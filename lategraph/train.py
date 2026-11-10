"""Training the TGN, following PyG's examples/tgn.py (batch 200, Adam 1e-4, one random negative per positive).

Optional: the "repairable memory" penalty of the proposal (Eq. 6),
    L = L_link + lam * E_e[(||dF/ds_u||_2 + ||dF/ds_v||_2 - rho)_+^2],
where F(s_u, s_v) = GRU([s_u || s_v || x_e || phi(t_e - tau_u)], s_u) is the memory update. Both spectral norms are
estimated by power iteration with finite-difference Jacobian-vector products on 64 sampled events per batch.
"""
import json
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch_geometric.nn.models.tgn import LastNeighborLoader

from .data import EventStream
from .model import TGN, TGNConfig, save_checkpoint


def time_gap_stats(E, hi: int):
    """Mean and std (seconds) of the time between consecutive events of the same node, over events [0, hi)."""
    node = torch.cat([E.src[:hi], E.dst[:hi]]).numpy(); t = torch.cat([E.t[:hi], E.t[:hi]]).numpy().astype(np.float64)
    order = np.lexsort((t, node)); node, t = node[order], t[order]
    gap = np.diff(t)[node[1:] == node[:-1]]
    return float(gap.mean()), float(gap.std())


def fixed_batches(lo: int, hi: int, size: int):
    return [(b, min(b + size, hi)) for b in range(lo, hi, size)]


def contraction_penalty(memory, src, dst, t, msg, rho: float, n_sample=64, iters=3, h=1e-2):
    """Returns (penalty without lam, median of a = ||J_own|| + ||J_partner|| over the sampled updates)."""
    idx = torch.randperm(src.numel())[:n_sample]
    nodes, part = torch.cat([src[idx], dst[idx]]), torch.cat([dst[idx], src[idx]])
    x = torch.cat([msg[idx], msg[idx]]); tt = torch.cat([t[idx], t[idx]])
    s0 = memory.memory[nodes].detach(); p0 = memory.memory[part].detach()
    phi = memory.time_enc((tt - memory.last_update[nodes]).clamp(min=0).to(s0.dtype))

    def F(s, p):
        return memory.gru(torch.cat([s, p, x, phi], -1), s)

    base = F(s0, p0)
    norms = []
    for which in (0, 1):                                    # 0: own memory, 1: partner memory
        v = torch.randn_like(s0); v = v / v.norm(dim=1, keepdim=True)
        pert = (lambda d: F(s0 + h * d, p0)) if which == 0 else (lambda d: F(s0, p0 + h * d))
        for _ in range(iters):                              # power iteration on J^T J (no graph through v)
            with torch.no_grad():
                u = (pert(v) - base.detach()) / h
            s = s0.clone().requires_grad_(True); p = p0.clone().requires_grad_(True)
            g = torch.autograd.grad((F(s, p) * u).sum(), (s, p))[which]
            v = g.detach(); v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
        norms.append(((pert(v) - base) / h).norm(dim=1))   # differentiable w.r.t. the GRU weights
    a = norms[0] + norms[1]
    return torch.relu(a - rho).pow(2).mean(), float(a.detach().median())


def train_tgn(stream: EventStream, cfg: TGNConfig = TGNConfig(), epochs=15, seed=0, lam=0.0, rho=0.9,
              batch_size=200, train_batches=None, val_batches=None, lr=1e-4, threads=4, ckpt_path=None, log=print,
              meta=None, init_state=None):
    """Train on events [0, val0), select the epoch with the best validation AP on [val0, test0); save the checkpoint.
    `init_state`: start from these weights (fine-tuning) instead of a fresh initialisation."""
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(threads)
    E = stream
    if cfg.emb == "proj" and cfg.proj_std == 1.0:              # JODIE: standardise the time since a node's last update
        cfg.proj_mean, cfg.proj_std = time_gap_stats(E, E.val0)
    model = TGN(E.num_nodes, E.msg_dim, cfg)
    if init_state is not None:
        model.load_state_dict(init_state)
    memory, gnn, link_pred = model.memory, model.gnn, model.link_pred
    neighbor_loader = LastNeighborLoader(E.num_nodes, size=cfg.n_neighbors)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.BCEWithLogitsLoss()
    assoc = torch.empty(E.num_nodes, dtype=torch.long)
    train_batches = train_batches or fixed_batches(0, E.val0, batch_size)
    val_batches = val_batches or fixed_batches(E.val0, E.test0, batch_size)
    gen = torch.Generator().manual_seed(seed)
    a_hist = []

    def step(lo, hi, train: bool, neg_gen):
        src, dst, t, msg = E.src[lo:hi], E.dst[lo:hi], E.t[lo:hi], E.msg[lo:hi]
        neg = torch.randint(E.dst_min, E.dst_max + 1, (hi - lo,), generator=neg_gen)
        n_id = torch.cat([src, dst, neg]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0))
        mem, last_update = memory(n_id)
        z = model.embed(mem, last_update, edge_index, E.t[e_id], E.msg[e_id], now=int(t.max()))
        pos = link_pred(z[assoc[src]], z[assoc[dst]])
        negs = link_pred(z[assoc[src]], z[assoc[neg]])
        loss = None
        if train:
            loss = crit(pos, torch.ones_like(pos)) + crit(negs, torch.zeros_like(negs))
            if lam > 0 and cfg.emb != "nomem":
                pen, a_med = contraction_penalty(memory, src, dst, t, msg, rho)
                loss = loss + lam * pen; a_hist.append(a_med)
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        return pos, negs, loss

    best, history, best_state = -1.0, [], None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train(); memory.reset_state(); neighbor_loader.reset_state()
        tot, n = 0.0, 0
        for lo, hi in train_batches:
            opt.zero_grad()
            _, _, loss = step(lo, hi, True, gen)
            loss.backward(); opt.step(); memory.detach()
            tot += loss.detach().item() * (hi - lo); n += hi - lo
        model.eval()
        aps = []
        vgen = torch.Generator().manual_seed(12345)        # same validation negatives every epoch
        with torch.no_grad():
            for lo, hi in val_batches:
                pos, negs, _ = step(lo, hi, False, vgen)
                y = torch.cat([torch.ones(pos.size(0)), torch.zeros(negs.size(0))])
                aps.append(average_precision_score(y.numpy(), torch.cat([pos, negs]).sigmoid().squeeze(-1).numpy()))
        val_ap = float(np.mean(aps))
        rec = dict(epoch=ep, loss=tot / max(n, 1), val_ap=val_ap, sec=time.time() - t0)
        if a_hist: rec["median_a"] = float(np.median(a_hist[-500:]))
        history.append(rec)
        log(f"{E.name} seed {seed} ep {ep:02d} loss {rec['loss']:.4f} val AP {val_ap:.4f}"
            + (f" median |Jo|+|Jp| {rec['median_a']:.3f}" if a_hist else "") + f" ({rec['sec']:.0f}s)")
        if val_ap > best:
            best = val_ap
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
    model.load_state_dict(best_state)
    model.eval()
    info = dict(dataset=E.name, seed=seed, epochs=epochs, best_epoch=best_epoch, best_val_ap=best, lam=lam, rho=rho,
                cfg=cfg.__dict__, history=history, **(meta or {}))
    if ckpt_path is not None:
        save_checkpoint(ckpt_path, model, info)
        with open(str(ckpt_path).replace(".pt", ".json"), "w") as f:
            json.dump(info, f, indent=1)
    return model, info
