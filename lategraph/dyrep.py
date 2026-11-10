"""DyRep-style memory (Trivedi et al., ICLR 2019, in the form used by DyGLib): the update of a node reads its partner's
*neighbourhood*, not only the partner's memory.

    h_o  = Attn( s_o ; { (s_w, phi(dt), x_e') : (w, e') in N_K(o) before the step } )      (TransformerConv, 2 heads)
    s_n <- RNN( [ s_n || h_o || x_e || phi(t_e - tau_n) ], s_n )

so an update of n reads s_n, s_o, the memories of o's last K neighbours, and o's neighbour list itself. The dependency
cone therefore grows through neighbour reads, and an edit also changes later updates whose partner's neighbour list
contains (or would contain) the edited event. RetroactiveDyRepMemory keeps a journal with the neighbour events every
update read and repairs exactly that cone (it walks every step of the window; the neighbour store is rebuilt to the
state before the earliest edited step and replayed with the corrected events).

Training follows PyG's lazy TGN schedule: the memory update for batch b-1 is computed at the start of batch b (with
gradients), from the memory and neighbour lists as they were before batch b-1.
"""
import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch_geometric.nn.models.tgn import TimeEncoder

from .memory import EventTable, RepairStats
from .model import GraphAttentionEmbedding, LinkPredictor
from .neighbors import TimeOrderedNeighborStore


class DyRep(torch.nn.Module):
    def __init__(self, num_nodes: int, msg_dim: int, memory_dim=100, time_dim=100, emb_dim=100, n_neighbors=10):
        super().__init__()
        self.num_nodes, self.msg_dim, self.d, self.K = num_nodes, msg_dim, memory_dim, n_neighbors
        self.time_enc = TimeEncoder(time_dim)
        self.msg_attn = GraphAttentionEmbedding(memory_dim, memory_dim, msg_dim, self.time_enc)   # h_o for messages
        self.updater = torch.nn.RNNCell(2 * memory_dim + msg_dim + time_dim, memory_dim)
        self.gnn = GraphAttentionEmbedding(memory_dim, emb_dim, msg_dim, self.time_enc)          # z for prediction
        self.link_pred = LinkPredictor(emb_dim)

    # ------------------------------------------------------------------------------------------- building blocks
    def attend(self, centers, nb_eid, mem, lu, E):
        """h for every node in `centers` from its neighbour events nb_eid [m, K] (-1 = empty slot)."""
        uc, inv = torch.unique(centers, return_inverse=True)
        first = torch.zeros(uc.numel(), dtype=torch.long)
        first[inv] = torch.arange(centers.numel())                       # one row of nb_eid per unique centre (rows of
        nb = nb_eid[first]                                               # the same centre are identical)
        mask = nb >= 0
        cen = uc.view(-1, 1).expand_as(nb)[mask]
        e = nb[mask]
        w = torch.where(E.src[e] == cen, E.dst[e], E.src[e])
        local = torch.unique(torch.cat([uc, w]))
        pos = lambda x: torch.searchsorted(local, x)
        ei = torch.stack([pos(w), pos(cen)])
        rel_t = (lu[local][ei[0]] - E.t[e]).to(mem.dtype)
        if rel_t.numel() == 1:                                           # batch invariance of cos(w * dt + b), as in
            enc = self.time_enc(rel_t.repeat(2))[:1]                     # RetroactiveTGNMemory.compute_update
        else:
            enc = self.time_enc(rel_t)
        z = self.msg_attn.conv(mem[local], ei, torch.cat([enc, E.msg[e].to(mem.dtype)], -1))
        return z[pos(uc)][inv]

    def update(self, nodes, other, eid, nb_eid, mem, lu, E):
        """New memory and last-update time of `nodes` (rows aligned), reading `mem`/`lu` (the pre-step state)."""
        h = self.attend(other, nb_eid, mem, lu, E)
        tt = E.t[eid]
        dt = (tt - lu[nodes]).clamp(min=0)
        m = torch.cat([mem[nodes], h, E.msg[eid].to(mem.dtype), self.time_enc(dt.to(mem.dtype))], -1)
        return self.updater(m, mem[nodes]), torch.maximum(tt, lu[nodes])

    def embed(self, mem, lu, edge_index, t_e, msg_e):
        return self.gnn(mem, lu, edge_index, t_e, msg_e)

    def score(self, z_src, z_dst):
        return self.link_pred(z_src, z_dst).squeeze(-1)


def choose(E: EventTable, ev: torch.Tensor):
    """LastAggregator: every node of the step is updated once, with its latest event (ties: larger event id)."""
    node = torch.cat([E.src[ev], E.dst[ev]]); other = torch.cat([E.dst[ev], E.src[ev]]); eid = torch.cat([ev, ev])
    order = torch.argsort(E.key[eid], descending=True)
    _, first = np.unique(node[order].numpy(), return_index=True)
    sel = order[torch.from_numpy(first)]
    return node[sel], other[sel], eid[sel]


def _single_row_pad(nodes, other, eid, nb):
    """Batch invariance (see RetroactiveTGNMemory.compute_update): never compute a single row on its own."""
    if nodes.numel() == 1:
        return nodes.repeat(2), other.repeat(2), eid.repeat(2), nb.repeat(2, 1), True
    return nodes, other, eid, nb, False


# ----------------------------------------------------------------------------------------------- training
def train_dyrep(S, epochs=15, seed=0, lr=1e-4, batch_size=200, threads=3, log=print, ckpt_path=None, meta=None):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(threads)
    model = DyRep(S.num_nodes, S.msg_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.BCEWithLogitsLoss()
    E = EventTable(S.src, S.dst, S.t, S.msg, torch.full((S.num_events,), -1, dtype=torch.long))
    gen = torch.Generator().manual_seed(seed)

    def run(lo_hi, train: bool, neg_gen, state=None):
        if state is None:
            mem = torch.zeros(S.num_nodes, model.d); lu = torch.zeros(S.num_nodes, dtype=torch.long)
            store = TimeOrderedNeighborStore(S.src, S.dst, S.t, S.num_nodes, model.K); pending = None
        else:
            mem, lu, store, pending = state
        aps, tot = [], 0.0
        for lo, hi in lo_hi:
            cur, cur_lu = mem, lu
            if pending is not None:                                  # apply batch b-1's updates (lazy, with gradients)
                nodes, other, eid, nb = pending
                nodes, other, eid, nb, single = _single_row_pad(nodes, other, eid, nb)
                new, new_lu = model.update(nodes, other, eid, nb, mem, lu, E)
                if single:
                    nodes, new, new_lu = nodes[:1], new[:1], new_lu[:1]
                cur = mem.clone(); cur[nodes] = new
                cur_lu = lu.clone(); cur_lu[nodes] = new_lu
            ev = torch.arange(lo, hi)
            src, dst = S.src[ev], S.dst[ev]
            neg = torch.randint(S.dst_min, S.dst_max + 1, (hi - lo,), generator=neg_gen)
            n_id = torch.cat([src, dst, neg]).unique()
            all_id, edge_index, e_id = store(n_id)
            assoc = torch.empty(S.num_nodes, dtype=torch.long); assoc[all_id] = torch.arange(all_id.numel())
            z = model.embed(cur[all_id], cur_lu[all_id], edge_index, S.t[e_id], S.msg[e_id])
            pos = model.score(z[assoc[src]], z[assoc[dst]]); negs = model.score(z[assoc[src]], z[assoc[neg]])
            if train:
                loss = crit(pos, torch.ones_like(pos)) + crit(negs, torch.zeros_like(negs))
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * (hi - lo)
            else:
                y = torch.cat([torch.ones(pos.numel()), torch.zeros(negs.numel())])
                aps.append(average_precision_score(y.numpy(), torch.cat([pos, negs]).sigmoid().detach().numpy()))
            mem, lu = cur.detach(), cur_lu.detach()
            nodes, other, eid = choose(E, ev)
            pending = (nodes, other, eid, store.e_id[other].clone())   # partners' neighbour lists before this batch
            store.insert(ev)
        return tot, aps, (mem, lu, store, pending)

    tr = [(b, min(b + batch_size, S.val0)) for b in range(0, S.val0, batch_size)]
    va = [(b, min(b + batch_size, S.test0)) for b in range(S.val0, S.test0, batch_size)]
    best, best_state, hist = -1.0, None, []
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train()
        tot, _, state = run(tr, True, gen)
        model.eval()
        with torch.no_grad():                                       # validation continues from the training memory
            _, aps, _ = run(va, False, torch.Generator().manual_seed(12345), state)
        val_ap = float(np.mean(aps))
        hist.append(dict(epoch=ep, loss=tot / S.val0, val_ap=val_ap, sec=time.time() - t0))
        log(f"{S.name} DyRep seed {seed} ep {ep:02d} loss {tot / S.val0:.4f} val AP {val_ap:.4f} ({time.time() - t0:.0f}s)")
        if val_ap > best:
            best, best_state, best_ep = val_ap, copy.deepcopy(model.state_dict()), ep
    model.load_state_dict(best_state); model.eval()
    info = dict(dataset=S.name, seed=seed, epochs=epochs, best_epoch=best_ep, best_val_ap=best, history=hist, **(meta or {}))
    if ckpt_path is not None:
        torch.save(dict(state=model.state_dict(), num_nodes=S.num_nodes, msg_dim=S.msg_dim, meta=info), ckpt_path)
    return model, info


def load_dyrep(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = DyRep(ck["num_nodes"], ck["msg_dim"]); m.load_state_dict(ck["state"]); m.eval(); m.meta = ck["meta"]
    return m


# ----------------------------------------------------------------------------------------------- inference + repair
@dataclass
class DyRepRecord:
    nodes: torch.Tensor; other: torch.Tensor; eid: torch.Tensor; nb: torch.Tensor     # nb: partners' neighbour events [n, K]
    pre_mem: torch.Tensor; pre_lu: torch.Tensor; post_mem: torch.Tensor; post_lu: torch.Tensor


class RetroactiveDyRepMemory:
    """Journaled DyRep memory with exact repair of the dependency cone (reads = own, partner, partner's neighbours)."""

    def __init__(self, model: DyRep, events: EventTable, history_end: int, batch_size=200):
        self.model, self.E, self.N, self.K = model, events, model.num_nodes, model.K
        self.memory = torch.zeros(self.N, model.d); self.last_update = torch.zeros(self.N, dtype=torch.long)
        self.store = TimeOrderedNeighborStore(events.src, events.dst, events.t, self.N, model.K)
        self.journal, self.batch_events, self.pending, self.step = {}, {}, {}, 0
        self.history_end, self.normal_updates = history_end, 0
        with torch.no_grad():
            for b in range(0, history_end, batch_size):
                self._apply(torch.arange(b, min(b + batch_size, history_end)), journal=False)
        self.base_store = self.store.clone()                        # neighbour lists after the history

    def clone(self) -> "RetroactiveDyRepMemory":
        c = RetroactiveDyRepMemory.__new__(RetroactiveDyRepMemory); c.__dict__.update(self.__dict__)
        return c.copy_state_from(self)

    def copy_state_from(self, o):
        self.memory, self.last_update = o.memory.clone(), o.last_update.clone()
        self.store = o.store.clone(); self.base_store = o.base_store
        self.journal, self.batch_events, self.pending = dict(o.journal), dict(o.batch_events), dict(o.pending)
        self.step, self.normal_updates = o.step, o.normal_updates
        return self

    @torch.no_grad()
    def _compute(self, nodes, other, eid, nb, mem, lu):
        nodes, other, eid, nb, single = _single_row_pad(nodes, other, eid, nb)
        new, new_lu = self.model.update(nodes, other, eid, nb, mem, lu, self.E)
        return (new[:1], new_lu[:1]) if single else (new, new_lu)

    @torch.no_grad()
    def _apply(self, ev, journal=True):
        if ev.numel():
            nodes, other, eid = choose(self.E, ev)
            nb = self.store.e_id[other].clone()
            pre_m, pre_l = self.memory[nodes].clone(), self.last_update[nodes].clone()
            new, new_lu = self._compute(nodes, other, eid, nb, self.memory, self.last_update)
            self.memory[nodes] = new; self.last_update[nodes] = new_lu
            self.store.insert(ev)
        else:
            nodes = other = eid = torch.empty(0, dtype=torch.long); nb = torch.empty(0, self.K, dtype=torch.long)
            pre_m = new = self.memory.new_empty(0, self.model.d); pre_l = new_lu = torch.empty(0, dtype=torch.long)
        if journal:
            self.journal[self.step] = DyRepRecord(nodes, other, eid, nb, pre_m, pre_l, new, new_lu)
            self.batch_events[self.step] = torch.sort(ev).values
            self.step += 1; self.normal_updates += int(nodes.numel())

    def advance(self, ev):
        self._apply(ev.long(), journal=True)

    def edit(self, insert=None, delete=None):
        for kind, ev in (("insert", insert), ("delete", delete)):
            if ev is None or ev.numel() == 0:
                continue
            st = self.E.step[ev]
            assert bool((st >= 0).all() & (st < self.step).all())
            for k in torch.unique(st).tolist():
                cur, chg = self.batch_events[k], ev[st == k]
                self.batch_events[k] = torch.unique(torch.cat([cur, chg])) if kind == "insert" else cur[~torch.isin(cur, chg)]
                self.pending[k] = True

    @torch.no_grad()
    def repair(self, mode="exact", eps=0.0) -> RepairStats:
        st = RepairStats()
        if not self.pending:
            return st
        k0, K = min(self.pending), self.step
        st.k0, st.steps = k0, K - k0
        mem, lu = self.memory, self.last_update
        for k in range(K - 1, k0 - 1, -1):                          # roll the memory back to before step k0
            r = self.journal[k]
            if r.nodes.numel():
                mem[r.nodes] = r.pre_mem; lu[r.nodes] = r.pre_lu
        store = self.base_store.clone()                              # neighbour lists before step k0 (unchanged by edits)
        for k in range(0, k0):
            store.insert(self.batch_events[k])
        dirty = torch.zeros(self.N, dtype=torch.bool)
        E = self.E
        for k in range(k0, K):
            old, ev = self.journal[k], self.batch_events[k]
            st.window_updates += int(old.nodes.numel())
            if ev.numel():
                nodes, other, eid = choose(E, ev)
            else:
                nodes = other = eid = torch.empty(0, dtype=torch.long)
            nb = store.e_id[other].clone()
            n_old = old.nodes.numel()
            if n_old:
                pos = torch.searchsorted(old.nodes, nodes).clamp(max=n_old - 1)
                in_old = old.nodes[pos] == nodes
                same = in_old & (old.eid[pos] == eid) & (old.other[pos] == other) & (old.nb[pos] == nb).all(1)
            else:
                pos = torch.zeros_like(nodes); in_old = same = torch.zeros(nodes.numel(), dtype=torch.bool)
            nbm = nb >= 0
            nbr = torch.where(E.src[nb.clamp(min=0)] == other.view(-1, 1), E.dst[nb.clamp(min=0)], E.src[nb.clamp(min=0)])
            reads_dirty = dirty[nodes] | dirty[other] | (dirty[nbr] & nbm).any(1)
            aff = torch.ones(nodes.numel(), dtype=torch.bool) if mode == "replay" else (~same | reads_dirty)
            pre_m, pre_l = mem[nodes].clone(), lu[nodes].clone()
            post_m, post_l = torch.empty_like(pre_m), torch.empty_like(pre_l)
            keep = ~aff
            post_m[keep] = old.post_mem[pos[keep]]; post_l[keep] = old.post_lu[pos[keep]]
            new_dirty = torch.zeros(nodes.numel(), dtype=torch.bool)
            if bool(aff.any()):
                na = aff.nonzero().view(-1)
                new_m, new_l = self._compute(nodes[na], other[na], eid[na], nb[na], mem, lu)
                st.recomputed += int(na.numel())
                if mode == "eps" and n_old:
                    can = same[na] & (new_l == old.post_lu[pos[na]]); ref = old.post_mem[pos[na]]
                    diff = (new_m - ref).norm(dim=1); skip = can & (diff <= eps * ref.norm(dim=1))
                    if bool(skip.any()):
                        st.skipped += int(skip.sum()); st.max_skipped_change = max(st.max_skipped_change, float(diff[skip].max()))
                        new_m[skip] = ref[skip]
                    new_dirty[na] = ~skip
                else:
                    new_dirty[na] = True
                post_m[na] = new_m; post_l[na] = new_l
            gone = old.nodes[~torch.isin(old.nodes, nodes)] if n_old else torch.empty(0, dtype=torch.long)
            mem[nodes] = post_m; lu[nodes] = post_l
            dirty[nodes] = new_dirty
            if gone.numel():
                dirty[gone] = True
            self.journal[k] = DyRepRecord(nodes, other, eid, nb, pre_m, pre_l, post_m, post_l)
            store.insert(ev)
        self.store = store
        self.pending = {}
        return st
