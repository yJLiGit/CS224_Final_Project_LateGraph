"""Contraction of the memory update and the epsilon-repair certificate (Proposition 2).

For the update F(s_n, s_o) = GRU([s_n || s_o || x_e || phi(t_e - tau_n)], s_n) we compute the exact Jacobians
J_own = dF/ds_n and J_partner = dF/ds_o (torch.func.jacrev + vmap) for EVERY update of the in-order test stream, and their
spectral norms. An update is "certified" if a = ||J_own||_2 + ||J_partner||_2 < 1 (the local version of Eq. 5).

Then we check the certificate itself: for random deletions repaired with epsilon-repair, the worst memory error
E = max_n ||s_eps - s_exact|| should satisfy E <= r / (1 - rho), with r the largest change epsilon-repair skipped and
rho the largest a measured on the repaired updates. (The bound uses local Jacobian norms, so it is a first-order bound.)

  python scripts/run_contraction.py --data wikipedia --model tgn_wikipedia_s0_rho0.75_lam10
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jacrev, vmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph import paths                                       # noqa: E402
from lategraph.data import BatchGrid, load                        # noqa: E402
from lategraph.memory import EventTable, RetroactiveTGNMemory     # noqa: E402
from lategraph.model import load_checkpoint                       # noqa: E402


def jac_norms(mem: RetroactiveTGNMemory, nodes, other, eid):
    """Spectral norms of dF/ds_own and dF/ds_partner for each (node, partner, event) at the current memory."""
    E = mem.events
    s, p = mem.memory[nodes], mem.memory[other]
    dt = (E.t[eid] - mem.last_update[nodes]).clamp(min=0)
    phi = mem.time_enc(dt.to(s.dtype)); x = E.msg[eid].to(s.dtype)
    gru = mem.gru

    def f(si, pi, xi, phii):
        inp = torch.cat([si, pi, xi, phii]).view(1, -1)
        return gru(inp, si.view(1, -1)).view(-1)

    Jo, Jp = vmap(jacrev(f, argnums=(0, 1)))(s, p, x, phi)
    no = torch.linalg.matrix_norm(Jo, ord=2); npn = torch.linalg.matrix_norm(Jp, ord=2)
    return no, npn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--bq", type=int, default=200); ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--steps", type=int, default=60, help="window for the certificate experiment")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    S = load(args.data)
    model = load_checkpoint(paths.CKPT / f"{args.model}.pt")
    grid = BatchGrid(S.t, S.test0, args.bq)
    step = torch.full((S.num_events,), -1, dtype=torch.long)
    test = torch.arange(S.test0, S.num_events)
    step[test] = grid.step_of_time(S.t[test])
    table = EventTable(S.src, S.dst, S.t, S.msg, step)
    mem = RetroactiveTGNMemory.from_trained(model.memory, table)
    mem.warmup(0, S.test0)
    # ---- (a) Jacobian norms on every test update
    no_all, np_all = [], []
    for k in range(grid.J):
        ev = test[step[test] == k]
        if ev.numel():
            nodes, other, eid = mem.choose(ev)
            with torch.no_grad():
                pass
            no, npn = jac_norms(mem, nodes, other, eid)
            no_all.append(no.detach()); np_all.append(npn.detach())
        mem.advance(ev, journal=False)
    no_all = torch.cat(no_all).numpy(); np_all = torch.cat(np_all).numpy(); a = no_all + np_all
    q = lambda v: dict(median=float(np.median(v)), p90=float(np.quantile(v, .9)), p99=float(np.quantile(v, .99)), max=float(v.max()))
    jac = dict(updates=int(len(a)), J_own=q(no_all), J_partner=q(np_all), sum=q(a), certified_frac=float((a < 1).mean()))
    print(f"[{time.time() - t0:.0f}s] {args.model}: {len(a)} updates; median ||J_own|| {jac['J_own']['median']:.3f}, "
          f"||J_p|| {jac['J_partner']['median']:.3f}; certified (sum < 1): {jac['certified_frac'] * 100:.1f}%", flush=True)

    # ---- (b) certificate tightness: random deletions repaired with epsilon-repair vs exact
    W = min(args.steps, grid.J)
    base = RetroactiveTGNMemory.from_trained(model.memory, table)
    base.warmup(0, S.test0)
    for k in range(W):
        base.advance(test[step[test] == k])
    win = test[step[test] < W].numpy()
    work = RetroactiveTGNMemory.from_trained(model.memory, table)
    cert = {}

    class Tracker:
        """Online first-order error bound for epsilon-repair: per node b_n, propagated through every update of the window,
        b_n <- ||J_own|| b_n + ||J_partner|| b_o + (skipped change, if epsilon-repair kept the old value here)."""
        def __init__(self):
            self.b = torch.zeros(S.num_nodes); self.local_rho = 0.0

        def __call__(self, k, nodes, other, eid, skipped, skip_diff, mem_, lu_):
            need = (self.b[nodes] > 0) | (self.b[other] > 0)
            new_b = skip_diff.clone()
            if bool(need.any()):
                idx = need.nonzero().view(-1)
                no, npn = jac_norms(work, nodes[idx], other[idx], eid[idx])
                no, npn = no.detach(), npn.detach()
                self.local_rho = max(self.local_rho, float((no + npn).max()))
                new_b[idx] += no * self.b[nodes[idx]] + npn * self.b[other[idx]]
            self.b[nodes] = new_b

    cell_rho = model.memory.gru.rho() if hasattr(model.memory.gru, "rho") else None

    class RigorousTracker:
        """For a memory update that is a contraction by construction (ContractiveRNNCell, constant rho in the inf-norm):
        b_n <- rho * max(b_n, b_o) + (skipped change at n), over EVERY update of the window (copied ones included).
        By induction ||s_eps,n - s_exact,n||_inf <= b_n at all times: a bound that holds for any input, not first order.
        (The skipped change is measured in L2, which is an upper bound of its inf-norm.)"""
        def __init__(self):
            self.b = torch.zeros(S.num_nodes)

        def __call__(self, k, nodes, other, eid, skipped, skip_diff, mem_, lu_):
            self.b[nodes] = cell_rho * torch.maximum(self.b[nodes], self.b[other]) + skip_diff

    def both(*hooks):
        def h(*a):
            for f in hooks:
                f(*a)
        return h

    for eps in (0.01, 0.05, 0.2):
        rows = []
        for i in range(args.trials):
            dele = torch.from_numpy(rng.choice(win, size=int(rng.integers(1, 6)), replace=False)).long()
            work.copy_state_from(base); work.edit(delete=dele); work.repair("exact")
            exact_mem = work.memory.clone()
            tr = Tracker(); rt = RigorousTracker() if cell_rho is not None else None
            work.copy_state_from(base); work.edit(delete=dele)
            st = work.repair("eps", eps=eps, hook=both(tr, rt) if rt else tr)
            err_n = (work.memory - exact_mem).norm(dim=1)
            row = dict(err=float(err_n.max()), bound=float(tr.b.max()), r=st.max_skipped_change, local_rho=tr.local_rho,
                       holds=bool((err_n <= tr.b + 1e-4).all()), skipped=st.skipped, recomputed=st.recomputed)
            if rt is not None:
                err_inf = (work.memory - exact_mem).abs().amax(dim=1)
                row.update(err_inf=float(err_inf.max()), bound_rig=float(rt.b.max()),
                           holds_rig=bool((err_inf <= rt.b + 1e-5).all()),
                           holds_global=bool(float(err_inf.max()) <= st.max_skipped_change / (1 - cell_rho) + 1e-5),
                           global_bound=st.max_skipped_change / (1 - cell_rho))
            rows.append(row)
        errs = np.array([x["err"] for x in rows]); bnd = np.array([x["bound"] for x in rows])
        nz = errs > 1e-7
        rho = float(a.max())
        cert[str(eps)] = dict(trials=len(rows), max_err=float(errs.max()), median_err=float(np.median(errs)),
                              frac_bound_holds=float(np.mean([x["holds"] for x in rows])),
                              median_bound_over_err=float(np.median(bnd[nz] / errs[nz])) if nz.any() else None,
                              p90_bound_over_err=float(np.quantile(bnd[nz] / errs[nz], 0.9)) if nz.any() else None,
                              max_local_rho=float(max(x["local_rho"] for x in rows)),
                              global_bound_valid=bool(rho < 1), rho_max_all_updates=rho,
                              skipped_frac=float(np.mean([x["skipped"] / max(x["recomputed"], 1) for x in rows])))
        if cell_rho is not None:
            ei = np.array([x["err_inf"] for x in rows]); br = np.array([x["bound_rig"] for x in rows])
            gb = np.array([x["global_bound"] for x in rows]); nzi = ei > 1e-7
            cert[str(eps)]["rigorous"] = dict(cell_rho=cell_rho, frac_holds=float(np.mean([x["holds_rig"] for x in rows])),
                                              frac_global_holds=float(np.mean([x["holds_global"] for x in rows])),
                                              median_bound_over_err=float(np.median(br[nzi] / ei[nzi])) if nzi.any() else None,
                                              median_global_over_err=float(np.median(gb[nzi] / ei[nzi])) if nzi.any() else None,
                                              max_err_inf=float(ei.max()))
            rg = cert[str(eps)]["rigorous"]
            print(f"  eps {eps}: rigorous bound (rho {cell_rho:.3f}) holds in {rg['frac_holds'] * 100:.0f}% of trials, global r/(1-rho) in "
                  f"{rg['frac_global_holds'] * 100:.0f}%; median bound/error {rg['median_bound_over_err'] or float('nan'):.1f}x "
                  f"(global {rg['median_global_over_err'] or float('nan'):.1f}x)", flush=True)
        c = cert[str(eps)]
        print(f"  eps {eps}: max err {c['max_err']:.4f}; first-order bound holds in {c['frac_bound_holds'] * 100:.0f}% of trials, "
              f"median bound/error {c['median_bound_over_err'] or float('nan'):.1f}x; global r/(1-rho) "
              + ("valid" if rho < 1 else f"vacuous (rho_max {rho:.2f})"), flush=True)
    # ---- (c) delta-certified repair (contractive memories only): skip a change only while the proven bound stays <= delta
    certified = {}
    if cell_rho is not None:
        for delta in (0.005, 0.02, 0.05):
            rows = []
            for i in range(args.trials):
                dele = torch.from_numpy(rng.choice(win, size=int(rng.integers(1, 6)), replace=False)).long()
                work.copy_state_from(base); work.edit(delete=dele); se = work.repair("exact"); exact_mem = work.memory.clone()
                work.copy_state_from(base); work.edit(delete=dele); sp = work.repair("eps", eps=0.2)
                work.copy_state_from(base); work.edit(delete=dele); sc = work.repair("cert", eps=0.2, delta=delta, rho=cell_rho)
                err = float((work.memory - exact_mem).abs().amax())
                rows.append(dict(err=err, bound=sc.extra["max_bound"], cost=sc.recomputed / max(se.recomputed, 1),
                                 eps_cost=sp.recomputed / max(se.recomputed, 1), skipped=sc.skipped, holds=err <= delta + 1e-6))
            certified[str(delta)] = dict(trials=len(rows), frac_holds=float(np.mean([x["holds"] for x in rows])),
                                         max_err=float(max(x["err"] for x in rows)), cost_vs_exact=float(np.mean([x["cost"] for x in rows])),
                                         eps_cost_vs_exact=float(np.mean([x["eps_cost"] for x in rows])),
                                         max_bound=float(max(x["bound"] for x in rows)))
            c = certified[str(delta)]
            print(f"  delta {delta}: error <= delta in {c['frac_holds'] * 100:.0f}% (max {c['max_err']:.4f}); cost {c['cost_vs_exact']:.2f}x of "
                  f"exact repair (plain eps-repair {c['eps_cost_vs_exact']:.2f}x)", flush=True)
    out = dict(config=vars(args), jacobian=jac, certificate=cert, certified_repair=certified, wall_sec=time.time() - t0,
               model_meta=dict(val_ap=model.meta.get("best_val_ap"), lam=model.meta.get("lam"), rho=model.meta.get("rho"),
                               cell_rho=cell_rho, updater=model.cfg.updater))
    od = paths.RESULTS / "contraction"; od.mkdir(parents=True, exist_ok=True)
    with open(od / f"contraction_{args.model}.json", "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(od / f"contraction_{args.model}_norms.npz", J_own=no_all.astype(np.float32), J_partner=np_all.astype(np.float32))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
