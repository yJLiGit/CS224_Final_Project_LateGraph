"""E4: after a deletion request, can the deleted interactions still be detected from the deployed TGN's memory?

Design (A/B pairs). Among user-item pairs that first interact in the test period, half (A) are observed and then deleted
at the request step; the other half (B) are never observed (their events are removed from every method's stream). After
an exact deletion, A and B are both "never observed", so no attack can tell them apart (AUC = 0.5). Model weights were
trained before the test period, so they never saw A or B: anything an attacker finds comes from the memory.

Methods at the request step: none, store (delete from the log / neighbour store only), reset (store + zero the
endpoints' memories), exact (cone repair), replay, eps:<x>, khop:<K>.
Attacks: logit (the model's own link score), rank (rank of v among all items for u), probe (logistic regression on
memory features [s_u*s_v, |s_u-s_v|, cos, norms, logit], 5-fold cross-validated).
Measured at +0, +5, +20, +50 steps after the request, while the stream continues.

  python scripts/run_deletion.py --data wikipedia --model tgn_wikipedia_s0 --seed 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph import paths                                                       # noqa: E402
from lategraph.data import BatchGrid, load                                        # noqa: E402
from lategraph.memory import EventTable, RetroactiveTGNMemory, khop_mask_for      # noqa: E402
from lategraph.model import load_checkpoint                                       # noqa: E402
from lategraph.neighbors import TimeOrderedNeighborStore                          # noqa: E402
from lategraph.stream import score_pairs                                          # noqa: E402

METHODS = "none,store,reset,restart,restart_cone,exact,replay,eps:0.01,eps:0.05,eps:0.2,khop:1,khop:2"


def auc(a, b):
    return float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--bq", type=int, default=200); ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--jdel", type=float, default=0.5); ap.add_argument("--after", default="0,5,20,50")
    ap.add_argument("--methods", default=METHODS); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--out", default=None)
    ap.add_argument("--no_rank", action="store_true", help="skip the rank attack (it scores every item; too slow on huge graphs)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    S = load(args.data)
    model = load_checkpoint(paths.CKPT / f"{args.model}.pt")
    grid = BatchGrid(S.t, S.test0, args.bq)
    J = grid.J
    step = torch.full((S.num_events,), -1, dtype=torch.long)
    test = torch.arange(S.test0, S.num_events)
    step[test] = grid.step_of_time(S.t[test])
    jdel = int(args.jdel * J)
    table = EventTable(S.src, S.dst, S.t, S.msg, step)
    src, dst = S.src.numpy(), S.dst.numpy()
    pre_end = int((step >= 0).logical_and(step < jdel).nonzero().max()) + 1

    # ---- A/B pairs: new in the test period before the request; the user has >= 3 other events
    hist_pairs = set(zip(src[:S.test0].tolist(), dst[:S.test0].tolist()))
    pair_events = {}
    for e in range(S.test0, pre_end):
        pair_events.setdefault((int(src[e]), int(dst[e])), []).append(e)
    ucnt = np.bincount(src[:pre_end], minlength=S.num_nodes)
    cands = [p for p, ev in pair_events.items() if p not in hist_pairs and ucnt[p[0]] - len(ev) >= 3]
    perm = rng.permutation(len(cands))[: 2 * min(args.K, len(cands) // 2)]
    A = [cands[i] for i in perm[0::2]]; B = [cands[i] for i in perm[1::2]]
    D = torch.tensor(sorted(e for p in A for e in pair_events[p]), dtype=torch.long)
    hidden = torch.zeros(S.num_events, dtype=torch.bool)
    hidden[torch.tensor(sorted(e for p in B for e in pair_events[p]), dtype=torch.long)] = True
    deleted = torch.zeros(S.num_events, dtype=torch.bool); deleted[D] = True
    batches = [test[(step[test] == j) & ~hidden[test]] for j in range(J)]
    print(f"[{time.time() - t0:.0f}s] {args.data}: {len(cands)} candidate pairs; A {len(A)} pairs / {D.numel()} events deleted, "
          f"B {len(B)} never observed; request at step {jdel}/{J}", flush=True)

    # ---- base trajectory up to the request (A observed, B never), with the journal
    base = RetroactiveTGNMemory.from_trained(model.memory, table)
    base.warmup(0, S.test0)
    for k in range(jdel):
        base.advance(batches[k])
    normal_cost = base.normal_updates
    store_all = TimeOrderedNeighborStore(S.src, S.dst, S.t, S.num_nodes, 10)
    keep = torch.zeros(S.num_events, dtype=torch.bool); keep[:pre_end] = True; keep &= ~hidden
    store_all.rebuild_from(keep)
    store_del = store_all.clone(); store_del.remove(D)
    k0 = int(step[D].min())
    window_events = torch.cat([batches[k] for k in range(k0, jdel)])

    _R = {}

    def restarter():
        if "R" not in _R:
            from lategraph.restarter import Restarter
            R = Restarter(model.memory.memory_dim, S.msg_dim)
            R.load_state_dict(torch.load(paths.CKPT / f"{args.model}_restarter.pt")); R.eval()
            _R["R"] = R
        return _R["R"]

    states = {}
    methods = args.methods.split(",")
    if not (paths.CKPT / f"{args.model}_restarter.pt").exists():
        methods = [m for m in methods if not m.startswith("restart")]
        print("  (no restarter checkpoint for this model: the restart methods are skipped)", flush=True)
    for m in methods:
        mem = RetroactiveTGNMemory.from_trained(model.memory, table).copy_state_from(base)
        kind, _, arg = m.partition(":")
        tt = time.perf_counter(); cost = 0; st = None
        if kind == "none":
            store = store_all.clone()
        elif kind == "store":
            store = store_del.clone()
        elif kind == "reset":
            ep = torch.cat([S.src[D], S.dst[D]]).unique()
            mem.memory[ep] = 0; mem.last_update[ep] = 0; store = store_del.clone()
        elif kind in ("restart", "restart_cone"):        # TIGER-style learned restart from the corrected neighbour list
            R = restarter()
            if kind == "restart":
                nodes = torch.cat([S.src[D], S.dst[D]]).unique()
            else:                                         # restart every node the exact repair would change
                ex = RetroactiveTGNMemory.from_trained(model.memory, table).copy_state_from(base)
                ex.edit(delete=D); ex.repair("exact")
                nodes = ((ex.memory - base.memory).abs().amax(1) > 0).nonzero().view(-1)
            store = store_del.clone()
            with torch.no_grad():
                mem.memory[nodes] = R(nodes, mem.memory, store, table, int(grid.T[jdel]))
            e0 = store.e_id[nodes, 0]                      # newest remaining event of each node (-1: none)
            mem.last_update[nodes] = torch.where(e0 >= 0, table.t[e0.clamp(min=0)], torch.zeros_like(e0))
            cost = int(nodes.numel())
        else:
            mem.edit(delete=D)
            mask = None
            if kind == "khop":
                seeds = torch.cat([S.src[D], S.dst[D]]).unique()
                mask = khop_mask_for(table, seeds, window_events, int(arg), S.num_nodes)
            mode = {"exact": "exact", "replay": "replay", "eps": "eps", "khop": "khop"}[kind]
            st = mem.repair(mode, eps=float(arg) if arg else 0.0, khop_mask=mask)
            cost = st.recomputed; store = store_del.clone()
        states[m] = dict(mem=mem, store=store, cost=cost, sec=time.perf_counter() - tt,
                         skipped=st.skipped if st else 0, max_skip=st.max_skipped_change if st else 0.0)
    ref = states["exact"]["mem"]
    cone = ((base.memory - ref.memory).abs().amax(1) > 0).nonzero().view(-1)
    ep = torch.cat([S.src[D], S.dst[D]]).unique()
    print(f"[{time.time() - t0:.0f}s] exact deletion changed {cone.numel()} of {S.num_nodes} memories "
          f"({ep.numel()} endpoints); cone cost {states['exact']['cost']} vs normal {normal_cost}", flush=True)

    # ---- attacks while the stream continues
    uA = torch.tensor([p[0] for p in A]); vA = torch.tensor([p[1] for p in A])
    uB = torch.tensor([p[0] for p in B]); vB = torch.tensor([p[1] for p in B])
    next_seen = {}
    for e in range(pre_end, S.num_events):
        p = (int(src[e]), int(dst[e]))
        if p not in next_seen:
            next_seen[p] = int(step[e])
    after = [int(a) for a in args.after.split(",")]
    items = torch.arange(S.dst_min, S.dst_max + 1)
    results, scores0 = {}, {}
    ref_mem = ref.memory.clone()                       # exact memory at the request step (ref itself keeps streaming)
    for m, d in states.items():                        # how far each method's memory is from the exact one, at request
        mm = d["mem"].memory
        d["resid_cone"] = float((mm[cone] - ref_mem[cone]).norm() / (ref_mem[cone].norm() + 1e-12)) if cone.numel() else 0.0
        d["resid_endpoints"] = float((mm[ep] - ref_mem[ep]).norm() / (ref_mem[ep].norm() + 1e-12))

    def features(mem, store, u, v, now):
        lg = score_pairs(model, mem, store, table, u, v, now).numpy()
        su, sv = mem.memory[u], mem.memory[v]
        cos = torch.nn.functional.cosine_similarity(su, sv, dim=1, eps=1e-12)
        F = torch.cat([su * sv, (su - sv).abs(), cos.view(-1, 1), su.norm(dim=1, keepdim=True), sv.norm(dim=1, keepdim=True)], 1)
        return lg, np.c_[F.numpy(), lg]

    def rank_score(mem, store, u, v, now):
        out = np.empty(len(u))
        for b in range(0, len(u), 100):
            uu, vv = u[b:b + 100], v[b:b + 100]
            s = score_pairs(model, mem, store, table, uu.repeat_interleave(items.numel()), items.repeat(uu.numel()), now).view(len(uu), -1)
            true = s.gather(1, (vv - S.dst_min).view(-1, 1))
            out[b:b + 100] = -(s > true).sum(1).numpy()                # fewer items above v -> more likely a member
        return out

    for m, d in states.items():
        mem, store = d["mem"], d["store"]
        res = dict(cost=d["cost"], cost_x_normal=d["cost"] / max(normal_cost, 1), sec=d["sec"],
                   skipped=d["skipped"], max_skipped_change=d["max_skip"])
        res["resid_cone"], res["resid_endpoints"] = d["resid_cone"], d["resid_endpoints"]
        attacks, rr = {}, []
        g = torch.Generator().manual_seed(4321)        # same MRR negatives for every method
        for j in range(jdel, J):
            a = j - jdel
            if a in after:
                okA = torch.tensor([next_seen.get(p, 10 ** 9) > j for p in A]); okB = torch.tensor([next_seen.get(p, 10 ** 9) > j for p in B])
                now = int(grid.T[j])
                lgA, FA = features(mem, store, uA[okA], vA[okA], now); lgB, FB = features(mem, store, uB[okB], vB[okB], now)
                rec = dict(nA=int(okA.sum()), nB=int(okB.sum()), logit=auc(lgA, lgB))
                if a == 0:
                    if not args.no_rank:
                        rec["rank"] = auc(rank_score(mem, store, uA[okA], vA[okA], now), rank_score(mem, store, uB[okB], vB[okB], now))
                    scores0[m] = (lgA, lgB)
                X = np.r_[FA, FB]; y = np.r_[np.ones(len(FA)), np.zeros(len(FB))]
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
                p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
                rec["probe"] = float(roc_auc_score(y, p))
                attacks[a] = rec
            b = batches[j]
            if b.numel():
                qs, qp = S.src[b], S.dst[b]
                nd = S.dst_max - S.dst_min + 1
                ng = (qp.view(-1, 1) - S.dst_min + torch.randint(1, nd, (qp.numel(), 100), generator=g)) % nd + S.dst_min
                sc = score_pairs(model, mem, store, table, torch.cat([qs, qs.repeat_interleave(100)]), torch.cat([qp, ng.reshape(-1)]), int(grid.T[j]))
                pos, neg = sc[:qp.numel()], sc[qp.numel():].view(qp.numel(), -1)
                rr.append((1.0 / (1 + (neg > pos.view(-1, 1)).sum(1) + 0.5 * (neg == pos.view(-1, 1)).sum(1))).numpy())
                mem.advance(b, journal=False); store.insert(b)
            if a >= max(after):
                break
        res["mrr_after"] = float(np.concatenate(rr).mean()); res["attacks"] = attacks
        results[m] = res
        print(f"{m:>9s}: cost x{res['cost_x_normal']:.2f} | resid cone {res['resid_cone']:.3f} | " +
              " ".join(f"+{a}: logit {v['logit']:.3f} probe {v['probe']:.3f}" + (f" rank {v['rank']:.3f}" if 'rank' in v else '')
                       for a, v in attacks.items()) + f" | MRR after {res['mrr_after']:.4f}", flush=True)
    # paired bootstrap of AUC(method) - AUC(exact) for the logit attack at the request step
    bs = np.random.default_rng(7)
    A0, B0 = scores0["exact"]
    for m, (A1, B1) in scores0.items():
        d = []
        for _ in range(1000):
            ia = bs.integers(0, len(A1), len(A1)); ib = bs.integers(0, len(B1), len(B1))
            d.append(auc(A1[ia], B1[ib]) - auc(A0[ia], B0[ib]))
        results[m]["logit_gap_vs_exact"] = [auc(A1, B1) - auc(A0, B0), float(np.quantile(d, .025)), float(np.quantile(d, .975))]
    summary = dict(config=vars(args), candidates=len(cands), pairs_A=len(A), pairs_B=len(B), deleted_events=int(D.numel()),
                   k0=k0, jdel=jdel, J=J, normal_updates_until_request=normal_cost, cone_nodes=int(cone.numel()),
                   endpoints=int(ep.numel()), methods=results, wall_sec=time.time() - t0)
    out = args.out or f"deletion_{args.model}_p{args.seed}"
    od = paths.RESULTS / "deletion"; od.mkdir(parents=True, exist_ok=True)
    with open(od / f"{out}.json", "w") as f:
        json.dump(summary, f, indent=1)
    np.savez(od / f"{out}_scores.npz", **{f"{m}__A": v[0] for m, v in scores0.items()}, **{f"{m}__B": v[1] for m, v in scores0.items()})
    for m in results:
        gp = results[m]["logit_gap_vs_exact"]
        print(f"  AUC({m}) - AUC(exact) = {gp[0]:+.3f} [{gp[1]:+.3f}, {gp[2]:+.3f}]")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
