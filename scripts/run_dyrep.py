"""DyRep-style memory: test MRR, exactness and cone size of repair, and the deletion leak (with a LiRA attack).

  python scripts/run_dyrep.py --data wikipedia --model dyrep_wikipedia_s0 --seed 0
Parts (results/dyrep/<model>.json):
  mrr        in-order test MRR (100 random negatives), as for the TGN models
  exactness  random single insertions/deletions in a 100-step window with 5% of the events missing: exact repair vs
             rollback + full replay (max difference, recomputed updates, wall-clock)
  deletion   the A/B design of run_deletion.py at the middle of the test period: no deletion, log-only deletion, exact
             repair; link-score and memory-probe AUC, and a LiRA attack with counterfactual shadow streams
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
from lategraph import paths                                         # noqa: E402
from lategraph.data import BatchGrid, load                          # noqa: E402
from lategraph.dyrep import RetroactiveDyRepMemory, load_dyrep       # noqa: E402
from lategraph.memory import EventTable                             # noqa: E402


def auc(a, b):
    return float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))


@torch.no_grad()
def scores(model, mem: RetroactiveDyRepMemory, store, table, u, v):
    n_id = torch.cat([u, v]).unique()
    all_id, edge_index, e_id = store(n_id)
    assoc = torch.empty(mem.N, dtype=torch.long); assoc[all_id] = torch.arange(all_id.numel())
    z = model.embed(mem.memory[all_id], mem.last_update[all_id], edge_index, table.t[e_id], table.msg[e_id])
    return model.score(z[assoc[u]], z[assoc[v]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--K", type=int, default=1000); ap.add_argument("--shadows", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    S = load(args.data)
    model = load_dyrep(paths.CKPT / f"{args.model}.pt")
    grid = BatchGrid(S.t, S.test0, 200)
    J = grid.J
    step = torch.full((S.num_events,), -1, dtype=torch.long)
    test = torch.arange(S.test0, S.num_events); step[test] = grid.step_of_time(S.t[test])
    table = EventTable(S.src, S.dst, S.t, S.msg, step)
    rng = np.random.default_rng(args.seed)
    base0 = RetroactiveDyRepMemory(model, table, S.test0)
    out = dict(config=vars(args), val_ap=model.meta.get("best_val_ap"))
    steps_ev = [test[step[test] == j] for j in range(J)]

    # ---- (1) in-order test MRR
    g = torch.Generator().manual_seed(1234 + args.seed)
    m = base0.clone()
    rr = []; nd = S.dst_max - S.dst_min + 1
    for j in range(J):
        b = steps_ev[j]
        if b.numel():
            qs, qp = S.src[b], S.dst[b]
            ng = (qp.view(-1, 1) - S.dst_min + torch.randint(1, nd, (qp.numel(), 100), generator=g)) % nd + S.dst_min
            sc = scores(model, m, m.store, table, torch.cat([qs, qs.repeat_interleave(100)]), torch.cat([qp, ng.reshape(-1)]))
            pos, neg = sc[:qp.numel()], sc[qp.numel():].view(qp.numel(), -1)
            rr.append((1.0 / (1 + (neg > pos.view(-1, 1)).sum(1) + 0.5 * (neg == pos.view(-1, 1)).sum(1))).numpy())
        m.advance(b)
    out["mrr"] = float(np.concatenate(rr).mean())
    print(f"[{time.time() - t0:.0f}s] {args.model}: val AP {out['val_ap']:.4f}, test MRR {out['mrr']:.4f}", flush=True)

    # ---- (2) exactness and cone size: 100-step window, 5% of its events missing (to be inserted)
    W = min(100, J)
    win = test[step[test] < W]
    missing = torch.zeros(S.num_events, dtype=torch.bool); missing[win[torch.from_numpy(rng.random(win.numel()) < 0.05)]] = True
    base = base0.clone()
    for j in range(W):
        base.advance(steps_ev[j][~missing[steps_ev[j]]])
    work = base.clone()
    present, absent = win[~missing[win]].numpy(), win[missing[win]].numpy()
    rows = []
    for i in range(args.trials):
        ins = torch.tensor([int(rng.choice(absent))]) if i % 2 else torch.empty(0, dtype=torch.long)
        dele = torch.empty(0, dtype=torch.long) if i % 2 else torch.tensor([int(rng.choice(present))])
        work.copy_state_from(base); work.edit(insert=ins, delete=dele)
        tt = time.perf_counter(); se = work.repair("exact"); te = time.perf_counter() - tt; me = work.memory.clone()
        work.copy_state_from(base); work.edit(insert=ins, delete=dele)
        tt = time.perf_counter(); sr = work.repair("replay"); tr = time.perf_counter() - tt
        rows.append(dict(diff=float((me - work.memory).abs().max()), cone=se.recomputed, window=sr.recomputed,
                         frac=se.recomputed / max(sr.recomputed, 1), exact_sec=te, replay_sec=tr, back=se.steps))
    out["exactness"] = dict(trials=len(rows), max_abs_diff=max(r["diff"] for r in rows),
                            cone_frac_median=float(np.median([r["frac"] for r in rows])),
                            cone_frac_mean=float(np.mean([r["frac"] for r in rows])),
                            cone_zero_frac=float(np.mean([r["cone"] == 0 for r in rows])),
                            speedup_median=float(np.median([r["replay_sec"] / max(r["exact_sec"], 1e-9) for r in rows])))
    e = out["exactness"]
    print(f"[{time.time() - t0:.0f}s] exactness: {e['trials']} trials, max |exact - replay| {e['max_abs_diff']:.2e}, median cone "
          f"{100 * e['cone_frac_median']:.2f}% of the window (mean {100 * e['cone_frac_mean']:.1f}%), no effect in "
          f"{100 * e['cone_zero_frac']:.0f}%", flush=True)

    # ---- (3) deletion leak at the middle of the test period
    jdel = J // 2
    src, dst = S.src.numpy(), S.dst.numpy()
    pre_end = int((step >= 0).logical_and(step < jdel).nonzero().max()) + 1
    hist_pairs = set(zip(src[:S.test0].tolist(), dst[:S.test0].tolist()))
    pair_events = {}
    for ev in range(S.test0, pre_end):
        pair_events.setdefault((int(src[ev]), int(dst[ev])), []).append(ev)
    ucnt = np.bincount(src[:pre_end], minlength=S.num_nodes)
    cands = [p for p, ev in pair_events.items() if p not in hist_pairs and ucnt[p[0]] - len(ev) >= 3]
    perm = np.random.default_rng(args.seed).permutation(len(cands))[: 2 * min(args.K, len(cands) // 2)]
    A = [cands[i] for i in perm[0::2]]; B = [cands[i] for i in perm[1::2]]
    P = A + B; pev = [torch.tensor(pair_events[p], dtype=torch.long) for p in P]
    u = torch.tensor([p[0] for p in P]); v = torch.tensor([p[1] for p in P]); y = np.r_[np.ones(len(A)), np.zeros(len(B))]

    def state(observed):
        hidden = torch.zeros(S.num_events, dtype=torch.bool)
        for i in np.flatnonzero(~observed):
            hidden[pev[i]] = True
        mm = base0.clone()
        for j in range(jdel):
            mm.advance(steps_ev[j][~hidden[steps_ev[j]]])
        D = torch.sort(torch.cat([pev[i] for i in np.flatnonzero(observed)])).values if observed.any() else torch.empty(0, dtype=torch.long)
        return mm, D

    def delete_with(method, mm, D):
        w = mm.clone()
        store = w.store.clone()
        if method in ("store", "exact"):
            store.remove(D)
        if method == "exact":
            w.edit(delete=D); w.repair("exact"); store = w.store
        return w, store

    def feats(w, store):
        lg = scores(model, w, store, table, u, v).numpy()
        su, sv = w.memory[u], w.memory[v]
        cos = torch.nn.functional.cosine_similarity(su, sv, dim=1, eps=1e-12)
        F = torch.cat([su * sv, (su - sv).abs(), cos.view(-1, 1), su.norm(dim=1, keepdim=True), sv.norm(dim=1, keepdim=True)], 1)
        return lg, cos.numpy(), np.c_[F.numpy(), lg]

    methods = ("none", "store", "exact")
    mt, Dt = state(np.r_[np.ones(len(A), bool), np.zeros(len(B), bool)])
    target, dres = {}, {}
    for meth in methods:
        w, store = delete_with(meth, mt, Dt)
        lg, cos, X = feats(w, store)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        pr = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        dres[meth] = dict(logit=auc(lg[y == 1], lg[y == 0]), probe=float(roc_auc_score(y, pr)))
        target[meth] = np.stack([lg, cos])
    rs = np.random.default_rng(1000 + args.seed)
    IN = np.zeros((args.shadows, len(P)), bool); SC = {mm_: np.zeros((args.shadows, 2, len(P))) for mm_ in ("store", "exact")}
    for r in range(args.shadows):
        IN[r] = rs.random(len(P)) < 0.5
        mr, Dr = state(IN[r])
        for meth in ("store", "exact"):
            w, store = delete_with(meth, mr, Dr)
            lg, cos, _ = feats(w, store)
            SC[meth][r] = np.stack([lg, cos])
    for meth in ("store", "exact"):
        for f, fname in enumerate(("logit", "cosine")):
            sc, s = SC[meth][:, f, :], target[meth][f]
            ci, co = IN.sum(0), (~IN).sum(0)
            mu_in = np.where(IN, sc, 0).sum(0) / np.maximum(ci, 1); mu_out = np.where(~IN, sc, 0).sum(0) / np.maximum(co, 1)
            floor = 1e-12 + 1e-6 * float(np.var(sc))
            vi = max(float(np.where(IN, (sc - mu_in) ** 2, 0).sum() / max(IN.sum() - len(P), 1)), floor)
            vo = max(float(np.where(~IN, (sc - mu_out) ** 2, 0).sum() / max((~IN).sum() - len(P), 1)), floor)
            lam = -(s - mu_in) ** 2 / (2 * vi) + (s - mu_out) ** 2 / (2 * vo) - 0.5 * np.log(vi / vo)
            good = (ci > 0) & (co > 0)
            dres[meth][f"lira_{fname}"] = auc(lam[good & (y == 1)], lam[good & (y == 0)])
    out["deletion"] = dict(pairs_A=len(A), pairs_B=len(B), shadows=args.shadows, methods=dres)
    for meth, r in dres.items():
        print(f"  {meth:>6}: logit {r['logit']:.3f} probe {r['probe']:.3f}" + (f" LiRA logit {r['lira_logit']:.3f} cosine {r['lira_cosine']:.3f}" if "lira_logit" in r else ""), flush=True)
    out["wall_sec"] = time.time() - t0
    od = paths.RESULTS / "dyrep"; od.mkdir(parents=True, exist_ok=True)
    with open(od / f"{args.model}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
