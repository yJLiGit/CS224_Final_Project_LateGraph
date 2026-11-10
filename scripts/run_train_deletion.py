"""E4c: deleting events of the training period, which both the weights and the memory remember.

Design. Pairs (u, v) all of whose events lie in the training period, whose user has >= 3 other events before the test
period. A random half (A) is trained on and processed; the other half (B) is removed from the training data and from
the stream (never seen). At the deployment start (the first test step) the A pairs must be forgotten. Methods:
  none      weights trained with A, memory with A, neighbour list with A (no deletion)
  store     delete from the log / neighbour store only
  memory    exact repair of the memory (the weights are unchanged): memory unlearning only
  finetune  memory repair + one epoch of fine-tuning the weights on the data without A (approximate weight unlearning)
  retrain   train from scratch without A and rebuild the memory (exact unlearning, the reference)
The same attacks as run_deletion.py (link score, rank of v among all items, memory probe with 5-fold CV) at the first
test step; utility is the MRR over the next 50 steps. The memory repair's cost is its cone in the (journaled) history.

  python scripts/run_train_deletion.py --data wikipedia --seed 0
"""
import argparse
import copy
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
from lategraph import paths                                              # noqa: E402
from lategraph.data import BatchGrid, load                               # noqa: E402
from lategraph.memory import EventTable, RetroactiveTGNMemory            # noqa: E402
from lategraph.model import TGNConfig, load_checkpoint                   # noqa: E402
from lategraph.neighbors import TimeOrderedNeighborStore                 # noqa: E402
from lategraph.stream import score_pairs                                 # noqa: E402
from lategraph.train import train_tgn                                    # noqa: E402


def auc(a, b):
    return float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=500); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ft_epochs", type=int, default=1); ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--mrr_steps", type=int, default=50); ap.add_argument("--out", default=None)
    ap.add_argument("--recent", type=float, default=0.0,
                    help="only pairs whose events all lie in the last `recent` share of the training period (e.g. 0.2)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    S = load(args.data)
    src, dst = S.src.numpy(), S.dst.numpy()
    rng = np.random.default_rng(args.seed)

    # ---- candidate pairs: every event in the training period, user with >= 3 other events before the test period
    pair_events, last_pos = {}, {}
    for e in range(S.num_events):
        p = (int(src[e]), int(dst[e]))
        pair_events.setdefault(p, []).append(e); last_pos[p] = e
    ucnt = np.bincount(src[:S.test0], minlength=S.num_nodes)
    lo = int((1 - args.recent) * S.val0) if args.recent else 0
    cands = [p for p, ev in pair_events.items() if last_pos[p] < S.val0 and ev[0] >= lo and ucnt[p[0]] - len(ev) >= 3]
    perm = rng.permutation(len(cands))[: 2 * min(args.K, len(cands) // 2)]
    A = [cands[i] for i in perm[0::2]]; B = [cands[i] for i in perm[1::2]]
    evA = torch.tensor(sorted(e for p in A for e in pair_events[p]), dtype=torch.long)
    evB = torch.tensor(sorted(e for p in B for e in pair_events[p]), dtype=torch.long)
    inA = torch.zeros(S.num_events, dtype=torch.bool); inA[evA] = True
    inB = torch.zeros(S.num_events, dtype=torch.bool); inB[evB] = True
    print(f"[{time.time() - t0:.0f}s] {args.data}: {len(cands)} candidate pairs; A {len(A)} pairs ({evA.numel()} events), "
          f"B {len(B)} pairs ({evB.numel()} events)", flush=True)

    # ---- the three models: trained with A (full), without A (reference), fine-tuned from full without A
    tag = f"tgn_{args.data}_s{args.seed}_trdel" + (f"_recent{args.recent:g}" if args.recent else "")
    log = lambda m: print(m, flush=True)
    cfg = TGNConfig()
    S_full, S_wo = S.subset(~inB, "full"), S.subset(~(inA | inB), "without A")
    models, train_sec = {}, {}
    for name, stream in (("full", S_full), ("retrain", S_wo)):
        ck = paths.CKPT / f"{tag}_{name}.pt"
        tt = time.time()
        if ck.exists():
            models[name] = load_checkpoint(ck)
        else:
            m, _ = train_tgn(stream, copy.deepcopy(cfg), epochs=args.epochs, seed=args.seed, threads=args.threads,
                             ckpt_path=ck, log=log, meta=dict(tag=f"{tag}_{name}"))
            models[name] = m
        train_sec[name] = time.time() - tt
    tt = time.time()
    m_ft, _ = train_tgn(S_wo, copy.deepcopy(cfg), epochs=args.ft_epochs, seed=args.seed, threads=args.threads,
                        log=log, init_state=models["full"].state_dict())
    models["finetune"] = m_ft; train_sec["finetune"] = time.time() - tt
    print(f"[{time.time() - t0:.0f}s] models ready (train min: " + ", ".join(f"{k} {v / 60:.1f}" for k, v in train_sec.items()) + ")", flush=True)

    # ---- memories at the deployment start (history = train + val, in fixed batches of 200 by position)
    grid = BatchGrid(S.t, S.test0, 200)
    step = torch.full((S.num_events,), -1, dtype=torch.long)
    step[:S.test0] = torch.arange(S.test0) // 200 - (S.test0 + 199) // 200        # history steps: negative, ordered
    test = torch.arange(S.test0, S.num_events)
    step[test] = grid.step_of_time(S.t[test])
    table = EventTable(S.src, S.dst, S.t, S.msg, step)
    batches = [torch.arange(b, min(b + 200, S.test0)) for b in range(0, S.test0, 200)]

    def memory(model, drop: torch.Tensor):
        """The model's memory after the history without the events `drop` (same batches, so the same batching)."""
        mem = RetroactiveTGNMemory.from_trained(model.memory, table)
        for b in batches:
            mem.apply(b[~drop[b]])
        return mem

    def store(drop):
        st = TimeOrderedNeighborStore(S.src, S.dst, S.t, S.num_nodes, 10)
        keep = torch.arange(S.num_events) < S.test0
        st.rebuild_from(keep & ~drop)
        return st

    full = models["full"]
    states = {
        "none": (full, memory(full, inB), store(inB)),
        "store": (full, memory(full, inB), store(inA | inB)),
        "memory": (full, memory(full, inA | inB), store(inA | inB)),
        "finetune": (models["finetune"], memory(models["finetune"], inA | inB), store(inA | inB)),
        "retrain": (models["retrain"], memory(models["retrain"], inA | inB), store(inA | inB)),
    }
    # cost of the memory repair: the cone of the deletion in a journaled history (relative to normal processing)
    jm = RetroactiveTGNMemory.from_trained(full.memory, table)
    jm.step = int(step[0])
    for b in batches:
        jm.advance(b[~inB[b]])
    jm.edit(delete=evA)
    tt = time.time(); rs = jm.repair("exact"); repair_sec = time.time() - tt
    exact_ok = float((jm.memory - states["memory"][1].memory).abs().max())
    # one request at a time: the cone of deleting a single A pair (share of the history updates after its first event)
    single = []
    one = RetroactiveTGNMemory.from_trained(full.memory, table)
    jm0 = RetroactiveTGNMemory.from_trained(full.memory, table); jm0.step = int(step[0])
    for b in batches:
        jm0.advance(b[~inB[b]])
    for i in rng.permutation(len(A))[:100]:
        ev = torch.tensor(pair_events[A[i]], dtype=torch.long)
        one.copy_state_from(jm0); one.edit(delete=ev)
        tt = time.time(); r1 = one.repair("exact")
        single.append((r1.recomputed / max(r1.window_updates, 1), r1.recomputed, time.time() - tt, r1.extra.get("visited_steps", 0), r1.steps))
    single = np.array(single)
    print(f"[{time.time() - t0:.0f}s] memory repair: {rs.recomputed} recomputed updates of {rs.window_updates} in the window "
          f"({jm.normal_updates} in the history), {repair_sec:.1f}s; max |repaired - rebuilt| = {exact_ok:.2e}; one pair at a time: "
          f"median cone {100 * np.median(single[:, 0]):.2f}% of the updates since its first event, {1000 * np.median(single[:, 2]):.0f} ms", flush=True)

    # ---- attacks at the first test step, and the MRR that follows
    uA = torch.tensor([p[0] for p in A]); vA = torch.tensor([p[1] for p in A])
    uB = torch.tensor([p[0] for p in B]); vB = torch.tensor([p[1] for p in B])
    items = torch.arange(S.dst_min, S.dst_max + 1)
    now = int(grid.T[0])

    def features(model, mem, st, u, v):
        lg = score_pairs(model, mem, st, table, u, v, now).numpy()
        su, sv = mem.memory[u], mem.memory[v]
        cos = torch.nn.functional.cosine_similarity(su, sv, dim=1, eps=1e-12)
        F = torch.cat([su * sv, (su - sv).abs(), cos.view(-1, 1), su.norm(dim=1, keepdim=True), sv.norm(dim=1, keepdim=True)], 1)
        return lg, np.c_[F.numpy(), lg]

    def rank_score(model, mem, st, u, v):
        out = np.empty(len(u))
        for b in range(0, len(u), 100):
            uu, vv = u[b:b + 100], v[b:b + 100]
            s = score_pairs(model, mem, st, table, uu.repeat_interleave(items.numel()), items.repeat(uu.numel()), now).view(len(uu), -1)
            out[b:b + 100] = -(s > s.gather(1, (vv - S.dst_min).view(-1, 1))).sum(1).numpy()
        return out

    res = {}
    g = torch.Generator().manual_seed(4321)
    negs = []
    nd = S.dst_max - S.dst_min + 1
    for j in range(min(args.mrr_steps, grid.J)):
        b = test[step[test] == j]
        negs.append((b, (S.dst[b].view(-1, 1) - S.dst_min + torch.randint(1, nd, (b.numel(), 100), generator=g)) % nd + S.dst_min))
    for m, (model, mem, st) in states.items():
        lgA, FA = features(model, mem, st, uA, vA); lgB, FB = features(model, mem, st, uB, vB)
        X = np.r_[FA, FB]; y = np.r_[np.ones(len(FA)), np.zeros(len(FB))]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        pr = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        rec = dict(logit=auc(lgA, lgB), rank=auc(rank_score(model, mem, st, uA, vA), rank_score(model, mem, st, uB, vB)),
                   probe=float(roc_auc_score(y, pr)))
        mm = RetroactiveTGNMemory.from_trained(model.memory, table).copy_state_from(mem); ss = st.clone(); rr = []
        for b, ng in negs:
            if b.numel():
                qs, qp = S.src[b], S.dst[b]
                sc = score_pairs(model, mm, ss, table, torch.cat([qs, qs.repeat_interleave(100)]), torch.cat([qp, ng.reshape(-1)]),
                                 int(grid.T[int(step[b[0]])]))
                pos, neg = sc[:qp.numel()], sc[qp.numel():].view(qp.numel(), -1)
                rr.append((1.0 / (1 + (neg > pos.view(-1, 1)).sum(1) + 0.5 * (neg == pos.view(-1, 1)).sum(1))).numpy())
                mm.apply(b); ss.insert(b)
        rec["mrr"] = float(np.concatenate(rr).mean())
        res[m] = rec
        print(f"{m:>9s}: logit {rec['logit']:.3f} rank {rec['rank']:.3f} probe {rec['probe']:.3f} | MRR {rec['mrr']:.4f}", flush=True)
    summary = dict(config=vars(args), candidates=len(cands), pairs_A=len(A), pairs_B=len(B), events_A=int(evA.numel()),
                   events_B=int(evB.numel()), train_minutes={k: v / 60 for k, v in train_sec.items()},
                   memory_repair=dict(recomputed=rs.recomputed, window_updates=rs.window_updates, history_updates=jm.normal_updates,
                                      sec=repair_sec, max_abs_vs_rebuild=exact_ok,
                                      single_pair=dict(cone_frac_median=float(np.median(single[:, 0])), recomputed_median=float(np.median(single[:, 1])),
                                                       sec_median=float(np.median(single[:, 2])), visited_median=float(np.median(single[:, 3])),
                                                       window_steps_median=float(np.median(single[:, 4])))),
                   methods=res, wall_sec=time.time() - t0)
    od = paths.RESULTS / "train_deletion"; od.mkdir(parents=True, exist_ok=True)
    with open(od / f"{args.out or tag}.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
