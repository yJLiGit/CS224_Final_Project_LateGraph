"""E4b: the strongest membership attack after a deletion request: a likelihood-ratio test with counterfactual shadow
streams (LiRA, Carlini et al. 2022, online variant with fixed variance).

Threat model (worst case for the defender). The attacker knows the deployed weights and every event of the stream except,
for each candidate pair, whether it was observed (and then deleted with method m) or never observed. Shadow runs: R
random halves I_r of the candidate pairs are observed up to the request step and then deleted with m; every other
candidate is never observed. For each pair p this gives scores under IN (p in I_r) and OUT, which are modelled as
Gaussians (per-pair means, one pooled variance per side). The target is the deployed memory, where the pairs A were
observed and deleted with m and the pairs B were never observed (the same A/B split as run_deletion.py with the same
seed). The statistic is
        Lambda(p) = log N(s_p; mu_in(p), var_in) - log N(s_p; mu_out(p), var_out),
with s_p the model's own link score of p (black box) or the cosine between the two memories (white box). We report the
AUC of Lambda for A against B and the true-positive rate at 5% and 1% false positives.

After exact repair the IN and OUT states are identical (the deleted events never happened), so every statistic is the
same for A and B and the attack is at chance by construction.

Weaker attacker (--know f < 1): the attacker knows only a random fraction f of the other test-period events (the history
before the test period stays known); its shadow streams leave the unknown events out, while the target is still the
deployed memory of the true stream. Results go to results/lira_know/.

  python scripts/run_lira.py --data wikipedia --model tgn_wikipedia_s0 --seed 0 --shadows 32
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph import paths                                                       # noqa: E402
from lategraph.data import BatchGrid, load                                        # noqa: E402
from lategraph.memory import EventTable, RetroactiveTGNMemory, khop_mask_for      # noqa: E402
from lategraph.model import load_checkpoint                                       # noqa: E402
from lategraph.neighbors import TimeOrderedNeighborStore                          # noqa: E402
from lategraph.stream import score_pairs                                          # noqa: E402

METHODS = "store,reset,restart,eps:0.05,khop:1,exact"


def select_pairs(S, step, jdel, K, rng):
    """A/B pairs exactly as in run_deletion.py: pairs new in the test period before the request whose user has >= 3
    other events; a random half is observed (A), the other half never observed (B)."""
    src, dst = S.src.numpy(), S.dst.numpy()
    pre_end = int((step >= 0).logical_and(step < jdel).nonzero().max()) + 1
    hist_pairs = set(zip(src[:S.test0].tolist(), dst[:S.test0].tolist()))
    pair_events = {}
    for e in range(S.test0, pre_end):
        pair_events.setdefault((int(src[e]), int(dst[e])), []).append(e)
    ucnt = np.bincount(src[:pre_end], minlength=S.num_nodes)
    cands = [p for p, ev in pair_events.items() if p not in hist_pairs and ucnt[p[0]] - len(ev) >= 3]
    perm = rng.permutation(len(cands))[: 2 * min(K, len(cands) // 2)]
    A = [cands[i] for i in perm[0::2]]; B = [cands[i] for i in perm[1::2]]
    return A, B, pair_events, pre_end


def tpr_at(stat_a, stat_b, fpr):
    thr = np.quantile(stat_b, 1 - fpr)
    return float((stat_a > thr).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--bq", type=int, default=200); ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--jdel", type=float, default=0.5); ap.add_argument("--shadows", type=int, default=32)
    ap.add_argument("--methods", default=METHODS); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--out", default=None)
    ap.add_argument("--know", type=float, default=1.0, help="fraction of the other test-period events the attacker knows")
    ap.add_argument("--after", type=int, default=0, help="attack this many steps after the request (the stream goes on)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
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
    A, B, pair_events, pre_end = select_pairs(S, step, jdel, args.K, np.random.default_rng(args.seed))
    P = A + B
    y = np.r_[np.ones(len(A)), np.zeros(len(B))]
    pev = [torch.tensor(pair_events[p], dtype=torch.long) for p in P]
    u = torch.tensor([p[0] for p in P]); v = torch.tensor([p[1] for p in P])
    # pairs seen again at the request step cannot be tested there (as in run_deletion.py)
    seen_at = {}
    for e in range(pre_end, min(S.num_events, pre_end + 5 * args.bq)):
        seen_at.setdefault((int(S.src[e]), int(S.dst[e])), int(step[e]))
    ok = np.array([seen_at.get(p, 10 ** 9) > jdel for p in P])
    now = int(grid.T[min(jdel + args.after, J)])
    step_ev = [test[step[test] == j] for j in range(jdel)]
    # --after k: the stream continues for k steps after the request, without the candidate pairs' later events
    pairs_set = set(P)
    later = [test[step[test] == j] for j in range(jdel, min(jdel + args.after, J))]
    later = [ev[torch.tensor([(int(S.src[e]), int(S.dst[e])) not in pairs_set for e in ev.tolist()], dtype=torch.bool)]
             if ev.numel() else ev for ev in later]
    # events the attacker does not know (only with --know < 1): a fixed random part of the other test-period events
    unknown = torch.zeros(S.num_events, dtype=torch.bool)
    if args.know < 1:
        cand_ev = torch.cat(pev)
        bg = torch.arange(S.test0, pre_end)
        bg = bg[~torch.isin(bg, cand_ev)]
        unknown[bg[torch.from_numpy(np.random.default_rng(2000 + args.seed).random(bg.numel()) >= args.know)]] = True
    print(f"[{time.time() - t0:.0f}s] {args.data}: {len(A)} + {len(B)} candidate pairs, request at step {jdel}/{J}, "
          f"{args.shadows} shadow streams", flush=True)

    warm = RetroactiveTGNMemory.from_trained(model.memory, table)
    warm.warmup(0, S.test0)
    hist_store = TimeOrderedNeighborStore(S.src, S.dst, S.t, S.num_nodes, 10)
    hist_store.rebuild_from(torch.arange(S.num_events) < S.test0)
    _R = {}

    def restarter():
        if "R" not in _R:
            from lategraph.restarter import Restarter
            R = Restarter(model.memory.memory_dim, S.msg_dim)
            R.load_state_dict(torch.load(paths.CKPT / f"{args.model}_restarter.pt")); R.eval()
            _R["R"] = R
        return _R["R"]

    def state(observed: np.ndarray, attacker: bool = False):
        """Memory and neighbour store at the request step when only the candidate pairs `observed` were observed
        (attacker=True: as simulated by the attacker, without the events it does not know)."""
        hidden = unknown.clone() if attacker else torch.zeros(S.num_events, dtype=torch.bool)
        for i in np.flatnonzero(~observed):
            hidden[pev[i]] = True
        mem = RetroactiveTGNMemory.from_trained(model.memory, table).copy_state_from(warm)
        for j in range(jdel):
            mem.advance(step_ev[j][~hidden[step_ev[j]]])
        store = hist_store.clone()
        ins = torch.arange(S.test0, pre_end); store.insert(ins[~hidden[ins]])
        D = torch.cat([pev[i] for i in np.flatnonzero(observed)]) if observed.any() else torch.empty(0, dtype=torch.long)
        k0 = int(step[D].min()) if D.numel() else jdel
        win = torch.cat([step_ev[j][~hidden[step_ev[j]]] for j in range(k0, jdel)] + [D[:0]])
        return mem, store, torch.sort(D).values, win

    def delete_with(method, mem0, store0, D, window_events):
        mem = RetroactiveTGNMemory.from_trained(model.memory, table).copy_state_from(mem0)
        store = store0.clone(); store.remove(D)
        kind, _, arg = method.partition(":")
        if kind == "reset":
            ep = torch.cat([S.src[D], S.dst[D]]).unique(); mem.memory[ep] = 0; mem.last_update[ep] = 0
        elif kind == "restart":
            nodes = torch.cat([S.src[D], S.dst[D]]).unique()
            with torch.no_grad():
                mem.memory[nodes] = restarter()(nodes, mem.memory, store, table, now)
            e0 = store.e_id[nodes, 0]
            mem.last_update[nodes] = torch.where(e0 >= 0, table.t[e0.clamp(min=0)], torch.zeros_like(e0))
        elif kind != "store":
            mem.edit(delete=D)
            mask = None
            if kind == "khop":
                seeds = torch.cat([S.src[D], S.dst[D]]).unique()
                mask = khop_mask_for(table, seeds, window_events, int(arg), S.num_nodes)
            mem.repair(kind, eps=float(arg) if kind == "eps" else 0.0, khop_mask=mask)
        return mem, store

    def go_on(mem, store):
        """--after k: process the k steps after the request (memory and neighbour store), then attack."""
        for ev in later:
            mem.advance(ev); store.insert(ev)
        return mem, store

    def stats(mem, store):
        mem, store = go_on(mem, store)
        lg = score_pairs(model, mem, store, table, u, v, now).numpy()
        cos = torch.nn.functional.cosine_similarity(mem.memory[u], mem.memory[v], dim=1, eps=1e-12).numpy()
        return np.stack([lg, cos])                                            # [2, |P|]

    methods = args.methods.split(",")
    if "restart" in methods and not (paths.CKPT / f"{args.model}_restarter.pt").exists():
        methods.remove("restart"); print("  (no restarter checkpoint for this model: 'restart' skipped)", flush=True)
    # target: A observed and deleted with each method, B never observed
    obs_t = np.r_[np.ones(len(A), bool), np.zeros(len(B), bool)]
    m0, s0, D0, w0 = state(obs_t)
    target = {m: stats(*delete_with(m, m0, s0, D0, w0)) for m in methods}
    target["none"] = stats(RetroactiveTGNMemory.from_trained(model.memory, table).copy_state_from(m0), s0.clone())
    print(f"[{time.time() - t0:.0f}s] target states done", flush=True)
    rng = np.random.default_rng(1000 + args.seed)
    IN = np.zeros((args.shadows, len(P)), bool)
    SC = {m: np.zeros((args.shadows, 2, len(P))) for m in methods}
    for r in range(args.shadows):
        IN[r] = rng.random(len(P)) < 0.5
        mr, sr, Dr, wr = state(IN[r], attacker=True)
        for m in methods:
            SC[m][r] = stats(*delete_with(m, mr, sr, Dr, wr))
        if r % 8 == 7:
            print(f"[{time.time() - t0:.0f}s] shadow {r + 1}/{args.shadows}", flush=True)

    res = {}
    for m in methods:
        out = {}
        for f, fname in enumerate(("logit", "cosine")):
            sc, s = SC[m][:, f, :], target[m][f]
            cnt_in, cnt_out = IN.sum(0), (~IN).sum(0)
            mu_in = np.where(IN, sc, 0).sum(0) / np.maximum(cnt_in, 1)
            mu_out = np.where(~IN, sc, 0).sum(0) / np.maximum(cnt_out, 1)
            floor = 1e-12 + 1e-6 * float(np.var(sc))                         # exact repair: all shadows are identical
            var_in = max(float(np.where(IN, (sc - mu_in) ** 2, 0).sum() / max(IN.sum() - len(P), 1)), floor)
            var_out = max(float(np.where(~IN, (sc - mu_out) ** 2, 0).sum() / max((~IN).sum() - len(P), 1)), floor)
            lam = -(s - mu_in) ** 2 / (2 * var_in) + (s - mu_out) ** 2 / (2 * var_out) - 0.5 * np.log(var_in / var_out)
            good = ok & (cnt_in > 0) & (cnt_out > 0)
            la, lb = lam[good & (y == 1)], lam[good & (y == 0)]
            off = np.abs(s - mu_out) / np.sqrt(var_out)                       # offline: distance from the OUT mean
            out[fname] = dict(auc=float(roc_auc_score(np.r_[np.ones(len(la)), np.zeros(len(lb))], np.r_[la, lb])),
                              tpr5=tpr_at(la, lb, 0.05), tpr1=tpr_at(la, lb, 0.01),
                              auc_offline=float(roc_auc_score(np.r_[np.ones(len(la)), np.zeros(len(lb))],
                                                              np.r_[off[good & (y == 1)], off[good & (y == 0)]])),
                              in_out_gap=float(np.mean(np.abs(mu_in - mu_out)[good]) / np.sqrt(var_out)),
                              nA=int(len(la)), nB=int(len(lb)))
        # the plain attack (no shadows): the raw link score of A against B
        s = target[m][0]
        out["raw_logit_auc"] = float(roc_auc_score(np.r_[np.ones(int((ok & (y == 1)).sum())), np.zeros(int((ok & (y == 0)).sum()))],
                                                   np.r_[s[ok & (y == 1)], s[ok & (y == 0)]]))
        res[m] = out
        print(f"{m:>9s}: LiRA logit AUC {out['logit']['auc']:.3f} (TPR@1%FPR {out['logit']['tpr1']:.3f}) | "
              f"LiRA cosine AUC {out['cosine']['auc']:.3f} (TPR@1% {out['cosine']['tpr1']:.3f}) | raw logit AUC {out['raw_logit_auc']:.3f}",
              flush=True)
    summary = dict(config=vars(args), pairs_A=len(A), pairs_B=len(B), tested=int(ok.sum()), jdel=jdel, J=J,
                   unknown_events=int(unknown.sum()),
                   methods=res, wall_sec=time.time() - t0)
    tag = (f"_know{args.know:g}" if args.know < 1 else "") + (f"_after{args.after}" if args.after else "") + (f"_K{args.K}" if args.K != 1000 else "")
    out = args.out or f"lira_{args.model}_p{args.seed}{tag}"
    od = paths.RESULTS / ("lira" if not tag else "lira_know"); od.mkdir(parents=True, exist_ok=True)
    with open(od / f"{out}.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
