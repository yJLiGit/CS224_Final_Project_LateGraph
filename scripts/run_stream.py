"""Streaming evaluation under late arrivals: every repair policy on one dataset / model / delay setting.

  python scripts/run_stream.py --data wikipedia --model tgn_wikipedia_s0 --delay rand --seed 0
  python scripts/run_stream.py --data mooc --model tgn_mooc_s1 --delay active --bq 50 --methods oracle,exact,eps:0.05,arrival

Writes results/stream/<out>.json (summary, paired bootstrap CIs vs. exact repair, strata) and <out>.npz (per query).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph import paths                                                  # noqa: E402
from lategraph.data import BatchGrid, load                                   # noqa: E402
from lategraph.delays import arrival_times, make_delays                      # noqa: E402
from lategraph.model import load_checkpoint                                  # noqa: E402
from lategraph.stats import paired_diff_ci                                   # noqa: E402
from lategraph.stream import (Policy, evaluate_stream, in_transit_flags, make_setup,   # noqa: E402
                              sample_negatives, warm_start)

DEFAULT = "oracle,exact,replay,eps:0.01,eps:0.05,eps:0.2,khop:1,khop:2,arrival,arrival_proc,drop,buffer:1,buffer:4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--delay", default="rand"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=float, default=0.2); ap.add_argument("--med", type=float, default=2.0)
    ap.add_argument("--bq", type=int, default=200); ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--methods", default=DEFAULT); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--amend", type=int, default=1, help="SEC: apply Form 4/A amendments as edits")
    ap.add_argument("--max_queries", type=int, default=None, help="score at most this many queries per step (fixed sample)")
    ap.add_argument("--no_index", action="store_true", help="repair by walking every step (timing comparison)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = args.out or f"{args.model}_{args.delay}_bq{args.bq}_d{args.seed}"
    out_dir = paths.RESULTS / "stream"; out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    S = load(args.data)
    model = load_checkpoint(paths.CKPT / f"{args.model}.pt")
    if S.arrival is not None and args.delay == "real":
        arrival = S.arrival
    else:
        delay = make_delays(S, args.delay, seed=args.seed, p=args.p, med=args.med)
        arrival = arrival_times(S, delay)
    if args.data.startswith("sec") or args.data.startswith("tgbl-"):   # one step per calendar day (daily timestamps)
        day0 = int(S.t[S.test0]); day_end = int(S.t[-1]) + 86400
        J = (day_end - day0) // 86400
        if args.max_steps: J = min(J, args.max_steps)
        grid = BatchGrid.from_boundaries([day0 + 86400 * j for j in range(J + 1)])
        replaces = dict(zip(S.extra["amend_new"].tolist(), S.extra["amend_old"].tolist())) if args.amend and "amend_new" in S.extra else None
        setup = make_setup(S, arrival, grid=grid, replaces=replaces, dedupe_queries=True)
    else:
        setup = make_setup(S, arrival, bq=args.bq, max_steps=args.max_steps)
    if args.max_queries:                            # a fixed random sample of each step's queries (large streams)
        g = torch.Generator().manual_seed(77)
        setup.queries = [q[torch.randperm(q.numel(), generator=g)[:args.max_queries].sort().values] if q.numel() > args.max_queries else q
                         for q in setup.queries]
        setup.info["queries"] = int(sum(q.numel() for q in setup.queries))
    print(f"{out}: {setup.info}", flush=True)
    negs_r, negs_h = sample_negatives(S, setup, seed=1234 + args.seed)
    base_mem, base_store = warm_start(model, S, setup.table)
    policies = [Policy(m, model, setup, base_mem, base_store) for m in args.methods.split(",")]
    for p in policies:
        p.mem.use_index = not args.no_index
    summary, arrays = evaluate_stream(model, setup, policies, negs_r, negs_h, ref="exact")
    # paired bootstrap of MRR differences against exact repair (per query)
    if "exact" in summary["methods"]:
        for name in summary["methods"]:
            for key, arr in (("rand", "rr_r"), ("hist", "rr_h")):
                m, lo, hi = paired_diff_ci(arrays[f"{arr}__{name}"], arrays[f"{arr}__exact"], n_boot=1000, seed=1)
                summary["methods"][name][f"dmrr_vs_exact_{key}"] = [m, lo, hi]
    # strata: queries whose user has an event in transit, and hub vs. tail items
    flags = in_transit_flags(setup)
    q_ev = torch.from_numpy(arrays["query_event"])
    item_deg = torch.bincount(S.dst[:S.test0], minlength=S.num_nodes)
    deg = item_deg[S.dst[q_ev]].numpy()
    hub = deg >= np.quantile(item_deg[S.dst_min:S.dst_max + 1].numpy(), 0.9)
    strata = dict(in_transit_frac=float(flags.mean()), hub_frac=float(hub.mean()), methods={})
    for name in summary["methods"]:
        rr = arrays[f"rr_r__{name}"]
        strata["methods"][name] = dict(in_transit=float(rr[flags].mean()) if flags.any() else None,
                                       not_in_transit=float(rr[~flags].mean()), hub_items=float(rr[hub].mean()),
                                       tail_items=float(rr[~hub].mean()))
    summary["strata"] = strata
    summary["config"] = vars(args)
    summary["wall_sec"] = time.time() - t0
    arrays["in_transit"] = flags; arrays["hub"] = hub
    with open(out_dir / f"{out}.json", "w") as f:
        json.dump(summary, f, indent=1)
    np.savez_compressed(out_dir / f"{out}.npz", **arrays)
    for name, s in summary["methods"].items():
        print(f"{name:>14}: MRR {s['mrr']:.4f} (hist {s['mrr_hist']:.4f})  cost x{s.get('cost_x_normal', float('nan')):6.2f}"
              f"  dMRR vs exact {s.get('dmrr_vs_exact_rand', [0])[0]:+.4f}  top1 {s.get('top1_agree', float('nan')):.3f}"
              f"  {s['sec']:.1f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
