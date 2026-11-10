"""How the cost of exact repair grows with the age of the edit (run on an otherwise idle machine).

  python scripts/bench_age.py --data wikipedia reddit mooc lastfm --threads 4 --trials 200
For each dataset (model seed 0): the longest test window available (at most --window steps of 200 events) with 5% of its
events held back; then single edits (alternating late insertions and deletions) at uniformly random steps of the window.
For every edit we record its age (steps between the edit and the end of the window), the time of exact repair with the
step index, of the same repair walking every step and of rollback + replay, the steps visited and the updates
recomputed. We also time one pass of the memory over the whole history (recomputing the memory from scratch with fixed
weights, the baseline for deletions of old events). Writes results/bench_age/bench_age_<data>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph import paths                                                   # noqa: E402
from lategraph.data import BatchGrid, load                                    # noqa: E402
from lategraph.memory import EventTable, RetroactiveTGNMemory                 # noqa: E402
from lategraph.model import load_checkpoint                                   # noqa: E402


def bench(name, threads, trials, window):
    torch.set_num_threads(threads)
    S = load(name)
    model = load_checkpoint(paths.CKPT / f"tgn_{name}_s0.pt")
    grid = BatchGrid(S.t, S.test0, 200, window)
    step = torch.full((S.num_events,), -1, dtype=torch.long)
    test = torch.arange(S.test0, S.num_events); step[test] = grid.step_of_time(S.t[test])
    table = EventTable(S.src, S.dst, S.t, S.msg, step)
    rng = np.random.default_rng(0)
    win = test[(step[test] >= 0) & (step[test] < grid.J)]
    missing = torch.zeros(S.num_events, dtype=torch.bool); missing[win[torch.from_numpy(rng.random(win.numel()) < 0.05)]] = True
    base = RetroactiveTGNMemory.from_trained(model.memory, table)
    t0 = time.perf_counter(); base.warmup(0, S.test0); history_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    for k in range(grid.J):
        base.advance(win[(step[win] == k) & ~missing[win]])
    window_sec = time.perf_counter() - t0
    present, absent = win[~missing[win]].numpy(), win[missing[win]].numpy()
    w = RetroactiveTGNMemory.from_trained(model.memory, table)
    recs = []
    for i in range(trials):
        pool = absent if i % 2 else present
        e = torch.tensor([int(rng.choice(pool))])
        ins, dele = (e, e[:0]) if i % 2 else (e[:0], e)
        r = dict(kind="insert" if i % 2 else "delete", age_steps=int(grid.J - step[e]))
        for m in ("indexed", "walk", "replay"):
            w.copy_state_from(base); w.use_index = m == "indexed"; w.edit(insert=ins, delete=dele)
            tt = time.perf_counter(); st = w.repair("replay" if m == "replay" else "exact"); r[f"ms_{m}"] = 1000 * (time.perf_counter() - tt)
            if m == "indexed":
                r["visited_steps"] = int(st.extra["visited_steps"]); r["recomputed"] = int(st.recomputed); r["window_steps"] = int(st.steps)
            if m == "replay":
                r["replayed"] = int(st.recomputed)
        recs.append(r)
    out = dict(data=name, threads=threads, window_steps=grid.J, history_events=int(S.test0), history_pass_sec=history_sec,
               window_pass_sec=window_sec, trials=recs)
    ages = np.array([r["age_steps"] for r in recs]); sp = np.array([r["ms_replay"] / max(r["ms_indexed"], 1e-6) for r in recs])
    for lo, hi in ((0, 10), (10, 50), (50, 200), (200, 10 ** 6)):
        sel = (ages > lo) & (ages <= hi)
        if sel.any():
            print(f"{name} age {lo}-{hi} steps: {sel.sum()} edits, median speed-up {np.median(sp[sel]):.1f}x, "
                  f"median cone {np.median([recs[j]['recomputed'] / max(recs[j]['replayed'], 1) for j in np.flatnonzero(sel)]):.3f} of replay", flush=True)
    print(f"{name}: one pass over the history ({S.test0:,} events) takes {history_sec:.2f} s", flush=True)
    od = paths.RESULTS / "bench_age"; od.mkdir(parents=True, exist_ok=True)
    with open(od / f"bench_age_{name}.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["wikipedia", "reddit", "mooc", "lastfm"])
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--window", type=int, default=500)
    a = ap.parse_args()
    for d in a.data:
        bench(d, a.threads, a.trials, a.window)
