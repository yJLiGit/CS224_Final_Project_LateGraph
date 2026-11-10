"""Timing benchmark of repair (run on an otherwise idle machine; the experiment queues measure costs, this measures time).

  python scripts/bench_repair.py --data wikipedia reddit mooc lastfm --threads 4
For each dataset (model seed 0): a 100-step window with 5% of its events missing, then
  (a) single edits (alternating late insertions and deletions) and 5-edit batches: exact repair with the step index, the
      same repair walking every step, and rollback + full replay; the indexed result is checked against the walk bit for bit;
  (b) streaming with random delays over the first 60 steps: time per step of exact repair (index / walk), full replay,
      eps-repair and applying late events on arrival.
Writes results/bench/bench_<data>.json.
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
from lategraph.delays import arrival_times, make_delays                       # noqa: E402
from lategraph.memory import EventTable, RetroactiveTGNMemory                 # noqa: E402
from lategraph.model import load_checkpoint                                   # noqa: E402
from lategraph.stream import Policy, evaluate_stream, make_setup, sample_negatives, warm_start   # noqa: E402


def bench(name, threads, trials, stream=True):
    torch.set_num_threads(threads)
    S = load(name)
    model = load_checkpoint(paths.CKPT / f"tgn_{name}_s0.pt")
    grid = BatchGrid(S.t, S.test0, 200, 100)
    step = torch.full((S.num_events,), -1, dtype=torch.long)
    test = torch.arange(S.test0, S.num_events); step[test] = grid.step_of_time(S.t[test])
    table = EventTable(S.src, S.dst, S.t, S.msg, step)
    rng = np.random.default_rng(0)
    win = test[step[test] < grid.J]
    missing = torch.zeros(S.num_events, dtype=torch.bool); missing[win[torch.from_numpy(rng.random(win.numel()) < 0.05)]] = True
    base = RetroactiveTGNMemory.from_trained(model.memory, table); base.warmup(0, S.test0)
    t0 = time.perf_counter()
    for k in range(grid.J):
        base.advance(win[(step[win] == k) & ~missing[win]])
    normal_ms = 1000 * (time.perf_counter() - t0) / grid.J
    present, absent = win[~missing[win]].numpy(), win[missing[win]].numpy()
    w = RetroactiveTGNMemory.from_trained(model.memory, table)
    out = dict(data=name, threads=threads, window_steps=grid.J, normal_ms_per_step=normal_ms)
    for kind, n_edits in (("single", 1), ("batch5", 5)):
        T = {m: [] for m in ("indexed", "walk", "replay")}; vis, back, equal = [], [], 0; per = []
        for i in range(trials):
            ins = torch.from_numpy(rng.choice(absent, n_edits if i % 2 else 0, replace=False)).long()
            dele = torch.from_numpy(rng.choice(present, 0 if i % 2 else n_edits, replace=False)).long()
            res = {}
            for m in T:
                w.copy_state_from(base); w.use_index = m == "indexed"; w.edit(insert=ins, delete=dele)
                tt = time.perf_counter(); st = w.repair("replay" if m == "replay" else "exact"); T[m].append(time.perf_counter() - tt)
                res[m] = (w.memory.clone(), w.last_update.clone())
                if m == "indexed":
                    vis.append(st.extra["visited_steps"]); back.append(st.steps)
                    per.append(dict(recomputed=int(st.recomputed), changed=int(st.changed_nodes), steps_back=int(st.steps)))
            equal += torch.equal(res["indexed"][0], res["walk"][0]) and torch.equal(res["indexed"][1], res["walk"][1])
        for r, ti, tw, tr in zip(per, T["indexed"], T["walk"], T["replay"]):
            r.update(ms_indexed=1000 * ti, ms_walk=1000 * tw, ms_replay=1000 * tr)
        med = {m: 1000 * float(np.median(v)) for m, v in T.items()}
        ratio = lambda a, b: float(np.median(np.array(T[a]) / np.maximum(np.array(T[b]), 1e-9)))
        out[kind] = dict(trials=trials, median_ms=med, speedup_vs_replay=ratio("replay", "indexed"),
                         walk_speedup_vs_replay=ratio("replay", "walk"), speedup_vs_walk=ratio("walk", "indexed"),
                         bitwise_equal=equal, visited_steps_median=float(np.median(vis)), window_steps_median=float(np.median(back)),
                         trials_detail=per)
        eff = [r for r in per if r["changed"] > 0]            # edits that change at least one memory
        if eff:
            out[kind]["effective_n"] = len(eff)
            out[kind]["speedup_vs_replay_effective"] = float(np.median([r["ms_replay"] / max(r["ms_indexed"], 1e-6) for r in eff]))
            out[kind]["walk_speedup_vs_replay_effective"] = float(np.median([r["ms_replay"] / max(r["ms_walk"], 1e-6) for r in eff]))
        print(f"{name} {kind}: median ms {med} | index vs replay {out[kind]['speedup_vs_replay']:.1f}x (walk {out[kind]['walk_speedup_vs_replay']:.1f}x) "
              f"| equal {equal}/{trials} | visited {np.median(vis):.0f} of {np.median(back):.0f} steps", flush=True)
    # streaming, first 60 steps (random delays with the model's defaults; skipped with --no_stream)
    od = paths.RESULTS / "bench"; od.mkdir(parents=True, exist_ok=True)
    if not stream:
        with open(od / f"bench_{name}.json", "w") as f:
            json.dump(out, f, indent=1)
        return
    arr = arrival_times(S, make_delays(S, "rand", seed=0))
    setup = make_setup(S, arr, bq=200, max_steps=60)
    negs_r, negs_h = sample_negatives(S, setup, seed=1234)
    m0, s0 = warm_start(model, S, setup.table)
    pols = [Policy(m, model, setup, m0, s0) for m in ("exact", "exact", "replay", "eps:0.05", "arrival")]
    pols[1].mem.use_index = False; pols[1].name = "exact_walk"
    summ, _ = evaluate_stream(model, setup, pols, negs_r, negs_h, ref="exact", log=None)
    out["stream_ms_per_step"] = {p.name: 1000 * p.sec / setup.grid.J for p in pols}
    print(f"{name} streaming ms/step: " + ", ".join(f"{k} {v:.1f}" for k, v in out["stream_ms_per_step"].items()), flush=True)
    with open(od / f"bench_{name}.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["wikipedia", "reddit", "mooc", "lastfm"])
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--no_stream", nargs="*", default=["tgbl-review"], help="datasets without the streaming part")
    a = ap.parse_args()
    for d in a.data:
        bench(d, a.threads, a.trials, stream=d not in a.no_stream)
