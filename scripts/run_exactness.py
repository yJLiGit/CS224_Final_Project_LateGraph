"""E2: is exact cone repair really exact? (Proposition 1)

For many random edits we compare the repaired memory with an independent reference: a fresh, sequential recomputation of
the corrected stream from a full checkpoint taken before the edited step (no journal, no cone logic). Edit types:
  insert  an event that was missing from the stream arrives late and is inserted into its own step
  delete  an event is removed
  amend   an event is replaced by a corrected version (new features and, sometimes, a new time / step)
  mixed   2-10 edits of all three kinds at once
With --pyg, the reference for single edits is PyG's own TGNMemory (only valid when the stream has no timestamp ties).

  python scripts/run_exactness.py --data wikipedia --model tgn_wikipedia_s0 --trials 4000 --pyg 1000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph import paths                                               # noqa: E402
from lategraph.data import BatchGrid, load                                # noqa: E402
from lategraph.memory import EventTable, RetroactiveTGNMemory             # noqa: E402
from lategraph.model import load_checkpoint                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--trials", type=int, default=3000); ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--bq", type=int, default=200); ap.add_argument("--missing", type=float, default=0.05)
    ap.add_argument("--pyg", type=int, default=0, help="first N single-edit trials also checked against PyG TGNMemory")
    ap.add_argument("--double", action="store_true", help="run everything in float64")
    ap.add_argument("--no_pad", action="store_true", help="disable the single-row padding (reproduces the float32 issue)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    S = load(args.data)
    model = load_checkpoint(paths.CKPT / f"{args.model}.pt")
    dtype = torch.float64 if args.double else torch.float32
    if args.double:
        model = model.double()
    grid = BatchGrid(S.t, S.test0, args.bq, args.steps)
    W = grid.J
    E = S.num_events
    lo, hi = S.test0, S.test0 + int((grid.step_of_time(S.t[S.test0:]) < W).sum())
    win = torch.arange(lo, hi)
    # amended versions: one per window event in a pool, appended to the event table
    pool = torch.from_numpy(rng.choice(win.numpy(), size=min(4000, win.numel()), replace=False))
    msg_std = S.msg[:S.test0].std(0)
    new_msg = S.msg[pool] + 0.5 * msg_std * torch.randn(pool.numel(), S.msg_dim)
    shift = torch.zeros(pool.numel(), dtype=torch.long)
    move = torch.from_numpy(rng.random(pool.numel()) < 0.3)      # 30% of amendments also correct the time
    span = int(grid.T[W] - grid.T[0])
    shift[move] = torch.from_numpy(rng.integers(-span // 10, span // 10, int(move.sum()))).long()
    new_t = (S.t[pool] + shift).clamp(int(grid.T[0]), int(grid.T[W]) - 1)
    src = torch.cat([S.src, S.src[pool]]); dst = torch.cat([S.dst, S.dst[pool]])
    tt = torch.cat([S.t, new_t]); msg = torch.cat([S.msg, new_msg]).to(dtype)
    step = torch.full((src.numel(),), -1, dtype=torch.long)
    test_ids = torch.cat([win, torch.arange(E, E + pool.numel())])
    step[test_ids] = grid.step_of_time(tt[test_ids])
    table = EventTable(src, dst, tt, msg, step)
    table.msg = msg
    version_of = {int(p): E + i for i, p in enumerate(pool.tolist())}

    # base trajectory: the window processed in order, with a random 5% of events still missing (they will arrive late)
    missing = torch.zeros(src.numel(), dtype=torch.bool)
    missing[win[torch.from_numpy(rng.random(win.numel()) < args.missing)]] = True
    base = RetroactiveTGNMemory.from_trained(model.memory, table)
    base.pad_single_rows = not args.no_pad
    base.warmup(0, S.test0)
    ckpt_mem, ckpt_lu = [], []                               # full state before every step (for the reference)
    content = []
    for k in range(W):
        ev = win[(step[win] == k) & ~missing[win]]
        ckpt_mem.append(base.memory.clone()); ckpt_lu.append(base.last_update.clone())
        content.append(ev)
        base.advance(ev)
    final_ref_state = (base.memory.clone(), base.last_update.clone())
    print(f"{args.data}: window {W} steps, {win.numel()} events, {int(missing.sum())} missing; pool {pool.numel()} amendments",
          flush=True)

    def reference(contents, k0, use_pyg=False):
        """Sequential recomputation of steps k0..W-1 from the checkpoint before step k0."""
        if use_pyg:
            m = model.memory; m.eval()
            m.memory.copy_(ckpt_mem[k0]); m.last_update.copy_(ckpt_lu[k0]); m._reset_message_store()
            for k in range(k0, W):
                ev = contents[k]
                if ev.numel():
                    m.update_state(table.src[ev], table.dst[ev], table.t[ev], table.msg[ev])
            return m.memory.clone(), m.last_update.clone()
        ref = RetroactiveTGNMemory.from_trained(model.memory, table)
        ref.memory.copy_(ckpt_mem[k0]); ref.last_update.copy_(ckpt_lu[k0])
        for k in range(k0, W):
            ref.apply(contents[k])
        return ref.memory, ref.last_update

    def has_ties(contents, k0):
        """Does some node have two events at its latest timestamp within a step? (PyG then picks by array order.)"""
        for k in range(k0, W):
            ev = contents[k]
            if ev.numel() < 2:
                continue
            node = torch.cat([table.src[ev], table.dst[ev]]); tt_ = torch.cat([table.t[ev], table.t[ev]])
            mx = torch.full((S.num_nodes,), -1, dtype=torch.long).scatter_reduce(0, node, tt_, reduce="amax")
            at_max = tt_ == mx[node]
            if bool((torch.bincount(node[at_max], minlength=S.num_nodes) > 1).any()):
                return True
        return False

    work = RetroactiveTGNMemory.from_trained(model.memory, table)
    present_ids = win[~missing[win]].numpy(); missing_ids = win[missing[win]].numpy()
    pool_ids = pool.numpy()
    records = []
    t0 = time.time()
    deg = torch.bincount(torch.cat([S.src[:S.test0], S.dst[:S.test0]]), minlength=S.num_nodes)
    for i in range(args.trials):
        kind = rng.choice(["insert", "delete", "amend", "mixed"], p=[0.3, 0.3, 0.3, 0.1])
        ins, dele = [], []
        n_edits = int(rng.integers(2, 11)) if kind == "mixed" else 1
        for _ in range(n_edits):
            k = kind if kind != "mixed" else rng.choice(["insert", "delete", "amend"])
            if k == "insert":
                ins.append(int(rng.choice(missing_ids)))
            elif k == "delete":
                dele.append(int(rng.choice(present_ids)))
            else:
                e = int(rng.choice(pool_ids))
                if missing[e]:
                    continue
                dele.append(e); ins.append(version_of[e])
        ins = sorted(set(ins) - set(dele)); dele = sorted(set(dele))
        if not ins and not dele:
            continue
        ins_t = torch.tensor(ins, dtype=torch.long); del_t = torch.tensor(dele, dtype=torch.long)
        # corrected contents
        contents = list(content)
        for k in torch.unique(torch.cat([step[ins_t], step[del_t]])).tolist():
            c = contents[k]
            c = c[~torch.isin(c, del_t)]
            c = torch.unique(torch.cat([c, ins_t[step[ins_t] == k]]))
            contents[k] = c
        k0 = int(torch.cat([step[ins_t], step[del_t]]).min())
        rec = dict(kind=kind, n_edits=len(ins) + len(dele), k0=k0, steps_back=W - k0)
        for mode in ("exact", "replay"):
            work.copy_state_from(base)
            work.edit(insert=ins_t, delete=del_t)
            tt0 = time.perf_counter()
            st = work.repair(mode)
            rec[f"{mode}_sec"] = time.perf_counter() - tt0
            if mode == "exact":
                rep_mem, rep_lu = work.memory.clone(), work.last_update.clone()
                rec.update(recomputed=st.recomputed, window_updates=st.window_updates, changed_nodes=st.changed_nodes)
            else:
                rec["replay_vs_exact"] = float((work.memory - rep_mem).abs().max())
        ref_mem, ref_lu = reference(contents, k0)
        rec["max_abs_diff"] = float((rep_mem - ref_mem).abs().max())
        rec["lu_equal"] = bool((rep_lu == ref_lu).all())
        rec["changed_vs_base"] = float((ref_mem - final_ref_state[0]).abs().max())
        if args.pyg and i < args.pyg and kind != "mixed":
            pm, pl = reference(contents, k0, use_pyg=True)
            rec["max_abs_diff_pyg"] = float((rep_mem - pm).abs().max())
            rec["ties"] = has_ties(contents, k0)
        nodes = torch.cat([table.src[torch.cat([ins_t, del_t])], table.dst[torch.cat([ins_t, del_t])]])
        rec["endpoint_max_degree"] = int(deg[nodes].max())
        records.append(rec)
        if (i + 1) % 250 == 0:
            md = max(r["max_abs_diff"] for r in records)
            print(f"[{time.time() - t0:5.0f}s] {i + 1} trials, worst |repair - reference| = {md:.2e}", flush=True)
    diffs = np.array([r["max_abs_diff"] for r in records])
    summ = dict(config=vars(args), trials=len(records), window_steps=W, window_events=int(win.numel()),
                max_abs_diff=float(diffs.max()), frac_below_1e5=float((diffs < 1e-5).mean()),
                frac_exact_zero=float((diffs == 0).mean()), lu_all_equal=bool(all(r["lu_equal"] for r in records)),
                replay_vs_exact_max=float(max(r["replay_vs_exact"] for r in records)),
                by_kind={k: dict(n=int(sum(r["kind"] == k for r in records)),
                                 max_abs_diff=float(max([r["max_abs_diff"] for r in records if r["kind"] == k] or [0])),
                                 median_cone_frac=float(np.median([r["recomputed"] / max(r["window_updates"], 1) for r in records if r["kind"] == k] or [0])))
                         for k in ("insert", "delete", "amend", "mixed")})
    pyg = [r["max_abs_diff_pyg"] for r in records if "max_abs_diff_pyg" in r]
    if pyg:
        nt = [r["max_abs_diff_pyg"] for r in records if "max_abs_diff_pyg" in r and not r["ties"]]
        summ["pyg_reference"] = dict(n=len(pyg), max_abs_diff=float(max(pyg)), frac_below_1e5=float(np.mean(np.array(pyg) < 1e-5)),
                                     n_without_ties=len(nt), max_abs_diff_without_ties=float(max(nt)) if nt else None,
                                     n_with_ties=len(pyg) - len(nt))
    out = args.out or f"exactness_{args.model}" + ("_f64" if args.double else "") + ("_nopad" if args.no_pad else "")
    od = paths.RESULTS / "exactness"; od.mkdir(parents=True, exist_ok=True)
    with open(od / f"{out}.json", "w") as f:
        json.dump(dict(summary=summ, trials=records), f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
