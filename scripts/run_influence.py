"""E7: exact counterfactual explanations. Which past events does a TGN prediction depend on, and does attention say so?

For sampled test queries (u, v) at step j0 we delete each of u's last C events (one at a time) and measure the exact
change of the model's link score s(u, v) (the deleted event never happened: exact repair of the memory, and the event
removed from the neighbour list). Two channels are measured separately:
  memory   repair the memory only (the neighbour list keeps the event)
  list     remove the event from the neighbour list only (the memory keeps it)
  both     the full deletion (= the exact leave-one-event-out influence)
and compared with the attention weights of PyG's TransformerConv for u's neighbour edges (the usual "explanation"),
which only ever see u's last k = 10 events.

  python scripts/run_influence.py --data wikipedia --model tgn_wikipedia_s0 --queries 200
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph import paths                                                 # noqa: E402
from lategraph.data import BatchGrid, load                                  # noqa: E402
from lategraph.memory import EventTable, RetroactiveTGNMemory               # noqa: E402
from lategraph.model import load_checkpoint                                 # noqa: E402
from lategraph.neighbors import TimeOrderedNeighborStore                    # noqa: E402
from lategraph.stream import score_pairs                                    # noqa: E402


@torch.no_grad()
def score_with_attention(model, mem, store, table, u: int, v: int, now: int):
    """s(u, v) and the attention (mean over heads) of u's incoming neighbour edges, as {event id: weight}."""
    n_id = torch.tensor(sorted({u, v}))
    all_id, edge_index, e_id = store(n_id)
    assoc = torch.empty(mem.num_nodes, dtype=torch.long); assoc[all_id] = torch.arange(all_id.numel())
    z, (ei, alpha) = model.gnn(mem.memory[all_id], mem.last_update[all_id], edge_index, table.t[e_id], table.msg[e_id],
                               return_attention=True)
    s = float(model.score(z[assoc[torch.tensor([u])]], z[assoc[torch.tensor([v])]]))
    # TransformerConv adds no self loops, so the returned edges are ours, in order
    into_u = (edge_index[1] == assoc[u]).nonzero().view(-1)
    att = {int(e_id[i]): float(alpha[i].mean()) for i in into_u.tolist()}
    return s, att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--queries", type=int, default=200); ap.add_argument("--C", type=int, default=30)
    ap.add_argument("--j0", type=float, default=0.5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    S = load(args.data)
    model = load_checkpoint(paths.CKPT / f"{args.model}.pt")
    assert model.cfg.emb == "attn", "the attention comparison needs the standard TGN embedding"
    grid = BatchGrid(S.t, S.test0, 200)
    step = torch.full((S.num_events,), -1, dtype=torch.long)
    test = torch.arange(S.test0, S.num_events); step[test] = grid.step_of_time(S.t[test])
    table = EventTable(S.src, S.dst, S.t, S.msg, step)
    j0 = int(args.j0 * grid.J)
    mem = RetroactiveTGNMemory.from_trained(model.memory, table); mem.warmup(0, S.test0)
    for j in range(j0):                                          # the journal covers the test period up to j0
        mem.advance(test[step[test] == j])
    store = TimeOrderedNeighborStore(S.src, S.dst, S.t, S.num_nodes, model.cfg.n_neighbors)
    seen = torch.arange(S.num_events) < S.test0
    seen[test[step[test] < j0]] = True
    store.rebuild_from(seen)
    now = int(grid.T[j0])
    # queries: events of step j0 whose user has >= C events in the journaled window (so every candidate is repairable)
    rng = np.random.default_rng(args.seed)
    win_ev = test[(step[test] >= 0) & (step[test] < j0)]
    user_ev = {}
    for e in win_ev.tolist():
        user_ev.setdefault(int(S.src[e]), []).append(e)
    q_all = [e for e in test[step[test] == j0].tolist() if len(user_ev.get(int(S.src[e]), [])) >= args.C]
    q = [q_all[i] for i in rng.permutation(len(q_all))[: args.queries]]
    print(f"[{time.time() - t0:.0f}s] {args.data}: {len(q_all)} eligible queries at step {j0}/{grid.J}; using {len(q)}", flush=True)
    work = RetroactiveTGNMemory.from_trained(model.memory, table)
    recs = []
    for qi, e_q in enumerate(q):
        u, v = int(S.src[e_q]), int(S.dst[e_q])
        s0, att = score_with_attention(model, mem, store, table, u, v, now)
        cand = user_ev[u][-args.C:][::-1]                        # u's last C events, newest first
        for r, e in enumerate(cand):
            et = torch.tensor([e])
            work.copy_state_from(mem); work.edit(delete=et); st = work.repair("exact")
            st_store = store.clone(); st_store.remove(et)
            s_mem = float(score_pairs(model, work, store, table, torch.tensor([u]), torch.tensor([v]), now))
            s_list = float(score_pairs(model, mem, st_store, table, torch.tensor([u]), torch.tensor([v]), now))
            s_both = float(score_pairs(model, work, st_store, table, torch.tensor([u]), torch.tensor([v]), now))
            recs.append(dict(q=qi, u=u, v=v, e=e, rank=r, item=int(S.dst[e]), same_item=int(S.dst[e]) == v,
                             age_h=(now - int(S.t[e])) / 3600, att=att.get(e), in_list=e in att,
                             d_mem=s0 - s_mem, d_list=s0 - s_list, d_both=s0 - s_both, cone=st.recomputed))
        if qi % 50 == 49:
            print(f"[{time.time() - t0:.0f}s] {qi + 1}/{len(q)} queries", flush=True)

    # ---- analysis
    R = {k: np.array([r[k] for r in recs], dtype=float if k not in ("in_list", "same_item") else bool) for k in
         ("q", "rank", "d_mem", "d_list", "d_both", "in_list", "same_item", "age_h", "cone")}
    att_v = np.array([r["att"] if r["att"] is not None else np.nan for r in recs])
    tot = np.abs(R["d_both"])
    out = dict(queries=len(q), candidates=args.C, j0=j0, J=grid.J, records=len(recs))
    out["share_outside_list"] = float(tot[~R["in_list"]].sum() / max(tot.sum(), 1e-12))     # influence attention never sees
    out["share_memory_channel"] = float(np.abs(R["d_mem"]).sum() / max(np.abs(R["d_mem"]).sum() + np.abs(R["d_list"]).sum(), 1e-12))
    out["additivity_err"] = float(np.median(np.abs(R["d_both"] - R["d_mem"] - R["d_list"]) / (tot + 1e-9)))
    rho, top1, topk = [], [], []
    for qi in range(len(q)):
        m = (R["q"] == qi) & R["in_list"]
        if m.sum() >= 3:
            a, d = att_v[m], tot[m]
            rho.append(spearmanr(a, d).statistic)
            top1.append(int(np.argmax(a) == np.argmax(d)))
        mq = R["q"] == qi
        top_ev = np.argmax(tot[mq]); topk.append(bool(R["in_list"][mq][top_ev]))
    out["spearman_att_vs_influence_median"] = float(np.nanmedian(rho)) if rho else None
    out["top1_attention_is_top1_influence"] = float(np.mean(top1)) if top1 else None
    out["top1_influence_inside_list"] = float(np.mean(topk))
    out["mean_abs_influence_by_rank"] = [float(tot[R["rank"] == r].mean()) for r in range(args.C)]
    out["mean_abs_mem_by_rank"] = [float(np.abs(R["d_mem"])[R["rank"] == r].mean()) for r in range(args.C)]
    out["mean_abs_list_by_rank"] = [float(np.abs(R["d_list"])[R["rank"] == r].mean()) for r in range(args.C)]
    out["same_item_mean_abs"] = float(tot[R["same_item"]].mean()) if R["same_item"].any() else None
    out["other_item_mean_abs"] = float(tot[~R["same_item"]].mean())
    out["median_cone"] = float(np.median(R["cone"]))
    out["wall_sec"] = time.time() - t0
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if not isinstance(v, list)}, indent=1))
    od = paths.RESULTS / "influence"; od.mkdir(parents=True, exist_ok=True)
    name = args.out or f"influence_{args.model}"
    with open(od / f"{name}.json", "w") as f:
        json.dump(dict(config=vars(args), summary=out), f, indent=1)
    with open(od / f"{name}_records.json", "w") as f:
        json.dump(recs[:4000], f)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
