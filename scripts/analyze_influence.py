"""Attention against exact influence, channel by channel (from the per-event records of run_influence.py).

  python scripts/analyze_influence.py      -> results/influence/attention_vs_channels.json
Attention only weighs the neighbour list, so besides the total effect of deleting an event we compare it with the effect
through the neighbour list alone (the memory left as it is). Per query: Spearman correlation of the attention weights
with |influence| over the events in the list, and whether the most-attended event is the most influential one; pooled
over the queries of all seeds, with 95% bootstrap intervals and the chance level of a random pick from the list.
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

R = Path(__file__).resolve().parents[1] / "results" / "influence"
rng = np.random.default_rng(0)


def boot(x, n=2000):
    x = np.asarray(x, float)
    if x.size == 0:
        return [float("nan")] * 2
    m = [np.mean(rng.choice(x, x.size)) for _ in range(n)]
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


out = {}
by_data = defaultdict(list)
for p in sorted(glob.glob(str(R / "influence_tgn_*_s[0-9]_records.json"))):
    data = re.search(r"influence_tgn_(.+)_s\d+_records", p).group(1)
    by_data[data].append(json.load(open(p)))
for data, runs in by_data.items():
    rho = {"total": [], "list": []}; top1 = {"total": [], "list": []}; chance = []; per_seed = []
    for recs in runs:
        n0 = len(top1["total"])
        q = defaultdict(list)
        for r in recs:
            if r["in_list"] and r["att"] is not None:
                q[r["q"]].append(r)
        for rs in q.values():
            if len(rs) < 3:
                continue
            att = np.array([r["att"] for r in rs])
            for ch, key in (("total", "d_both"), ("list", "d_list")):
                infl = np.abs([r[key] for r in rs])
                if np.ptp(infl) > 0 and np.ptp(att) > 0:
                    rho[ch].append(spearmanr(att, infl).statistic)
                top1[ch].append(float(np.argmax(att) == np.argmax(infl)))
            chance.append(1 / len(rs))
        per_seed.append(float(np.mean(top1["total"][n0:])) if len(top1["total"]) > n0 else float("nan"))
    out[data] = dict(seeds=len(runs), queries=len(chance),
                     **{f"spearman_{ch}_median": float(np.median(rho[ch])) for ch in rho},
                     **{f"top1_{ch}": float(np.mean(top1[ch])) for ch in top1},
                     **{f"top1_{ch}_ci": boot(top1[ch]) for ch in top1},
                     top1_chance=float(np.mean(chance)), top1_total_per_seed=per_seed)
    print(data, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[data].items()})
json.dump(out, open(R / "attention_vs_channels.json", "w"), indent=1)
