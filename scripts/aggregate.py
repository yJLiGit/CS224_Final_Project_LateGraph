"""Collect every result JSON into results/summary.json (the numbers the milestone report uses).

  python scripts/aggregate.py
Seeds are aggregated as mean, sample std and the per-seed values.
"""
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph import paths                     # noqa: E402

R = paths.RESULTS


def ms(v):
    v = np.asarray([x for x in v if x is not None], dtype=np.float64)
    if v.size == 0:
        return None
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if v.size > 1 else 0.0, n=int(v.size), values=[float(x) for x in v])


def load(p):
    with open(p) as f:
        return json.load(f)


def parse_model(tag):
    """'tgn_wikipedia_s0_rho0.75_lam10' -> ('wikipedia', 0, 'rho0.75_lam10'); 'dyrep_mooc_s1' -> ('mooc', 1, 'dyrep')."""
    m = re.match(r"(tgn|dyrep)_(tgbl-\w+|\w+?)_s(\d+)(.*)", tag)
    kind, data, seed, rest = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    variant = rest.strip("_") or ("attn" if kind == "tgn" else kind)
    return data, seed, variant


def training():
    out = defaultdict(list)
    for p in sorted(glob.glob(str(R / "training" / "*.json"))):
        d = load(p)
        data, seed, var = parse_model(d["tag"])
        out[f"{data}|{var}"].append(d)
    return {k: dict(val_ap=ms([x["best_val_ap"] for x in v]), best_epoch=[x["best_epoch"] for x in v],
                    minutes=ms([x.get("wall_sec", 0) / 60 for x in v]), seeds=[x["seed"] for x in v]) for k, v in out.items()}


def streaming():
    groups = defaultdict(list)
    for p in sorted(glob.glob(str(R / "stream" / "*.json"))):
        if "debug" in p:
            continue
        d = load(p)
        c = d["config"]
        data, seed, var = parse_model(c["model"])
        kind = "horizon" if Path(p).stem.startswith("horizon") else "main"
        if c["data"].startswith("sec"):
            kind += "_amend" if c.get("amend", 1) else "_noamend"
        key = f"{kind}|{data}|{var}|{c['delay']}|bq{c['bq']}"
        groups[key].append(d)
    out = {}
    for key, runs in groups.items():
        methods = runs[0]["methods"].keys()
        agg = {}
        for m in methods:
            rs = [r["methods"][m] for r in runs if m in r["methods"]]
            agg[m] = {k: ms([r.get(k) for r in rs]) for k in ("mrr", "mrr_hist", "cost_x_normal", "top1_agree", "mem_div", "dlogit", "sec", "fallback", "journal_MB")}
            agg[m]["dmrr_vs_exact"] = ms([r["dmrr_vs_exact_rand"][0] for r in rs if "dmrr_vs_exact_rand" in r])
            agg[m]["dmrr_vs_exact_hist"] = ms([r["dmrr_vs_exact_hist"][0] for r in rs if "dmrr_vs_exact_hist" in r])
            agg[m]["dmrr_rel_pct"] = ms([100 * r["dmrr_vs_exact_rand"][0] / max(runs[i]["methods"]["exact"]["mrr"], 1e-9)
                                         for i, r in enumerate(rs) if "dmrr_vs_exact_rand" in r and "exact" in runs[i]["methods"]])
            st = [r["strata"]["methods"][m] for r in runs if "strata" in r and m in r["strata"]["methods"]]
            if st:
                agg[m]["strata"] = {k: ms([s[k] for s in st]) for k in ("in_transit", "not_in_transit", "hub_items", "tail_items")}
        info = runs[0]["info"]
        out[key] = dict(methods=agg, seeds=[r["config"]["seed"] for r in runs], n=len(runs),
                        late_frac=ms([r["info"]["late_frac"] for r in runs]),
                        in_transit_frac=ms([r["strata"]["in_transit_frac"] for r in runs if "strata" in r]),
                        J=info["J"], test_events=info["test_events"], queries=info.get("queries"),
                        late_steps_median=ms([r["info"]["late_steps_median"] for r in runs]),
                        late_steps_p90=ms([r["info"]["late_steps_p90"] for r in runs]),
                        amendments=info.get("amendments"),
                        wall_min=ms([r.get("wall_sec", 0) / 60 for r in runs]))
    return out


def exactness():
    out = {}
    for p in sorted(glob.glob(str(R / "exactness" / "exactness_*.json"))):
        if "debug" in p:
            continue
        d = load(p)
        s, tr = d["summary"], d["trials"]
        frac = np.array([t["recomputed"] / max(t["window_updates"], 1) for t in tr])
        cone = np.array([t["recomputed"] for t in tr]); back = np.array([t["steps_back"] for t in tr])
        deg = np.array([t["endpoint_max_degree"] for t in tr])
        sec_e = np.array([t["exact_sec"] for t in tr]); sec_r = np.array([t["replay_sec"] for t in tr])
        hub = deg >= np.quantile(deg, 0.9)
        out[Path(p).stem] = dict(summary=s, cone_frac_median=float(np.median(frac)), cone_frac_p90=float(np.quantile(frac, .9)),
                                 cone_zero_frac=float((cone == 0).mean()),
                                 speedup_median=float(np.median(sec_r / np.maximum(sec_e, 1e-9))),
                                 cone_frac_hub=float(np.median(frac[hub])), cone_frac_tail=float(np.median(frac[~hub])),
                                 corr_cone_back=float(np.corrcoef(cone, back)[0, 1]),
                                 corr_cone_deg=float(np.corrcoef(cone, deg)[0, 1]),
                                 **single_edit_cones(tr))
    return out


def single_edit_cones(tr):
    """Cone sizes of single edits only: share without any effect, median and 90th percentile among the effective ones."""
    one = [t for t in tr if t.get("n_edits", 1) == 1]
    if not one:
        return {}
    f = np.array([t["recomputed"] / max(t["window_updates"], 1) for t in one])
    eff = np.array([t["changed_nodes"] > 0 for t in one])            # edits that change at least one memory
    back = np.array([t["steps_back"] for t in one])
    return dict(single_n=len(one), single_no_effect=float((~eff).mean()),
                single_cone_median_eff=float(np.median(f[eff])) if eff.any() else 0.0,
                single_cone_p90=float(np.quantile(f[eff], 0.9)) if eff.any() else 0.0, single_cone_mean=float(f.mean()),
                single_corr_age=float(np.corrcoef(back, f)[0, 1]) if f.std() > 0 else 0.0)


def deletion():
    groups = defaultdict(list)
    for p in sorted(glob.glob(str(R / "deletion" / "deletion_*.json"))):
        if "debug" in p:
            continue
        d = load(p)
        data, seed, var = parse_model(d["config"]["model"])
        extra = "pairs" if d["config"]["seed"] != seed else "main"
        groups[f"{extra}|{data}|{var}"].append(d)
    out = {}
    for key, runs in groups.items():
        methods = runs[0]["methods"].keys()
        agg = {}
        for m in methods:
            rs = [r["methods"][m] for r in runs]
            a = {}
            for att in ("logit", "probe", "rank"):
                for after in ("0", "5", "20", "50"):
                    vals = [r["attacks"].get(after, {}).get(att) for r in rs]
                    if any(v is not None for v in vals):
                        a[f"{att}_{after}"] = ms(vals)
            agg[m] = dict(attacks=a, cost_x_normal=ms([r["cost_x_normal"] for r in rs]), mrr_after=ms([r["mrr_after"] for r in rs]),
                          resid_cone=ms([r["resid_cone"] for r in rs]), sec=ms([r["sec"] for r in rs]),
                          logit_gap=ms([r["logit_gap_vs_exact"][0] for r in rs]),
                          logit_gap_ci=[r["logit_gap_vs_exact"] for r in rs])
        out[key] = dict(methods=agg, n=len(runs), pairs_A=[r["pairs_A"] for r in runs], pairs_B=[r["pairs_B"] for r in runs],
                        deleted_events=[r["deleted_events"] for r in runs], cone_nodes=ms([r["cone_nodes"] for r in runs]),
                        endpoints=ms([r["endpoints"] for r in runs]), candidates=[r["candidates"] for r in runs])
    return out


def contraction():
    groups = defaultdict(list)
    for p in sorted(glob.glob(str(R / "contraction" / "contraction_*.json"))):
        d = load(p)
        data, seed, var = parse_model(d["config"]["model"])
        groups[f"{data}|{var}"].append(d)
    out = {}
    for key, runs in groups.items():
        j = [r["jacobian"] for r in runs]
        cert = {}
        for eps in ("0.01", "0.05", "0.2"):
            cs = [r["certificate"][eps] for r in runs if eps in r["certificate"]]
            cert[eps] = dict(holds=ms([c["frac_bound_holds"] for c in cs]), ratio=ms([c["median_bound_over_err"] for c in cs]),
                             ratio_p90=ms([c.get("p90_bound_over_err") for c in cs]), max_err=ms([c["max_err"] for c in cs]))
        rig = {}
        for eps in ("0.01", "0.05", "0.2"):
            rr = [r["certificate"][eps].get("rigorous") for r in runs if eps in r["certificate"]]
            rr = [x for x in rr if x]
            if rr:
                rig[eps] = dict(holds=ms([x["frac_holds"] for x in rr]), global_holds=ms([x["frac_global_holds"] for x in rr]),
                                ratio=ms([x["median_bound_over_err"] for x in rr]), global_ratio=ms([x["median_global_over_err"] for x in rr]),
                                cell_rho=ms([x["cell_rho"] for x in rr]))
        cr = {}
        for dl in ("0.005", "0.02", "0.05"):
            xs = [r.get("certified_repair", {}).get(dl) for r in runs]
            xs = [x for x in xs if x]
            if xs:
                cr[dl] = dict(holds=ms([x["frac_holds"] for x in xs]), max_err=max(x["max_err"] for x in xs),
                              cost=ms([x["cost_vs_exact"] for x in xs]), eps_cost=ms([x["eps_cost_vs_exact"] for x in xs]))
        out[key] = dict(certified_repair=cr, rigorous=rig, n=len(runs), updates=[x["updates"] for x in j], certified=ms([x["certified_frac"] for x in j]),
                        J_own=ms([x["J_own"]["median"] for x in j]), J_partner=ms([x["J_partner"]["median"] for x in j]),
                        sum_median=ms([x["sum"]["median"] for x in j]), sum_max=ms([x["sum"]["max"] for x in j]),
                        sum_p99=ms([x["sum"]["p99"] for x in j]), val_ap=ms([r["model_meta"]["val_ap"] for r in runs]),
                        certificate=cert)
    return out


def lira():
    """Likelihood-ratio membership attacks (scripts/run_lira.py), grouped by dataset and model variant."""
    groups = defaultdict(list)
    for p in sorted(glob.glob(str(R / "lira" / "lira_*.json"))):
        d = load(p); data, seed, var = parse_model(d["config"]["model"]); groups[f"{data}|{var}"].append(d)
    out = {}
    for key, runs in groups.items():
        meths = {}
        for m in runs[0]["methods"]:
            rs = [r["methods"][m] for r in runs if m in r["methods"]]
            meths[m] = {f"{f}_{k}": ms([r[f][k] for r in rs]) for f in ("logit", "cosine") for k in ("auc", "tpr1", "tpr5", "auc_offline")}
            meths[m]["raw_logit_auc"] = ms([r["raw_logit_auc"] for r in rs])
        out[key] = dict(n=len(runs), seeds=[r["config"]["seed"] for r in runs], shadows=runs[0]["config"]["shadows"],
                        pairs=ms([r["pairs_A"] for r in runs]), methods=meths)
    return out


def lira_know():
    """Variants of run_lira.py (--know f: weaker attacker; --after k: attack k steps after the request; --K: fewer
    candidate pairs), grouped by dataset, model variant and variant tag."""
    groups = defaultdict(list)
    for p in sorted(glob.glob(str(R / "lira_know" / "lira_*.json"))):
        d = load(p); data, seed, var = parse_model(d["config"]["model"]); c = d["config"]
        tag = "|".join(t for t in (f"know{c.get('know', 1):g}" if c.get("know", 1) < 1 else "",
                                   f"after{c['after']}" if c.get("after", 0) else "",
                                   f"K{c['K']}" if c.get("K", 1000) != 1000 else "") if t)
        groups[f"{data}|{var}|{tag}"].append(d)
    out = {}
    for key, runs in groups.items():
        meths = {}
        for m in runs[0]["methods"]:
            rs = [r["methods"][m] for r in runs if m in r["methods"]]
            meths[m] = {f"{f}_{k}": ms([r[f][k] for r in rs]) for f in ("logit", "cosine") for k in ("auc", "tpr1", "auc_offline")}
            meths[m]["raw_logit_auc"] = ms([r["raw_logit_auc"] for r in rs])
        out[key] = dict(n=len(runs), seeds=[r["config"]["seed"] for r in runs], know=runs[0]["config"].get("know", 1),
                        after=runs[0]["config"].get("after", 0), K=runs[0]["config"].get("K", 1000), methods=meths)
    return out


def train_deletion():
    """Deletions of training-period events (scripts/run_train_deletion.py)."""
    groups = defaultdict(list)
    for p in sorted(glob.glob(str(R / "train_deletion" / "*.json"))):
        d = load(p); groups[d["config"]["data"] + ("|recent" if d["config"].get("recent") else "")].append(d)
    out = {}
    for data, runs in groups.items():
        meths = {m: {k: ms([r["methods"][m][k] for r in runs]) for k in ("logit", "rank", "probe", "mrr")} for m in runs[0]["methods"]}
        mr = [r["memory_repair"] for r in runs]
        out[data] = dict(n=len(runs), seeds=[r["config"]["seed"] for r in runs], pairs=ms([r["pairs_A"] for r in runs]),
                         events=ms([r["events_A"] for r in runs]), methods=meths,
                         train_minutes={k: ms([r["train_minutes"][k] for r in runs]) for k in runs[0]["train_minutes"]},
                         repair_frac=ms([x["recomputed"] / max(x["history_updates"], 1) for x in mr]),
                         repair_sec=ms([x["sec"] for x in mr]), max_abs_vs_rebuild=max(x["max_abs_vs_rebuild"] for x in mr),
                         single_cone_frac=ms([x["single_pair"]["cone_frac_median"] for x in mr]),
                         single_sec=ms([x["single_pair"]["sec_median"] for x in mr]))
    return out


def influence():
    """Exact counterfactual influence vs attention (scripts/run_influence.py)."""
    groups = defaultdict(list)
    for p in sorted(glob.glob(str(R / "influence" / "influence_*.json"))):
        if p.endswith("_records.json"):
            continue
        d = load(p); data, seed, var = parse_model(d["config"]["model"]); groups[data].append(d["summary"])
    keys = ("share_outside_list", "share_memory_channel", "spearman_att_vs_influence_median", "top1_attention_is_top1_influence",
            "top1_influence_inside_list", "same_item_mean_abs", "other_item_mean_abs")
    return {data: dict(n=len(rs), **{k: ms([r[k] for r in rs]) for k in keys}, queries=ms([r["queries"] for r in rs]),
                       by_rank=np.mean([r["mean_abs_influence_by_rank"] for r in rs], 0).tolist(),
                       mem_by_rank=np.mean([r["mean_abs_mem_by_rank"] for r in rs], 0).tolist(),
                       list_by_rank=np.mean([r["mean_abs_list_by_rank"] for r in rs], 0).tolist())
            for data, rs in groups.items()}


def dyrep():
    """DyRep-style memory (scripts/run_dyrep.py)."""
    groups = defaultdict(list)
    for p in sorted(glob.glob(str(R / "dyrep" / "dyrep_*.json"))):
        d = load(p); groups[d["config"]["data"]].append(d)
    out = {}
    for data, runs in groups.items():
        dm = {m: {k: ms([r["deletion"]["methods"][m].get(k) for r in runs]) for k in runs[0]["deletion"]["methods"][m]}
              for m in runs[0]["deletion"]["methods"]}
        out[data] = dict(n=len(runs), val_ap=ms([r["val_ap"] for r in runs]), mrr=ms([r["mrr"] for r in runs]),
                         cone_frac_median=ms([r["exactness"]["cone_frac_median"] for r in runs]),
                         cone_frac_mean=ms([r["exactness"]["cone_frac_mean"] for r in runs]),
                         cone_zero_frac=ms([r["exactness"]["cone_zero_frac"] for r in runs]),
                         speedup=ms([r["exactness"]["speedup_median"] for r in runs]),
                         max_abs_diff=max(r["exactness"]["max_abs_diff"] for r in runs), deletion=dm)
    return out


def main():
    summary = dict(training=training(), streaming=streaming(), exactness=exactness(), deletion=deletion(), contraction=contraction(),
                   lira=lira(), lira_know=lira_know(), train_deletion=train_deletion(), influence=influence(), dyrep=dyrep())
    with open(R / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(" | ".join(f"{k}: {len(v)}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
