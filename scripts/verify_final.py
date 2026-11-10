"""Independent check of the final paper's headline numbers: recompute each from the per-run result files (not from
summary.json / aggregate.py) and compare with the generated macros (report/numbers.json = ../paper/numbers.tex).

  python scripts/verify_final.py          (exit status 1 if any number differs)
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
N = json.load(open(REPO / "report" / "numbers.json"))                     # = ../blog/numbers.json = ../paper/numbers.tex
ok, bad = 0, 0


def check(name, value, fmt):
    global ok, bad
    if name not in N or N[name] == "??":
        print(f"  skip {name} (not generated yet)"); return
    want = fmt.format(value)
    if want == N[name]:
        ok += 1; print(f"  ok   {name} = {want}")
    else:
        bad += 1; print(f"  FAIL {name}: recomputed {want}, paper says {N[name]}")


load = lambda p: json.load(open(p))
seed_of = lambda tag: int(re.search(r"_s(\d+)", tag).group(1))
print("== deletion attacks (results/lira, results/deletion)")
for d, n in (("wikipedia", "Wiki"), ("reddit", "Reddit"), ("lastfm", "Lastfm"), ("mooc", "Mooc"), ("tgbl-review", "Review")):
    runs = [load(p) for p in glob.glob(str(R / "lira" / f"lira_tgn_{d}_s*_p*.json")) if re.fullmatch(rf"tgn_{d}_s\d+", load(p)["config"]["model"])]
    if runs:
        check(f"LiraStore{n}", np.mean([r["methods"]["store"]["cosine"]["auc"] for r in runs]), "{:.2f}")
        check(f"RawStore{n}", np.mean([r["methods"]["store"]["raw_logit_auc"] for r in runs]), "{:.2f}")
        check(f"LiraExact{n}", np.mean([r["methods"]["exact"]["cosine"]["auc"] for r in runs]), "{:.2f}")
        if all("restart" in r["methods"] for r in runs):
            check(f"LiraRestart{n}", np.mean([r["methods"]["restart"]["cosine"]["auc"] for r in runs]), "{:.2f}")
        for m, lab in (("store", "Store"), ("reset", "Reset"), ("restart", "Restart"), ("khop:1", "Khop"), ("eps:0.05", "Eps"), ("exact", "Exact")):
            if all(m in r["methods"] for r in runs):      # the stronger of the two LiRA statistics, by mean AUC over seeds
                check(f"LiraBest{lab}{n}", max(np.mean([r["methods"][m][st]["auc"] for r in runs]) for st in ("cosine", "logit")), "{:.2f}")
    dl = [load(p) for p in glob.glob(str(R / "deletion" / f"deletion_tgn_{d}_s*_p*.json"))]
    dl = [r for r in dl if re.fullmatch(rf"tgn_{d}_s\d+", r["config"]["model"]) and r["config"]["seed"] == seed_of(r["config"]["model"])]
    if dl:
        check(f"DelExactCost{n}", np.mean([r["methods"]["exact"]["cost_x_normal"] for r in dl]), "{:.2f}")
        check(f"DelReplayCost{n}", np.mean([r["methods"]["replay"]["cost_x_normal"] for r in dl]), "{:.2f}")
print("== attackers by access and weaker attackers (results/lira, results/lira_know)")
for d, n in (("wikipedia", "Wiki"), ("reddit", "Reddit"), ("mooc", "Mooc"), ("lastfm", "Lastfm"), ("tgbl-review", "Review")):
    runs = [load(p) for p in glob.glob(str(R / "lira" / f"lira_tgn_{d}_s*_p*.json")) if re.fullmatch(rf"tgn_{d}_s\d+", load(p)["config"]["model"])]
    for m, lab in (("store", "Store"), ("reset", "Reset"), ("restart", "Restart"), ("exact", "Exact")):
        if runs and all(m in r["methods"] for r in runs):
            check(f"LiraQ{lab}{n}", np.mean([r["methods"][m]["logit"]["auc"] for r in runs]), "{:.2f}")
            check(f"LiraW{lab}{n}", np.mean([r["methods"][m]["cosine"]["auc"] for r in runs]), "{:.2f}")
    for tag, vl in (("know0.5", "KnowHalf"), ("know0", "KnowNone"), ("after50", "AfterFifty"), ("K25", "KTwentyFive")):
        vr = []
        for p in glob.glob(str(R / "lira_know" / f"lira_tgn_{d}_s*_p*.json")):
            c = load(p)["config"]
            t = "|".join(x for x in (f"know{c.get('know', 1):g}" if c.get("know", 1) < 1 else "", f"after{c['after']}" if c.get("after", 0) else "",
                                       f"K{c['K']}" if c.get("K", 1000) != 1000 else "") if x)
            if t == tag and re.fullmatch(rf"tgn_{d}_s\d+", c["model"]):
                vr.append(load(p))
        if vr:
            check(f"Lira{vl}QStore{n}", np.mean([r["methods"]["store"]["logit"]["auc"] for r in vr]), "{:.2f}")
print("== exactness (results/exactness)")
ex = {Path(p).stem: load(p)["summary"] for p in glob.glob(str(R / "exactness" / "exactness_*.json"))}
main3 = [ex.get(f"exactness_tgn_{d}_s0") for d in ("wikipedia", "reddit", "mooc")]
if all(main3):
    check("ExTotal", sum(s["trials"] for s in main3), "{:,}")
    check("ExMax", max(s["max_abs_diff"] for s in main3), "{:.1e}")
five = [ex.get(f"exactness_tgn_{d}_s0") for d in ("wikipedia", "reddit", "mooc", "lastfm", "tgbl-review")]
if all(five):
    check("ExTotalAll", sum(s["trials"] for s in five), "{:,}")
    check("ExMaxAll", max(s["max_abs_diff"] for s in five), "{:.1e}")
raw = load(R / "exactness" / "exactness_tgn_wikipedia_s0.json")["trials"]
one = [t for t in raw if t["n_edits"] == 1]
check("ExNoEffectWiki", 100 * np.mean([t["changed_nodes"] == 0 for t in one]), "{:.0f}")       # single edits that change no memory
if "exactness_tgn_wikipedia_s0_nopad" in ex:
    check("ExNoPadFail", 100 * (1 - ex["exactness_tgn_wikipedia_s0_nopad"]["frac_below_1e5"]), "{:.0f}")
print("== contraction (results/contraction)")
for var, n in (("rho0.85_lam10", "EightFive"), ("rho0.75_lam10", "SevenFive")):
    cs = [load(p) for p in glob.glob(str(R / "contraction" / f"contraction_tgn_wikipedia_s*_{var}.json"))]
    if cs:
        check(f"Cert{n}", 100 * np.mean([c["jacobian"]["certified_frac"] for c in cs]), "{:.1f}")
cs = [load(p) for p in glob.glob(str(R / "contraction" / "contraction_tgn_wikipedia_s[0-9].json"))]
if cs:
    check("ConVanillaJown", np.mean([c["jacobian"]["J_own"]["median"] for c in cs]), "{:.2f}")
print("== late events (results/stream)")
for d, n in (("wikipedia", "Wiki"), ("reddit", "Reddit"), ("mooc", "Mooc"), ("lastfm", "Lastfm")):
    rs = [load(p) for p in glob.glob(str(R / "stream" / f"tgn_{d}_s[0-9]_rand_bq200_d[0-9].json"))]
    if rs:
        rel = lambda r, m: 100 * r["methods"][m]["dmrr_vs_exact_rand"][0] / r["methods"]["exact"]["mrr"]
        check(f"ArrivalRel{n}", np.mean([rel(r, "arrival") for r in rs]), "{:+.2f}")
        check(f"BufOneRel{n}", np.mean([rel(r, "buffer:1") for r in rs]), "{:+.1f}")
        check(f"ExactCost{n}", np.mean([r["methods"]["exact"]["cost_x_normal"] for r in rs]), "{:.2f}")
sec = [load(p) for p in glob.glob(str(R / "stream" / "tgn_sec_s[0-9]_real_bq200_d[0-9].json"))]
if sec:
    check("SecAmend", sec[0]["info"]["amendments"], "{:,}")
    check("SecLate", 100 * np.mean([r["info"]["late_frac"] for r in sec]), "{:.0f}")
    check("SecExactCost", np.mean([r["methods"]["exact"]["cost_x_normal"] for r in sec]), "{:.2f}")
print("== influence, DyRep, training-period deletion")
inf = [load(p)["summary"] for p in glob.glob(str(R / "influence" / "influence_tgn_wikipedia_s[0-9].json"))]
if inf:
    check("ShareOutsideWiki", 100 * np.mean([s["share_outside_list"] for s in inf]), "{:.0f}")
dy = [load(p) for p in glob.glob(str(R / "dyrep" / "dyrep_wikipedia_s*.json"))]
if dy:
    check("DyrepLiraStoreWiki", np.mean([r["deletion"]["methods"]["store"]["lira_cosine"] for r in dy]), "{:.2f}")
    check("DyrepConeWiki", 100 * np.mean([r["exactness"]["cone_frac_median"] for r in dy]), "{:.1f}")
recs = [r for p in glob.glob(str(R / "influence" / "influence_tgn_wikipedia_s[0-9]_records.json")) for r in [load(p)]]
if recs:
    top = []
    for rr in recs:
        q = {}
        for r in rr:
            if r["in_list"] and r["att"] is not None:
                q.setdefault(r["q"], []).append(r)
        for v in q.values():
            if len(v) >= 3:
                top.append(int(np.argmax([r["att"] for r in v]) == np.argmax(np.abs([r["d_list"] for r in v]))))
    check("AttTopOneListWiki", 100 * np.mean(top), "{:.0f}")
print("== timing (results/bench)")
for d, n in (("wikipedia", "Wiki"), ("reddit", "Reddit"), ("mooc", "Mooc")):
    p = R / "bench" / f"bench_{d}.json"
    if p.exists():
        check(f"BenchSpeed{n}", load(p)["single"]["speedup_vs_replay"], "{:.0f}")
        b = load(p)
        check(f"BenchExactMs{n}", float(np.median(b["single"]["median_ms"]["indexed"])), "{:.1f}")
        if "stream_ms_per_step" in b:
            check(f"BenchStreamExactMs{n}", b["stream_ms_per_step"]["exact"], "{:.1f}")
print(f"\n{ok} numbers re-derived and equal, {bad} different")
sys.exit(1 if bad else 0)
