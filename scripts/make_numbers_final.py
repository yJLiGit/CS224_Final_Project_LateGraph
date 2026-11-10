"""Every number of the final paper and the blog post, computed from the results (never typed by hand).

  python scripts/make_numbers_final.py    -> report/numbers.tex + report/numbers.json (and ../paper/numbers.tex,
                                             ../blog/numbers.json when the paper and the blog post are next to the repository)
A number whose results are missing is written as '??' (and listed at the end), so a draft can be built at any time.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
S = json.load(open(R / "summary.json"))
N, missing = {}, []


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def put(name, value, fmt="{:.2f}"):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        N[name] = "??"; missing.append(name); return
    N[name] = fmt.format(value) if not isinstance(value, str) else value


def mean(x):
    return None if x is None else x["mean"]


def lira_best(L, m):
    """The stronger of the two LiRA statistics (link score, memory cosine) of a deletion method, by mean AUC over seeds;
    an attacker can make this choice on its own shadow runs."""
    xs = [x for x in (get(L, m, "cosine_auc"), get(L, m, "logit_auc")) if x is not None]
    return max(xs, key=lambda x: x["mean"]) if xs else None


def pm(x, d=3):
    return None if x is None else f"{x['mean']:.{d}f} $\\pm$ {x['std']:.{d}f}"


DS = dict(Wiki="wikipedia", Reddit="reddit", Mooc="mooc", Lastfm="lastfm", Review="tgbl-review")
# ---------------------------------------------------------------------------------------------------- data
for n, d in DS.items():
    st = get(S, "streaming", f"main|{d}|attn|rand|bq200")
    put(f"N{n}TestSteps", st["J"] if st else None, "{:,}")
# ---------------------------------------------------------------------------------------------------- deletion and attacks
for n, d in DS.items():
    L = get(S, "lira", f"{d}|attn", "methods")
    put(f"LiraStore{n}", mean(get(L, "store", "cosine_auc")), "{:.2f}")
    put(f"LiraStoreLogit{n}", mean(get(L, "store", "logit_auc")), "{:.2f}")
    put(f"RawStore{n}", mean(get(L, "store", "raw_logit_auc")), "{:.2f}")
    put(f"LiraRestart{n}", mean(get(L, "restart", "cosine_auc")), "{:.2f}")
    put(f"LiraReset{n}", mean(get(L, "reset", "cosine_auc")), "{:.2f}")
    put(f"LiraEps{n}", mean(get(L, "eps:0.05", "cosine_auc")), "{:.2f}")
    put(f"LiraExact{n}", mean(get(L, "exact", "cosine_auc")), "{:.2f}")
    put(f"RawReset{n}", mean(get(L, "reset", "raw_logit_auc")), "{:.2f}")
    for m, lab in (("store", "Store"), ("reset", "Reset"), ("restart", "Restart"), ("khop:1", "Khop"), ("eps:0.05", "Eps"), ("exact", "Exact")):
        put(f"LiraBest{lab}{n}", mean(lira_best(L, m)), "{:.2f}")
    put(f"LiraTprStore{n}", None if get(L, "store", "cosine_tpr1") is None else 100 * mean(get(L, "store", "cosine_tpr1")), "{:.0f}")
    put(f"LiraSeeds{n}", get(S, "lira", f"{d}|attn", "n"), "{:d}")
    Dm = get(S, "deletion", f"main|{d}|attn", "methods")
    put(f"DelExactCost{n}", mean(get(Dm, "exact", "cost_x_normal")), "{:.2f}")
    put(f"DelReplayCost{n}", mean(get(Dm, "replay", "cost_x_normal")), "{:.2f}")
    put(f"DelEpsCost{n}", mean(get(Dm, "eps:0.05", "cost_x_normal")), "{:.2f}")
    put(f"DelKhopCost{n}", mean(get(Dm, "khop:1", "cost_x_normal")), "{:.2f}")
    put(f"DelProbeStore{n}", mean(get(Dm, "store", "attacks", "probe_0")), "{:.2f}")
    put(f"DelProbeRestart{n}", mean(get(Dm, "restart", "attacks", "probe_0")), "{:.2f}")
    put(f"DelProbeExact{n}", mean(get(Dm, "exact", "attacks", "probe_0")), "{:.2f}")
# ---------------------------------------------------------------------------------------------------- attackers by access
METH = (("store", "Store"), ("reset", "Reset"), ("restart", "Restart"), ("khop:1", "Khop"), ("eps:0.05", "Eps"), ("exact", "Exact"))
for n, d in DS.items():
    L = get(S, "lira", f"{d}|attn", "methods")
    Dm = get(S, "deletion", f"main|{d}|attn", "methods")
    for m, lab in METH:
        put(f"LiraQ{lab}{n}", mean(get(L, m, "logit_auc")), "{:.2f}")            # query access: LiRA on the link score
        put(f"LiraW{lab}{n}", mean(get(L, m, "cosine_auc")), "{:.2f}")           # white-box: LiRA on the memories
        put(f"LiraQTprFive{lab}{n}", None if get(L, m, "logit_tpr5") is None else 100 * mean(get(L, m, "logit_tpr5")), "{:.0f}")
    put(f"DelProbeReset{n}", mean(get(Dm, "reset", "attacks", "probe_0")), "{:.2f}")
    put(f"LiraPairs{n}", mean(get(S, "lira", f"{d}|attn", "pairs")), "{:,.0f}")
    put(f"DelExactMrrAfter{n}", mean(get(Dm, "exact", "mrr_after")), "{:.3f}")
    put(f"DelNoneMrrAfter{n}", mean(get(Dm, "none", "mrr_after")), "{:.3f}")
    put(f"DelResetMrrAfter{n}", mean(get(Dm, "reset", "mrr_after")), "{:.3f}")
# largest AUC of eps-repair, the 1-hop rebuild and exact repair against any attacker with partial knowledge
part = [mean(get(S, "lira_know", f"{d}|attn|{tag}", "methods", m, "logit_auc")) for d in DS.values()
        for tag in ("know0.5", "know0.1", "know0") for m in ("eps:0.05", "khop:1", "exact")]          # query access, K = 1,000
part = [x for x in part if x is not None]
put("PartialApproxMax", max(part) if part else None, "{:.2f}")
# with K = 25 the AUCs are noisy (25 pairs per group): the range of exact repair, which leaks nothing, shows how much
k25 = [mean(get(S, "lira_know", f"{d}|attn|{tag}", "methods", "exact", st)) for d in DS.values()
       for tag in ("K25", "know0.5|K25", "know0|K25") for st in ("logit_auc", "cosine_auc")]
k25 = [x for x in k25 if x is not None]
put("ExactKTwentyFiveMin", min(k25) if k25 else None, "{:.2f}")
put("ExactKTwentyFiveMax", max(k25) if k25 else None, "{:.2f}")
SL = json.load(open(R / "step_lengths.json")) if (R / "step_lengths.json").exists() else {}
for n, d in DS.items():
    put(f"StepMin{n}", get(SL, d, "minutes_per_step"), "{:.0f}")
put("SecLagMed", get(SL, "sec", "lag_median_days"), "{:.1f}")
put("SecLagNinetyNine", get(SL, "sec", "lag_p99_days"), "{:.0f}")
put("SecLagFour", None if get(SL, "sec", "lag_over4_frac") is None else 100 * SL["sec"]["lag_over4_frac"], "{:.0f}")
# models trained for 30 instead of 15 epochs: does the leak depend on under-training?
for n, d in DS.items():
    L30 = get(S, "lira", f"{d}|ep30", "methods")
    for m, lab in METH:
        put(f"LiraQ{lab}EpThirty{n}", mean(get(L30, m, "logit_auc")), "{:.2f}")
        put(f"LiraW{lab}EpThirty{n}", mean(get(L30, m, "cosine_auc")), "{:.2f}")
    put(f"ValApEpThirty{n}", mean(get(S, "training", f"{d}|ep30", "val_ap")), "{:.3f}")
    put(f"ValAp{n}", mean(get(S, "training", f"{d}|attn", "val_ap")), "{:.3f}")
VARIANTS = (("know0.5", "KnowHalf"), ("know0.1", "KnowTenth"), ("know0", "KnowNone"), ("after20", "AfterTwenty"),
            ("after50", "AfterFifty"), ("K100", "KHundred"), ("K25", "KTwentyFive"), ("know0.5|K25", "KnowHalfKTwentyFive"),
            ("know0|K25", "KnowNoneKTwentyFive"))
for tag, vl in VARIANTS:
    for n, d in DS.items():
        Lv = get(S, "lira_know", f"{d}|attn|{tag}", "methods")
        for m, lab in METH:
            put(f"Lira{vl}Q{lab}{n}", mean(get(Lv, m, "logit_auc")), "{:.2f}")
            put(f"Lira{vl}W{lab}{n}", mean(get(Lv, m, "cosine_auc")), "{:.2f}")
# ---------------------------------------------------------------------------------------------------- weaker attacker
KN = S.get("lira_know", {})
for know, kl in ((0.5, "Half"), (0.1, "Tenth"), (0.0, "None")):
    for n, d in DS.items():
        Lk = get(KN, f"{d}|attn|know{know:g}", "methods")
        for m, lab in (("store", "Store"), ("reset", "Reset"), ("restart", "Restart"), ("khop:1", "Khop"), ("eps:0.05", "Eps"), ("exact", "Exact")):
            put(f"LiraKnow{kl}{lab}{n}", mean(lira_best(Lk, m)), "{:.2f}")
# ---------------------------------------------------------------------------------------------------- exactness
EX = S.get("exactness", {})
tot = sum(get(EX, k, "summary", "trials") or 0 for k in ("exactness_tgn_wikipedia_s0", "exactness_tgn_reddit_s0", "exactness_tgn_mooc_s0"))
put("ExTotal", tot or None, "{:,}")
mx = [get(EX, k, "summary", "max_abs_diff") for k in ("exactness_tgn_wikipedia_s0", "exactness_tgn_reddit_s0", "exactness_tgn_mooc_s0")]
put("ExMax", max(mx) if all(x is not None for x in mx) else None, "{:.1e}")
five = [f"exactness_tgn_{d}_s0" for d in ("wikipedia", "reddit", "mooc", "lastfm", "tgbl-review")]
put("ExTotalAll", sum(get(EX, k, "summary", "trials") or 0 for k in five) or None, "{:,}")
mx5 = [get(EX, k, "summary", "max_abs_diff") for k in five]
put("ExMaxAll", max(mx5) if all(x is not None for x in mx5) else None, "{:.1e}")
put("ExMaxJodie", get(EX, "exactness_tgn_wikipedia_s0_jodie", "summary", "max_abs_diff"), "{:.1e}")
put("ExMaxDyrep", get(S, "dyrep", "wikipedia", "max_abs_diff"), "{:.1e}")

for n, d in DS.items():
    put(f"ExCone{n}", None if get(EX, f"exactness_tgn_{d}_s0", "cone_frac_median") is None else 100 * get(EX, f"exactness_tgn_{d}_s0", "cone_frac_median"), "{:.2f}")
    e = get(EX, f"exactness_tgn_{d}_s0")
    put(f"ExNoEffect{n}", None if get(e, "single_no_effect") is None else 100 * e["single_no_effect"], "{:.0f}")     # single edits
    put(f"ExConeEff{n}", None if get(e, "single_cone_median_eff") is None else 100 * e["single_cone_median_eff"], "{:.1f}")
    put(f"ExConePNinety{n}", None if get(e, "single_cone_p90") is None else 100 * e["single_cone_p90"], "{:.0f}")
nopad = get(EX, "exactness_tgn_wikipedia_s0_nopad", "summary")
put("ExNoPadFail", None if nopad is None else 100 * (1 - nopad["frac_below_1e5"]), "{:.0f}")
f64 = get(EX, "exactness_tgn_wikipedia_s0_f64", "summary")
put("ExDoubleTrials", None if f64 is None else f64["trials"], "{:,}")
put("ExDoubleMax", None if f64 is None else f64["max_abs_diff"], "{:.0f}")
# ---------------------------------------------------------------------------------------------------- timing benchmark
for n, d in DS.items():
    b = R / "bench" / f"bench_{d}.json"
    bj = json.load(open(b)) if b.exists() else {}
    put(f"BenchSpeed{n}", get(bj, "single", "speedup_vs_replay"), "{:.0f}")
    put(f"BenchWalkSpeed{n}", get(bj, "single", "walk_speedup_vs_replay"), "{:.1f}")
    put(f"BenchExactMs{n}", get(bj, "single", "median_ms", "indexed"), "{:.1f}")
    put(f"BenchReplayMs{n}", get(bj, "single", "median_ms", "replay"), "{:.1f}")
    put(f"BenchBatchSpeed{n}", get(bj, "batch5", "speedup_vs_replay"), "{:.1f}")
    put(f"BenchSpeedEff{n}", get(bj, "single", "speedup_vs_replay_effective"), "{:.1f}")
    put(f"BenchWalkSpeedEff{n}", get(bj, "single", "walk_speedup_vs_replay_effective"), "{:.1f}")
    for k, lab in (("exact", "Exact"), ("exact_walk", "Walk"), ("replay", "Replay"), ("eps:0.05", "Eps"), ("arrival", "Arrival")):
        put(f"BenchStream{lab}Ms{n}", get(bj, "stream_ms_per_step", k), "{:.1f}")
AGE_OLD = 200                                    # "old" edits: at least this many steps before the end of the window
put("AgeOldSteps", AGE_OLD, "{:d}")
for n, d in DS.items():
    b = R / "bench_age" / f"bench_age_{d}.json"
    ba = json.load(open(b)) if b.exists() else None
    if ba:
        tr = ba["trials"]; age = np.array([t["age_steps"] for t in tr])
        sp = np.array([t["ms_replay"] / max(t["ms_indexed"], 1e-6) for t in tr])
        put(f"BenchAgeOldSpeed{n}", float(np.median(sp[age >= AGE_OLD])) if (age >= AGE_OLD).sum() >= 5 else None, "{:.1f}")
        put(f"BenchAgeRecentSpeed{n}", float(np.median(sp[age <= 20])) if (age <= 20).sum() >= 5 else None, "{:.0f}")
        mid = (age > 10) & (age <= 50)
        put(f"BenchAgeMidSpeed{n}", float(np.median(sp[mid])) if mid.sum() >= 5 else None, "{:.0f}")
        eff = np.array([t["recomputed"] > 0 for t in tr])                 # edits that recompute something
        for lab_, sel in (("Recent", eff & (age <= 20)), ("Mid", eff & (age > 10) & (age <= 50)), ("Old", eff & (age >= AGE_OLD)),
                          ("All", eff)):
            put(f"BenchAge{lab_}SpeedEff{n}", float(np.median(sp[sel])) if sel.sum() >= 4 else None, "{:.1f}")
        put(f"BenchAgeWindow{n}", ba["window_steps"], "{:d}")
        put(f"HistPassSec{n}", ba["history_pass_sec"], "{:.2f}")
    else:
        for k in ("BenchAgeOldSpeed", "BenchAgeRecentSpeed", "BenchAgeWindow", "HistPassSec"):
            put(f"{k}{n}", None)
# ---------------------------------------------------------------------------------------------------- contraction / certificate
C = S.get("contraction", {})
put("ConVanillaJown", mean(get(C, "wikipedia|attn", "J_own")), "{:.2f}")
put("ConVanillaJsum", mean(get(C, "wikipedia|attn", "sum_median")), "{:.2f}")
put("ConVanillaSumMed", mean(get(C, "wikipedia|attn", "sum_median")), "{:.2f}")
for var, n in (("rho0.85_lam10", "EightFive"), ("rho0.75_lam10", "SevenFive"), ("rho0.95_lam10", "NineFive")):
    put(f"Cert{n}", None if get(C, f"wikipedia|{var}", "certified") is None else 100 * mean(get(C, f"wikipedia|{var}", "certified")), "{:.1f}")
for var, n in (("crnn0.9", "NineCell"), ("crnn0.8", "EightCell"), ("crnn0.95", "NineFiveCell")):
    rg = get(C, f"wikipedia|{var}", "rigorous", "0.05")
    put(f"Rho{n}", mean(get(rg, "cell_rho")), "{:.2f}")
    put(f"RigHolds{n}", None if rg is None else 100 * mean(rg["holds"]), "{:.0f}")
    put(f"RigRatio{n}", mean(get(rg, "ratio")), "{:.1f}")
    cr = get(C, f"wikipedia|{var}", "certified_repair", "0.02")
    put(f"CertRepairCost{n}", mean(get(cr, "cost")), "{:.2f}")
    put(f"CertRepairHolds{n}", None if cr is None else 100 * mean(cr["holds"]), "{:.0f}")
St = S.get("streaming", {})
ora = lambda v: mean(get(St, f"main|wikipedia|{v}|rand|bq200", "methods", "oracle", "mrr"))
base_mrr = ora("attn")
for var, n in (("rho0.85_lam10", "EightFive"), ("rho0.75_lam10", "SevenFive"), ("crnn0.9", "NineCell"), ("crnn0.8", "EightCell")):
    put(f"MrrLoss{n}", None if base_mrr is None or ora(var) is None else 100 * (round(base_mrr, 3) - round(ora(var), 3)), "{:.1f}")
N["CrnnMrrLoss"] = N["MrrLossNineCell"]
# ---------------------------------------------------------------------------------------------------- late events
for n, d in DS.items():
    g = get(St, f"main|{d}|attn|rand|bq200", "methods")
    put(f"ExactCost{n}", mean(get(g, "exact", "cost_x_normal")), "{:.2f}")
    put(f"EpsCost{n}", mean(get(g, "eps:0.05", "cost_x_normal")), "{:.2f}")
    put(f"ReplayCost{n}", mean(get(g, "replay", "cost_x_normal")), "{:.1f}")
    put(f"ArrivalRel{n}", mean(get(g, "arrival", "dmrr_rel_pct")), "{:+.2f}")
    put(f"ArrivalStd{n}", get(g, "arrival", "dmrr_rel_pct", "std"), "{:.1f}")
    put(f"BufOneRel{n}", mean(get(g, "buffer:1", "dmrr_rel_pct")), "{:+.1f}")
    put(f"EpsRel{n}", mean(get(g, "eps:0.05", "dmrr_rel_pct")), "{:+.2f}")
    put(f"Seeds{n}", get(St, f"main|{d}|attn|rand|bq200", "n"), "{:d}")
    put(f"JournalMB{n}", mean(get(g, "exact", "journal_MB")), "{:.0f}")
N["JournalWikiMB"] = N["JournalMBWiki"]
# the journal needed to delete from the whole history, scaled from the measured test-period journal by the event counts
for n, d in DS.items():
    st = get(St, f"main|{d}|attn|rand|bq200"); jm = mean(get(st, "methods", "exact", "journal_MB"))
    tr = get(S, "training", f"{d}|attn")
    ev = {"wikipedia": 157474, "reddit": 672447, "mooc": 411749, "lastfm": 1293103, "tgbl-review": 4873540}.get(d)
    put(f"HistJournalGB{n}", None if jm is None or st is None else jm * (ev - st["test_events"]) / st["test_events"] / 1000, "{:.1f}")
# ---------------------------------------------------------------------------------------------------- SEC
sec = get(St, "main_amend|sec|attn|real|bq200")
put("SecAmend", None if sec is None else sec["amendments"], "{:,}")
put("SecLate", None if sec is None else 100 * mean(sec["late_frac"]), "{:.0f}")
put("SecExactCost", mean(get(sec, "methods", "exact", "cost_x_normal")), "{:.2f}")
put("SecSeeds", None if sec is None else sec["n"], "{:d}")
for n, d in DS.items():
    put(f"TrainSingleSec{n}", mean(get(S, "train_deletion", d, "single_sec")), "{:.2f}")
rel = lambda m: get(sec, "methods", m, "dmrr_rel_pct")
spread = [abs(mean(rel(m))) for m in ("arrival", "buffer:24", "buffer:96", "eps:0.05", "khop:1") if rel(m)]
put("SecSpreadPct", max(spread) if spread else None, "{:.2f}")
put("SecDropRel", mean(rel("drop")), "{:+.2f}")
put("SecExactMrr", mean(get(sec, "methods", "exact", "mrr")), "{:.2f}")
put("SecOracleRel", mean(rel("oracle")), "{:+.1f}")
# ---------------------------------------------------------------------------------------------------- influence
I = S.get("influence", {})
for n, d in DS.items():
    put(f"ShareOutside{n}", None if get(I, d) is None else 100 * mean(I[d]["share_outside_list"]), "{:.0f}")
    put(f"InflQueries{n}", mean(get(I, d, "queries")), "{:.0f}")
    put(f"SpearmanAtt{n}", mean(get(I, d, "spearman_att_vs_influence_median")), "{:+.2f}")
    put(f"TopOneAtt{n}", None if get(I, d) is None else 100 * mean(I[d]["top1_attention_is_top1_influence"]), "{:.0f}")
    put(f"MemChannel{n}", None if get(I, d) is None else 100 * mean(I[d]["share_memory_channel"]), "{:.0f}")
AV = json.load(open(R / "influence" / "attention_vs_channels.json")) if (R / "influence" / "attention_vs_channels.json").exists() else {}
for n, d in DS.items():
    a_ = AV.get(d)
    put(f"AttTopOneList{n}", None if a_ is None else 100 * a_["top1_list"], "{:.0f}")
    put(f"AttTopOneTotal{n}", None if a_ is None else 100 * a_["top1_total"], "{:.0f}")
    put(f"AttTopOneTotalLo{n}", None if a_ is None else 100 * a_["top1_total_ci"][0], "{:.0f}")
    put(f"AttTopOneTotalHi{n}", None if a_ is None else 100 * a_["top1_total_ci"][1], "{:.0f}")
    put(f"AttSpearmanList{n}", None if a_ is None else a_["spearman_list_median"], "{:+.2f}")
    ps = None if a_ is None else a_.get("top1_total_per_seed")
    put(f"AttTopOneTotalSeedMin{n}", None if not ps else 100 * min(ps), "{:.0f}")
    put(f"AttTopOneTotalSeedMax{n}", None if not ps else 100 * max(ps), "{:.0f}")
# ---------------------------------------------------------------------------------------------------- DyRep
Dy = S.get("dyrep", {})
for n, d in DS.items():
    put(f"DyrepCone{n}", None if get(Dy, d) is None else 100 * mean(Dy[d]["cone_frac_median"]), "{:.1f}")
    put(f"DyrepMrr{n}", mean(get(Dy, d, "mrr")), "{:.3f}")
    put(f"DyrepLiraStore{n}", mean(get(Dy, d, "deletion", "store", "lira_cosine")), "{:.2f}")
cw, dw = get(EX, "exactness_tgn_wikipedia_s0", "cone_frac_median"), get(Dy, "wikipedia", "cone_frac_median")
put("DyrepConeRatio", None if cw is None or dw is None or cw == 0 else mean(dw) / cw, "{:.0f}")

# ---------------------------------------------------------------------------------------------------- sentences with numbers
T = S.get("train_deletion", {})


def tdm(ds, m, k):
    return mean(get(T, ds, "methods", m, k))


if get(T, "wikipedia"):
    have = [(d, lab) for d, lab in (("wikipedia", "Wikipedia"), ("reddit", "Reddit"), ("mooc", "MOOC")) if get(T, d)]
    per = lambda m, k="logit": " / ".join(f"{tdm(d, m, k):.2f}" for d, _ in have)
    names = " / ".join(lab for _, lab in have)
    txt = (f"Before any deletion the deployed model separates these pairs from never-seen ones only weakly (link-score AUC "
           f"{per('none')} on {names}); log-only deletion leaves {per('store')}, and repairing the memory with unchanged "
           f"weights {per('memory')}, level with retraining ({per('retrain')}); the probe agrees, also for pairs from the last "
           f"20\\% of the training period. These attacks are weak here, so we claim only that no signal remains that they "
           f"detect; a shadow-model attack on the weights is future work.")
    cone = get(T, "wikipedia", "single_cone_frac")
    hist = N.get("HistPassSecWiki", "??")
    if cone is not None:
        hr = N.get("HistPassSecReddit", "??")
        rr = T.get("reddit", {}).get("single_sec", {}).get("mean")
        txt += (f" The cone of one training-period pair covers {100 * cone['mean']:.0f}\\% of the later history updates on "
                f"Wikipedia, so recomputing the memory from scratch with fixed weights, one batched pass over the history "
                f"({hist}\\,s on an idle machine; Reddit: {hr}\\,s), is faster than repairing it ("
                f"{T['wikipedia']['single_sec']['mean']:.2f}\\,s and " + (f"{rr:.2f}" if rr else "??") + "\\,s, measured during the "
                f"experiment runs); both are exact, and far cheaper than retraining "
                f"({T['wikipedia']['train_minutes']['retrain']['mean']:.0f} minutes on Wikipedia).")
    N["TrainDelText"] = txt
else:
    N["TrainDelText"] = "(results pending)"; missing.append("TrainDelText")
put("LiraStoreJodieWiki", mean(lira_best(get(S, "lira", "wikipedia|jodie", "methods"), "store")), "{:.2f}")
put("LiraQStoreJodieWiki", mean(get(S, "lira", "wikipedia|jodie", "methods", "store", "logit_auc")), "{:.2f}")
put("DyrepLiraQStoreWiki", mean(get(S, "dyrep", "wikipedia", "deletion", "store", "lira_logit")), "{:.2f}")
put("MrrOracleWiki", mean(get(S, "streaming", "main|wikipedia|attn|rand|bq200", "methods", "oracle", "mrr")), "{:.3f}")
put("MrrOracleNomemWiki", mean(get(S, "streaming", "main|wikipedia|nomem|rand|bq200", "methods", "oracle", "mrr")), "{:.3f}")
A = get(S, "lira", "wikipedia|attn", "methods", "store", "logit_auc")          # LiRA on the link score (query access)
J = get(S, "lira", "wikipedia|jodie", "methods", "store", "logit_auc")
Dyl = get(S, "dyrep", "wikipedia", "deletion", "store", "lira_logit")
if A and J and Dyl:
    N["ArchText"] = (f"Every memory model leaks after a log-only deletion (LiRA AUC on scores {A['mean']:.2f} for TGN, {J['mean']:.2f} for "
                     f"JODIE, {Dyl['mean']:.2f} for DyRep), while the memory-free control forgets by construction but loses "
                     f"accuracy. DyRep's neighbourhood reads make its cones much larger than TGN's and JODIE's, which read two "
                     f"memories; the contractive cell trades accuracy for a provable error bound.")
else:
    N["ArchText"] = "(results pending)"; missing.append("ArchText")

# the report inside the repository always; the paper and the blog post next to it when they exist
tex_dirs = [REPO / "report"] + [d for d in (REPO.parent / "paper",) if d.exists()]
json_dirs = [REPO / "report"] + [d for d in (REPO.parent / "blog",) if d.exists()]
for d in tex_dirs:
    with open(d / "numbers.tex", "w") as f:
        f.write("% generated by scripts/make_numbers_final.py -- do not edit\n")
        for k in sorted(N):
            f.write(f"\\newcommand{{\\{k}}}{{{N[k]}}}\n")
for d in json_dirs:
    json.dump(N, open(d / "numbers.json", "w"), indent=1, sort_keys=True)
print(f"{len(N)} numbers; {len(missing)} still missing" + (f": {missing[:12]}" if missing else ""))
