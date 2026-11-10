"""Write report/numbers.tex: every number quoted in the milestone report, as a LaTeX macro computed from results/.

  python scripts/make_numbers.py
Macro names use letters only (LaTeX rule): One/Four for 1 h / 4 h buffers, Eps for epsilon = 0.05, etc.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph import paths                     # noqa: E402

S = json.load(open(paths.RESULTS / "summary.json"))
OUT = paths.REPORT / "numbers.tex"
lines = []


def put(name, value):
    assert name.isalpha(), name
    lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def f(x, d=3):
    return f"{x:.{d}f}"


def sci(x, d=1):
    """5.4e-07 -> $5.4{\times}10^{-7}$ (0 stays 0)."""
    if x == 0:
        return "0"
    m, e = f"{x:.{d}e}".split("e")
    return f"${m}{{\\times}}10^{{{int(e)}}}$"


def pm(s, d=3, scale=1.0):
    if s is None:
        return "--"
    mean = f"{scale * s['mean']:.{d}f}"
    if mean.startswith("-"):
        mean = "$-$" + mean[1:] if float(mean) != 0 else mean[1:]
    return f"{mean}\\,$\\pm$\\,{scale * s['std']:.{d}f}"


def sg(x, d=2):
    v = f"{x:+.{d}f}"
    if float(v) == 0:
        v = f"{0:.{d}f}"
    return v.replace("-", "$-$")


DSN = {"wikipedia": "Wiki", "reddit": "Reddit", "mooc": "Mooc"}
st = S["streaming"]

# ---------------------------------------------------------------- training
for ds, n in list(DSN.items()) + [("sec", "Sec")]:
    t = S["training"].get(f"{ds}|attn")
    if t:
        put(f"ValAP{n}", pm(t["val_ap"], 3))
for var, n in (("id", "WikiMemOnly"), ("nomem", "WikiNoMem"), ("rho0.75_lam10", "WikiRhoSevenFive"),
               ("rho0.85_lam10", "WikiRhoEightFive"), ("rho0.95_lam10", "WikiRhoNineFive")):
    t = S["training"].get(f"wikipedia|{var}")
    if t:
        put(f"ValAP{n}", pm(t["val_ap"], 3))

# ---------------------------------------------------------------- streaming (random delays, steps of 200)
for ds, n in DSN.items():
    g = st.get(f"main|{ds}|attn|rand|bq200")
    if not g:
        continue
    m = g["methods"]
    put(f"{n}Late", f(100 * g["late_frac"]["mean"], 1))
    put(f"{n}InTransit", f(100 * g["in_transit_frac"]["mean"], 0))
    put(f"{n}LateStepsMed", f(g["late_steps_median"]["mean"], 0))
    put(f"{n}Oracle", pm(m["oracle"]["mrr"])); put(f"{n}Exact", pm(m["exact"]["mrr"]))
    for meth, key in (("arrival", "Arrival"), ("arrival_proc", "Proc"), ("drop", "Drop"), ("buffer:1", "BufOne"),
                      ("buffer:4", "BufFour"), ("eps:0.05", "Eps"), ("eps:0.01", "EpsSmall"), ("eps:0.2", "EpsBig"),
                      ("khop:1", "KhopOne"), ("khop:2", "KhopTwo"), ("oracle", "OracleRel")):
        if meth in m and m[meth]["dmrr_rel_pct"]:
            put(f"{n}{key}Rel", pm(m[meth]["dmrr_rel_pct"], 2))
            put(f"{n}{key}RelMean", sg(m[meth]["dmrr_rel_pct"]["mean"], 2))
    for meth, key in (("replay", "Replay"), ("exact", "Exact"), ("eps:0.01", "EpsSmall"), ("eps:0.05", "Eps"),
                      ("eps:0.2", "EpsBig"), ("khop:1", "KhopOne"), ("khop:2", "KhopTwo"), ("arrival", "Arrival"),
                      ("buffer:1", "BufOne")):
        if meth in m:
            c = m[meth]["cost_x_normal"]["mean"]
            put(f"{n}{key}Cost", f(c, 1) if c < 10 else f(c, 0))
            put(f"{n}{key}Agree", f(100 * m[meth]["top1_agree"]["mean"], 1))
            put(f"{n}{key}Div", f(m[meth]["mem_div"]["mean"], 3))
            put(f"{n}{key}Ms", f(1000 * m[meth]["sec"]["mean"] / g["J"], 0))
    st_ = m["exact"].get("strata")
    if st_:
        for meth, key in (("arrival", "Arrival"), ("buffer:1", "BufOne")):
            a, b = m[meth]["strata"]["in_transit"]["mean"], m["exact"]["strata"]["in_transit"]["mean"]
            c, d = m[meth]["strata"]["not_in_transit"]["mean"], m["exact"]["strata"]["not_in_transit"]["mean"]
            put(f"{n}{key}TransitRel", sg(100 * (a - b) / b, 1)); put(f"{n}{key}NotTransitRel", sg(100 * (c - d) / d, 1))
    put(f"{n}EpsAbsDiff", f(abs(m["eps:0.05"]["mrr"]["mean"] - m["exact"]["mrr"]["mean"]), 4))

# ranges over all JODIE delay settings (attn models, steps of 200 and 50)
arr_abs, buf1, buf4, eps_abs, n_settings, settings = [], [], [], [], 0, []
for k, g in st.items():
    kind, ds, var, delay, bq = k.split("|")
    if kind != "main" or ds not in DSN or var != "attn":
        continue
    n_settings += 1; m = g["methods"]
    settings.append(k)
    for s_ in m["arrival"]["dmrr_rel_pct"]["values"]:
        arr_abs.append(abs(s_))
    buf1 += m["buffer:1"]["dmrr_rel_pct"]["values"]; buf4 += m["buffer:4"]["dmrr_rel_pct"]["values"]
    eps_abs += [abs(x) for x in m["eps:0.05"]["dmrr_vs_exact"]["values"]]
put("NSettings", str(n_settings)); put("NRuns", str(len(arr_abs)))
put("ArrivalMaxAbs", f(max(arr_abs), 1)); put("ArrivalMedAbs", f(float(np.median(arr_abs)), 2))
put("BufOneMin", f(-max(buf1), 1)); put("BufOneMax", f(-min(buf1), 1))
put("BufFourMin", f(-max(buf4), 1)); put("BufFourMax", f(-min(buf4), 1))
put("EpsMaxAbs", f(max(eps_abs), 4))
arr_abs_mrr = []
per_ds_buf1 = {"wikipedia": [], "reddit": [], "mooc": []}
for k, g in st.items():
    kind, ds, var, delay, bq = k.split("|")
    if kind != "main" or ds not in DSN or var != "attn":
        continue
    arr_abs_mrr += [abs(x) for x in g["methods"]["arrival"]["dmrr_vs_exact"]["values"]]
    per_ds_buf1[ds] += g["methods"]["buffer:1"]["dmrr_rel_pct"]["values"]
put("ArrivalMaxAbsMRR", f(max(arr_abs_mrr), 3))
for ds, n in DSN.items():
    put(f"{n}BufOneLo", f(-max(per_ds_buf1[ds]), 1)); put(f"{n}BufOneHi", f(-min(per_ds_buf1[ds]), 1))
for ds, n in (("wikipedia", "Wiki"), ("mooc", "Mooc")):
    g = st.get(f"main|{ds}|attn|rand|bq50")
    if g:
        m = g["methods"]
        put(f"{n}FiftyBufOneRel", sg(m["buffer:1"]["dmrr_rel_pct"]["mean"], 1)); put(f"{n}FiftyBufFourRel", sg(m["buffer:4"]["dmrr_rel_pct"]["mean"], 1))
        put(f"{n}FiftyExactCost", f(m["exact"]["cost_x_normal"]["mean"], 0 if m["exact"]["cost_x_normal"]["mean"] >= 10 else 1))
        put(f"{n}FiftyEpsCost", f(m["eps:0.05"]["cost_x_normal"]["mean"], 1)); put(f"{n}FiftyReplayCost", f(m["replay"]["cost_x_normal"]["mean"], 0))
        put(f"{n}FiftyArrivalRel", sg(m["arrival"]["dmrr_rel_pct"]["mean"], 1))
# memory-only / memory-free / contractive
for var, n in (("id", "MemOnly"), ("nomem", "NoMem"), ("rho0.75_lam10", "RhoSevenFive"), ("rho0.85_lam10", "RhoEightFive"),
               ("rho0.95_lam10", "RhoNineFive")):
    g = st.get(f"main|wikipedia|{var}|rand|bq200")
    if g:
        m = g["methods"]
        put(f"Wiki{n}Oracle", pm(m["oracle"]["mrr"])); put(f"Wiki{n}OracleMean", f(m["oracle"]["mrr"]["mean"], 3))
        put(f"Wiki{n}ArrivalRel", pm(m["arrival"]["dmrr_rel_pct"], 2)); put(f"Wiki{n}BufOneRel", pm(m["buffer:1"]["dmrr_rel_pct"], 2))
g = st.get("main|wikipedia|attn|rand|bq200")
put("WikiOracleMean", f(g["methods"]["oracle"]["mrr"]["mean"], 3))
nm = st.get("main|wikipedia|nomem|rand|bq200")
if nm:
    put("MemoryWorth", f(100 * (g["methods"]["oracle"]["mrr"]["mean"] - nm["methods"]["oracle"]["mrr"]["mean"]), 1))
# horizon ablation
for ds, n in (("wikipedia", "Wiki"), ("mooc", "Mooc")):
    h = st.get(f"horizon|{ds}|attn|rand|bq200")
    if h:
        for H, w in (("5", "Five"), ("10", "Ten"), ("20", "Twenty"), ("50", "Fifty")):
            m = h["methods"][f"exact@{H}"]
            put(f"{n}Hor{w}Rel", sg(100 * (m["mrr"]["mean"] - h["methods"]["exact"]["mrr"]["mean"]) / h["methods"]["exact"]["mrr"]["mean"], 2))
            put(f"{n}Hor{w}Cost", f(m["cost_x_normal"]["mean"], 1)); put(f"{n}Hor{w}MB", f(m["journal_MB"]["mean"], 0))
            put(f"{n}Hor{w}Fallback", f"{int(m['fallback']['mean']):,}")
        put(f"{n}HorFullMB", f(h["methods"]["exact"]["journal_MB"]["mean"], 0))

# ---------------------------------------------------------------- SEC
g = st.get("main_amend|sec|attn|real|bq200")
if g:
    m = g["methods"]
    put("SecLate", f(100 * g["late_frac"]["mean"], 0)); put("SecDays", str(g["J"])); put("SecQueries", f"{g['queries']:,}")
    put("SecAmend", str(g["amendments"])); put("SecInTransit", f(100 * g["in_transit_frac"]["mean"], 0))
    for meth, key in (("oracle", "Oracle"), ("exact", "Exact"), ("eps:0.05", "Eps"), ("khop:1", "KhopOne"), ("arrival", "Arrival"),
                      ("arrival_logfix", "LogFix"), ("drop", "Drop"), ("buffer:24", "BufDay"), ("buffer:96", "BufFourDays"), ("replay", "Replay")):
        if meth in m:
            put(f"Sec{key}", pm(m[meth]["mrr"])); put(f"Sec{key}Mean", f(m[meth]["mrr"]["mean"], 2)); put(f"Sec{key}Cost", f(m[meth]["cost_x_normal"]["mean"], 2 if m[meth]["cost_x_normal"]["mean"] < 10 else 0))
            if m[meth]["dmrr_rel_pct"]:
                put(f"Sec{key}Rel", pm(m[meth]["dmrr_rel_pct"], 2))
    put("SecFreshGap", f(100 * (m["oracle"]["mrr"]["mean"] - m["exact"]["mrr"]["mean"]), 1))
    put("SecFreshGapRel", f(100 * (m["oracle"]["mrr"]["mean"] - m["exact"]["mrr"]["mean"]) / m["exact"]["mrr"]["mean"], 1))
c = S["contraction"].get("sec|attn")
if c:
    put("SecJown", f(c["J_own"]["mean"], 2)); put("SecJsum", f(c["sum_median"]["mean"], 2))

# ---------------------------------------------------------------- exactness
ex = S["exactness"]
tot = 0
for ds, n in DSN.items():
    e = ex.get(f"exactness_tgn_{ds}_s0")
    if not e:
        continue
    s = e["summary"]; tot += s["trials"]
    put(f"Ex{n}Trials", f"{s['trials']:,}"); put(f"Ex{n}Max", sci(s['max_abs_diff']))
    put(f"Ex{n}Zero", f(100 * s["frac_exact_zero"], 1)); put(f"Ex{n}Below", f(100 * s["frac_below_1e5"], 0))
    put(f"Ex{n}ConeMed", f(100 * e["cone_frac_median"], 2)); put(f"Ex{n}ConePNinety", f(100 * e["cone_frac_p90"], 1))
    put(f"Ex{n}ConeZero", f(100 * e["cone_zero_frac"], 0)); put(f"Ex{n}Speedup", f(e["speedup_median"], 1))
    put(f"Ex{n}Window", f"{s['window_events']:,}")
    if "pyg_reference" in s:
        p = s["pyg_reference"]
        put("ExPygN", str(p["n"])); put("ExPygTies", str(p["n_with_ties"])); put("ExPygNoTiesMax", sci(p['max_abs_diff_without_ties']))
put("ExTotal", f"{tot:,}")
e64 = ex.get("exactness_tgn_wikipedia_s0_f64")
if e64:
    put("ExDoubleTrials", f"{e64['summary']['trials']:,}"); put("ExDoubleMax", sci(e64['summary']['max_abs_diff']))

# ---------------------------------------------------------------- deletion
de = S["deletion"]
for key, n in (("main|wikipedia|attn", "Wiki"), ("main|wikipedia|id", "WikiMemOnly"), ("main|reddit|attn", "Reddit"),
               ("main|mooc|attn", "Mooc"), ("main|wikipedia|rho0.75_lam10", "WikiRhoSevenFive"),
               ("main|wikipedia|rho0.95_lam10", "WikiRhoNineFive"), ("main|wikipedia|nomem", "WikiNoMem")):
    d = de.get(key)
    if not d:
        continue
    m = d["methods"]
    put(f"Del{n}Pairs", f"{int(np.mean(d['pairs_A'])):,}")
    for meth, k2 in (("none", "None"), ("store", "Store"), ("reset", "Reset"), ("restart", "Restart"), ("restart_cone", "RestartCone"),
                     ("exact", "Exact"), ("eps:0.05", "Eps"), ("eps:0.2", "EpsBig"), ("khop:1", "KhopOne"), ("replay", "Replay")):
        if meth not in m:
            continue
        a = m[meth]["attacks"]
        put(f"Del{n}{k2}Logit", pm(a["logit_0"])); put(f"Del{n}{k2}LogitMean", f(a["logit_0"]["mean"], 3))
        if "probe_0" in a:
            put(f"Del{n}{k2}Probe", pm(a["probe_0"])); put(f"Del{n}{k2}ProbeMean", f(a["probe_0"]["mean"], 3))
        if "rank_0" in a:
            put(f"Del{n}{k2}Rank", pm(a["rank_0"]))
        if "logit_50" in a:
            put(f"Del{n}{k2}LogitLate", f(a["logit_50"]["mean"], 3))
        if "probe_50" in a:
            put(f"Del{n}{k2}ProbeLate", f(a["probe_50"]["mean"], 3))
        put(f"Del{n}{k2}Gap", pm(m[meth]["logit_gap"]))
        put(f"Del{n}{k2}Cost", f(m[meth]["cost_x_normal"]["mean"], 2)); put(f"Del{n}{k2}MRR", f(m[meth]["mrr_after"]["mean"], 3))
        put(f"Del{n}{k2}Resid", f(m[meth]["resid_cone"]["mean"], 3))
    if "reset" in m:
        put(f"Del{n}ResetMRRLoss", f(100 * (m["exact"]["mrr_after"]["mean"] - m["reset"]["mrr_after"]["mean"]), 1))
    put(f"Del{n}Cone", f"{int(d['cone_nodes']['mean']):,}"); put(f"Del{n}Endpoints", f"{int(d['endpoints']['mean']):,}")
p = de.get("pairs|wikipedia|attn")
if p:
    put("DelWikiPairSeedsGap", ", ".join(f"{sg(x, 3)}" for x in p["methods"]["store"]["logit_gap"]["values"]))

# ---------------------------------------------------------------- contraction
co = S["contraction"]
for var, n in (("attn", "Vanilla"), ("rho0.95_lam10", "RhoNineFive"), ("rho0.85_lam10", "RhoEightFive"), ("rho0.75_lam10", "RhoSevenFive"),
               ("id", "MemOnly")):
    c = co.get(f"wikipedia|{var}")
    if not c:
        continue
    put(f"Con{n}Cert", pm(c["certified"], 1, 100)); put(f"Con{n}CertMean", f(100 * c["certified"]["mean"], 1))
    put(f"Con{n}Jown", f(c["J_own"]["mean"], 2)); put(f"Con{n}Jpartner", f(c["J_partner"]["mean"], 2))
    put(f"Con{n}SumMed", f(c["sum_median"]["mean"], 2)); put(f"Con{n}SumMax", f(max(c["sum_max"]["values"]), 2))
    put(f"Con{n}Updates", f"{c['updates'][0]:,}")
    for eps, w in (("0.01", "Small"), ("0.05", ""), ("0.2", "Big")):
        ce = c["certificate"][eps]
        if ce["ratio"]:
            put(f"Con{n}Ratio{w}", f(ce["ratio"]["mean"], 1)); put(f"Con{n}Holds{w}", f(100 * ce["holds"]["mean"], 0))
if co.get("wikipedia|rho0.75_lam10"):
    put("ConRhoSevenFiveAmplify", f(100 - 100 * co["wikipedia|rho0.75_lam10"]["certified"]["mean"], 1))
for ds, n in (("mooc", "Mooc"), ("reddit", "Reddit")):
    c = co.get(f"{ds}|attn")
    if c:
        put(f"Con{n}Jown", f(c["J_own"]["mean"], 2)); put(f"Con{n}CertMean", f(100 * c["certified"]["mean"], 1))

# ---------------------------------------------------------------- extra derived numbers used in the text
import glob as _glob                                      # noqa: E402
ev = {}
for fpath in _glob.glob(str(paths.RESULTS / "restarter" / "*.json")):
    d_ = json.load(open(fpath)); ev[d_["model"]] = d_["explained_variance"]
for pre, n in (("tgn_wikipedia_s", "Wiki"), ("tgn_reddit_s", "Reddit")):
    vals = [v for k, v in ev.items() if k.startswith(pre) and k[len(pre):].isdigit()]
    if vals:
        put(f"RestarterEV{n}", f(float(np.mean(vals)), 2))
van = st["main|wikipedia|attn|rand|bq200"]["methods"]["oracle"]["mrr"]["mean"]
for var, n in (("rho0.95_lam10", "RhoNineFive"), ("rho0.85_lam10", "RhoEightFive"), ("rho0.75_lam10", "RhoSevenFive")):
    g_ = st.get(f"main|wikipedia|{var}|rand|bq200")
    if g_:
        put(f"Con{n}Loss", f(100 * (van - g_["methods"]["oracle"]["mrr"]["mean"]), 1))
        put(f"Con{n}Gain", f(100 * (g_["methods"]["oracle"]["mrr"]["mean"] - van), 1))
mo = st["main|mooc|attn|rand|bq200"]["methods"]["oracle"]["mrr"]["values"]
put("MoocSeedMin", f(min(mo), 2)); put("MoocSeedMax", f(max(mo), 2))
dis = []
for ds in DSN:
    m_ = st[f"main|{ds}|attn|rand|bq200"]["methods"]
    dis += [100 * (1 - m_["arrival"]["top1_agree"]["mean"]), 100 * (1 - m_["drop"]["top1_agree"]["mean"])]
put("ArrDropDisMin", f(min(dis), 0)); put("ArrDropDisMax", f(max(dis), 0))
g_ = st.get("main_amend|sec|attn|real|bq200")
if g_:
    m_ = g_["methods"]
    a_, b_ = m_["oracle"]["strata"]["in_transit"]["mean"], m_["exact"]["strata"]["in_transit"]["mean"]
    put("SecOracleTransitRel", f(100 * (a_ - b_) / b_, 0))
for ds, n in DSN.items():
    m_ = st[f"main|{ds}|attn|rand|bq200"]["methods"]
    put(f"{n}ExactHist", pm(m_["exact"]["mrr_hist"])); put(f"{n}ArrivalHistRel", pm(m_["arrival"]["dmrr_vs_exact_hist"], 3, 100 / m_["exact"]["mrr_hist"]["mean"]))
    put(f"{n}BufOneHistRel", pm(m_["buffer:1"]["dmrr_vs_exact_hist"], 2, 100 / m_["exact"]["mrr_hist"]["mean"]))
npd = ex.get("exactness_tgn_wikipedia_s0_nopad")
if npd:
    s_ = npd["summary"]
    put("ExNoPadTrials", f"{s_['trials']:,}"); put("ExNoPadFail", f(100 * (1 - s_["frac_below_1e5"]), 0))
    put("ExNoPadMax", sci(s_['max_abs_diff'], 0))
dl = de.get("main|wikipedia|attn")
if dl:
    put("DelWikiStoreGapMean", f(dl["methods"]["store"]["logit_gap"]["mean"], 3))

for ds, n in DSN.items():                                   # single edits that change no memory at all
    fpath = paths.RESULTS / "exactness" / f"exactness_tgn_{ds}_s0.json"
    if fpath.exists():
        tr_ = [t for t in json.load(open(fpath))["trials"] if t["kind"] != "mixed"]
        put(f"Ex{n}NoEffect", f(100 * float(np.mean([t["recomputed"] == 0 for t in tr_])), 0))

# ---------------------------------------------------------------- data sizes (Table 1)
from lategraph.data import load as load_stream          # noqa: E402
for ds, n in list(DSN.items()) + [("sec", "Sec")]:
    S_ = load_stream(ds)
    put(f"N{n}Events", f"{S_.num_events:,}"); put(f"N{n}Test", f"{S_.num_events - S_.test0:,}")
    put(f"N{n}Src", f"{(S_.dst_min if ds != 'sec' else S_.extra['n_insiders']):,}"); put(f"N{n}Dst", f"{S_.dst_max - S_.dst_min + 1:,}")
    put(f"N{n}Days", f"{float((S_.t[-1] - S_.t[0]) / 86400):.0f}")
    if ds == "sec":
        lag = (S_.arrival[S_.test0:] - S_.t[S_.test0:]).double() / 86400
        put("SecLagMed", f"{float(lag.median()):.1f}"); put("SecLagNinety", f"{float(lag.quantile(0.9)):.1f}")
        put("SecLagNinetyNine", f"{float(lag.quantile(0.99)):.0f}"); put("SecLagFour", f"{100 * float((lag > 4).double().mean()):.0f}")
        put("NSecTrain", f"{S_.val0:,}"); put("NSecVal", f"{S_.test0 - S_.val0:,}")

# ---------------------------------------------------------------- appendix tables (written as LaTeX rows)
GEN = OUT.parent / "generated"; GEN.mkdir(exist_ok=True)
rows = []
DELAY_NAMES = {"rand": "random", "active": "active users", "burst": "outage bursts", "offline": "offline users", "itemshard": "slow item shard"}
for ds, name in (("wikipedia", "Wikipedia"), ("reddit", "Reddit"), ("mooc", "MOOC")):
    for delay in ("rand", "active", "burst", "offline", "itemshard"):
        for bq in ("bq200", "bq50"):
            g = st.get(f"main|{ds}|attn|{delay}|{bq}")
            if not g:
                continue
            m = g["methods"]
            r = lambda meth: pm(m[meth]["dmrr_rel_pct"], 2) if m[meth]["dmrr_rel_pct"]["n"] > 1 else sg(m[meth]["dmrr_rel_pct"]["mean"], 2)
            cst = lambda meth: (f(m[meth]["cost_x_normal"]["mean"], 1) if m[meth]["cost_x_normal"]["mean"] < 10 else f(m[meth]["cost_x_normal"]["mean"], 0))
            rows.append(f"{name} & {DELAY_NAMES[delay]} & {bq[2:]} & {g['n']} & {f(100 * g['late_frac']['mean'], 1)} & {r('arrival')} & {r('drop')} & {r('buffer:1')} & {r('buffer:4')} & {r('eps:0.05')} & {cst('replay')} & {cst('exact')} & {cst('eps:0.05')} & {cst('khop:1')} \\\\")
(GEN / "table_delays.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
rows = []
for ds, name in (("wikipedia", "Wikipedia"), ("mooc", "MOOC")):
    h = st.get(f"horizon|{ds}|attn|rand|bq200")
    if not h:
        continue
    ex = h["methods"]["exact"]
    for meth, lab in (("exact", "full journal"), ("exact@50", "50 steps"), ("exact@20", "20 steps"), ("exact@10", "10 steps"), ("exact@5", "5 steps"),
                      ("eps:0.05@20", "$\\varepsilon$=0.05, 20 steps")):
        m = h["methods"][meth]
        rows.append(f"{name} & {lab} & {f(m['journal_MB']['mean'], 1)} & {int(m['fallback']['mean']):,} & {f(m['cost_x_normal']['mean'], 1)} & "
                    f"{f(100 * m['top1_agree']['mean'], 1)} & {sg(100 * (m['mrr']['mean'] - ex['mrr']['mean']) / ex['mrr']['mean'], 2)} \\\\")
(GEN / "table_horizon.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
rows = []
for ds, name in (("wikipedia", "Wikipedia"), ("reddit", "Reddit"), ("mooc", "MOOC")):
    g = st.get(f"main|{ds}|attn|rand|bq200")
    m = g["methods"]
    def strat(meth, key):
        return m[meth]["strata"][key]["mean"]
    ex_it, ex_nt = strat("exact", "in_transit"), strat("exact", "not_in_transit")
    rel = lambda meth, key, base: sg(100 * (strat(meth, key) - base) / base, 1)
    rows.append(f"{name} & {f(100 * g['in_transit_frac']['mean'], 0)} & {f(ex_it, 3)} & {rel('oracle', 'in_transit', ex_it)} & {rel('arrival', 'in_transit', ex_it)} & {rel('buffer:1', 'in_transit', ex_it)} & "
                f"{f(ex_nt, 3)} & {rel('oracle', 'not_in_transit', ex_nt)} & {rel('arrival', 'not_in_transit', ex_nt)} & {rel('buffer:1', 'not_in_transit', ex_nt)} \\\\")
g = st.get("main_amend|sec|attn|real|bq200")
if g:
    m = g["methods"]
    ex_it, ex_nt = m["exact"]["strata"]["in_transit"]["mean"], m["exact"]["strata"]["not_in_transit"]["mean"]
    rel2 = lambda meth, key, base: sg(100 * (m[meth]["strata"][key]["mean"] - base) / base, 1)
    rows.append(f"SEC Form 4 & {f(100 * g['in_transit_frac']['mean'], 0)} & {f(ex_it, 3)} & {rel2('oracle', 'in_transit', ex_it)} & {rel2('arrival', 'in_transit', ex_it)} & {rel2('buffer:96', 'in_transit', ex_it)}$^\\dagger$ & "
                f"{f(ex_nt, 3)} & {rel2('oracle', 'not_in_transit', ex_nt)} & {rel2('arrival', 'not_in_transit', ex_nt)} & {rel2('buffer:96', 'not_in_transit', ex_nt)}$^\\dagger$ \\\\")
(GEN / "table_strata.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

# ---------------------------------------------------------------- runtimes and job counts (Appendix C)
_tr = [json.load(open(x))["wall_sec"] / 60 for x in _glob.glob(str(paths.RESULTS / "training" / "*.json"))]
_st = [json.load(open(x))["wall_sec"] / 60 for x in _glob.glob(str(paths.RESULTS / "stream" / "*.json")) if "debug" not in x]
put("NModels", str(len(_tr))); put("TrainMinLo", str(int(np.floor(min(_tr))))); put("TrainMinHi", str(int(np.ceil(max(_tr)))))
put("StreamMinHi", str(int(np.ceil(max(_st)))))
_jobs = [l for j in ("main_jobs.txt", "sec_jobs.txt") for l in open(paths.REPO / "scripts" / "jobs" / j) if l.strip() and not l.startswith("#")]
put("NJobs", str(len(_jobs)))

OUT.write_text("% generated by scripts/make_numbers.py from results/ -- do not edit by hand\n" + "\n".join(lines) + "\n")
print(f"wrote {len(lines)} macros to {OUT}")
