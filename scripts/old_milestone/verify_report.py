"""Independent check of the milestone report.

make_numbers.py writes report/numbers.tex from results/summary.json. This script does not use summary.json: it
recomputes the headline numbers straight from the per-run JSON files in results/ and compares them with the macros
the report prints. It also checks two claims that are not single numbers ("at chance", "held every time") and looks
for unresolved references in the compiled PDF.

  python scripts/verify_report.py            # exits with status 1 if any check fails
"""
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph import paths                     # noqa: E402

R = paths.RESULTS
REPORT = paths.REPORT
MACROS = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", (REPORT / "numbers.tex").read_text()))
DSN = {"wikipedia": "Wiki", "reddit": "Reddit", "mooc": "Mooc"}
fails = []


def num(macro):
    """First number in a macro value: '$-$1.01\\,$\\pm$\\,0.26' -> -1.01, '$5.4{\\times}10^{-7}$' -> 5.4e-7."""
    s = MACROS[macro].replace("$-$", "-").replace(",", "")
    m = re.search(r"(-?\d+(?:\.\d+)?)(?:\{\\times\}10\^\{(-?\d+)\})?", s)
    return float(m.group(1)) * (10 ** int(m.group(2)) if m.group(2) else 1)


def decimals(macro):
    s = MACROS[macro].replace(",", "")
    m = re.search(r"\d+(?:\.(\d+))?", s)
    return len(m.group(1)) if m and m.group(1) else 0


def check(macro, value, what):
    """Pass if the recomputed value rounds to the printed one (half a unit of the last printed digit)."""
    printed = num(macro)
    if "times}10" in MACROS[macro]:
        ok = abs(value - printed) <= 0.051 * 10 ** np.floor(np.log10(abs(printed)))
    else:
        ok = abs(value - printed) <= 0.5 * 10 ** -decimals(macro) + 1e-9
    print(f"  {'ok  ' if ok else 'FAIL'} {what:<62s} report {MACROS[macro]:<28s} recomputed {value:.6g}")
    if not ok:
        fails.append(what)


def claim(ok, what, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {what:<62s} {detail}")
    if not ok:
        fails.append(what)


def load(pattern):
    return [json.load(open(p)) for p in sorted(glob.glob(str(R / pattern)))]


# --------------------------------------------------------------------------------------------------- training
print("Training (best validation AP over 3 seeds)")
for ds, n in list(DSN.items()) + [("sec", "Sec")]:
    ap = [d["best_val_ap"] for d in load(f"training/tgn_{ds}_s[0-9].json")]
    check(f"ValAP{n}", float(np.mean(ap)), f"{ds}: mean val AP ({len(ap)} seeds)")

# --------------------------------------------------------------------------------------------------- exactness
print("Exact repair vs. sequential recomputation")
total = 0
for ds, n in DSN.items():
    tr = json.load(open(R / "exactness" / f"exactness_tgn_{ds}_s0.json"))["trials"]
    diff = np.array([t["max_abs_diff"] for t in tr]); total += len(tr)
    check(f"Ex{n}Trials", len(tr), f"{ds}: trials")
    check(f"Ex{n}Max", diff.max(), f"{ds}: largest |repaired - reference|")
    check(f"Ex{n}Zero", 100 * np.mean(diff == 0), f"{ds}: bit-identical (%)")
    single = [t for t in tr if t["kind"] != "mixed"]
    check(f"Ex{n}NoEffect", 100 * np.mean([t["recomputed"] == 0 for t in single]), f"{ds}: single edits that change nothing (%)")
    claim(diff.max() < 1e-5, f"{ds}: every trial within 1e-5")
check("ExTotal", total, "all datasets: trials")
f64 = json.load(open(R / "exactness" / "exactness_tgn_wikipedia_s0_f64.json"))["trials"]
claim(all(t["max_abs_diff"] == 0 for t in f64), "float64: every trial bit-identical", f"({len(f64)} trials)")
check("ExDoubleTrials", len(f64), "float64: trials")
nopad = json.load(open(R / "exactness" / "exactness_tgn_wikipedia_s0_nopad.json"))["trials"]
check("ExNoPadFail", 100 * np.mean([t["max_abs_diff"] > 1e-5 for t in nopad]), "without padding: trials off by > 1e-5 (%)")

# --------------------------------------------------------------------------------------------------- streaming
print("Late events (every JODIE delay setting, standard TGN)")
runs = []
for p in sorted(glob.glob(str(R / "stream" / "tgn_*_s[0-9]_*_bq*_d[0-9].json"))):
    m = re.match(r"tgn_(wikipedia|reddit|mooc)_s(\d)_([a-z]+)_bq(\d+)_d\d\.json", Path(p).name)
    if m:
        runs.append((m.group(1), m.group(3), int(m.group(4)), json.load(open(p))["methods"]))
check("NRuns", len(runs), "runs")
check("NSettings", len({(d, k, b) for d, k, b, _ in runs}), "settings (dataset, delay model, step size)")
arr = [abs(m["arrival"]["mrr"] - m["exact"]["mrr"]) for *_, m in runs]
check("ArrivalMaxAbsMRR", max(arr), "arrival order: largest |MRR - exact MRR|")
check("ArrivalMedAbs", float(np.median([100 * a / m["exact"]["mrr"] for a, (*_, m) in zip(arr, runs)])),
      "arrival order: median |relative MRR change| (%)")
check("EpsMaxAbs", max(abs(m["eps:0.05"]["mrr"] - m["exact"]["mrr"]) for *_, m in runs), "eps=0.05: largest |MRR - exact MRR|")
for ds, n in DSN.items():
    b1 = [100 * (m["buffer:1"]["mrr"] - m["exact"]["mrr"]) / m["exact"]["mrr"] for d, *_, m in runs if d == ds]
    check(f"{n}BufOneLo", -max(b1), f"{ds}: smallest loss of a 1-hour buffer (%)")
    check(f"{n}BufOneHi", -min(b1), f"{ds}: largest loss of a 1-hour buffer (%)")
print("Late events (Table 3: random delays, steps of 200, 3 seeds)")
for ds, n in DSN.items():
    ms = [m for d, k, b, m in runs if d == ds and k == "rand" and b == 200]
    check(f"{n}Exact", float(np.mean([m["exact"]["mrr"] for m in ms])), f"{ds}: exact-repair MRR ({len(ms)} seeds)")
    for meth, key in (("replay", "Replay"), ("exact", "Exact"), ("eps:0.05", "Eps"), ("khop:1", "KhopOne")):
        check(f"{n}{key}Cost", float(np.mean([m[meth]["cost_x_normal"] for m in ms])), f"{ds}: compute of {meth} (x normal)")
    check(f"{n}ArrivalAgree", 100 * float(np.mean([m["arrival"]["top1_agree"] for m in ms])), f"{ds}: arrival top-1 agreement (%)")
    J = [json.load(open(p))["info"]["J"] for p in sorted(glob.glob(str(R / "stream" / f"tgn_{ds}_s[0-9]_rand_bq200_d[0-9].json")))]
    for meth, key in (("replay", "Replay"), ("exact", "Exact"), ("arrival", "Arrival")):
        check(f"{n}{key}Ms", float(np.mean([1000 * m[meth]["sec"] / j for m, j in zip(ms, J)])), f"{ds}: time per step of {meth} (ms)")

# --------------------------------------------------------------------------------------------------- deletion
print("Deletion (membership AUC at the request step)")
for ds, n in (("wikipedia", "Wiki"), ("reddit", "Reddit")):
    D = load(f"deletion/deletion_tgn_{ds}_s[0-9]_p[0-9].json")
    gap = [d["methods"]["store"]["attacks"]["0"]["logit"] - d["methods"]["exact"]["attacks"]["0"]["logit"] for d in D]
    check(f"Del{n}StoreGap", float(np.mean(gap)), f"{ds}: log-only minus exact link-score AUC ({len(D)} seeds)")
    claim(min(gap) > 0, f"{ds}: log-only deletion leaks in every seed", f"(gaps {', '.join(f'{g:+.3f}' for g in gap)})")
    worst = 0.0
    for d in D:
        for meth in ("exact", "eps:0.05"):
            a = d["methods"][meth]["attacks"]["0"]
            se = np.sqrt((a["nA"] + a["nB"] + 1) / (12 * a["nA"] * a["nB"]))          # AUC std. error with no signal
            worst = max(worst, max(abs(a[k] - 0.5) / se for k in ("logit", "rank", "probe")))
    claim(worst < 3, f"{ds}: exact and eps-repair at chance for all 3 attacks", f"(largest |AUC - 0.5| = {worst:.1f} std. errors)")
    same = max(abs(d["methods"][meth]["attacks"][h][k] - d["methods"]["replay"]["attacks"][h][k])
               for d in D for meth in ("exact", "eps:0.05") for h in ("0", "5", "20", "50")
               for k in ("logit", "rank", "probe") if k in d["methods"]["replay"]["attacks"][h])
    claim(same < 0.005, f"{ds}: exact and eps-repair AUCs equal full replay", f"(largest difference {same:.4f}, all horizons)")
W = load("deletion/deletion_tgn_wikipedia_s[0-9]_p[0-9].json")
check("DelWikiRestartProbeMean", float(np.mean([d["methods"]["restart"]["attacks"]["0"]["probe"] for d in W])), "wikipedia: learned-restart probe AUC")
check("DelWikiResetProbeMean", float(np.mean([d["methods"]["reset"]["attacks"]["0"]["probe"] for d in W])), "wikipedia: reset probe AUC")

# --------------------------------------------------------------------------------------------------- contraction
print("Contraction and the online error bound")
C = {Path(p).stem.replace("contraction_tgn_", ""): json.load(open(p)) for p in glob.glob(str(R / "contraction" / "*.json"))}
held = [c["certificate"][e]["frac_bound_holds"] for c in C.values() for e in c["certificate"]]
claim(min(held) == 1.0, "online bound holds in every trial (all models, all eps)", f"({len(C)} models x 3 eps)")
for var, n in (("rho0.85_lam10", "RhoEightFive"), ("rho0.75_lam10", "RhoSevenFive"), ("rho0.95_lam10", "RhoNineFive")):
    cs = [C[f"wikipedia_s{s}_{var}"] for s in range(3)]
    check(f"Con{n}CertMean", 100 * float(np.mean([c["jacobian"]["certified_frac"] for c in cs])), f"wikipedia {var}: contractive updates (%)")
    check(f"Con{n}Ratio", float(np.mean([c["certificate"]["0.05"]["median_bound_over_err"] for c in cs])),
          f"wikipedia {var}: median bound / true error (eps=0.05)")
cs = [C[f"wikipedia_s{s}"] for s in range(3)]
check("ConVanillaCertMean", 100 * float(np.mean([c["jacobian"]["certified_frac"] for c in cs])), "wikipedia standard TGN: contractive updates (%)")

# --------------------------------------------------------------------------------------------------- SEC
print("SEC Form 4")
S = load("stream/tgn_sec_s[0-9]_real_bq200_d[0-9].json")
check("SecLate", 100 * float(np.mean([s["info"]["late_frac"] for s in S])), f"late for their daily step (%) ({len(S)} seeds)")
check("SecAmend", S[0]["info"]["amendments"], "amendments in the test period")
check("SecExactCost", float(np.mean([s["methods"]["exact"]["cost_x_normal"] for s in S])), "exact repair compute (x normal)")
check("SecReplayCost", float(np.mean([s["methods"]["replay"]["cost_x_normal"] for s in S])), "full replay compute (x normal)")
rel = [100 * (s["methods"]["oracle"]["mrr"] - s["methods"]["exact"]["mrr"]) / s["methods"]["exact"]["mrr"] for s in S]
check("SecOracleRel", float(np.mean(rel)), "hindsight oracle vs. exact repair (%)")

# --------------------------------------------------------------------------------------------------- the PDF
print("Compiled report")
pdf = REPORT / "main.pdf"
txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout if pdf.exists() else ""
claim(bool(txt), "report/main.pdf exists and has text")
claim("??" not in txt, "no unresolved references or citations")
if "YOUR-ACCOUNT" in txt:
    print("  note the code link is still the placeholder; set \\CodeURL in report/main.tex")

print(f"\n{'ALL CHECKS PASSED' if not fails else f'{len(fails)} CHECK(S) FAILED: ' + '; '.join(fails)}")
sys.exit(1 if fails else 0)
