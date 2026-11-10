"""LaTeX tables of the report (report/generated/*.tex) and of the paper (../paper/generated/*.tex, when it exists),
from results/summary.json and results/bench/.

  python scripts/make_tables_final.py
Missing entries are written as '--'.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
S = json.load(open(R / "summary.json"))
OUTS = [REPO / "report" / "generated"] + ([REPO.parent / "paper" / "generated"] if (REPO.parent / "paper").exists() else [])
for _o in OUTS:
    _o.mkdir(parents=True, exist_ok=True)
DS = [("wikipedia", "Wikipedia"), ("reddit", "Reddit"), ("mooc", "MOOC"), ("lastfm", "LastFM")]


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def f(x, fmt="{:.3f}", scale=1.0, pm=True):
    if x is None:
        return "--"
    if isinstance(x, dict) and "mean" in x:
        s = fmt.format(scale * x["mean"])
        if pm and x.get("n", 1) > 1:
            s += r"\,{\scriptsize$\pm$" + fmt.replace("{:+", "{:").format(scale * x["std"]) + "}"
        return s
    return fmt.format(scale * x)


def write(name, s):
    for o in OUTS:
        (o / f"{name}.tex").write_text(s)
    print("wrote", name)


def err(x):
    """A largest difference: 0 when the repaired memory is bit-identical in every trial."""
    return "0" if x == 0 else f"{x:.1e}"


def bench(d):
    p = R / "bench" / f"bench_{d}.json"
    return json.load(open(p)) if p.exists() else {}


# ------------------------------------------------------------------------------------------------ exactness and speed
def tab_exact():
    E = S.get("exactness", {})
    rows = []

    def cones(e):
        if e is None or "single_no_effect" not in e:
            return "-- & -- & --"
        p90 = 100 * e["single_cone_p90"]
        return (f"{100 * e['single_no_effect']:.0f}\\% & {100 * e['single_cone_median_eff']:.2f}\\% & "
                f"{p90:.2f}\\%" if p90 < 1 else f"{100 * e['single_no_effect']:.0f}\\% & {100 * e['single_cone_median_eff']:.2f}\\% & {p90:.1f}\\%")
    for d, lab in DS + [("tgbl-review", "tgbl-review")]:
        e = E.get(f"exactness_tgn_{d}_s0"); b = bench(d)
        if e is None:
            rows.append(f"{lab} (TGN) & -- & -- & -- & -- & -- & -- & -- \\\\"); continue
        s = e["summary"]
        rows.append(f"{lab} (TGN) & {s['trials']:,} & {err(s['max_abs_diff'])} & {100 * s['frac_exact_zero']:.0f}\\% & {cones(e)} & "
                    + (f"{b['single']['speedup_vs_replay_effective']:.1f}$\\times$ ({b['single']['walk_speedup_vs_replay_effective']:.1f}$\\times$)"
                       if b and "speedup_vs_replay_effective" in b["single"] else "--") + " \\\\")
    for key, lab in (("exactness_tgn_wikipedia_s0_jodie", "Wikipedia (JODIE)"), ("exactness_tgn_wikipedia_s0_crnn0.9", "Wikipedia (contractive)")):
        e = E.get(key)
        if e:
            s = e["summary"]
            rows.append(f"{lab} & {s['trials']:,} & {err(s['max_abs_diff'])} & {100 * s['frac_exact_zero']:.0f}\\% & {cones(e)} & -- \\\\")
    dy = g(S, "dyrep", "wikipedia")
    if dy:
        rows.append(f"Wikipedia (DyRep) & {300 * dy['n']:,} & {dy['max_abs_diff']:.1e} & -- & {f(dy['cone_zero_frac'], '{:.0f}', 100, False)}\\%$^\\ast$ & "
                    f"{f(dy['cone_frac_median'], '{:.2f}', 100, False)}\\%$^\\ast$ & -- & -- \\\\")
    e64 = E.get("exactness_tgn_wikipedia_s0_f64")
    if e64:
        rows.append(f"Wikipedia (TGN, float64) & {e64['summary']['trials']:,} & {e64['summary']['max_abs_diff']:.0f} & 100\\% & -- & -- & -- & -- \\\\")
    write("tab_exact", r"""\begin{table}[t]
\centering\small
\caption{Exact repair on random edits of a 100-step test window (insertions, deletions and amendments; 10\% of the trials edit
2--10 events at once). Max.\ error: largest $|$repaired $-$ reference$|$ over all memory entries, where the reference is a
sequential recomputation of the corrected stream (float32 unless noted). Single edits: share that changes no memory, and
for the others the cone as a share of all updates since the edit (median and 90th percentile; $^\ast$DyRep: median over all
trials). Speed-up: median, over single edits that change a memory, of the time of rollback + replay over the time of exact
repair with the step index (in brackets: walking every step), on an idle laptop CPU; edits that change nothing are
recognised at their own step.}
\label{tab:exact}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
 & & & & \multicolumn{3}{c}{single edits} & \\
\cmidrule(lr){5-7}
 & trials & max.\ error & bit-identical & no change & cone & cone p90 & speed-up \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ------------------------------------------------------------------------------------------------ training-period deletions
def tab_train_del():
    T = S.get("train_deletion", {})
    cols = [(k, lab) for k, lab in (("wikipedia", "Wikipedia"), ("reddit", "Reddit"), ("mooc", "MOOC"), ("wikipedia|recent", "Wikipedia, recent"))]
    meths = [("none", "no deletion"), ("store", "log-only"), ("memory", "memory repair"), ("finetune", "memory repair + fine-tune"), ("retrain", "retrain (reference)")]
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab} ({g(T, k, 'n') or 0})}}" for k, lab in cols)
    rows = []
    for m, lab in meths:
        cells = []
        for k, _ in cols:
            M = g(T, k, "methods", m)
            cells += [f(g(M, "logit"), "{:.2f}"), f(g(M, "probe"), "{:.2f}")]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    cost = []
    for k, _ in cols:
        x = g(T, k)
        ba = R / "bench_age" / f"bench_age_{k.split('|')[0]}.json"
        hist = f"{json.load(open(ba))['history_pass_sec']:.2f} s" if ba.exists() else "--"
        cost += [f"\\multicolumn{{2}}{{c}}{{{f(x['single_sec'], '{:.2f}', 1, False)} s / {hist} / {f(x['train_minutes']['retrain'], '{:.0f}', 1, False)} min}}" if x else "\\multicolumn{2}{c}{--}"]
    write("tab_train_del", r"""\begin{table}[t]
\centering\scriptsize
\caption{Deleting training-period pairs (membership AUC of the link score / the memory probe at the deployment start; mean
$\pm$ std over seeds, number of seeds in brackets). ``Recent'': pairs from the last 20\% of the training period. Last row:
time of one exact memory repair for a single pair / of recomputing the memory over the whole history with fixed weights
(one batched pass, idle machine) / of retraining one model.}
\label{tab:traindel}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}l""" + "cc" * len(cols) + r"""@{}}
\toprule
 & """ + head + r""" \\
 & """ + " & ".join(["score", "probe"] * len(cols)) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
cost: repair / recompute / retrain & """ + " & ".join(cost) + r""" \\
\bottomrule
\end{tabular}
\end{table}
""")


# ------------------------------------------------------------------------------------------------ certificates
def tab_cert():
    C, St, Tr = S.get("contraction", {}), S.get("streaming", {}), S.get("training", {})
    rows = []
    for var, lab in (("attn", "standard TGN (GRU)"), ("rho0.95_lam10", "penalty, $\\rho=0.95$"), ("rho0.85_lam10", "penalty, $\\rho=0.85$"),
                     ("rho0.75_lam10", "penalty, $\\rho=0.75$"), ("rho0.85_lam1", "penalty, $\\rho=0.85$, $\\lambda=1$"),
                     ("rho0.85_lam100", "penalty, $\\rho=0.85$, $\\lambda=100$"), ("crnn0.95", "contractive cell, $\\rho\\le0.95$"),
                     ("crnn0.9", "contractive cell, $\\rho\\le0.9$"), ("crnn0.8", "contractive cell, $\\rho\\le0.8$"),
                     ("crnn0.95c0.7", "contractive cell, $\\rho\\le0.95$, $c=0.7$"), ("crnn0.99c0.9", "contractive cell, $\\rho\\le0.99$, $c=0.9$")):
        c = C.get(f"wikipedia|{var}"); s = g(St, f"main|wikipedia|{var}|rand|bq200", "methods", "oracle", "mrr")
        tr = g(Tr, f"wikipedia|{var}", "val_ap")
        crnn = var.startswith("crnn")
        cert = "100" if crnn and c else f(g(c, "certified"), "{:.1f}", 100)
        held = (f(g(c, "rigorous", "0.05", "holds"), "{:.0f}", 100, False) + " (proven)") if crnn else f(g(c, "certificate", "0.05", "holds"), "{:.0f}", 100, False)
        ratio = f(g(c, "rigorous", "0.05", "ratio"), "{:.1f}", 1, False) if crnn else f(g(c, "certificate", "0.05", "ratio"), "{:.1f}", 1, False)
        dc = f(g(c, "certified_repair", "0.02", "cost"), "{:.2f}", 1, False) if crnn else "--"
        rows.append(f"{lab} & {f(tr, '{:.3f}')} & {f(s, '{:.3f}')} & {cert} & {held} & {ratio} & {dc} \\\\")
    write("tab_cert", r"""\begin{table}[t]
\centering\scriptsize
\caption{Contraction and the error bound on Wikipedia ($\varepsilon=0.05$; 100 random deletions per model and seed). Certified:
updates with $\|J_{\text{own}}\|+\|J_{\text{partner}}\|<1$ (all updates for the contractive cell, which contracts by
construction). Bound held: share of trials in which the per-node bound held for every node (first order for the
GRU, proven for the cell). $\delta$-cert.: updates recomputed by $\delta$-certified repair ($\delta=0.02$) relative to exact repair.}
\label{tab:cert}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & val.\ AP & test MRR & certified (\%) & bound held (\%) & bound / error & $\delta$-cert.\ cost \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ------------------------------------------------------------------------------------------------ late events
def tab_late():
    St = S.get("streaming", {})
    DSL = DS + [("tgbl-review", "tgbl-review$^\\ddagger$"), ("sec", "SEC$^\\S$")]
    G = [g(St, "main_amend|sec|attn|real|bq200" if d == "sec" else f"main|{d}|attn|rand|bq200") for d, _ in DSL]
    day = {"buffer:1": "buffer:24", "buffer:4": "buffer:96"}          # daily steps: buffers of 1 and 4 days
    M = lambda i, m, k: g(G[i], "methods", day.get(m, m) if DSL[i][0] in ("tgbl-review", "sec") else m, k)
    lines = []

    def row(lab, fn):
        lines.append(lab + " & " + " & ".join(fn(i) for i in range(len(DSL))) + " \\\\")
    row("seeds", lambda i: str(G[i]["n"]) if G[i] else "--")
    row("test events that arrive late (\\%)", lambda i: f(g(G[i], "late_frac"), "{:.1f}", 100, False))
    row("MRR of exact repair", lambda i: f(M(i, "exact", "mrr"), "{:.3f}"))
    lines.append("\\midrule")
    lines.append("MRR vs.\\ exact repair (\\%) & " + " & " * (len(DSL) - 1) + " \\\\")
    for m, lab in (("oracle", "oracle (all on time)"), ("arrival", "arrival order"), ("drop", "drop late events"),
                   ("buffer:1", "buffer 1 h (1 day$^{\\ddagger\\S}$)"), ("buffer:4", "buffer 4 h (4 days)"), ("eps:0.05", "$\\varepsilon$-repair, $\\varepsilon=0.05$")):
        row(f"\\quad {lab}", lambda i, m=m: f(M(i, m, "dmrr_rel_pct"), "{:+.2f}"))
    lines.append("\\midrule")
    lines.append("compute ($\\times$ normal processing) & " + " & " * (len(DSL) - 1) + " \\\\")
    for m, lab in (("replay", "full replay"), ("exact", "exact repair (cone)"), ("khop:1", "1-hop rebuild"), ("eps:0.05", "$\\varepsilon$-repair")):
        row(f"\\quad {lab}", lambda i, m=m: f(M(i, m, "cost_x_normal"), "{:.2f}", 1, False))
    lines.append("\\midrule")
    lines.append("top-1 prediction as exact repair (\\%) & " + " & " * (len(DSL) - 1) + " \\\\")
    for m, lab in (("arrival", "arrival order"), ("eps:0.05", "$\\varepsilon$-repair"), ("khop:1", "1-hop rebuild")):
        row(f"\\quad {lab}", lambda i, m=m: f(M(i, m, "top1_agree"), "{:.1f}", 100, False))
    write("tab_late", r"""\begin{table}[t]
\centering\scriptsize
\caption{Late events (random delays), steps of 200 events, mean $\pm$ std over seeds. MRR changes are relative to exact repair
and paired within each seed; compute is in multiples of normal in-order processing. $^\ddagger$Daily steps (the timestamps
are days), delays with a median of 2 days, the first 200 test days with 300 queries per day; buffers of 1 and 4 days.
$^\S$Real arrival times and amendments, daily steps; buffers of 1 and 4 days.}
\label{tab:late}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}l""" + "c" * len(DSL) + r"""@{}}
\toprule
 & """ + " & ".join(lab for _, lab in DSL) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ------------------------------------------------------------------------------------------------ architectures
def tab_arch():
    St, Tr, E, L, Dy = S.get("streaming", {}), S.get("training", {}), S.get("exactness", {}), S.get("lira", {}), S.get("dyrep", {})
    rows = []
    for var, lab, ex in (("attn", "TGN (GRU memory, attention)", "exactness_tgn_wikipedia_s0"), ("id", "memory-only TGN", None),
                         ("jodie", "JODIE (RNN memory, time projection)", "exactness_tgn_wikipedia_s0_jodie"),
                         ("crnn0.9", "TGN with the contractive cell ($\\rho\\le0.9$)", "exactness_tgn_wikipedia_s0_crnn0.9"),
                         ("nomem", "memory-free control (attention only)", None)):
        mrr = g(St, f"main|wikipedia|{var}|rand|bq200", "methods", "oracle", "mrr")
        cone = f(g(E, ex, "cone_frac_median"), "{:.2f}", 100, False) + "\\%" if ex and g(E, ex) else "--"
        lira = f(g(L, f"wikipedia|{var}", "methods", "store", "logit_auc"), "{:.2f}")        # LiRA on the link score (query access)
        if var == "nomem":
            lira = "exact by design"
        rows.append(f"{lab} & {f(g(Tr, f'wikipedia|{var}', 'val_ap'), '{:.3f}')} & {f(mrr, '{:.3f}')} & {cone} & {lira} \\\\")
    dy = Dy.get("wikipedia")
    rows.insert(3, f"DyRep (RNN memory, reads the partner's neighbourhood) & {f(g(dy, 'val_ap'), '{:.3f}')} & {f(g(dy, 'mrr'), '{:.3f}')}$^\\dagger$ & "
                   f"{f(g(dy, 'cone_frac_median'), '{:.2f}', 100, False)}\\% & "
                   f"{f(g(dy, 'deletion', 'store', 'lira_logit'), '{:.2f}')} \\\\")
    write("tab_arch", r"""\begin{table}[!htb]
\centering\scriptsize
\caption{Memory models on Wikipedia (mean $\pm$ std over seeds). Test MRR: oracle policy of the streaming evaluation
($^\dagger$DyRep: in-order test MRR with 100 random negatives). Cone: median share of the updates since an edit that exact
repair recomputes. Leak: LiRA AUC on the link score (query access) after a log-only deletion (0.5 = nothing left).}
\label{tab:arch}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
 & val.\ AP & test MRR & cone & leak after log-only deletion \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ------------------------------------------------------------------------------------------------ appendix tables
def tab_delays():
    St = S.get("streaming", {})
    rows = []
    for d, lab in DS:
        for dl, dlab in (("rand", "random"), ("active", "active users"), ("burst", "outages"), ("offline", "offline users"), ("itemshard", "slow item shard")):
            for bq in (200, 50):
                G = St.get(f"main|{d}|attn|{dl}|bq{bq}")
                if not G:
                    continue
                M = G["methods"]
                cells = [f(g(M, m, "dmrr_rel_pct"), "{:+.2f}", 1, False) for m in ("arrival", "drop", "buffer:1", "buffer:4", "eps:0.05")]
                cells += [f(g(M, m, "cost_x_normal"), "{:.2f}", 1, False) for m in ("replay", "exact", "eps:0.05", "khop:1")]
                rows.append(f"{lab} & {dlab} & {bq} & {G['n']} & {f(G['late_frac'], '{:.1f}', 100, False)} & " + " & ".join(cells) + " \\\\")
    write("tab_delays", r"""\begin{table}[!htb]
\centering\scriptsize
\caption{Every delay setting: MRR change relative to exact repair (\%) and compute (multiples of normal processing), mean
over seeds.}
\label{tab:delays}
\setlength{\tabcolsep}{2.4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lllrrrrrrrrrrr@{}}
\toprule
 & delay model & step & seeds & late \% & arrival & drop & buffer 1h & buffer 4h & $\varepsilon$=0.05 & replay & exact & $\varepsilon$=0.05 & 1-hop \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")


def tab_jac():
    C = S.get("contraction", {})
    rows = []
    for key, lab in (("wikipedia|attn", "Wikipedia"), ("reddit|attn", "Reddit"), ("mooc|attn", "MOOC"), ("lastfm|attn", "LastFM"),
                     ("sec|attn", "SEC Form 4"), ("wikipedia|id", "Wikipedia, memory-only"), ("wikipedia|jodie", "Wikipedia, JODIE")):
        c = C.get(key)
        if not c:
            continue
        rows.append(f"{lab} & {c['n']} & {f(c['J_own'], '{:.2f}')} & {f(c['J_partner'], '{:.2f}')} & {f(c['sum_median'], '{:.2f}')} & "
                    f"{f(c['certified'], '{:.1f}', 100)} \\\\")
    write("tab_jac", r"""\begin{table}[!htb]
\centering\scriptsize
\caption{Spectral norms of the memory update's Jacobians on every test update (median over updates; mean $\pm$ std over
models). Certified: updates with $\|J_{\text{own}}\|+\|J_{\text{partner}}\|<1$.}
\label{tab:jac}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
 & models & $\|J_{\text{own}}\|$ & $\|J_{\text{partner}}\|$ & sum & certified (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tab_horizon():
    St = S.get("streaming", {})
    rows = []
    for d, lab in (("wikipedia", "Wikipedia"), ("mooc", "MOOC")):
        G = St.get(f"horizon|{d}|attn|rand|bq200")
        if not G:
            continue
        M = G["methods"]
        for m, h in (("exact", "full"), ("exact@50", "50 steps"), ("exact@20", "20 steps"), ("exact@10", "10 steps"), ("exact@5", "5 steps")):
            if m not in M:
                continue
            rows.append(f"{lab} & {h} & {f(g(M, m, 'journal_MB'), '{:.1f}', 1, False)} & {f(g(M, m, 'fallback'), '{:.0f}', 1, False)} & "
                        f"{f(g(M, m, 'cost_x_normal'), '{:.2f}', 1, False)} & {f(g(M, m, 'top1_agree'), '{:.1f}', 100, False)} \\\\")
    write("tab_horizon", r"""\begin{table}[!htb]
\centering\scriptsize
\caption{Journal length (random delays, mean over seeds). Late events older than the journal are applied on arrival.}
\label{tab:horizon}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llrrrr@{}}
\toprule
 & journal & memory (MB) & applied on arrival & compute ($\times$) & top-1 same as exact (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tab_data():
    import torch                                                   # noqa: F401  (load() needs it)
    import sys
    sys.path.insert(0, str(REPO))
    from lategraph.data import load
    rows = []
    for d, lab in DS + [("tgbl-review", "tgbl-review"), ("sec", "SEC Form 4")]:
        try:
            s = load(d).summary()
        except Exception:
            rows.append(f"{lab} & -- & -- & -- & -- & -- \\\\"); continue
        rows.append(f"{lab} & {s['events']:,} & {s['nodes']:,} & {s['test']:,} & {s['msg_dim']} & {s['span_days']:,.0f} \\\\")
    write("tab_data", r"""\begin{table}[!htb]
\centering\scriptsize
\caption{Datasets (chronological splits).}
\label{tab:data}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
 & events & nodes & test events & edge features & days \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ------------------------------------------------------------------------------------------------ every deletion attack
def tab_del_full():
    meths = [("store", "log-only"), ("reset", "reset"), ("restart", "learned restart"), ("khop:1", "1-hop rebuild"),
             ("eps:0.05", "$\\varepsilon$-repair"), ("exact", "exact repair")]
    rows = []
    for d, lab in DS + [("tgbl-review", "tgbl-review")]:
        Lm = g(S, "lira", f"{d}|attn", "methods"); Dm = g(S, "deletion", f"main|{d}|attn", "methods")
        if not Lm:
            continue
        n = g(S, "lira", f"{d}|attn", "n")
        for i, (k, ml) in enumerate(meths):
            if k not in Lm and not g(Dm, k):
                continue
            cells = [f(g(Lm, k, "raw_logit_auc"), "{:.2f}"), f(g(Dm, k, "attacks", "probe_0"), "{:.2f}"),
                     f(g(Lm, k, "logit_auc"), "{:.2f}"), f(g(Lm, k, "cosine_auc"), "{:.2f}"),
                     f(g(Lm, k, "cosine_tpr1"), "{:.1f}", 100, False), f(g(Dm, k, "cost_x_normal"), "{:.2f}", 1, False)]
            rows.append((f"{lab} ({n})" if i == 0 else "") + f" & {ml} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    write("tab_del_full", r"""\begin{table}[!htb]
\centering\scriptsize
\caption{Every deletion attack (membership AUC, mean $\pm$ std over seeds; number of seeds in brackets). Score: the model's
own link score; probe: logistic regression on the two memories; LiRA: likelihood-ratio attack on the link score or on the
cosine of the two memories, with counterfactual shadow streams; TPR: true-positive rate of the memory LiRA at 1\% false
positives; cost: updates computed by the deletion method, as a multiple of normal processing up to the request.}
\label{tab:delfull}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llcccccr@{}}
\toprule
 & method & score & probe & LiRA (score) & LiRA (memory) & TPR@1\% & cost \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ------------------------------------------------------------------------------------------------ weaker attackers
VARS = [("", "full knowledge, at the request"), ("know0.5", "knows 50\\% of the other events"), ("know0.1", "knows 10\\%"),
        ("know0", "knows none of them"), ("after20", "20 steps after the request"), ("after50", "50 steps after"),
        ("K100", "$K=100$ candidate pairs"), ("K25", "$K=25$"), ("know0.5|K25", "$K=25$, knows 50\\%"),
        ("know0|K25", "$K=25$, knows none")]


def lira_variant(d, tag):
    return g(S, "lira", f"{d}|attn", "methods") if not tag else g(S, "lira_know", f"{d}|attn|{tag}", "methods")


def tab_lira_robust():
    dsets = DS + [("tgbl-review", "tgbl-review")]
    rows = []
    for tag, lab in VARS:
        cells = []
        for d, _ in dsets:
            M = lira_variant(d, tag)
            cells += [f(g(M, m, "logit_auc"), "{:.2f}", 1, False) for m in ("store", "reset")]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab in dsets)
    rule = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(dsets)))
    write("tab_lira_robust", r"""\begin{table}[t]
\centering\scriptsize
\caption{The leak under other query-access attackers: LiRA AUC on the link score after log-only deletion (del.) and after
resetting the memories (reset), mean over seeds. The attacker's shadow streams leave out the other test-period events it does
not know; or it attacks 20 or 50 steps after the request; or fewer candidate pairs are unknown at once ($K$; at most 1{,}000
per group by default, 338 on Wikipedia). With partial knowledge and the default $K$, $\varepsilon$-repair, the 1-hop rebuild and
exact repair stay at or below \PartialApproxMax{}; the other rows expose them (Table~\ref{tab:liraknow}). With $K=25$ the AUCs are
noisy: exact repair, which leaks nothing, ranges from \ExactKTwentyFiveMin{} to \ExactKTwentyFiveMax.}
\label{tab:robust}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}l""" + "cc" * len(dsets) + r"""@{}}
\toprule
attacker & """ + head + r""" \\
""" + rule + "\n & " + " & ".join(["del.", "reset"] * len(dsets)) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tab_lira_know():
    meths = [("store", "log-only"), ("reset", "reset"), ("restart", "restart"), ("khop:1", "1-hop"), ("eps:0.05", "$\\varepsilon$-repair"),
             ("exact", "exact")]
    rows = []
    for d, lab in DS + [("tgbl-review", "tgbl-review")]:
        first = True
        for tag, vl in VARS:
            M = lira_variant(d, tag)
            if not M:
                continue
            cell = lambda m: (f"{g(M, m, 'logit_auc')['mean']:.2f}\\,/\\,{g(M, m, 'cosine_auc')['mean']:.2f}" if g(M, m, "logit_auc") else "--")
            rows.append((lab if first else "") + f" & {vl} & " + " & ".join(cell(m) for m, _ in meths) + " \\\\")
            first = False
        rows.append("\\midrule")
    write("tab_lira_know", r"""\begin{table}[!htb]
\centering\scriptsize
\caption{The likelihood-ratio attack for every attacker variant and deletion method: AUC with query access (LiRA on the link
score) / with white-box access (LiRA on the cosine of the two memories), mean over seeds (5 for Wikipedia and Reddit, 3
otherwise; 16 shadow streams for tgbl-review, 64 otherwise).}
\label{tab:liraknow}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}ll""" + "c" * len(meths) + r"""@{}}
\toprule
 & attacker & """ + " & ".join(l for _, l in meths) + r""" \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


if __name__ == "__main__":
    import sys
    for fn in (tab_exact, tab_train_del, tab_cert, tab_late, tab_arch, tab_delays, tab_jac, tab_horizon, tab_del_full, tab_lira_know, tab_lira_robust) + ((tab_data,) if "--data" in sys.argv else ()):
        try:
            fn()
        except Exception as e:
            print(f"skipped {fn.__name__}: {type(e).__name__}: {e}")
