"""Result figures of the final paper and the blog post, drawn from results/summary.json (scripts/aggregate.py).

  python scripts/make_figures_final.py        -> ../paper/figures/*.pdf and ../blog/figures/*.png
Missing results are skipped (the script can be run while experiments are still going).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

REPO = Path(__file__).resolve().parents[1]
S = json.load(open(REPO / "results" / "summary.json"))
OUT_PDF = REPO / "report" / "figures"                                 # the report inside the repository
OUT_PDFS = [OUT_PDF] + ([REPO.parent / "paper" / "figures"] if (REPO.parent / "paper").exists() else [])
OUT_PNG = REPO.parent / "blog" / "figures" if (REPO.parent / "blog").exists() else REPO / "report" / "figures"
for _o in OUT_PDFS + [OUT_PNG]:
    _o.mkdir(parents=True, exist_ok=True)
C = dict(store="#c53030", reset="#dd6b20", restart="#d69e2e", eps="#3182ce", khop="#805ad5", exact="#2f855a", none="#718096",
         replay="#4a5568", arrival="#d53f8c", drop="#975a16", buf="#e53e3e", oracle="#1a202c", memory="#3182ce",
         finetune="#805ad5", retrain="#2f855a")
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.dpi": 150})
DSN = dict(wikipedia="Wikipedia", reddit="Reddit", mooc="MOOC", lastfm="LastFM", sec="SEC Form 4", **{"tgbl-review": "tgbl-review"})


def save(fig, name):
    for o in OUT_PDFS:
        fig.savefig(o / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_PNG / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)


def m(d, *keys, default=None):
    """Nested lookup that tolerates missing results; returns (mean, std) of an aggregated entry."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    if isinstance(d, dict) and "mean" in d:
        return d["mean"], d["std"]
    return d


def deletion_full():
    ds = [d for d in ("wikipedia", "reddit", "mooc", "lastfm", "tgbl-review") if f"{d}|attn" in S.get("lira", {})]
    if not ds:
        return
    meths = [("store", "log-only"), ("reset", "reset"), ("restart", "restart"), ("eps:0.05", "ε-repair"), ("exact", "exact")]
    attacks = [("raw", "link score"), ("probe", "memory probe"), ("lira_logit", "LiRA (score)"), ("lira_cos", "LiRA (memory)")]
    fig, axes = plt.subplots(1, len(ds), figsize=(2.75 * len(ds), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, ds):
        L = S["lira"][f"{d}|attn"]["methods"]; Dl = S.get("deletion", {}).get(f"main|{d}|attn", {}).get("methods", {})
        w = 0.19
        for j, (a, alab) in enumerate(attacks):
            xs, ys, es = [], [], []
            for i, (k, _) in enumerate(meths):
                if a == "raw":
                    v = m(L, k, "raw_logit_auc")
                elif a == "probe":
                    v = m(Dl, k, "attacks", "probe_0")
                elif a == "lira_logit":
                    v = m(L, k, "logit_auc")
                else:
                    v = m(L, k, "cosine_auc")
                if v is None:
                    continue
                xs.append(i + (j - 1.5) * w); ys.append(v[0]); es.append(v[1])
            ax.bar(xs, ys, w, yerr=es, capsize=1.5, label=alab, color=["#a0aec0", "#4a5568", "#c53030", "#ed8936"][j])
        ax.axhline(0.5, color="k", lw=0.8, ls="--")
        ax.set_xticks(range(len(meths))); ax.set_xticklabels([l for _, l in meths], rotation=30, ha="right")
        for i, (k, _) in enumerate(meths):
            if m(L, k, "cosine_auc") is None:
                ax.text(i, 0.52, "not run", ha="center", va="bottom", fontsize=7, color="#718096", rotation=90)
        n = S["lira"][f"{d}|attn"]["n"]
        ax.set_title(f"{DSN[d]} ({n} seed{'s' if n > 1 else ''})", fontsize=9); ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("membership AUC (0.5 = forgotten)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.08))
    save(fig, "deletion_full")


def deletion():
    """The deletion leak for two attacker classes (the paper and the blog). (a) Query access: the model's link score of the
    pair, alone (hollow) and in the likelihood-ratio test (filled). (b) White-box access: a probe on the two memories
    (hollow) and the likelihood-ratio test on their cosine (filled). Mean +- std over seeds, every dataset."""
    ds = [d for d in ("wikipedia", "reddit", "mooc", "lastfm", "tgbl-review") if f"{d}|attn" in S.get("lira", {})]
    if not ds:
        return
    meths = [("store", "log-only"), ("reset", "reset"), ("restart", "restart"), ("khop:1", "1-hop"), ("eps:0.05", "ε-repair"),
             ("exact", "exact")]
    cols = {"wikipedia": "#3182ce", "reddit": "#e53e3e", "mooc": "#38a169", "lastfm": "#805ad5", "tgbl-review": "#dd6b20"}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55), sharey=True)
    off = np.linspace(-0.3, 0.3, len(ds))
    panels = [("(a) query access: link scores", lambda L, D, k: m(L, k, "raw_logit_auc"), lambda L, D, k: m(L, k, "logit_auc")),
              ("(b) white-box access: memories", lambda L, D, k: m(D, k, "attacks", "probe_0"), lambda L, D, k: m(L, k, "cosine_auc"))]
    for ax, (title, weak, strong) in zip(axes, panels):
        for j, d in enumerate(ds):
            L = S["lira"][f"{d}|attn"]["methods"]; D = S.get("deletion", {}).get(f"main|{d}|attn", {}).get("methods", {})
            for i, (k, _) in enumerate(meths):
                x = i + off[j]
                st, wk = strong(L, D, k), weak(L, D, k)
                if st is not None:
                    ax.errorbar(x, st[0], yerr=st[1], fmt="o", ms=4, color=cols[d], ecolor=cols[d], elinewidth=0.8, capsize=1.2,
                                label=DSN[d] if i == 0 else None, zorder=3)
                if wk is not None:
                    ax.plot(x, wk[0], "o", ms=4, mfc="white", mec=cols[d], mew=0.9, zorder=2)
        ax.axhline(0.5, color="k", lw=0.8, ls="--", zorder=1)
        for i in range(1, len(meths)):
            ax.axvline(i - 0.5, color="#e2e8f0", lw=0.8, zorder=0)
        ax.set_xticks(range(len(meths))); ax.set_xticklabels([l for _, l in meths], fontsize=7.5, rotation=25, ha="right")
        ax.set_xlim(-0.5, len(meths) - 0.5); ax.set_ylim(0.0, 1.03); ax.set_title(title, fontsize=8.5)
        ax.tick_params(axis="y", labelsize=7.5)
    axes[0].set_ylabel("membership AUC (0.5 = forgotten)", fontsize=8)
    h, lab = axes[0].get_legend_handles_labels()
    keys = [plt.Line2D([], [], marker="o", ls="", color="#4a5568", ms=4.5),
            plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#4a5568", ms=4.5)]
    fig.legend(h + keys, lab + ["likelihood-ratio attack", "score / probe alone"], loc="upper center", ncol=7,
               bbox_to_anchor=(0.5, 1.07), fontsize=7, handletextpad=0.2, columnspacing=0.9)
    fig.tight_layout(w_pad=0.6)
    save(fig, "deletion")


def train_deletion():
    T = S.get("train_deletion", {})
    ds = [d for d in ("wikipedia", "reddit", "mooc") if d in T]
    if not ds:
        return
    meths = [("none", "no deletion"), ("store", "log-only"), ("memory", "memory repair"), ("finetune", "+ fine-tune"), ("retrain", "retrain")]
    fig, axes = plt.subplots(1, len(ds), figsize=(3.2 * len(ds), 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, ds):
        M = T[d]["methods"]
        for j, (att, col) in enumerate((("logit", "#a0aec0"), ("probe", "#4a5568"))):
            xs = np.arange(len(meths)) + (j - 0.5) * 0.36
            ys = [M[k][att]["mean"] for k, _ in meths]; es = [M[k][att]["std"] for k, _ in meths]
            ax.bar(xs, ys, 0.36, yerr=es, capsize=1.5, color=col, label={"logit": "link score", "probe": "memory probe"}[att])
        ax.axhline(0.5, color="k", lw=0.8, ls="--")
        ax.set_xticks(range(len(meths))); ax.set_xticklabels([l for _, l in meths], rotation=30, ha="right")
        ax.set_title(f"{DSN[d]} ({T[d]['n']} seeds)"); ax.set_ylim(0.4, 0.8)
    axes[0].set_ylabel("membership AUC"); axes[-1].legend(fontsize=8)
    save(fig, "train_deletion")


def certificate():
    Cn, St = S.get("contraction", {}), S.get("streaming", {})
    pts = []
    for var, lab, col, mk in (("attn", "standard TGN", "#1a202c", "o"), ("rho0.95_lam10", "penalty ρ=0.95 (λ=10)", "#90cdf4", "s"),
                              ("rho0.85_lam10", "penalty ρ=0.85 (λ=10)", "#3182ce", "s"), ("rho0.75_lam10", "penalty ρ=0.75 (λ=10)", "#1a365d", "s"),
                              ("rho0.85_lam1", "penalty ρ=0.85, λ=1", "#3182ce", "^"), ("rho0.85_lam100", "penalty ρ=0.85, λ=100", "#3182ce", "v"),
                              ("crnn0.95", "contractive cell ρ≤0.95 (c=0.5)", "#9ae6b4", "D"), ("crnn0.9", "contractive cell ρ≤0.9 (c=0.5)", "#48bb78", "D"),
                              ("crnn0.8", "contractive cell ρ≤0.8 (c=0.5)", "#22543d", "D"),
                              ("crnn0.95c0.7", "contractive cell ρ≤0.95, c=0.7", "#f6ad55", "D"),
                              ("crnn0.99c0.9", "contractive cell ρ≤0.99, c=0.9", "#dd6b20", "D")):
        c = Cn.get(f"wikipedia|{var}"); s = St.get(f"main|wikipedia|{var}|rand|bq200")
        if not c or not s:
            continue
        cert = 100 * np.array(c["certified"]["values"]) if "crnn" not in var else np.full(c["n"], 100.0)
        mrr = np.array(s["methods"]["oracle"]["mrr"]["values"])
        k = min(len(cert), len(mrr))
        pts.append((lab, col, mk, cert[:k], mrr[:k]))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    for lab, col, mk, x, y in pts:
        ax.scatter(x, y, s=12, color=col, marker=mk, alpha=0.35)
        ax.scatter([x.mean()], [y.mean()], s=64, color=col, marker=mk, edgecolor="k", lw=0.6, label=lab)
    ax.set_xlabel("updates that contract (%; contractive cell: all, by construction)"); ax.set_ylabel("test MRR (Wikipedia)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.5, scatterpoints=1)
    ax.set_xlim(-5, 105)
    save(fig, "certificate")


def late_events():
    St = S.get("streaming", {})
    ds = [d for d in ("wikipedia", "reddit", "mooc") if f"main|{d}|attn|rand|bq200" in St]
    if not ds:
        return
    pol = [("replay", "full replay", C["replay"]), ("exact", "exact repair", C["exact"]), ("eps:0.05", "ε-repair", C["eps"]),
           ("khop:1", "1-hop rebuild", C["khop"]), ("arrival", "apply on arrival", C["arrival"]), ("drop", "drop late", C["drop"]),
           ("buffer:1", "buffer 1 h", C["buf"])]
    fig, axes = plt.subplots(1, len(ds), figsize=(3.4 * len(ds), 2.8))
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, ds):
        g = St[f"main|{d}|attn|rand|bq200"]["methods"]
        for k, lab, col in pol:
            if k not in g:
                continue
            cost = np.array(g[k]["cost_x_normal"]["values"]); dis = 100 * (1 - np.array(g[k]["top1_agree"]["values"]))
            ax.errorbar(cost.mean(), dis.mean(), xerr=cost.std(), yerr=dis.std(), fmt="o", color=col, ms=5, capsize=2, label=lab)
        ax.set_xscale("log"); ax.set_xlabel("compute (× normal processing)"); ax.set_title(f"{DSN[d]} ({St[f'main|{d}|attn|rand|bq200']['n']} seeds)")
    axes[0].set_ylabel("top-1 predictions that differ\nfrom exact repair (%)")
    axes[-1].legend(fontsize=7, loc="upper right")
    save(fig, "late_events")


def influence():
    """Exact leave-one-event-out influence: by recency and channel (left); what attention sees and tracks (right). One colour
    per dataset in both panels; right: shares with 95% bootstrap intervals over the queries of all seeds."""
    I = S.get("influence", {})
    ds = [d for d in ("wikipedia", "reddit", "mooc") if d in I]
    if not ds:
        return
    AV = json.load(open(REPO / "results" / "influence" / "attention_vs_channels.json")) if (REPO / "results" / "influence" / "attention_vs_channels.json").exists() else {}
    cols = {"wikipedia": "#3182ce", "reddit": "#e53e3e", "mooc": "#38a169"}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6), gridspec_kw=dict(width_ratios=[1.2, 1]))
    for d in ds:
        r = np.arange(1, len(I[d]["by_rank"]) + 1)
        axes[0].plot(r, I[d]["mem_by_rank"], "-o", ms=2.5, color=cols[d], lw=1.2)
        axes[0].plot(r, I[d]["list_by_rank"], "--", color=cols[d], lw=1)
    axes[0].axvline(10.5, color="k", lw=0.8, ls=":")
    axes[0].text(10.8, 0.92 * axes[0].get_ylim()[1], "attention sees only\nthe 10 newest", fontsize=6.5, ha="left", va="top", color="#4a5568")
    axes[0].set_xlabel("how recent the deleted event is (1 = newest)", fontsize=8); axes[0].set_ylabel("mean |change of the score|", fontsize=8)
    axes[0].tick_params(labelsize=7.5); axes[0].set_xticks([1, 5, 10, 15, 20])
    h = [plt.Line2D([], [], color="#4a5568", marker="o", ms=2.5, lw=1.2), plt.Line2D([], [], color="#4a5568", ls="--", lw=1)]
    axes[0].legend(h, ["through the memory", "through the neighbour list"], fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.72))
    bars = [("outside", "influence that\nattention\nnever sees"), ("top1_list", "top-attended\n= top event\n(list channel)"),
            ("top1_total", "top-attended\n= top event\n(total)")]
    x = np.arange(len(bars)); w = 0.8 / len(ds)
    for i, d in enumerate(ds):
        a = AV.get(d, {})
        vals = [100 * I[d]["share_outside_list"]["mean"], 100 * a.get("top1_list", np.nan), 100 * a.get("top1_total", np.nan)]
        lo = [100 * I[d]["share_outside_list"]["std"], 100 * (a.get("top1_list", 0) - a.get("top1_list_ci", [0, 0])[0]),
              100 * (a.get("top1_total", 0) - a.get("top1_total_ci", [0, 0])[0])]
        hi = [100 * I[d]["share_outside_list"]["std"], 100 * (a.get("top1_list_ci", [0, 0])[1] - a.get("top1_list", 0)),
              100 * (a.get("top1_total_ci", [0, 0])[1] - a.get("top1_total", 0))]
        axes[1].bar(x + (i - (len(ds) - 1) / 2) * w, vals, w, yerr=[lo, hi], capsize=2, color=cols[d], label=DSN[d])
    axes[1].set_xticks(x); axes[1].set_xticklabels([l for _, l in bars], fontsize=6.5); axes[1].set_ylabel("% of influence / of predictions", fontsize=8)
    axes[1].tick_params(axis="y", labelsize=7.5)
    axes[1].plot([0.55, 2.45], [10, 10], color="k", ls="--", lw=0.8); axes[1].text(2.47, 10, "chance", fontsize=6.5, va="center")
    axes[1].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "influence")


def repair_age():
    """Speed-up of exact repair over replay against the age of the edit (scripts/bench_age.py), one panel per dataset;
    edits that change a memory (blue, median line per age bin) and edits that change nothing (grey)."""
    ds = [d for d in ("wikipedia", "reddit", "mooc", "lastfm") if (REPO / "results" / "bench_age" / f"bench_age_{d}.json").exists()]
    if not ds:
        return
    fig, axes = plt.subplots(1, len(ds), figsize=(2.2 * len(ds), 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, ds):
        b = json.load(open(REPO / "results" / "bench_age" / f"bench_age_{d}.json"))
        age = np.array([t["age_steps"] for t in b["trials"]]); sp = np.array([t["ms_replay"] / max(t["ms_indexed"], 1e-6) for t in b["trials"]])
        eff = np.array([t["recomputed"] > 0 for t in b["trials"]])
        ax.scatter(age[~eff], sp[~eff], s=5, alpha=0.4, color="#a0aec0", lw=0, label="changes nothing")
        ax.scatter(age[eff], sp[eff], s=5, alpha=0.5, color="#3182ce", lw=0, label="changes a memory")
        if eff.sum() >= 8:
            bins = np.unique(np.quantile(age[eff], np.linspace(0, 1, 6)).astype(int))
            mid, med = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                sel = eff & (age >= lo) & (age <= hi)
                if sel.sum() >= 3:
                    mid.append((lo + hi) / 2); med.append(np.median(sp[sel]))
            ax.plot(mid, med, color="#1a365d", lw=1.5, marker="o", ms=3)
        ax.axhline(1, color="#c53030", lw=0.8, ls="--")
        ax.set_yscale("log"); ax.set_title(f"{DSN[d]} ({b['window_steps']} steps)", fontsize=8.5)
        ax.set_xlabel("age of the edit (steps)", fontsize=8); ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("replay time / repair time", fontsize=8)
    axes[0].legend(fontsize=6.5, loc="upper left", markerscale=2, handletextpad=0.2)
    axes[0].text(0.97, 0.03, "below the dashed line,\nreplay is faster", transform=axes[0].transAxes, ha="right", va="bottom",
                 fontsize=6.3, color="#c53030")
    for ax in axes:
        lo, hi = ax.get_ylim(); ax.set_ylim(min(lo, 0.2), hi)
    fig.tight_layout()
    save(fig, "repair_age")


if __name__ == "__main__":
    for f in (deletion, deletion_full, train_deletion, certificate, late_events, influence, repair_age):
        try:
            f()
        except Exception as e:                                    # keep going while results are incomplete
            print(f"skipped {f.__name__}: {type(e).__name__}: {e}", file=sys.stderr)
        if not (OUT_PDF / f"{f.__name__}.pdf").exists():          # a placeholder so that drafts still compile
            fig, ax = plt.subplots(figsize=(6, 1.2)); ax.axis("off")
            ax.text(0.5, 0.5, f"{f.__name__}: results pending", ha="center", va="center", color="#c53030")
            for o in OUT_PDFS:
                fig.savefig(o / f"{f.__name__}.pdf")
            plt.close(fig)
