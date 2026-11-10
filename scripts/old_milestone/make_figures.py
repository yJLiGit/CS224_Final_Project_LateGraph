"""Figures for the milestone report (PDF, from results/summary.json and the raw result files).

  python scripts/make_figures.py [--out report/figures]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph import paths              # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
C = dict(exact="#2a9d8f", replay="#264653", eps="#8e7dbe", khop="#e9c46a", arrival="#457b9d", drop="#a8dadc",
         buffer="#e76f51", oracle="#6c757d", store="#e76f51", reset="#9d0208", restart="#f4a261", none="#adb5bd", restart_cone="#f4a261")
DS = [("wikipedia", "Wikipedia"), ("reddit", "Reddit"), ("mooc", "MOOC")]


def color(m):
    return C.get(m.split(":")[0].split("@")[0], "#333333")


def label(m):
    k, _, a = m.partition(":")
    return {"exact": "exact (cone)", "replay": "full replay", "eps": f"$\\varepsilon$={a}", "khop": f"{a}-hop rebuild",
            "arrival": "arrival order", "arrival_proc": "re-stamped", "drop": "drop late", "buffer": f"buffer {a} h",
            "oracle": "oracle", "store": "log-only delete", "reset": "reset", "restart": "learned restart",
            "restart_cone": "restart cone", "none": "no deletion", "arrival_logfix": "arrival + log fix"}.get(k, m)


MARK = {"replay": "s", "exact": "o", "eps:0.01": "v", "eps:0.05": "^", "eps:0.2": "<", "khop:1": "D", "khop:2": "d",
        "arrival": "P", "drop": "X", "buffer:1": "*", "buffer:4": "h"}
ECOL = {"eps:0.01": "#b8a9e3", "eps:0.05": "#8e7dbe", "eps:0.2": "#5e4b9c", "khop:1": "#e9c46a", "khop:2": "#c9a227",
        "buffer:1": "#e76f51", "buffer:4": "#9d0208"}


def fig_cost_error(S, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3), sharey=True)
    methods = ["replay", "exact", "eps:0.01", "eps:0.05", "eps:0.2", "khop:1", "khop:2", "arrival", "drop", "buffer:1"]
    for ax, (ds, name) in zip(axes, DS):
        g = S["streaming"].get(f"main|{ds}|attn|rand|bq200")
        if not g:
            continue
        for m in methods:
            if m not in g["methods"]:
                continue
            x = g["methods"][m]["cost_x_normal"]["mean"]; y = 100 * (1 - g["methods"][m]["top1_agree"]["mean"])
            ax.errorbar(x, y, yerr=100 * g["methods"][m]["top1_agree"]["std"], fmt=MARK[m], ms=5 if m != "buffer:1" else 7,
                        color=ECOL.get(m, color(m)), capsize=1.5, lw=0.8, label=label(m), mec="white", mew=0.4)
        ax.set_xscale("log"); ax.set_title(f"{name} ({100 * g['late_frac']['mean']:.0f}% of test events late)")
        ax.set_xlabel("compute (× normal processing)")
        ax.axvline(1, color="#cccccc", lw=0.6, zorder=0); ax.grid(axis="y", color="#eeeeee", lw=0.5)
    axes[0].set_ylabel("top-1 prediction differs\nfrom exact repair (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=6.5, handletextpad=0.3, columnspacing=1.2)
    plt.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(out / "cost_error.pdf", bbox_inches="tight"); plt.close(fig)


def fig_accuracy(S, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.9), sharey=False)
    methods = ["oracle", "eps:0.05", "khop:1", "arrival", "arrival_proc", "drop", "buffer:1", "buffer:4"]
    for ax, (ds, name) in zip(axes, DS):
        g = S["streaming"].get(f"main|{ds}|attn|rand|bq200")
        if not g:
            continue
        ms_ = [m for m in methods if m in g["methods"]]
        vals = [g["methods"][m]["dmrr_rel_pct"]["mean"] for m in ms_]
        errs = [g["methods"][m]["dmrr_rel_pct"]["std"] for m in ms_]
        ax.bar(range(len(ms_)), vals, yerr=errs, color=[color(m) for m in ms_], capsize=1.5, error_kw=dict(lw=0.7))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(ms_))); ax.set_xticklabels([label(m) for m in ms_], rotation=55, ha="right", fontsize=6)
        ax.set_title(f"{name}: exact MRR {g['methods']['exact']['mrr']['mean']:.3f}")
    axes[0].set_ylabel("MRR vs. exact repair (%)")
    plt.tight_layout(); fig.savefig(out / "accuracy_policies.pdf", bbox_inches="tight"); plt.close(fig)


def fig_deletion(S, out):
    groups = [("main|wikipedia|attn", "Wikipedia\nTGN"), ("main|wikipedia|id", "Wikipedia\nmemory-only"), ("main|reddit|attn", "Reddit\nTGN"),
              ("main|wikipedia|rho0.75_lam10", "Wikipedia\ncontractive")]
    methods = ["none", "store", "reset", "restart", "eps:0.05", "exact"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35), sharey=True)
    for ax, att, title in ((axes[0], "logit", "attack 1: the model's own link score"), (axes[1], "probe", "attack 2: learned memory probe")):
        w = 0.82 / len(methods)
        for gi, (key, name) in enumerate(groups):
            g = S["deletion"].get(key)
            if not g:
                continue
            for mi, m in enumerate(methods):
                if m not in g["methods"]:
                    continue
                a = g["methods"][m]["attacks"].get(f"{att}_0")
                if not a:
                    continue
                ax.bar(gi + (mi - (len(methods) - 1) / 2) * w, a["mean"], w * 0.92, yerr=a["std"], color=color(m), capsize=1,
                       error_kw=dict(lw=0.6), label=label(m) if gi == 0 else None)
        ax.axhline(0.5, color="k", ls="--", lw=0.7)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels([n for _, n in groups], fontsize=6.5)
        ax.set_ylim(0.4, 0.95); ax.set_title(title)
    axes[1].annotate("chance", xy=(1.0, 0.5), xycoords=("axes fraction", "data"), xytext=(2, 0), textcoords="offset points",
                     va="center", fontsize=6)
    axes[0].set_ylabel("membership AUC\n(deleted vs. never-seen pairs)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=6.5)
    plt.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out / "deletion_auc.pdf", bbox_inches="tight"); plt.close(fig)


def fig_contraction(S, out):
    fig, ax = plt.subplots(figsize=(3.3, 2.25))
    variants = [("attn", "standard TGN", "#6c757d"), ("rho0.95_lam10", "penalty, $\\rho$=0.95", "#457b9d"),
                ("rho0.85_lam10", "penalty, $\\rho$=0.85", "#8e7dbe"), ("rho0.75_lam10", "penalty, $\\rho$=0.75", "#2a9d8f")]
    for var, name, col in variants:
        c = S["contraction"].get(f"wikipedia|{var}")
        g = S["streaming"].get(f"main|wikipedia|{var}|rand|bq200")
        if not c or not g:
            continue
        unc = 100 - np.array(c["certified"]["values"]) * 100          # share of updates that amplify (not certified)
        mrr = np.array(g["methods"]["oracle"]["mrr"]["values"])
        ax.scatter(unc, mrr, color=col, s=12, alpha=0.55)
        ax.errorbar(unc.mean(), mrr.mean(), yerr=mrr.std(ddof=1), fmt="D", ms=5, color=col, capsize=2, lw=0.8, label=name)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("test updates that amplify, $\\|J_{own}\\|+\\|J_{partner}\\|\\geq 1$ (%)")
    ax.set_ylabel("MRR (no delays)")
    ax.set_xticks([100, 30, 10, 1, 0.1]); ax.set_xticklabels(["100", "30", "10", "1", "0.1"])
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    plt.tight_layout(); fig.savefig(out / "contraction_frontier.pdf", bbox_inches="tight"); plt.close(fig)


def fig_cone(out):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.05))
    for ds, name, col in (("wikipedia", "Wikipedia", "#2a9d8f"), ("reddit", "Reddit", "#457b9d"), ("mooc", "MOOC", "#e76f51")):
        p = paths.RESULTS / "exactness" / f"exactness_tgn_{ds}_s0.json"
        if not p.exists():
            continue
        tr = [t for t in json.load(open(p))["trials"] if t["kind"] != "mixed"]
        back = np.array([t["steps_back"] for t in tr]); frac = np.array([100 * t["recomputed"] / max(t["window_updates"], 1) for t in tr])
        deg = np.array([t["endpoint_max_degree"] for t in tr])
        bins = np.array([1, 10, 20, 40, 60, 80, 101])
        mid = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        mean = [frac[(back >= bins[i]) & (back < bins[i + 1])].mean() for i in range(len(bins) - 1)]
        axes[0].plot(mid, mean, "o-", color=col, ms=3, label=name)
        q = np.quantile(deg, [0, .5, .9, .99, 1.0])
        vals = [frac[(deg >= q[i]) & (deg <= q[i + 1])].mean() for i in range(4)]
        axes[1].plot(range(4), vals, "o-", color=col, ms=3, label=name)
    for ax in axes:
        ax.set_yscale("log"); ax.grid(axis="y", color="#eeeeee", lw=0.5)
    axes[0].set_xlabel("how many steps back the edited event is"); axes[0].set_ylabel("cone size: recomputed updates\n(% of all updates since the edit)")
    axes[0].legend(frameon=False)
    axes[1].set_xticks(range(4)); axes[1].set_xticklabels(["< p50", "p50-p90", "p90-p99", "> p99"])
    axes[1].set_xlabel("degree of the edited event's busiest endpoint (percentile)")
    plt.tight_layout(); fig.savefig(out / "cone_size.pdf", bbox_inches="tight"); plt.close(fig)


def fig_sec(S, out):
    from lategraph.sec import load_sec_stream
    st = load_sec_stream()
    lag = ((st.arrival[st.test0:] - st.t[st.test0:]).double() / 86400).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.1), gridspec_kw=dict(width_ratios=[1, 1.35]))
    x = np.sort(lag); y = np.arange(1, len(x) + 1) / len(x)
    axes[0].plot(np.maximum(x, 0.05), 100 * y, color="#264653")
    axes[0].set_xscale("log"); axes[0].set_xlabel("filing lag: trade date to EDGAR acceptance (days)")
    axes[0].set_ylabel("test transactions (%)")
    axes[0].axvline(4, color="#e76f51", lw=0.6, ls="--")
    axes[0].text(4.6, 12, f"{100 * (lag > 4).mean():.0f}% filed more\nthan 4 days late", fontsize=6, color="#e76f51")
    axes[0].set_xticks([0.5, 1, 2, 4, 10, 30, 100, 365]); axes[0].set_xticklabels(["0.5", "1", "2", "4", "10", "30", "100", "365"])
    axes[0].grid(color="#eeeeee", lw=0.5)
    g = S["streaming"].get("main_amend|sec|attn|real|bq200")
    if g:
        methods = [m for m in ["oracle", "eps:0.05", "khop:1", "arrival", "arrival_logfix", "buffer:24", "buffer:96", "drop"] if m in g["methods"]]
        vals = [g["methods"][m]["dmrr_rel_pct"]["mean"] for m in methods]; errs = [g["methods"][m]["dmrr_rel_pct"]["std"] for m in methods]
        ys = np.arange(len(methods))[::-1]
        axes[1].barh(ys, vals, xerr=errs, height=0.65, color=[color(m) for m in methods], capsize=1.5, error_kw=dict(lw=0.7))
        axes[1].axvline(0, color="k", lw=0.6)
        axes[1].set_yticks(ys); axes[1].set_yticklabels([label(m) for m in methods], fontsize=6.5)
        axes[1].set_xlabel("MRR vs. exact repair (%)"); axes[1].grid(axis="x", color="#eeeeee", lw=0.5)
        axes[1].set_title(f"exact repair: MRR {g['methods']['exact']['mrr']['mean']:.3f} (3 seeds, 6-month test)")
    plt.tight_layout(); fig.savefig(out / "sec.pdf", bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(paths.REPORT / "figures"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    S = json.load(open(paths.RESULTS / "summary.json"))
    for f in (fig_cost_error, fig_accuracy, fig_deletion, fig_contraction):
        try:
            f(S, out); print("ok", f.__name__)
        except Exception as e:
            print("FAILED", f.__name__, repr(e))
    try:
        fig_cone(out); print("ok fig_cone")
    except Exception as e:
        print("FAILED fig_cone", repr(e))
    try:
        fig_sec(S, out); print("ok fig_sec")
    except Exception as e:
        print("FAILED fig_sec", repr(e))


if __name__ == "__main__":
    main()
