"""
Generate the paper figures from committed result files only.

Every number plotted here traces to a file tracked in git:
  v2/results/hardtail_replication_results.json          (FATE CHB-MIT + Siena AUCs, seizure counts)
  experiments/data/msctev1-20260829/aggregate/loso_results.csv        (MSCTEV1 CHB-MIT per-patient AUC)
  data/generated/msctev1_siena_loso.csv                               (MSCTEV1 Siena per-patient AUC)
  experiments/data/msctev1-20260829/aggregate/tta_results.json        (AdaBN before/after)
  experiments/data/msctev1-20260829/aggregate/experiment3_results_AB.json (mechanism tests)

Usage: python3 make_paper_figures.py   (writes ../../figures/fig_*.pdf)
"""
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # projects/SeizureDetection
AGG = ROOT / "experiments/data/msctev1-20260829/aggregate"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# Clean sans-serif per project figure rule. Arial/Liberation are not installed
# on this host; DejaVu Sans is the available metric-similar sans fallback.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

BLUE, RED, GREY = "#2b6cb0", "#c53030", "#718096"


def save(fig, name):
    """PDF for LaTeX, PNG for quick review and notebooks."""
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
COL1 = 3.4   # IEEE single column width, inches
COL2 = 7.1   # IEEE double column width


def load_repl():
    with open(ROOT / "v2/results/hardtail_replication_results.json") as fh:
        return json.load(fh)


def load_msctev1_auc():
    aucs = {}
    with open(AGG / "loso_results.csv") as fh:
        for r in csv.DictReader(fh):
            pid = r.get("patient") or r.get("pid")
            if not pid or pid in ("mean", "std"):
                continue
            for k in ("auc", "AUC", "test_auc"):
                if k in r and r[k]:
                    aucs[pid] = float(r[k])
                    break
    return aucs


def load_msctev1_siena_auc():
    aucs = {}
    with open(ROOT / "data/generated/msctev1_siena_loso.csv") as fh:
        for r in csv.DictReader(fh):
            pid = r.get("patient")
            if not pid or pid in ("mean", "std"):
                continue
            if r.get("auc"):
                aucs[pid] = float(r["auc"])
    return aucs


# ---------------------------------------------------------------- Figure 1
def fig_hard_tail(repl, msc, msc_siena):
    """Per-patient LOSO AUC, sorted, both architectures on CHB-MIT and Siena."""
    fate = {r["patient"]: r["auc"] for r in repl["chbmit_fate"]["rows"]}
    siena = {r["patient"]: r["auc"] for r in repl["siena"]["rows"]}

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5),
                             gridspec_kw={"width_ratios": [24, 14]})

    ax = axes[0]
    order = sorted(fate, key=lambda p: fate[p])
    x = np.arange(len(order))
    ax.bar(x - 0.2, [fate[p] for p in order], width=0.4, color=BLUE, label="FATE")
    have = [p for p in order if p in msc]
    ax.bar([order.index(p) + 0.2 for p in have], [msc[p] for p in have],
           width=0.4, color=RED, label="MSCTEV1")
    ax.axhline(0.5, ls=":", c=GREY, lw=0.9)
    ax.axhline(0.70, ls="--", c="k", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=90)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Held-out ROC-AUC")
    ax.set_title("(a) CHB-MIT ($n=24$)")
    # Legend is attached to the figure, above both panels.
    handles, labels = ax.get_legend_handles_labels()

    ax = axes[1]
    order = sorted(siena, key=lambda p: siena[p])
    x2 = np.arange(len(order))
    ax.bar(x2 - 0.2, [siena[p] for p in order], width=0.4, color=BLUE)
    have2 = [p for p in order if p in msc_siena]
    ax.bar([order.index(p) + 0.2 for p in have2], [msc_siena[p] for p in have2],
           width=0.4, color=RED)
    ax.axhline(0.5, ls=":", c=GREY, lw=0.9)
    ax.axhline(0.70, ls="--", c="k", lw=0.9)
    ax.set_xticks(x2)
    ax.set_xticklabels(order, rotation=90)
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) Siena ($n=14$)")

    fig.legend(handles, labels, frameon=False, ncol=2, fontsize=7.5,
               loc="upper center", bbox_to_anchor=(0.5, 1.04), handlelength=1.3)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig_hard_tail")
    plt.close(fig)
    print("wrote fig_hard_tail.pdf")


# ---------------------------------------------------------------- Figure 2
def fig_predictor(repl):
    """Seizure count vs LOSO AUC: holds on CHB-MIT, does not transfer to Siena."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6), sharey=True)
    for ax, key, name in zip(axes, ["chbmit_fate", "siena"],
                             ["(a) CHB-MIT (FATE)", "(b) Siena"]):
        rows = repl[key]["rows"]
        x = np.array([r["n_seizures"] for r in rows], float)
        y = np.array([r["auc"] for r in rows], float)
        rho, p = spearmanr(x, y)
        ax.scatter(x, y, s=26, color=BLUE if key == "chbmit_fate" else GREY,
                   edgecolor="white", linewidth=0.5, zorder=3)
        # Label hard-tail points, alternating vertical offset so neighbours do
        # not collide, and flipping to the left near the right edge so labels
        # stay inside the axes.
        hard = sorted([r for r in rows if r["auc"] < 0.70],
                      key=lambda r: (r["n_seizures"], r["auc"]))
        xmax = x.max()
        for i, r in enumerate(hard):
            right_edge = r["n_seizures"] > 0.75 * xmax
            dx = -7 if right_edge else 7
            ha = "right" if right_edge else "left"
            dy = 6 if i % 2 == 0 else -10
            ax.annotate(r["patient"], (r["n_seizures"], r["auc"]),
                        fontsize=6, xytext=(dx, dy), ha=ha,
                        textcoords="offset points", color=RED, zorder=5,
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground="white")])
        ax.set_xlabel("Number of seizures")
        ax.set_title(f"{name} ($n={len(rows)}$)\n$\\rho$={rho:+.3f}, $p$={p:.4f}")
        ax.axhline(0.70, ls="--", c="k", lw=0.8)
    axes[0].set_ylabel("Held-out ROC-AUC")
    axes[0].set_ylim(0.15, 1.05)
    fig.tight_layout()
    save(fig, "fig_predictor")
    plt.close(fig)
    print("wrote fig_predictor.pdf")


# ---------------------------------------------------------------- Figure 3
def fig_mechanisms():
    """The two intuitive mechanisms, both ruled out."""
    path = AGG / "experiment3_results_AB.json"
    if not path.exists():
        print("skip fig_mechanisms: no experiment3 results")
        return
    d = json.load(open(path))
    var, isi = d["variability_data"], d["isi_data"]

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.4))

    def panel(ax, data, field, title, ylabel, alt):
        hard = [float(v[field]) for v in data.values()
                if v.get("group") == "Hard" or float(v["auc"]) < 0.70]
        easy = [float(v[field]) for v in data.values() if float(v["auc"]) >= 0.85]
        hard = [h for h in hard if np.isfinite(h)]
        easy = [e for e in easy if np.isfinite(e)]
        _, p = mannwhitneyu(hard, easy, alternative="two-sided")
        bp = ax.boxplot([hard, easy], tick_labels=[f"Hard\n(n={len(hard)})",
                                              f"Easy\n(n={len(easy)})"],
                        widths=0.55, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], [RED, BLUE]):
            patch.set_facecolor(c); patch.set_alpha(0.3); patch.set_edgecolor(c)
        for med in bp["medians"]:
            med.set_color("k")
        for i, vals in enumerate([hard, easy], start=1):
            ax.scatter(np.random.default_rng(0).normal(i, 0.05, len(vals)), vals,
                       s=14, color=[RED, BLUE][i - 1], zorder=3, alpha=0.85)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n two-sided $p$={p:.3f}")

    panel(axes[0], var, "cv", "(a) Seizure-duration variability",
          "Coefficient of variation", "greater")
    isi_f = {k: v for k, v in isi.items() if np.isfinite(float(v["mean_isi"]))}
    panel(axes[1], isi_f, "mean_isi", "(b) Inter-seizure interval",
          "Mean ISI (s)", "less")
    fig.tight_layout()
    save(fig, "fig_mechanisms")
    plt.close(fig)
    print("wrote fig_mechanisms.pdf")


# ---------------------------------------------------------------- Figure 4
def fig_adabn():
    """AdaBN before/after, per patient, sorted by baseline."""
    d = json.load(open(AGG / "tta_results.json"))
    d = sorted(d, key=lambda r: r["baseline_auc"])
    base = np.array([r["baseline_auc"] for r in d])
    adap = np.array([r["adapted_auc"] for r in d])
    pids = [r["patient"] for r in d]
    x = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(COL2, 2.5))
    for xi, b, a in zip(x, base, adap):
        ax.plot([xi, xi], [b, a], color=(RED if a < b else BLUE),
                lw=1.1, zorder=1, alpha=0.8)
    ax.scatter(x, base, s=22, color=GREY, zorder=3, label="Baseline")
    ax.scatter(x, adap, s=22, color=BLUE, zorder=3, label="After AdaBN")
    ax.axhline(0.70, ls="--", c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(pids, rotation=90)
    ax.set_ylabel("Held-out ROC-AUC")
    ax.set_ylim(0.2, 1.05)
    # The adjacent table reports the aggregate tests; keep this paired figure
    # descriptive rather than duplicating an inferential result in the title.
    ax.set_title(f"MSCTEV1 CHB-MIT folds: unlabeled BN recalibration ($n$={len(d)})")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_adabn")
    plt.close(fig)
    print("wrote fig_adabn.pdf")

    tail = [(r["patient"], r["baseline_auc"], r["adapted_auc"])
            for r in d if r["baseline_auc"] < 0.70]
    print("  hard-tail patients (baseline<0.70):")
    for pid, b, a in tail:
        print(f"    {pid}: {b:.4f} -> {a:.4f}  ({a-b:+.4f})")


# ---------------------------------------------------------------- Figure 5
def fig_trace(pid="chb01"):
    """Per-window seizure probability against ground truth for one patient.

    Replaces a notebook export whose axis labels were unreadable at print size.
    """
    f = ROOT / f"experiments/data/results-fate-preds/{pid}/preds_{pid}.npz"
    if not f.exists():
        print(f"skip fig_trace: {f} missing")
        return
    d = np.load(f)
    y, s_, thr = d["y_true"], d["y_score"], float(d["threshold"])

    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    x = np.arange(len(s_))
    ax.plot(x, s_, lw=0.35, color=BLUE, zorder=2)
    ict = y == 1
    ax.fill_between(x, 0, 1, where=ict, color=RED, alpha=0.30, step="mid",
                    linewidth=0, zorder=1)
    ax.axhline(thr, ls="--", lw=0.9, color="k", zorder=3)
    ax.set_xlim(0, len(s_))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Window index")
    ax.set_ylabel("P(ictal)")
    ax.set_title(f"{pid}: per-window probability against annotation")
    # No legend: the trace is dense enough that any in-axes placement covers
    # signal. The caption identifies the shading and the threshold line.
    fig.tight_layout()
    save(fig, "fig_trace")
    plt.close(fig)
    print("wrote fig_trace.pdf")


# ---------------------------------------------------------------- Figure 6
def fig_bimodal():
    """GMM fit to the graph model's per-patient AUC distribution.

    Panel (b) plots BIC relative to the best model rather than raw BIC: the
    raw values are all negative, so bars drawn from zero obscure the
    differences that actually drive model selection.
    """
    from sklearn.mixture import GaussianMixture
    src = ROOT / "v2/results/experiment2_results.json"
    if not src.exists():
        print("skip fig_bimodal: no experiment2 results")
        return
    aucs = np.array(list(json.load(open(src))["loso_aucs"].values())).reshape(-1, 1)

    bic, gmms = {}, {}
    for n in (1, 2, 3):
        g = GaussianMixture(n_components=n, random_state=42, n_init=30).fit(aucs)
        bic[n], gmms[n] = g.bic(aucs), g
    best = min(bic, key=bic.get)
    gb = gmms[best]

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5),
                             gridspec_kw={"width_ratios": [2.3, 1]})

    # ---- (a) distribution and fit
    ax = axes[0]
    xs = np.linspace(0.25, 1.06, 400).reshape(-1, 1)
    ax.hist(aucs.ravel(), bins=12, density=True, color="#d5dbe3",
            edgecolor="white", linewidth=0.6, zorder=1)
    for i, idx in enumerate(np.argsort(gb.means_.ravel())):
        mu = gb.means_.ravel()[idx]
        w = gb.weights_.ravel()[idx]
        sd = np.sqrt(gb.covariances_.ravel()[idx])
        comp = w * np.exp(-0.5 * ((xs.ravel() - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        col = [RED, BLUE][i % 2]
        ax.fill_between(xs.ravel(), comp, alpha=0.4, color=col, zorder=2)
        # label each component in place rather than in a legend box
        ax.annotate(f"$\\mu$={mu:.2f}\nw={w:.2f}", (mu, comp.max()),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=6.5, color=col, zorder=5,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.plot(xs, np.exp(gb.score_samples(xs)), color="k", lw=1.5, zorder=4,
            label=f"{best}-component fit")
    ax.set_xlabel("Held-out AUC")
    ax.set_ylabel("Density")
    ax.set_ylim(0, None)
    ax.set_title("(a) Per-patient AUC distribution")
    ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.4)

    # ---- (b) BIC relative to the selected model
    ax = axes[1]
    ns = sorted(bic)
    delta = [bic[n] - bic[best] for n in ns]
    cols = [BLUE if n == best else GREY for n in ns]
    ax.bar([str(n) for n in ns], delta, color=cols, width=0.62)
    for n, d in zip(ns, delta):
        ax.annotate("selected" if d == 0 else f"+{d:.1f}",
                    (str(n), d), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=6.5,
                    color=BLUE if d == 0 else "k")
    ax.set_xlabel("GMM components")
    ax.set_ylabel(r"$\Delta$BIC vs. best")
    ax.set_ylim(0, max(delta) * 1.28)
    ax.set_title("(b) Model selection")

    fig.tight_layout()
    save(fig, "fig_bimodal")
    plt.close(fig)
    print("wrote fig_bimodal.pdf  BIC " +
          ", ".join(f"{n}:{bic[n]:.1f}" for n in ns) + f"  best={best}")


# ---------------------------------------------------------------- Figure 7
def fig_roc():
    """Per-patient ROC curves from the committed per-window predictions.

    Cannot be reconstructed from summary AUC, which is why the raw
    predictions are stored.
    """
    from sklearn.metrics import roc_curve, auc as auc_fn
    pred_dir = ROOT / "experiments/data/results-fate-preds"
    files = sorted(pred_dir.glob("chb*/preds_chb*.npz"))
    if not files:
        print("skip fig_roc: no prediction files")
        return
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    aucs = []
    for f in files:
        d = np.load(f)
        y, sc = d["y_true"], d["y_score"]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        fpr, tpr, _ = roc_curve(y, sc)
        a = auc_fn(fpr, tpr)
        aucs.append(a)
        ax.plot(fpr, tpr, lw=0.7, alpha=0.75,
                color=RED if a < 0.70 else GREY)
    ax.plot([0, 1], [0, 1], ls=":", lw=0.9, color="k")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Per-patient ROC, n={len(aucs)}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=RED, lw=1.2, label="AUC $<$ 0.70"),
                       Line2D([], [], color=GREY, lw=1.2, label="AUC $\\geq$ 0.70")],
              frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_roc")
    plt.close(fig)
    print(f"wrote fig_roc.pdf  (n={len(aucs)}, {sum(a<0.70 for a in aucs)} below 0.70)")


if __name__ == "__main__":
    repl = load_repl()
    msc = load_msctev1_auc()
    msc_siena = load_msctev1_siena_auc()
    fig_hard_tail(repl, msc, msc_siena)
    fig_predictor(repl)
    fig_mechanisms()
    fig_adabn()
    fig_trace()
    fig_bimodal()
    fig_roc()
