"""
Aggregates a multi-seed LOSO sweep into per-fold median/variance AUC and a
5-seed ensemble AUC, plus a cohort summary.

The sweep writes per-(model, seed, fold) results to:
  results-multiseed/<model>/seed<SEED>/<fold>/loso_<model>.csv
where <model> is fate or graphst, SEED in 1..5, fold is chb01..chb24
(graphst has no chb12). Each loso_<model>.csv has a header, one patient row
for THAT fold, then a mean row and a std row. The fold-row AUC is read from
the first data row (the one whose patient column is not mean or std).

The data may not be present locally yet (it is harvested from a cluster PVC
later), so every step tolerates missing seeds, missing folds, and missing
predictions. A tiny synthetic fixture (--selftest) exercises the logic
without any real data.

Usage:
  python aggregate_multiseed.py --model fate
  python aggregate_multiseed.py --model graphst --root /data/results-multiseed
  python aggregate_multiseed.py --selftest
"""
import argparse
import csv
import glob
import os

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

SEEDS = [1, 2, 3, 4, 5]
HARD_TAIL = 0.70


def repo_results_root():
    # src/ -> experiments/ -> results-multiseed/
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "results-multiseed"))


def repo_figures_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "figures"))


def read_fold_auc(csv_path):
    """Return the fold-row AUC from one loso csv, or None if unreadable.

    The fold row is the first data row whose patient is not mean or std.
    """
    try:
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                pid = (row.get("patient") or "").strip().lower()
                if pid in ("mean", "std", ""):
                    continue
                val = row.get("auc")
                if val is None or val == "":
                    return None
                auc = float(val)
                if np.isnan(auc):
                    return None
                return auc
    except Exception:
        return None
    return None


def find_pred_file(fold_dir, fold):
    """Return the preds npz path for a fold dir, or None.

    Accepts preds_<fold>.npz or any preds_*.npz in the dir.
    """
    direct = os.path.join(fold_dir, "preds_%s.npz" % fold)
    if os.path.exists(direct):
        return direct
    hits = sorted(glob.glob(os.path.join(fold_dir, "preds_*.npz")))
    return hits[0] if hits else None


def load_preds(npz_path):
    """Return (y_true, y_score) as float arrays, or None on any problem."""
    try:
        d = np.load(npz_path)
        y_true = np.asarray(d["y_true"]).ravel().astype(float)
        y_score = np.asarray(d["y_score"]).ravel().astype(float)
    except Exception:
        return None
    if y_true.shape[0] == 0 or y_true.shape[0] != y_score.shape[0]:
        return None
    return y_true, y_score


def ensemble_auc(pred_list):
    """Average y_score across seeds (aligned by y_true length) and score AUC.

    pred_list is a list of (y_true, y_score). Entries whose length differs
    from the first entry are skipped. Returns None if sklearn is missing,
    fewer than 2 seeds align, y_true disagrees, or only one class is present.
    """
    if not HAVE_SKLEARN or not pred_list:
        return None
    n = pred_list[0][0].shape[0]
    kept_true = None
    scores = []
    for y_true, y_score in pred_list:
        if y_true.shape[0] != n:
            continue
        if kept_true is None:
            kept_true = y_true
        elif not np.array_equal(kept_true, y_true):
            # same-length but different labels; cannot trust alignment
            continue
        scores.append(y_score)
    if len(scores) < 2 or kept_true is None:
        return None
    if len(np.unique(kept_true)) < 2:
        return None
    mean_score = np.mean(np.stack(scores, axis=0), axis=0)
    try:
        return float(roc_auc_score(kept_true, mean_score))
    except Exception:
        return None


def discover_folds(root, model):
    """Return sorted list of fold names that appear under any seed."""
    pattern = os.path.join(root, model, "seed*", "*", "loso_%s.csv" % model)
    folds = set()
    for path in glob.glob(pattern):
        fold = os.path.basename(os.path.dirname(path))
        folds.add(fold)
    return sorted(folds)


def aggregate(root, model):
    """Return a list of per-fold dicts with AUC stats and ensemble AUC."""
    folds = discover_folds(root, model)
    rows = []
    for fold in folds:
        aucs = []
        preds = []
        for seed in SEEDS:
            fold_dir = os.path.join(root, model, "seed%d" % seed, fold)
            csv_path = os.path.join(fold_dir, "loso_%s.csv" % model)
            auc = read_fold_auc(csv_path)
            if auc is not None:
                aucs.append(auc)
            npz = find_pred_file(fold_dir, fold)
            if npz is not None:
                p = load_preds(npz)
                if p is not None:
                    preds.append(p)
        arr = np.array(aucs, dtype=float)
        row = {
            "fold": fold,
            "n_seeds": len(aucs),
            "auc_median": float(np.median(arr)) if arr.size else float("nan"),
            "auc_mean": float(np.mean(arr)) if arr.size else float("nan"),
            "auc_std": float(np.std(arr)) if arr.size else float("nan"),
            "auc_min": float(np.min(arr)) if arr.size else float("nan"),
            "auc_max": float(np.max(arr)) if arr.size else float("nan"),
            "ensemble_auc": ensemble_auc(preds),
        }
        rows.append(row)
    return rows


def cohort_summary(rows):
    medians = np.array(
        [r["auc_median"] for r in rows if not np.isnan(r["auc_median"])],
        dtype=float,
    )
    below = int(np.sum(medians < HARD_TAIL)) if medians.size else 0
    return {
        "n_folds": len(rows),
        "n_folds_scored": int(medians.size),
        "mean_of_medians": float(np.mean(medians)) if medians.size else float("nan"),
        "median_of_medians": float(np.median(medians)) if medians.size else float("nan"),
        "n_folds_below_0.70": below,
    }


def fmt(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    if isinstance(x, float):
        return "%.4f" % x
    return str(x)


def write_csv(out_path, rows):
    fields = [
        "fold", "n_seeds", "auc_median", "auc_mean", "auc_std",
        "auc_min", "auc_max", "ensemble_auc",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in rows:
            w.writerow([fmt(r[f]) for f in fields])
    return out_path


def setup_font():
    """Use Arial if its TTFs are installed, else leave the default sans."""
    try:
        import matplotlib
        from matplotlib import font_manager
    except Exception:
        return
    font_dir = os.path.expanduser("~/.local/share/fonts")
    arial = os.path.join(font_dir, "Arial.ttf")
    if os.path.exists(arial):
        try:
            for name in ("Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"):
                p = os.path.join(font_dir, name)
                if os.path.exists(p):
                    font_manager.addfont(p)
            matplotlib.rcParams["font.family"] = "Arial"
            return
        except Exception:
            pass
    matplotlib.rcParams["font.family"] = "sans-serif"


def make_figure(rows, model, out_path):
    """Grayscale-safe sorted per-fold median AUC with min/max error bars.

    Returns the path written, or None if matplotlib is unavailable or there
    is nothing to plot.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    setup_font()

    scored = [r for r in rows if not np.isnan(r["auc_median"])]
    if not scored:
        return None
    scored = sorted(scored, key=lambda r: r["auc_median"])
    x = np.arange(len(scored))
    med = np.array([r["auc_median"] for r in scored])
    lo = med - np.array([r["auc_min"] for r in scored])
    hi = np.array([r["auc_max"] for r in scored]) - med
    labels = [r["fold"] for r in scored]

    fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * len(scored)), 4.0))
    ax.errorbar(
        x, med, yerr=[lo, hi], fmt="o", color="black", ecolor="0.5",
        elinewidth=1.0, capsize=3, markersize=4,
    )
    ax.axhline(HARD_TAIL, linestyle="--", color="0.3", linewidth=1.0)
    ax.text(len(scored) - 1, HARD_TAIL + 0.005, "AUC 0.70",
            ha="right", va="bottom", fontsize=8, color="0.3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("per-fold median AUC (min/max band)")
    ax.set_xlabel("fold (sorted by median AUC)")
    ax.set_title("%s multi-seed LOSO" % model)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle=":", color="0.85")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def print_summary(model, rows, summary):
    print("model: %s" % model)
    print("folds discovered: %d  (scored: %d)"
          % (summary["n_folds"], summary["n_folds_scored"]))
    print("mean of per-fold medians:   %s" % fmt(summary["mean_of_medians"]))
    print("median of per-fold medians: %s" % fmt(summary["median_of_medians"]))
    print("folds below AUC %.2f (hard tail): %d"
          % (HARD_TAIL, summary["n_folds_below_0.70"]))
    print("")
    print("%-8s %6s %9s %9s %9s %9s %9s %9s"
          % ("fold", "nseeds", "median", "mean", "std", "min", "max", "ens"))
    for r in rows:
        print("%-8s %6d %9s %9s %9s %9s %9s %9s" % (
            r["fold"], r["n_seeds"], fmt(r["auc_median"]), fmt(r["auc_mean"]),
            fmt(r["auc_std"]), fmt(r["auc_min"]), fmt(r["auc_max"]),
            fmt(r["ensemble_auc"]),
        ))


def build_fixture(root, model):
    """Write a tiny synthetic results-multiseed tree for self-testing.

    Three folds, five seeds. One fold (chb17) is a hard-tail case with AUC
    well below 0.70. chb01 is missing seed 3 to exercise the missing-seed
    path. Writes preds npz per fold so the ensemble path runs.
    """
    rng = np.random.default_rng(0)
    fold_auc = {"chb01": 0.90, "chb05": 0.82, "chb17": 0.58}
    n = 200
    # one fixed label vector per fold, shared across seeds (same test set)
    fold_labels = {f: rng.integers(0, 2, size=n).astype(float) for f in fold_auc}
    for fold, base in fold_auc.items():
        for seed in SEEDS:
            if fold == "chb01" and seed == 3:
                continue  # missing seed
            fold_dir = os.path.join(root, model, "seed%d" % seed, fold)
            os.makedirs(fold_dir, exist_ok=True)
            auc = float(np.clip(base + rng.normal(0, 0.02), 0.0, 1.0))
            csv_path = os.path.join(fold_dir, "loso_%s.csv" % model)
            with open(csv_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["patient", "auc", "auprc", "sensitivity",
                            "specificity", "f1"])
                w.writerow([fold, "%.4f" % auc, "0.5", "0.8", "0.8", "0.6"])
                w.writerow(["mean", "%.4f" % auc, "0.5", "0.8", "0.8", "0.6"])
                w.writerow(["std", "0.0", "0.0", "0.0", "0.0", "0.0"])
            # synthetic preds: separable for high-AUC folds, noisy for tail
            y_true = fold_labels[fold]
            signal = (base - 0.5) * 4.0  # stronger separation for higher base
            y_score = rng.normal(0, 1, size=n) + signal * y_true
            np.savez(os.path.join(fold_dir, "preds_%s.npz" % fold),
                     y_true=y_true, y_score=y_score)


def selftest():
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="multiseed_selftest_")
    try:
        root = os.path.join(tmp, "results-multiseed")
        model = "fate"
        build_fixture(root, model)
        rows = aggregate(root, model)
        summary = cohort_summary(rows)
        out_csv = os.path.join(root, "aggregate_%s.csv" % model)
        write_csv(out_csv, rows)
        print_summary(model, rows, summary)
        print("")
        print("wrote %s" % out_csv)
        # sanity checks
        assert summary["n_folds"] == 3, summary
        assert summary["n_folds_below_0.70"] == 1, summary
        c1 = [r for r in rows if r["fold"] == "chb01"][0]
        assert c1["n_seeds"] == 4, c1  # seed 3 missing
        print("selftest: OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="fate", choices=["fate", "graphst"])
    ap.add_argument("--root", default=None,
                    help="results-multiseed root (default: repo location)")
    ap.add_argument("--no-figure", action="store_true",
                    help="skip the matplotlib figure")
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic fixture self-test and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    root = args.root or repo_results_root()
    rows = aggregate(root, args.model)
    summary = cohort_summary(rows)

    out_csv = os.path.join(root, "aggregate_%s.csv" % args.model)
    write_csv(out_csv, rows)
    print_summary(args.model, rows, summary)
    print("")
    print("wrote %s" % out_csv)

    if not args.no_figure:
        fig_path = os.path.join(repo_figures_dir(),
                                "fig_multiseed_%s.png" % args.model)
        written = make_figure(rows, args.model, fig_path)
        if written:
            print("wrote %s" % written)
        else:
            print("figure skipped (matplotlib missing or nothing to plot)")


if __name__ == "__main__":
    main()
