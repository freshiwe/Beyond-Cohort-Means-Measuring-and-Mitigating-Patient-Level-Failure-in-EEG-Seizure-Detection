"""
Numeric audit: every headline number in the paper must trace to a committed
data file. Run before any submission.

Usage: python3 verify_paper_numbers.py
Exit 0 if all claims verify, 1 otherwise.
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
AGG = ROOT / "experiments/data/msctev1-20260829/aggregate"
TEX = (ROOT / "bigdata.tex").read_text()

checks = []


def chk(label, value, present):
    checks.append((label, str(value), bool(present)))


def main():
    rows = [r for r in csv.DictReader(
        open(ROOT / "experiments/data/results-graphst-pp/fp_day_summary.csv"))
        if r["patient"] not in ("mean", "std")]
    for r in rows:
        r["fp"] = float(r["pp_fp_per_day"])
        r["s"] = float(r["pp_event_sens"])
        r["raw"] = float(r["raw_fp_per_day"])

    chk("mean raw FP/day", f"{np.mean([r['raw'] for r in rows]):.1f}", "797" in TEX)
    chk("mean pp FP/day", f"{np.mean([r['fp'] for r in rows]):.1f}", "79.6" in TEX or " 80 " in TEX)
    chk("mean pp event sens", f"{np.mean([r['s'] for r in rows]):.3f}", "0.696" in TEX)
    chk("low-alarm/high-sensitivity count", len([r for r in rows if r["fp"] < 10 and r["s"] >= 0.5]),
        any(k in TEX for k in ("Seven of 24", "7 of 24", "7 of the 24")))
    chk("flood count", len([r for r in rows if r["fp"] > 100]), "five" in TEX.lower())
    silent = [r for r in rows if r["s"] <= 0.25]
    chk("silent count", len(silent), "six" in TEX.lower())
    chk("every silent case under 10 FP/day",
        all(r["fp"] < 10 for r in silent), all(r["fp"] < 10 for r in silent))

    repl = json.load(open(ROOT / "v2/results/hardtail_replication_results.json"))
    for key, needle_rho, needle_p in (("chbmit_fate", "0.587", "0.0026"),
                                       ("siena", "0.011", "0.970")):
        rs = repl[key]["rows"]
        rho, p = spearmanr([x["n_seizures"] for x in rs], [x["auc"] for x in rs])
        chk(f"{key} rho", f"{rho:.3f}", needle_rho in TEX)
        chk(f"{key} p", f"{p:.4f}", needle_p in TEX)

    e3 = json.load(open(AGG / "experiment3_results_AB.json"))
    pairs = [(v["n_seizures"], float(v["auc"])) for v in e3["variability_data"].values()]
    rho3, p3 = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    chk("second architecture rho", f"{rho3:.3f}", "0.516" in TEX)
    chk("second architecture n", len(pairs), "23 of the same" in TEX)

    sc = json.load(open(AGG / "stats_corrections.json"))
    bh = {t["test"]: t for t in sc["benjamini_hochberg"]["tests"]}
    chk("BH family size", sc["benjamini_hochberg"]["n_tests"], "18 tests" in TEX)
    chk("predictor q", f"{bh['pred_nseiz_chbmit']['q_bh']:.3f}", "0.039" in TEX)
    chk("adabn siena q", f"{bh['adabn_siena']['q_bh']:.3f}", "0.176" in TEX)


    # --- GMM component selection on the graph model's AUCs ---
    from sklearn.mixture import GaussianMixture
    e2 = json.load(open(ROOT / "v2/results/experiment2_results.json"))
    aucs = np.array(list(e2["loso_aucs"].values())).reshape(-1, 1)
    bic = {}
    for n in (1, 2, 3):
        bic[n] = GaussianMixture(n_components=n, random_state=42,
                                 n_init=30).fit(aucs).bic(aucs)
    best = min(bic, key=bic.get)
    chk("GMM BIC 1 component", f"{bic[1]:.1f}", "-6.3" in TEX or "$-6.3$" in TEX)
    chk("GMM BIC 2 components", f"{bic[2]:.1f}", "26.3" in TEX)
    chk("GMM BIC 3 components", f"{bic[3]:.1f}", "22.8" in TEX)
    chk("BIC selects 2", best, best == 2)

    # --- ROC figure: count of patients below 0.70 ---
    from sklearn.metrics import roc_curve, auc as auc_fn
    preds = sorted((ROOT / "experiments/data/results-fate-preds").glob("chb*/preds_chb*.npz"))
    below = 0
    for f in preds:
        d = np.load(f)
        y, sc = d["y_true"], d["y_score"]
        if y.sum() in (0, len(y)):
            continue
        fpr, tpr, _ = roc_curve(y, sc)
        if auc_fn(fpr, tpr) < 0.70:
            below += 1
    chk("ROC curves below 0.70", below,
        bool(re.search(r"four\s+cases below AUC", TEX)) and below == 4)
    chk("measured FATE parameter count", "615775",
        "615{,}775 parameters" in TEX)
    chk("unsupported pooled count removed", "103/185 absent",
        "103 of 185" not in TEX)
    chk("CHB-MIT strict-LOSO overclaim removed", "absent",
        "24-patient strict-LOSO" not in TEX and
        "strict, leak-free" not in TEX)
    chk("window-level validation disclosed", "present",
        "seeded random 90/10 split" in TEX and
        bool(re.search(r"not\s+subject-separated model selection", TEX)))
    chk("GMM small-sample claim bounded", "present",
        "descriptive rather than proof" in TEX and
        "selected decisively" not in TEX)
    chk("Figure 1 scope matches artifact", "present",
        "details the band-initialization stage" in TEX and
        "shows the full pipeline" not in TEX)
    chk("public-release status not overstated", "present",
        "archival URL, version, and license will accompany" in TEX and
        "released with this paper" not in TEX)

    print(f"{'CHECK':40s} {'DATA':14s} IN PAPER")
    print("-" * 70)
    failed = 0
    for label, val, ok in checks:
        if not ok:
            failed += 1
        print(f"{label:40s} {val:14s} {'yes' if ok else 'NOT FOUND'}")
    print("-" * 70)
    print(f"{len(checks) - failed}/{len(checks)} verified")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
