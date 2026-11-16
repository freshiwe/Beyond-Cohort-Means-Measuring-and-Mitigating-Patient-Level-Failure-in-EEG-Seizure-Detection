"""
Merge FATEv3's 5-way ablation shard outputs into v3_ablation_summary.csv, same
shape as the existing experiments/data/results-abl/ablation_summary.csv
(condition,mean_auc,delta_vs_full,wilcoxon_p_auc,mean_f1), so the same
plotting code (spectral_figures.fig_ablation_bar / fig_pareto) works unchanged.

Each condition's shards write <results>/<condition>/<fold>/loso_fate_v3<_ablation>.csv
(per-fold, one row) -- same per-shard-subdirectory convention as merge_shards.py
expects, produced by job-v3-ablation.yaml / gen-v3-ablation-shards.sh.

Usage: python v3_ablation_summary.py --results /data/results-v3-abl \
    --out /data/results-v3-abl/v3_ablation_summary.csv
"""
import os
import csv
import glob
import argparse
import numpy as np
from scipy.stats import wilcoxon

from model_v3 import ABLATIONS_V3


def load_condition(results_root, cond):
    """<results_root>/<cond>/<fold>/loso_fate_v3[_<cond>].csv -> {patient: auc}."""
    tag = "fate_v3" if cond == "none" else f"fate_v3_{cond}"
    per_patient = {}
    for f in sorted(glob.glob(os.path.join(results_root, cond, "*", f"loso_{tag}.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r["patient"] in ("mean", "std"):
                    continue
                per_patient[r["patient"]] = {"auc": float(r["auc"]), "f1": float(r["f1"])}
    return per_patient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/results-v3-abl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.results, "v3_ablation_summary.csv")

    by_cond = {c: load_condition(args.results, c) for c in ABLATIONS_V3}
    full = by_cond.get("none", {})
    if not full:
        raise SystemExit(f"no shard CSVs found for condition 'none' under {args.results}/none/*/")
    full_auc_full_mean = np.mean([v["auc"] for v in full.values()])

    rows = []
    for cond in ABLATIONS_V3:
        pp = by_cond[cond]
        if not pp:
            print(f"  [skip] {cond}: no shard CSVs found yet")
            continue
        aucs = np.array([v["auc"] for v in pp.values()])
        f1s = np.array([v["f1"] for v in pp.values()])
        mean_auc = float(aucs.mean())
        delta = mean_auc - full_auc_full_mean
        if cond == "none":
            p = 1.0
        else:
            common = sorted(set(full) & set(pp))
            if len(common) < 2:
                p = float("nan")
            else:
                a = [full[k]["auc"] for k in common]
                b = [pp[k]["auc"] for k in common]
                try:
                    p = float(wilcoxon(a, b).pvalue)
                except ValueError:
                    p = float("nan")  # all differences zero, or too few non-tied pairs
        rows.append({"condition": "full_v3" if cond == "none" else cond,
                     "mean_auc": f"{mean_auc:.4f}", "delta_vs_full": f"{delta:.4f}",
                     "wilcoxon_p_auc": f"{p:.4f}", "mean_f1": f"{f1s.mean():.4f}"})
        print(f"  {cond:20s} n={len(pp)} mean_auc={mean_auc:.4f} delta={delta:+.4f} p={p:.4f}")

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["condition", "mean_auc", "delta_vs_full",
                                           "wilcoxon_p_auc", "mean_f1"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
