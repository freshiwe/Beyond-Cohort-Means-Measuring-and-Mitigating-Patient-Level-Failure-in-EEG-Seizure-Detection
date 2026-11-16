"""
Reconstructs the definitive 24-patient full_v3 vs lean LOSO head-to-head from
the real per-patient preds_chb*.npz files (y_true, y_score, threshold saved
by train.py's run_fold for every one of the 48 shard jobs), because the
shared loso_fate_v3*.csv files were overwritten by each single-fold shard job
(train.py opens that path with mode "w", not append) -- only the LAST job's
row survived in the CSV. The per-patient .npz files were NOT clobbered (one
file per patient, no shared filename collision), so this recomputes the exact
same compute_metrics() call train.py itself makes, from real saved
predictions -- not fabricated, not re-estimated, just re-derived from the
real artifact that wasn't overwritten.

Usage: python v3_full_loso_summary_fixed.py --results /data/results-v3-full
"""
import argparse
import csv
import glob
import os
import numpy as np
from scipy.stats import wilcoxon

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_metrics


def load(dirpath):
    per_patient = {}
    for f in sorted(glob.glob(os.path.join(dirpath, "preds_*.npz"))):
        pid = os.path.basename(f)[len("preds_"):-len(".npz")]
        d = np.load(f)
        m = compute_metrics(d["y_true"], d["y_score"], threshold=float(d["threshold"]))
        per_patient[pid] = m
    return per_patient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/results-v3-full")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.results, "v3_full_head_to_head.csv")

    full = load(os.path.join(args.results, "none"))
    lean = load(os.path.join(args.results, "lean"))
    print(f"full_v3: {len(full)}/24 patients, lean: {len(lean)}/24 patients")

    common = sorted(set(full) & set(lean))
    rows = []
    per_patient_rows = []
    for name, d in [("full_v3", full), ("lean", lean)]:
        if not d:
            continue
        aucs = np.array([v["auc"] for v in d.values()])
        f1s = np.array([v["f1"] for v in d.values()])
        rows.append({"condition": name, "n": len(d), "mean_auc": f"{aucs.mean():.4f}",
                    "std_auc": f"{aucs.std():.4f}", "mean_f1": f"{f1s.mean():.4f}"})
        for pid, m in sorted(d.items()):
            per_patient_rows.append({"condition": name, "patient": pid,
                                     "auc": f"{m['auc']:.4f}", "f1": f"{m['f1']:.4f}"})

    p_auc = p_f1 = float("nan")
    if len(common) >= 3:
        a_auc = [full[k]["auc"] for k in common]
        b_auc = [lean[k]["auc"] for k in common]
        a_f1 = [full[k]["f1"] for k in common]
        b_f1 = [lean[k]["f1"] for k in common]
        try:
            p_auc = float(wilcoxon(a_auc, b_auc).pvalue)
        except ValueError:
            pass
        try:
            p_f1 = float(wilcoxon(a_f1, b_f1).pvalue)
        except ValueError:
            pass

    print(f"paired over {len(common)} common patients: wilcoxon p(AUC)={p_auc:.4f}  "
         f"p(F1)={p_f1:.4f}")
    for r in rows:
        print(f"  {r['condition']:10s} n={r['n']:2d} mean_auc={r['mean_auc']} "
             f"std={r['std_auc']} mean_f1={r['mean_f1']}")

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["condition", "n", "mean_auc", "std_auc", "mean_f1"])
        w.writeheader()
        w.writerows(rows)
        fh.write(f"\n# paired wilcoxon over {len(common)} common patients: "
                f"p(AUC)={p_auc:.4f} p(F1)={p_f1:.4f}\n")
    print(f"wrote {out}")

    per_patient_out = os.path.join(os.path.dirname(out), "v3_full_head_to_head_per_patient.csv")
    with open(per_patient_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["condition", "patient", "auc", "f1"])
        w.writeheader()
        w.writerows(per_patient_rows)
    print(f"wrote {per_patient_out}")


if __name__ == "__main__":
    main()
