"""
Replication of the E09 hard-tail predictor finding (N seizures / N ictal
files correlate with LOSO AUC) on two independent targets:
  1. FATE's own real CHB-MIT AUCs (24 patients) -- same dataset, different
     model than the original MSCTEV1-based finding.
  2. Siena's real AUCs (14 patients) -- different dataset, different
     patients, different montage entirely.

Reuses the already-fixed CHB-MIT summary parser (data.py's parse_summary)
and the already-correct Siena annotation parser (data_siena.py's
parse_annotation) -- no new parsing logic, avoiding any repeat of the
numbered-seizure bug fixed in the original E09 script.
"""
import csv
import glob
import os
import sys
import json
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import parse_summary
from data_siena import parse_annotation

CHBMIT_ROOT = "/data/chbmit"
SIENA_ROOT = "/data/siena"


def load_real_auc(csv_path):
    aucs = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["patient"] in ("mean", "std"):
                continue
            aucs[r["patient"]] = float(r["auc"])
    return aucs


def chbmit_stats(pid):
    summary_paths = glob.glob(os.path.join(CHBMIT_ROOT, pid, "*summary.txt"))
    if not summary_paths:
        return None
    seiz = parse_summary(summary_paths[0])  # {edf_name: [(start,end),...]}
    n_seizures = sum(len(v) for v in seiz.values())
    n_ictal_files = sum(1 for v in seiz.values() if len(v) > 0)
    return n_seizures, n_ictal_files


def siena_stats(pid):
    ann_paths = glob.glob(os.path.join(SIENA_ROOT, pid, "Seizures-list-*.txt"))
    if not ann_paths:
        return None
    seiz = parse_annotation(ann_paths[0])
    n_seizures = sum(len(v) for v in seiz.values())
    n_ictal_files = sum(1 for v in seiz.values() if len(v) > 0)
    return n_seizures, n_ictal_files


def run(name, patients, stats_fn, real_auc):
    rows = []
    for pid in patients:
        s = stats_fn(pid)
        if s is None or pid not in real_auc:
            print(f"  {pid}: skipped (no data)")
            continue
        n_seiz, n_files = s
        rows.append({"patient": pid, "n_seizures": n_seiz,
                     "n_ictal_files": n_files, "auc": real_auc[pid]})
        print(f"  {pid}: n_seizures={n_seiz} n_ictal_files={n_files} auc={real_auc[pid]:.4f}")

    aucs = np.array([r["auc"] for r in rows])
    result = {"dataset": name, "n": len(rows), "rows": rows}
    for feat in ["n_seizures", "n_ictal_files"]:
        x = np.array([r[feat] for r in rows])
        rho, p = spearmanr(x, aucs)
        result[feat] = {"rho": float(rho), "p": float(p)}
        print(f"  [{name}] {feat}: rho={rho:+.4f} p={p:.4f} "
             f"{'SIGNIFICANT' if p < 0.05 else ''}")
    return result


def main():
    real_fate_auc = load_real_auc("../results/loso_fate.csv")
    real_siena_auc = load_real_auc("../data/results-siena/loso_siena_summary.csv")

    chbmit_patients = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CHBMIT_ROOT, "chb*")))
    siena_patients = sorted(os.path.basename(p) for p in glob.glob(os.path.join(SIENA_ROOT, "PN*")))

    print("=== CHB-MIT (FATE real AUCs, n=24) ===")
    r1 = run("chbmit_fate", chbmit_patients, chbmit_stats, real_fate_auc)

    print("\n=== Siena (real AUCs, n=14) ===")
    r2 = run("siena", siena_patients, siena_stats, real_siena_auc)

    out = {"chbmit_fate": r1, "siena": r2}
    with open("/data/hardtail_replication_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote /data/hardtail_replication_results.json")


if __name__ == "__main__":
    main()
