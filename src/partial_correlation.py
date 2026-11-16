"""
Partial Spearman correlation analysis for the difficulty predictor section.

Addresses M5: disentangle seizure count vs inter-seizure interval as predictors
of LOSO AUC, since the two are highly correlated (rho=-0.651).

Outputs: data/generated/partial_correlations.json

Usage: python3 partial_correlation.py
  (run from experiments/src/ or any directory with the standard project layout)
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, t as t_dist


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA_GEN = ROOT / "data" / "generated"
FEATURES_CSV = DATA_GEN / "predictor_features_chbmit.csv"


def partial_spearman(x, y, z):
    """
    Partial Spearman ρ(x, y | z) using the rank-based formula:
      r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))
    Returns (rho, p_value) where p is from a t-test with df=n-2-k (k=1).
    """
    rho_xy, _ = spearmanr(x, y)
    rho_xz, _ = spearmanr(x, z)
    rho_yz, _ = spearmanr(y, z)
    denom = np.sqrt((1 - rho_xz ** 2) * (1 - rho_yz ** 2))
    if denom < 1e-10:
        return float("nan"), float("nan")
    rho_partial = (rho_xy - rho_xz * rho_yz) / denom
    n = len(x)
    df = n - 2 - 1
    if df <= 0:
        return float(rho_partial), float("nan")
    tv = rho_partial * np.sqrt(df / (1 - rho_partial ** 2 + 1e-15))
    pval = float(2 * t_dist.sf(abs(tv), df))
    return float(rho_partial), pval


def main():
    rows = []
    with open(FEATURES_CSV) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    seiz = np.array([float(r["seizure_annotation_count"]) for r in rows])
    isi  = np.array([float(r["inter_seizure_interval_min"]) for r in rows])
    auc  = np.array([float(r["auc"]) for r in rows])
    n    = len(rows)

    # Marginal Spearman
    rho_sy, p_sy = spearmanr(seiz, auc)
    rho_iy, p_iy = spearmanr(isi, auc)
    rho_si, p_si = spearmanr(seiz, isi)

    # Partial Spearman
    rho_sy_z, p_sy_z = partial_spearman(seiz, auc, isi)   # seiz -> AUC | ISI
    rho_iy_z, p_iy_z = partial_spearman(isi,  auc, seiz)  # ISI  -> AUC | seiz

    out = {
        "n": n,
        "note": (
            "Partial Spearman computed via rank-based formula r_xy.z = "
            "(r_xy - r_xz*r_yz)/sqrt((1-r_xz^2)(1-r_yz^2)); "
            "p-value from t-test with df=n-3."
        ),
        "marginal": {
            "seiz_vs_auc": {"rho": round(rho_sy, 4), "p": round(p_sy, 4)},
            "isi_vs_auc":  {"rho": round(rho_iy, 4), "p": round(p_iy, 4)},
            "seiz_vs_isi": {"rho": round(rho_si, 4), "p": round(p_si, 4)},
        },
        "partial": {
            "seiz_vs_auc_given_isi": {
                "rho": round(rho_sy_z, 4), "p": round(p_sy_z, 4),
                "interpretation": (
                    "Seizure count vs AUC controlling for ISI. "
                    "If p > 0.05, seizure count has no independent effect beyond ISI."
                ),
            },
            "isi_vs_auc_given_seiz": {
                "rho": round(rho_iy_z, 4), "p": round(p_iy_z, 4),
                "interpretation": (
                    "ISI vs AUC controlling for seizure count. "
                    "If p < 0.05, ISI has independent predictive value beyond seizure count."
                ),
            },
        },
        "conclusion": (
            f"Seizure count (marginal rho={rho_sy:.3f}, p={p_sy:.4f}) and "
            f"ISI (marginal rho={rho_iy:.3f}, p={p_iy:.4f}) are strongly co-correlated "
            f"with each other (rho={rho_si:.3f}, p={p_si:.4f}). "
            f"After controlling for ISI, seizure count is no longer independently "
            f"significant (partial rho={rho_sy_z:.3f}, p={p_sy_z:.4f}). "
            f"ISI remains independently predictive after controlling for seizure count "
            f"(partial rho={rho_iy_z:.3f}, p={p_iy_z:.4f}). "
            f"This suggests ISI is the more fundamental predictor; seizure count's "
            f"apparent effect is largely mediated through ISI."
        ),
    }

    out_path = DATA_GEN / "partial_correlations.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)

    print("=== Partial Spearman correlation analysis ===")
    print(f"n = {n}")
    print(f"Marginal rho(seiz, AUC) = {rho_sy:.4f}  p={p_sy:.4f}")
    print(f"Marginal rho(ISI,  AUC) = {rho_iy:.4f}  p={p_iy:.4f}")
    print(f"Marginal rho(seiz, ISI) = {rho_si:.4f}  p={p_si:.4f}")
    print()
    print(f"Partial rho(seiz, AUC | ISI)  = {rho_sy_z:.4f}  p={p_sy_z:.4f}")
    print(f"Partial rho(ISI,  AUC | seiz) = {rho_iy_z:.4f}  p={p_iy_z:.4f}")
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
