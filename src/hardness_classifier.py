"""
Preemptive hardness classifier: given only the expanded, leak-free feature
set from hardness_features.py (computed from a new patient's unlabeled
calibration window), predict whether that patient will be in the bottom
quartile (real LOSO AUC) for the frozen FATE detector -- before any seizure
of theirs is scored.

Deliberately more rigorous than E05/E07-A's single-feature Spearman
correlations:
  - combines CHB-MIT (n=24) + Siena (n=14) = 38 patients, with dataset as an
    explicit covariate (not a "real" feature -- included so the classifier
    can account for the two cohorts' different baselines, e.g. pediatric vs
    adult, rather than confusing cohort membership for hardness itself).
  - "hard" label = bottom quartile BY REAL AUC, computed WITHIN each dataset
    separately (so Siena's generally-lower baseline AUC doesn't just make
    every Siena patient "hard" relative to CHB-MIT).
  - a small, L2-regularized logistic regression -- deliberately not a
    complex model, n=38 cannot support one.
  - strict nested leave-one-patient-out: for each held-out patient, the
    classifier is fit ONLY on the other 37 (StandardScaler fit inside the
    fold too, not globally -- same leakage discipline this project has
    caught and fixed twice already this session for other scripts).
  - permutation test for real significance: shuffle the hard/not-hard labels
    (within each dataset, to preserve the same quartile-per-dataset
    structure) and rerun the ENTIRE nested LOPO procedure many times: the
    real ranking AUC is compared against this null distribution, not a
    single feature's Spearman p-value.

Usage: python hardness_classifier.py --in-chbmit /data/.../hardness_features_chbmit.csv \
    --in-siena /data/.../hardness_features_siena.csv --out /data/results-hardness \
    --n-permutations 500
"""
import argparse
import csv
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

FEATURES = ["spectral_shift", "mean_ictal_prob", "score_entropy",
           "line_length_mean", "variance_mean", "kurtosis_mean",
           "embed_dist", "bn_shift"]


def load_rows(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    return rows


def build_matrix(rows):
    """Returns X (n, len(FEATURES)+1 [dataset_is_siena]), y_auc (n,), pids, datasets."""
    pids = [r["patient"] for r in rows]
    datasets = [r["dataset"] for r in rows]
    y_auc = np.array([float(r["real_auc"]) for r in rows])
    X = np.zeros((len(rows), len(FEATURES) + 1))
    for i, r in enumerate(rows):
        for j, f in enumerate(FEATURES):
            v = r[f]
            X[i, j] = float(v) if v not in ("", "nan") else np.nan
        X[i, -1] = 1.0 if r["dataset"] == "siena" else 0.0
    return X, y_auc, pids, datasets


def hard_labels(y_auc, datasets):
    """Bottom quartile by real AUC, computed within each dataset separately."""
    labels = np.zeros(len(y_auc), dtype=int)
    for ds in set(datasets):
        idx = [i for i, d in enumerate(datasets) if d == ds]
        vals = y_auc[idx]
        thresh = np.quantile(vals, 0.25)
        for i in idx:
            labels[i] = 1 if y_auc[i] <= thresh else 0
    return labels


def nested_lopo_scores(X, labels):
    """Strict leave-one-patient-out: fit scaler+model on the other n-1 each
    time, predict the held-out patient's hard-probability. Median-impute NaN
    features using ONLY the training fold's median (leak-free)."""
    n = len(labels)
    scores = np.zeros(n)
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        Xtr, Xte = X[train_idx].copy(), X[i:i + 1].copy()
        ytr = labels[train_idx]
        # leak-free median imputation: fold-only statistic
        med = np.nanmedian(Xtr, axis=0)
        for col in range(Xtr.shape[1]):
            Xtr[np.isnan(Xtr[:, col]), col] = med[col]
            Xte[np.isnan(Xte[:, col]), col] = med[col]
        if len(np.unique(ytr)) < 2:
            scores[i] = 0.5
            continue
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        clf = LogisticRegression(penalty="l2", C=0.5, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr_s, ytr)
        scores[i] = clf.predict_proba(Xte_s)[0, 1]
    return scores


def ranking_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def permutation_test(X, labels, datasets, real_auc_score, n_perm, seed=42):
    rng = np.random.default_rng(seed)
    ds_arr = np.array(datasets)
    null_aucs = []
    for _ in range(n_perm):
        perm_labels = labels.copy()
        for ds in set(datasets):
            idx = np.where(ds_arr == ds)[0]
            perm_labels[idx] = rng.permutation(labels[idx])
        perm_scores = nested_lopo_scores(X, perm_labels)
        null_aucs.append(ranking_auc(perm_labels, perm_scores))
    null_aucs = np.array([a for a in null_aucs if not np.isnan(a)])
    p = float((null_aucs >= real_auc_score).sum() + 1) / (len(null_aucs) + 1)
    return p, null_aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-chbmit", required=True)
    ap.add_argument("--in-siena", required=True)
    ap.add_argument("--out", default="/data/results-hardness")
    ap.add_argument("--n-permutations", type=int, default=500)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = load_rows([args.in_chbmit, args.in_siena])
    X, y_auc, pids, datasets = build_matrix(rows)
    labels = hard_labels(y_auc, datasets)
    print(f"n={len(rows)} ({datasets.count('chbmit')} CHB-MIT, {datasets.count('siena')} Siena), "
         f"n_hard={labels.sum()}")

    scores = nested_lopo_scores(X, labels)
    real_auc_score = ranking_auc(labels, scores)
    print(f"Real nested-LOPO ranking AUC: {real_auc_score:.4f}")

    print(f"running {args.n_permutations} permutations for significance...")
    p_value, null_aucs = permutation_test(X, labels, datasets, real_auc_score, args.n_permutations)
    print(f"Permutation p-value: {p_value:.4f} (null mean={null_aucs.mean():.4f}, "
         f"null std={null_aucs.std():.4f}, n_perm_valid={len(null_aucs)})")

    # explicit hard-tail flagging check
    per_patient = []
    for i, pid in enumerate(pids):
        per_patient.append({"patient": pid, "dataset": datasets[i], "real_auc": float(y_auc[i]),
                            "is_real_hard": int(labels[i]), "predicted_risk": float(scores[i])})
    per_patient.sort(key=lambda r: -r["predicted_risk"])
    top_quartile_n = max(1, len(pids) // 4)
    flagged = per_patient[:top_quartile_n]
    flagged_pids = {r["patient"] for r in flagged}
    real_hard_pids = {r["patient"] for r in per_patient if r["is_real_hard"]}
    overlap = flagged_pids & real_hard_pids
    print(f"\nTop-risk-quartile flagged ({top_quartile_n}): {sorted(flagged_pids)}")
    print(f"Real bottom-quartile-hard patients: {sorted(real_hard_pids)}")
    print(f"Overlap: {sorted(overlap)} ({len(overlap)}/{len(real_hard_pids)} real-hard patients flagged)")

    with open(os.path.join(args.out, "hardness_classifier_per_patient.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["patient", "dataset", "real_auc", "is_real_hard", "predicted_risk"])
        w.writeheader()
        w.writerows(sorted(per_patient, key=lambda r: r["patient"]))

    summary = {
        "n": len(rows), "n_chbmit": datasets.count("chbmit"), "n_siena": datasets.count("siena"),
        "n_hard": int(labels.sum()), "ranking_auc": real_auc_score, "permutation_p_value": p_value,
        "n_permutations": len(null_aucs), "null_mean": float(null_aucs.mean()), "null_std": float(null_aucs.std()),
        "flagged_top_quartile": sorted(flagged_pids), "real_hard": sorted(real_hard_pids),
        "overlap": sorted(overlap), "features_used": FEATURES,
    }
    with open(os.path.join(args.out, "hardness_classifier_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {args.out}/hardness_classifier_per_patient.csv and hardness_classifier_summary.json")

    # Refit on ALL 38 patients (no held-out patient) -- this is the "final"
    # classifier a downstream user (e.g. scoring a third architecture's
    # patients) should reuse via score_new_patients() below, NOT any of the
    # 38 per-fold models above (those exist only to produce leak-free scores
    # for validating THIS classifier on THIS data).
    med_all = np.nanmedian(X, axis=0)
    X_imputed = X.copy()
    for col in range(X.shape[1]):
        X_imputed[np.isnan(X_imputed[:, col]), col] = med_all[col]
    scaler_final = StandardScaler().fit(X_imputed)
    clf_final = LogisticRegression(penalty="l2", C=0.5, max_iter=2000, class_weight="balanced")
    clf_final.fit(scaler_final.transform(X_imputed), labels)
    import pickle
    with open(os.path.join(args.out, "hardness_classifier_final.pkl"), "wb") as fh:
        pickle.dump({"scaler": scaler_final, "clf": clf_final, "median": med_all,
                    "features": FEATURES}, fh)
    print(f"wrote {args.out}/hardness_classifier_final.pkl "
         f"(final classifier refit on all {len(rows)} patients -- "
         f"for scoring NEW patients from a different architecture, NOT for "
         f"re-validating on these same 38; use score_new_patients() for that)")


def score_new_patients(pkl_path, feature_rows):
    """Score new patients (e.g. from a different architecture) using an
    ALREADY-FITTED classifier -- no refitting. feature_rows: list of dicts
    with the same FEATURES keys (dataset covariate not required -- defaults
    to 0/CHB-MIT-like if absent, since a third architecture's patients won't
    have a natural "dataset" label in this scheme; caller should think about
    whether that default is appropriate for their case)."""
    import pickle
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    X = np.zeros((len(feature_rows), len(d["features"]) + 1))
    for i, r in enumerate(feature_rows):
        for j, f in enumerate(d["features"]):
            v = r.get(f, np.nan)
            X[i, j] = float(v) if v not in (None, "", "nan") else np.nan
        X[i, -1] = float(r.get("dataset_is_siena", 0.0))
    for col in range(X.shape[1]):
        X[np.isnan(X[:, col]), col] = d["median"][col]
    Xs = d["scaler"].transform(X)
    return d["clf"].predict_proba(Xs)[:, 1]


if __name__ == "__main__":
    main()
