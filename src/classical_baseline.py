"""
Classical (non-deep) baseline: real hand-engineered spectral + time-domain
features per window, fed to a gradient-boosted-tree classifier, under the
EXACT same strict-LOSO protocol and validation-selected threshold discipline
as train.py -- built to directly test whether the deep frequency-aware
machinery earns its complexity over a much simpler, well-understood approach
at this data scale (see project memory / conversation: current 2025-2026 EEG
literature finds compact deep models often just match classical spectral
decoders on clinical-scale cohorts).

Features per window, per channel, real (not fabricated):
  - relative band power (Welch PSD, delta/theta/alpha/beta/gamma-adj, matching
    FATEv3's 5-band edges) -- same real spectral computation used throughout
    spectral_figures.py this session, not a new invented feature.
  - line length (sum of |successive differences|) -- classic, cheap EEG
    seizure-detection feature (Esteller et al.), captures signal complexity.
  - variance -- classic amplitude-based feature.
18 channels x 5 bands x 1 (power) + 18 x 2 (line length, variance) = 126 features/window.

Usage: python classical_baseline.py --data-root /data/chbmit --cache /data/cache \
    --out /data/results-classical
"""
import os
import glob
import argparse
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import HistGradientBoostingClassifier

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_patient, loso_splits, FS
from metrics import compute_metrics, best_f1_threshold

BANDS = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0), ("gamma_adj", 30.0, 45.0)]


def extract_features(X, chunk=2000):
    """X: (N, 18, T) real cached windows -> (N, 126) real feature matrix.

    Chunked: scipy.signal.welch on the full concatenated training array (which
    can be 10s of thousands of windows) allocates an (N, C, F) PSD array
    simultaneously with the (N, C, T) raw array, which OOM-killed the first
    attempt at this job (32Gi limit) well before training even started.
    Processing in chunks and only keeping the much smaller (chunk, 126)
    feature output bounds peak memory regardless of N.
    """
    N, C, T = X.shape
    feats = np.zeros((N, C * (len(BANDS) + 2)), dtype=np.float32)
    for s in range(0, N, chunk):
        Xc = X[s:s + chunk]
        f, psd = welch(Xc, fs=FS, nperseg=min(256, T), axis=-1)  # (n, C, F)
        total_power = psd.sum(axis=-1) + 1e-8  # (n, C)
        col = 0
        for _, lo, hi in BANDS:
            band_mask = (f >= lo) & (f <= hi)
            band_power = psd[:, :, band_mask].sum(axis=-1)  # (n, C)
            feats[s:s + chunk, col:col + C] = band_power / total_power
            col += C
        line_length = np.abs(np.diff(Xc, axis=-1)).sum(axis=-1)  # (n, C)
        feats[s:s + chunk, col:col + C] = line_length
        col += C
        variance = Xc.var(axis=-1)  # (n, C)
        feats[s:s + chunk, col:col + C] = variance
    return feats


def run_fold(test_pid, train_pids, dirs, cache, seed=42):
    # Convert each patient's raw windows to features immediately and discard
    # the raw array before loading the next patient -- features are ~150x
    # smaller than raw windows (126 floats/window vs 18*1024), so this keeps
    # peak memory bounded by one patient's raw array at a time instead of all
    # 23 concatenated (the actual cause of the OOMKill).
    Ftr_list, ytr_list = [], []
    for p in train_pids:
        Xp, yp = load_patient(dirs[p], cache)
        Ftr_list.append(extract_features(Xp))
        ytr_list.append(yp)
        del Xp
    Ftr_all = np.concatenate(Ftr_list); del Ftr_list
    ytr = np.concatenate(ytr_list); del ytr_list

    Xte, yte = load_patient(dirs[test_pid], cache)
    if len(np.unique(yte)) < 2:
        print(f"  {test_pid}: no positive test windows, skipping")
        return None
    Fte = extract_features(Xte)
    del Xte

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Ftr_all))
    cut = int(0.9 * len(idx))
    tr, va = idx[:cut], idx[cut:]

    Ftr, Fva = Ftr_all[tr], Ftr_all[va]

    n0, n1 = int((ytr[tr] == 0).sum()), int((ytr[tr] == 1).sum())
    sw = np.where(ytr[tr] == 1, n0 / max(n1, 1), 1.0)  # up-weight ictal, same
                                                        # imbalance-handling spirit as FATE's focal loss

    clf = HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                         learning_rate=0.1, random_state=seed)
    clf.fit(Ftr, ytr[tr], sample_weight=sw)

    pva = clf.predict_proba(Fva)[:, 1]
    thr = best_f1_threshold(ytr[va], pva) if len(np.unique(ytr[va])) > 1 else 0.5
    pte = clf.predict_proba(Fte)[:, 1]
    m = compute_metrics(yte, pte, threshold=thr)
    print(f"  {test_pid}: AUC={m['auc']:.4f} Sens={m['sensitivity']:.3f} "
         f"Spec={m['specificity']:.3f} F1={m['f1']:.3f} thr={thr:.3f}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/data/chbmit"))
    ap.add_argument("--cache", default=os.environ.get("CACHE_DIR", "/data/cache"))
    ap.add_argument("--out", default="../../data")
    ap.add_argument("--folds", default="", help="comma list, default all found")
    args = ap.parse_args()

    pdirs = sorted(glob.glob(os.path.join(args.data_root, "chb*")))
    dirs = {os.path.basename(p): p for p in pdirs if os.path.isdir(p)}
    pool = sorted(dirs)
    folds = args.folds.split(",") if args.folds else pool
    folds = [p for p in folds if p in dirs]
    assert pool and folds, f"no patient dirs under {args.data_root}"

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for test_pid in folds:
        train_pids = [p for p in pool if p != test_pid]
        m = run_fold(test_pid, train_pids, dirs, args.cache)
        if m:
            m["patient"] = test_pid
            rows.append(m)

    out_path = os.path.join(args.out, "loso_classical.csv")
    import csv
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["patient", "auc", "sensitivity", "specificity", "f1", "threshold"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
        if rows:
            aucs = [r["auc"] for r in rows]
            w.writerow({"patient": "mean", "auc": f"{np.mean(aucs):.4f}"})
            w.writerow({"patient": "std", "auc": f"{np.std(aucs):.4f}"})
    print(f"\nMean AUC={np.mean([r['auc'] for r in rows]):.4f} +/- {np.std([r['auc'] for r in rows]):.4f}"
         f"  ({len(rows)} folds) -> {out_path}")


if __name__ == "__main__":
    main()
