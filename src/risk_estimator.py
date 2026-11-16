"""
Experiment A: can a small set of real, unlabeled, pre-registered features
computed from a new patient's first ~15 minutes of EEG predict how well the
frozen v2 FATE detector will do on the rest of their recording?

Deliberately NOT a fitted regressor -- with only 24 patients, fitting a
multi-feature model to predict AUC would almost certainly just fit noise.
Instead: 3 small, named, pre-registered candidate features, each correlated
(Spearman) against the real per-patient AUC from results/loso_fate.csv,
independently. No feature selection after seeing the correlations.

Features (all computed from X_calib only, labels never touched):
  1. spectral_shift: Euclidean distance between this patient's calibration
     mean relative band-power vector (Welch, same 5 bands as
     classical_baseline.py) and the mean of the OTHER 23 patients'
     calibration band-power vectors (computed within the same LOSO fold --
     the held-out patient never contributes to its own comparison mean).
  2. mean_ictal_prob: mean predicted ictal probability over calibration
     windows from that patient's own frozen checkpoint. All calibration
     windows are background signal in expectation; a well-calibrated model
     should score them low. Elevated scores on known-background data are a
     candidate sign the model doesn't "understand" this patient's baseline.
  3. score_entropy: mean binary prediction entropy over the same calibration
     scores -- a second, related but distinct uncertainty signal (a model
     that is uniformly unsure, not just biased high, also entropy-high).

Usage: python risk_estimator.py --data-root /data/chbmit --ckpt-dir /data/results-ckpt \
    --out /data/results-risk-tta
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
from scipy.signal import welch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib_split import calib_windows
from model import FATE


def _fs_for(dataset):
    if dataset == "siena":
        import data_siena as _m
    else:
        import data as _m
    return _m.FS

BANDS = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0), ("gamma_adj", 30.0, 45.0)]


def band_power_vector(X, fs, chunk=1000):
    """X: (N, C, T) -> (n_bands,) relative band power, averaged over windows
    and channels. Same Welch computation as classical_baseline.py."""
    if len(X) == 0:
        return np.full(len(BANDS), np.nan)
    accum = np.zeros(len(BANDS))
    n = 0
    for s in range(0, len(X), chunk):
        Xc = X[s:s + chunk]
        f, psd = welch(Xc, fs=fs, nperseg=min(256, Xc.shape[-1]), axis=-1)  # (n,C,F)
        total = psd.sum(axis=-1) + 1e-8  # (n,C)
        for bi, (_, lo, hi) in enumerate(BANDS):
            mask = (f >= lo) & (f <= hi)
            rel = (psd[:, :, mask].sum(axis=-1) / total).mean()  # scalar, avg over n,C
            accum[bi] += rel * len(Xc)
        n += len(Xc)
    return accum / max(n, 1)


@torch.no_grad()
def model_uncertainty_features(model, X_calib, device, batch=256):
    if len(X_calib) == 0:
        return float("nan"), float("nan")
    model.eval()
    probs = []
    for s in range(0, len(X_calib), batch):
        xb = torch.from_numpy(X_calib[s:s + batch]).to(device)
        logits = model(xb)
        p = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        probs.append(p)
    p = np.concatenate(probs)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return float(p.mean()), float(entropy.mean())


def load_real_auc(loso_csv):
    aucs = {}
    with open(loso_csv) as fh:
        for r in csv.DictReader(fh):
            if r["patient"] in ("mean", "std"):
                continue
            aucs[r["patient"]] = float(r["auc"])
    return aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/data/chbmit"))
    ap.add_argument("--dataset", default="chbmit", choices=["chbmit", "siena"])
    ap.add_argument("--ckpt-dir", default="/data/results-ckpt")
    # NOT /data/results-ckpt/loso_fate.csv -- that shared PVC path was clobbered
    # by the 24 single-patient ckpt-shard jobs each writing with mode "w" (same
    # overwrite bug found and fixed for v3_full_loso_summary.py this session);
    # only chb01's row survived there. The real 24-row aggregate lives in the
    # repo, written once from a proper merge, not per-shard.
    ap.add_argument("--loso-csv", default="../results/loso_fate.csv")
    ap.add_argument("--out", default="/data/results-risk-tta")
    ap.add_argument("--patients", default="", help="comma list, default all found")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    subj_glob = "PN*" if args.dataset == "siena" else "chb*"
    pdirs = sorted(glob.glob(os.path.join(args.data_root, subj_glob)))
    dirs = {os.path.basename(p): p for p in pdirs if os.path.isdir(p)}
    fs = _fs_for(args.dataset)
    pool = args.patients.split(",") if args.patients else sorted(dirs)
    pool = [p for p in pool if p in dirs]
    assert pool, f"no patient dirs under {args.data_root}"

    real_auc = load_real_auc(args.loso_csv)

    # Only the small ~15-min calibration window is ever loaded/retained here --
    # NOT the rest of each patient's (multi-hour) recording -- so this is safe
    # to hold for all 24 patients at once. Materializing full eval sets for all
    # 24 patients simultaneously OOM-killed an earlier version of this script.
    print(f"loading calibration windows + band power for {len(pool)} patients...")
    calib_cache, bandpower_cache = {}, {}
    for pid in pool:
        Xc, cov = calib_windows(dirs[pid], dataset=args.dataset)
        calib_cache[pid] = (Xc, cov)
        bandpower_cache[pid] = band_power_vector(Xc, fs)
        print(f"  {pid}: calib={len(Xc)} windows (~{cov:.0f}s)")

    rows = []
    for pid in pool:
        Xc, cov = calib_cache[pid]
        if pid not in real_auc:
            print(f"  {pid}: no real AUC in {args.loso_csv}, skipping")
            continue

        other = [p for p in pool if p != pid]
        pool_mean_bp = np.nanmean(np.stack([bandpower_cache[p] for p in other]), axis=0)
        spectral_shift = float(np.linalg.norm(bandpower_cache[pid] - pool_mean_bp))

        ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{pid}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  {pid}: no checkpoint at {ckpt_path}, skipping model features")
            mean_ictal_prob = score_entropy = float("nan")
        else:
            model = FATE(ablation="none").to(args.device)
            model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
            mean_ictal_prob, score_entropy = model_uncertainty_features(model, Xc, args.device)
            del model

        rows.append({"patient": pid, "real_auc": real_auc[pid],
                     "spectral_shift": spectral_shift,
                     "mean_ictal_prob": mean_ictal_prob,
                     "score_entropy": score_entropy,
                     "calib_windows": len(Xc), "calib_seconds": cov})
        print(f"  {pid}: real_auc={real_auc[pid]:.4f} spectral_shift={spectral_shift:.4f} "
             f"mean_ictal_prob={mean_ictal_prob:.4f} score_entropy={score_entropy:.4f}")

    per_patient_path = os.path.join(args.out, "risk_estimator_per_patient.csv")
    with open(per_patient_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["patient", "real_auc", "spectral_shift",
                                           "mean_ictal_prob", "score_entropy",
                                           "calib_windows", "calib_seconds"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {per_patient_path}")

    aucs = np.array([r["real_auc"] for r in rows])
    corr_rows = []
    feat_names = ["spectral_shift", "mean_ictal_prob", "score_entropy"]
    for feat in feat_names:
        vals = np.array([r[feat] for r in rows])
        valid = ~np.isnan(vals)
        if valid.sum() >= 3:
            rho, p = spearmanr(vals[valid], aucs[valid])
        else:
            rho, p = float("nan"), float("nan")
        corr_rows.append({"feature": feat, "n": int(valid.sum()),
                          "spearman_rho": rho, "p_value": p})
        print(f"  corr({feat}, real_auc): rho={rho:.4f} p={p:.4f} (n={int(valid.sum())})")

    # unweighted combined z-score, NOT fit/tuned -- explicitly just an average
    # of z-scored features, reported for completeness alongside, not instead
    # of, the individual feature correlations above.
    z_stack = []
    for feat in feat_names:
        vals = np.array([r[feat] for r in rows], dtype=float)
        mu, sd = np.nanmean(vals), np.nanstd(vals)
        z = (vals - mu) / (sd + 1e-8)
        z_stack.append(z)
    combined = np.nanmean(np.stack(z_stack), axis=0)
    valid = ~np.isnan(combined)
    if valid.sum() >= 3:
        rho, p = spearmanr(combined[valid], aucs[valid])
    else:
        rho, p = float("nan"), float("nan")
    corr_rows.append({"feature": "combined_unweighted_zscore", "n": int(valid.sum()),
                      "spearman_rho": rho, "p_value": p})
    print(f"  corr(combined_unweighted_zscore, real_auc): rho={rho:.4f} p={p:.4f}")

    corr_path = os.path.join(args.out, "risk_estimator_correlations.csv")
    with open(corr_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["feature", "n", "spearman_rho", "p_value"])
        w.writeheader()
        w.writerows(corr_rows)
    print(f"wrote {corr_path}")


if __name__ == "__main__":
    main()
