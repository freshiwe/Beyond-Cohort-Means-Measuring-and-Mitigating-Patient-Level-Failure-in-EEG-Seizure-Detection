"""
Expanded, principled feature set for the preemptive hardness predictor
(replaces E05/E07-A's 3-ad-hoc-feature, single-correlation approach with a
richer feature set meant to be fed into a properly validated nested-LOPO
classifier, see hardness_classifier.py).

All features are computed leak-free: from calibration-window data only
(first ~15 min, unlabeled, via calib_split.calib_windows -- same window
already used by risk_estimator.py / tta_adabn.py this session), with any
"distance from the pool" style feature computed within-dataset,
leave-one-patient-out (the held-out patient never contributes to its own
comparison statistic).

Features:
  1-3. spectral_shift, mean_ictal_prob, score_entropy -- REUSED, not
       recomputed: read directly from risk_estimator.py's already-real,
       already-pushed per-patient CSVs (experiments/results/
       risk_estimator_per_patient[_siena].csv). Recomputing would risk
       divergence from the numbers already reported in TEAM_PROCESS.md.
  4. line_length_mean, 5. variance_mean -- reused from
     classical_baseline.py's extract_features() (real Welch/line-length/
     variance features already built and validated this session), averaged
     across the calibration window's channels and windows.
  6. kurtosis_mean -- new, classic EEG amplitude-distribution feature not
     previously computed here, same aggregation.
  7. embed_dist -- Euclidean distance between this patient's mean FATE
     penultimate-layer embedding (the 128-dim vector immediately before
     model.head, extracted via a manual replay of FATE.forward()'s early
     stages -- NOT a model.py edit, this file owns the extra computation)
     over calibration windows, and the same-dataset pool's mean embedding
     (leave-one-out). Richer than the existing mean_ictal_prob/score_entropy,
     which only see the final scalar probability.
  8. bn_shift -- L2 norm of the change in every BatchNorm layer's
     running_mean/running_var after AdaBN-adapting to this patient's
     calibration window (reuses tta_adabn.py's own adabn_update() and
     set_bn_train() exactly, adds only the before/after stat snapshot).

Usage: python hardness_features.py --data-root /data/chbmit --dataset chbmit \
    --ckpt-dir /data/results-ckpt \
    --existing-csv /path/to/risk_estimator_per_patient.csv \
    --out /data/results-hardness/hardness_features_chbmit.csv
"""
import argparse
import copy
import csv
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kurtosis as _kurtosis

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib_split import calib_windows
from model import FATE
from classical_baseline import extract_features as classical_extract_features, BANDS as CLASSICAL_BANDS
from tta_adabn import set_bn_train, adabn_update


def _fs_for(dataset):
    if dataset == "siena":
        import data_siena as _m
    else:
        import data as _m
    return _m.FS


@torch.no_grad()
def embed_windows(model, X, device, batch=256):
    """Replay FATE.forward() up to (but not including) model.head -- the
    128-dim pooled embedding z. Kept here (not a model.py edit) since it's a
    read-only replay of forward()'s own ops, not a model architecture change."""
    model.eval()
    zs = []
    for s in range(0, len(X), batch):
        xb = torch.from_numpy(X[s:s + batch]).to(device)
        bands = model.decomp(xb)
        B, nb, C, T = bands.shape
        feats = model.extractor(bands.reshape(B * nb, C, T))
        feats = feats.reshape(B, nb, feats.size(1), feats.size(2))
        if model.faa is None:
            A = feats
        else:
            A, _ = model.faa(feats)
        Fz = model.fusion(A)
        Z = model.encoder(Fz)
        w = torch.softmax(model.pool(Z), dim=1)
        z = (w * Z).sum(dim=1)
        zs.append(z.cpu().numpy())
    return np.concatenate(zs, axis=0) if zs else np.zeros((0, 128), dtype=np.float32)


def classical_signal_features(X):
    """line_length_mean, variance_mean (reused columns from
    classical_baseline.extract_features), kurtosis_mean (new)."""
    if len(X) == 0:
        return float("nan"), float("nan"), float("nan")
    feats = classical_extract_features(X)  # (N, 18*(5+2)) = (N, 126)
    C = X.shape[1]
    n_bands = len(CLASSICAL_BANDS)
    ll_start = C * n_bands
    var_start = ll_start + C
    line_length_mean = float(feats[:, ll_start:ll_start + C].mean())
    variance_mean = float(feats[:, var_start:var_start + C].mean())
    kurt = _kurtosis(X, axis=-1, fisher=True, bias=False)  # (N, C)
    kurtosis_mean = float(np.nanmean(kurt))
    return line_length_mean, variance_mean, kurtosis_mean


def bn_stat_snapshot(model):
    snap = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            snap.append((m.running_mean.clone(), m.running_var.clone()))
    return snap


def bn_shift_magnitude(before, after):
    total = 0.0
    for (rm0, rv0), (rm1, rv1) in zip(before, after):
        total += float(((rm1 - rm0) ** 2).sum()) + float(((rv1 - rv0) ** 2).sum())
    return total ** 0.5


def load_existing_csv(path):
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows[r["patient"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/data/chbmit"))
    ap.add_argument("--dataset", default="chbmit", choices=["chbmit", "siena"])
    ap.add_argument("--ckpt-dir", default="/data/results-ckpt")
    ap.add_argument("--existing-csv", required=True,
                    help="risk_estimator.py's per-patient CSV for this dataset "
                         "(spectral_shift/mean_ictal_prob/score_entropy/real_auc reused, not recomputed)")
    ap.add_argument("--out", default="/data/results-hardness/hardness_features.csv")
    ap.add_argument("--patients", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    subj_glob = "PN*" if args.dataset == "siena" else "chb*"
    pdirs = sorted(glob.glob(os.path.join(args.data_root, subj_glob)))
    dirs = {os.path.basename(p): p for p in pdirs if os.path.isdir(p)}
    pool = args.patients.split(",") if args.patients else sorted(dirs)
    pool = [p for p in pool if p in dirs]
    assert pool, f"no patient dirs under {args.data_root}"

    existing = load_existing_csv(args.existing_csv)

    print(f"loading calibration windows + computing new features for {len(pool)} patients...")
    calib_cache = {}
    embed_cache = {}
    classical_cache = {}
    for pid in pool:
        if pid not in existing:
            print(f"  {pid}: not in {args.existing_csv}, skipping")
            continue
        Xc, cov = calib_windows(dirs[pid], dataset=args.dataset)
        calib_cache[pid] = Xc
        ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{pid}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  {pid}: no checkpoint at {ckpt_path}, skipping model features")
            continue
        model = FATE(ablation="none").to(args.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        emb = embed_windows(model, Xc, args.device)
        embed_cache[pid] = emb.mean(axis=0) if len(emb) else np.full(128, np.nan)
        classical_cache[pid] = classical_signal_features(Xc)
        del model
        print(f"  {pid}: calib={len(Xc)} windows (~{cov:.0f}s), embedding+classical computed")

    rows = []
    for pid in pool:
        if pid not in embed_cache:
            continue
        e = existing[pid]
        other = [p for p in pool if p != pid and p in embed_cache]
        pool_mean_embed = np.nanmean(np.stack([embed_cache[p] for p in other]), axis=0)
        embed_dist = float(np.linalg.norm(embed_cache[pid] - pool_mean_embed))

        line_length_mean, variance_mean, kurtosis_mean = classical_cache[pid]

        # BN-adaptation shift magnitude: reuse tta_adabn's own adabn_update exactly.
        ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{pid}.pt")
        model = FATE(ablation="none").to(args.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        before = bn_stat_snapshot(model)
        Xc = calib_cache[pid]
        if len(Xc) > 0:
            adabn_update(model, Xc, device=args.device)
        after = bn_stat_snapshot(model)
        bn_shift = bn_shift_magnitude(before, after)
        del model

        rows.append({
            "patient": pid, "dataset": args.dataset,
            "real_auc": float(e["real_auc"]),
            "spectral_shift": float(e["spectral_shift"]),
            "mean_ictal_prob": float(e["mean_ictal_prob"]),
            "score_entropy": float(e["score_entropy"]),
            "line_length_mean": line_length_mean,
            "variance_mean": variance_mean,
            "kurtosis_mean": kurtosis_mean,
            "embed_dist": embed_dist,
            "bn_shift": bn_shift,
        })
        print(f"  {pid}: embed_dist={embed_dist:.4f} bn_shift={bn_shift:.6f} "
             f"line_length={line_length_mean:.2f} variance={variance_mean:.4f} kurtosis={kurtosis_mean:.3f}")

    fieldnames = ["patient", "dataset", "real_auc", "spectral_shift", "mean_ictal_prob",
                 "score_entropy", "line_length_mean", "variance_mean", "kurtosis_mean",
                 "embed_dist", "bn_shift"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} patients)")


if __name__ == "__main__":
    main()
