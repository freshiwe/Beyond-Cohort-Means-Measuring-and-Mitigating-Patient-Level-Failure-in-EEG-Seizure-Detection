"""
Experiment B: does adapting BatchNorm running statistics to a new patient's
unlabeled calibration window (same 15-minute window Experiment A uses)
improve detection on the rest of their recording -- and does the benefit,
if any, concentrate in the patients Experiment A calls high-risk?

AdaBN only (no gradient updates): forward-pass X_calib through the frozen
v2 FATE checkpoint with ONLY its BatchNorm modules in train() mode (so their
running_mean/running_var update via the standard momentum rule as a side
effect of forward()); everything else (including Dropout) stays in eval()
mode. No optimizer, no backward pass, no label ever touched.

AUC_before and AUC_after are both computed on X_eval (the same reduced
evaluation set, from calib_split.calib_eval_split) -- NOT the original
full-recording loso_fate.csv number, since carving out 15 min of calibration
changes what "before" means; this isolates the pure effect of BN adaptation.

Usage: python tta_adabn.py --data-root /data/chbmit --ckpt-dir /data/results-ckpt \
    --risk-csv /data/results-risk-tta/risk_estimator_per_patient.csv \
    --out /data/results-risk-tta
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib_split import calib_windows, eval_stream, CALIB_SECONDS
from model import FATE
from metrics import compute_metrics


def set_bn_train(model, train):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.train(train)


@torch.no_grad()
def score_streaming(model, patient_dir, calib_seconds, batch=256, device="cpu", dataset="chbmit"):
    """Score the evaluation portion of a patient's recording one EDF file at a
    time (via calib_split.eval_stream), never materializing the whole
    (multi-hour) recording in memory at once -- only (score, label) pairs are
    accumulated, not raw windows. Returns (y_eval, p_eval) concatenated."""
    model.eval()
    ys, ps = [], []
    for X_chunk, y_chunk in eval_stream(patient_dir, calib_seconds, dataset=dataset):
        for s in range(0, len(X_chunk), batch):
            xb = torch.from_numpy(X_chunk[s:s + batch]).to(device)
            logits = model(xb)
            ps.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        ys.append(y_chunk)
    if not ys:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def adabn_update(model, X_calib, batch=256, device="cpu"):
    model.eval()               # dropout etc stay off
    set_bn_train(model, True)  # BN layers only: forward passes update running stats
    for s in range(0, len(X_calib), batch):
        xb = torch.from_numpy(X_calib[s:s + batch]).to(device)
        model(xb)
    set_bn_train(model, False)
    model.eval()


def load_real_auc(loso_csv):
    aucs = {}
    with open(loso_csv) as fh:
        for r in csv.DictReader(fh):
            if r["patient"] in ("mean", "std"):
                continue
            aucs[r["patient"]] = float(r["auc"])
    return aucs


def load_risk_scores(risk_csv):
    """Reproduce risk_estimator.py's unweighted combined z-score from its saved
    per-patient raw features -- not re-fit, same formula, so this script only
    needs risk_estimator.py's output CSV as input, no shared in-memory state."""
    rows = []
    with open(risk_csv) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    feats = ["spectral_shift", "mean_ictal_prob", "score_entropy"]
    z_sum = np.zeros(len(rows))
    n_valid_feats = 0
    for f in feats:
        v = np.array([float(r[f]) if r[f] not in ("", "nan") else np.nan for r in rows])
        if np.sum(~np.isnan(v)) < 2:
            continue
        mu, sd = np.nanmean(v), np.nanstd(v)
        z_sum = z_sum + np.nan_to_num((v - mu) / (sd + 1e-8), nan=0.0)
        n_valid_feats += 1
    combined = z_sum / max(n_valid_feats, 1)
    return {r["patient"]: float(c) for r, c in zip(rows, combined)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/data/chbmit"))
    ap.add_argument("--dataset", default="chbmit", choices=["chbmit", "siena"])
    ap.add_argument("--ckpt-dir", default="/data/results-ckpt")
    # See risk_estimator.py: the PVC path was clobbered by the sharded jobs,
    # this repo-committed file is the real 24-row aggregate.
    ap.add_argument("--loso-csv", default="../results/loso_fate.csv")
    ap.add_argument("--risk-csv", default="/data/results-risk-tta/risk_estimator_per_patient.csv")
    ap.add_argument("--out", default="/data/results-risk-tta")
    ap.add_argument("--patients", default="", help="comma list, default all found")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    subj_glob = "PN*" if args.dataset == "siena" else "chb*"
    pdirs = sorted(glob.glob(os.path.join(args.data_root, subj_glob)))
    dirs = {os.path.basename(p): p for p in pdirs if os.path.isdir(p)}
    pool = args.patients.split(",") if args.patients else sorted(dirs)
    pool = [p for p in pool if p in dirs]
    assert pool, f"no patient dirs under {args.data_root}"

    real_auc = load_real_auc(args.loso_csv)
    risk_scores = load_risk_scores(args.risk_csv) if os.path.exists(args.risk_csv) else {}
    if not risk_scores:
        print(f"  WARNING: no risk scores loaded from {args.risk_csv} -- "
             f"run risk_estimator.py first for the risk-quartile breakdown.")

    rows = []
    for pid in pool:
        ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{pid}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  {pid}: no checkpoint at {ckpt_path}, skipping")
            continue
        Xc, cov = calib_windows(dirs[pid], dataset=args.dataset)

        model = FATE(ablation="none").to(args.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))

        # First streaming pass: score the eval portion with the frozen model
        # ("before"). Never materializes the full (multi-hour) recording.
        ye, p_before = score_streaming(model, dirs[pid], CALIB_SECONDS, device=args.device, dataset=args.dataset)
        if len(ye) == 0 or len(np.unique(ye)) < 2:
            print(f"  {pid}: eval set has <2 classes ({len(ye)} windows), skipping")
            del model
            continue
        m_before = compute_metrics(ye, p_before)

        if len(Xc) > 0:
            adabn_update(model, Xc, device=args.device)
        else:
            print(f"  {pid}: 0 calibration windows, AUC_after == AUC_before (no-op)")
        # Second streaming pass: score the SAME eval portion with the BN-adapted
        # model ("after"). eval_stream() re-reads from disk rather than caching
        # in memory -- CPU-cheap relative to the memory this saves.
        ye2, p_after = score_streaming(model, dirs[pid], CALIB_SECONDS, device=args.device, dataset=args.dataset)
        m_after = compute_metrics(ye2, p_after)

        rscore = risk_scores.get(pid, float("nan"))
        row = {"patient": pid, "risk_score": rscore, "n_calib": len(Xc), "n_eval": len(ye),
              "auc_before": m_before["auc"], "auc_after": m_after["auc"],
              "f1_before": m_before["f1"], "f1_after": m_after["f1"]}
        rows.append(row)
        print(f"  {pid}: risk={rscore:.3f} calib_win={len(Xc)} eval_win={len(ye)} "
             f"AUC before={m_before['auc']:.4f} after={m_after['auc']:.4f} "
             f"(delta={m_after['auc'] - m_before['auc']:+.4f})")
        del model

    out_path = os.path.join(args.out, "tta_adabn_head_to_head.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["patient", "risk_score", "n_calib", "n_eval",
                                           "auc_before", "auc_after", "f1_before", "f1_after"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path}")

    if len(rows) >= 3:
        before = np.array([r["auc_before"] for r in rows])
        after = np.array([r["auc_after"] for r in rows])
        try:
            p = float(wilcoxon(before, after).pvalue)
        except ValueError:
            p = float("nan")
        print(f"\nAll {len(rows)} patients: mean AUC before={before.mean():.4f} "
             f"after={after.mean():.4f} delta={after.mean() - before.mean():+.4f} "
             f"wilcoxon p={p:.4f}")

        has_risk = [r for r in rows if not np.isnan(r["risk_score"])]
        if len(has_risk) >= 4:
            has_risk_sorted = sorted(has_risk, key=lambda r: r["risk_score"])
            q = max(1, len(has_risk_sorted) // 4)
            low_risk = has_risk_sorted[:q]    # lowest combined risk score
            high_risk = has_risk_sorted[-q:]  # highest combined risk score -- the
                                               # hard-tail candidates per Experiment A
            for label, grp in [("bottom-risk-quartile (lowest risk score)", low_risk),
                               ("top-risk-quartile (highest risk score)", high_risk)]:
                b = np.array([r["auc_before"] for r in grp])
                a = np.array([r["auc_after"] for r in grp])
                print(f"  {label} (n={len(grp)}): before={b.mean():.4f} "
                     f"after={a.mean():.4f} delta={a.mean() - b.mean():+.4f}  "
                     f"patients={[r['patient'] for r in grp]}")


if __name__ == "__main__":
    main()
