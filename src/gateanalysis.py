"""Gate-faithfulness + band-occlusion analysis for FATE.

Tests whether the frequency machinery is actually used (vs decoration):
  (1) gate weights on ICTAL vs INTERICTAL test windows, per band -- does the gate
      shift toward Delta/Theta on seizures, as the paper claims?
  (2) band occlusion: bandstop each clinical band out of the raw input and measure
      the AUC drop -- which band does the model rely on, and does it agree with (1)?
A faithful, meaningful gate shows a clear ictal shift AND a matching occlusion drop.
A near-constant gate / flat occlusion means the frequency story is decoration.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.signal import butter, sosfiltfilt

BANDS = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0),
         ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0)]
FS = 256


def _bandstop(X, lo, hi, fs=FS):
    sos = butter(4, [lo, hi], btype="bandstop", fs=fs, output="sos")
    return np.ascontiguousarray(sosfiltfilt(sos, X, axis=-1)).astype(np.float32)


@torch.no_grad()
def _scores(model, X, device, batch=128):
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        out.append(torch.softmax(model(xb), 1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


@torch.no_grad()
def gate_and_occlusion(model, X, y, device, batch=128):
    model.eval()
    y = np.asarray(y).astype(int)
    # gates + base scores
    gates, probs = [], []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        logits, g = model(xb, return_gate=True)
        gates.append(g.cpu().numpy())
        probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    G = np.concatenate(gates)              # (N, 4)
    P = np.concatenate(probs)
    has_both = len(np.unique(y)) > 1
    gate_ictal = G[y == 1].mean(0) if (y == 1).any() else np.full(G.shape[1], np.nan)
    gate_inter = G[y == 0].mean(0) if (y == 0).any() else np.full(G.shape[1], np.nan)
    base_auc = roc_auc_score(y, P) if has_both else float("nan")
    # band occlusion (bandstop each band out of the raw input, re-score)
    occ_auc = []
    for _, lo, hi in BANDS:
        Po = _scores(model, _bandstop(X, lo, hi), device, batch)
        occ_auc.append(roc_auc_score(y, Po) if has_both else float("nan"))
    return {
        "base_auc": float(base_auc),
        "gate_ictal": [float(v) for v in gate_ictal],   # delta,theta,alpha,beta
        "gate_inter": [float(v) for v in gate_inter],
        "occ_auc": [float(v) for v in occ_auc],          # AUC with that band removed
    }
