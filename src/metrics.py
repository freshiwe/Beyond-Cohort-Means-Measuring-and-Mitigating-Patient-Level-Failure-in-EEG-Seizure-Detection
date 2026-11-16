"""Evaluation metrics for cross-patient seizure detection."""
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score


def best_f1_threshold(y_true, y_score):
    """Threshold on the ictal score that maximizes F1 (as in the paper)."""
    order = np.argsort(y_score)
    ts = np.unique(y_score[order])
    if len(ts) > 512:
        ts = np.quantile(y_score, np.linspace(0, 1, 512))
    best_t, best = 0.5, -1.0
    for t in ts:
        f = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f > best:
            best, best_t = f, t
    return best_t


def compute_metrics(y_true, y_score, threshold=None):
    """Return AUC + sensitivity/specificity/F1.

    threshold: if provided, use this fixed operating point (chosen on a held-out
    validation split) instead of fitting it on y_true/y_score. Supplying a
    validation-derived threshold removes test-set leakage from Sens/Spec/F1.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    out = {}
    out["auc"] = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    # AUPRC (average precision) is the imbalance-appropriate threshold-free metric.
    out["auprc"] = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    t = float(threshold) if threshold is not None else best_f1_threshold(y_true, y_score)
    y_pred = (y_score >= t).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    out["sensitivity"] = tp / (tp + fn) if (tp + fn) else float("nan")
    out["specificity"] = tn / (tn + fp) if (tn + fp) else float("nan")
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["threshold"] = float(t)
    return out
