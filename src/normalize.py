"""Per-channel normalization schemes for the cross-patient seizure pipeline.

Motivation (traceable to the paper's own finding): patient chb05 is an amplitude
outlier whose ictal band power is 11--107x a representative peer, and per-channel
z-scoring does NOT neutralize that shift -- z-score divides by the standard
deviation, which is itself inflated by the very high-amplitude ictal segments, so
the relative structure the model keys on is distorted. These schemes are the
alternatives we ablate against plain z-score to test whether amplitude-invariant
normalization lifts the cross-patient hard tail (chb08, chb12, chb16, chb20).

Each function takes sig: np.ndarray of shape (C, T) and returns (C, T), operating
per channel over the whole recording (matching the existing pipeline's placement).
The full 24-patient LOSO ablation over these schemes is the intended experiment;
it must be run full-cohort with per-patient reporting, not on a small probe.
"""
import numpy as np

EPS = 1e-7
SCHEMES = ("zscore", "robust", "mad", "log", "instance")


def zscore(sig):
    """Current baseline: per-channel mean/std over the recording."""
    return (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + EPS)


def robust(sig):
    """Median/IQR (interquartile-range) scaling: amplitude-outlier resistant, since
    the IQR ignores the extreme ictal tails that inflate the standard deviation."""
    med = np.median(sig, axis=1, keepdims=True)
    q75 = np.percentile(sig, 75, axis=1, keepdims=True)
    q25 = np.percentile(sig, 25, axis=1, keepdims=True)
    return (sig - med) / ((q75 - q25) + EPS)


def mad(sig):
    """Median / median-absolute-deviation scaling (a stronger robust estimator)."""
    med = np.median(sig, axis=1, keepdims=True)
    m = np.median(np.abs(sig - med), axis=1, keepdims=True)
    return (sig - med) / (1.4826 * m + EPS)  # 1.4826 -> consistent with std for Gaussian


def log(sig):
    """Signed log1p amplitude compression, then z-score. Compresses the dynamic
    range so a 100x amplitude patient lands nearer the training amplitude regime."""
    comp = np.sign(sig) * np.log1p(np.abs(sig))
    return zscore(comp)


def instance(sig):
    """Per-channel min-max to [-1, 1] over the recording (bounded dynamic range)."""
    lo = sig.min(1, keepdims=True)
    hi = sig.max(1, keepdims=True)
    return 2.0 * (sig - lo) / ((hi - lo) + EPS) - 1.0


_FN = {"zscore": zscore, "robust": robust, "mad": mad, "log": log, "instance": instance}


def normalize(sig, scheme="zscore"):
    """Apply the named per-channel normalization scheme to sig (C, T)."""
    if scheme not in _FN:
        raise ValueError(f"unknown norm scheme: {scheme} (choose from {SCHEMES})")
    return _FN[scheme](sig).astype(sig.dtype)
