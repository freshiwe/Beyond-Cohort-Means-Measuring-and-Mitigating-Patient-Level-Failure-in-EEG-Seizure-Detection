"""Phase A — SzCORE-style event-level scoring.

Makes our numbers directly comparable to the SzCORE benchmark / leaderboard
(epilepsybenchmarks.com; Dan et al., Epilepsia 2024). Event-based scoring answers the
clinically real questions: how many seizures were caught, how many missed, and how many
false alarms were raised.

Matching rule (SzCORE): a reference (ground-truth) seizure and a hypothesis (predicted)
event match if they OVERLAP, after expanding each reference event by a tolerance on both
sides (default 30 s pre / 60 s post is common; we expose a symmetric `tolerance_sec`).
  TP = reference events overlapped by >=1 hypothesis event
  FN = reference events not overlapped
  FP = hypothesis events overlapping NO reference event
Then sensitivity = TP/(TP+FN), precision = TP/(TP+FP), F1, and FP/day + FP/hour.

Pure numpy/stdlib; operates on (t0, t1) interval lists, so it composes directly with
postproc.apply_postproc() and eventeval's per-window scoring.
"""
import numpy as np


def _overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def score_events(pred_events, ref_events, total_hours,
                 tolerance_pre=30.0, tolerance_post=60.0):
    """Event-level TP/FP/FN + sensitivity/precision/F1 + false-alarm rates.

    pred_events, ref_events: lists of (t0, t1) seconds.
    total_hours: recording duration (for FP/day, FP/hour).
    tolerance_pre/post: seconds to expand each reference event before overlap test.
    """
    ref = [(t0 - tolerance_pre, t1 + tolerance_post) for (t0, t1) in ref_events]
    pred = list(pred_events)
    matched_ref = set()
    fp = 0
    for pe in pred:
        hit = False
        for j, re in enumerate(ref):
            if _overlap(pe, re):
                matched_ref.add(j)
                hit = True
        if not hit:
            fp += 1
    tp = len(matched_ref)
    fn = len(ref) - tp
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * prec * sens / (prec + sens)
          if prec and sens and not np.isnan(prec) and not np.isnan(sens) and (prec + sens) > 0
          else 0.0)
    fp_per_hour = fp / total_hours if total_hours else float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "n_ref": len(ref), "n_pred": len(pred),
        "sensitivity": sens, "precision": prec, "f1": float(f1),
        "fp_per_hour": fp_per_hour, "fp_per_day": fp_per_hour * 24 if total_hours else float("nan"),
        "hours": round(float(total_hours), 3),
    }


def aggregate(per_patient):
    """Macro-average a list of score_events() dicts across patients (SzCORE reports
    both macro and pooled; macro is the fair cross-patient summary)."""
    keys = ["sensitivity", "precision", "f1", "fp_per_hour", "fp_per_day"]
    out = {}
    for k in keys:
        vals = [d[k] for d in per_patient if not np.isnan(d.get(k, float("nan")))]
        out["mean_" + k] = float(np.mean(vals)) if vals else float("nan")
    tp = sum(d["tp"] for d in per_patient); fp = sum(d["fp"] for d in per_patient)
    fn = sum(d["fn"] for d in per_patient)
    out["pooled_sensitivity"] = tp / (tp + fn) if (tp + fn) else float("nan")
    out["pooled_precision"] = tp / (tp + fp) if (tp + fp) else float("nan")
    out["n_patients"] = len(per_patient)
    return out


if __name__ == "__main__":
    # one 40s seizure at t=1800; a perfect hit + one far-away false alarm.
    ref = [(1800, 1840)]
    pred_good = [(1795, 1850)]
    pred_fp = [(1795, 1850), (600, 610)]
    print("perfect :", score_events(pred_good, ref, total_hours=1.0))
    print("1 FA    :", score_events(pred_fp, ref, total_hours=1.0))
