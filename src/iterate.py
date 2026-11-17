"""Iteration engine — tune post-processing on validation, lock it, score test with SzCORE.

This is the automated loop that lets us push toward a clinically viable operating point
WITHOUT touching the network: for a held-out patient we have a per-window score stream;
we search postproc params on the validation stream to hit a target false-alarm budget at
max sensitivity, freeze those params, then report SzCORE metrics on test. No test leakage:
params are chosen only from validation.

Inputs are plain arrays so this composes with eventeval (which already produces per-window
probs + start times + reference intervals per patient).
"""
import itertools
import numpy as np

from postproc import apply_postproc, DEFAULT_GRID
from szcore import score_events


def sweep_val(scores, starts, ref_events, hours, win_sec, stride_sec,
              grid=None, fp_per_day_budget=10.0):
    """Grid-search postproc params on VALIDATION data.

    Objective: maximize event sensitivity subject to fp_per_day <= budget; break ties
    by lower fp_per_day. Returns (best_params, best_val_metrics).
    """
    grid = grid or DEFAULT_GRID
    keys = list(grid.keys())
    best, best_params = None, None
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        if params.get("off_thr", 0) > params.get("on_thr", 1):
            continue  # invalid hysteresis
        ev = apply_postproc(scores, starts, win_sec, stride_sec, **params)
        m = score_events(ev, ref_events, hours)
        feasible = (not np.isnan(m["fp_per_day"])) and m["fp_per_day"] <= fp_per_day_budget
        # rank: feasibility, then sensitivity, then fewer FP/day
        key = (1 if feasible else 0, m["sensitivity"] if not np.isnan(m["sensitivity"]) else -1,
               -(m["fp_per_day"] if not np.isnan(m["fp_per_day"]) else 1e9))
        if best is None or key > best:
            best, best_params = key, params
    val_ev = apply_postproc(scores, starts, win_sec, stride_sec, **best_params)
    return best_params, score_events(val_ev, ref_events, hours)


def score_test(scores, starts, ref_events, hours, win_sec, stride_sec, params):
    """Apply FROZEN val params to the test patient and return SzCORE metrics."""
    ev = apply_postproc(scores, starts, win_sec, stride_sec, **params)
    m = score_events(ev, ref_events, hours)
    m["_params"] = params
    return m


# ---- per-file (per-patient, multi-recording) scoring for the train.py wiring ----

def score_patient_files(files, win_sec, stride_sec, params,
                        tol_pre=30.0, tol_post=60.0):
    """Apply post-proc per recording file and aggregate SzCORE counts across a patient.

    files: list of {starts, scores, intervals, dur} from eventeval.sequential_windows().
    Events are scored within each file (never matched across files); TP/FP/FN and hours
    are summed, then sensitivity/precision/F1 + FP/day computed on the totals.
    """
    tp = fp = fn = 0
    hours = 0.0
    for f in files:
        hours += f["dur"] / 3600.0
        ref = f["intervals"]
        if len(f["scores"]) == 0:
            fn += len(ref)
            continue
        ev = apply_postproc(f["scores"], f["starts"], win_sec, stride_sec, **params)
        m = score_events(ev, ref, f["dur"] / 3600.0, tol_pre, tol_post)
        tp += m["tp"]; fp += m["fp"]; fn += m["fn"]
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * prec * sens / (prec + sens)
          if prec and sens and not np.isnan(prec) and not np.isnan(sens) and (prec + sens) > 0
          else 0.0)
    fp_per_hour = fp / hours if hours else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "sensitivity": sens, "precision": prec,
            "f1": float(f1), "fp_per_hour": fp_per_hour,
            "fp_per_day": fp_per_hour * 24 if hours else float("nan"),
            "hours": round(hours, 3)}


def sweep_val_files(files, win_sec, stride_sec, grid=None, fp_per_day_budget=10.0):
    """Grid-search post-proc params on calibration patient(s); maximize sensitivity s.t.
    FP/day <= budget, tie-break on lower FP/day. Returns (best_params, best_metrics)."""
    grid = grid or DEFAULT_GRID
    keys = list(grid.keys())
    best, best_params = None, None
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        if params.get("off_thr", 0) > params.get("on_thr", 1):
            continue
        m = score_patient_files(files, win_sec, stride_sec, params)
        feasible = (not np.isnan(m["fp_per_day"])) and m["fp_per_day"] <= fp_per_day_budget
        key = (1 if feasible else 0,
               m["sensitivity"] if not np.isnan(m["sensitivity"]) else -1,
               -(m["fp_per_day"] if not np.isnan(m["fp_per_day"]) else 1e9))
        if best is None or key > best:
            best, best_params = key, params
    return best_params, score_patient_files(files, win_sec, stride_sec, best_params)


if __name__ == "__main__":
    # Synthetic: show the sweep drives FP/day down while keeping the true seizure.
    win, stride = 2.0, 1.0
    n = 3600
    starts = np.arange(n) * stride
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 0.35, n)
    scores[rng.integers(0, n, 40)] = 0.95            # 40 FP spikes
    ictal = (starts + win > 1800) & (starts < 1840)
    scores[ictal] = rng.uniform(0.85, 0.99, ictal.sum())
    ref = [(1800.0, 1840.0)]
    raw = score_events(
        [(float(starts[i]), float(starts[i] + win)) for i in np.where(scores >= 0.7)[0]],
        ref, 1.0)
    params, valm = sweep_val(scores, starts, ref, 1.0, win, stride, fp_per_day_budget=24)
    print(f"raw (thr=0.7)     : sens={raw['sensitivity']:.2f} FP/day={raw['fp_per_day']:.0f}")
    print(f"best val params   : {params}")
    print(f"after postproc    : sens={valm['sensitivity']:.2f} FP/day={valm['fp_per_day']:.1f} "
          f"F1={valm['f1']:.2f}")
