"""Event-level evaluation for seizure detection (clinical metrics).

Window-level AUC/F1 hide how a detector behaves in deployment. This module scores
the held-out patient over ALL windows in time order (no interictal cap) and reports:
  - AUPRC            : imbalance-appropriate, threshold-free
  - event_sensitivity: fraction of annotated seizures overlapped by any positive window
  - fa_per_hour      : false-alarm EVENTS per recording hour (clinical unit)
A predicted "event" is a run of consecutive positive windows (with a small merge gap).
"""
import numpy as np
import torch
from sklearn.metrics import average_precision_score


def _merge_events(starts, pred, win_sec, stride_sec, merge_gap_sec=2.0):
    """Group consecutive positive windows into (t0, t1) predicted-event intervals."""
    pos = np.where(pred == 1)[0]
    if len(pos) == 0:
        return []
    events, cur0, cur1 = [], starts[pos[0]], starts[pos[0]] + win_sec
    for i in pos[1:]:
        t0 = starts[i]
        if t0 <= cur1 + merge_gap_sec:
            cur1 = max(cur1, t0 + win_sec)
        else:
            events.append((cur0, cur1)); cur0, cur1 = t0, t0 + win_sec
    events.append((cur0, cur1))
    return events


@torch.no_grad()
def sequential_windows(model, patient_dir, device, loader_mod, win_sec, stride_sec, batch=128):
    """Per-file window scores for a patient, time-ordered and uncapped.

    Returns a list of dicts {starts, scores, intervals, dur} — one per recording file.
    Used by the Phase-B post-processing path (iterate.sweep_val_files / score_patient_files):
    post-processing is applied per recording (each file has its own timeline), so events are
    never matched across files.
    """
    model.eval()
    files = []
    for name, starts, X, y, intervals, dur in loader_mod.load_patient_sequential(patient_dir):
        if len(X) == 0:
            files.append({"starts": np.array([]), "scores": np.array([]),
                          "intervals": intervals, "dur": dur})
            continue
        probs = []
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            probs.append(torch.softmax(model(xb), 1)[:, 1].cpu().numpy())
        files.append({"starts": np.asarray(starts, float), "scores": np.concatenate(probs),
                      "intervals": intervals, "dur": dur})
    return files


@torch.no_grad()
def event_level_eval(model, patient_dir, device, threshold, loader_mod,
                     win_sec, stride_sec, batch=128):
    model.eval()
    all_p, all_y = [], []
    tp_events = fp_events = total_events = 0
    total_hours = 0.0
    for name, starts, X, y, intervals, dur in loader_mod.load_patient_sequential(patient_dir):
        total_hours += dur / 3600.0
        total_events += len(intervals)
        if len(X) == 0:
            continue
        probs = []
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            probs.append(torch.softmax(model(xb), 1)[:, 1].cpu().numpy())
        p = np.concatenate(probs)
        all_p.append(p); all_y.append(y)
        pred = (p >= threshold).astype(int)
        pred_events = _merge_events(starts, pred, win_sec, stride_sec)
        matched = set()
        for (pt0, pt1) in pred_events:
            hit = False
            for j, (st, en) in enumerate(intervals):
                if pt1 >= st and pt0 <= en:
                    matched.add(j); hit = True
            if not hit:
                fp_events += 1
        tp_events += len(matched)
    y = np.concatenate(all_y) if all_y else np.array([])
    p = np.concatenate(all_p) if all_p else np.array([])
    auprc = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return {
        "auprc": auprc,
        "event_sensitivity": (tp_events / total_events) if total_events else float("nan"),
        "fa_per_hour": (fp_events / total_hours) if total_hours else float("nan"),
        "n_seizures": int(total_events),
        "hours": round(total_hours, 2),
        "n_test_windows": int(len(y)),
    }
