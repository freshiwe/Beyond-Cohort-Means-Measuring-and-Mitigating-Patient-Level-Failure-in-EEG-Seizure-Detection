"""Phase B — false-alarm post-processing for window-level seizure detectors.

A window classifier emits one ictal probability per window. Raw thresholding produces
hundreds of false alarms/day (our eventeval: mean 18.6 FA/h ~= 446/day). The standard,
network-free route to a clinically viable false-alarm rate is temporal post-processing:

  firing power  : moving average of the ictal probability over a window of windows
                  (Teixeira et al.); smooths isolated spikes.
  hysteresis    : two thresholds (on > off). An alarm starts when FP crosses `on_thr`
                  and ends when it falls below `off_thr` -> stable events, fewer flickers.
  min duration  : drop predicted events shorter than `min_event_sec` (kills 1-window blips).
  refractory    : after an event ends, suppress new alarms for `refractory_sec`
                  (a real EEG reviewer would not re-alarm instantly).

All parameters are meant to be tuned on the VALIDATION patient only (no test leakage),
mirroring the val-selected threshold discipline already in train.py.

Pure numpy; unit-testable on synthetic score streams (see __main__).
"""
import numpy as np


def firing_power(scores, k=5):
    """Causal moving average of ictal probability over the last `k` windows.

    Causal (uses only past+current) so it is deployable as a real-time alarm.
    k=1 disables smoothing.
    """
    scores = np.asarray(scores, dtype=float)
    if k <= 1 or len(scores) == 0:
        return scores
    out = np.empty_like(scores)
    csum = np.cumsum(np.insert(scores, 0, 0.0))
    for i in range(len(scores)):
        lo = max(0, i - k + 1)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def hysteresis_alarm(fp, on_thr, off_thr):
    """Two-threshold state machine -> binary alarm per window.

    Alarm turns ON when firing-power >= on_thr, stays on until it drops below off_thr.
    Requires off_thr <= on_thr.
    """
    fp = np.asarray(fp, dtype=float)
    on = np.zeros(len(fp), dtype=int)
    state = 0
    for i, v in enumerate(fp):
        if state == 0 and v >= on_thr:
            state = 1
        elif state == 1 and v < off_thr:
            state = 0
        on[i] = state
    return on


def _runs_to_events(alarm, starts, win_sec):
    """Contiguous runs of alarm==1 -> list of (t0, t1) time intervals."""
    events = []
    i, n = 0, len(alarm)
    while i < n:
        if alarm[i] == 1:
            j = i
            while j + 1 < n and alarm[j + 1] == 1:
                j += 1
            events.append((float(starts[i]), float(starts[j] + win_sec)))
            i = j + 1
        else:
            i += 1
    return events


def apply_postproc(scores, starts, win_sec, stride_sec,
                   on_thr=0.7, off_thr=None, fp_k=5,
                   min_event_sec=0.0, refractory_sec=0.0):
    """Turn a per-window score stream into predicted seizure events (t0, t1).

    off_thr defaults to on_thr (plain thresholding on the smoothed signal) unless given.
    Returns a list of (t0, t1) intervals in seconds, after min-duration + refractory.
    """
    scores = np.asarray(scores, dtype=float)
    starts = np.asarray(starts, dtype=float)
    if len(scores) == 0:
        return []
    off_thr = on_thr if off_thr is None else off_thr
    fp = firing_power(scores, fp_k)
    alarm = hysteresis_alarm(fp, on_thr, off_thr)
    events = _runs_to_events(alarm, starts, win_sec)
    # minimum duration
    if min_event_sec > 0:
        events = [(a, b) for (a, b) in events if (b - a) >= min_event_sec]
    # refractory: merge/suppress events that start within refractory_sec of the last end
    if refractory_sec > 0 and events:
        merged = [events[0]]
        for (a, b) in events[1:]:
            la, lb = merged[-1]
            if a - lb <= refractory_sec:
                merged[-1] = (la, max(lb, b))  # absorb into previous event
            else:
                merged.append((a, b))
        events = merged
    return events


# Default grids for the val-only sweep (Phase B iteration).
DEFAULT_GRID = {
    "on_thr": [0.5, 0.6, 0.7, 0.8, 0.9],
    "off_thr": [0.3, 0.4, 0.5],
    "fp_k": [1, 3, 5, 9],
    "min_event_sec": [0.0, 4.0, 8.0],
    "refractory_sec": [0.0, 30.0, 60.0],
}


if __name__ == "__main__":
    # Synthetic sanity check: 1-hour recording, 2s windows, 1s stride, one 40s seizure.
    win, stride = 2.0, 1.0
    n = 3600
    starts = np.arange(n) * stride
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 0.3, n)            # interictal background
    scores[np.random.randint(0, n, 25)] = 0.95  # 25 isolated FP spikes
    sz0, sz1 = 1800, 1840
    ictal = (starts + win > sz0) & (starts < sz1)
    scores[ictal] = rng.uniform(0.85, 0.99, ictal.sum())
    raw = _runs_to_events((scores >= 0.7).astype(int), starts, win)
    pp = apply_postproc(scores, starts, win, stride,
                        on_thr=0.7, off_thr=0.4, fp_k=5,
                        min_event_sec=4.0, refractory_sec=30.0)
    print(f"raw threshold events : {len(raw)}  (1 real seizure + spikes)")
    print(f"post-processed events: {len(pp)}  -> {pp}")
