"""
Shared calibration/evaluation split for the Risk-Aware TTA experiments
(risk_estimator.py, tta_adabn.py).

Simulates a realistic deployment scenario: before trusting the detector on a
brand-new patient, we have observed their first ~15 minutes of EEG,
unlabeled (we do not know whether a seizure occurred in that window -- the
label is discarded even if one did; a real system would not know either).
That calibration window is used ONLY for computing risk features / adapting
BatchNorm statistics, never for choosing a threshold or measuring AUC. All
remaining windows (rest of the first EDF + every subsequent EDF, in time
order) form the evaluation set that both "before" and "after" performance are
measured on, for a fair, leak-free before/after comparison.

Reuses data.py's load_patient_sequential(), built earlier this session for
event-level evaluation -- already time-ordered, per-EDF, with real window
start timestamps.

IMPORTANT (memory): CHB-MIT recordings run many hours per patient with NO
interictal cap in load_patient_sequential (unlike the capped load_patient
used for training) -- materializing the full uncapped eval set for all 24
patients simultaneously OOM-killed the first smoke test at 16Gi even for
just 3 patients. So this module intentionally offers two separate entry
points instead of one that returns everything:
  - calib_windows(): stops reading as soon as ~calib_seconds is covered,
    never touches the rest of the recording. Cheap, small, safe to hold for
    all 24 patients at once (risk_estimator.py's use case).
  - eval_stream(): a generator over the remaining windows, one EDF file's
    worth at a time, for streaming (score-then-discard) use
    (tta_adabn.py's use case) -- never materializes the whole recording.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def _get_loader(dataset):
    if dataset == "siena":
        import data_siena as _m
    else:
        import data as _m
    return _m.load_patient_sequential, _m.WIN, _m.FS

CALIB_SECONDS = 900.0  # 15 min
DEFAULT_DATASET = "chbmit"


def calib_windows(patient_dir, calib_seconds=CALIB_SECONDS, dataset=DEFAULT_DATASET):
    """Return (X_calib, calib_actual_seconds). Stops reading EDFs once
    calib_seconds of coverage is reached -- does not load the rest of the
    recording at all, so this is cheap regardless of total recording length.
    """
    load_patient_sequential, WIN, FS = _get_loader(dataset)
    calib_X = []
    elapsed = 0.0
    for edf_name, starts, X, y, intervals, dur in load_patient_sequential(patient_dir):
        if len(starts) == 0:
            continue
        if elapsed >= calib_seconds:
            break
        cutoff_mask = (elapsed + starts) < calib_seconds
        n_calib = int(cutoff_mask.sum())
        if n_calib > 0:
            calib_X.append(X[:n_calib])
        elapsed += dur
        if elapsed >= calib_seconds:
            break
    X_calib = np.concatenate(calib_X, axis=0) if calib_X else np.zeros((0, 18, WIN), np.float32)
    return X_calib, min(elapsed, calib_seconds)


def eval_stream(patient_dir, calib_seconds=CALIB_SECONDS, dataset=DEFAULT_DATASET):
    """Yield (X_chunk, y_chunk) for the evaluation portion (everything after
    the first calib_seconds of recording), one EDF file's worth per yield --
    never materializes the whole recording at once. Caller should score each
    chunk and discard it (accumulate only scores/labels, not raw windows).
    """
    load_patient_sequential, WIN, FS = _get_loader(dataset)
    elapsed = 0.0
    for edf_name, starts, X, y, intervals, dur in load_patient_sequential(patient_dir):
        if len(starts) == 0:
            continue
        if elapsed >= calib_seconds:
            if len(X) > 0:
                yield X, y
            elapsed += dur
            continue
        cutoff_mask = (elapsed + starts) < calib_seconds
        n_calib = int(cutoff_mask.sum())
        if n_calib < len(X):
            yield X[n_calib:], y[n_calib:]
        elapsed += dur


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/data/chbmit"))
    ap.add_argument("--patients", default="chb01,chb02,chb03")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, choices=["chbmit", "siena"])
    args = ap.parse_args()
    for pid in args.patients.split(","):
        pdir = os.path.join(args.data_root, pid)
        Xc, cov = calib_windows(pdir, dataset=args.dataset)
        n_eval, n_ictal = 0, 0
        for Xch, ych in eval_stream(pdir, dataset=args.dataset):
            n_eval += len(Xch)
            n_ictal += int((ych == 1).sum())
        print(f"{pid}: calib={Xc.shape[0]} windows (~{cov:.0f}s)  "
             f"eval={n_eval} windows ({n_ictal} ictal)")
