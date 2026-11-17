"""
CHB-MIT loading, windowing and labeling for cross-patient seizure detection.

Pipeline (matches the paper):
  - select the 18 bipolar channels common to all patients
  - resample to 256 Hz, per-channel z-score over the whole recording
  - 2 s windows (512 samples) with 1 s stride (50% overlap)
  - label a window ictal if >50% of it overlaps an annotated seizure
  - cap interictal windows per file (random subsample) to control imbalance

Requires: mne (EDF reading), numpy. Data from PhysioNet chbmit/1.0.0.
Windows are cached to <cache>/chbNN.npz so this parse runs once per patient.
"""
import os
import re
import glob
import numpy as np
import normalize as _norm

FS = 256
WIN = 2 * FS          # 512  (2 s windows: more ictal samples from short seizures)
STRIDE = 1 * FS       # 256  (1 s stride, 50% overlap)
ICTAL_OVERLAP = 0.5
MAX_INTERICTAL_PER_FILE = 200

# Per-channel normalization scheme (see normalize.py). Overridable by train.py
# via --norm; default "zscore" reproduces the existing pipeline exactly.
NORM_SCHEME = os.environ.get("NORM_SCHEME", "zscore")

# 18 bipolar channels present for all CHB-MIT patients
CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]


def parse_summary(summary_path):
    """Return {edf_filename: [(start_sec, end_sec), ...]} from a chbNN-summary.txt."""
    seizures, cur = {}, None
    starts, ends = [], []
    with open(summary_path, "r", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"File Name:\s*(\S+)", line)
            if m:
                if cur is not None:
                    seizures[cur] = list(zip(starts, ends))
                cur, starts, ends = m.group(1), [], []
            elif "Seizure" in line and "Start Time" in line:
                starts.append(int(re.search(r"(\d+)\s*seconds", line).group(1)))
            elif "Seizure" in line and "End Time" in line:
                ends.append(int(re.search(r"(\d+)\s*seconds", line).group(1)))
        if cur is not None:
            seizures[cur] = list(zip(starts, ends))
    return seizures


def _pick_channels(raw):
    # CHB-MIT EDFs contain duplicate labels (e.g. T8-P8 appears twice); mne renames
    # them T8-P8-0 / T8-P8-1. Match the exact label first, then fall back to the
    # first-duplicate "-0" variant so the standard 18-channel montage resolves.
    names = {c.upper().replace(" ", ""): c for c in raw.ch_names}
    picks = []
    for want in CHANNELS:
        key = want.upper().replace(" ", "")
        if key in names:
            picks.append(names[key])
        elif key + "-0" in names:
            picks.append(names[key + "-0"])
    return picks


def load_patient(patient_dir, cache_dir=None, seed=42):
    """Load one patient -> (X: [N,18,1024] float32, y: [N] int).  Cached as .npz."""
    import mne
    pid = os.path.basename(patient_dir.rstrip("/"))
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f"{pid}.npz")
        if os.path.exists(cache):
            try:
                d = np.load(cache)
                return d["X"], d["y"]
            except Exception:
                # partial or corrupt cache (e.g. a concurrent writer mid-write);
                # fall through and regenerate it atomically below.
                pass

    summary = glob.glob(os.path.join(patient_dir, "*summary.txt"))
    seiz = parse_summary(summary[0]) if summary else {}
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for edf in sorted(glob.glob(os.path.join(patient_dir, "*.edf"))):
        try:
            raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
        except Exception:
            continue
        picks = _pick_channels(raw)
        if len(picks) < 6:                        # too few montage channels: unusable
            continue
        raw.pick(picks[:18])
        if int(raw.info["sfreq"]) != FS:
            raw.resample(FS, verbose="ERROR")
        sig = raw.get_data()                      # (<=18, T)
        if sig.shape[0] < 18:                     # pad rare short montages to 18 rows
            sig = np.vstack([sig, np.zeros((18 - sig.shape[0], sig.shape[1]), sig.dtype)])
        sig = _norm.normalize(sig, NORM_SCHEME)

        intervals = seiz.get(os.path.basename(edf), [])
        n = sig.shape[1]
        ict, inter = [], []
        for s in range(0, n - WIN + 1, STRIDE):
            w = sig[:, s:s + WIN].astype(np.float32)
            ws, we = s / FS, (s + WIN) / FS
            ov = sum(max(0, min(we, e) - max(ws, st)) for st, e in intervals)
            (ict if ov > ICTAL_OVERLAP * (WIN / FS) else inter).append(w)
        if len(inter) > MAX_INTERICTAL_PER_FILE:
            idx = rng.choice(len(inter), MAX_INTERICTAL_PER_FILE, replace=False)
            inter = [inter[i] for i in idx]
        X_list += ict + inter
        y_list += [1] * len(ict) + [0] * len(inter)

    X = np.stack(X_list).astype(np.float32) if X_list else np.zeros((0, 18, WIN), np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    if cache_dir:
        # Atomic write: build a unique temp file, then os.replace into place so a
        # concurrent reader never sees a half-written npz (the shared RWX cache is
        # read by many shards at once). Pass a file handle so savez does not append
        # a second .npz suffix.
        tmp = f"{cache}.tmp.{os.getpid()}"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, X=X, y=y)
        os.replace(tmp, cache)
    return X, y


def load_patient_sequential(patient_dir):
    """For event-level test scoring: yield per-EDF, in time order, with NO interictal
    cap: (edf_name, starts_sec[np], X[N,18,WIN], y[N], seizure_intervals, dur_sec).

    Unlike load_patient (which caps interictal and shuffles for training), this keeps
    every window and its start time so we can score in temporal order and compute
    event-level sensitivity and false-alarms/hour on the real interictal load.
    """
    import mne
    summ = glob.glob(os.path.join(patient_dir, "*summary.txt"))
    seiz = parse_summary(summ[0]) if summ else {}
    for edf in sorted(glob.glob(os.path.join(patient_dir, "*.edf"))):
        try:
            raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
        except Exception:
            continue
        picks = _pick_channels(raw)
        if len(picks) < 6:
            continue
        raw.pick(picks[:18])
        if int(raw.info["sfreq"]) != FS:
            raw.resample(FS, verbose="ERROR")
        sig = raw.get_data()
        if sig.shape[0] < 18:
            sig = np.vstack([sig, np.zeros((18 - sig.shape[0], sig.shape[1]), sig.dtype)])
        sig = _norm.normalize(sig, NORM_SCHEME)
        intervals = seiz.get(os.path.basename(edf), [])
        n = sig.shape[1]
        starts, X, y = [], [], []
        for s in range(0, n - WIN + 1, STRIDE):
            ws, we = s / FS, (s + WIN) / FS
            ov = sum(max(0, min(we, e) - max(ws, st)) for st, e in intervals)
            starts.append(ws)
            X.append(sig[:, s:s + WIN].astype(np.float32))
            y.append(1 if ov > ICTAL_OVERLAP * (WIN / FS) else 0)
        X = np.stack(X) if X else np.zeros((0, 18, WIN), np.float32)
        yield (os.path.basename(edf), np.asarray(starts), X,
               np.asarray(y, dtype=np.int64), intervals, n / FS)


def loso_splits(patient_ids):
    """Yield (test_pid, train_pids) for leave-one-subject-out."""
    for pid in patient_ids:
        yield pid, [p for p in patient_ids if p != pid]
