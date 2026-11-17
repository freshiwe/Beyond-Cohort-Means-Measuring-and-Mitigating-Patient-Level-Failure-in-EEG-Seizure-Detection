"""
Siena Scalp EEG loading, windowing and labeling — drop-in replacement for
data.py so the same LOSO harness (train.py --dataset siena) runs on a second,
independent dataset.

Differences from CHB-MIT handled here:
  - referential (monopolar) channels -> derive the same 18 bipolar pairs by
    subtraction, with old 10-20 naming (T3/T4/T5/T6 = T7/T8/P7/P8).
  - 512 Hz -> resampled to 256 Hz (FS).
  - annotations are clock times (HH.MM.SS): seizure onset in-file =
    seizure_start - registration_start (mod 24h for overnight recordings).
  - real-world messiness: filename typos (PNO6 vs PN06), "(in sleep)" suffixes.

Windowing/labeling/interictal-capping match data.py exactly for a fair
cross-dataset comparison. Window size is overridable via module globals (set by
train.py from --window-sec), same as data.py.
"""
import os
import re
import glob
import numpy as np

FS = 256
WIN = 2 * FS          # matches data.py default (overridden by train.py --window-sec)
STRIDE = 1 * FS
ICTAL_OVERLAP = 0.5
MAX_INTERICTAL_PER_FILE = 200

# Same 18-channel bipolar montage as CHB-MIT, expressed as (anode, cathode)
# referential electrodes. Old->new naming: T3=T7, T4=T8, T5=P7, T6=P8.
BIPOLAR = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("FZ", "CZ"), ("CZ", "PZ"),
]
N_CH = len(BIPOLAR)  # 18


def _norm(name):
    """Normalize an EDF channel label to a bare electrode key (e.g. 'EEG Fp1-Ref' -> 'FP1')."""
    s = name.upper().replace("EEG", "").replace("-REF", "").replace("REF", "")
    s = s.replace(" ", "").replace(".", "")
    # strip a trailing reference suffix like 'FP1-LE' or 'FP1-A1'
    s = re.split(r"-", s)[0]
    return s


def _norm_file(fn):
    """Normalize an annotation/EDF file field to a comparable key (PNO6 -> pn06)."""
    b = os.path.basename(fn).strip().lower()
    b = re.sub(r"^pno", "pn0", b)      # letter O is a typo for zero in these ids
    if not b.endswith(".edf"):
        b += ".edf"
    return b


# Siena annotation times use inconsistent delimiters across subjects: dots
# (19.39.33) and colons (21:51:02), sometimes mixed within one entry.
_TIME_RE = re.compile(r"(\d{1,2})[.:](\d{2})[.:](\d{2})")


def _to_sec(m):
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def parse_annotation(path):
    """Return {norm_edf_name: [(onset_sec, offset_sec), ...]} for one subject.

    Handles both annotation dialects seen in Siena:
      - "Seizure start time:"/"Seizure end time:" (PN00 style), and
      - "Start time:"/"End time:" (PN01 style), with dot OR colon delimiters.
    Registration start is subtracted so onsets are seconds-into-file.
    """
    seiz = {}
    cur, reg, s_on = None, None, None
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            l = line.strip()
            low = l.lower()
            m = re.match(r"file name:\s*(\S+)", low)
            if m:
                cur = _norm_file(m.group(1))
                seiz.setdefault(cur, [])
                reg, s_on = None, None
            elif "registration start" in low:
                mt = _TIME_RE.search(l)
                if mt:
                    reg = _to_sec(mt)
            elif "registration end" in low:
                continue
            elif "start time" in low:            # seizure start (Seizure/plain)
                mt = _TIME_RE.search(l)
                if mt:
                    s_on = _to_sec(mt)
            elif "end time" in low:              # seizure end
                mt = _TIME_RE.search(l)
                if mt and cur is not None and reg is not None and s_on is not None:
                    on = (s_on - reg) % 86400
                    off = (_to_sec(mt) - reg) % 86400
                    if off >= on:
                        seiz[cur].append((on, off))
                    s_on = None
    return seiz


def _intervals_for(seiz, edf_base):
    """Match an EDF to its seizure intervals, tolerant of annotation filename quirks:
    the annotation file field can be 'PN01.edf' or even 'PN11-.edf' while the real
    file is 'PN01-1.edf'/'PN11-1.edf'. For subjects whose annotation lacks proper
    per-segment names, fall back to the union of all that subject's seizures."""
    k = _norm_file(edf_base)
    if seiz.get(k):
        return seiz[k]
    subj = re.match(r"(pn\d+)", k)
    if subj:
        s = subj.group(1)
        # If the annotation already uses proper per-segment keys (e.g. pn00-1.edf),
        # exact matching handles it; do NOT merge across a subject's segments.
        has_segmented = any(re.fullmatch(rf"{s}-\d+\.edf", key) for key in seiz)
        if not has_segmented:
            merged = []
            for key, ivs in seiz.items():
                ks = re.match(r"(pn\d+)", key)
                if ks and ks.group(1) == s:
                    merged += ivs
            if merged:
                return merged
    return seiz.get(k, [])


def _bipolar_signal(raw):
    """Derive the 18 bipolar channels from a referential Siena recording. None if incomplete."""
    idx = {}
    for i, ch in enumerate(raw.ch_names):
        idx.setdefault(_norm(ch), i)
    rows = []
    for a, c in BIPOLAR:
        if a not in idx or c not in idx:
            return None
        rows.append((idx[a], idx[c]))
    data = raw.get_data()
    return np.stack([data[a] - data[c] for a, c in rows])   # (18, T)


def load_patient(patient_dir, cache_dir=None, seed=42):
    """Load one Siena subject -> (X:[N,18,WIN] float32, y:[N] int). Cached as .npz."""
    import mne
    pid = os.path.basename(patient_dir.rstrip("/"))
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f"{pid}.npz")
        if os.path.exists(cache):
            d = np.load(cache)
            return d["X"], d["y"]

    ann = glob.glob(os.path.join(patient_dir, "Seizures-list-*.txt"))
    seiz = parse_annotation(ann[0]) if ann else {}
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for edf in sorted(glob.glob(os.path.join(patient_dir, "*.edf"))):
        try:
            raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
        except Exception:
            continue
        sig = _bipolar_signal(raw)
        if sig is None:                                   # montage incomplete -> skip file
            del raw; continue
        sfreq = raw.info["sfreq"]
        if int(sfreq) != FS:                              # 512 -> 256
            import scipy.signal as sps
            n_new = int(sig.shape[1] * FS / sfreq)
            sig = sps.resample(sig, n_new, axis=1)
        del raw
        sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-7)

        intervals = _intervals_for(seiz, os.path.basename(edf))
        n = sig.shape[1]
        ict, inter = [], []
        for s in range(0, n - WIN + 1, STRIDE):
            w = sig[:, s:s + WIN].astype(np.float32)
            ws, we = s / FS, (s + WIN) / FS
            ov = sum(max(0, min(we, e) - max(ws, st)) for st, e in intervals)
            (ict if ov > ICTAL_OVERLAP * (WIN / FS) else inter).append(w)
        if len(inter) > MAX_INTERICTAL_PER_FILE:
            keep = rng.choice(len(inter), MAX_INTERICTAL_PER_FILE, replace=False)
            inter = [inter[i] for i in keep]
        X_list += ict + inter
        y_list += [1] * len(ict) + [0] * len(inter)

    X = np.stack(X_list).astype(np.float32) if X_list else np.zeros((0, N_CH, WIN), np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    if cache_dir:
        np.savez_compressed(cache, X=X, y=y)
    return X, y


def loso_splits(patient_ids):
    for pid in patient_ids:
        yield pid, [p for p in patient_ids if p != pid]


def load_patient_sequential(patient_dir):
    """For event-level test scoring / calibration-split use: yield per-EDF, in time
    order, with NO interictal cap: (edf_name, starts_sec[np], X[N,18,WIN], y[N],
    seizure_intervals, dur_sec). Mirrors data.py's load_patient_sequential exactly,
    using Siena's bipolar derivation + annotation parsing instead of CHB-MIT's.
    """
    import mne
    ann = glob.glob(os.path.join(patient_dir, "Seizures-list-*.txt"))
    seiz = parse_annotation(ann[0]) if ann else {}
    for edf in sorted(glob.glob(os.path.join(patient_dir, "*.edf"))):
        try:
            raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
        except Exception:
            continue
        sig = _bipolar_signal(raw)
        if sig is None:
            del raw
            continue
        sfreq = raw.info["sfreq"]
        if int(sfreq) != FS:
            import scipy.signal as sps
            n_new = int(sig.shape[1] * FS / sfreq)
            sig = sps.resample(sig, n_new, axis=1)
        del raw
        sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-7)
        intervals = _intervals_for(seiz, os.path.basename(edf))
        n = sig.shape[1]
        starts, X, y = [], [], []
        for s in range(0, n - WIN + 1, STRIDE):
            ws, we = s / FS, (s + WIN) / FS
            ov = sum(max(0, min(we, e) - max(ws, st)) for st, e in intervals)
            starts.append(ws)
            X.append(sig[:, s:s + WIN].astype(np.float32))
            y.append(1 if ov > ICTAL_OVERLAP * (WIN / FS) else 0)
        X = np.stack(X) if X else np.zeros((0, N_CH, WIN), np.float32)
        yield (os.path.basename(edf), np.asarray(starts), X,
               np.asarray(y, dtype=np.int64), intervals, n / FS)
