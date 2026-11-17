"""Clinical metrics reviewers asked for: SzCORE event scoring, detection latency,
and per-patient event counts with Wilson 95% CIs.

Inputs (all committed):
  experiments/data/results-fate-preds/<pid>/preds_<pid>.npz   (y_true, y_score, threshold)
  experiments/data/chbmit-summaries/<pid>/<pid>-summary.txt   (PhysioNet CHB-MIT annotations)
  experiments/data/results-graphst-pp/fp_day_summary.csv      (post-processed event sens)

Reuses szcore.score_events() and data.parse_summary() unmodified.

TIMELINE RECONSTRUCTION (the load-bearing step)
-----------------------------------------------
preds_<pid>.npz stores the output of data.load_patient(): per EDF file, all ictal
windows in time order, then <=200 RANDOMLY SUBSAMPLED interictal windows in random
order. There are no window start times. So:
  * ICTAL windows are recoverable on the real clock. Windows are 2 s with a 1 s
    stride starting at integer second 0, and a window is ictal iff >50% of it
    overlaps the annotation. For an annotation [st, en) in integer seconds that
    means window starts ws = st, st+1, ..., en-2, i.e. exactly (en-st-1) windows,
    the k-th of which starts at absolute second st+k. Verified exactly against
    y_true.sum() for 23/24 patients (chb12: 3 EDFs are dropped by the loader's
    montage check, verified by run-length matching).
  * INTERICTAL windows are NOT recoverable: order destroyed, 200/file cap, no
    timestamps, no recording duration. Anything requiring the interictal timeline
    (false alarms, FP/24 h, pre-onset tolerance) is therefore NOT computable here.
"""
import csv
import glob
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import parse_summary          # noqa: E402  (do not modify)
from szcore import score_events         # noqa: E402  (do not modify)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/
PREDS = os.path.join(ROOT, "data", "results-fate-preds")
SUMM = os.path.join(ROOT, "data", "chbmit-summaries")
PPDIR = os.path.join(ROOT, "data", "results-graphst-pp")
SZOUT = os.path.join(ROOT, "data", "results-szcore")

WIN_SEC = 2.0
STRIDE_SEC = 1.0


def wilson(k, n, z=1.959963984540054):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def annotated_seizures(pid):
    """[(edf, start_sec, end_sec, n_ictal_windows)] in loader (sorted-filename) order."""
    sm = parse_summary(os.path.join(SUMM, pid, f"{pid}-summary.txt"))
    out = []
    for edf in sorted(sm.keys()):
        for st, en in sm[edf]:
            out.append((edf, int(st), int(en), int(en) - int(st) - 1))
    return out


def ictal_runs(y_true):
    """[(start_idx, end_idx_exclusive)] of maximal y_true==1 runs."""
    y = np.asarray(y_true).astype(int)
    idx = np.flatnonzero(np.diff(np.concatenate([[0], y, [0]])) != 0)
    return list(zip(idx[0::2].tolist(), idx[1::2].tolist()))


def map_seizures_to_indices(pid, y_true):
    """Assign each annotated seizure its [a, b) slice of the prediction array.

    A run in the array is one EDF file's ictal block (possibly several seizures
    concatenated). Files are matched to runs greedily by their total ictal-window
    count, which skips EDFs the loader dropped. Returns (mapped, skipped_files).
    """
    seiz = annotated_seizures(pid)
    per_file = {}
    for edf, st, en, nw in seiz:
        per_file.setdefault(edf, []).append((st, en, nw))
    files = [f for f in sorted(per_file) if sum(w for _, _, w in per_file[f]) > 0]
    runs = ictal_runs(y_true)

    mapped, skipped, fi = [], [], 0
    for (a, b) in runs:
        need = b - a
        while fi < len(files) and sum(w for _, _, w in per_file[files[fi]]) != need:
            skipped.append(files[fi])
            fi += 1
        if fi >= len(files):
            raise RuntimeError(f"{pid}: cannot match run of length {need}")
        cur = a
        for (st, en, nw) in per_file[files[fi]]:
            mapped.append({"pid": pid, "edf": files[fi], "onset": st, "offset": en,
                           "a": cur, "b": cur + nw})
            cur += nw
        assert cur == b, f"{pid}: run/seizure length mismatch"
        fi += 1
    skipped += files[fi:]
    return mapped, skipped


def merged_pred_events(above):
    """Runs of above-threshold windows -> (t0, t1) on the array-index pseudo-clock
    (1 index == 1 s, window length 2 s)."""
    idx = np.flatnonzero(np.asarray(above).astype(int))
    if len(idx) == 0:
        return []
    ev, c0, c1 = [], float(idx[0]), float(idx[0]) + WIN_SEC
    for i in idx[1:]:
        if i <= c1:
            c1 = max(c1, float(i) + WIN_SEC)
        else:
            ev.append((c0, c1)); c0, c1 = float(i), float(i) + WIN_SEC
    ev.append((c0, c1))
    return ev


def main():
    os.makedirs(SZOUT, exist_ok=True)
    pids = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(PREDS, "*", "preds_*.npz")))

    sz_rows, lat_rows, per_patient_dicts = [], [], []
    for pid in pids:
        d = np.load(os.path.join(PREDS, pid, f"preds_{pid}.npz"))
        y, s, thr = d["y_true"], d["y_score"], float(d["threshold"])
        above = s >= thr
        mapped, skipped = map_seizures_to_indices(pid, y)

        # --- 1. SzCORE any-overlap, zero tolerance (interictal timeline unavailable)
        ref = [(float(m["a"]), float(m["b"] - 1) + WIN_SEC) for m in mapped]
        pred = merged_pred_events(above)
        r = score_events(pred, ref, total_hours=float("nan"),
                         tolerance_pre=0.0, tolerance_post=0.0)
        # strict check: TP only if an above-threshold window lies INSIDE the seizure
        strict_tp = sum(1 for m in mapped if above[m["a"]:m["b"]].any())
        n = len(mapped)
        lo, hi = wilson(strict_tp, n)
        sz_rows.append({
            "patient": pid, "n_ref_events": n,
            "tp": strict_tp, "fn": n - strict_tp,
            "event_sensitivity": round(strict_tp / n, 4) if n else float("nan"),
            "sens_ci_lo": round(lo, 4), "sens_ci_hi": round(hi, 4),
            "szcore_overlap_tp": r["tp"],
            "fp_events": "NOT_COMPUTABLE", "fp_per_24h": "NOT_COMPUTABLE",
            "recording_hours": "NOT_COMPUTABLE",
            "threshold": round(thr, 6),
            "note": "interictal windows are capped/shuffled and carry no timestamps",
        })
        per_patient_dicts.append({"tp": strict_tp, "fn": n - strict_tp, "fp": 0,
                                  "sensitivity": strict_tp / n if n else float("nan"),
                                  "precision": float("nan"), "f1": float("nan"),
                                  "fp_per_hour": float("nan"),
                                  "fp_per_day": float("nan")})

        # --- 2. detection latency (onset -> end of first above-threshold ictal window)
        lats = []
        for m in mapped:
            w = np.flatnonzero(above[m["a"]:m["b"]])
            if len(w):
                lats.append(float(w[0]) * STRIDE_SEC + WIN_SEC)
        lat_rows.append({
            "patient": pid, "n_events": n, "n_detected": len(lats),
            "median_latency_s": round(float(np.median(lats)), 2) if lats else float("nan"),
            "mean_latency_s": round(float(np.mean(lats)), 2) if lats else float("nan"),
            "min_latency_s": round(float(np.min(lats)), 2) if lats else float("nan"),
            "max_latency_s": round(float(np.max(lats)), 2) if lats else float("nan"),
            "n_skipped_edfs": len(skipped),
        })

    # ---- write SzCORE csv
    p1 = os.path.join(SZOUT, "szcore_summary.csv")
    with open(p1, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sz_rows[0].keys()))
        w.writeheader()
        for r in sz_rows:
            w.writerow(r)
        tp = sum(r["tp"] for r in sz_rows); fn = sum(r["fn"] for r in sz_rows)
        macro = float(np.mean([r["event_sensitivity"] for r in sz_rows]))
        plo, phi = wilson(tp, tp + fn)
        w.writerow({"patient": "MACRO_MEAN", "n_ref_events": len(sz_rows),
                    "event_sensitivity": round(macro, 4),
                    "fp_events": "NOT_COMPUTABLE", "fp_per_24h": "NOT_COMPUTABLE"})
        w.writerow({"patient": "POOLED", "n_ref_events": tp + fn, "tp": tp, "fn": fn,
                    "event_sensitivity": round(tp / (tp + fn), 4),
                    "sens_ci_lo": round(plo, 4), "sens_ci_hi": round(phi, 4),
                    "fp_events": "NOT_COMPUTABLE", "fp_per_24h": "NOT_COMPUTABLE"})

    # ---- write latency csv
    p2 = os.path.join(SZOUT, "detection_latency.csv")
    with open(p2, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lat_rows[0].keys()))
        w.writeheader()
        for r in lat_rows:
            w.writerow(r)
        med = [r["median_latency_s"] for r in lat_rows if r["n_detected"]]
        mea = [r["mean_latency_s"] for r in lat_rows if r["n_detected"]]
        w.writerow({"patient": "COHORT_MACRO",
                    "n_events": sum(r["n_events"] for r in lat_rows),
                    "n_detected": sum(r["n_detected"] for r in lat_rows),
                    "median_latency_s": round(float(np.median(med)), 2),
                    "mean_latency_s": round(float(np.mean(mea)), 2)})

    # ---- 3. event counts + Wilson CI on the post-processed GraphST sensitivities
    pp = list(csv.DictReader(open(os.path.join(PPDIR, "fp_day_summary.csv"))))
    p3 = os.path.join(PPDIR, "event_counts_ci.csv")
    ci_rows = []
    for row in pp:
        pid = row["patient"]
        if pid == "mean":
            continue
        n_annot = len(annotated_seizures(pid))
        y = np.load(os.path.join(PREDS, pid, f"preds_{pid}.npz"))["y_true"]
        mapped, skipped = map_seizures_to_indices(pid, y)
        n_eval = len(mapped)
        sens = float(row["pp_event_sens"])
        k = int(round(sens * n_eval))
        lo, hi = wilson(k, n_eval)
        ci_rows.append({
            "patient": pid,
            "n_seizures_annotated": n_annot,
            "n_seizures_evaluated": n_eval,
            "n_edfs_skipped_by_loader": len(skipped),
            "tp_events": k, "fn_events": n_eval - k,
            "pp_event_sens": round(k / n_eval, 4),
            "wilson95_lo": round(lo, 4), "wilson95_hi": round(hi, 4),
            "pp_fp_per_day": row["pp_fp_per_day"],
            "reported_pp_event_sens": row["pp_event_sens"],
        })
    with open(p3, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ci_rows[0].keys()))
        w.writeheader()
        for r in ci_rows:
            w.writerow(r)
        K = sum(r["tp_events"] for r in ci_rows); N = sum(r["n_seizures_evaluated"] for r in ci_rows)
        lo, hi = wilson(K, N)
        w.writerow({"patient": "POOLED", "n_seizures_annotated":
                    sum(r["n_seizures_annotated"] for r in ci_rows),
                    "n_seizures_evaluated": N, "tp_events": K, "fn_events": N - K,
                    "pp_event_sens": round(K / N, 4),
                    "wilson95_lo": round(lo, 4), "wilson95_hi": round(hi, 4)})
        w.writerow({"patient": "MACRO_MEAN",
                    "pp_event_sens": round(float(np.mean([r["pp_event_sens"] for r in ci_rows])), 4)})

    for path in (p1, p2, p3):
        print("\n=== " + path + " ===")
        print(open(path).read().rstrip())


if __name__ == "__main__":
    main()
