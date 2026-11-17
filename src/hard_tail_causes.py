"""
Experiment 3: Why Do High-Seizure Patients Fail?

Three independent, publishable analyses of MSCTEV1's real per-patient LOSO
hard tail (chb06/13/14/16, run msctev1-20260829):

  A. Seizure duration variability (coefficient of variation of durations)
  B. Inter-seizure interval / clustering (short ISI -> post/pre-ictal
     contamination of "interictal" windows)
  C. Ictal morphology consistency (pairwise PSD cosine similarity across a
     patient's own seizures)

Uploaded as analysis.txt by a co-author; moved into experiments/src/ and
fixed here before running:

  - parse_summary_full() used substring match "Seizure Start Time" (misses
    numbered "Seizure N Start Time:" files) and digit-concatenation via
    filter(str.isdigit, line), which corrupts timestamps for the numbered
    format -- the exact bug already found and fixed once this session in
    the E09 hard-tail-predictor script. Fixed here by reusing data.py's
    already-correct parse_summary() instead of re-implementing parsing.
  - Analysis C read from a "chbmit_windows/{patient}*.npz" directory with
    keys "windows"/"labels" that does not exist anywhere in this project.
    MSCTEV1 (whose AUCs this script analyzes) was itself trained from
    "/data/cache-2s/{patient}.npz" (keys X, y) -- confirmed against
    experiments/nrp/msctev1_batch/job-indexed.yaml's own --data-root flag.
    Fixed to read the real cache instead.
"""
import numpy as np
import json
import os
import sys
from scipy.stats import mannwhitneyu
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import parse_summary

SAVE_DIR = Path(os.environ.get("SAVE_DIR", "/data/msctev1/runs/msctev1-20260829"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data/chbmit"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/data/cache-2s"))

LOSO_AUCS = {
    "chb01": 0.9705, "chb02": 0.9661, "chb03": 0.9940,
    "chb04": 0.9252, "chb05": 0.8066, "chb06": 0.5236,
    "chb07": 0.9617, "chb08": 0.8292, "chb09": 0.9918,
    "chb10": 0.9552, "chb11": 0.9482, "chb13": 0.3702,
    "chb14": 0.3949, "chb15": 0.9008, "chb16": 0.6222,
    "chb17": 0.8706, "chb18": 0.9245, "chb19": 0.9617,
    "chb20": 0.8688, "chb21": 0.9461, "chb22": 0.9989,
    "chb23": 0.9902, "chb24": 0.8906,
}

PATIENT_NAMES = list(LOSO_AUCS.keys())

HARD_THR = 0.70
EASY_THR = 0.85
hard_patients = [p for p, a in LOSO_AUCS.items() if a < HARD_THR]
easy_patients = [p for p, a in LOSO_AUCS.items() if a >= EASY_THR]
print(f"Hard patients (AUC < {HARD_THR}): {hard_patients}")
print(f"Easy patients (AUC >= {EASY_THR}): {easy_patients}")


def get_seizures(summary_path):
    """List[{file, onset, offset, duration}] in file order (fixed parser)."""
    seiz_by_file = parse_summary(summary_path)
    seizures = []
    for fname in sorted(seiz_by_file.keys()):
        for onset, offset in seiz_by_file[fname]:
            seizures.append({"file": fname, "onset": onset,
                              "offset": offset, "duration": offset - onset})
    return seizures


# ============================================================
# ANALYSIS A -- Seizure Duration Variability
# ============================================================
print("\n" + "=" * 70)
print("  Analysis A: Seizure Duration Variability (Heterogeneity)")
print("=" * 70)
print(f"  {'Patient':<10} {'N Seiz':>7} {'Mean Dur':>9} {'Std Dur':>9} "
      f"{'CV':>8} {'AUC':>8} {'Group':>8}")
print("  " + "-" * 62)

variability_data = {}
for patient in PATIENT_NAMES:
    summary_path = DATA_DIR / patient / f"{patient}-summary.txt"
    if not summary_path.exists():
        continue
    seizures = get_seizures(summary_path)
    if len(seizures) < 2:
        continue

    durations = [s["duration"] for s in seizures]
    mean_dur = np.mean(durations)
    std_dur = np.std(durations)
    cv = std_dur / mean_dur if mean_dur > 0 else 0
    auc = LOSO_AUCS[patient]
    group = ("Hard" if auc < HARD_THR else
             "Easy" if auc >= EASY_THR else "Mid")

    variability_data[patient] = {
        "n_seizures": len(seizures), "mean_dur": mean_dur,
        "std_dur": std_dur, "cv": cv, "durations": durations,
        "auc": auc, "group": group,
    }
    print(f"  {patient:<10} {len(seizures):>7} {mean_dur:>9.1f} "
          f"{std_dur:>9.1f} {cv:>8.3f} {auc:>8.4f} {group:>8}")

hard_cvs = [variability_data[p]["cv"] for p in hard_patients if p in variability_data]
easy_cvs = [variability_data[p]["cv"] for p in easy_patients if p in variability_data]

pval_a = None
if hard_cvs and easy_cvs:
    stat, pval_a = mannwhitneyu(hard_cvs, easy_cvs, alternative="two-sided")
    print(f"\n  Hard CV mean : {np.mean(hard_cvs):.3f}")
    print(f"  Easy CV mean : {np.mean(easy_cvs):.3f}")
    print(f"  Mann-Whitney U (two-sided): p = {pval_a:.4f}")
    print(f"  {'SIGNIFICANT' if pval_a < 0.05 else 'not significant'}")


# ============================================================
# ANALYSIS B -- Inter-Seizure Interval
# ============================================================
print("\n" + "=" * 70)
print("  Analysis B: Inter-Seizure Interval (Clustering)")
print("=" * 70)
print(f"  {'Patient':<10} {'N Seiz':>7} {'Mean ISI(s)':>12} "
      f"{'Min ISI(s)':>11} {'Clustered':>10} {'AUC':>8}")
print("  " + "-" * 62)

CLUSTER_THR_SEC = 300

isi_data = {}
for patient in PATIENT_NAMES:
    summary_path = DATA_DIR / patient / f"{patient}-summary.txt"
    if not summary_path.exists():
        continue
    seizures = get_seizures(summary_path)
    if len(seizures) < 2:
        isi_data[patient] = {"mean_isi": np.inf, "min_isi": np.inf,
                              "n_clustered": 0, "frac_clustered": 0.0,
                              "auc": LOSO_AUCS[patient]}
        continue

    isis = []
    for i in range(1, len(seizures)):
        isi = seizures[i]["onset"] - seizures[i - 1]["offset"]
        if isi >= 0:
            isis.append(isi)
    if not isis:
        continue

    mean_isi = np.mean(isis)
    min_isi = min(isis)
    n_clustered = sum(1 for isi in isis if isi < CLUSTER_THR_SEC)
    frac_clustered = n_clustered / len(isis)
    auc = LOSO_AUCS[patient]

    isi_data[patient] = {"mean_isi": mean_isi, "min_isi": min_isi,
                          "n_clustered": n_clustered,
                          "frac_clustered": frac_clustered,
                          "isis": isis, "auc": auc}
    print(f"  {patient:<10} {len(seizures):>7} {mean_isi:>12.0f} "
          f"{min_isi:>11.0f} {f'{n_clustered}/{len(isis)}':>10} {auc:>8.4f}")

hard_isis = [isi_data[p]["mean_isi"] for p in hard_patients
             if p in isi_data and not np.isinf(isi_data[p]["mean_isi"])]
easy_isis = [isi_data[p]["mean_isi"] for p in easy_patients
             if p in isi_data and not np.isinf(isi_data[p]["mean_isi"])]

pval_b = None
if hard_isis and easy_isis:
    stat, pval_b = mannwhitneyu(hard_isis, easy_isis, alternative="two-sided")
    print(f"\n  Hard mean ISI : {np.mean(hard_isis):.0f}s")
    print(f"  Easy mean ISI : {np.mean(easy_isis):.0f}s")
    print(f"  Mann-Whitney U (two-sided): p = {pval_b:.4f}")
    print(f"  {'SIGNIFICANT' if pval_b < 0.05 else 'not significant'}")


# ============================================================
# ANALYSIS C -- Ictal Morphology Consistency
# ============================================================
print("\n" + "=" * 70)
print("  Analysis C: Ictal Morphology Consistency")
print("=" * 70)
print(f"  {'Patient':<10} {'N Seiz':>7} {'Mean PSD Cos-Sim':>17} "
      f"{'Std Cos-Sim':>12} {'AUC':>8}")
print("  " + "-" * 55)

from sklearn.metrics.pairwise import cosine_similarity

consistency_data = {}
rng = np.random.default_rng(42)

for patient in PATIENT_NAMES:
    cache_path = CACHE_DIR / f"{patient}.npz"
    if not cache_path.exists():
        continue
    d = np.load(cache_path)
    X, y = d["X"], d["y"]
    X_ictal = X[y == 1]
    if len(X_ictal) < 4:
        continue

    psd = np.abs(np.fft.rfft(X_ictal, axis=-1)) ** 2
    psd_mean = psd.mean(axis=1)  # (N_windows, freq_bins), avg over channels

    norms = np.linalg.norm(psd_mean, axis=1, keepdims=True) + 1e-8
    psd_norm = psd_mean / norms

    sample_idx = rng.choice(len(psd_norm), min(100, len(psd_norm)), replace=False)
    psd_sample = psd_norm[sample_idx]
    sim_matrix = cosine_similarity(psd_sample)

    n = len(sim_matrix)
    off_diag = sim_matrix[np.triu_indices(n, k=1)]
    mean_sim = off_diag.mean()
    std_sim = off_diag.std()
    auc = LOSO_AUCS[patient]

    consistency_data[patient] = {"mean_cos_sim": float(mean_sim),
                                  "std_cos_sim": float(std_sim),
                                  "n_windows": int(len(X_ictal)), "auc": auc}
    print(f"  {patient:<10} {len(X_ictal):>7} {mean_sim:>17.4f} "
          f"{std_sim:>12.4f} {auc:>8.4f}")

hard_sims = [consistency_data[p]["mean_cos_sim"] for p in hard_patients if p in consistency_data]
easy_sims = [consistency_data[p]["mean_cos_sim"] for p in easy_patients if p in consistency_data]

pval_c = None
if hard_sims and easy_sims:
    stat, pval_c = mannwhitneyu(hard_sims, easy_sims, alternative="two-sided")
    print(f"\n  Hard mean cosine similarity : {np.mean(hard_sims):.4f}")
    print(f"  Easy mean cosine similarity : {np.mean(easy_sims):.4f}")
    print(f"  Mann-Whitney U (hard < easy): p = {pval_c:.4f}")
    print(f"  {'SIGNIFICANT' if pval_c < 0.05 else 'not significant'}")
else:
    print("  SKIPPED -- no cached windows found under", CACHE_DIR)


# ============================================================
# SYNTHESIS
# ============================================================
print("\n" + "=" * 70)
print("  Synthesis: Which Explanation Is Supported?")
print("=" * 70)

explanations = {
    "Seizure heterogeneity (CV of durations)": {
        "supported": bool(pval_a is not None and pval_a < 0.05),
        "p": pval_a,
        "direction": "Hard patients have more variable seizure durations",
    },
    "Seizure clustering (short ISI)": {
        "supported": bool(pval_b is not None and pval_b < 0.05),
        "p": pval_b,
        "direction": "Hard patients have shorter inter-seizure intervals",
    },
    "Ictal morphology inconsistency (low cosine sim)": {
        "supported": bool(pval_c is not None and pval_c < 0.05),
        "p": pval_c,
        "direction": "Hard patients have less consistent ictal PSD across seizures",
    },
}

for name, result in explanations.items():
    status = "SUPPORTED" if result["supported"] else "not supported"
    p_str = f"{result['p']:.4f}" if result["p"] is not None else "n/a"
    print(f"\n  {name}:")
    print(f"    Status : {status} (p={p_str})")
    print(f"    Direction: {result['direction']}")

out_dir = SAVE_DIR / "aggregate"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "experiment3_results.json", "w") as f:
    json.dump({
        "variability_data": {p: {k: v for k, v in d.items() if k != "durations"}
                              for p, d in variability_data.items()},
        "isi_data": {p: {k: v for k, v in d.items() if k != "isis"}
                     for p, d in isi_data.items()},
        "consistency_data": consistency_data,
        "explanations": explanations,
    }, f, indent=2, default=str)

print(f"\n  Results saved to {out_dir / 'experiment3_results.json'}")
print("=" * 70)
