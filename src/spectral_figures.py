"""
Real spectral / time-frequency figures for FATE, computed from actual EEG signal
(not schematic diagrams, not gate scalars).

Why this file exists: fig5_banddecomp.tex and fig2_faa.tex in ../figures are TikZ
architecture *schematics* -- the "band" traces in fig5 are hand-drawn sine curves,
not real filtered EEG. fig6_band_gate_heatmap.png is real data, but it's the
*learned scalar gate* g in [0,1]^4 (one number per band per patient, from
gateanalysis.py), not a spectral/time-frequency plot. Nowhere in the pipeline does
a script compute a PSD, spectrogram, or per-patient band-power figure from actual
signal. This script closes that gap using the exact same 18-channel bipolar
montage, 256 Hz resample, and band edges as data.py / model.py, so the figures are
consistent with what the model actually sees and does.

Outputs (patient chb01 by default; point --edf/--summary elsewhere for others).
Figures needing --ckpt are skipped (with a printed reason, not fabricated) if none
is given:
  raw_eeg.png                  raw multichannel EEG around a real seizure
  psd_overlay.png               PSD (Welch), ictal vs interictal windows, bands shaded
  spectrogram.png                STFT time-frequency spectrogram spanning an ictal
                                  window, seizure onset/offset marked
  band_power_bar.png             mean relative band power, ictal vs interictal
  multiband_tensor.png           multi-band EEG feature tensor via scipy bandpass
                                  (always available; the real content of the
                                  fig5_banddecomp.tex schematic)
  multiband_tensor_model.png     [--ckpt] the SAME tensor via the model's actual
                                  FrequencyBandDecomposition forward pass
  filter_bank_response.png       [--ckpt] learned vs sinc-init filter frequency
                                  response -- shows what training actually changed
  cross_band_influence.png       [--ckpt] real band x band coupling derived from
                                  the gate FC's trained weights (FATE has no
                                  literal band x band attention layer -- see the
                                  function docstring for why this is the honest
                                  substitute)
  freq_attention_heatmap.png     [--ckpt] band x time attention, from the model's
                                  actual per-band MultiheadAttention weights

Usage:
  python spectral_figures.py --edf data/chbmit_demo/chb01/chb01_03.edf \
      --summary data/chbmit_demo/chb01/chb01-summary.txt --out ../figures/spectral

Requires: numpy, scipy, matplotlib, pyedflib (lighter than mne, no compiled C deps
beyond libc). torch only needed for --ckpt.
"""
import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram, butter, sosfiltfilt

FS = 256
BANDS = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0),
         ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0)]
BAND_COLORS = {"delta": "#3b6fb0", "theta": "#2f9e8f", "alpha": "#d98e2a", "beta": "#c0392b"}

CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]


def parse_summary(path, edf_name):
    """Return [(start_sec, end_sec), ...] seizure intervals for one EDF, from a
    chbNN-summary.txt (same parser logic as data.py, standalone here)."""
    seizures, cur, starts, ends = {}, None, [], []
    with open(path, "r", errors="ignore") as fh:
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
    return seizures.get(edf_name, [])


def load_edf_bipolar(edf_path):
    """Read an EDF with pyedflib and return (sig [18,T] float32 z-scored, fs)."""
    import pyedflib
    f = pyedflib.EdfReader(edf_path)
    labels = f.getSignalLabels()
    fs_native = f.getSampleFrequency(0)
    norm = {l.upper().replace(" ", ""): i for i, l in enumerate(labels)}
    rows = []
    for want in CHANNELS:
        key = want.upper().replace(" ", "")
        idx = norm.get(key, norm.get(key + "-0"))
        if idx is None:
            raise ValueError(f"channel {want} not found in {edf_path}: {labels}")
        rows.append(f.readSignal(idx))
    f.close()
    sig = np.stack(rows).astype(np.float64)
    if int(round(fs_native)) != FS:
        from scipy.signal import resample_poly
        sig = resample_poly(sig, FS, int(round(fs_native)), axis=1)
    sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-7)
    return sig.astype(np.float32), FS


def windows_ictal_interictal(sig, intervals, win_sec=4.0, stride_sec=1.0, overlap=0.5):
    """Same windowing convention as data.py (bigger 4s window here for cleaner PSDs)."""
    win, stride = int(win_sec * FS), int(stride_sec * FS)
    n = sig.shape[1]
    ict, inter = [], []
    for s in range(0, n - win + 1, stride):
        ws, we = s / FS, (s + win) / FS
        ov = sum(max(0, min(we, e) - max(ws, st)) for st, e in intervals)
        (ict if ov > overlap * win_sec else inter).append(sig[:, s:s + win])
    return ict, inter


def bandpass(sig, lo, hi, fs=FS):
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig, axis=-1)


# ---------------------------------------------------------------- figures ----

def fig_psd_overlay(ict, inter, out, pid="chb01"):
    """PSD (Welch), averaged over channels+windows, ictal vs interictal, bands shaded."""
    def avg_psd(wins):
        if not wins:
            return None, None
        Ps = []
        for w in wins:
            f, p = welch(w, fs=FS, nperseg=min(512, w.shape[-1]), axis=-1)
            Ps.append(p.mean(0))  # average over 18 channels
        return f, np.mean(Ps, axis=0)

    f_i, p_i = avg_psd(ict)
    f_n, p_n = avg_psd(inter)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for name, lo, hi in BANDS:
        ax.axvspan(lo, hi, color=BAND_COLORS[name], alpha=0.08)
    if p_n is not None:
        ax.semilogy(f_n, p_n, label=f"interictal (n={len(inter)} win)", color="#3b6fb0", lw=1.6)
    if p_i is not None:
        ax.semilogy(f_i, p_i, label=f"ictal (n={len(ict)} win)", color="#c0392b", lw=1.6)
    ax.set_xlim(0, 40)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (log)")
    ax.set_title(f"CHB-MIT {pid}: real PSD, ictal vs interictal (Welch, 18-ch avg)")
    ax.legend(frameon=False)
    # Band labels placed AFTER plotting + in axes-fraction y-coords (not data
    # coords at the pre-plot default ylim) so they sit just inside the axes,
    # below the title, instead of colliding with it (the original bug: labels
    # were pinned to ax.get_ylim()[1] computed before any data existed).
    for name, lo, hi in BANDS:
        ax.text((lo + hi) / 2, 0.97, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color=BAND_COLORS[name])
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_spectrogram(sig, intervals, out, channel_idx=2, pad_sec=15.0, pid="chb01"):
    """STFT spectrogram spanning one real seizure (channel T7-P7 by default -- a
    common CHB-MIT ictal-onset channel), with onset/offset marked."""
    if not intervals:
        print("  [spectrogram] no seizure interval in this file, skipping")
        return
    st, en = intervals[0]
    lo_s = max(0, int((st - pad_sec) * FS))
    hi_s = min(sig.shape[1], int((en + pad_sec) * FS))
    seg = sig[channel_idx, lo_s:hi_s]
    f, t, Sxx = spectrogram(seg, fs=FS, nperseg=256, noverlap=224)
    t0 = lo_s / FS
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    im = ax.pcolormesh(t0 + t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud", cmap="magma")
    ax.axvline(st, color="cyan", lw=1.4, ls="--", label="seizure onset")
    ax.axvline(en, color="lime", lw=1.4, ls="--", label="seizure offset")
    ax.set_ylim(0, 40)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"CHB-MIT {pid}, channel {CHANNELS[channel_idx]}: real STFT spectrogram")
    fig.colorbar(im, ax=ax, label="power (dB)")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_band_power_bar(ict, inter, out, pid="chb01"):
    """Relative band power (bandpass energy / broadband energy), ictal vs interictal."""
    def rel_power(wins):
        if not wins:
            return None
        vals = {name: [] for name, _, _ in BANDS}
        for w in wins:
            total = (w ** 2).sum(axis=-1) + 1e-8
            for name, lo, hi in BANDS:
                bp = bandpass(w, lo, hi)
                vals[name].append(((bp ** 2).sum(axis=-1) / total).mean())
        return {k: (np.mean(v), np.std(v)) for k, v in vals.items()}

    ri, rn = rel_power(ict), rel_power(inter)
    names = [n for n, _, _ in BANDS]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    w = 0.35
    if rn:
        ax.bar(x - w / 2, [rn[n][0] for n in names], w, yerr=[rn[n][1] for n in names],
               label="interictal", color="#3b6fb0", capsize=3)
    if ri:
        ax.bar(x + w / 2, [ri[n][0] for n in names], w, yerr=[ri[n][1] for n in names],
               label="ictal", color="#c0392b", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Relative band power")
    ax.set_title(f"CHB-MIT {pid}: real relative band power")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_multiband_tensor(sig, intervals, out, n_channels=6, win_sec=4.0):
    """The REAL multi-band EEG feature tensor: actual bandpassed signal for a few
    channels, stacked delta/theta/alpha/beta x channel x time -- what fig5's
    hand-drawn schematic claims to depict, computed on real data instead."""
    if intervals:
        st, _ = intervals[0]
        s0 = max(0, int(st * FS))
    else:
        s0 = sig.shape[1] // 2
    seg = sig[:n_channels, s0:s0 + int(win_sec * FS)]
    t = np.arange(seg.shape[1]) / FS
    fig, axes = plt.subplots(len(BANDS), 1, figsize=(7.0, 6.0), sharex=True)
    for row, (name, lo, hi) in enumerate(BANDS):
        band_sig = bandpass(seg, lo, hi)
        ax = axes[row]
        for c in range(n_channels):
            ax.plot(t, band_sig[c] + c * 4, color=BAND_COLORS[name], lw=0.7, alpha=0.85)
        ax.set_ylabel(f"{name}\n{lo}-{hi} Hz", fontsize=9)
        ax.set_yticks([])
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Real multi-band EEG feature tensor: {n_channels}x18 channels x "
                 f"{len(BANDS)} bands x {seg.shape[1]} samples (rows offset per channel)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_raw_eeg(sig, intervals, out, n_channels=8, pad_sec=10.0, pid="chb01"):
    """Raw multichannel EEG around a real seizure -- the most basic 'what is
    actually happening' figure, missing from the paper entirely until now."""
    if intervals:
        st, en = intervals[0]
        lo_s = max(0, int((st - pad_sec) * FS))
        hi_s = min(sig.shape[1], int((en + pad_sec) * FS))
    else:
        lo_s, hi_s = 0, min(sig.shape[1], int(30 * FS))
        st = en = None
    seg = sig[:n_channels, lo_s:hi_s]
    t = np.arange(seg.shape[1]) / FS + lo_s / FS
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for c in range(n_channels):
        ax.plot(t, seg[c] + c * 6, color="#333333", lw=0.6)
        ax.text(t[0] - 0.5, c * 6, CHANNELS[c], ha="right", va="center", fontsize=7)
    if st is not None:
        ax.axvspan(st, en, color="#c0392b", alpha=0.12, label="annotated seizure")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"CHB-MIT {pid}: raw z-scored bipolar EEG, {n_channels} channels "
                 f"(real signal, not schematic)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _load_model(ckpt_path, device="cpu"):
    import torch
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import FATE
    model = FATE()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def fig_filter_bank_response(ckpt_path, out):
    """Learned vs ideal-sinc frequency response of the FrequencyBandDecomposition
    filter bank -- shows whether training actually moved the filters, or whether
    they stayed at their sinc initialization (i.e. is the "learnable" part real)."""
    if ckpt_path is None:
        print("  [filter_bank_response] skipped: no --ckpt given.")
        return
    import torch
    from model import _sinc_bandpass
    model = _load_model(ckpt_path)
    decomp = model.decomp
    fft_n = decomp.kernel_size * 8
    freqs = np.fft.rfftfreq(fft_n, d=1.0 / decomp.fs)

    fig, axes = plt.subplots(len(BANDS), 1, figsize=(6.5, 7.0), sharex=True)
    for b, (name, lo, hi) in enumerate(BANDS):
        w_learned = decomp.filters[b].weight.detach().numpy()[:, 0, :]  # (C, K)
        H_learned = np.abs(np.fft.rfft(w_learned, n=fft_n, axis=-1)).mean(0)
        h_ideal = _sinc_bandpass(lo, hi, decomp.fs, decomp.kernel_size).numpy()
        H_ideal = np.abs(np.fft.rfft(h_ideal, n=fft_n))
        ax = axes[b]
        ax.plot(freqs, H_ideal / H_ideal.max(), "--", color="gray", lw=1.2,
                label="sinc init (target)")
        ax.plot(freqs, H_learned / H_learned.max(), color=BAND_COLORS[name], lw=1.6,
                label="learned (trained)")
        ax.axvspan(lo, hi, color=BAND_COLORS[name], alpha=0.08)
        ax.set_xlim(0, 45)
        ax.set_ylabel(f"{name}\n{lo}-{hi} Hz", fontsize=9)
        if b == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Real learned filter-bank frequency response vs. sinc initialization\n"
                 "(from the trained checkpoint -- shows what training actually changed)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_cross_band_influence(ckpt_path, out):
    """Band-to-band coupling derived from the gate FC layer's REAL learned weights.

    FATE has no literal band x band self-attention layer -- FrequencyAwareAttention
    does per-band *temporal* self-attention plus a scalar cross-band sigmoid gate
    (see model.py). This is the honest substitute for a "cross-band attention
    matrix": self.faa.gate is nn.Linear(n_bands*dim -> n_bands); its weight matrix
    W has shape (n_bands_out, n_bands_in*dim). Reshaping to
    (n_bands_out, n_bands_in, dim) and taking the L2 norm over dim gives a real
    n_bands x n_bands matrix: how strongly band i's average activation drives the
    gate output that rescales band j. This is a genuine derived quantity from
    trained parameters, not a fabricated attention matrix.
    """
    if ckpt_path is None:
        print("  [cross_band_influence] skipped: no --ckpt given.")
        return
    model = _load_model(ckpt_path)
    if model.faa is None:
        print("  [cross_band_influence] skipped: this checkpoint has no FAA module "
              "(ablation=no_faa).")
        return
    W = model.faa.gate.weight.detach().numpy()   # (n_bands_out, n_bands_in*dim)
    nb = model.faa.gate.out_features
    dim = W.shape[1] // nb
    W = W.reshape(nb, nb, dim)
    infl = np.linalg.norm(W, axis=-1)             # (n_bands_out, n_bands_in)
    names = [n for n, _, _ in BANDS]

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    im = ax.imshow(infl, cmap="magma")
    ax.set_xticks(range(nb)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(nb)); ax.set_yticklabels(names)
    ax.set_xlabel("input band (activation)")
    ax.set_ylabel("output band (gated)")
    for i in range(nb):
        for j in range(nb):
            ax.text(j, i, f"{infl[i, j]:.2f}", ha="center", va="center",
                    color="white" if infl[i, j] > infl.max() * 0.5 else "black", fontsize=8)
    # Short on-figure title (full caveat lives in the docstring + notebook markdown
    # cell above this figure) -- a 3-line title on a 4.2"-tall figure was getting
    # clipped at the top edge even with tight_layout(), since tight_layout doesn't
    # always reserve enough headroom for a multi-line ax title.
    ax.set_title("Real gate-influence matrix\n(not literal attention -- see docstring)")
    fig.colorbar(im, ax=ax, label="||W_ij|| (L2 over feature dim)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_multiband_tensor_model(ckpt_path, sig, intervals, out, n_channels=6, win_sec=4.0):
    """The model's ACTUAL FrequencyBandDecomposition forward pass output (not the
    scipy-bandpass stand-in used by fig_multiband_tensor when no checkpoint is
    available) -- the real multi-band EEG feature tensor as FATE computes it."""
    if ckpt_path is None:
        print("  [multiband_tensor_model] skipped: no --ckpt given, see multiband_tensor.png "
              "(scipy bandpass stand-in) instead.")
        return
    import torch
    model = _load_model(ckpt_path)
    if intervals:
        st, _ = intervals[0]
        s0 = max(0, int(st * FS))
    else:
        s0 = sig.shape[1] // 2
    win = int(win_sec * FS)
    x = torch.from_numpy(sig[:, s0:s0 + win]).unsqueeze(0)  # (1, 18, T)
    with torch.no_grad():
        bands = model.decomp(x)[0].numpy()  # (n_bands, 18, T)
    t = np.arange(bands.shape[-1]) / FS
    fig, axes = plt.subplots(len(BANDS), 1, figsize=(7.0, 6.0), sharex=True)
    for row, (name, lo, hi) in enumerate(BANDS):
        ax = axes[row]
        for c in range(n_channels):
            ax.plot(t, bands[row, c] + c * 4, color=BAND_COLORS[name], lw=0.7, alpha=0.85)
        ax.set_ylabel(f"{name}\n{lo}-{hi} Hz", fontsize=9)
        ax.set_yticks([])
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Real model multi-band feature tensor: actual FrequencyBandDecomposition\n"
                 f"forward pass, {n_channels}/18 channels shown (trained checkpoint)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


# Approximate 10-20 scalp position for each bipolar channel (midpoint of its two
# electrodes' standard 10-20 coordinates, unit head circle, nose at +y). Good
# enough for a qualitative topoplot; not clinical-grade electrode geometry.
_CHANNEL_XY = {
    "FP1-F7": (-0.55, 0.65), "F7-T7": (-0.90, 0.20), "T7-P7": (-0.90, -0.20),
    "P7-O1": (-0.55, -0.65), "FP1-F3": (-0.35, 0.65), "F3-C3": (-0.45, 0.20),
    "C3-P3": (-0.45, -0.20), "P3-O1": (-0.35, -0.65), "FP2-F4": (0.35, 0.65),
    "F4-C4": (0.45, 0.20), "C4-P4": (0.45, -0.20), "P4-O2": (0.35, -0.65),
    "FP2-F8": (0.55, 0.65), "F8-T8": (0.90, 0.20), "T8-P8": (0.90, -0.20),
    "P8-O2": (0.55, -0.65), "FZ-CZ": (0.0, 0.20), "CZ-PZ": (0.0, -0.20),
}


def fig_per_patient_loso(csv_path, out, metric="auc"):
    """Real per-patient LOSO performance (figure 15 in the request: 'don't hide
    everything behind one mean'). Reads experiments/results/loso_fate.csv --
    already-computed, already-real, just never plotted per-patient."""
    import csv as _csv
    rows = []
    with open(csv_path) as fh:
        for r in _csv.DictReader(fh):
            if r["patient"] in ("mean", "std"):
                continue
            rows.append(r)
    rows.sort(key=lambda r: float(r[metric]))
    pids = [r["patient"] for r in rows]
    vals = [float(r[metric]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#c0392b" if v < np.median(vals) else "#3b6fb0" for v in vals]
    ax.bar(range(len(pids)), vals, color=colors)
    ax.axhline(np.mean(vals), color="black", ls="--", lw=1, label=f"mean={np.mean(vals):.3f}")
    ax.set_xticks(range(len(pids)))
    ax.set_xticklabels(pids, rotation=90, fontsize=7)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Real per-patient strict-LOSO {metric.upper()} (24 patients, sorted)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_ablation_bar(csv_path, out):
    """Real ablation comparison (experiments/data/results-abl/ablation_summary.csv):
    full FATE vs no-FAA / no-gate / fixed-bands / EEGNet baseline, with the
    Wilcoxon p-value already computed against the full model."""
    import csv as _csv
    rows = []
    with open(csv_path) as fh:
        for r in _csv.DictReader(fh):
            rows.append(r)
    names = [r["condition"] for r in rows]
    aucs = [float(r["mean_auc"]) for r in rows]
    ps = [float(r["wilcoxon_p_auc"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    colors = ["#2f9e8f" if n == "full_fate" else "#7f8c8d" for n in names]
    bars = ax.bar(names, aucs, color=colors)
    for b, p in zip(bars, ps):
        star = "*" if p < 0.05 else "ns"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, star,
                ha="center", fontsize=9)
    ax.set_ylim(min(aucs) - 0.05, max(aucs) + 0.03)
    ax.set_ylabel("mean AUC")
    ax.set_title("Real ablation study (Wilcoxon vs full FATE; * = p<0.05)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_pareto(ablation_csv, efficiency_csv, out, x="mflops"):
    """Accuracy-vs-compute Pareto plot. Joins the ablation study's real mean AUC
    (per condition) with efficiency.csv's real params/MFLOPs for the SAME
    conditions (full_fate<->fate, nofaa<->fate_no_faa, nogate<->fate_no_gate,
    fixedbands<->fate_fixed_bands, eegnet<->eegnet) -- both tables were produced
    independently on NRP, this just plots their intersection honestly."""
    import csv as _csv
    join = {"full_fate": "fate", "nofaa": "fate_no_faa", "nogate": "fate_no_gate",
            "fixedbands": "fate_fixed_bands", "eegnet": "eegnet"}
    auc = {}
    with open(ablation_csv) as fh:
        for r in _csv.DictReader(fh):
            auc[r["condition"]] = float(r["mean_auc"])
    eff = {}
    with open(efficiency_csv) as fh:
        for r in _csv.DictReader(fh):
            eff[r["model"]] = r
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for cond, model_key in join.items():
        if cond not in auc or model_key not in eff:
            continue
        xv = float(eff[model_key][x])
        yv = auc[cond]
        marker = "*" if cond == "full_fate" else "o"
        size = 220 if cond == "full_fate" else 90
        ax.scatter(xv, yv, s=size, marker=marker,
                  color="#2f9e8f" if cond == "full_fate" else "#7f8c8d", zorder=3)
        ax.annotate(cond, (xv, yv), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel(x.upper())
    ax.set_ylabel("mean AUC")
    ax.set_title(f"Real accuracy-vs-compute Pareto plot ({x}, log scale)")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_channel_band_sensitivity(ckpt_path, sig, intervals, out, win_sec=4.0):
    """Real channel x frequency-band occlusion sensitivity map: honest substitute
    for a "channel x frequency attention map" (FATE's spatial channel dimension is
    collapsed by EEGNetExtractor's depthwise conv BEFORE FrequencyAwareAttention
    runs, so there is no literal per-channel attention weight to read out -- see
    model.py). Instead, for a real ictal window, zero one channel within one
    band's decomposed signal at a time (72 = 18 ch x 4 bands forward passes) and
    measure the drop in the model's real ictal probability. A channel/band the
    model actually relies on shows a large drop when occluded.
    """
    if ckpt_path is None or not intervals:
        print("  [channel_band_sensitivity] skipped: needs --ckpt and a seizure interval.")
        return
    import torch
    model = _load_model(ckpt_path)
    st, _ = intervals[0]
    s0 = max(0, int(st * FS))
    win = int(win_sec * FS)
    x = torch.from_numpy(sig[:, s0:s0 + win]).unsqueeze(0)  # (1, 18, T)

    with torch.no_grad():
        bands = model.decomp(x)                      # (1, nb, 18, T)
        base_p = torch.softmax(_forward_from_bands(model, bands), 1)[0, 1].item()
        nb, nc = bands.shape[1], bands.shape[2]
        sens = np.zeros((nc, nb))
        for b in range(nb):
            for c in range(nc):
                occ = bands.clone()
                occ[:, b, c, :] = 0.0
                p = torch.softmax(_forward_from_bands(model, occ), 1)[0, 1].item()
                sens[c, b] = base_p - p

    names = [n for n, _, _ in BANDS]
    fig, ax = plt.subplots(figsize=(4.6, 7.2))
    im = ax.imshow(sens, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(sens).max(), vmax=np.abs(sens).max())
    ax.set_xticks(range(nb)); ax.set_xticklabels(names)
    ax.set_yticks(range(nc)); ax.set_yticklabels(CHANNELS, fontsize=7)
    ax.set_title("Real channel x band occlusion\nsensitivity (drop in ictal prob.\n"
                 "when that channel/band is zeroed)")
    fig.colorbar(im, ax=ax, label="base_p - occluded_p")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return sens


def _forward_from_bands(model, bands):
    """Replicate FATE.forward() starting from an already-decomposed band tensor
    (so occlusion figures can intervene on `bands` directly)."""
    import torch.nn.functional as F
    B, nb, C, T = bands.shape
    feats = model.extractor(bands.reshape(B * nb, C, T))
    feats = feats.reshape(B, nb, feats.size(1), feats.size(2))
    if model.faa is None:
        A = feats
    else:
        A, _ = model.faa(feats)
    Fz = model.fusion(A)
    Z = model.encoder(Fz)
    import torch
    w = torch.softmax(model.pool(Z), dim=1)
    z = (w * Z).sum(dim=1)
    return model.head(z)


def fig_topoplot_bands(sens_matrix, out):
    """Approximate EEG scalp topography, one head per band, colored by the
    channel x band occlusion sensitivity computed above (real derived quantity;
    electrode positions are an approximate 10-20 layout, not clinical geometry --
    see _CHANNEL_XY)."""
    if sens_matrix is None:
        print("  [topoplot_bands] skipped: no sensitivity matrix (needs --ckpt).")
        return
    names = [n for n, _, _ in BANDS]
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.6))
    vmax = np.abs(sens_matrix).max()
    for b, name in enumerate(names):
        ax = axes[b]
        head = plt.Circle((0, 0), 1.05, fill=False, color="black", lw=1.2)
        ax.add_patch(head)
        ax.plot([-0.08, 0, 0.08], [1.05, 1.18, 1.05], color="black", lw=1.2)  # nose
        for c, ch in enumerate(CHANNELS):
            x, y = _CHANNEL_XY[ch]
            v = sens_matrix[c, b]
            ax.scatter(x, y, s=260, c=[v], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      edgecolors="black", linewidths=0.4, zorder=3)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(name, fontsize=10)
    fig.suptitle("Real band-specific occlusion-sensitivity scalp topography "
                 "(approximate 10-20 layout)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_event_timeline(ckpt_path, sig, intervals, out, win_sec=4.0, stride_sec=1.0, device="cpu"):
    """Real continuous seizure-probability timeline: scores the model over every
    window across the whole recording in time order and overlays the ground-truth
    seizure interval(s) -- the clinically meaningful "does the alarm track the
    real event" figure, not just a summary AUC number."""
    if ckpt_path is None:
        print("  [event_timeline] skipped: no --ckpt given.")
        return
    import torch
    model = _load_model(ckpt_path, device)
    win, stride = int(win_sec * FS), int(stride_sec * FS)
    n = sig.shape[1]
    starts = list(range(0, n - win + 1, stride))
    probs = []
    with torch.no_grad():
        for i in range(0, len(starts), 128):
            batch = np.stack([sig[:, s:s + win] for s in starts[i:i + 128]])
            xb = torch.from_numpy(batch)
            p = torch.softmax(model(xb), 1)[:, 1].numpy()
            probs.append(p)
    probs = np.concatenate(probs)
    t = np.array(starts) / FS

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t, probs, color="#3b6fb0", lw=0.8)
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="0.5 threshold")
    for st, en in intervals:
        ax.axvspan(st, en, color="#c0392b", alpha=0.15,
                  label="ground-truth seizure" if (st, en) == intervals[0] else None)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("P(seizure)")
    ax.set_ylim(0, 1)
    ax.set_title("Real continuous seizure-probability timeline vs. ground truth")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_freq_attention_heatmap(ckpt_path, sig, intervals, out, device="cpu"):
    """Band x time attention from the model's REAL per-band MultiheadAttention
    weights. Requires a saved checkpoint (train.py --save-ckpt writes one to <out>/ckpt_<pid>.pt;
    see nrp/ckpt-shards/ for the 24-way sharded jobs that produce them). Refuses to
    fabricate data if no checkpoint is given."""
    if ckpt_path is None:
        print("  [freq_attention_heatmap] skipped: no --ckpt given. Pass one from "
              "nrp/ckpt-shards/ (train.py --save-ckpt) once a shard job has finished.")
        return
    import torch
    model = _load_model(ckpt_path, device)

    if intervals:
        st, _ = intervals[0]
        s0 = max(0, int(st * FS) - 512)
    else:
        s0 = sig.shape[1] // 2
    x = torch.from_numpy(sig[:, s0:s0 + 1024]).unsqueeze(0)  # (1, 18, 1024)

    attn_maps = []
    with torch.no_grad():
        bands = model.decomp(x)
        B, nb, C, T = bands.shape
        feats = model.extractor(bands.reshape(B * nb, C, T))
        feats = feats.reshape(B, nb, feats.size(1), feats.size(2))
        for b, attn in enumerate(model.faa.attn):
            xb = feats[:, b]
            _, w = attn(xb, xb, xb, need_weights=True, average_attn_weights=True)
            attn_maps.append(w[0].mean(0).numpy())  # (T', T') attn -> diagonal-ish energy per step

    band_names = [n for n, _, _ in BANDS]
    energy = np.stack(attn_maps)  # (n_bands, T') -- attn_maps entries are already
    # (T',) per band (w[0].mean(0) above); an earlier version called .mean(0) again
    # here, collapsing each to a 0-d scalar and breaking energy.shape[1] below.
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    im = ax.imshow(energy, aspect="auto", cmap="magma",
                   extent=[0, energy.shape[1], 0, len(band_names)])
    ax.set_yticks(np.arange(len(band_names)) + 0.5)
    ax.set_yticklabels(band_names[::-1])
    ax.set_xlabel("token (time) index")
    ax.set_title("Real per-band self-attention energy (from the trained checkpoint)")
    fig.colorbar(im, ax=ax, label="mean attention received")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default="spectral_out")
    ap.add_argument("--ckpt", default=None, help="optional FATE checkpoint (.pt state_dict)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    edf_name = os.path.basename(args.edf)
    pid = edf_name.split("_")[0]
    intervals = parse_summary(args.summary, edf_name)
    print(f"loaded seizure intervals for {edf_name}: {intervals}")

    sig, fs = load_edf_bipolar(args.edf)
    print(f"signal shape {sig.shape} @ {fs} Hz")

    ict, inter = windows_ictal_interictal(sig, intervals)
    print(f"{len(ict)} ictal windows, {len(inter)} interictal windows (4s, 1s stride)")

    fig_raw_eeg(sig, intervals, os.path.join(args.out, "raw_eeg.png"), pid=pid)
    fig_psd_overlay(ict, inter, os.path.join(args.out, "psd_overlay.png"), pid=pid)
    fig_spectrogram(sig, intervals, os.path.join(args.out, "spectrogram.png"), pid=pid)
    fig_band_power_bar(ict, inter, os.path.join(args.out, "band_power_bar.png"), pid=pid)
    fig_multiband_tensor(sig, intervals, os.path.join(args.out, "multiband_tensor.png"))
    fig_multiband_tensor_model(args.ckpt, sig, intervals,
                               os.path.join(args.out, "multiband_tensor_model.png"))
    fig_filter_bank_response(args.ckpt, os.path.join(args.out, "filter_bank_response.png"))
    fig_cross_band_influence(args.ckpt, os.path.join(args.out, "cross_band_influence.png"))
    fig_freq_attention_heatmap(args.ckpt, sig, intervals,
                               os.path.join(args.out, "freq_attention_heatmap.png"))
    print(f"done -> {args.out}/")


if __name__ == "__main__":
    main()
