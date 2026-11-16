"""
FATE-v3: makes FATE's "frequency-aware" claims literally true.

model.py's FrequencyAwareAttention does per-band *temporal* self-attention plus a
bare Linear+sigmoid scalar gate -- there is no layer where bands attend to each
other, and no per-channel attention weight anywhere (EEGNetExtractor's depthwise
conv collapses the 18-channel dimension before frequency attention ever runs).
Both facts were confirmed by direct code reading, and are why spectral_figures.py
had to build "honest substitute" figures (a gate-weight-derived influence matrix,
an occlusion-sensitivity map) instead of plotting real attention weights.

v3 adds two small modules, grounded in 2025-2026 literature (see plan doc /
project memory project_seizuredetection_spectral_gap.md), that make those two
figures literal instead of derived:

  CrossBandAttention  : bands become tokens; one real MultiheadAttention layer
                         lets them attend to each other (replaces the scalar
                         gate). This is the actual "cross-band attention
                         matrix" / "frequency-band self-attention matrix" from
                         the literature and the request -- not derived.
  SEChannelAttention   : a squeeze-and-excitation block on the raw per-band
                         channel signal, BEFORE EEGNetExtractor collapses
                         channels, giving a real per-channel attention weight
                         (informed by TFCANet's SE-Residual design).

Also extends the band edges to 5 (adds a 30-45 Hz band, staying under 60 Hz US
mains interference) and adds an optional Hilbert-transform phase channel
(phase-amplitude coupling is a known epilepsy biomarker; informed by dual-branch
CNN-ViT phase+power work) -- both ablatable.

Everything else (the learnable sinc filter bank, EEGNetExtractor, BandFusion,
Transformer encoder, attention pooling, head) is reused as-is from model.py --
those parts are real and were not the problem.

Run `python model_v3.py` to sanity-check shapes and parameter counts.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import _sinc_bandpass, EEGNetExtractor

_FS = 256
_BAND_EDGES_V3 = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
_BAND_EDGES_V3_4 = _BAND_EDGES_V3[:4]   # for the no_5th_band ablation


class FrequencyBandDecompositionV3(nn.Module):
    """Same learnable sinc-init depthwise filter bank as model.py, just built
    for a configurable band count (5 by default, 4 for the no_5th_band
    ablation) so this file doesn't duplicate FrequencyBandDecomposition."""

    def __init__(self, n_channels=18, n_bands=5, kernel_size=51, fs=_FS, band_edges=None):
        super().__init__()
        self.n_bands = n_bands
        self.kernel_size = kernel_size
        self.fs = fs
        self.band_edges = (band_edges or _BAND_EDGES_V3)[:n_bands]
        pad = kernel_size // 2
        self.filters = nn.ModuleList(
            nn.Conv1d(n_channels, n_channels, kernel_size,
                      padding=pad, groups=n_channels, bias=False)
            for _ in range(n_bands)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(n_channels) for _ in range(n_bands))
        self._init_bandpass()

    def _init_bandpass(self):
        for b, (f_low, f_high) in enumerate(self.band_edges):
            h = _sinc_bandpass(f_low, f_high, self.fs, self.kernel_size)
            with torch.no_grad():
                self.filters[b].weight.copy_(
                    h.view(1, 1, -1).expand_as(self.filters[b].weight))

    def freq_reg_loss(self) -> torch.Tensor:
        device = self.filters[0].weight.device
        fft_n = self.kernel_size * 8
        freqs = torch.fft.rfftfreq(fft_n, d=1.0 / self.fs).to(device)
        loss = torch.zeros(1, device=device)
        for b, (f_low, f_high) in enumerate(self.band_edges):
            w = self.filters[b].weight
            W = torch.fft.rfft(w, n=fft_n, dim=-1)
            power = W.abs().pow(2).mean(dim=(0, 1))
            in_band = (freqs >= f_low) & (freqs <= f_high)
            total = power.sum() + 1e-8
            loss = loss + power[~in_band].sum() / total
        return loss / self.n_bands

    def forward(self, x):
        bands = [bn(f(x)) for f, bn in zip(self.filters, self.bns)]
        return torch.stack(bands, dim=1)  # (B, n_bands, C, T)


class HilbertPhaseAmplitude(nn.Module):
    """Fixed (non-learned) analytic-signal transform: (B,C,T) -> (B,2C,T) with
    instantaneous amplitude and phase interleaved per channel. FFT-based Hilbert
    transform (no external dependency). Phase-amplitude coupling across bands is
    a known epilepsy biomarker that amplitude-only FATE (model.py) cannot see.
    """

    def forward(self, x):
        B, C, T = x.shape
        # Complex FFT under fp16/bf16 autocast is numerically unstable (produces
        # NaN -- confirmed by the v3 smoke test) and PyTorch's ComplexHalf support
        # is explicitly experimental. Force this transform to run in float32
        # regardless of the enclosing autocast context, then cast back.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            Xf = torch.fft.fft(x32, dim=-1)
            h = torch.zeros(T, device=x.device, dtype=torch.float32)
            if T % 2 == 0:
                h[0] = h[T // 2] = 1
                h[1:T // 2] = 2
            else:
                h[0] = 1
                h[1:(T + 1) // 2] = 2
            analytic = torch.fft.ifft(Xf * h)
            amp = analytic.abs()
            phase = analytic.angle()
        out = torch.stack([amp, phase], dim=2).reshape(B, 2 * C, T).to(x.dtype)
        return out


class SEChannelAttention(nn.Module):
    """Squeeze-and-excitation on the raw per-band channel signal, BEFORE
    EEGNetExtractor's depthwise conv collapses the channel dimension -- this is
    the real per-channel attention weight for the channel x frequency figure
    (model.py has none; occlusion sensitivity was an honest substitute for
    exactly this reason). (B,C,T) -> (B,C,T) rescaled, weights (B,C) returned.
    """

    def __init__(self, n_channels, reduction=4):
        super().__init__()
        hidden = max(4, n_channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_channels), nn.Sigmoid(),
        )

    def forward(self, x):               # x: (B, C, T)
        s = x.mean(dim=-1)               # (B, C) squeeze
        w = self.fc(s)                   # (B, C) excite
        return x * w.unsqueeze(-1), w


class CrossBandAttention(nn.Module):
    """Bands become tokens; one real MultiheadAttention layer lets them attend
    to each other. Replaces FrequencyAwareAttention's scalar Linear+sigmoid gate
    (model.py) with genuine learned band-to-band attention weights -- the
    literal "cross-band attention matrix" from the literature/request.
    """

    def __init__(self, n_bands, dim, heads=4, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, band_summary):     # (B, n_bands, D) -- per-band pooled features
        out, weights = self.attn(band_summary, band_summary, band_summary,
                                 need_weights=True, average_attn_weights=True)
        # weights: (B, n_bands, n_bands) real cross-band attention matrix
        gate = torch.sigmoid(self.norm(out))   # (B, n_bands, D) per-band, per-feature scale
        return gate, weights


class FrequencyAwareAttentionV3(nn.Module):
    """Per-band temporal self-attention (reused idea from model.py, still real
    and still good) + CrossBandAttention instead of the old scalar gate."""

    def __init__(self, n_bands, dim, heads=4, drop=0.1, use_cross_band=True):
        super().__init__()
        self.use_cross_band = use_cross_band
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
            for _ in range(n_bands)
        )
        self.cross_band = CrossBandAttention(n_bands, dim, heads, drop) if use_cross_band else None
        self.scalar_gate = None if use_cross_band else nn.Linear(n_bands * dim, n_bands)

    def forward(self, x):                # x: (B, n_bands, T', D)
        A = []
        for b, attn in enumerate(self.attn):
            xb = x[:, b]
            ab, _ = attn(xb, xb, xb)
            A.append(ab)
        A = torch.stack(A, dim=1)         # (B, nb, T', D)
        summary = A.mean(dim=2)           # (B, nb, D) time-pooled, same as model.py's gate input

        if self.use_cross_band:
            gate, cross_band_weights = self.cross_band(summary)   # (B, nb, D), (B, nb, nb)
            A = A * gate.unsqueeze(2)                              # broadcast over time
            g_scalar = gate.mean(dim=-1)                           # (B, nb) for logging parity w/ v2
        else:
            ctx = summary.flatten(1)
            g_scalar = torch.sigmoid(self.scalar_gate(ctx))        # (B, nb) -- v2-equivalent fallback
            A = A * g_scalar[:, :, None, None]
            cross_band_weights = None
        return A, g_scalar, cross_band_weights


class BandFusionV3(nn.Module):
    def __init__(self, n_bands, dim):
        super().__init__()
        self.proj = nn.Linear(n_bands * dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, A):
        B, nb, T, D = A.shape
        x = A.permute(0, 2, 1, 3).reshape(B, T, nb * D)
        return F.gelu(self.norm(self.proj(x)))


ABLATIONS_V3 = ("none", "no_cross_band_attn", "no_channel_se", "no_phase", "no_5th_band",
               "lean")
# lean: combines the two changes that did NOT show a positive margin in the
# n=3 fast-ablation (no_cross_band_attn +0.0095, no_5th_band +0.0230 vs
# full_v3) with keeping the one that did (phase: no_phase was -0.0406, the
# only condition that clearly hurt when removed). This is the leaner variant
# to run the DEFINITIVE full 24-fold LOSO head-to-head against full_v3 --
# see v3_ablation_summary.csv and project memory for the n=3 result that
# motivated this.


class FATEv3(nn.Module):
    """v3 and its ablations, selected by `ablation`:
      none               : full v3 (5 bands, phase, SE channel attn, cross-band attn)
      no_cross_band_attn : real per-channel/phase/5-band features, but the old
                           scalar Linear+sigmoid gate instead of CrossBandAttention
                           -- direct, controlled test of whether cross-band
                           attention earns its parameters
      no_channel_se      : drop the SE channel-attention block
      no_phase           : amplitude-only per band (drop the Hilbert phase channel)
      no_5th_band        : 4 clinical bands only (delta/theta/alpha/beta), like v2
      lean               : phase only -- 4 bands (no_5th_band) + no cross-band
                           attention (scalar gate instead), SE channel attn kept.
                           The leaner competitor to full_v3 for the definitive
                           24-fold head-to-head (see ABLATIONS_V3 comment).
    """

    def __init__(self, n_channels=18, dim=128, heads=4, enc_layers=2, ffn=256,
                 drop=0.25, n_classes=2, ablation="none"):
        super().__init__()
        assert ablation in ABLATIONS_V3, ablation
        self.ablation = ablation
        n_bands = 4 if ablation in ("no_5th_band", "lean") else 5
        self.use_phase = ablation != "no_phase"
        self.use_se = ablation != "no_channel_se"
        use_cross_band = ablation not in ("no_cross_band_attn", "lean")

        edges = _BAND_EDGES_V3_4 if n_bands == 4 else _BAND_EDGES_V3
        self.decomp = FrequencyBandDecompositionV3(n_channels, n_bands, band_edges=edges)
        self.hilbert = HilbertPhaseAmplitude() if self.use_phase else None
        extractor_channels = n_channels * 2 if self.use_phase else n_channels
        self.se = SEChannelAttention(extractor_channels) if self.use_se else None
        self.extractor = EEGNetExtractor(extractor_channels, D=dim, drop=drop)
        self.faa = FrequencyAwareAttentionV3(n_bands, dim, heads, use_cross_band=use_cross_band)
        self.fusion = BandFusionV3(n_bands, dim)
        layer = nn.TransformerEncoderLayer(
            dim, heads, ffn, dropout=drop, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, enc_layers, enable_nested_tensor=False)
        self.pool = nn.Linear(dim, 1)
        self.head = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, n_classes))

    def forward(self, x, return_gate=False, return_attn=False):
        bands = self.decomp(x)                        # (B, nb, C, T)
        B, nb, C, T = bands.shape
        flat = bands.reshape(B * nb, C, T)
        if self.hilbert is not None:
            flat = self.hilbert(flat)                  # (B*nb, 2C, T)
        se_weights = None
        if self.se is not None:
            flat, se_weights = self.se(flat)            # (B*nb, C', T), (B*nb, C')
        feats = self.extractor(flat)
        feats = feats.reshape(B, nb, feats.size(1), feats.size(2))
        A, g, cross_band_weights = self.faa(feats)
        Fz = self.fusion(A)
        Z = self.encoder(Fz)
        w = torch.softmax(self.pool(Z), dim=1)
        z = (w * Z).sum(dim=1)
        logits = self.head(z)
        if return_attn:
            se = se_weights.reshape(B, nb, -1) if se_weights is not None else None
            return logits, {"gate": g, "cross_band": cross_band_weights, "channel_se": se}
        return (logits, g) if return_gate else logits


def build_v3(ablation="none", **kw):
    return FATEv3(ablation=ablation, **kw)


if __name__ == "__main__":
    x = torch.randn(2, 18, 1024)
    for abl in ABLATIONS_V3:
        m = FATEv3(ablation=abl)
        logits, attn = m(x, return_attn=True)
        n = sum(p.numel() for p in m.parameters())
        assert tuple(logits.shape) == (2, 2), abl
        cb = attn["cross_band"]
        cb_shape = tuple(cb.shape) if cb is not None else None
        se = attn["channel_se"]
        se_shape = tuple(se.shape) if se is not None else None
        print(f"{abl:20s} logits {tuple(logits.shape)} gate {tuple(attn['gate'].shape)} "
              f"cross_band {cb_shape} channel_se {se_shape} params {n:,}")
