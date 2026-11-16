"""
FATE: Frequency-Aware Transformer for cross-patient EEG seizure detection.

Reference implementation of the architecture described in the paper (Figs 1-5).
Shapes for the default config: input (B, 18, 1024) -> tokens (B, 32, 128).

v2 filter bank: depthwise Conv1d filters are initialized with windowed-sinc
bandpass responses at the clinical EEG band edges (delta/theta/alpha/beta) and
a frequency regularization loss penalizes out-of-band energy during training.
Without this, default Kaiming-uniform init produces arbitrary filters that give
the FAA module nothing meaningful to attend to per band.

Run `python model.py` to sanity-check shapes and parameter count (~617k).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# CHB-MIT sampling rate and clinical EEG band edges (Hz)
_FS = 256
_BAND_EDGES = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)]


def _sinc_bandpass(f_low: float, f_high: float, fs: int, N: int) -> torch.Tensor:
    """Hamming-windowed sinc bandpass FIR: bandpass = LP(f_high) - LP(f_low)."""
    n = np.arange(N, dtype=np.float64) - N // 2
    # np.sinc is the normalized sinc: sinc(x) = sin(πx)/(πx)
    h = (2 * f_high / fs) * np.sinc(2 * f_high / fs * n) \
      - (2 * f_low  / fs) * np.sinc(2 * f_low  / fs * n)
    h *= np.hamming(N)
    h /= np.abs(h).sum() + 1e-8
    return torch.tensor(h, dtype=torch.float32)


class FrequencyBandDecomposition(nn.Module):
    """Learnable depthwise Conv1d filter bank: (B,C,T) -> (B,n_bands,C,T).

    Filters are initialized at the clinical EEG band edges (delta/theta/alpha/beta)
    using windowed-sinc bandpass FIR responses, then kept learnable so training can
    fine-tune the response to the CHB-MIT data. freq_reg_loss() returns an
    out-of-band energy penalty (averaged over bands) that is added to the
    training loss to keep filters near their target band.
    """

    def __init__(self, n_channels=18, n_bands=4, kernel_size=51,
                 fs=_FS, band_edges=None):
        super().__init__()
        self.n_bands = n_bands
        self.kernel_size = kernel_size
        self.fs = fs
        self.band_edges = (band_edges or _BAND_EDGES)[:n_bands]
        pad = kernel_size // 2
        self.filters = nn.ModuleList(
            nn.Conv1d(n_channels, n_channels, kernel_size,
                      padding=pad, groups=n_channels, bias=False)
            for _ in range(n_bands)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(n_channels) for _ in range(n_bands))
        self._init_bandpass()

    def _init_bandpass(self):
        """Copy the sinc bandpass response into every channel of each filter."""
        for b, (f_low, f_high) in enumerate(self.band_edges):
            h = _sinc_bandpass(f_low, f_high, self.fs, self.kernel_size)
            # depthwise Conv1d weight: (n_channels, 1, kernel_size)
            with torch.no_grad():
                self.filters[b].weight.copy_(
                    h.view(1, 1, -1).expand_as(self.filters[b].weight))

    def freq_reg_loss(self) -> torch.Tensor:
        """Out-of-band energy ratio, averaged over bands (scalar, in [0,1]).

        Penalizes filter weight that drifts outside the target EEG band.
        Multiply by lambda_freq (~0.01) and add to training loss.
        """
        device = self.filters[0].weight.device
        fft_n = self.kernel_size * 8          # zero-pad for frequency resolution
        freqs = torch.fft.rfftfreq(fft_n, d=1.0 / self.fs).to(device)
        loss = torch.zeros(1, device=device)
        for b, (f_low, f_high) in enumerate(self.band_edges):
            w = self.filters[b].weight        # (C, 1, kernel_size)
            W = torch.fft.rfft(w, n=fft_n, dim=-1)
            power = W.abs().pow(2).mean(dim=(0, 1))   # (fft_n//2+1,)
            in_band = (freqs >= f_low) & (freqs <= f_high)
            total = power.sum() + 1e-8
            loss = loss + power[~in_band].sum() / total
        return loss / self.n_bands

    def forward(self, x):
        bands = [bn(f(x)) for f, bn in zip(self.filters, self.bns)]
        return torch.stack(bands, dim=1)  # (B, n_bands, C, T)


class EEGNetExtractor(nn.Module):
    """Shared EEGNet-style feature extractor: (B,C,T) -> (B,T',D)."""

    def __init__(self, n_channels=18, F1=16, depth_mult=2, D=128, drop=0.25):
        super().__init__()
        F2 = F1 * depth_mult
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, 51), padding=(0, 25), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 15), padding=(0, 7), groups=F2, bias=False),
            nn.Conv2d(F2, D, (1, 1), bias=False),
            nn.BatchNorm2d(D), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop),
        )

    def forward(self, x):
        x = x.unsqueeze(1)              # (B,1,C,T)
        x = self.temporal(x)           # (B,F1,C,T)
        x = self.depthwise(x)          # (B,F2,1,T/4)
        x = self.separable(x)          # (B,D,1,T/32)
        return x.squeeze(2).transpose(1, 2)  # (B, T', D)


class FrequencyAwareAttention(nn.Module):
    """Per-band self-attention + cross-band sigmoid gating."""

    def __init__(self, n_bands=4, dim=128, heads=4, drop=0.1, use_gate=True):
        super().__init__()
        self.use_gate = use_gate
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
            for _ in range(n_bands)
        )
        self.gate = nn.Linear(n_bands * dim, n_bands)

    def forward(self, x):               # x: (B, n_bands, T', D)
        A = []
        for b, attn in enumerate(self.attn):
            xb = x[:, b]
            ab, _ = attn(xb, xb, xb)
            A.append(ab)
        A = torch.stack(A, dim=1)        # (B, nb, T', D)
        ctx = A.mean(dim=2).flatten(1)   # (B, nb*D)
        g = torch.sigmoid(self.gate(ctx))  # (B, nb)
        if self.use_gate:
            A = A * g[:, :, None, None]
        return A, g


class BandFusion(nn.Module):
    def __init__(self, n_bands=4, dim=128):
        super().__init__()
        self.proj = nn.Linear(n_bands * dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, A):               # (B, nb, T', D)
        B, nb, T, D = A.shape
        x = A.permute(0, 2, 1, 3).reshape(B, T, nb * D)
        return F.gelu(self.norm(self.proj(x)))  # (B, T', D)


ABLATIONS = ("none", "no_faa", "no_gate", "fixed_bands", "no_transformer")


class FATE(nn.Module):
    """Full model and its ablations, selected by `ablation`:
      none           : full FATE
      no_faa         : drop per-band attention + gating (bands go straight to fusion)
      no_gate        : keep per-band attention, disable cross-band gating (g=1)
      fixed_bands    : freeze the learnable filter bank (non-adaptive decomposition)
      no_transformer : drop the Transformer encoder (fused tokens -> pooling)
    """

    def __init__(self, n_channels=18, n_bands=4, dim=128, heads=4,
                 enc_layers=2, ffn=256, drop=0.25, n_classes=2, ablation="none"):
        super().__init__()
        assert ablation in ABLATIONS, ablation
        self.ablation = ablation
        self.decomp = FrequencyBandDecomposition(n_channels, n_bands)
        if ablation == "fixed_bands":
            for p in self.decomp.parameters():
                p.requires_grad_(False)
        self.extractor = EEGNetExtractor(n_channels, D=dim, drop=drop)
        self.faa = (None if ablation == "no_faa"
                    else FrequencyAwareAttention(n_bands, dim, heads,
                                                 use_gate=(ablation != "no_gate")))
        self.fusion = BandFusion(n_bands, dim)
        if ablation == "no_transformer":
            self.encoder = nn.Identity()
        else:
            layer = nn.TransformerEncoderLayer(
                dim, heads, ffn, dropout=drop, activation="gelu",
                batch_first=True, norm_first=True)
            # enable_nested_tensor=False: the nested-tensor fast path only helps padded
            # variable-length batches (we use fixed-length windows) and is auto-disabled
            # under norm_first=True anyway. Setting it explicitly silences the PyTorch
            # UserWarning with no change to outputs.
            self.encoder = nn.TransformerEncoder(layer, enc_layers,
                                                 enable_nested_tensor=False)
        self.pool = nn.Linear(dim, 1)
        self.head = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, n_classes))

    def forward(self, x, return_gate=False):
        bands = self.decomp(x)                       # (B, nb, C, T)
        B, nb, C, T = bands.shape
        feats = self.extractor(bands.reshape(B * nb, C, T))
        feats = feats.reshape(B, nb, feats.size(1), feats.size(2))
        if self.faa is None:                         # no_faa
            A = feats
            g = torch.ones(B, nb, device=x.device)
        else:
            A, g = self.faa(feats)                   # use_gate handles no_gate
        Fz = self.fusion(A)                          # (B, T', D)
        Z = self.encoder(Fz)                         # (B, T', D)
        w = torch.softmax(self.pool(Z), dim=1)       # (B, T', 1)
        z = (w * Z).sum(dim=1)                       # (B, D)
        logits = self.head(z)
        return (logits, g) if return_gate else logits


if __name__ == "__main__":
    x = torch.randn(2, 18, 1024)
    for abl in ABLATIONS:
        m = FATE(ablation=abl)
        logits, g = m(x, return_gate=True)
        n = sum(p.numel() for p in m.parameters())
        assert tuple(logits.shape) == (2, 2), abl
        print(f"{abl:16s} logits {tuple(logits.shape)} gate {tuple(g.shape)} params {n:,}")
