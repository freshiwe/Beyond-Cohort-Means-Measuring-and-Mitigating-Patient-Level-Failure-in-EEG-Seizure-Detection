"""
TFCANet: Time-Frequency Cross-Attention Network for epileptic seizure detection.

Reference: "A time-frequency cross-attention network model for epileptic seizure
detection", Scientific Reports (Nature), ~2026-03. https://www.nature.com/articles/s41598-026-41636-7
(PMC13111615). Added as a real literature baseline for the FATEv3 SOTA comparison
-- see project memory project_seizuredetection_spectral_gap.md.

Architecture (per the paper, adapted here to this repo's I/O contract --
(B, 18, 1024) 18-channel bipolar EEG -> (B, 2) seizure/non-seizure logits,
vs. the paper's 16-channel/2000-sample/5-class TUSZ-style setup):

  Frequency branch : short-window FFT magnitude (STFT-like) -> 4 stacked 1D
                      SE-Residual conv blocks (channel progression
                      128->256->256->512, stride 2, kernel 3, each block =
                      conv+BN+ReLU + squeeze-excitation gate + residual skip)
                      -> 1x1 conv down to 64 channels.
  Time branch      : 4 alternating ResWMSA blocks -- an SE-Residual local
                      feature extraction stage followed by Windowed
                      Multi-Head Self-Attention (non-overlapping windows,
                      4 heads, window size halving each block: 128/64/32/16
                      for this repo's 1024-sample input, vs. the paper's
                      200/100/50/25 for their 2000-sample input) -- windowed
                      attention keeps the O(L^2) full-attention cost down to
                      O(L*w) per the paper.
  Fusion           : both branches projected to a common (64, L) shape, then
                      cross-attention with Q,K from the TIME branch and V from
                      the FREQUENCY branch ("time context selects the most
                      relevant spectral characteristics" -- paper's phrasing),
                      not simple concatenation.
  Head             : global average pool + linear classifier.

Run `python tfcanet.py` to sanity-check output shape and parameter count.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1D(nn.Module):
    """Squeeze-and-excitation gate: global-avg-pool -> 2-layer MLP -> sigmoid
    -> per-channel rescale. Matches the paper's SE description ("learned
    weights are then multiplied element-wise with the original feature map
    channels")."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x):            # (B, C, L)
        s = x.mean(dim=-1)
        w = self.fc(s).unsqueeze(-1)
        return x * w


class SEResidualBlock1D(nn.Module):
    """conv -> BN -> ReLU -> SE gate, with a (possibly strided/projected)
    residual skip -- the paper's basic building block, used in both branches."""

    def __init__(self, in_ch, out_ch, stride=1, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.se = SEBlock1D(out_ch)
        self.proj = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                                    nn.BatchNorm1d(out_ch))
                     if (in_ch != out_ch or stride != 1) else nn.Identity())

    def forward(self, x):
        h = F.relu(self.bn(self.conv(x)), inplace=True)
        h = self.se(h)
        return F.relu(h + self.proj(x), inplace=True)


class WindowedMHSA(nn.Module):
    """Non-overlapping windowed multi-head self-attention (paper's ResWMSA
    attention half): splits the sequence into windows of size `window`, runs
    independent MHSA per window, concatenates back. Reduces full O(L^2)
    attention to O(L*window)."""

    def __init__(self, dim, window, heads=4, drop=0.1):
        super().__init__()
        self.window = window
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):            # x: (B, L, D)
        B, L, D = x.shape
        w = self.window
        pad = (w - L % w) % w
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        Lp = x.shape[1]
        xw = x.reshape(B * (Lp // w), w, D)
        out, _ = self.attn(xw, xw, xw)
        out = xw + out                     # residual (the "Res" in ResWMSA)
        out = self.norm(out)
        out = out.reshape(B, Lp, D)[:, :L]
        return out


class ResWMSABlock(nn.Module):
    """SE-Residual local extraction -> Windowed MHSA global mixing, one block
    of the paper's time-domain branch."""

    def __init__(self, in_ch, out_ch, window, stride=1, heads=4, drop=0.1):
        super().__init__()
        self.conv = SEResidualBlock1D(in_ch, out_ch, stride=stride)
        self.wmsa = WindowedMHSA(out_ch, window, heads=heads, drop=drop)

    def forward(self, x):            # x: (B, C, L)
        h = self.conv(x)                       # (B, C', L')
        h_t = h.transpose(1, 2)                # (B, L', C')
        h_t = self.wmsa(h_t)
        return h_t.transpose(1, 2)             # (B, C', L')


class TFCANet(nn.Module):
    def __init__(self, n_channels=18, n_classes=2, fft_win=64, fft_hop=32,
                 drop=0.1):
        super().__init__()
        self.fft_win = fft_win
        self.fft_hop = fft_hop
        n_freq_bins = fft_win // 2 + 1

        # --- frequency branch: STFT magnitude -> embed -> 4 SE-Residual blocks ---
        self.freq_embed = nn.Sequential(
            nn.Conv1d(n_channels * n_freq_bins, 128, 1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        freq_ch = [128, 256, 256, 512]
        self.freq_blocks = nn.ModuleList()
        in_ch = 128
        for out_ch in freq_ch:
            self.freq_blocks.append(SEResidualBlock1D(in_ch, out_ch, stride=2))
            in_ch = out_ch
        self.freq_down = nn.Conv1d(freq_ch[-1], 64, 1, bias=False)

        # --- time branch: embed -> 4 alternating ResWMSA blocks ---
        self.time_embed = nn.Sequential(
            nn.Conv1d(n_channels, 64, 1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        # window sizes halve each block (paper: 200/100/50/25 for T=2000;
        # here: 128/64/32/16 for T=1024), channels grow toward 512 like the
        # frequency branch.
        time_ch = [128, 256, 256, 512]
        windows = [128, 64, 32, 16]
        self.time_blocks = nn.ModuleList()
        in_ch = 64
        for out_ch, win in zip(time_ch, windows):
            self.time_blocks.append(ResWMSABlock(in_ch, out_ch, win, stride=2, drop=drop))
            in_ch = out_ch
        self.time_down = nn.Conv1d(time_ch[-1], 64, 1, bias=False)

        # --- alignment + cross-attention fusion (paper: Q,K=time, V=freq) ---
        self.time_fc = nn.Linear(64, 64)
        self.freq_fc = nn.Linear(64, 64)
        self.cross_attn = nn.MultiheadAttention(64, 4, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(64)

        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(drop), nn.Linear(32, n_classes))

    def _stft_magnitude(self, x):
        """x: (B, C, T) -> (B, C*F, T') windowed FFT magnitude, concatenated
        across channels (paper: "points are subsequently concatenated to
        obtain the initial frequency domain data")."""
        B, C, T = x.shape
        win = torch.hann_window(self.fft_win, device=x.device)
        flat = x.reshape(B * C, T)
        spec = torch.stft(flat, n_fft=self.fft_win, hop_length=self.fft_hop,
                          window=win, return_complex=True).abs()   # (B*C, F, T')
        F_bins, Tp = spec.shape[1], spec.shape[2]
        spec = spec.reshape(B, C * F_bins, Tp)
        return spec

    def forward(self, x):            # x: (B, 18, 1024)
        # frequency branch
        f = self._stft_magnitude(x)              # (B, C*F, T')
        f = self.freq_embed(f)
        for blk in self.freq_blocks:
            f = blk(f)
        f = self.freq_down(f)                     # (B, 64, Lf)

        # time branch
        t = self.time_embed(x)                    # (B, 64, T)
        for blk in self.time_blocks:
            t = blk(t)
        t = self.time_down(t)                      # (B, 64, Lt)

        # align to a common length (adaptive pool to the shorter of the two)
        L = min(f.shape[-1], t.shape[-1])
        f = F.adaptive_avg_pool1d(f, L).transpose(1, 2)   # (B, L, 64)
        t = F.adaptive_avg_pool1d(t, L).transpose(1, 2)   # (B, L, 64)
        f = self.freq_fc(f)
        t = self.time_fc(t)

        # cross-attention: Q,K = time (context), V = frequency
        out, attn_weights = self.cross_attn(t, t, f, need_weights=True,
                                            average_attn_weights=True)
        out = self.norm(out + t)
        z = out.mean(dim=1)                        # global average pool over time
        return self.head(z)


def build_tfcanet(**kw):
    return TFCANet(**kw)


if __name__ == "__main__":
    x = torch.randn(2, 18, 1024)
    m = build_tfcanet()
    y = m(x)
    n = sum(p.numel() for p in m.parameters())
    assert tuple(y.shape) == (2, 2)
    print(f"tfcanet   logits {tuple(y.shape)}  params {n:,}")
