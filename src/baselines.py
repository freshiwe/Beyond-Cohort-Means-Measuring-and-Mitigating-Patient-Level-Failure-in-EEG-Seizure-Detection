"""
Baseline models for a matched-protocol comparison, run through the same
LOSO harness as FATE (same preprocessing, splits, SMOTE, metrics).

  eegnet  : EEGNet (Lawhern et al. 2018), the compact reference backbone.
  cnnlstm : 1D CNN + BiLSTM (a common hybrid), ~1-2M params.
  runet   : Ru et al. style STFT -> CNN -> multi-head self-attention.

All take (B, C=18, T=1024) and return (B, 2) logits, so build_model() can
swap any of them (or FATE) into train.py.
"""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, C=18, T=1024, F1=8, D=2, F2=16, drop=0.25, n_classes=2):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop),
        )
        self.classify = nn.Linear(F2 * (T // 32), n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block2(self.block1(x))
        return self.classify(x.flatten(1))


class CNNLSTM(nn.Module):
    def __init__(self, C=18, hidden=128, drop=0.3, n_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(C, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
        )
        self.lstm = nn.LSTM(128, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=drop)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, n_classes))

    def forward(self, x):
        h = self.cnn(x).transpose(1, 2)     # (B, T', 128)
        out, _ = self.lstm(h)               # (B, T', 2*hidden)
        return self.head(out.mean(1))


class RuNet(nn.Module):
    """STFT time-frequency input -> CNN -> lightweight multi-head self-attention."""

    def __init__(self, C=18, n_fft=128, dim=128, heads=8, drop=0.25, n_classes=2):
        super().__init__()
        self.n_fft, self.hop = n_fft, n_fft // 2
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, dim, 3, padding=1, stride=2), nn.BatchNorm2d(dim), nn.GELU(),
        )
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=drop)
        self.head = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, n_classes))

    def forward(self, x):
        B, C, T = x.shape
        win = torch.hann_window(self.n_fft, device=x.device)
        spec = torch.stft(x.reshape(B * C, T), n_fft=self.n_fft, hop_length=self.hop,
                          window=win, return_complex=True).abs()
        spec = spec.reshape(B, C, spec.size(1), spec.size(2))  # (B,C,F,T')
        h = self.cnn(spec).flatten(2).transpose(1, 2)          # (B, tokens, dim)
        a, _ = self.attn(h, h, h)
        return self.head(a.mean(1))


def build_model(name="fate", ablation="none", **kw):
    name = name.lower()
    # n_steps is a harness-level knob only the temporal-context models use; pop it
    # so it never leaks into backbones that do not accept it (FATE, EEGNet, ...).
    n_steps = kw.pop("n_steps", 8)
    if name == "fate":
        from model import FATE
        return FATE(ablation=ablation, **kw)
    if name == "eegnet":
        return EEGNet(**kw)
    if name == "cnnlstm":
        return CNNLSTM(**kw)
    if name == "runet":
        return RuNet(**kw)
    if name == "graphst":
        from model_graphst import build_graphst
        # harness passes drop=<dropout> + ablation (none|nograph|noscale)
        return build_graphst(ablation, dropout=kw.get("drop", 0.25))
    if name == "fate_v3":
        from model_v3 import build_v3
        # harness passes drop=<dropout> + ablation (see model_v3.ABLATIONS_V3)
        return build_v3(ablation, drop=kw.get("drop", 0.25))
    if name == "tfcanet":
        from tfcanet import build_tfcanet
        # literature baseline (Sci Rep ~2026-03, time-frequency cross-attention) --
        # no ablation variants, ignores the ablation arg like eegnet/cnnlstm/runet.
        return build_tfcanet(drop=kw.get("drop", 0.1))
    if name in ("tempctx", "tempxf"):
        from model_temporal import build_temporal
        # Phase C temporal-context models (LSTM / Transformer over consecutive
        # intra-window sub-windows). n_steps read from kw (train.py --temporal-steps).
        return build_temporal(name, n_steps=n_steps, drop=kw.get("drop", 0.25))
    raise ValueError(f"unknown model {name}")


if __name__ == "__main__":
    x = torch.randn(2, 18, 1024)
    for name in ("eegnet", "cnnlstm", "runet"):
        m = build_model(name)
        y = m(x)
        n = sum(p.numel() for p in m.parameters())
        print(f"{name:8s} out {tuple(y.shape)}  params {n:,}")
