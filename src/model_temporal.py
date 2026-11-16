"""
Temporal-context models (Phase C): classify a short sequence of consecutive
sub-windows instead of a single flat window.

Motivation: the hard-tail patients (e.g. chb05-style amplitude shifts, brief
focal onsets) fail when the model sees only one pooled embedding per window;
seizure evolution is a temporal process. These models keep the harness I/O
contract -- input (B, 18, T), output (B, 2) logits -- but internally split the
T-sample window into `n_steps` consecutive chunks, embed each chunk with a
shared EEGNet-style extractor, then model the ordered embedding sequence with
either an LSTM or a small Transformer encoder before classifying.

Because the split is internal, these drop straight into build_model() /
train.py via --model tempctx (LSTM, default) or --model tempxf (Transformer),
with no change to the data loader or the (B,18,T) window contract.

Run `python model_temporal.py` to sanity-check shapes and parameter counts.
"""
import torch
import torch.nn as nn

from model import EEGNetExtractor


class _StepEmbedder(nn.Module):
    """Shared per-chunk embedder: (B*S, C, Tc) -> (B*S, D) via EEGNet + attn-pool."""

    def __init__(self, n_channels=18, dim=128, drop=0.25):
        super().__init__()
        self.extractor = EEGNetExtractor(n_channels, D=dim, drop=drop)
        self.pool = nn.Linear(dim, 1)

    def forward(self, x):                       # (N, C, Tc)
        feats = self.extractor(x)               # (N, T', D)
        w = torch.softmax(self.pool(feats), dim=1)
        return (w * feats).sum(dim=1)           # (N, D)


def _to_steps(x, n_steps):
    """(B, C, T) -> (B, n_steps, C, T//n_steps), dropping any remainder tail."""
    B, C, T = x.shape
    step = T // n_steps
    x = x[:, :, : step * n_steps]
    return x.reshape(B, C, n_steps, step).permute(0, 2, 1, 3).contiguous()


class TemporalContextLSTM(nn.Module):
    """Per-chunk EEGNet embeddings -> BiLSTM over the ordered sequence -> (B,2).

    Keeps the (B,18,T) contract: the window is split into `n_steps` consecutive
    sub-windows internally, so the model sees intra-window temporal evolution.
    """

    def __init__(self, n_channels=18, dim=128, n_steps=8, hidden=128,
                 layers=1, drop=0.25, n_classes=2):
        super().__init__()
        self.n_steps = n_steps
        self.embed = _StepEmbedder(n_channels, dim, drop)
        self.lstm = nn.LSTM(dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True,
                            dropout=drop if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, 64), nn.GELU(), nn.Dropout(drop),
            nn.Linear(64, n_classes))

    def forward(self, x):
        B = x.size(0)
        steps = _to_steps(x, self.n_steps)               # (B, S, C, Tc)
        S, C, Tc = steps.shape[1], steps.shape[2], steps.shape[3]
        e = self.embed(steps.reshape(B * S, C, Tc))      # (B*S, D)
        e = e.reshape(B, S, -1)                          # (B, S, D)
        out, _ = self.lstm(e)                            # (B, S, 2*hidden)
        return self.head(out.mean(dim=1))


class TemporalContextTransformer(nn.Module):
    """Per-chunk EEGNet embeddings -> Transformer encoder over the sequence -> (B,2).

    Same internal window split as TemporalContextLSTM; a learned CLS token and
    positional embedding let the encoder attend across the ordered sub-windows.
    """

    def __init__(self, n_channels=18, dim=128, n_steps=8, heads=4, layers=2,
                 ffn=256, drop=0.25, n_classes=2):
        super().__init__()
        self.n_steps = n_steps
        self.embed = _StepEmbedder(n_channels, dim, drop)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_steps + 1, dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, ffn, dropout=drop, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers,
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(), nn.Dropout(drop),
            nn.Linear(64, n_classes))

    def forward(self, x):
        B = x.size(0)
        steps = _to_steps(x, self.n_steps)               # (B, S, C, Tc)
        S, C, Tc = steps.shape[1], steps.shape[2], steps.shape[3]
        e = self.embed(steps.reshape(B * S, C, Tc)).reshape(B, S, -1)
        cls = self.cls.expand(B, -1, -1)                 # (B, 1, D)
        z = torch.cat([cls, e], dim=1) + self.pos[:, : S + 1]
        z = self.encoder(z)                              # (B, S+1, D)
        return self.head(z[:, 0])                        # CLS token


def build_temporal(name="tempctx", n_steps=8, drop=0.25, **kw):
    name = name.lower()
    if name == "tempctx":
        return TemporalContextLSTM(n_steps=n_steps, drop=drop, **kw)
    if name == "tempxf":
        return TemporalContextTransformer(n_steps=n_steps, drop=drop, **kw)
    raise ValueError(f"unknown temporal model {name}")


if __name__ == "__main__":
    x = torch.randn(2, 18, 1024)
    for name in ("tempctx", "tempxf"):
        m = build_temporal(name, n_steps=8)
        y = m(x)
        n = sum(p.numel() for p in m.parameters())
        assert tuple(y.shape) == (2, 2), name
        print(f"{name:8s} out {tuple(y.shape)}  params {n:,}")
