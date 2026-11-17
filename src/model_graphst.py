"""
GraphSpatioTemporalNet — co-author's drop-in replacement for FreqAwareCNNTransformer.

Multi-scale temporal front-end + graph spatial mixing + temporal encoder.
Takes (B, C=18, T) and returns (B, n_classes) logits, so build_model() can swap
it into the same LOSO harness as FATE / EEGNet / CNNLSTM / RuNet.

NOTE (integration audit): GraphSpatialMixer computes a learnable electrode
adjacency (self.adj_logits -> softmax) but the forward pass applies the graph
convolutions with torch.eye(T) (identity), so the electrode graph is currently
unused. Kept faithful to the co-author's original CELL 3 so cluster runs match
the notebook exactly; revisit if we want the adjacency to actually drive mixing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTemporalExtractor(nn.Module):
    """Multi-scale temporal feature extractor (replaces rigid band decomposition + EEGNet)."""

    def __init__(self, n_channels, embed_dim=128, dropout=0.25):
        super().__init__()
        # Parallel temporal branches with different receptive fields
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1, k), padding=(0, k // 2), bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
            )
            for k in [25, 51, 75]  # short / medium / long temporal context
        ])
        self.spatial = nn.Sequential(
            nn.Conv2d(16 * 3, 32, kernel_size=(n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), padding=(0, 7), groups=32, bias=False),
            nn.Conv2d(32, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # (B, C, T)
        x = x.unsqueeze(1)  # (B, 1, C, T)
        multi = [branch(x) for branch in self.scales]
        x = torch.cat(multi, dim=1)  # (B, 48, C, T)
        x = self.spatial(x)
        x = self.separable(x)  # (B, D, 1, tokens)
        x = x.squeeze(2)  # (B, D, tokens)
        return x.permute(0, 2, 1)  # (B, tokens, D)


class GraphConvolution(nn.Module):
    """Simple graph convolution with learnable adjacency."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, adj):  # x: (B, N, F), adj: (N, N)
        support = self.weight(x)  # (B, N, out_dim)
        out = torch.matmul(adj, support) + self.bias
        return out


class GraphSpatialMixer(nn.Module):
    """Learnable electrode graph + residual graph convolution."""

    def __init__(self, n_channels, embed_dim, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        # Learnable adjacency (will be softmax-normalized)
        self.adj_logits = nn.Parameter(torch.randn(n_channels, n_channels) * 0.01)
        self.gcn1 = GraphConvolution(embed_dim, embed_dim)
        self.gcn2 = GraphConvolution(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # (B, tokens, D) - we treat tokens as temporary "nodes"
        # Here we re-interpret the channel dimension that was collapsed earlier.
        # For simplicity we apply a channel-wise graph after expanding back.
        # In this design we apply a lightweight residual GCN on the feature sequence
        # and also maintain a global channel graph that is broadcast.
        B, T, D = x.shape
        adj = F.softmax(self.adj_logits, dim=-1)  # (C, C) - used later if needed

        # Residual graph-style mixing on the token dimension (acts as a soft spatial mixer)
        h = self.norm1(x)
        h = self.gcn1(h, torch.eye(T, device=x.device))  # identity + residual for stability
        h = self.dropout(F.gelu(h))
        x = x + h

        h = self.norm2(x)
        h = self.gcn2(h, torch.eye(T, device=x.device))
        h = self.dropout(F.gelu(h))
        x = x + h
        return x, adj


class TemporalDynamicsEncoder(nn.Module):
    """Lightweight temporal encoder (TransformerEncoder style, same as original)."""

    def __init__(self, embed_dim, num_layers=2, num_heads=4, dropout=0.25):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor=False: no padding mask is ever passed, and norm_first=True
        # already disqualifies the nested-tensor fast path — set explicitly to silence the
        # PyTorch warning without changing outputs.
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

    def forward(self, x):  # (B, T, D)
        return self.transformer(x)


class GraphSpatioTemporalNet(nn.Module):
    """
    New architecture: Multi-scale temporal front-end + Graph spatial mixing + Temporal encoder.
    Drop-in replacement for FreqAwareCNNTransformer.
    """

    def __init__(
        self,
        n_channels=18,
        embed_dim=128,
        num_layers=2,
        num_heads=4,
        num_classes=2,
        dropout=0.25,
    ):
        super().__init__()
        self.extractor = MultiScaleTemporalExtractor(n_channels, embed_dim, dropout)
        self.graph_mixer = GraphSpatialMixer(n_channels, embed_dim, dropout)
        self.temporal = TemporalDynamicsEncoder(embed_dim, num_layers, num_heads, dropout)

        self.seq_pool = nn.Linear(embed_dim, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # (B, C, T)
        tokens = self.extractor(x)          # (B, tokens, D)
        tokens, _ = self.graph_mixer(tokens)  # (B, tokens, D)
        tokens = self.temporal(tokens)      # (B, tokens, D)

        # Attention pooling (same as the original FATE head)
        weights = F.softmax(self.seq_pool(tokens), dim=1)
        pooled = (weights * tokens).sum(dim=1)  # (B, D)
        return self.head(pooled)            # (B, num_classes)


class GraphSpatioTemporalNOGRAPH(nn.Module):
    """Ablation 1: remove GraphSpatialMixer -- pure multi-scale CNN + Transformer."""

    def __init__(self, n_channels=18, embed_dim=128,
                 num_layers=2, num_heads=4, num_classes=2, dropout=0.25):
        super().__init__()
        self.extractor = MultiScaleTemporalExtractor(n_channels, embed_dim, dropout)
        self.temporal = TemporalDynamicsEncoder(embed_dim, num_layers, num_heads, dropout)
        self.seq_pool = nn.Linear(embed_dim, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes),
        )

    def forward(self, x):
        tokens = self.extractor(x)
        tokens = self.temporal(tokens)
        weights = F.softmax(self.seq_pool(tokens), dim=1)
        pooled = (weights * tokens).sum(dim=1)
        return self.head(pooled)


class SingleScaleTemporalExtractor(nn.Module):
    """Single-scale version of the temporal front-end (for ablation)."""

    def __init__(self, n_channels, embed_dim=128, dropout=0.25, kernel_size=51):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2), bias=False),
            nn.BatchNorm2d(16), nn.ELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), padding=(0, 7), groups=32, bias=False),
            nn.Conv2d(32, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )

    def forward(self, x):  # (B, C, T)
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        x = x.squeeze(2)
        return x.permute(0, 2, 1)  # (B, tokens, D)


class GraphSpatioTemporalNOSCALE(nn.Module):
    """Ablation 2: remove multi-scale front-end -- single-scale + Graph + Transformer."""

    def __init__(self, n_channels=18, embed_dim=128,
                 num_layers=2, num_heads=4, num_classes=2, dropout=0.25):
        super().__init__()
        self.extractor = SingleScaleTemporalExtractor(n_channels, embed_dim, dropout)
        self.graph_mixer = GraphSpatialMixer(n_channels, embed_dim, dropout)
        self.temporal = TemporalDynamicsEncoder(embed_dim, num_layers, num_heads, dropout)
        self.seq_pool = nn.Linear(embed_dim, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes),
        )

    def forward(self, x):
        tokens, _ = self.graph_mixer(self.extractor(x))
        tokens = self.temporal(tokens)
        weights = F.softmax(self.seq_pool(tokens), dim=1)
        pooled = (weights * tokens).sum(dim=1)
        return self.head(pooled)


def build_graphst(ablation="none", dropout=0.25):
    """Dispatch the graphst family through the LOSO harness's ablation flag.

    none    -> full GraphSpatioTemporalNet (MultiScale + Graph + Transformer)
    nograph -> GraphSpatioTemporalNOGRAPH (drop the graph spatial mixer)
    noscale -> GraphSpatioTemporalNOSCALE (single-scale temporal front-end)
    """
    a = (ablation or "none").lower()
    if a in ("none", ""):
        return GraphSpatioTemporalNet(dropout=dropout)
    if a == "nograph":
        return GraphSpatioTemporalNOGRAPH(dropout=dropout)
    if a == "noscale":
        return GraphSpatioTemporalNOSCALE(dropout=dropout)
    raise ValueError(f"unknown graphst ablation {ablation!r} (use none|nograph|noscale)")


if __name__ == "__main__":
    _x = torch.randn(2, 18, 1024)
    _model = GraphSpatioTemporalNet()
    print(f"Output shape : {_model(_x).shape}")
    print(f"Parameters   : {sum(p.numel() for p in _model.parameters() if p.requires_grad):,}")
