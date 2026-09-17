import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from pathlib import Path
import src

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import FreqAwareCNNTransformer
from src.training.focal_loss import FocalLoss

print("=" * 50)
print("  Synthetic End-to-End Test")
print("=" * 50)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")

# ── Synthetic data ────────────────────────────────────
rng = np.random.default_rng(42)
T   = 512   # 2s at 256Hz

def pink_noise(shape):
    white = rng.standard_normal(shape)
    f     = np.fft.rfftfreq(shape[-1]) + 1e-6
    psd   = (1.0 / f) ** 0.5
    return np.fft.irfft(
        np.fft.rfft(white) * psd, n=shape[-1]
    ).astype(np.float32)

n_ictal, n_inter = 100, 400
X_inter = pink_noise((n_inter, 18, T))
X_ictal = pink_noise((n_ictal, 18, T))
t       = np.linspace(0, 2, T, dtype=np.float32)
X_ictal += np.sin(2 * np.pi * 3 * t)[np.newaxis, np.newaxis, :]

X = np.concatenate([X_inter, X_ictal]).astype(np.float32)
y = np.array([0]*n_inter + [1]*n_ictal, dtype=np.int64)
idx = rng.permutation(len(y))
X, y = X[idx], y[idx]

dataset = TensorDataset(
    torch.tensor(X), torch.tensor(y))
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

# ── Model ─────────────────────────────────────────────
model = FreqAwareCNNTransformer(
    n_channels=18, n_bands=4, embed_dim=128,
    num_layers=2, num_heads=4, num_classes=2, dropout=0.3,
).to(DEVICE)

params = sum(p.numel() for p in model.parameters()
             if p.requires_grad)
print(f"  Parameters: {params:,}")
assert params < 1_000_000, f"Too large: {params:,}"
print("   Model output shape : "
      f"{model(torch.randn(4,18,512).to(DEVICE)).shape}")

# ── 3-epoch training loop ─────────────────────────────
criterion = FocalLoss(alpha=0.75, gamma=2.0)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(1, 4):
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        logits   = model(X_b)
        loss     = criterion(logits, y_b)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        probs = F.softmax(logits, dim=1)[:,1]
        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(y_b.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    print(f"  Epoch {epoch}: loss={total_loss/len(loader):.4f}  "
          f"AUC={auc:.4f}")

print("\n   Training loop      : 3 epochs completed")
print(f"   Final AUC          : {auc:.4f}")
print(f"   Parameters         : {params:,}")
