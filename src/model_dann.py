"""
Domain-adversarial training wrapper (Phase D2): DANN-style gradient reversal on
a patient/domain discriminator, on top of any existing backbone.

Idea (Ganin et al. 2016, "Domain-Adversarial Training of Neural Networks"): the
seizure classifier and a patient discriminator share a feature extractor. A
gradient-reversal layer (GRL) sits between the shared features and the
discriminator, so minimizing the discriminator loss during the backward pass
*maximizes* patient-confusion in the features -- pushing the backbone toward
patient-invariant representations that transfer to the unseen LOSO test patient.

Leak-free w.r.t. the test patient: the discriminator's domain labels are the
TRAINING patient ids only (the held-out test patient never appears in training,
so it is never a discriminator class and never contributes a domain gradient).

Contract preserved: DANNWrapper.forward(x) returns (B, 2) class logits exactly
like the bare backbone, so it is a drop-in for evaluation / _scores / metrics.
Training calls forward_da(x, alpha) to also get the (B, n_domains) domain logits.

The wrapper taps a backbone feature vector via a registered forward hook on a
named penultimate module (default: the backbone's `.head`), so it works for
FATE, EEGNet, CNNLSTM, RuNet, GraphST, tempctx/tempxf, etc. without editing them.

Run `python model_dann.py` to sanity-check shapes over several backbones.
"""
import torch
import torch.nn as nn

from baselines import build_model


class GradientReversal(torch.autograd.Function):
    """Identity forward; negate-and-scale gradient on the backward pass."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


def _find_feature_module(backbone):
    """Return (module, in_features) for the classifier the GRL taps in front of.

    The module's INPUT is the shared pooled feature vector we want the domain
    discriminator to see. Backbones in this repo expose the classifier as `head`
    (FATE/RuNet/CNNLSTM/temporal), `classify` (EEGNet), or `classifier`; fall back
    to the last nn.Linear anywhere in the backbone.
    """
    for attr in ("head", "classify", "classifier"):
        mod = getattr(backbone, attr, None)
        if mod is not None:
            first = mod[0] if isinstance(mod, nn.Sequential) else mod
            if isinstance(first, nn.Linear):
                return first, first.in_features
    last = None
    for m in backbone.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is not None:
        return last, last.in_features
    return None, None


class DANNWrapper(nn.Module):
    """Wrap a backbone with a GRL patient discriminator.

    Parameters
    ----------
    backbone : nn.Module returning (B, 2) class logits.
    n_domains : number of training patients (discriminator output size).
    feat_dim : dimensionality of the tapped feature vector.
    """

    def __init__(self, backbone, n_domains, feat_dim=64, disc_hidden=64, drop=0.25):
        super().__init__()
        self.backbone = backbone
        self.n_domains = n_domains
        self._feat = None
        target, in_feat = _find_feature_module(backbone)
        if in_feat is not None:
            feat_dim = in_feat
        # capture the feature vector that flows INTO the classifier layer
        (target or backbone).register_forward_pre_hook(self._capture)
        self.discriminator = nn.Sequential(
            nn.Linear(feat_dim, disc_hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(disc_hidden, n_domains))

    def _capture(self, module, inputs):
        self._feat = inputs[0]

    def forward(self, x):
        return self.backbone(x)                     # (B, 2), contract preserved

    def forward_da(self, x, alpha=1.0):
        logits = self.backbone(x)                   # populates self._feat via hook
        feat = self._feat
        dom_logits = self.discriminator(grad_reverse(feat, alpha))
        return logits, dom_logits


def build_dann(model="fate", ablation="none", n_domains=1, drop=0.25, **kw):
    """Build a backbone via build_model() and wrap it for domain-adversarial training."""
    backbone = build_model(model, ablation, drop=drop, **kw)
    return DANNWrapper(backbone, n_domains=n_domains, drop=drop)


if __name__ == "__main__":
    x = torch.randn(4, 18, 1024)
    for name in ("fate", "eegnet", "cnnlstm"):
        m = build_dann(name, n_domains=5)
        logits = m(x)
        logits2, dom = m.forward_da(x, alpha=0.7)
        n = sum(p.numel() for p in m.parameters())
        assert tuple(logits.shape) == (4, 2), name
        assert tuple(dom.shape) == (4, 5), name
        # verify GRL is differentiable end to end
        (logits2.sum() + dom.sum()).backward()
        print(f"{name:8s} class {tuple(logits.shape)} domain {tuple(dom.shape)} "
              f"params {n:,}")
