"""Cross-Scale High-Frequency Recalibration (CHFR) and its shared building blocks.

Self-contained copy of the released HReCSNet module set.  ``up``, ``highpass``,
``cgn``, ``ResidualAdapter`` and ``CHFR`` are byte-for-byte copies of the
authoritative research implementation
(``tools/smt_structure_20261006/model.py``); the mathematics is unchanged.

Notation (paper Sec. III-D  <->  identifiers in this file):

    f_i              re-arranged 2-D feature map produced by the encoder stage
    x_i = phi_i      adapt[i] output
    q_i              fuse{i} input concatenation
    d_i              level-i decoder feature; the module writes back in place,
                     i.e. d3 <- CHFR_3(d3, f4)
    z                base.refine output (the saliency-head feature)

Inside ``CHFR``:

    L, H      ->  low, high
    X         ->  x  = low + up(high)
    M_c       ->  ch = sigmoid(mlp(avgpool(x)) + mlp(maxpool(x)))
    Z         ->  z  = x * ch
    H_f       ->  hp = highpass(z)
    M_s       ->  sp = sigmoid(spatial([mean_c(z), mean_c(|hp|)]))
    Delta     ->  the second argument of ``residual``
    alpha, a  ->  0.2 * tanh(strength); ``a`` is the learnable scalar ``strength``
                  (initialised to atanh(0.5), so alpha starts at 0.1)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def up(x, reference):
    """Bilinear resize of ``x`` to the spatial size of ``reference``."""
    return F.interpolate(x, size=reference.shape[-2:], mode='bilinear', align_corners=False)


def highpass(x):
    """3x3 local high-frequency residual.  Replicated padding avoids introducing
    an artificial black image border."""
    return x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), 3, 1)


def cgn(ic, oc, kernel=1, groups=1, dilation=1):
    """Conv + GroupNorm + GELU used throughout the decoder."""
    return nn.Sequential(nn.Conv2d(ic, oc, kernel, padding=dilation * (kernel // 2),
                                   dilation=dilation, groups=groups, bias=False),
                         nn.GroupNorm(8 if oc >= 64 else 4, oc), nn.GELU())


class ResidualAdapter(nn.Module):
    """Bounded residual write-back shared by every decoder adapter."""

    def __init__(self):
        super().__init__()
        self.strength = nn.Parameter(torch.tensor(math.atanh(.5)))

    def residual(self, x, delta):
        # Starts at 0.1, bounded in (-0.2, 0.2); all branches get gradients at step 1.
        return x + .2 * self.strength.tanh() * delta


class CHFR(ResidualAdapter):
    """Cross-Scale High-Frequency Recalibration.

    Aligns the low-level fused feature with the previous higher-level feature,
    recalibrates channels with a shared pooled MLP, forms a local high-frequency
    residual with a replicated-pad average pool, gates it spatially with the
    channel means of the recalibrated feature and of the high-frequency residual,
    and writes the result back through a bounded residual (|alpha| < 0.2).
    """

    def __init__(self, channels=64, reduction=8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.GELU(),
                                 nn.Conv2d(hidden, channels, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.out = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, low, high, gradient=None):
        # Paper notation: x->x, ch->M_c, z->Z, hp->H_f, sp->M_s.
        # The low + alpha*Delta write-back is performed by self.residual
        # (alpha = 0.2 * tanh(strength)).
        high = up(high, low)
        x = low + high
        ch = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1)))
        z = x * ch
        hp = highpass(z)
        sp = torch.sigmoid(self.spatial(torch.cat([z.mean(1, keepdim=True), hp.abs().mean(1, keepdim=True)], 1)))
        return self.residual(low, self.out(z * sp - low))
