"""HReCSNet: SMT-T backbone + shared progressive decoder + CHFR.

Paper composition:

    HReCSNet = SMT-T (lightweight backbone)
             + shared progressive decoder (adapt / fuse1-3 / refine)
             + CHFR (Cross-Scale High-Frequency Recalibration) at D3 and D2
             + RBE-CS (Region-Boundary-Error Collaborative Supervision, training only)

Inference uses the saliency head only.  ``StructureGenerator.forward(..., only_sal=True)``
returns ``{'sal': ...}`` and, for the CHFR branch, never evaluates the auxiliary
heads at all -- it is the first return of the branch, not a slice of a six-head
output.  :class:`SalOnly` is the same generator restricted to that path.

``SDPG`` is the study's earlier "shared progressive decoder guidance" adapter.
It is retained **only** so that the module tree is bit-identical to the training
code and the released checkpoints load with ``strict=True``; the CHFR branch
returns before any SDPG code is reached, so it is never executed at inference
and contributes no computation.  Its implementation is an exact copy of the
authoritative ``tools/smt_structure_20261006/model.py``.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from losses.rbe_cs import ALL_HEADS, GROUPS
from models.chfr import CHFR, ResidualAdapter, cgn, highpass, up
from models.decoder import SMTNewGANGenerator, resolve_smt_root

#: The only released HReCSNet variant (CHFR adapters at D3/D2).
VARIANT = 'chfr'
SUPPORTED_VARIANTS = (VARIANT,)
#: Position of 'chfr' in the study's ATTENTION_VARIANTS tuple; fixes the
#: initialization seeds used by the research code (10020 + 2*index, 10021 + 2*index).
_ATTENTION_VARIANT_INDEX = 2
#: Groups whose auxiliary heads are training-time only.
AUX_HEADS = ('edge', 'coarse', 'aux2', 'aux3', 'unc')


class SDPG(ResidualAdapter):
    """Study legacy adapter (exact copy).  Never executed by the CHFR inference path."""

    def __init__(self, channels=64):
        super().__init__()
        self.detail = nn.Sequential(cgn(2 * channels, 32), cgn(32, 32, 3, groups=32),
                                    nn.Conv2d(32, channels, 1, bias=False))
        self.gate = nn.Sequential(cgn(2 * channels + 1, 32), nn.Conv2d(32, 1, 1))

    def forward(self, low, high, image_gradient):
        guide = up(high, low)
        edge = F.interpolate(image_gradient, size=low.shape[-2:], mode='area')
        gate = self.gate(torch.cat([low, guide, edge], 1)).sigmoid()
        return self.residual(low, gate * self.detail(torch.cat([low, highpass(low)], 1)))


class GroupGenerator(nn.Module):
    """Shared progressive decoder + the six RBE-CS heads.

    ``group`` selects which heads the objective in :mod:`losses.rbe_cs` uses; the
    released model always uses the full ``'ABC'`` head set.  Heads outside the
    selected group are set to ``None`` so they are absent from ``state_dict()``.
    """

    def __init__(self, group='ABC', dataset='eorssd', pretrained=False,
                 smt_root=None, pretrained_weights=None, image_size=384):
        super().__init__()
        if group not in GROUPS:
            raise ValueError(f'Unknown RBE-CS group: {group}')
        self.group = group
        self.dataset = dataset
        self.heads = list(GROUPS[group]['heads'])
        self.base = SMTNewGANGenerator('biformer_sge', pretrained=pretrained,
                                       weights_path=pretrained_weights or 'external/SMT/smt_tiny_full.pth',
                                       image_size=image_size, smt_root=smt_root)
        # Identical to the training code: the biformer/sge combination is disabled.
        self.base.f4_biformer = nn.Identity()
        self.base.d3_sge = nn.Identity()
        for key in ALL_HEADS:
            if key not in self.heads:
                setattr(self.base, key, None)
        self.pretrained_report = self.base.pretrained_report

    def forward(self, x):
        b = x.shape[0]
        z = x
        fs = []
        base = self.base
        for i in range(4):
            z, h, w = getattr(base.encoder, f'patch_embed{i+1}')(z)
            for blk in getattr(base.encoder, f'block{i+1}'):
                z = blk(z, h, w)
            z = getattr(base.encoder, f'norm{i+1}')(z).transpose(1, 2).reshape(b, -1, h, w).contiguous()
            fs.append(base.adapt[i](z))
        f1, f2, f3, f4 = fs
        d3 = base.fuse3(torch.cat([f3, up(f4, f3)], 1))
        d2 = base.fuse2(torch.cat([f2, up(d3, f2)], 1))
        d1 = base.fuse1(torch.cat([f1, up(d2, f1)], 1))
        feat = base.refine(d1)
        sources = {'sal': feat, 'edge': feat, 'unc': feat, 'coarse': d3, 'aux2': d2, 'aux3': d3}
        return {key: up(getattr(base, key)(sources[key]), x) for key in self.heads}


class StructureGenerator(GroupGenerator):
    """HReCSNet generator: CHFR_3 at D3 and CHFR_2 at D2.

        d3 <- CHFR_3(Fuse_3(f3, f4), f4)
        d2 <- CHFR_2(Fuse_2(f2, d3), d3)
        z  <- Refine(Fuse_1(f1, d2))
    """

    def __init__(self, variant=VARIANT, pretrained=False, dataset='eorssd',
                 smt_root=None, pretrained_weights=None, image_size=384):
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f'This release only ships variant {VARIANT!r}, got {variant!r}')
        super().__init__('ABC', dataset, pretrained=pretrained, smt_root=smt_root,
                         pretrained_weights=pretrained_weights, image_size=image_size)
        self.variant = variant
        # Isolated per-module initialization, exactly as in the research code.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(10001)
            self.sdpg1 = SDPG()
            torch.manual_seed(10002)
            self.sdpg2 = SDPG()
            torch.manual_seed(10020 + _ATTENTION_VARIANT_INDEX * 2)
            self.attn3 = CHFR()
            torch.manual_seed(10021 + _ATTENTION_VARIANT_INDEX * 2)
            self.attn2 = CHFR()

    @staticmethod
    def input_gradient(x):
        rgb = x * x.new_tensor([.229, .224, .225]).view(1, 3, 1, 1) + x.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
        gray = (rgb * x.new_tensor([.299, .587, .114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
        padded = F.pad(gray, (1, 1, 1, 1), mode='replicate')
        dx = .5 * (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2])
        dy = .5 * (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1])
        return (dx.abs() + dy.abs()).clamp(0, 1)

    def forward(self, x, only_sal=False):
        b = x.shape[0]
        z = x
        fs = []
        base = self.base
        for i in range(4):
            z, h, w = getattr(base.encoder, f'patch_embed{i+1}')(z)
            for block in getattr(base.encoder, f'block{i+1}'):
                z = block(z, h, w)
            z = getattr(base.encoder, f'norm{i+1}')(z).transpose(1, 2).reshape(b, -1, h, w).contiguous()
            fs.append(base.adapt[i](z))
        f1, f2, f3, f4 = fs
        gradient = self.input_gradient(x)
        # CHFR decoder branch (paper Sec. III-D).  q_i = Fuse_i output (CHFR input),
        # d_i = CHFR output (that level's feature).
        q3 = base.fuse3(torch.cat([f3, up(f4, f3)], 1))
        d3 = self.attn3(q3, f4, gradient)
        q2 = base.fuse2(torch.cat([f2, up(d3, f2)], 1))
        d2 = self.attn2(q2, d3, gradient)
        q1 = base.fuse1(torch.cat([f1, up(d2, f1)], 1))
        feat = base.refine(q1)
        if only_sal:
            # Pure saliency inference: the auxiliary heads are not evaluated.
            return {'sal': up(base.sal(feat), x)}
        sources = {'sal': feat, 'edge': feat, 'unc': feat, 'coarse': d3, 'aux2': d2, 'aux3': d3}
        return {key: up(getattr(base, key)(sources[key]), x) for key in ALL_HEADS}


class SalOnly(nn.Module):
    """View of a generator restricted to the saliency path (training-time heads are not run)."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x):
        return self.generator(x, only_sal=True)


class _CountingHead(nn.Module):
    """Transparent wrapper that counts how often a head module is evaluated."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.module(x)


def check_single_head(generator, size=384):
    """Instrumental proof that pure saliency inference never touches the auxiliary heads.

    Every one of the six heads is wrapped in a call counter, a single
    ``forward(x, only_sal=True)`` is executed, and the counters are returned
    (``{'sal': 1, 'edge': 0, 'coarse': 0, 'aux2': 0, 'aux3': 0, 'unc': 0}``).
    The wrappers are then removed, leaving the generator unchanged.
    """
    originals, counters = {}, {}
    for key in ALL_HEADS:
        head = getattr(generator.base, key, None)
        if head is None:
            continue
        originals[key] = head
        counters[key] = _CountingHead(head)
        setattr(generator.base, key, counters[key])
    was_training = generator.training
    generator.eval()
    try:
        with torch.no_grad():
            out = generator(torch.zeros(1, 3, size, size, dtype=torch.float32), only_sal=True)
        if set(out) != {'sal'}:
            raise AssertionError(f'only_sal=True returned heads {sorted(out)}, expected only sal')
    finally:
        for key, module in originals.items():
            setattr(generator.base, key, module)
        if was_training:
            generator.train()
    return {key: wrapper.calls for key, wrapper in counters.items()}


def _dataset_from_checkpoint(ck):
    name = str(ck.get('config', {}).get('data', {}).get('name', '')).lower()
    return {'eorssd': 'eorssd', 'orsi4199': 'orsi4199', 'orssd': 'orssd'}.get(name, name or 'orssd')


def load_hrecsnet(checkpoint, device='cpu', smt_root=None, strict=True, verify_single_head=True):
    """Load a released HReCSNet checkpoint and return an eval-mode :class:`SalOnly` generator.

    The checkpoints store the trained parameters under the ``'generator'`` key and
    are tagged with ``format='smt_structure_20261006_v1'``.  Their key set is
    identical to :class:`StructureGenerator` (``base.*``, ``attn3.*``, ``attn2.*``
    and the training-time-only ``sdpg1.*``/``sdpg2.*``), so no prefix rewriting or
    key filtering is required.

    When ``verify_single_head`` is set, the five training-only heads are counted
    during one ``only_sal=True`` forward before being detached, and the counts are
    returned in ``report['head_calls']`` (must be ``{'sal': 1, others: 0}``).
    """
    ck = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
    if not isinstance(ck, dict):
        raise ValueError(f'Unrecognised checkpoint object in {checkpoint}')
    if ck.get('format') != 'smt_structure_20261006_v1':
        raise ValueError(f'Not an HReCSNet structure checkpoint: {checkpoint}')
    variant = ck.get('config', {}).get('ablation', {}).get('variant', VARIANT)
    state = ck['generator'] if 'generator' in ck else ck
    g = StructureGenerator(variant, pretrained=False, dataset=_dataset_from_checkpoint(ck),
                           smt_root=smt_root)
    g.load_state_dict(state, strict=strict)
    report = {'variant': variant, 'strict': strict, 'unexpected_keys': [], 'missing_keys': []}
    if verify_single_head:
        calls = check_single_head(g)
        report['head_calls'] = calls
        expected = {key: (1 if key == 'sal' else 0) for key in ALL_HEADS}
        if calls != expected:
            raise AssertionError(f'Auxiliary head evaluated during inference: {calls} (expected {expected})')
    # Training-time-only heads: drop them so only the saliency path remains.
    for head in AUX_HEADS:
        setattr(g.base, head, None)
    report['detached_heads'] = [h for h in AUX_HEADS if getattr(g.base, h, None) is None]
    return SalOnly(g).to(device).eval(), ck, report


__all__ = ['VARIANT', 'SUPPORTED_VARIANTS', 'AUX_HEADS', 'SDPG', 'GroupGenerator',
           'StructureGenerator', 'SalOnly', 'check_single_head', 'load_hrecsnet',
           'resolve_smt_root']
