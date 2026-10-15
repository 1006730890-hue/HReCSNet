"""Minimal HReCSNet decoder wiring on top of the official SMT-T encoder.

This is the released, inference-oriented subset of the study generator
``SMTNewGANGenerator``: the four ``adapt`` projections, the shared progressive
decoder (``fuse3``/``fuse2``/``fuse1`` + ``refine``) and the six prediction
heads (``sal``/``edge``/``coarse``/``aux2``/``aux3``/``unc``).  The forward pass
is unchanged; only the code that is irrelevant for HReCSNet was dropped.

The SMT-T backbone is **not** vendored here.  Clone the official SMT repository
and place it at ``<repo>/external/SMT`` so that ``external/SMT/models/smt.py``
exists (see ``README.md``).  The location can also be given with ``--smt-root``
or the ``SMT_ROOT`` environment variable.

The study's ``biformer_sge`` module combination (``f4_biformer`` + ``d3_sge``)
is neutralised by :class:`models.hrecsnet.GroupGenerator` exactly as in the
research code, so HReCSNet carries no parameters for those two slots and they do
not appear in the released checkpoints.  They are created here as ``Identity``
placeholders so the module tree matches the training-time model.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Relative locations searched for the official SMT repository, in order.
SMT_SEARCH_PATHS = (
    REPO_ROOT / 'external' / 'SMT',
    REPO_ROOT.parent / 'external' / 'SMT',
    Path.cwd() / 'external' / 'SMT',
    Path.cwd().parent / 'external' / 'SMT',
)


def resolve_smt_root(smt_root=None):
    """Return the directory of the official SMT repository (containing ``models/smt.py``)."""
    candidates = []
    if smt_root:
        candidates.append(Path(smt_root).expanduser())
    env = os.environ.get('SMT_ROOT')
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(SMT_SEARCH_PATHS)
    for candidate in candidates:
        if (candidate / 'models' / 'smt.py').is_file():
            return candidate.resolve()
    tried = '\n  '.join(str(c) for c in candidates)
    raise FileNotFoundError(
        'Official SMT repository not found. Clone it so that '
        '<SMT_ROOT>/models/smt.py exists, then pass --smt-root or set SMT_ROOT.\n'
        f'Searched:\n  {tried}')


def load_smt_module(smt_root=None):
    """Import ``models/smt.py`` from the official SMT repository as a private module."""
    root = resolve_smt_root(smt_root)
    source = root / 'models' / 'smt.py'
    spec = importlib.util.spec_from_file_location('hrecsnet_official_smt', source)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load SMT backbone from {source}')
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('hrecsnet_official_smt', module)
    spec.loader.exec_module(module)
    return module


class ConvBNAct(nn.Sequential):
    def __init__(self, ic, oc, k=3):
        super().__init__(nn.Conv2d(ic, oc, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(oc), nn.GELU())


class SMTNewGANGenerator(nn.Module):
    """SMT-T pyramid + 64-channel shared progressive decoder + six heads."""

    def __init__(self, variant='biformer_sge', pretrained=False,
                 weights_path='external/SMT/smt_tiny_full.pth', image_size=384,
                 smt_root=None):
        super().__init__()
        if variant != 'biformer_sge':
            raise ValueError("HReCSNet release only ships the 'biformer_sge' SMT-T variant")
        module = load_smt_module(smt_root)
        self.encoder = module.smt_t(img_size=384, num_classes=1000)
        self.pretrained_report = {'loaded': False, 'missing_keys': [], 'unexpected_keys': []}
        if pretrained:
            path = Path(weights_path)
            path = path if path.is_absolute() else Path.cwd() / path
            weights = torch.load(path, map_location='cpu', weights_only=False)
            state = weights.get('model', weights.get('state_dict', weights))
            state = {k.replace('module.', '', 1): v for k, v in state.items()
                     if not k.startswith('head.')}
            report = self.encoder.load_state_dict(state, strict=False)
            self.pretrained_report = {'loaded': True, 'path': str(path),
                                      'missing_keys': list(report.missing_keys),
                                      'unexpected_keys': list(report.unexpected_keys)}

        self.variant = variant
        self.image_size = image_size
        self.adapt = nn.ModuleList([ConvBNAct(c, 64, 1) for c in [64, 128, 256, 512]])
        self.fuse3 = ConvBNAct(128, 64)
        self.fuse2 = ConvBNAct(128, 64)
        self.fuse1 = ConvBNAct(128, 64)
        self.refine = nn.Sequential(ConvBNAct(64, 64), ConvBNAct(64, 64))
        self.edge = nn.Conv2d(64, 1, 1)
        self.sal = nn.Conv2d(64, 1, 1)
        self.coarse = nn.Conv2d(64, 1, 1)
        self.aux2 = nn.Conv2d(64, 1, 1)
        self.aux3 = nn.Conv2d(64, 1, 1)
        self.unc = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        # The study's biformer_sge combination is disabled for every HReCSNet run.
        self.f4_biformer = nn.Identity()
        self.d3_sge = nn.Identity()

    def forward(self, x):
        b = x.shape[0]
        z = x
        fs = []
        for i in range(4):
            z, h, w = getattr(self.encoder, f'patch_embed{i+1}')(z)
            for blk in getattr(self.encoder, f'block{i+1}'):
                z = blk(z, h, w)
            z = getattr(self.encoder, f'norm{i+1}')(z).transpose(1, 2).reshape(b, -1, h, w).contiguous()
            fs.append(self.adapt[i](z))
        f1, f2, f3, f4 = fs
        d3 = self.fuse3(torch.cat([f3, F.interpolate(f4, size=f3.shape[-2:], mode='bilinear', align_corners=False)], 1))
        d2 = self.fuse2(torch.cat([f2, F.interpolate(d3, size=f2.shape[-2:], mode='bilinear', align_corners=False)], 1))
        d1 = self.fuse1(torch.cat([f1, F.interpolate(d2, size=f1.shape[-2:], mode='bilinear', align_corners=False)], 1))
        feat = self.refine(d1)
        size = x.shape[-2:]
        out = {'sal': self.sal(feat), 'edge': self.edge(feat), 'unc': self.unc(feat),
               'coarse': self.coarse(d3), 'aux2': self.aux2(d2), 'aux3': self.aux3(d3)}
        return {k: F.interpolate(v, size=size, mode='bilinear', align_corners=False)
                for k, v in out.items()}


__all__ = ['SMTNewGANGenerator', 'ConvBNAct', 'load_smt_module', 'resolve_smt_root']
