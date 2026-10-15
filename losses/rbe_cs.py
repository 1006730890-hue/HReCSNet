"""Region-Boundary-Error Collaborative Supervision (RBE-CS).

**Training only.**  HReCSNet performs inference with the saliency head alone;
none of the functions in this module are used by ``tools/infer.py`` or
``tools/evaluate.py``.  They are kept because RBE-CS is part of the method and
because the released checkpoints store the six-head parameter set that this
objective trains.

Exact copy of the authoritative research implementation
(``tools/g3_ablation_36/group_supervision.py``) with its two training-time
helper imports inlined (``sal_loss`` from ``train_m2_fixed.py`` and
``sobel_edge`` from ``supervised_reference.py``) so that the release stays
self-contained.  The loss mathematics is unchanged.

Notation (paper Sec. III-E  <->  identifiers)

    S, E, C, A2, A3, U   ->  the six heads 'sal','edge','coarse','aux2','aux3','unc'
    region loss          ->  .25*BCE(coarse) + .10*(BCE(aux2) + BCE(aux3))
    boundary loss        ->  .20*L1(edge)   + .12*L1(edge)
    error loss           ->  .04*L1(unc, |sigmoid(sal) - Y|)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

ALL_HEADS = ('sal', 'edge', 'coarse', 'aux2', 'aux3', 'unc')

GROUPS = {
    'A': {'id': 'm2a_eorssd', 'label': 'M2+A', 'name': '区域深监督',
          'heads': ['sal', 'coarse', 'aux2', 'aux3'],
          'weights': {'coarse_bce': .25, 'aux2_bce': .10, 'aux3_bce': .10}},
    'B': {'id': 'm2b_eorssd', 'label': 'M2+B', 'name': '边缘监督',
          'heads': ['sal', 'edge'], 'weights': {'edge_l1': .20, 'extra_edge_l1': .12}},
    'C': {'id': 'm2c_eorssd', 'label': 'M2+C', 'name': '误差回归',
          'heads': ['sal', 'unc'], 'weights': {'unc_l1': .04}},
    'AB': {'id': 'm2ab', 'label': 'M2+A+B', 'name': '区域深监督 + 边缘监督',
           'heads': ['sal', 'edge', 'coarse', 'aux2', 'aux3'],
           'weights': {'coarse_bce': .25, 'aux2_bce': .10, 'aux3_bce': .10,
                       'edge_l1': .20, 'extra_edge_l1': .12}},
    'AC': {'id': 'm2ac_eorssd', 'label': 'M2+A+C', 'name': '区域深监督 + 误差回归',
           'heads': ['sal', 'coarse', 'aux2', 'aux3', 'unc'],
           'weights': {'coarse_bce': .25, 'aux2_bce': .10, 'aux3_bce': .10, 'unc_l1': .04}},
    'BC': {'id': 'm2bc_eorssd', 'label': 'M2+B+C', 'name': '边缘监督 + 误差回归',
           'heads': ['sal', 'edge', 'unc'],
           'weights': {'edge_l1': .20, 'extra_edge_l1': .12, 'unc_l1': .04}},
    'ABC': {'id': 'r0_eorssd', 'label': 'M2+A+B+C / R0', 'name': '完整多头监督',
            'heads': ['sal', 'edge', 'coarse', 'aux2', 'aux3', 'unc'],
            'weights': {'coarse_bce': .25, 'aux2_bce': .10, 'aux3_bce': .10,
                        'edge_l1': .20, 'extra_edge_l1': .12, 'unc_l1': .04}},
}


def sobel_edge(x: torch.Tensor) -> torch.Tensor:
    """Differentiable normalized Sobel magnitude for structural conditioning."""
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError("x must be a Bx1xHxW tensor")
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    return torch.sqrt(F.conv2d(x, kx, padding=1).square() + F.conv2d(x, ky, padding=1).square() + 1e-6).clamp(0, 1)


def sal_loss(outputs, mask):
    """Primary region objective on the saliency head: boundary-band BCE + Dice."""
    logits = outputs['sal']
    p = torch.sigmoid(logits)
    band = 1 + 4 * torch.abs(F.avg_pool2d(mask, 15, 1, 7) - mask)
    b = (band * F.binary_cross_entropy_with_logits(logits, mask, reduction='none')).mean()
    inter = (p * mask).sum((2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((2, 3)) + mask.sum((2, 3)) + 1)
    return b + dice.mean(), float(b.detach()), float(dice.mean().detach())


def grouped_loss(outputs, mask, group):
    """RBE-CS objective for one supervision group.  Training only."""
    expected = set(GROUPS[group]['heads'])
    if set(outputs) != expected:
        raise ValueError(f'{group} expected heads {expected}, got {set(outputs)}')
    primary, bce, dice = sal_loss(outputs, mask)
    zero = primary.new_zeros(())
    region, boundary, uncertainty = zero, zero, zero
    raw = {}
    if group == 'AB':
        for key in ('coarse', 'aux2', 'aux3'):
            target = F.interpolate(mask, size=outputs[key].shape[-2:], mode='area')
            raw[key] = F.binary_cross_entropy_with_logits(outputs[key], target)
        raw['edge'] = F.l1_loss(torch.sigmoid(outputs['edge']), sobel_edge(mask))
        region = .25 * raw['coarse'] + .10 * (raw['aux2'] + raw['aux3'])
        boundary = .20 * raw['edge'] + .12 * raw['edge']
        # Preserve original R0 summation order, omitting only the unc term.
        loss = primary + .20 * raw['edge'] + .25 * raw['coarse'] + .10 * (raw['aux2'] + raw['aux3']) + .12 * raw['edge']
    elif group == 'A':
        for key in ('coarse', 'aux2', 'aux3'):
            target = F.interpolate(mask, size=outputs[key].shape[-2:], mode='area')
            raw[key] = F.binary_cross_entropy_with_logits(outputs[key], target)
        region = .25 * raw['coarse'] + .10 * (raw['aux2'] + raw['aux3'])
        loss = primary + region
    elif group == 'B':
        raw['edge'] = F.l1_loss(torch.sigmoid(outputs['edge']), sobel_edge(mask))
        boundary = .20 * raw['edge'] + .12 * raw['edge']
        loss = primary + .20 * raw['edge'] + .12 * raw['edge']
    elif group == 'C':
        target = (torch.sigmoid(outputs['sal']) - mask).abs().detach()
        raw['unc'] = F.l1_loss(outputs['unc'], target)
        uncertainty = .04 * raw['unc']
        loss = primary + uncertainty
    elif group in ('AC', 'BC', 'ABC'):
        # Keep the R0 summation order and detach its prediction-error target.
        loss = primary
        if 'B' in group:
            raw['edge'] = F.l1_loss(torch.sigmoid(outputs['edge']), sobel_edge(mask))
            boundary = .20 * raw['edge'] + .12 * raw['edge']
            loss = loss + .20 * raw['edge']
        if 'A' in group:
            for key in ('coarse', 'aux2', 'aux3'):
                target = F.interpolate(mask, size=outputs[key].shape[-2:], mode='area')
                raw[key] = F.binary_cross_entropy_with_logits(outputs[key], target)
            region = .25 * raw['coarse'] + .10 * (raw['aux2'] + raw['aux3'])
            loss = loss + .25 * raw['coarse'] + .10 * (raw['aux2'] + raw['aux3'])
        target = (torch.sigmoid(outputs['sal']) - mask).abs().detach()
        raw['unc'] = F.l1_loss(outputs['unc'], target)
        uncertainty = .04 * raw['unc']
        loss = loss + uncertainty
        if 'B' in group:
            loss = loss + .12 * raw['edge']
    else:
        raise ValueError(f'Unsupported group: {group}')
    logs = dict(sal_bce=bce, sal_dice=dice, supervised=float(primary.detach()),
                region_loss=float(region.detach()), edge_loss=float(boundary.detach()),
                unc_loss=float(uncertainty.detach()))
    return loss, logs


__all__ = ['GROUPS', 'ALL_HEADS', 'grouped_loss', 'sal_loss', 'sobel_edge']
