"""Separate official joint-motion and uncertain surface-pseudo objectives."""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

AUXILIARY_MOTION_METRICS = tuple(
    'kinoptic_'+family+'_'+field
    for family in ('joint_motion', 'pseudo_motion')
    for field in ('loss', 'points', 'epe_m', 'uncertainty_m', 'outside_uncertainty_fraction'))


@dataclass(frozen=True)
class AuxiliaryMotionOptions:
    joint_weight: float = 0.
    pseudo_weight: float = 0.
    huber_delta: float = .01
    pseudo_uncertainty_scale: float = 1.

    def __post_init__(self):
        for name in ('joint_weight', 'pseudo_weight', 'huber_delta', 'pseudo_uncertainty_scale'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.huber_delta <= 0 or self.pseudo_uncertainty_scale < 1.:
            raise ValueError('Huber delta must be positive; pseudo uncertainty cannot be reduced below its configured band')


def auxiliary_motion_losses(pred, target, options):
    """No absolute XYZ, visibility, confidence fitting, or GT/pseudo merging.

    Scene-normalized displacement is sampled from the existing dense outputs.
    Pseudo residuals inside their measured uncertainty band have zero gradient.
    A low loss on pseudo labels is not an independent accuracy measurement.
    """
    from .tasks import sample_tracks
    zero = pred['displacement'].sum()*0.
    total = zero; terms = {}; supervised = False
    for prefix, weight, uncertain in [('joint_motion', options.joint_weight, False),
                                      ('pseudo_motion', options.pseudo_weight, True)]:
        metric = 'kinoptic_'+prefix
        terms.update({metric+'_loss': zero, metric+'_points': zero.detach(),
                      metric+'_epe_m': zero.detach(), metric+'_uncertainty_m': zero.detach(),
                      metric+'_outside_uncertainty_fraction': zero.detach()})
        if prefix+'_queries' not in target:
            continue
        queries = target[prefix+'_queries'].to(zero.device)
        truth = target[prefix+'_displacement'].to(zero.device).float()
        valid = target[prefix+'_valid'].to(zero.device).bool()
        confidence = target[prefix+'_confidence'].to(zero.device).float()
        query_valid = target[prefix+'_query_valid'].to(zero.device).bool()
        if truth.shape != (*valid.shape, 3) or confidence.shape != valid.shape or query_valid.shape != valid.shape[:2]:
            raise ValueError('auxiliary motion target shapes disagree')
        if not torch.isfinite(truth).all() or not torch.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
            raise ValueError('auxiliary targets must be finite; confidence must lie in [0,1]')
        if not queries.shape[1] or not (valid & query_valid[..., None] & (confidence > 0)).any():
            continue
        sampled = sample_tracks(pred, queries, strict=False)
        cross_time = sampled['source_time_idx'][..., None] != pred['query_time_idx'][:, None]
        valid = valid & query_valid[..., None] & sampled['query_valid'][..., None] & cross_time & (confidence > 0)
        if sampled['displacement'].shape != truth.shape:
            raise ValueError('auxiliary motion time/point axes disagree with predictions')
        if not valid.any():
            continue
        if not torch.isfinite(sampled['displacement'][valid]).all():
            raise ValueError('nonfinite predicted auxiliary displacement')
        error = torch.linalg.vector_norm(sampled['displacement'].float()-truth, dim=-1)
        radius = torch.zeros_like(error)
        if uncertain:
            radius = target[prefix+'_uncertainty'].to(zero.device).float()
            if radius.shape != valid.shape or not torch.isfinite(radius).all() or (radius < 0).any():
                raise ValueError('pseudo uncertainty must be finite nonnegative [B,N,T]')
            if (radius[valid] <= 0).any():
                raise ValueError('surface pseudo labels require a strictly positive uncertainty bound')
            radius = radius*options.pseudo_uncertainty_scale
        residual = (error-radius).clamp_min(0.)
        robust = F.huber_loss(residual, torch.zeros_like(residual), reduction='none', delta=options.huber_delta)
        # Dividing by delta gives L1-like units away from the uncertainty band.
        robust = robust/options.huber_delta
        loss = (robust[valid]*confidence[valid]).sum()/confidence[valid].sum().clamp_min(1e-8)
        scale = target.get('scene_scale', torch.ones(error.shape[0], device=error.device)).to(error.device)
        physical_scale = scale[:, None, None].expand_as(error)
        terms.update({metric+'_loss': loss, metric+'_points': valid.sum().to(error.dtype).detach(),
            metric+'_epe_m': (error[valid]*physical_scale[valid]).mean().detach(),
            metric+'_uncertainty_m': (radius[valid]*physical_scale[valid]).mean().detach(),
            metric+'_outside_uncertainty_fraction': (error[valid]>radius[valid]).float().mean().detach()})
        if weight > 0:
            total = total+weight*loss; supervised = True
    return terms, total, supervised
