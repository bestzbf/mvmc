"""Confidence-independent dense depth objectives in the existing scene gauge.

Temporal consistency compares prediction/GT *scale errors*, not depth values or
material points. Dynamic objects and camera motion therefore remain legitimate.
All eligibility and near/far groups are determined exclusively by ground truth.
"""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


DEPTH_OBJECTIVE_METRICS = tuple('depth_objective_'+name for name in (
    'relative', 'log_bias', 'temporal_scale', 'observations', 'temporal_pairs',
    'log_bias_signed', 'log_bias_abs', 'floor_fraction',
    'prediction_below_floor_fraction', 'prediction_above_scale_fraction',
    'confidence_saturated_fraction',
)) + tuple(f'depth_objective_{bucket}_{suffix}'
           for bucket in ('near', 'mid', 'far')
           for suffix in ('pixels', 'relative_error'))


@dataclass(frozen=True)
class DepthObjectiveOptions:
    relative_weight: float = 0.
    log_bias_weight: float = 0.
    temporal_scale_weight: float = 0.
    relative_floor_fraction: float = .01
    robust_beta: float = .1
    temporal_single_camera_only: bool = True
    diagnostics: bool = False
    near_ratio: float = .5
    far_ratio: float = 2.
    confidence_saturation: float = 10000.
    prediction_upper_ratio: float = 100.

    def __post_init__(self):
        for name in ('relative_weight', 'log_bias_weight', 'temporal_scale_weight'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('relative_floor_fraction', 'robust_beta', 'near_ratio',
                     'far_ratio', 'confidence_saturation', 'prediction_upper_ratio'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not self.near_ratio < self.far_ratio:
            raise ValueError('near_ratio must be smaller than far_ratio')
        if self.relative_floor_fraction >= 1:
            raise ValueError('relative_floor_fraction must be smaller than one')
        if self.confidence_saturation < 1 or self.prediction_upper_ratio <= 1:
            raise ValueError('saturation and upper prediction thresholds must be >=1 and >1')
        if not isinstance(self.temporal_single_camera_only, bool) or not isinstance(self.diagnostics, bool):
            raise ValueError('temporal_single_camera_only and diagnostics must be boolean')

    @property
    def enabled(self):
        return bool(self.relative_weight or self.log_bias_weight
                    or self.temporal_scale_weight or self.diagnostics)


def _clip_mean(values, supported):
    """Average observations within each clip, then clips with actual support."""
    count = supported.sum(1)
    per_clip = torch.where(supported, values, 0).sum(1)/count.clamp_min(1)
    return per_clip[count > 0].mean()


def dense_depth_objectives(prediction, truth, valid, pred, options):
    """Return unweighted terms/diagnostics and their configured weighted loss.

    Tensors have shape [B,S,H,W] after the caller resizes prediction to the GT
    grid. The denominator floor is a fraction of each observation's mean valid
    GT depth, so changing scene units does not change any objective. Invalid
    pixels are replaced before arithmetic, including logarithms and reductions.
    """
    if prediction.shape != truth.shape or truth.ndim != 4 or valid.shape != truth.shape:
        raise ValueError('depth objectives require matching [B,S,H,W] tensors')
    truth = truth.to(device=prediction.device).float().detach()
    valid = valid.to(device=prediction.device).bool() & torch.isfinite(truth) & (truth > 0)
    p = prediction.float()
    if not valid.any():
        # Selecting first avoids NaN/Inf invalid padding leaking into a zero.
        return {}, p[valid].sum()*0
    if not torch.isfinite(p[valid]).all() or (p[valid] <= 0).any():
        raise FloatingPointError('depth objectives require finite positive predictions on valid GT')
    safe_p = torch.where(valid, p, 1.)
    safe_g = torch.where(valid, truth, 1.)
    counts = valid.sum((-1, -2))
    supported = counts > 0
    gt_mean = torch.where(valid, truth, 0.).sum((-1, -2))/counts.clamp_min(1)
    floor = (gt_mean*options.relative_floor_fraction).clamp_min(torch.finfo(torch.float32).tiny)
    denominator = safe_g.clamp_min(floor[..., None, None])
    residual = (safe_p-safe_g)/denominator
    log_ratio = safe_p.log()-safe_g.log()
    if not torch.isfinite(residual[valid]).all():
        raise FloatingPointError('nonfinite relative depth residual on valid GT')
    def observation_mean(values):
        return torch.where(valid, values, 0.).sum((-1, -2))/counts.clamp_min(1)
    def robust(values):
        return F.smooth_l1_loss(values, torch.zeros_like(values),
                                beta=options.robust_beta, reduction='none')
    bias = observation_mean(log_ratio)
    relative = _clip_mean(observation_mean(robust(residual)), supported)
    anchor = _clip_mean(robust(bias), supported)
    terms = dict(depth_objective_relative=relative, depth_objective_log_bias=anchor)
    total = options.relative_weight*relative+options.log_bias_weight*anchor
    temporal_losses, pair_count = [], 0
    if options.temporal_scale_weight:
        if truth.shape[1] > 1 and not all(key in pred for key in ('camera_id', 'time_idx')):
            raise ValueError('temporal depth scale supervision needs camera_id and time_idx')
        if all(key in pred for key in ('camera_id', 'time_idx')):
            camera = torch.as_tensor(pred['camera_id'], device=p.device)
            times = torch.as_tensor(pred['time_idx'], device=p.device)
            if camera.shape != counts.shape or times.shape != counts.shape:
                raise ValueError('depth objective camera_id/time_idx must have shape [B,S]')
            if camera.is_floating_point() or times.is_floating_point():
                raise ValueError('depth objective camera_id/time_idx must be integer labels')
            observed = torch.as_tensor(pred.get('observation_valid', torch.ones_like(counts, dtype=torch.bool)), device=p.device)
            if observed.shape != counts.shape or observed.dtype != torch.bool:
                raise ValueError('depth objective observation_valid must be bool [B,S]')
            for b in range(truth.shape[0]):
                eligible = supported[b] & (camera[b] >= 0)
                cameras = camera[b, eligible].unique()
                physical_cameras = camera[b, observed[b] & (camera[b] >= 0)].unique()
                if options.temporal_single_camera_only and len(physical_cameras) != 1:
                    continue
                streams = []
                for cam in cameras:
                    ids = ((camera[b] == cam) & eligible).nonzero(as_tuple=True)[0]
                    ids = ids[times[b, ids].argsort()]
                    if len(ids) < 2:
                        continue
                    if (times[b, ids].diff() <= 0).any():
                        raise ValueError('temporal depth scale requires distinct times for each camera')
                    # Different surfaces may be visible in successive frames.
                    # Only their observation-level log prediction/GT bias is compared.
                    streams.append(robust(bias[b, ids].diff()).mean())
                    pair_count += len(ids)-1
                if streams:
                    temporal_losses.append(torch.stack(streams).mean())
    temporal = torch.stack(temporal_losses).mean() if temporal_losses else (safe_p[valid][:1]*0).sum()
    terms['depth_objective_temporal_scale'] = temporal
    total = total+options.temporal_scale_weight*temporal
    if options.diagnostics:
        with torch.no_grad():
            terms.update(depth_objective_observations=supported.sum().float(),
                         depth_objective_temporal_pairs=p.new_tensor(float(pair_count)),
                         depth_objective_log_bias_signed=_clip_mean(bias, supported),
                         depth_objective_log_bias_abs=_clip_mean(bias.abs(), supported),
                         depth_objective_floor_fraction=(truth[valid] < floor[..., None, None].expand_as(p)[valid]).float().mean(),
                         depth_objective_prediction_below_floor_fraction=(p[valid] < floor[..., None, None].expand_as(p)[valid]).float().mean(),
                         depth_objective_prediction_above_scale_fraction=(p[valid] > (gt_mean*options.prediction_upper_ratio)[..., None, None].expand_as(p)[valid]).float().mean())
            ratio = safe_g/gt_mean.clamp_min(torch.finfo(torch.float32).tiny)[..., None, None]
            buckets = {'near': ratio < options.near_ratio,
                       'mid': (ratio >= options.near_ratio) & (ratio <= options.far_ratio),
                       'far': ratio > options.far_ratio}
            for name, bucket in buckets.items():
                mask = valid & bucket
                terms[f'depth_objective_{name}_pixels'] = mask.sum().float()
                if mask.any():
                    terms[f'depth_objective_{name}_relative_error'] = residual[mask].abs().mean()
            if 'depth_conf' in pred:
                confidence = pred['depth_conf'].to(p.device)
                if confidence.shape != valid.shape:
                    raise ValueError('depth objective diagnostic confidence must use the depth GT grid')
                if not torch.isfinite(confidence[valid]).all():
                    raise FloatingPointError('nonfinite depth confidence on valid GT')
                terms['depth_objective_confidence_saturated_fraction'] = (confidence[valid] >= options.confidence_saturation).float().mean()
    return terms, total
