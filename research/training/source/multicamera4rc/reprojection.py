"""GT-supported image correspondence supervision, independent of visibility scores.

Project predicted world trajectories with predicted cameras and compare with
the corresponding GT projection. An independently supplied GT trajectory and
GT visibility (or conservative measured-depth witness) are required. Comparing
two projections made from the same prediction would provide no supervision.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .query_pair import QueryPairOptions, _project, _visible


REPROJECTION_METRICS = tuple('joint_reprojection_' + name for name in (
    'loss', 'pixels', 'views', 'epe_px', 'behind_fraction', 'dense_pixels', 'sparse_pixels'))


@dataclass(frozen=True)
class ReprojectionOptions:
    weight: float = 0.
    huber_delta: float = .01  # fractions of the RGB canvas, not scene units
    max_points_per_view: int = 2048
    minimum_depth_ratio: float = .05

    def __post_init__(self):
        if isinstance(self.weight, bool) or not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError('reprojection weight must be finite and nonnegative')
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError('reprojection Huber delta must be finite and positive')
        if (isinstance(self.max_points_per_view, bool)
                or not isinstance(self.max_points_per_view, int) or self.max_points_per_view <= 0):
            raise ValueError('reprojection point budget must be a positive integer')
        if not math.isfinite(self.minimum_depth_ratio) or not 0 < self.minimum_depth_ratio < 1:
            raise ValueError('reprojection minimum depth ratio must be in (0,1)')


def reprojection_losses(pred, target, options):
    """Equal weight per nonempty source/time/view, with a bounded pixel budget.

    No model confidence or predicted visibility can remove a GT residual.
    The depth denominator floor is tied to detached GT depth so points behind
    the camera remain penalized without division by almost zero. Existing
    depth/XYZ objectives still supervise metric scale. Source self-projection
    is excluded because unprojection followed by projection is an identity.
    """
    zero = pred['trajectory'].new_zeros((), dtype=torch.float32)
    if not options.weight:
        return {}, zero
    required = {'K', 'c2w', 'camera_id', 'time_idx', 'query_time_idx',
                'source_obs_idx', 'source_valid', 'observation_valid', 'output_camera_id'}
    if required - set(pred):
        raise ValueError('reprojection needs camera/time/source metadata')
    if not {'K', 'c2w'} <= set(target):
        return {}, zero
    pred_h, pred_w = pred['trajectory'].shape[-3:-1]
    if 'trajectory' in target and target['trajectory'].ndim == 6:
        height, width = target['trajectory'].shape[-3:-1]
    elif 'depth' in target and target['depth'].ndim == 4:
        height, width = target['depth'].shape[-2:]
    else:
        height, width = pred_h, pred_w
    canvas = zero.new_tensor([width, height])
    options_depth = QueryPairOptions()
    losses, errors, behinds, counts, dense_counts, sparse_counts = [], [], [], [], [], []
    observed = pred['observation_valid'].bool()
    camera_known = observed & target.get('camera_valid', observed).bool()
    camera_known &= target.get('camera_translation_valid', observed).bool()

    def _scale_pred_K(pk):
        if (height, width) == (pred_h, pred_w):
            return pk
        ratio = pk.new_tensor((width / pred_w, height / pred_h, 1.0))
        scaled = (pk * ratio[:, None]).clone()
        scaled[:2, 2] += (ratio[:2] - 1.0) / 2.0
        return scaled

    def add_view(estimate, truth, valid, bi, si, visible=None, visibility_known=None, *, sparse=False):
        if not sparse and estimate.shape[-3:-1] != truth.shape[-3:-1]:
            from .losses import _resize_vector
            estimate = _resize_vector(estimate, tuple(truth.shape[-3:-1]))
        # Flatten only this source/time, never form S*Q*T*H*W camera copies.
        truth = truth.detach().float().reshape(-1, 3)
        estimate = estimate.float().reshape(-1, 3)
        valid = valid.bool().reshape(-1) & torch.isfinite(truth).all(-1)
        ids = valid.nonzero().flatten()
        if not len(ids):
            return
        ge, gk = target['c2w'][bi, si].detach().float(), target['K'][bi, si].detach().float()
        if not torch.isfinite(ge).all() or not torch.isfinite(gk).all():
            raise ValueError('nonfinite GT camera marked valid')
        with torch.no_grad():
            uv, gz, finite = _project(truth[ids], gk, ge)
            admitted = (finite & (gz > 0) & (uv[:, 0] >= -1e-4) & (uv[:, 0] <= width-1+1e-4)
                        & (uv[:, 1] >= -1e-4) & (uv[:, 1] <= height-1+1e-4))
            if visible is not None and visibility_known is not None:
                admitted &= visibility_known.reshape(-1)[ids].bool() & (visible.reshape(-1)[ids] > .5)
            elif {'depth', 'depth_valid', 'scene_scale'} <= set(target):
                _, witness = _visible(truth[ids][None], si, target, bi,
                                      target['scene_scale'][bi], options_depth)
                admitted &= witness[0]
            else:
                return  # Unknown target visibility is not a positive label.
            ids, uv, gz = ids[admitted], uv[admitted], gz[admitted]
            if len(ids) > options.max_points_per_view:
                selected = torch.linspace(0, len(ids)-1, options.max_points_per_view,
                                          device=ids.device).long()
                ids, uv, gz = ids[selected], uv[selected], gz[selected]
        if not len(ids):
            return
        pe, pk = pred['c2w'][bi, si].float(), _scale_pred_K(pred['K'][bi, si].float())
        points = estimate[ids]
        if not all(torch.isfinite(x).all() for x in (points, pe, pk)):
            raise FloatingPointError('nonfinite prediction on admitted reprojection GT')
        pc = (points-pe[:3, 3]) @ pe[:3, :3]
        positive_floor = options.minimum_depth_ratio*gz
        projected = pc @ pk.T
        puv = projected[:, :2]/pc[:, 2:].clamp_min(positive_floor[:, None])
        residual = (puv-uv)/canvas
        robust = F.huber_loss(residual, torch.zeros_like(residual), reduction='none',
                              delta=options.huber_delta)/options.huber_delta
        positive_penalty = (positive_floor-pc[:, 2]).clamp_min(0)/gz
        losses.append(robust.mean()+positive_penalty.mean())
        errors.append((puv-uv).norm(dim=-1).mean().detach())
        behinds.append((pc[:, 2] <= 0).float().mean().detach())
        counts.append(len(ids))
        (sparse_counts if sparse else dense_counts).append(len(ids))

    with torch.autocast(device_type=zero.device.type, enabled=False):
        if {'trajectory', 'trajectory_valid'} <= set(target):
            for bi in range(len(observed)):
                for qi, source in enumerate(pred['source_obs_idx'][bi]):
                    if not pred['source_valid'][bi, qi]:
                        continue
                    for ti, time in enumerate(pred['query_time_idx'][bi]):
                        slots = (camera_known[bi] & (pred['time_idx'][bi] == time)).nonzero().flatten()
                        for si in slots:
                            if si == source:
                                continue
                            visible = known = None
                            if {'visibility', 'visibility_valid'} <= set(target):
                                ci = (pred['output_camera_id'][bi] == pred['camera_id'][bi, si]).nonzero().flatten()
                                if len(ci) != 1:
                                    raise ValueError('ambiguous reprojection camera axis')
                                visible = target['visibility'][bi, qi, ti, ci[0]]
                                known = target['visibility_valid'][bi, qi, ti, ci[0]]
                            add_view(pred['trajectory'][bi, qi, ti], target['trajectory'][bi, qi, ti],
                                     target['trajectory_valid'][bi, qi, ti], bi, si, visible, known)
        if {'sparse_trajectory', 'sparse_trajectory_valid', 'sparse_queries'} <= set(target):
            from .tasks import sample_tracks
            sampled = sample_tracks(pred, target['sparse_queries'].to(zero.device), strict=False)
            for bi in range(len(observed)):
                valid_query = sampled['query_valid'][bi] & target['sparse_query_valid'][bi].bool()
                sources = target['sparse_queries'][bi, :, 0].long()
                for ti, time in enumerate(pred['query_time_idx'][bi]):
                    slots = (camera_known[bi] & (pred['time_idx'][bi] == time)).nonzero().flatten()
                    for si in slots:
                        valid = valid_query & target['sparse_trajectory_valid'][bi, :, ti].bool() & (sources != si)
                        visible = known = None
                        if {'sparse_visibility', 'sparse_visibility_valid'} <= set(target):
                            ci = (pred['output_camera_id'][bi] == pred['camera_id'][bi, si]).nonzero().flatten()
                            if len(ci) != 1:
                                raise ValueError('ambiguous sparse reprojection camera axis')
                            visible = target['sparse_visibility'][bi, :, ti, ci[0]]
                            known = target['sparse_visibility_valid'][bi, :, ti, ci[0]]
                        add_view(sampled['trajectory'][bi, :, ti], target['sparse_trajectory'][bi, :, ti],
                                 valid, bi, si, visible, known, sparse=True)
    if not losses:
        return {}, zero
    loss = torch.stack(losses).mean()
    values = (loss, zero.new_tensor(sum(counts)), zero.new_tensor(len(counts)), torch.stack(errors).mean(),
              torch.stack(behinds).mean(), zero.new_tensor(sum(dense_counts)), zero.new_tensor(sum(sparse_counts)))
    return dict(zip(REPROJECTION_METRICS, values)), options.weight*loss
