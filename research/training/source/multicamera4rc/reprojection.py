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


_GEOMETRY_METRICS = tuple('joint_reprojection_' + name for name in (
    'loss', 'pixels', 'views', 'epe_px', 'behind_fraction', 'dense_pixels', 'sparse_pixels'))
_STEREO_METRICS = tuple('joint_reprojection_stereo_' + name for name in ('loss', 'pixels', 'epe_px'))
REPROJECTION_METRICS = _GEOMETRY_METRICS + _STEREO_METRICS


@dataclass(frozen=True)
class ReprojectionOptions:
    weight: float = 0.
    huber_delta: float = .01  # fractions of the RGB canvas, not scene units
    max_points_per_view: int = 2048
    minimum_depth_ratio: float = .05
    stereo_certified_weight: float = 0.

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
        if (isinstance(self.stereo_certified_weight, bool) or not math.isfinite(self.stereo_certified_weight)
                or self.stereo_certified_weight < 0):
            raise ValueError('certified stereo reprojection weight must be finite and nonnegative')


def _robust_projection(points, uv, gz, k, c, canvas, options):
    pc = (points - c[:3, 3]) @ c[:3, :3]
    positive_floor = options.minimum_depth_ratio * gz
    projected = (pc @ k.T)[:, :2] / pc[:, 2:].clamp_min(positive_floor[:, None])
    error = projected - uv
    residual = error / canvas
    robust = F.huber_loss(residual, torch.zeros_like(residual), reduction='none',
                          delta=options.huber_delta) / options.huber_delta
    positive_penalty = (positive_floor - pc[:, 2]).clamp_min(0) / gz
    return robust.mean() + positive_penalty.mean(), error.norm(dim=-1).mean().detach(), (
        pc[:, 2] <= 0
    ).float().mean().detach()


def _certified_stereo_losses(pred, target, sampled, queries, query_valid, camera_known, canvas, zero, options):
    """Compare motion with a directly matched right-camera pixel.

    Use the fixed pseudo-metric source anchor and dataset cameras from the
    existing stereo gauge. Only displacement is trainable in this residual;
    it cannot optimize the predicted depth, anchor or camera/baseline.
    """
    measured = target['sparse_stereo_uv'].detach().float()
    known = target['sparse_stereo_uv_valid'].bool()
    truth = target['sparse_trajectory'].detach().float()
    truth_valid = target['sparse_trajectory_valid'].bool()
    if (known.flatten(1).any(-1) & ~target['sparse_trajectory_is_pseudo'].bool()).any():
        raise ValueError('certified stereo UV requires the Stereo4D pseudo-metric gauge')
    if (measured.shape != (*known.shape, 2) or known.shape != (*truth_valid.shape, pred['output_camera_id'].shape[1])):
        raise ValueError('certified stereo UV must use sparse query, time and camera axes')
    losses, errors, counts = [], [], []
    for bi in range(len(known)):
        if not known[bi].any():
            continue
        times = pred['query_time_idx'][bi]
        source_times = sampled['source_time_idx'][bi]
        source_time_matches = source_times[:, None] == times[None, :]
        anchors = source_time_matches.long().argmax(-1)
        nodes = torch.arange(len(anchors), device=anchors.device)
        base = truth[bi, nodes, anchors]
        source_slots = queries[bi, :, 0].long().clamp(0, pred['camera_id'].shape[1] - 1)
        anchored = (query_valid[bi] & (pred['camera_id'][bi, source_slots] == 0)
                    & source_time_matches.any(-1)
                    & truth_valid[bi, nodes, anchors] & torch.isfinite(base).all(-1))
        for ti, time in enumerate(times):
            slots = (camera_known[bi] & (pred['time_idx'][bi] == time)
                     & (pred['camera_id'][bi] == 1)).nonzero().flatten()
            for si in slots:
                ci = (pred['output_camera_id'][bi] == 1).nonzero().flatten()
                if len(ci) != 1:
                    raise ValueError('ambiguous certified stereo camera axis')
                uv = measured[bi, :, ti, ci[0]]
                valid = (known[bi, :, ti, ci[0]] & anchored & (source_times < time)
                         & truth_valid[bi, :, ti] & ((uv >= 0) & (uv <= canvas - 1)).all(-1))
                ids = valid.nonzero().flatten()
                if not len(ids):
                    continue
                k = target['K'][bi, si].detach().float()
                c = target['c2w'][bi, si].detach().float()
                if not torch.isfinite(k).all() or not torch.isfinite(c).all():
                    raise ValueError('nonfinite GT camera marked valid')
                _, gz, finite = _project(truth[bi, ids, ti], k, c)
                admit = finite & (gz > 0)
                ids, gz = ids[admit], gz[admit]
                if len(ids) > options.max_points_per_view:
                    selected = torch.linspace(0, len(ids)-1, options.max_points_per_view,
                                              device=ids.device).long()
                    ids, gz = ids[selected], gz[selected]
                if not len(ids):
                    continue
                points = base[ids] + sampled['displacement'][bi, ids, ti].float()
                if not torch.isfinite(points).all():
                    raise FloatingPointError('nonfinite displacement on certified stereo match')
                loss, error, _ = _robust_projection(points, uv[ids], gz, k, c, canvas, options)
                losses.append(loss)
                errors.append(error)
                counts.append(len(ids))
    if not losses:
        return {}, zero
    loss = torch.stack(losses).mean()
    values = (loss, zero.new_tensor(sum(counts)), torch.stack(errors).mean())
    return dict(zip(_STEREO_METRICS, values)), options.stereo_certified_weight * loss


def reprojection_losses(pred, target, options):
    """Equal weight per nonempty source/time/view, with a bounded pixel budget.

    No model confidence or predicted visibility can remove a GT residual.
    The depth denominator floor is tied to detached GT depth so points behind
    the camera remain penalized without division by almost zero. Existing
    depth/XYZ objectives still supervise metric scale. Source self-projection
    is excluded because unprojection followed by projection is an identity.

    Sparse query pixels use the GT canvas from the dense trajectory or depth
    shape. With no such raster label, they use the prediction canvas.
    """
    zero = pred['trajectory'].new_zeros((), dtype=torch.float32)
    stereo_enabled = options.stereo_certified_weight and {
        'sparse_stereo_uv', 'sparse_stereo_uv_valid', 'sparse_trajectory_is_pseudo',
    } <= set(target)
    if not (options.weight or stereo_enabled):
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
        loss, error, behind = _robust_projection(points, uv, gz, pk, pe, canvas, options)
        losses.append(loss)
        errors.append(error)
        behinds.append(behind)
        counts.append(len(ids))
        (sparse_counts if sparse else dense_counts).append(len(ids))

    stereo_terms, stereo_total = {}, zero
    with torch.autocast(device_type=zero.device.type, enabled=False):
        if options.weight and {'trajectory', 'trajectory_valid'} <= set(target):
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
        if (options.weight or stereo_enabled) and {'sparse_trajectory', 'sparse_trajectory_valid', 'sparse_queries'} <= set(target):
            from .tasks import rescale_queries_to_prediction, sample_tracks
            queries = target['sparse_queries'].to(zero.device)
            gt_query_valid = target['sparse_query_valid'].bool()
            queries, gt_query_valid = rescale_queries_to_prediction(
                queries, (pred_h, pred_w), target, gt_query_valid,
            )
            sampled = sample_tracks(pred, queries, strict=False)
            if stereo_enabled:
                stereo_terms, stereo_total = _certified_stereo_losses(
                    pred, target, sampled, queries, sampled['query_valid'] & gt_query_valid,
                    camera_known, canvas, zero, options,
                )
            for bi in range(len(observed) if options.weight else 0):
                valid_query = sampled['query_valid'][bi] & gt_query_valid[bi]
                sources = queries[bi, :, 0].long()
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
        return stereo_terms, stereo_total
    loss = torch.stack(losses).mean()
    values = (loss, zero.new_tensor(sum(counts)), zero.new_tensor(len(counts)), torch.stack(errors).mean(),
              torch.stack(behinds).mean(), zero.new_tensor(sum(dense_counts)), zero.new_tensor(sum(sparse_counts)))
    total = options.weight * loss
    if stereo_terms:
        total = total + stereo_total
    return {**dict(zip(_GEOMETRY_METRICS, values)), **stereo_terms}, total
