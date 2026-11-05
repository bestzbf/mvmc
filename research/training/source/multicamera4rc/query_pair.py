"""Dense agreement between independent queries of one GT surface point.

GT alone chooses correspondences. No GT camera enters the model forward.
The two trajectories already share the scene reference frame: never align
each camera separately. All admitted pixels participate, without a query cap.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class QueryPairOptions:
    depth_absolute_tolerance_m: float = .01
    depth_relative_tolerance: float = .005
    gt_agreement_tolerance_m: float = .01
    gt_agreement_tolerance_normalized: float = .001
    source_reprojection_tolerance_px: float = .01
    huber_delta: float = .01  # scene-normalized XYZ units, like other 3D losses
    # Opt-in class budget AFTER GT correspondence/visibility admission. None
    # preserves the historical loss exactly; no pixels are subsampled.
    dynamic_temporal_fraction: float | None = None

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == 'dynamic_temporal_fraction':
                if value is not None and (not math.isfinite(value) or not 0 < value < 1):
                    raise ValueError('dynamic_temporal_fraction must be None or strictly between 0 and 1')
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')


QUERY_PAIR_METRICS = tuple('query_pair_' + name for name in (
    'consistency', 'points', 'temporal_points', 'dynamic_temporal_points',
    'source_pairs', 'examples', 'epe', 'temporal_epe', 'dynamic_temporal_epe',
    'gt_residual', 'corrected_epe', 'grouped_active', 'dynamic_weight_fraction',
    'dynamic_group_loss', 'other_group_loss'))


def attach_pair_surface_ids(target, recipe):
    """Add a reference to existing GT labels; no allocation or model input."""
    for key in ('dense_segmentation', 'dense_syn4d_face_map'):
        if key in recipe:
            return dict(target, query_pair_surface_id=recipe[key])
    return target


def _sample(image, uv):
    """[C,H,W] -> [C,Hout,Wout], uv denotes integer-array pixel centres."""
    h, w = image.shape[-2:]
    grid = torch.stack((2*(uv[..., 0]+.5)/w-1, 2*(uv[..., 1]+.5)/h-1), -1)
    return F.grid_sample(image[None].float(), grid[None].float(),
                         align_corners=False)[0]


def _corners(uv, h, w):
    x0, y0 = uv[..., 0].floor().long(), uv[..., 1].floor().long()
    wx, wy = uv[..., 0]-x0, uv[..., 1]-y0
    for dx, dy, weight in ((0, 0, (1-wx)*(1-wy)), (1, 0, wx*(1-wy)),
                           (0, 1, (1-wx)*wy), (1, 1, wx*wy)):
        yield (y0+dy).clamp(0, h-1), (x0+dx).clamp(0, w-1), weight > 1e-6


@torch.no_grad()
def _project(points, K, E):
    pc = (points-E[:3, 3]) @ E[:3, :3]
    projected = pc @ K.T
    uv = projected[..., :2]/pc[..., 2:].clamp_min(1e-8)
    finite = torch.isfinite(uv).all(-1) & torch.isfinite(pc).all(-1)
    # Keep invalid coordinates finite before floor/grid_sample/indexing.
    return torch.nan_to_num(uv, nan=-1e6, posinf=1e6, neginf=-1e6), pc[..., 2], finite


@torch.no_grad()
def _visible(points, slot, target, b, scale, options, surface=None, source_ids=None):
    """Conservative GT co-visibility with valid bilinear depth support.

    Depth boundaries and (when available) persistent object/mesh IDs must
    agree at every nonzero interpolation contributor. Unknown is not visible.
    """
    depth = target['depth'][b, slot].detach().float()
    h, w = depth.shape
    uv, z, finite = _project(points, target['K'][b, slot].detach().float(),
                              target['c2w'][b, slot].detach().float())
    # Float32 transforms can put an exact border centre a few ulps outside;
    # tolerate numerical roundoff only, never an actual off-image pixel.
    eps = 1e-4
    inside = (finite & (z > 0) & (uv[..., 0] >= -eps) & (uv[..., 0] <= w-1+eps)
              & (uv[..., 1] >= -eps) & (uv[..., 1] <= h-1+eps))
    uv = torch.stack((uv[..., 0].clamp(0, w-1), uv[..., 1].clamp(0, h-1)), -1)
    known = target['depth_valid'][b, slot].bool() & torch.isfinite(depth) & (depth > 0)
    d = _sample(torch.where(known, depth, 0)[None], uv)[0]
    tolerance = options.depth_absolute_tolerance_m/scale + options.depth_relative_tolerance*d
    mask = inside & (d > 0) & ((z-d).abs() <= tolerance)
    for y, x, active in _corners(uv, h, w):
        good = known[y, x] & ((depth[y, x]-d).abs() <= tolerance)
        if surface is not None:
            good &= surface[b, slot, y, x] == source_ids
        mask &= ~active | good
    return uv, mask


def _all_support(mask, uv):
    """Do not accept a partially valid interpolated GT point."""
    h, w = mask.shape
    result = torch.ones(uv.shape[:-1], dtype=torch.bool, device=mask.device)
    for y, x, active in _corners(uv, h, w):
        result &= ~active | mask[y, x]
    return result


def _sample_xyz(xyz, uv):
    return _sample(xyz.permute(2, 0, 1), uv).permute(1, 2, 0)


def paired_source_order(candidates, camera_id, time_idx, count):
    """Preserve randomized priority; reserve a same-time different-view partner.

    Does not inspect predicted visibility, resample the dataset, add decoder
    sources or change Q*T. Monocular/no-synchronized-partner order is unchanged.
    The identical policy is usable in a zero-weight control experiment.
    """
    if count < 2 or len(candidates) < 2:
        return candidates
    remaining = candidates.tolist()
    chosen = []
    while len(chosen)+2 <= count:
        pair = None
        for a in remaining:
            partner = next((b for b in remaining if camera_id[a] != camera_id[b]
                            and time_idx[a] == time_idx[b]), None)
            if partner is not None:
                pair = (a, partner)
                break
        if pair is None:
            break
        chosen.extend(pair)
        remaining = [item for item in remaining if item not in pair]
    return candidates.new_tensor(chosen+remaining)


def query_pair_consistency(pred, target, estimate, truth, valid,
                           pixel_weights=None, *, options=None):
    """Masked robust agreement of the two branches' 3D prediction errors.

    At corresponding source pixels x and u(x), the residual is
        (P_A - sample(P_B,u)) - stopgrad(G_A - sample(G_B,u)).
    The GT term removes the small rasterization/interpolation discrepancy;
    it does NOT admit inaccurate matches. GT disagreement is bounded in both
    original GT units and scene-normalized units; packed SfM scene_units must
    not be mistaken for a calibrated metric scale.
    It is zero for an exact physical correspondence and makes GT-as-prediction
    an exact optimum even on finite raster grids. Original absolute GT losses
    remain necessary to anchor the shared solution.

    Normalize locally by admitted foreground/background weights, as for the
    existing GT objectives. DDP averages per-example losses, so monocular and
    zero-overlap examples contribute zero, retaining the mixture weighting.
    No distributed collective occurs here: a rank may have no valid pairs.
    Iterate loss maps over time, without another model pass or Q^2*T volume.
    """
    options = options or QueryPairOptions()
    zero = estimate[:0].float().sum()  # view only, finite even with NaN padding
    result = {k: zero.detach().clone() for k in QUERY_PAIR_METRICS[:6]}
    result['query_pair_consistency'] = zero
    required_pred = {'source_obs_idx', 'camera_id', 'time_idx', 'query_time_idx', 'observation_valid'}
    required_gt = {'K', 'c2w', 'depth', 'depth_valid', 'camera_valid'}
    if required_pred-set(pred) or required_gt-set(target):
        return result
    B, Q, T, H, W, _ = truth.shape
    if estimate.shape != truth.shape or valid.shape != truth.shape[:-1]:
        raise ValueError('query pair trajectories/masks must match [B,Q,T,H,W,3]')
    if target['depth'].shape != (B, pred['camera_id'].shape[1], H, W):
        raise ValueError('query pair depth and trajectory must use one GT pixel grid')
    surface = target.get('query_pair_surface_id')
    if surface is not None and surface.shape != target['depth'].shape:
        raise ValueError('query pair surface IDs must match the GT depth pixel grid')
    counts = estimate.new_zeros(6, dtype=torch.float32)
    sums = estimate.new_zeros(5, dtype=torch.float32)
    numerator, denominator = zero, zero.detach().clone()
    grouped = options.dynamic_temporal_fraction is not None
    dynamic_numerator, other_numerator = zero, zero
    dynamic_denominator, other_denominator = zero.detach().clone(), zero.detach().clone()
    with torch.autocast(device_type=estimate.device.type, enabled=False):
        truth = truth.detach().float()
        for b in range(B):
            scale = target.get('scene_scale', truth.new_ones(B))[b].detach().float()
            if not torch.isfinite(scale) or scale <= 0:
                raise ValueError('query pair scene scale must be finite and positive')
            gt_tolerance = torch.minimum(options.gt_agreement_tolerance_m/scale,
                                         scale.new_tensor(options.gt_agreement_tolerance_normalized))
            camera, time = pred['camera_id'][b], pred['time_idx'][b]
            pose_ok = pred['observation_valid'][b].bool() & target['camera_valid'][b].bool()
            pose_ok &= torch.isfinite(target['c2w'][b]).all(dim=(-2, -1))
            pose_ok &= torch.isfinite(target['K'][b]).all(dim=(-2, -1))
            pose_ok &= (target['K'][b, :, 0, 0] > 0) & (target['K'][b, :, 1, 1] > 0)
            if 'camera_translation_valid' in target:
                pose_ok &= target['camera_translation_valid'][b].bool()
            # Duplicate camera/time labels are ambiguous, so do not guess.
            slots = {}
            for s in pose_ok.nonzero().flatten().tolist():
                key = (int(camera[s]), int(time[s]))
                slots[key] = s if key not in slots else None
            if len({key[0] for key in slots}) < 2 or Q < 2:
                continue
            anchors = {}
            with torch.no_grad():
                x = torch.arange(W, device=truth.device)[None]
                y = torch.arange(H, device=truth.device)[:, None]
                for q in range(Q):
                    s = int(pred['source_obs_idx'][b, q])
                    if not 0 <= s < len(camera) or not pose_ok[s]:
                        continue
                    if 'source_valid' in pred and not pred['source_valid'][b, q]:
                        continue
                    at = (pred['query_time_idx'][b] == time[s]).nonzero().flatten()
                    if len(at) != 1 or slots.get((int(camera[s]), int(time[s]))) != s:
                        continue
                    at = int(at[0])
                    ids = None if surface is None else surface[b, s]
                    gt = torch.where(valid[b, q, at, ..., None], truth[b, q, at], 0)
                    # This is an integer source query, so compare its own
                    # depth/labels directly. Interpolating a numerically
                    # reprojected pixel here would reject mesh boundaries for
                    # harmless float32 roundoff in the projection.
                    own_uv, own_z, finite = _project(gt, target['K'][b, s].detach().float(),
                                                     target['c2w'][b, s].detach().float())
                    depth = target['depth'][b, s].detach().float()
                    good = valid[b, q, at] & finite & target['depth_valid'][b, s].bool()
                    good &= torch.isfinite(depth) & (depth > 0) & (own_z > 0)
                    tolerance = options.depth_absolute_tolerance_m/scale + options.depth_relative_tolerance*depth
                    good &= (own_z-depth).abs() <= tolerance
                    good &= (own_uv[..., 0]-x).abs() <= options.source_reprojection_tolerance_px
                    good &= (own_uv[..., 1]-y).abs() <= options.source_reprojection_tolerance_px
                    anchors[q] = (s, at, gt, good, ids)
            example_points = counts[0].clone()
            for qa, (sa, at, anchor_gt, own_valid, source_ids) in anchors.items():
                for qb, (sb, _, _, partner_valid, _) in anchors.items():
                    if qa == qb:
                        continue
                    if camera[sa] == camera[sb] or time[sa] != time[sb]:
                        continue
                    with torch.no_grad():
                        uv, anchor = _visible(anchor_gt, sb, target, b, scale, options, surface, source_ids)
                        anchor &= own_valid & _all_support(partner_valid, uv)
                        partner_gt = _sample_xyz(torch.where(valid[b, qb, at, ..., None], truth[b, qb, at], 0), uv)
                        anchor &= (anchor_gt-partner_gt).norm(dim=-1) <= gt_tolerance
                    if not anchor.any():
                        continue
                    pair_has_points = False
                    for ti in range(T):
                        # Geometry-only clips have no temporal GT: never turn
                        # absent motion into a static consistency label.
                        with torch.no_grad():
                            mask = anchor & valid[b, qa, ti] & _all_support(valid[b, qb, ti], uv)
                            if not mask.any():
                                continue
                            ta = int(pred['query_time_idx'][b, ti])
                            va, vb = slots.get((int(camera[sa]), ta)), slots.get((int(camera[sb]), ta))
                            if va is None or vb is None:
                                continue
                            ga = torch.where(valid[b, qa, ti, ..., None], truth[b, qa, ti], 0)
                            gb = _sample_xyz(torch.where(valid[b, qb, ti, ..., None], truth[b, qb, ti], 0), uv)
                            gt_delta = ga-gb
                            mask &= gt_delta.norm(dim=-1) <= gt_tolerance
                            if ti != at:
                                _, seen_a = _visible(ga, va, target, b, scale, options, surface, source_ids)
                                _, seen_b = _visible(ga, vb, target, b, scale, options, surface, source_ids)
                                mask &= seen_a & seen_b
                            if not mask.any():
                                continue
                        pa, pb = estimate[b, qa, ti].float(), estimate[b, qb, ti].float()
                        finite_b = torch.isfinite(pb).all(-1)
                        if not _all_support(finite_b, uv)[mask].all() or not torch.isfinite(pa[mask]).all():
                            raise FloatingPointError('non-finite prediction at an admitted query pair')
                        pb = _sample_xyz(torch.where(finite_b[..., None], pb, 0), uv)
                        difference = pa[mask]-pb[mask]
                        raw = difference.detach().norm(dim=-1)
                        error = (difference-gt_delta[mask]).norm(dim=-1)
                        delta = options.huber_delta
                        robust = torch.where(error <= delta, .5*error.square()/delta, error-.5*delta)
                        weight = (torch.ones_like(error) if pixel_weights is None else
                                  torch.broadcast_to(pixel_weights, valid.shape)[b, qa, ti][mask].float())
                        numerator = numerator+(robust*weight).sum()
                        denominator = denominator+weight.sum()
                        n = mask.sum()
                        counts[0] += n
                        sums[0] += raw.sum()
                        sums[3] += gt_delta[mask].norm(dim=-1).sum()
                        sums[4] += error.detach().sum()
                        if ti != at:
                            counts[1] += n
                            sums[1] += raw.sum()
                            fg = target.get('trajectory_foreground')
                            # Foreground membership alone does not establish
                            # motion (e.g. a stationary object on the ground).
                            dynamic = (ga[mask] - anchor_gt[mask]).norm(dim=-1) > 1e-5
                            if fg is not None:
                                dynamic &= torch.broadcast_to(fg, valid.shape)[b, qa, ti][mask].bool()
                            counts[2] += dynamic.sum()
                            sums[2] += raw[dynamic].sum()
                        elif grouped:
                            dynamic = torch.zeros_like(error, dtype=torch.bool)
                        if grouped:
                            # Classification uses the exact same admitted GT
                            # point set as the diagnostic above. Source-time
                            # geometry belongs to 'other', never dynamic GT.
                            dynamic_numerator = dynamic_numerator+(robust[dynamic]*weight[dynamic]).sum()
                            dynamic_denominator = dynamic_denominator+weight[dynamic].sum()
                            other_numerator = other_numerator+(robust[~dynamic]*weight[~dynamic]).sum()
                            other_denominator = other_denominator+weight[~dynamic].sum()
                        pair_has_points = True
                    counts[3] += int(pair_has_points)
            counts[4] += (counts[0] > example_points).float()
        result.update(query_pair_consistency=numerator/denominator.clamp_min(1e-8),
                      query_pair_points=counts[0], query_pair_temporal_points=counts[1],
                      query_pair_dynamic_temporal_points=counts[2], query_pair_source_pairs=counts[3],
                      query_pair_examples=counts[4])
        if counts[0] > 0:
            result.update(query_pair_epe=sums[0]/counts[0], query_pair_gt_residual=sums[3]/counts[0],
                          query_pair_corrected_epe=sums[4]/counts[0])
            if grouped:
                dynamic_mean = dynamic_numerator/dynamic_denominator.clamp_min(1e-8)
                other_mean = other_numerator/other_denominator.clamp_min(1e-8)
                both = (dynamic_denominator > 0) & (other_denominator > 0)
                fraction = float(options.dynamic_temporal_fraction)
                # A missing group contributes no invented label and does not
                # reduce the budget of the remaining group. Empty samples keep
                # the graph-connected zero returned by the original path.
                result['query_pair_consistency'] = torch.where(both,
                    fraction*dynamic_mean+(1-fraction)*other_mean,
                    numerator/denominator.clamp_min(1e-8))
                result['query_pair_grouped_active'] = both.float()
                result['query_pair_dynamic_weight_fraction'] = torch.where(both,
                    denominator.new_tensor(fraction), (dynamic_denominator > 0).float())
                if dynamic_denominator > 0:
                    result['query_pair_dynamic_group_loss'] = dynamic_mean.detach()
                if other_denominator > 0:
                    result['query_pair_other_group_loss'] = other_mean.detach()
        for i, name in ((1, 'temporal'), (2, 'dynamic_temporal')):
            if counts[i] > 0:
                result[f'query_pair_{name}_epe'] = sums[i]/counts[i]
    return result
