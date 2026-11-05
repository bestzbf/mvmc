"""Dense 3D residuals in synchronized *other* physical cameras.

Both XYZ and c2w translations use the existing common scene scale and gauge.
GT visibility is not required: a known 3D point remains supervised when hidden
or outside the destination image. Neither predicted confidence nor visibility
can switch off this objective. No extra model forward or point sampling occurs.
"""
from collections.abc import Mapping

import torch
from torch import Tensor

CROSS_CAMERA_METRICS = (
    'cross_camera_3d', 'cross_camera_3d_epe', 'cross_camera_3d_points',
    'cross_camera_3d_identity_epe', 'cross_camera_3d_temporal_epe',
    'cross_camera_3d_delta_points', 'cross_camera_3d_delta_px',
    'cross_camera_3d_static_delta_px', 'cross_camera_3d_static_epe', 'cross_camera_3d_static_points',
    'cross_camera_3d_dynamic_delta_px', 'cross_camera_3d_dynamic_epe', 'cross_camera_3d_dynamic_points',
)


def cross_camera_3d(pred: Mapping[str, Tensor], target: Mapping[str, Tensor],
                    estimate: Tensor, truth: Tensor, valid: Tensor,
                    pixel_weights: Tensor | None = None, *, report_delta=False) -> dict[str, Tensor]:
    """Return a pair/pixel weighted mean in normalized scene units.

    XYZ: [B,Q,T,H,W,3]; poses: [B,S,4,4]. Match target query times to
    observation times, and exclude *all* observations from the source camera,
    not just the source slot. M eligible (batch,source,time,target-view) tuples
    become [M,H*W,3] batched row-vector transforms (P - t) @ R_c2w.
    All H*W pixels with authoritative GT are retained; normalization prevents
    camera count or image area from multiplying the loss coefficient.
    """
    zero = estimate.new_zeros((), dtype=torch.float32)
    # Missing diagnostic GT is absent, not a zero-error observation. This
    # matters when a different DDP rank trains monocular/calibration-only data.
    result = {'cross_camera_3d': zero, 'cross_camera_3d_points': zero}
    if not valid.any() or 'c2w' not in target:
        return result
    required = {'c2w', 'camera_id', 'time_idx', 'query_time_idx', 'source_obs_idx'}
    if required - set(pred):
        raise ValueError('cross-camera 3D supervision needs camera/time/source metadata')
    camera_id, times = pred['camera_id'], pred['time_idx']
    observation = pred.get('observation_valid', torch.ones_like(camera_id, dtype=torch.bool)).bool()
    source = pred['source_obs_idx']
    source_valid = pred.get('source_valid', torch.ones_like(source, dtype=torch.bool)).bool()
    B, S = camera_id.shape
    if (estimate.shape != truth.shape or truth.ndim != 6 or truth.shape[-1] != 3
            or valid.shape != truth.shape[:-1] or times.shape != (B,S)
            or source.shape != truth.shape[:2] or observation.shape != (B,S)
            or source_valid.shape != source.shape
            or pred['query_time_idx'].shape != (B,truth.shape[2])):
        raise ValueError('incompatible dense cross-camera tensor axes')
    if (((source < 0) | (source >= S)) & source_valid).any():
        raise ValueError('valid source observation is out of bounds')
    index = source.clamp(0, S-1)
    source_valid = source_valid & observation.gather(1,index)
    source_camera = camera_id.gather(1,index)
    duplicate = ((camera_id[:,:,None] == camera_id[:,None,:])
                 & (times[:,:,None] == times[:,None,:])
                 & observation[:,:,None] & observation[:,None,:]
                 & ~torch.eye(S,device=times.device,dtype=torch.bool)[None])
    if duplicate.any():
        raise ValueError('duplicate physical camera/time observations')
    gt_pose = target['c2w'].to(estimate.device).detach()
    if gt_pose.shape != (B,S,4,4) or pred['c2w'].shape != gt_pose.shape:
        raise ValueError('cross-camera c2w must be matching [B,S,4,4]')
    pose_valid = observation & torch.isfinite(gt_pose).all(dim=(-1,-2))
    for key in ('camera_valid', 'camera_translation_valid'):
        if key in target:
            pose_valid = pose_valid & torch.broadcast_to(target[key].to(estimate.device).bool(), (B,S))
    match_time = pred['query_time_idx'][:,:,None] == times[:,None,:]
    eligible = (pose_valid[:,None,None,:] & match_time[:,None,:,:]
                & (source_camera[:,:,None,None] != camera_id[:,None,None,:])
                & source_valid[:,:,None,None] & valid.any(dim=(-1,-2))[...,None])
    bi, qi, ti, si = eligible.nonzero(as_tuple=True)
    if bi.numel() == 0:
        return result
    # Compute the transforms in FP32 even inside the bf16 model autocast.
    with torch.autocast(device_type=estimate.device.type, enabled=False):
        mask = valid[bi,qi,ti]
        p = torch.where(mask[...,None], estimate[bi,qi,ti].float(), 0.)
        g = torch.where(mask[...,None], truth[bi,qi,ti].detach().float(), 0.)
        pe, ge = pred['c2w'][bi,si].float(), gt_pose[bi,si].float()
        if not torch.isfinite(p).all() or not torch.isfinite(pe).all():
            raise FloatingPointError('non-finite prediction in cross-camera 3D supervision')
        pc = torch.bmm(p.flatten(1,2)-pe[:,None,:3,3], pe[:,:3,:3])
        gc = torch.bmm(g.flatten(1,2)-ge[:,None,:3,3], ge[:,:3,:3])
        error = torch.linalg.vector_norm(pc-gc,dim=-1).reshape_as(mask)
        if not torch.isfinite(error[mask]).all():
            raise FloatingPointError('non-finite cross-camera 3D residual')
        weights = mask.float()
        if pixel_weights is not None:
            weights = weights * torch.broadcast_to(pixel_weights,valid.shape)[bi,qi,ti].float()
        result['cross_camera_3d'] = (error*weights).sum()/weights.sum().clamp_min(1e-8)
        result['cross_camera_3d_points'] = mask.sum().float()
        result['cross_camera_3d_epe'] = error[mask].mean()
        if report_delta and 'K' in target:
            # Training-only diagnostic, fixed GT pixel-equivalent radii. No
            # per-camera fitting, prediction gating or benchmark-score claim.
            k = target['K'].to(estimate.device).float()[bi,si]
            focal = (k[:,0,0]+k[:,1,1])/2
            z = gc[...,2].reshape_as(mask)
            measure = mask & (z>0) & torch.isfinite(z) & (focal[:,None,None]>0)
            if measure.any():
                tolerance = z/focal[:,None,None]
                result['cross_camera_3d_delta_px'] = torch.stack([
                    (error[measure] < radius*tolerance[measure]).float().mean()
                    for radius in (1,2,4,8,16)]).mean()*100
                result['cross_camera_3d_delta_points'] = measure.sum().float()
                if 'displacement' in target and 'displacement_valid' in target:
                    motion = target['displacement'].to(estimate.device)[bi,qi,ti].detach().float()
                    known = target['displacement_valid'].to(estimate.device)[bi,qi,ti].bool()
                    temporal = times.gather(1,index)[bi,qi] != pred['query_time_idx'][bi,ti]
                    supported = measure & known & temporal[:,None,None] & torch.isfinite(motion).all(-1)
                    amplitude = motion.norm(dim=-1)
                    dynamic = supported & (amplitude>1e-5)
                    if 'trajectory_foreground' in target:
                        dynamic = dynamic & target['trajectory_foreground'].to(estimate.device)[bi,qi,ti].bool()
                    static = torch.zeros_like(supported)
                    if 'static_mask' in target:
                        sm = target['static_mask'].to(estimate.device).bool()
                        if sm.ndim == 4:sm = sm[:,:,None].expand_as(valid)
                        static = supported & sm[bi,qi,ti] & (amplitude<=1e-5)
                    for name, selected in (('static',static),('dynamic',dynamic)):
                        if selected.any():
                            prefix = 'cross_camera_3d_'+name
                            result[prefix+'_epe'] = error[selected].mean()
                            result[prefix+'_delta_px'] = torch.stack([
                                (error[selected]<radius*tolerance[selected]).float().mean()
                                for radius in (1,2,4,8,16)]).mean()*100
                            result[prefix+'_points'] = selected.sum().float()
        identity = times.gather(1,index)[bi,qi] == pred['query_time_idx'][bi,ti]
        for name, selected in (('identity',identity),('temporal',~identity)):
            selected = selected[:,None,None] & mask
            if selected.any():
                result[f'cross_camera_3d_{name}_epe'] = error[selected].mean()
    return result
