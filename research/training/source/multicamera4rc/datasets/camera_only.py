"""Calibration supervision for real multiview clips without surface geometry.

Human skeleton joints are not depth or material-point labels. Keep measured
intrinsics/rotations and explicitly mask translation, depth, rays and motion.
No scene scale is inferred from a camera baseline or a fitted human skeleton.
"""
import torch

from .adapters.base import InvalidClipError


def attach_camera_only_recipe(clip, sample):
    _attach_camera_recipe(clip, sample, kind='camera_only')


def _attach_camera_recipe(clip, sample, *, kind):
    if clip.metadata.get('supervision_kind') != kind:
        raise ValueError('camera-only supervision needs an explicit adapter contract')
    target = {k:v for k,v in sample['targets'].items() if not k.startswith('sparse_')}
    if target['depth_valid'].any():
        raise ValueError('calibration-only adapter unexpectedly supplied surface depth')
    if not target['camera_valid'].any():
        raise ValueError('calibration-only clip has no measured camera')
    target['camera_translation_valid'] = torch.zeros_like(target['camera_valid'])
    target['dense_camera_only'] = torch.tensor(True)
    sample['targets'] = target


def attach_camera_joint_motion_recipe(clip, sample):
    """Retain relative joint displacement without supervising joint depth or XYZ."""
    sparse = sample['targets']
    queries = sparse['sparse_queries']
    source_times = sample['time_idx'][queries[:, 0].long()]
    cross_time = source_times[:, None] != sample['query_time_idx'][None]
    valid = sparse['sparse_displacement_valid'] & cross_time & sparse['sparse_query_valid'][:, None]
    if not valid.any():
        raise InvalidClipError("prepared EgoPose clip has no labeled cross-frame queries")
    # Positions from the bridge are already expressed in observation zero's
    # gauge. Use the median distance of unique annotated joints as a scale
    # proxy. These positions never become absolute trajectory or depth targets.
    points = sparse['sparse_trajectory'][sparse['sparse_trajectory_valid']]
    distances = torch.linalg.vector_norm(points.unique(dim=0), dim=-1)
    scale = distances.median()
    if not torch.isfinite(scale) or scale <= 1e-6:
        raise InvalidClipError("EgoPose joint scale must be finite and positive")
    _attach_camera_recipe(clip, sample, kind='camera_joint_motion')
    sample['targets'].update(
        sparse_queries=queries,
        sparse_query_valid=valid.any(dim=-1),
        sparse_displacement=torch.where(valid[..., None], sparse['sparse_displacement'], 0),
        sparse_displacement_valid=valid,
        dense_joint_motion_scale=scale,
    )


@torch.no_grad()
def materialize_camera_only_targets(batch, source_indices, *, normalize_scene=True):
    recipe = batch['targets']
    if not recipe['dense_camera_only'].all():
        raise ValueError('camera-only recipe cannot be mixed with surface recipes within a local batch')
    target = {k:v for k,v in recipe.items() if not k.startswith('dense_')}
    b,s,h,w = target['depth'].shape
    if source_indices.ndim != 2 or source_indices.shape[0] != b:
        raise ValueError('source indices must be [B,Q]')
    if target['depth_valid'].any() or target['camera_translation_valid'].any():
        raise ValueError('camera-only recipe must not supervise depth or metric translation')
    q,t = source_indices.shape[1], batch['query_time_idx'].shape[1]
    zero = target['depth'].new_zeros(())
    false = zero.bool()
    motion_shape = (b,q,t,h,w)
    scale = zero.new_ones(b)
    if 'dense_joint_motion_scale' in recipe:
        if t != 2:
            raise ValueError("EgoPose displacement needs two target times")
        if normalize_scene:
            scale = recipe['dense_joint_motion_scale']
        target['sparse_displacement'] = target['sparse_displacement'] / scale[:, None, None, None]
    target.update(ray=zero.expand(b,s,h,w,6), ray_valid=false.expand(b,s,h,w),
        trajectory=zero.expand(*motion_shape,3), trajectory_valid=false.expand(motion_shape),
        displacement=zero.expand(*motion_shape,3), displacement_valid=false.expand(motion_shape),
        trajectory_foreground=false.expand(motion_shape), source_pixel_valid=false.expand(b,q,h,w),
        static_mask=false.expand(b,q,h,w), scene_scale=scale)
    return target
