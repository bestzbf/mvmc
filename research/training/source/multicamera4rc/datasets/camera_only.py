"""Calibration supervision for real multiview clips without surface geometry.

Human skeleton joints are not depth or material-point labels. Keep measured
intrinsics/rotations and explicitly mask translation, depth, rays and motion.
No scene scale is inferred from a camera baseline or a fitted human skeleton.
"""
import torch


def attach_camera_only_recipe(clip, sample):
    if clip.metadata.get('supervision_kind') != 'camera_only':
        raise ValueError('camera-only supervision needs an explicit adapter contract')
    target = {k:v for k,v in sample['targets'].items() if not k.startswith('sparse_')}
    if target['depth_valid'].any():
        raise ValueError('calibration-only adapter unexpectedly supplied surface depth')
    if not target['camera_valid'].any():
        raise ValueError('calibration-only clip has no measured camera')
    target['camera_translation_valid'] = torch.zeros_like(target['camera_valid'])
    target['dense_camera_only'] = torch.tensor(True)
    sample['targets'] = target


@torch.no_grad()
def materialize_camera_only_targets(batch, source_indices):
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
    target.update(ray=zero.expand(b,s,h,w,6), ray_valid=false.expand(b,s,h,w),
        trajectory=zero.expand(*motion_shape,3), trajectory_valid=false.expand(motion_shape),
        displacement=zero.expand(*motion_shape,3), displacement_valid=false.expand(motion_shape),
        trajectory_foreground=false.expand(motion_shape), source_pixel_valid=false.expand(b,q,h,w),
        static_mask=false.expand(b,q,h,w), scene_scale=zero.new_ones(b))
    return target
