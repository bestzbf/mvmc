"""Known integer ROI and clip-shared RGB appearance augmentation.

All times of a physical camera share one ROI. No world-space value, camera
extrinsic or displacement vector is scaled. Dense renderer/native recipes
are cropped exactly, then targets are rebuilt with the translated intrinsics.
"""
import math
from collections.abc import Mapping

import numpy as np
import torch
from torch.nn import functional as F


DEFAULT_PHOTOMETRIC = dict(probability=.8, brightness=.2, contrast=.2,
    saturation=.2, grayscale_probability=.05, blur_probability=.1,
    blur_sigma=[.1, 1.], blur_kernel_size=5)


def validate_photometric_augmentation(config):
    """Normalize a serializable, clip-shared RGB-only augmentation contract."""
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise ValueError('photometric augmentation must be a mapping')
    if set(config)-set(DEFAULT_PHOTOMETRIC):
        raise ValueError('unknown photometric augmentation options')
    result = dict(DEFAULT_PHOTOMETRIC, **config)
    for name in ('probability', 'grayscale_probability', 'blur_probability',
                 'brightness', 'contrast', 'saturation'):
        value = result[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f'photometric {name} must be finite and between zero and one')
        result[name] = float(value)
    sigma = result['blur_sigma']
    if (not isinstance(sigma, (list, tuple)) or len(sigma) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or v <= 0 for v in sigma)
            or sigma[0] > sigma[1]):
        raise ValueError('blur_sigma must be two ordered positive finite values')
    result['blur_sigma'] = list(map(float, sigma))
    kernel = result['blur_kernel_size']
    if isinstance(kernel, bool) or not isinstance(kernel, int) or not 1 <= kernel <= 31 or kernel % 2 != 1:
        raise ValueError('blur_kernel_size must be an odd integer from 1 to 31')
    return result


def photometric_sample(sample, config, seed, content_hw=None):
    """Alter RGB only, sharing every random parameter across cameras and time.

    Sampling uses a separate namespace of the deterministic dataset sample
    seed; worker scheduling and Torch's global RNG cannot change the result.
    Native patch padding and invalid observation slots remain bitwise intact.
    No target, camera parameter, pixel coordinate or ROI metadata is rewritten.
    """
    config = validate_photometric_augmentation(config)
    if config is None:
        return sample
    images = sample['images']
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError('photometric augmentation expects [observations,3,H,W] RGB')
    if images.dtype != torch.uint8 and not images.is_floating_point():
        raise ValueError('photometric RGB must be floating point [0,1] or uint8')
    if not torch.isfinite(images).all() or (images < 0).any() or (images > (255 if images.dtype == torch.uint8 else 1)).any():
        raise ValueError('photometric RGB is outside its finite input range')
    observations, _, h, w = images.shape
    if content_hw is None:
        content_hw = [[h, w]]*observations
    content = torch.as_tensor(content_hw, device=images.device)
    if (content.shape != (observations, 2) or not torch.isfinite(content).all()
            or not (content == content.long()).all() or (content <= 0).any()
            or (content > content.new_tensor([h, w])).any()):
        raise ValueError('photometric content sizes must be valid integer RGB bounds')
    valid = sample.get('observation_valid', torch.ones(observations, dtype=torch.bool, device=images.device))
    if valid.shape != (observations,):
        raise ValueError('photometric observation mask has the wrong shape')
    valid = valid.to(device=images.device, dtype=torch.bool)
    rng = np.random.default_rng((int(seed) ^ 0x50484F54) & ((1 << 64)-1))
    info = dict(kind='clip_shared_photometric_v1', applied=False)
    if rng.random() >= config['probability'] or not valid.any():
        return dict(sample, photometric_augmentation=info)
    factors = {name: float(rng.uniform(1-config[name], 1+config[name]))
               for name in ('brightness', 'contrast', 'saturation')}
    grayscale = bool(rng.random() < config['grayscale_probability'])
    blur = bool(rng.random() < config['blur_probability'])
    sigma = float(rng.uniform(*config['blur_sigma'])) if blur else None
    with torch.no_grad():
        rgb = images.float() / (255 if images.dtype == torch.uint8 else 1)
        yy = torch.arange(h, device=images.device)[None, :, None]
        xx = torch.arange(w, device=images.device)[None, None, :]
        mask = ((yy < content[:, 0, None, None]) & (xx < content[:, 1, None, None])
                & valid[:, None, None])[:, None]
        coefficients = rgb.new_tensor([.2989, .5870, .1140])[None, :, None, None]
        if factors['brightness'] != 1.:
            rgb = (rgb*factors['brightness']).clamp(0, 1)
        luminance = (rgb*coefficients).sum(1, keepdim=True)
        # A single contrast anchor preserves relative clip illumination, rather
        # than normalizing each time or physical camera independently.
        anchor = (luminance*mask).sum()/mask.sum().clamp_min(1)
        if factors['contrast'] != 1.:
            rgb = (anchor+(rgb-anchor)*factors['contrast']).clamp(0, 1)
        luminance = (rgb*coefficients).sum(1, keepdim=True)
        if factors['saturation'] != 1.:
            rgb = (luminance+(rgb-luminance)*factors['saturation']).clamp(0, 1)
        if grayscale:
            rgb = (rgb*coefficients).sum(1, keepdim=True).expand_as(rgb).clone()
        if blur and config['blur_kernel_size'] > 1:
            kernel_size = config['blur_kernel_size']
            radius = kernel_size//2
            axis = torch.arange(-radius, radius+1, device=rgb.device, dtype=rgb.dtype)
            kernel = torch.exp(-.5*(axis/sigma).square())
            kernel /= kernel.sum()
            horizontal = kernel[None, None, None, :].expand(3, 1, 1, -1)
            vertical = kernel[None, None, :, None].expand(3, 1, -1, 1)
            blurred = rgb.clone()
            # Blur within actual RGB bounds so native black patch padding does
            # not leak into content. Equal native canvases share a batched op.
            sizes = {tuple(map(int, row)) for row in content[valid].tolist()}
            for ch, cw in sizes:
                slots = valid & (content[:, 0] == ch) & (content[:, 1] == cw)
                field = rgb[slots, :, :ch, :cw]
                mode = 'reflect' if min(ch, cw) > radius else 'replicate'
                field = F.conv2d(F.pad(field, (radius, radius, 0, 0), mode=mode), horizontal, groups=3)
                field = F.conv2d(F.pad(field, (0, 0, radius, radius), mode=mode), vertical, groups=3)
                indices = torch.nonzero(slots, as_tuple=False).flatten()
                blurred[indices, :, :ch, :cw] = field
            rgb = blurred
        if images.dtype == torch.uint8:
            rgb = (rgb.clamp(0, 1)*255).round().to(images.dtype)
        else:
            rgb = rgb.clamp(0, 1).to(images.dtype)
        result = torch.where(mask, rgb, images)
    info.update(applied=True, **factors, grayscale=grayscale, blur=blur,
                blur_sigma=sigma, blur_kernel_size=config['blur_kernel_size'])
    return dict(sample, images=result, photometric_augmentation=info)


def principal_canvas_for_sample(sample):
    """Known crop metadata [S,4]; independent of targets and camera calibration."""
    observations, _, h, w = sample['images'].shape
    identity = torch.tensor([h, w, 0, 0], dtype=torch.float32).expand(observations, 4).clone()
    info = sample.get('augmentation') or {}
    if not info.get('applied', False):
        return identity
    if info.get('kind') != 'integer_camera_roi_v1' or info.get('output_hw') != [h, w]:
        raise ValueError('principal metadata requires an exact integer ROI and matching output canvas')
    boxes = torch.as_tensor(info['boxes'])
    original = torch.as_tensor(info['source_hw'])
    if (boxes.shape != (observations, 4) or original.shape != (2,)
            or not torch.isfinite(boxes).all() or not torch.isfinite(original).all()
            or not (boxes == boxes.long()).all() or not (original == original.long()).all()
            or (original <= 0).any() or (boxes[:, :2] < 0).any() or (boxes[:, 2:] <= 0).any()
            or (boxes[:, :2]+boxes[:, 2:] > original).any()
            or not (((boxes[:, 2:]+13)//14)*14 == boxes.new_tensor([h, w])).all()):
        raise ValueError('inconsistent principal crop metadata')
    for camera in sample['camera_id'].unique():
        rows = boxes[sample['camera_id'] == camera]
        if not (rows == rows[0]).all():
            raise ValueError('a physical camera must share its ROI over time')
    identity[:, :2] = original
    identity[:, 2:] = boxes[:, :2]
    return identity


def validate_augmentation(config):
    if config is None:
        return None
    config = dict(config)
    if set(config)-{'probability','min_crop_fraction','max_crop_fraction'}:
        raise ValueError('unknown camera crop augmentation options')
    defaults = dict(probability=.35,min_crop_fraction=.75,max_crop_fraction=1.)
    defaults.update(config)
    if (any(isinstance(x,bool) or not math.isfinite(x) for x in defaults.values())
            or not 0 <= defaults['probability'] <= 1
            or not .5 <= defaults['min_crop_fraction'] <= defaults['max_crop_fraction'] <= 1):
        raise ValueError('invalid ROI augmentation probability/fractions')
    return defaults


def crop_sample(sample, boxes):
    """Apply [top,left,height,width] per observation, exact integer sampling.

    Boxes share size; cameras may have different origins, fixed over time.
    This public deterministic operation also supports GT/projection audits.
    """
    s,_,height,width = sample['images'].shape
    boxes = torch.as_tensor(boxes)
    if not torch.isfinite(boxes).all() or not (boxes == boxes.long()).all():
        raise ValueError('ROI coordinates must be finite integers')
    boxes = boxes.long()
    if boxes.shape != (s,4) or len(set(map(tuple,boxes[:,2:].tolist()))) != 1:
        raise ValueError('one same-size integer ROI is required per observation')
    h,w = boxes[0,2:].tolist()
    if (min(h,w)<28 or (boxes[:,:2]<0).any()
            or (boxes[:,0]+h>height).any() or (boxes[:,1]+w>width).any()):
        raise ValueError('ROI outside source canvas')
    for camera in sample['camera_id'].unique():
        rows=boxes[sample['camera_id']==camera]
        if not (rows==rows[0]).all():
            raise ValueError('a physical camera must share its ROI over time')
    hp,wp=((v+13)//14*14 for v in (h,w))

    def crop(value, *, vector=False, fill=0):
        if value.shape[0]!=s:
            raise ValueError('spatial recipe must have observation as its first axis')
        if tuple(value.shape[-3:-1] if vector else value.shape[-2:])!=(height,width):
            raise ValueError('spatial recipe resolution disagrees with RGB')
        rows=[]
        for i,(top,left,_,_) in enumerate(boxes.tolist()):
            field=value[i].movedim(-1,-3) if vector else value[i]
            field=F.pad(field[...,top:top+h,left:left+w],(0,wp-w,0,hp-h),value=fill)
            rows.append(field.movedim(-3,-1) if vector else field)
        return torch.stack(rows)

    out=dict(sample,images=crop(sample['images']))
    original=sample['targets'];target=dict(original)
    fields={'depth':0,'depth_valid':False,'dense_segmentation':-1,
            'dense_pair_valid':False,'dense_syn4d_face_map':-1,'dense_syn4d_environment':False}
    for name,fill in fields.items():
        if name in target:target[name]=crop(target[name],fill=fill)
    if 'dense_pair_delta' in target:
        target['dense_pair_delta']=crop(target['dense_pair_delta'],vector=True)
    target['K']=original['K'].clone()
    target['K'][:,0,2]-=boxes[:,1].to(target['K'])
    target['K'][:,1,2]-=boxes[:,0].to(target['K'])
    if 'dense_syn4d_content_hw' in target:
        target['dense_syn4d_content_hw']=torch.tensor([[h,w]]*s,dtype=torch.long)
        target['dense_motion_source_valid']=(target['dense_syn4d_face_map']>=0).flatten(1).any(1)

    has_kinoptic_queries = False
    kinoptic_source_valid = torch.zeros(s, dtype=torch.bool)
    for prefix in ('joint_motion','pseudo_motion'):
        if prefix+'_queries' not in target:continue
        has_kinoptic_queries = True
        queries=target[prefix+'_queries'].clone();source=queries[:,0].long()
        queries[:,1]-=boxes[source,1].to(queries)
        queries[:,2]-=boxes[source,0].to(queries)
        retained=(queries[:,1]>=0)&(queries[:,1]<=w-1)&(queries[:,2]>=0)&(queries[:,2]<=h-1)
        target[prefix+'_queries']=queries
        target[prefix+'_query_valid']=target[prefix+'_query_valid']&retained
        target[prefix+'_valid']=target[prefix+'_valid']&retained[:,None]
        active = target[prefix+'_query_valid']
        if active.any():
            kinoptic_source_valid[source[active]] = True
    if has_kinoptic_queries and 'dense_motion_source_valid' in target:
        target['dense_motion_source_valid'] = kinoptic_source_valid

    if 'sparse_queries' in target:
        queries=target['sparse_queries'].clone()
        source=queries[:,0].long()
        queries[:,1]-=boxes[source,1].to(queries)
        queries[:,2]-=boxes[source,0].to(queries)
        retained=(queries[:,1]>=0)&(queries[:,1]<=w-1)&(queries[:,2]>=0)&(queries[:,2]<=h-1)
        target['sparse_queries']=queries
        target['sparse_query_valid']=target['sparse_query_valid']&retained
        # World positions and vector values are not rewritten. Only valid
        # source queries and target-view visibility change under a crop.
        for name in ['sparse_trajectory_valid','sparse_displacement_valid']:
            if name in target:target[name]=target[name]&retained[:,None]
        if 'sparse_visibility' in target:
            truth=target['sparse_trajectory']
            geometry_valid=target['sparse_trajectory_valid'] & torch.isfinite(truth).all(-1)
            visibility=target['sparse_visibility'].clone()
            known=target['sparse_visibility_valid'].clone() & retained[:,None,None]
            cameras=sample['camera_id'].unique(sorted=True)
            for ti,time in enumerate(sample['query_time_idx']):
                for ci,camera in enumerate(cameras):
                    slots=torch.nonzero((sample['time_idx']==time)&(sample['camera_id']==camera)).flatten()
                    if not len(slots):continue
                    slot=int(slots[0]);c=target['c2w'][slot]
                    local=(truth[:,ti]-c[:3,3])@c[:3,:3]
                    projected=local@target['K'][slot].T
                    xy=projected[:,:2]/projected[:,2:].clamp_min(1e-8)
                    inside=(local[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<=w-1)&(xy[:,1]>=0)&(xy[:,1]<=h-1)
                    old=visibility[:,ti,ci].bool()
                    known[:,ti,ci] &= (~old | geometry_valid[:,ti])
                    visibility[:,ti,ci] = old & geometry_valid[:,ti] & inside
            target.update(sparse_visibility=visibility,sparse_visibility_valid=known)
    out['targets']=target
    out['augmentation']=dict(kind='integer_camera_roi_v1',applied=True,boxes=boxes.tolist(),
                             source_hw=[height,width],content_hw=[h,w],output_hw=[hp,wp])
    return out


def augment_sample(sample, config, seed, content_hw):
    config=validate_augmentation(config)
    if config is None:return sample
    rng=np.random.default_rng(int(seed)^0x4C410C7)
    if rng.random()>=config['probability']:
        return dict(sample,augmentation=dict(kind='integer_camera_roi_v1',applied=False))
    content=np.asarray(content_hw,dtype=np.int64)
    if content.shape!=(len(sample['images']),2):
        raise ValueError('real RGB content sizes required, excluding padding')
    if (content<28).any():
        return dict(sample,augmentation=dict(kind='integer_camera_roi_v1',applied=False,
                                           reason='rgb_smaller_than_minimum_roi'))
    fraction=rng.uniform(config['min_crop_fraction'],config['max_crop_fraction'])
    h,w=np.maximum(28,np.floor(content.min(0)*fraction).astype(int))
    boxes=np.zeros((len(content),4),dtype=np.int64)
    for camera in sample['camera_id'].unique(sorted=True):
        ids=(sample['camera_id']==camera).numpy()
        ch,cw=content[ids].min(0)
        if h>ch or w>cw:raise ValueError('RGB too small for calibrated crop')
        top,left=int(rng.integers(ch-h+1)),int(rng.integers(cw-w+1))
        boxes[ids]=[top,left,h,w]
    result=crop_sample(sample,boxes)
    if not result['targets']['depth_valid'].any() and 'dense_camera_only' not in result['targets']:
        return dict(sample,augmentation=dict(kind='integer_camera_roi_v1',applied=False,
                                           reason='no_valid_depth_in_roi'))
    return result
