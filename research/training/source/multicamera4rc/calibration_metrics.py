"""Independent camera calibration measurements; no confidence weighting."""
import torch

INTRINSICS_METRICS = (
    'intrinsics_fx_ratio', 'intrinsics_fy_ratio', 'intrinsics_focal_abs_rel',
    'intrinsics_log_focal_error', 'intrinsics_fov_error_degrees',
    'intrinsics_principal_error_px', 'intrinsics_principal_error_normalized',
    'intrinsics_focal_below_90pct', 'intrinsics_focal_above_110pct',
    'intrinsics_narrow_focal_abs_rel', 'intrinsics_normal_focal_abs_rel',
    'intrinsics_wide_focal_abs_rel',
)


@torch.no_grad()
def intrinsics_metrics(prediction, target):
    if 'K' not in prediction or 'K' not in target:
        return {}
    p, t = prediction['K'].float(), target['K'].to(prediction['K'].device).float()
    sh, sw = prediction['depth'].shape[-2:]
    h, w = target.get('depth', prediction['depth']).shape[-2:]
    p = p*p.new_tensor([w/sw,h/sh,1])[None,None,:,None]
    p[..., :2, 2] += p.new_tensor([(w/sw-1)/2,(h/sh-1)/2])
    valid = torch.isfinite(t).all((-1,-2)) & (t[...,0,0]>0) & (t[...,1,1]>0)
    for data,key in ((target,'camera_valid'),(prediction,'observation_valid')):
        if key in data:
            valid &= data[key].to(valid.device).bool()
    if not valid.any():
        return {}
    p,t = p[valid],t[valid]
    pf = torch.stack((p[:,0,0],p[:,1,1]),-1)
    tf = torch.stack((t[:,0,0],t[:,1,1]),-1)
    if not torch.isfinite(p).all() or (pf<=0).any():
        raise FloatingPointError('invalid supervised camera prediction')
    ratio=pf/tf
    half=pf.new_tensor([w/2,h/2])
    pfov,tfov=2*torch.atan(half/pf),2*torch.atan(half/tf)
    result={
        'intrinsics_fx_ratio':ratio[:,0].mean(), 'intrinsics_fy_ratio':ratio[:,1].mean(),
        'intrinsics_focal_abs_rel':(ratio-1).abs().mean(),
        'intrinsics_log_focal_error':ratio.log().abs().mean(),
        'intrinsics_fov_error_degrees':torch.rad2deg((pfov-tfov).abs()).mean(),
        'intrinsics_principal_error_px':torch.linalg.vector_norm(p[:,:2,2]-t[:,:2,2],dim=-1).mean(),
        'intrinsics_principal_error_normalized':torch.linalg.vector_norm((p[:,:2,2]-t[:,:2,2])/pf.new_tensor([w,h]),dim=-1).mean(),
        'intrinsics_focal_below_90pct':(ratio<.9).float().mean(),
        'intrinsics_focal_above_110pct':(ratio>1.1).float().mean(),
    }
    degrees=torch.rad2deg(tfov[:,0])
    for name,mask in [('narrow',degrees<45),('normal',(degrees>=45)&(degrees<80)),('wide',degrees>=80)]:
        if mask.any():
            result[f'intrinsics_{name}_focal_abs_rel']=(ratio[mask]-1).abs().mean()
    return result
