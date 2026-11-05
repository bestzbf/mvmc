"""Opt-in source anchoring, relative calibration and separated motion objectives.

All translations/depths stay in the existing common training scene scale.
Authoritative GT defines masks. Confidence and predicted visibility cannot
remove a residual. These are candidate objectives, not benchmark protocols.
"""
from dataclasses import dataclass
import math
import torch
from .metadata import gather_observations
from .geometry import unproject_depth

JOINT_GEOMETRY_METRICS = (
    'joint_source_log_depth', 'joint_source_xyz_epe', 'joint_source_pixels', 'joint_source_delta_px',
    'joint_relative_rotation_chordal', 'joint_rotation_pairs',
    'joint_relative_translation_epe', 'joint_translation_pairs',
    'joint_dynamic_displacement_epe', 'joint_dynamic_point_times',
    'joint_static_displacement_epe', 'joint_static_point_times',
    'joint_baseline_relative', 'joint_baseline_ratio', 'joint_baseline_pairs',
    'joint_depth_baseline_scale', 'joint_depth_scale_log_bias',
    'joint_baseline_scale_log_bias', 'joint_scale_pairs',
)


@dataclass(frozen=True)
class JointGeometryOptions:
    source_anchor_weight: float = 0.
    relative_pose_weight: float = 0.
    dynamic_displacement_weight: float = 0.
    static_displacement_weight: float = 0.
    diagnostics: bool = True
    baseline_relative_weight: float = 0.
    depth_baseline_scale_weight: float = 0.
    baseline_min_normalized: float = 1e-3

    def __post_init__(self):
        for k in ('source_anchor_weight','relative_pose_weight',
                  'dynamic_displacement_weight','static_displacement_weight',
                  'baseline_relative_weight','depth_baseline_scale_weight'):
            v=getattr(self,k)
            if isinstance(v,bool) or not math.isfinite(v) or v<0:
                raise ValueError(k+' must be finite and nonnegative')
        if not math.isfinite(self.baseline_min_normalized) or self.baseline_min_normalized <= 0:
            raise ValueError('baseline_min_normalized must be positive and finite')
        if not isinstance(self.diagnostics,bool):raise ValueError('diagnostics must be bool')


def _mean_maps(error, valid):
    """Each nonempty source/time map has equal weight, then each batch item."""
    count=valid.sum((-2,-1)); present=count>0
    value=torch.where(valid,error,0.).sum((-2,-1))/count.clamp_min(1)
    value=value.flatten(1);present=present.flatten(1)
    per_batch=(value*present).sum(1)/present.sum(1).clamp_min(1)
    return per_batch[present.any(1)].mean()


def _checked(error, valid):
    if not torch.isfinite(error[valid]).all():
        raise FloatingPointError('nonfinite joint geometry prediction on valid GT')
    return _mean_maps(error,valid)


def joint_geometry_losses(pred,target,options):
    tensor=next(x for x in pred.values() if isinstance(x,torch.Tensor) and x.is_floating_point())
    zero=tensor.new_zeros((),dtype=torch.float32);total=zero;metrics={}
    device=tensor.device
    with torch.autocast(device_type=device.type,enabled=False):
        if (options.source_anchor_weight or options.diagnostics) and all(k in target for k in ('depth','depth_valid','K')):
            required={'depth','K','source_obs_idx','observation_valid'}
            if required-set(pred):raise ValueError('source anchor requires source indices and predicted depth/K')
            source=pred['source_obs_idx']; S=pred['depth'].shape[1]
            sv=pred.get('source_valid',torch.ones_like(source,dtype=torch.bool)).bool()
            if (((source<0)|(source>=S))&sv).any():raise ValueError('invalid source index')
            source=source.clamp(0,S-1)
            pz=gather_observations(pred['depth'].float(),source)
            gz=gather_observations(target['depth'].to(device).float().detach(),source)
            if pz.shape!=gz.shape:raise ValueError('joint anchors require a common pixel grid')
            pk=gather_observations(pred['K'].float(),source)
            gk=gather_observations(target['K'].to(device).float().detach(),source)
            cam=pred['observation_valid'].bool()
            if 'camera_valid' in target:cam=cam&target['camera_valid'].to(device).bool()
            sv=sv&gather_observations(cam,source)
            sv=sv&torch.isfinite(gk).all((-1,-2))&(gk[...,[0,1],[0,1]]>0).all(-1)
            valid=gather_observations(target['depth_valid'].to(device).bool(),source)&sv[...,None,None]
            valid=valid&torch.isfinite(gz)&(gz>0)&torch.isfinite(gk).all((-1,-2))[...,None,None]
            if valid.any():
                if (pk[sv][...,[0,1],[0,1]]<=0).any() or not torch.isfinite(pk[sv]).all():
                    raise FloatingPointError('invalid predicted focal length')
                if (pz[valid]<=0).any():raise FloatingPointError('nonpositive source depth')
                # Sanitize invalid observations/pixels BEFORE nonlinear arithmetic.
                eye=torch.eye(3,device=device).expand_as(pk)
                pk=torch.where(sv[...,None,None],pk,eye)
                gk=torch.where(sv[...,None,None],gk,eye)
                pz=torch.where(valid,pz,1.);gz=torch.where(valid,gz,1.)
                identity=torch.eye(4,device=device).expand(*pz.shape[:2],4,4)
                pxyz=unproject_depth(pz,pk,identity);gxyz=unproject_depth(gz,gk,identity)
                epe=(pxyz-gxyz).norm(dim=-1)
                logz=_checked((pz.log()-gz.log()).abs(),valid)
                base=_checked(epe,valid)
                metrics.update(joint_source_log_depth=logz,joint_source_xyz_epe=base,
                    joint_source_pixels=valid.sum().float())
                total=total+options.source_anchor_weight*(logz+base)
                if options.diagnostics:
                    tolerance=gz/gk[...,[0,1],[0,1]].mean(-1)[...,None,None]
                    metrics['joint_source_delta_px']=torch.stack([
                        _mean_maps((epe<r*tolerance).float(),valid) for r in (1,2,4,8,16)]).mean()*100

        if (options.relative_pose_weight or options.diagnostics) and 'c2w' in target:
            needed={'c2w','camera_id','time_idx','observation_valid'}
            if needed-set(pred):raise ValueError('relative pose needs camera/time metadata')
            p=pred['c2w'].float();g=target['c2w'].to(device).float().detach()
            cam=pred['camera_id'];times=pred['time_idx'];valid=pred['observation_valid'].bool()
            valid=valid&torch.isfinite(g).all((-1,-2))
            if 'camera_valid' in target:valid=valid&target['camera_valid'].to(device).bool()
            pair=(valid[:,:,None]&valid[:,None,:]&(cam[:,:,None]!=cam[:,None,:])
                &(times[:,:,None]==times[:,None,:]))
            pair=pair&torch.ones_like(pair).triu(1)
            b,i,j=pair.nonzero(as_tuple=True)
            if len(b):
                pi,pj=p[b,i],p[b,j];gi,gj=g[b,i],g[b,j]
                if not torch.isfinite(torch.stack((pi,pj))).all():raise FloatingPointError('nonfinite relative pose')
                pr=pj[:,:3,:3].transpose(-1,-2)@pi[:,:3,:3]
                gr=gj[:,:3,:3].transpose(-1,-2)@gi[:,:3,:3]
                rot=(pr-gr).norm(dim=(-1,-2))/math.sqrt(2)
                def batch_mean(x,ids):return torch.stack([x[ids==v].mean() for v in ids.unique()]).mean()
                rotation=batch_mean(rot,b)
                metrics.update(joint_relative_rotation_chordal=rotation,joint_rotation_pairs=rot.new_tensor(len(b)))
                total=total+options.relative_pose_weight*rotation
                translation_valid=target.get('camera_translation_valid',valid).to(device).bool()
                known=translation_valid[b,i]&translation_valid[b,j]
                if known.any():
                    # Relative transform j <- i, translation in destination camera.
                    pt=(pj[:,:3,:3].transpose(-1,-2)@(pi[:,:3,3]-pj[:,:3,3])[...,None])[...,0]
                    gt=(gj[:,:3,:3].transpose(-1,-2)@(gi[:,:3,3]-gj[:,:3,3])[...,None])[...,0]
                    te=batch_mean((pt-gt).norm(dim=-1)[known],b[known])
                    metrics.update(joint_relative_translation_epe=te,joint_translation_pairs=known.sum().float())
                    total=total+options.relative_pose_weight*te

        if (options.dynamic_displacement_weight or options.static_displacement_weight or options.diagnostics) and 'displacement' in target:
            if 'displacement_valid' not in target:raise ValueError('motion objective requires authoritative displacement validity')
            p=pred['displacement'].float();g=target['displacement'].to(device).float().detach()
            valid=target['displacement_valid'].to(device).bool()&torch.isfinite(g).all(-1)
            valid=valid&pred['source_valid'][...,None,None,None]
            temporal=pred['source_time_idx'][:,:,None]!=pred['query_time_idx'][:,None]
            valid=valid&temporal[...,None,None]
            g=torch.where(valid[...,None],g,0.);p=torch.where(valid[...,None],p,0.)
            error=(p-g).norm(dim=-1)
            # Foreground flags alone are not evidence of motion: require nonzero GT displacement.
            dynamic=(valid&(g.norm(dim=-1)>1e-5))
            if 'trajectory_foreground' in target:dynamic=dynamic&target['trajectory_foreground'].to(device).bool()
            static=torch.zeros_like(valid)
            if 'static_mask' in target:
                sm=target['static_mask'].to(device).bool()
                if sm.ndim==4:sm=sm[:,:,None]
                static=valid&sm&(g.norm(dim=-1)<=1e-5)
            for name,mask,weight in (('dynamic',dynamic,options.dynamic_displacement_weight),
                                     ('static',static,options.static_displacement_weight)):
                if mask.any():
                    # Static targets retain their tiny GT displacement rather than replacing labels.
                    value=_checked(error,mask);total=total+weight*value
                    metrics['joint_'+name+'_displacement_epe']=value
                    metrics['joint_'+name+'_point_times']=mask.sum().float()
        if options.baseline_relative_weight or options.depth_baseline_scale_weight:
            scale_metrics, scale_loss = depth_baseline_losses(pred, target, options)
            metrics.update(scale_metrics)
            total = total + scale_loss
    return metrics,total


def depth_baseline_losses(pred, target, options):
    """GT-anchored scale calibration; no predicted-confidence/visibility masking.

    Same-time different-camera pairs only. Baselines use camera centers in the
    common gauge, avoiding a rotation gradient in the scale objective. A vector
    residual also recovers a collapsed predicted baseline. Dense log-depth
    means use every valid pixel in each observation, not sampled queries.
    No GT-metric translation (e.g. camera-only Panoptic) means no scale loss.
    """
    p = pred['c2w'].float()
    zero = p.new_zeros(())
    if 'c2w' not in target:
        return {}, zero
    g = target['c2w'].to(p.device).float().detach()
    valid = pred['observation_valid'].bool() & torch.isfinite(g).all((-1, -2))
    for key in ('camera_valid', 'camera_translation_valid'):
        if key in target:
            valid = valid & torch.broadcast_to(target[key].to(p.device).bool(), valid.shape)
    cam, time = pred['camera_id'], pred['time_idx']
    pair = (valid[:, :, None] & valid[:, None, :]
            & (cam[:, :, None] != cam[:, None, :])
            & (time[:, :, None] == time[:, None, :])).triu(1)
    b, i, j = pair.nonzero(as_tuple=True)
    if not len(b):
        return {}, zero
    gt = g[b, i, :3, 3] - g[b, j, :3, 3]
    length = gt.norm(dim=-1)
    keep = length >= options.baseline_min_normalized
    b, i, j, gt, length = b[keep], i[keep], j[keep], gt[keep], length[keep]
    if not len(b):
        return {}, zero
    pt = p[b, i, :3, 3] - p[b, j, :3, 3]
    if not torch.isfinite(pt).all():
        raise FloatingPointError('nonfinite predicted baseline on valid GT')
    def mean(x, ids=b):
        return torch.stack([x[ids == k].mean() for k in ids.unique()]).mean()
    def robust(x):
        return torch.nn.functional.smooth_l1_loss(x, torch.zeros_like(x), beta=.1, reduction='none')
    relative = mean(robust(((pt-gt)/length[:, None]).norm(dim=-1)))
    # Symmetric smoothing: exact GT yields zero, collapsed baselines stay finite.
    eps = 1e-6
    log_ratio = .5 * ((pt.square().sum(-1)+eps**2).log()
                      - (length.square()+eps**2).log())
    metrics = dict(joint_baseline_relative=relative,
                   joint_baseline_ratio=mean(pt.norm(dim=-1)/length),
                   joint_baseline_pairs=zero.new_tensor(len(b)))
    loss = options.baseline_relative_weight * relative
    if options.depth_baseline_scale_weight and all(k in target for k in ('depth','depth_valid')):
        pz, gz = pred['depth'].float(), target['depth'].to(p.device).float().detach()
        if pz.shape != gz.shape:
            raise ValueError('depth/baseline scale requires common dense pixel grid')
        mask = target['depth_valid'].to(p.device).bool() & valid[...,None,None]
        mask = mask & torch.isfinite(gz) & (gz > 0)
        if not torch.isfinite(pz[mask]).all() or (pz[mask] <= 0).any():
            raise FloatingPointError('invalid predicted depth on valid scale GT')
        safe_p = torch.where(mask,pz,1.)
        safe_g = torch.where(mask,gz,1.)
        counts = mask.sum((-1,-2))
        bias = (safe_p.log()-safe_g.log()).sum((-1,-2))/counts.clamp_min(1)
        both = (counts[b,i]>0) & (counts[b,j]>0)
        if both.any():
            a, c, r = bias[b,i][both], bias[b,j][both], log_ratio[both]
            # Anchors are essential: consistency alone admits both at half scale.
            value = .5*(robust(a)+robust(c)) + robust(r)
            value = value + .25*(robust(a-r)+robust(c-r))
            joint = mean(value,b[both])
            loss = loss + options.depth_baseline_scale_weight*joint
            metrics.update(joint_depth_baseline_scale=joint,
                joint_depth_scale_log_bias=mean((a+c)/2,b[both]),
                joint_baseline_scale_log_bias=mean(r,b[both]),
                joint_scale_pairs=zero.new_tensor(int(both.sum())))
    return metrics, loss
