"""Keep Kinoptic depth GT, joint displacement, and surface pseudo labels distinct."""
import numpy as np
import torch

from .partial_dense import attach_partial_dense_recipe


def _project_joints(points, K, E):
    camera = points@E[:3, :3].T+E[:3, 3]
    projected = camera@K.T
    uv = projected[:, :2]/np.maximum(projected[:, 2:], 1e-9)
    return uv, camera[:, 2]


def _depth_supported_queries(uv, z, depth, support, *, radius=3):
    """Conservative source eligibility only; this is not visibility annotation.

    Joint centres may sit behind their own surface. Require measured surface
    samples within 3 pixels, no more than 20 cm in front or 5 cm behind the
    joint. Labels never become absolute surface depth or occlusion targets.
    """
    h, w = depth.shape;valid = np.zeros(len(uv), bool)
    for j, (pixel, joint_z) in enumerate(zip(uv, z)):
        if not np.isfinite(pixel).all() or not np.isfinite(joint_z) or joint_z <= .1: continue
        if not (radius <= pixel[0] < w-radius and radius <= pixel[1] < h-radius): continue
        x, y = np.floor(pixel+.5).astype(int)
        patch = depth[y-radius:y+radius+1, x-radius:x+radius+1]
        mask = support[y-radius:y+radius+1, x-radius:x+radius+1] & (patch > 0) & np.isfinite(patch)
        good = patch[mask]
        if len(good) < 3: continue
        median = float(np.median(good))
        # Reject discontinuities instead of accepting a nearby background/body mix.
        valid[j] = (float(np.quantile(good, .9)-np.quantile(good, .1)) <= .08
                    and -.05 <= joint_z-median <= .20)
    return valid


def attach_kinoptic_human_recipe(clip, sample, *, image_mode, max_pseudo_queries=512):
    if clip.metadata.get('supervision_kind') != 'kinoptic_human' or clip.metadata.get('pseudo_is_true_gt') is not False:
        raise ValueError('Kinoptic human recipe requires explicit separated provenance')
    if len(clip.images) != 4 or len(sample['images']) != 4 or len(sample['query_time_idx']) != 2:
        raise ValueError('Kinoptic human recipe requires its complete C2T2 clip')
    evidence = clip.metadata['kinoptic_human_evidence']
    start, end = np.asarray(evidence['joint_start_world_m']), np.asarray(evidence['joint_end_world_m'])
    confidence = np.asarray(evidence['joint_confidence'])
    if start.ndim != 2 or start.shape[1:] != (3,) or start.shape != end.shape or confidence.shape != start.shape[:1]:
        raise ValueError('invalid official joint target shapes')
    if not all(np.isfinite(v).all() for v in [start, end, confidence]) or (confidence < 0).any():
        raise ValueError('invalid official joint target values')
    positions = [start, end];uvs = [];eligible = []
    for obs in range(4):
        uv, z = _project_joints(positions[obs % 2], clip.intrinsics[obs], clip.extrinsics[obs])
        ok = _depth_supported_queries(uv, z, clip.depths[obs], clip.metadata['pixel_support'][obs])
        uvs.append(uv);eligible.append(ok)
    eligible = np.asarray(eligible)
    # Both timestamps must have supporting surfaces in both input cameras.
    common = eligible.all(axis=0) & (confidence >= .3)
    attach_partial_dense_recipe(clip, sample, kind='geometry', image_mode=image_mode)
    target = sample['targets'];reference_R = np.asarray(clip.extrinsics[0])[:3, :3]
    affines = [target['K'][i].numpy()@np.linalg.inv(clip.intrinsics[i]) for i in range(4)]
    joint_queries=[];joint_delta=[];joint_valid=[];joint_conf=[]
    for obs in range(4):
        for j in np.flatnonzero(common):
            pixel = affines[obs]@np.r_[uvs[obs][j], 1.]
            joint_queries.append([obs, pixel[0], pixel[1]])
            direction = end[j]-start[j] if obs % 2 == 0 else start[j]-end[j]
            delta = np.zeros((2, 3), np.float32);valid = np.zeros(2, bool)
            delta[1-obs % 2] = direction@reference_R.T;valid[1-obs % 2] = True
            joint_delta.append(delta);joint_valid.append(valid);joint_conf.append(valid*np.clip(confidence[j], 0, 1))
    def attach(prefix, queries, delta, valid, conf, uncertainty=None):
        n=len(queries)
        target[prefix+'_queries']=torch.tensor(np.asarray(queries, np.float32).reshape(n, 3))
        target[prefix+'_query_valid']=torch.tensor(np.asarray(valid, bool).reshape(n, 2).any(axis=-1))
        target[prefix+'_displacement']=torch.tensor(np.asarray(delta, np.float32).reshape(n, 2, 3))
        target[prefix+'_valid']=torch.tensor(np.asarray(valid, bool).reshape(n, 2))
        target[prefix+'_confidence']=torch.tensor(np.asarray(conf, np.float32).reshape(n, 2))
        if uncertainty is not None:
            target[prefix+'_uncertainty']=torch.tensor(np.asarray(uncertainty, np.float32).reshape(n, 2))
    attach('joint_motion', joint_queries, joint_delta, joint_valid, joint_conf)
    pseudo_uv=np.asarray(evidence['pseudo_source_uv']);pseudo_delta=np.asarray(evidence['pseudo_delta_world_m'])
    radius=np.asarray(evidence['pseudo_uncertainty_m']);cycle=np.asarray(evidence['pseudo_cycle_error_px'])
    cross=np.asarray(evidence['pseudo_cross_view_error_px'])
    n=len(pseudo_uv)
    if pseudo_uv.shape != (n, 2) or pseudo_delta.shape != (n, 3) or any(x.shape != (n,) for x in (radius, cycle, cross)):
        raise ValueError('invalid surface pseudo target shapes')
    if not all(np.isfinite(v).all() for v in (pseudo_uv, pseudo_delta, radius, cycle, cross)) or (radius <= 0).any():
        raise ValueError('invalid surface pseudo target values or uncertainty')
    select=np.arange(n)
    if n>max_pseudo_queries:select=np.linspace(0, n-1, max_pseudo_queries, dtype=int)
    uv=np.c_[pseudo_uv[select],np.ones(len(select))]@affines[0].T
    query=np.c_[np.zeros(len(select)),uv[:, :2]]
    delta=np.zeros((len(select),2,3),np.float32);delta[:,1]=pseudo_delta[select]@reference_R.T
    valid=np.zeros((len(select),2),bool);valid[:,1]=True
    conf=np.zeros_like(valid,dtype=np.float32);conf[:,1]=np.clip((1-cycle[select]/2)*(1-cross[select]/3),.1,1.)
    uncertainty=np.zeros_like(conf);uncertainty[:,1]=radius[select]
    attach('pseudo_motion',query,delta,valid,conf,uncertainty)
    h,w=sample['images'].shape[-2:]
    source_valid=torch.zeros(4,dtype=torch.bool)
    for prefix in ('joint_motion','pseudo_motion'):
        q=target[prefix+'_queries'];inside=(q[:,1]>=0)&(q[:,1]<=w-1)&(q[:,2]>=0)&(q[:,2]<=h-1)
        target[prefix+'_query_valid'] &= inside;target[prefix+'_valid'] &= inside[:,None]
        source_valid[q[inside,0].long()]=True
    target['dense_motion_source_valid']=source_valid
    sample['kinoptic_human']=dict(capture=clip.metadata['kinoptic_capture'],
        pack_sha256=clip.metadata['kinoptic_pack_sha256'],manifest_sha256=clip.metadata['kinoptic_manifest_sha256'],
        official_joint_queries=len(joint_queries),surface_pseudo_queries=len(select),
        joint_source_evidence='two_view_measured_surface_proximity_not_visibility_gt',
        pseudo_is_true_gt=False)
