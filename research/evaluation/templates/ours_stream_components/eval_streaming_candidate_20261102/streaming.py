"""Private TAPVid probe: full timeline refinement, streamed independent heads.

Batch-dependent kernels may differ numerically. This is NOT admitted for scoring.
"""
from contextlib import contextmanager
from types import MethodType
import torch
from torch.nn import functional as F
from multicamera4rc.refinement import gather_observations, convex_upsample


class MotionFeatures:
    def __init__(self, coarse, selected, indices, shape):
        self.coarse, self.selected, self.indices, self.shape = coarse, selected, indices, shape

    def __getitem__(self, index):
        b,u=index
        _,_,channels,h,w=self.shape
        # Native visibility is pointwise in source pixels. Preserve all queried
        # bilinear neighbours, and keep its original full-image GEMM shapes.
        value=self.selected.new_zeros((channels,h*w))
        value.index_copy_(1,self.indices,self.selected[b,u])
        return value.reshape(channels,h,w)


@contextmanager
def streamed(model, inputs, time_chunk=8):
    if model.training or model.config.decoder_memory!='target_time':
        raise ValueError('This trial requires the native target_time inference configuration')
    if getattr(model,'_stream_active',False):raise RuntimeError('Nested streamed context')
    frames=len(inputs.frame_indices)
    height,width=inputs.video.shape[-2:]
    queries=inputs.queries_xytv.cpu()
    source_ids=queries[:,3].long()*frames+queries[:,2].long()
    support={int(src):queries[source_ids==src,:2].clone() for src in source_ids.unique()}
    state=dict(sources=0,decoder_chunks=0,head_chunks=0,time_chunk=time_chunk,
               full_timeline_refiner=True,cpu_feature_offload=False)
    saved=[]
    def replace(obj,name,fn):
        saved.append((obj,name,name in obj.__dict__,getattr(obj,name)))
        setattr(obj,name,MethodType(fn,obj))
    original_decode=model.decode
    original_decoder=model.motion_decoder.forward
    original_head=model.track_head.forward
    original_refiner=model.trajectory_refiner.forward
    old_identity=getattr(model,'_stream_identity_memory',False)
    active={}

    def decode(this,scene,query_time_idx=None,source_obs_idx=None,return_visibility=True):
        ids=torch.as_tensor(source_obs_idx).reshape(-1)
        if len(ids)!=1:raise ValueError('Streamed export requires one source observation per call')
        active['xy']=support[int(ids.item())]
        state['sources']+=1
        return original_decode(scene,query_time_idx,source_obs_idx,return_visibility)

    def decoder(this,source_patches,memory,cond,source_camera,memory_camera,memory_valid,
                patch_grid,prepared_memory=None):
        if prepared_memory is not None:raise ValueError('Do not mix experimental streaming and prepared KV')
        if memory.ndim!=4:raise ValueError('Expected per-target multi-view memory')
        result=None
        for start in range(0,cond.shape[1],time_chunk):
            stop=min(start+time_chunk,cond.shape[1])
            part=original_decoder(source_patches,memory[:,start:stop],cond[:,start:stop],source_camera,
                memory_camera[:,start:stop],memory_valid[:,start:stop],patch_grid,None)
            if result is None:result=part.new_empty((len(part),cond.shape[1],*part.shape[2:]))
            result[:,start:stop].copy_(part)
            state['decoder_chunks']+=1
            del part
        return result

    def head(this,decoded_taps,output_hw):
        batch,targets=decoded_taps[0].shape[:2]
        if batch!=1:raise ValueError('This trial is scoped to one clip')
        h,w=output_hw;hp,wp=h//this.patch_size,w//this.patch_size
        xy=active['xy'].floor().long()
        shifts=torch.arange(-2,3)
        yy=(xy[:,1,None,None]+shifts[None,:,None]).expand(-1,5,5).clamp(0,height-1)
        xx=(xy[:,0,None,None]+shifts[None,None,:]).expand(-1,5,5).clamp(0,width-1)
        indices=(yy*w+xx).reshape(-1).unique(sorted=True).to(decoded_taps[0].device)
        outputs={};coarse=selected=None
        for start in range(0,targets,time_chunk):
            stop=min(start+time_chunk,targets)
            out=original_head([tap[:,start:stop] for tap in decoded_taps],output_hw)
            feature=out.pop('features');channels=feature.shape[2]
            if coarse is None:
                coarse=feature.new_empty((batch,targets,channels,hp,wp))
                selected=feature.new_empty((batch,targets,channels,len(indices)))
            coarse[:,start:stop].copy_(F.interpolate(feature.reshape(-1,channels,h,w),
                (hp,wp),mode='bilinear',align_corners=False).reshape(batch,stop-start,channels,hp,wp))
            selected[:,start:stop].copy_(feature.flatten(-2).index_select(-1,indices))
            for key,value in out.items():
                if key not in outputs:outputs[key]=value.new_empty((batch,targets,*value.shape[2:]))
                outputs[key][:,start:stop].copy_(value)
            state['head_chunks']+=1
            del feature,out,value
        decoded_taps.clear()
        outputs['features']=MotionFeatures(coarse,selected,indices,(batch,targets,channels,h,w))
        return outputs

    def refiner(this,*args,**kwargs):
        if not isinstance(args[1],MotionFeatures):return original_refiner(*args,**kwargs)
        return refine_streamed(this,*args,**kwargs)

    def prepare_targets(this,target_features,patch_grid):
        hp,wp=patch_grid;b,u,c=target_features.shape[:3]
        return torch.stack([this.target_projection(target_features[bi,ti,ci].contiguous())
            .transpose(0,1).reshape(128,hp,wp)
            for bi in range(b) for ti in range(u) for ci in range(c)]).reshape(b,u,c,128,hp,wp)

    try:
        model._stream_active=True;model._stream_identity_memory=True
        replace(model,'decode',decode)
        replace(model.motion_decoder,'forward',decoder)
        replace(model.track_head,'forward',head)
        replace(model.trajectory_refiner,'forward',refiner)
        replace(model.visibility_head,'prepare_targets',prepare_targets)
        yield state
    finally:
        for obj,name,owned,value in reversed(saved):
            if owned:setattr(obj,name,value)
            else:delattr(obj,name)
        model._stream_active=False;model._stream_identity_memory=old_identity


def refine_streamed(self, trajectory, motion_features, source_features, target_features,
            target_depth, target_conf, target_K, target_c2w, target_valid,
            same_camera, times, time_valid, anchor_slot, scale, target_pyramid=None):
    b, u, H, W, _ = trajectory.shape
    h, w = source_features.shape[-2:]
    n, d = h * w, self.config.refinement_dim
    coarse = F.interpolate(trajectory.permute(0, 1, 4, 2, 3).reshape(b * u, 3, H, W),
                           (h, w), mode="bilinear", align_corners=False)
    initial = coarse.reshape(b, u, 3, n).permute(0, 1, 3, 2)
    anchor = gather_observations(initial, anchor_slot)
    feature = motion_features.coarse.reshape(b * u, -1, h, w)
    feature = self.motion_projection(feature).reshape(b, u, d, n).permute(0, 1, 3, 2)
    source = source_features.flatten(-2).transpose(-1, -2)
    source_time = gather_observations(times, anchor_slot)
    time = self.time_projection(self.time_embedding(times - source_time[:, None]).float())
    hidden = feature + source[:, None] + time[:, :, None]
    correction = torch.zeros_like(initial)
    support = trajectory.new_zeros(b)
    for _ in range(self.config.refinement_iterations):
        current = initial + correction
        xyz = torch.cat((current.detach() - anchor.detach()[:, None],
                         anchor.detach()[:, None].expand_as(current)), -1) / scale[:, None, None, None]
        state = xyz.sign() * torch.log1p(xyz.abs())
        evidence = torch.zeros_like(hidden)
        if self.config.refinement_matching:
            evidence, support = self.matching_evidence(current, source_features, target_features,
                target_depth, target_conf, target_K, target_c2w, target_valid,
                same_camera, scale, (H, W), target_pyramid)
        hidden = hidden + evidence + self.state_encoder(state.clamp(-10, 10))
        if self.config.refinement_temporal:
            hidden = self.temporal(hidden, time_valid)
        hidden = (hidden + self.update(hidden)) * time_valid[:, :, None, None]
        step = self.delta_head(hidden).tanh() * scale[:, None, None, None] * self.config.refinement_max_update
        step = step - gather_observations(step, anchor_slot)[:, None]
        correction = correction + step * time_valid[:, :, None, None]
    mask = self.upsample_mask(source_features)
    dense = convex_upsample(correction.reshape(b, u, h, w, 3), mask, self.config.patch_size)
    return dict(correction=dense, match_fraction=support.detach(),
                correction_relative_norm=(correction.detach().norm(dim=-1) / scale[:, None, None]).mean((1, 2)))
