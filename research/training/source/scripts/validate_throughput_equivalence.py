"""Production-model, native-data output/gradient checks; no optimizer updates.

Run with torchrun on reserved GPUs. Each process checks different native cases
independently; it does not initialize a collective or consume training state.
"""
import argparse
from dataclasses import replace
import gc
import json
import math
import os
from pathlib import Path
import traceback

import torch

from multicamera4rc import MultiCamera4RC
from multicamera4rc.checkpoint import config_from_checkpoint
from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.query_pair import attach_pair_surface_ids
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.train import _sources, _TrainingStep


def capture_prediction(outputs):
    """Observe the actual loss input, including the native geometry-only path."""
    def capture(module, inputs):
        prediction = inputs[0]
        for key in ('depth','depth_conf','K','c2w','base_points','trajectory',
                    'displacement','motion_confidence','visibility_logits'):
            if key in prediction:
                outputs[key] = prediction[key].detach().cpu()
    return capture


def case_indices(dataset, rank):
    # Cover different native sources on all GPUs, using real mixture
    # indices and deterministic adapter plans (no synthetic or resized inputs).
    wanted=(('mv_kubric_official','kubric_variable'),('syn4d_sim','syn4d_v1'),
            ('pointodyssey','dynamic_replica_left'),('vkitti2_stereo','mv_kubric_official'),
            ('dynamic_replica_right','pointodyssey'))[rank%5]
    cases=[]
    for name in wanted:
        si=next(i for i,s in enumerate(dataset.sources) if s['display_name']==name)
        begin=dataset.quota_ends[si-1] if si else 0
        index=begin+min(11+rank,dataset.quotas[si]-1)
        for offset in range(min(128,dataset.quotas[si])):
            candidate=begin+((11+rank+offset)%dataset.quotas[si])
            plan=dataset.sample_plan(candidate)
            monocular = name in ('pointodyssey','dynamic_replica_left','dynamic_replica_right')
            if plan['num_times']>1 and (monocular or plan['cameras']>1):
                index=candidate;break
        else:raise ValueError('no dynamic native test case for '+name)
        cases.append(index)
    return cases


def check(args):
    rank=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    device=torch.device('cuda',rank)
    data=torch.load(args.checkpoint,mmap=True,map_location='cpu',weights_only=True)
    state=data['training_state'];cfg=config_from_checkpoint(data)
    dataset=create_dataset(args.data_config);dataset.set_epoch(state['data_epoch'])
    cases=case_indices(dataset,rank)
    model=MultiCamera4RC(cfg);model.load_state_dict(data['model'],strict=True)
    model=model.to(device).train()
    criterion=MultiCamera4RCLoss(**state['loss_config'])
    step=_TrainingStep(model,criterion)
    records=[]
    for index in cases:
        batch=collate_adapter_samples([dataset[index]])
        for k,v in list(batch.items()):
            if isinstance(v,torch.Tensor):batch[k]=v.to(device)
        batch['targets']={k:v.to(device) for k,v in batch['targets'].items()}
        torch.manual_seed(713+rank)
        source=_sources(batch,state['source_count'],state['max_source_time_pairs'],
                        state['prefer_motion_sources'],criterion.query_pair_source_sampling)
        targets=materialize_dense_targets(batch,source)
        if criterion.query_pair_weight>0 or criterion.query_pair_source_sampling:
            targets=attach_pair_surface_ids(targets,batch['targets'])
        for name in ('visibility','sparse_visibility'):
            if name in targets and not targets.get(name+'_valid',torch.tensor(True,device=device)).any():
                targets.pop(name);targets.pop(name+'_valid',None)
        visibility=any(k in targets for k in ('visibility','sparse_visibility'))
        record={'index':index,'dataset':batch['dataset_name'],'clip':batch['clip_id'],
            'image_hw':list(batch['images'].shape[-2:]),'observations':len(batch['camera_id'][0]),'arms':{}}
        baseline=None
        for mode in ('reference','shared','shared_kv'):
            model.config=replace(cfg,reuse_shared_computation=mode!='reference',reuse_decoder_kv=mode=='shared_kv')
            model.zero_grad(set_to_none=True);torch.manual_seed(713+rank)
            outputs={}
            gradients=loss=metrics=None
            motion_called=[False]
            hook=criterion.register_forward_pre_hook(capture_prediction(outputs))
            motion_hook=model.register_forward_hook(lambda *unused: motion_called.__setitem__(0,True))
            torch.cuda.reset_peak_memory_stats(device)
            try:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,metrics=step(batch['images'],batch['camera_id'],batch['time_idx'],batch['query_time_idx'],
                        source,batch['observation_valid'],targets,visibility,batch.get('principal_canvas'))
                loss.backward();torch.cuda.synchronize(device)
                if not {'trajectory','depth','K','c2w'}<=outputs.keys():
                    raise ValueError('native loss input lacks required geometric predictions')
                gradients={n:None if p.grad is None else p.grad.detach().float().cpu() for n,p in model.named_parameters()}
                arm={'loss':float(loss.detach()),'decoded_motion':motion_called[0],
                     'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30,
                     'finite':all(g is None or torch.isfinite(g).all().item() for g in gradients.values())}
                if not arm['finite']:raise ValueError('nonfinite gradient')
                if mode=='reference':
                    record['decoded_motion']=motion_called[0]
                    baseline=(outputs,gradients,arm['loss']);arm['passed']=True
                else:
                    if motion_called[0]!=record['decoded_motion']:raise ValueError('supervised execution branch changed')
                    a,ga,la=baseline
                    if a.keys()!=outputs.keys():raise ValueError('output fields changed')
                    arm['output_max_abs']={}
                    for key in a:
                        torch.testing.assert_close(a[key],outputs[key],rtol=3e-5,atol=3e-5,msg=key)
                        arm['output_max_abs'][key]=(a[key]-outputs[key]).abs().max().item()
                    if not math.isclose(la,arm['loss'],rel_tol=3e-5,abs_tol=3e-5):raise ValueError('loss changed')
                    aa=bb=dot=error=0.;worst=0.;groups={}
                    for name,old in ga.items():
                        new=gradients[name]
                        if (old is None)!=(new is None):raise ValueError('gradient participation changed: '+name)
                        if old is None:continue
                        values=[old.square().sum(dtype=torch.float64).item(),new.square().sum(dtype=torch.float64).item(),
                                (old*new).sum(dtype=torch.float64).item(),(old-new).square().sum(dtype=torch.float64).item()]
                        aa+=values[0];bb+=values[1];dot+=values[2];error+=values[3]
                        group=('backbone' if name.startswith('backbone.') else 'refinement' if name.startswith('trajectory_refiner.')
                               else 'geometry' if name.startswith(('head.','cam_dec.')) else 'motion')
                        groups[group]=[x+y for x,y in zip(groups.get(group,[0.,0.,0.,0.]),values)]
                        worst=max(worst,(old-new).abs().max().item())
                    arm.update(gradient_relative_l2=math.sqrt(error/max(aa,1e-30)),
                        gradient_cosine=dot/max(math.sqrt(aa*bb),1e-30),gradient_max_abs_difference=worst)
                    arm['gradient_groups']={g:dict(relative_l2=math.sqrt(e/max(a,1e-30)),
                        cosine=1. if a==b==0 else dot/max(math.sqrt(a*b),1e-30)) for g,(a,b,dot,e) in groups.items()}
                    arm['passed']=(arm['gradient_relative_l2']<=.005 and arm['gradient_cosine']>=.99999
                        and all(v['relative_l2']<=.005 and v['cosine']>=.99999 for v in arm['gradient_groups'].values()))
                    if not arm['passed']:arm['reason']='BF16 shared-gradient accumulation exceeded tolerance'
                record['arms'][mode]=arm
            except Exception:
                record['arms'][mode]={'passed':False,'error':traceback.format_exc()}
                if mode=='reference':raise
            finally:
                hook.remove();motion_hook.remove();model.zero_grad(set_to_none=True)
                del loss,metrics,gradients,outputs
                gc.collect();torch.cuda.empty_cache()
        records.append(record)
        # Publish completed cases even if a later high-resolution case fails.
        dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
        temporary=dest/f'rank_{rank}_partial.tmp'
        temporary.write_text(json.dumps(dict(rank=rank,checkpoint=args.checkpoint,records=records),indent=2))
        temporary.replace(dest/f'rank_{rank}_partial.json')
        del baseline,batch,targets
        gc.collect();torch.cuda.empty_cache()
    result={'rank':rank,'checkpoint':args.checkpoint,'records':records,
        'passed_modes':[mode for mode in ('shared','shared_kv') if all(r['arms'].get(mode,{}).get('passed') for r in records)]}
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    (dest/f'rank_{rank}.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','data-config','output'):p.add_argument('--'+key,required=True)
    check(p.parse_args())
