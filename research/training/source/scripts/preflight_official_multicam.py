"""Check every new C/T layout with the actual giant checkpoint before admission.

Uses one idle main-training GPU after its segment boundary, never an ablation
GPU. Forward/backward only: no optimizer update and no checkpoint rewrite.
"""
import argparse
from dataclasses import replace
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from multicamera4rc.checkpoint import config_from_checkpoint
from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples, read_data_config
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.model import MultiCamera4RC
from multicamera4rc.train import _TrainingStep, _sources
from scripts.continue_quality_recovery import digest
from scripts.cutover_native_motion_stereo import write_json


def existing_layout_cases(dataset):
    """Real mixed-sampler cases: largest camera count and longest multiview clip.

    Cover every existing multiview adapter; keep the two Syn4D releases separate.
    Repeated TartanAir environment sources share an adapter/layout contract.
    """
    cases={}
    for si,source in enumerate(dataset.sources):
        if source['name'] in ('mv_kubric','panoptic'):
            raise ValueError('loss-only preflight must not introduce official sources')
        name=source['display_name'] if source['name'] in ('syn4d','mugen_h5') else source['name']
        layouts=source.get('layouts') or [source]
        layouts=[(x['cameras'],x['num_times']) for x in layouts if x['cameras']>1]
        if not layouts:
            continue
        selected={max(layouts),max(layouts,key=lambda x:(x[1],x[0]))}
        expected={(name,c,t) for c,t in selected}
        start=dataset.quota_ends[si-1] if si else 0
        for index in range(start,dataset.quota_ends[si]):
            if expected.issubset(cases):break
            p=dataset.sample_plan(index);key=(name,p['cameras'],p['num_times'])
            if key in expected:cases.setdefault(key,index)
        if not expected.issubset(cases):
            raise ValueError('actual sampler omitted an existing multiview stress layout')
    required={'kubric_multicam','vkitti2_stereo','syn4d_v1','syn4d_sim'}
    if not required.issubset({key[0] for key in cases}):
        raise ValueError('existing cross-camera source coverage is incomplete')
    return cases


def layout_cases(dataset, include_existing_cross_camera=False, existing_only=False):
    if existing_only:
        return existing_layout_cases(dataset)
    cases={}
    for si,source in enumerate(dataset.sources):
        name=source['name']  # internal adapter key; display_name is the user alias
        if name in ('mv_kubric','panoptic'):
            expected={(name,x['cameras'],x['num_times']) for x in source['layouts']}
        elif include_existing_cross_camera and source['display_name'] in ('syn4d_v1','syn4d_sim'):
            # Largest other-camera pixel expansion at each native resolution.
            # These existing sources also receive the new loss, so checking
            # only the new 512px MV-Kubric packs would miss a memory regression.
            name=source['display_name']
            c,t=(8,2) if name=='syn4d_v1' else (4,4)
            expected={(name,c,t)}
        else:
            continue
        start=dataset.quota_ends[si-1] if si else 0
        for index in range(start,dataset.quota_ends[si]):
            p=dataset.sample_plan(index)
            key=(name,p['cameras'],p['num_times'])
            if key in expected:cases.setdefault(key,index)
            if expected.issubset(cases):
                break
        if not expected.issubset(cases):
            raise ValueError('actual mixed sampler omitted a proposed layout')
    names={'mv_kubric','panoptic'} | ({'syn4d_v1','syn4d_sim'} if include_existing_cross_camera else set())
    if {x[0] for x in cases} != names:
        raise ValueError('all required new and existing source layouts must be checked')
    return cases


def run(config, checkpoint, output, gpu_uuid, loss_config=None, existing_only=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid:
        raise ValueError('preflight must expose only the selected main-training GPU UUID')
    output=Path(output)
    with open('/tmp/multicamera4rc_'+gpu_uuid+'.lock','a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        if gpu_uuid in apps:
            raise RuntimeError('selected main-training GPU is occupied; refusing to share')
        torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.90,0)
        before=(Path(checkpoint).stat().st_size,Path(checkpoint).stat().st_mtime_ns)
        payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
        cfg=replace(config_from_checkpoint(payload),gradient_checkpointing=True,attention_chunk_size=None,visibility_chunk_size=None)
        model=MultiCamera4RC(cfg);model.load_state_dict(payload['model'],strict=True)
        settings=payload['training_state']['loss_config']
        if loss_config:
            from scripts.continue_official_multicam import cross_camera_loss_config
            candidate=json.loads(Path(loss_config).read_text())
            if candidate!=cross_camera_loss_config(settings,candidate.get('cross_camera_3d_weight',0)):
                raise ValueError('preflight may only add the declared cross-camera term')
            settings=candidate
        criterion=MultiCamera4RCLoss(**settings)
        result=dict(passed=False,checkpoint=str(checkpoint),step=payload['step'],
            config=str(config),config_sha256=digest(config),gpu_uuid=gpu_uuid,
            optimizer_update_performed=False,loss_config=criterion.to_config(),rows=[])
        del payload
        model=model.cuda().train();step=_TrainingStep(model,criterion)
        dataset=create_dataset(dict(read_data_config(config),skip_invalid=False))
        if existing_only and not criterion.cross_camera_3d_weight>0:
            raise ValueError('loss-only preflight requires cross-camera supervision')
        cases=layout_cases(dataset,include_existing_cross_camera=criterion.cross_camera_3d_weight>0,
                           existing_only=existing_only)
        result['expected_cases']=[list(key) for key in sorted(cases)]
        for (name,c,t),index in sorted(cases.items()):
            torch.manual_seed(987+index);torch.cuda.manual_seed(987+index)
            begun=time.monotonic();sample=dataset[index];batch=collate_adapter_samples([sample])
            batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            batch['targets']={k:v.cuda() for k,v in batch['targets'].items()}
            source=_sources(batch,4,64,True)
            target=materialize_dense_targets(batch,source)
            for key in ('ray','visibility'):
                if criterion.weights[key]==0 and key in target:
                    target.pop(key);target.pop(key+'_valid')
            visibility=any(k in target for k in ('visibility','sparse_visibility'))
            model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss,metrics=step(batch['images'],batch['camera_id'],batch['time_idx'],batch['query_time_idx'],
                    source,batch['observation_valid'],target,visibility,batch.get('principal_canvas'))
            loss.backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads):
                raise FloatingPointError(f'nonfinite new-source loss or gradients: {name}, C{c}T{t}')
            if name=='panoptic' and (any(p.grad is not None for p in model.motion_decoder.parameters())
                                    or any(p.grad is not None for p in model.head.parameters())):
                raise ValueError('calibration-only clip produced depth/motion-head gradients')
            if criterion.cross_camera_3d_weight:
                count=float(metrics.get('cross_camera_3d_points',torch.tensor(0)))
                if (name!='panoptic' and c>1 and count<=0) or ((name=='panoptic' or c==1) and count!=0):
                    raise ValueError('cross-camera supervision selected the wrong source/layout')
            row=dict(source=name,cameras=c,times=t,index=index,input_shape=list(batch['images'].shape),
                decoded_sources=source.shape[1],loss=float(loss),finite_gradients=True,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-begun,
                cross_camera_metrics={k:float(v) for k,v in metrics.items() if k.startswith('cross_camera_3d')})
            result['rows'].append(row);write_json(output,result);print(json.dumps(row),flush=True)
            model.zero_grad(set_to_none=True);del sample,batch,source,target,loss,metrics,grads
            gc.collect();torch.cuda.empty_cache()
        if before != (Path(checkpoint).stat().st_size,Path(checkpoint).stat().st_mtime_ns):
            raise ValueError('source checkpoint changed during preflight')
        result.update(passed=True,checkpoint_modified=False);write_json(output,result)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('config','checkpoint','output','gpu-uuid'):
        p.add_argument('--'+flag,required=True)
    p.add_argument('--loss-config')
    p.add_argument('--existing-only',action='store_true')
    a=p.parse_args();run(a.config,a.checkpoint,a.output,a.gpu_uuid,a.loss_config,a.existing_only)
