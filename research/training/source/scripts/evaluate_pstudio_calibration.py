"""Final PStudio camera-head and tracking audit on the frozen 954-clip plan.

Uses the already corrected comparison runner's RGB preprocessing, inference
and official APD3D/EPE scoring without changing its 60k release-only loader.
Only RGB and supplied starting XY queries reach the model. Test results are
reported after the held-out calibration decision, never used to select an arm.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import gc
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

from multicamera4rc import MultiCamera4RC
from multicamera4rc.checkpoint import config_from_checkpoint
from scripts.cutover_native_motion_stereo import write_json


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def read_protocol(config):
    root=Path(config['comparison_root'])
    for relative,expected in config['reference_sha256'].items():
        if digest(root/relative)!=expected:raise ValueError('PStudio reference source changed: '+relative)
    for key in ('plan','official_metrics'):
        if digest(config[key])!=config[key+'_sha256']:raise ValueError('PStudio '+key+' changed')
    plan=json.loads(Path(config['plan']).read_text())
    if (plan.get('subset')!='pstudio' or plan.get('split')!='full_eval'
            or plan.get('schema')!='omnix16-first-visible-v2'
            or plan.get('metric_eval_short_side')!=256 or plan.get('frames')!=16
            or len(plan['cases'])!=config['clips']
            or len({c['sequence'] for c in plan['cases']})!=config['sequences']
            or len({c['id'] for c in plan['cases']})!=config['clips']):
        raise ValueError('PStudio full-test protocol changed')
    sys.path.insert(0,str(root))
    common=importlib.import_module('evaluation.common')
    protocol=importlib.import_module('evaluation.run_tapvid')
    if (Path(common.__file__).resolve()!= (root/'evaluation/common.py').resolve()
            or Path(protocol.__file__).resolve()!= (root/'evaluation/run_tapvid.py').resolve()):
        raise ValueError('another evaluation module shadows the frozen reference')
    official=common.load_module('_intrinsics_trial_pstudio_scorer',Path(config['official_metrics']))
    return plan,common,protocol,official


class CaptureCamera:
    """Observe exactly the encoder invocation used by reference tracking."""
    def __init__(self,model):
        self.model=model
        self.intrinsics=None

    def encode(self,*args,**kwargs):
        scene=self.model.encode(*args,**kwargs)
        self.intrinsics=scene.geometry['K'][0].detach().float().cpu().numpy()
        return scene

    def decode(self,*args,**kwargs):
        return self.model.decode(*args,**kwargs)


def camera_metrics(predicted,raw_gt,metadata):
    """Compare model K to GT under the exact half-pixel RGB resize."""
    predicted=np.asarray(predicted,dtype=np.float64)
    gt=np.asarray(raw_gt,dtype=np.float64)
    if (predicted.ndim!=3 or predicted.shape[1:]!=(3,3) or len(predicted)==0
            or gt.shape!=(4,) or not np.isfinite(gt).all() or (gt[:2]<=0).any()):
        raise ValueError('invalid PStudio camera GT/head shape')
    scale=np.asarray(metadata['content_hw'][::-1])/np.asarray(metadata['raw_hw'][::-1])
    actual=np.stack((predicted[:,0,0],predicted[:,1,1],predicted[:,0,2],predicted[:,1,2]),-1)
    if not np.isfinite(actual).all() or (actual[:,:2]<=0).any():
        raise ValueError('nonfinite or nonpositive predicted focal length')
    # Undo content resize; patch padding only extends bottom/right.
    actual[:,:2]/=scale
    actual[:,2:]=(actual[:,2:]+.5)/scale-.5
    ratio=actual[:,:2]/gt[:2]
    pp=actual[:,2:]-gt[2:]
    hw=np.asarray(metadata['raw_hw'][::-1])
    def metrics(r,p):
        return dict(fx_ratio=float(r[:,0].mean()),fy_ratio=float(r[:,1].mean()),
            focal_abs_rel=float(np.abs(r-1).mean()),log_focal_error=float(np.abs(np.log(r)).mean()),
            focal_below_90pct=float((r<.9).mean()),focal_above_110pct=float((r>1.1).mean()),
            principal_error_raw_px=float(np.linalg.norm(p,axis=-1).mean()),
            principal_error_normalized=float(np.linalg.norm(p/hw,axis=-1).mean()))
    result={'intrinsics_'+k:v for k,v in metrics(ratio,pp).items()}
    result.update({'source_intrinsics_'+k:v for k,v in metrics(ratio[:1],pp[:1]).items()})
    return result


def summarize(records):
    grouped=defaultdict(list)
    for row in records:grouped[row['sequence']].append(row['metrics'])
    keys=sorted(set.intersection(*(set(row['metrics']) for row in records)))
    keys=[k for k in keys if k not in ('global_scale','queries','visible_point_times')]
    sequences={name:{key:float(np.mean([row[key] for row in rows])) for key in keys}
               for name,rows in sorted(grouped.items())}
    return dict(by_sequence=sequences,
                mean_metrics={key:float(np.mean([row[key] for row in sequences.values()])) for key in keys},
                clip_mean_metrics={key:float(np.mean([row['metrics'][key] for row in records])) for key in keys})


def compare_reports(reports):
    control=reports['control']
    identities=[(r['id'],r['sequence']) for r in control['records']]
    comparisons={}
    for label,report in reports.items():
        if (report['stage']!='passed' or report['contract']['protocol']!=control['contract']['protocol']
                or report['contract']['loading']['step']!=control['contract']['loading']['step']
                or any(report['contract'].get(key)!=control['contract'].get(key)
                       for key in ('implementation','model_implementation','preflight_clips'))
                or [(r['id'],r['sequence']) for r in report['records']]!=identities):
            raise ValueError('final PStudio arms must share step, protocol and all clips')
        if label=='control':continue
        comparisons[label]={}
        names=sorted(control['by_sequence'])
        rng=np.random.default_rng(20261013)
        indices=rng.integers(0,len(names),size=(2000,len(names)))
        for key in ('intrinsics_log_focal_error','source_intrinsics_log_focal_error',
                    'intrinsics_fx_ratio','intrinsics_fy_ratio','apd3d','epe_m'):
            a=np.array([control['by_sequence'][name][key] for name in names])
            b=np.array([report['by_sequence'][name][key] for name in names])
            comparisons[label][key]=dict(control=float(a.mean()),candidate=float(b.mean()),
                delta=float((b-a).mean()),paired_sequence_delta_95pct=
                np.quantile((b-a)[indices].mean(1),[.025,.975]).tolist())
    return dict(stage='passed',test_used_for_selection=False,clips=len(identities),
                sequences=len(control['by_sequence']),
                means={label:report['mean_metrics'] for label,report in reports.items()},
                comparisons=comparisons,protocol=control['contract']['protocol'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preflight-clips',type=int,default=0,
                        help='Execute only this many initial clips; never produces a passed full-test report')
    parser.add_argument('--memory-fraction',type=float,default=.70)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    plan,common,protocol,official=read_protocol(config)
    torch.set_num_threads(4)
    device=torch.device('cuda:0')
    torch.cuda.set_device(device)
    if not 0<args.memory_fraction<=1 or not 0<=args.preflight_clips<=len(plan['cases']):
        raise ValueError('invalid preflight/resource limit')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device)
    cases=plan['cases'][:args.preflight_clips] if args.preflight_clips else plan['cases']
    payload=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=True)
    cfg=replace(config_from_checkpoint(payload),gradient_checkpointing=False)
    model=MultiCamera4RC(cfg)
    model.load_state_dict(payload['model'],strict=True)
    loading=dict(path=str(args.checkpoint.resolve()),sha256=digest(args.checkpoint),
                 step=payload['step'],strict=True,config=cfg.to_dict())
    del payload
    model=model.to(device).eval().requires_grad_(False)
    captured=CaptureCamera(model)
    contract=dict(protocol=config,loading=loading,implementation=digest(__file__),
                  preflight_clips=args.preflight_clips,memory_fraction=args.memory_fraction,
                  model_implementation={str(p.relative_to(Path(__file__).resolve().parents[1])):digest(p)
                    for p in sorted((Path(__file__).resolve().parents[1]/'multicamera4rc').rglob('*.py'))},
                  torch=torch.__version__,numpy=np.__version__)
    # Normalize library version wrapper types and keep the resume contract exact.
    contract=json.loads(json.dumps(contract))
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'results.json';records=[]
    if path.exists():
        previous=json.loads(path.read_text())
        if not args.resume or previous['contract']!=contract:
            raise ValueError('refusing to mix PStudio checkpoints or protocols')
        records=previous['records']
        if [r['id'] for r in records]!=[c['id'] for c in cases[:len(records)]]:
            raise ValueError('PStudio resume records are not the completed plan prefix')
    report=dict(stage='running',contract=contract,records=records,failures=[],
                requested_clips=config['clips'],requested_sequences=config['sequences'],
                test_used_for_selection=False)
    write_json(path,report);verified=set()
    try:
        for case in cases[len(records):]:
            began=time.monotonic()
            if case['path'] not in verified:
                if digest(case['path'])!=case['sha256']:raise ValueError('PStudio RGB/GT changed')
                verified.add(case['path'])
            with np.load(case['path'],allow_pickle=True) as data:
                start,points=case['start'],case['point_ids']
                rgb=[np.asarray(Image.open(io.BytesIO(bytes(x))).convert('RGB'))
                     for x in data['images_jpeg_bytes'][start:start+16]]
                truth=data['tracks_XYZ'][start:start+16,points].astype(np.float32)
                visible=data['visibility'][start:start+16,points].astype(bool)
                gt=data['fx_fy_cx_cy'].astype(np.float64)
            if (truth.shape!=(16,len(points),3) or not len(points) or not visible[0].all()
                    or not np.isfinite(truth[visible]).all() or (truth[0,:,2]<=0).any()):
                raise ValueError('invalid PStudio starting queries/GT')
            xy=truth[0,:,:2]/truth[0,:,2:]*gt[:2]+gt[2:]
            images,metadata=common.prepare_rgb(rgb,plan['long_side'])
            torch.cuda.synchronize(device);inference_start=time.monotonic()
            predicted=protocol.predict(captured,'multid4rt',images,xy,metadata,start,device)
            torch.cuda.synchronize(device);inference_seconds=time.monotonic()-inference_start
            metrics=protocol.score(official,predicted,truth,visible,gt,metadata['raw_hw'])
            metrics.update(camera_metrics(captured.intrinsics,gt,metadata))
            np.savez_compressed(args.output/(case['id']+'.npz'),tracks_XYZ=predicted,
                                predicted_K=captured.intrinsics,point_ids=points,start=start)
            records.append(dict(id=case['id'],sequence=case['sequence'],input=metadata,
                                metrics=metrics,seconds=time.monotonic()-began,
                                inference_seconds=inference_seconds))
            write_json(path,report)
            print(json.dumps(dict(completed=len(records),total=config['clips'],**records[-1])),flush=True)
            del images,predicted
            gc.collect();torch.cuda.empty_cache()
        report.update(stage='preflight_passed' if args.preflight_clips else 'passed',**summarize(records))
        write_json(path,report)
    except Exception:
        report.update(stage='failed',error=traceback.format_exc())
        write_json(path,report)
        raise


if __name__=='__main__':main()
