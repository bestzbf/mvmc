"""Write a proposed mixed-data recipe without changing a running training job."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from multicamera4rc.datasets.mixture import read_data_config


def build_mixture(base, mv_root, panoptic_root, panoptic_audit, *, mv_fraction=.10, panoptic_fraction=.03):
    if base.get('supervision') != 'dense' or base.get('image_mode') != 'native':
        raise ValueError('official data integration requires the existing native dense recipe')
    if any(s['adapter'] in ('mv_kubric','panoptic') for s in base['sources']):
        raise ValueError('official sources already exist; refusing duplicate weighting')
    if not 0 < mv_fraction < 1 or not 0 < panoptic_fraction < 1 or mv_fraction+panoptic_fraction >= 1:
        raise ValueError('source fractions must be positive with sum below one')
    if panoptic_audit.get('stage') != 'passed' or not panoptic_audit.get('strict_exclusion_recheck_passed'):
        raise ValueError('Panoptic requires a passed audit and strict reviewed-exclusion recheck')
    audit_path = Path(panoptic_audit['index_report']['root'])
    audited_root = audit_path.resolve()
    if audited_root != Path(panoptic_root).resolve():
        raise ValueError('Panoptic audit belongs to a different data root')
    excluded = panoptic_audit.get('reviewed_exclude_videos')
    if not isinstance(excluded,list) or len(set(excluded)) != len(excluded):
        raise ValueError('Panoptic audit must supply an explicit unique reviewed video exclusion list')
    result=deepcopy(base)
    total=sum(float(s.get('weight',1)) for s in result['sources'])
    if total <= 0:
        raise ValueError('base source weights must have positive sum')
    remaining=1-mv_fraction-panoptic_fraction
    for source in result['sources']:
        source['weight']=float(source.get('weight',1))/total*remaining
    mv=dict(name='mv_kubric_official',adapter='mv_kubric',root=str(Path(mv_root).resolve()),
        split='train',cameras=10,num_times=24,weight=mv_fraction,strides=[1,2],dense_kind='kubric',
        layouts=[dict(cameras=c,num_times=t,weight=w) for c,t,w in
                 [(1,24,1),(2,12,2),(3,8,2),(4,6,3),(6,4,1),(8,3,1),(10,2,1)]],
        options=dict(require_full_conversion=True,val_ratio=.05,split_strategy='tail',
                     random_cameras=True,index_workers=2,verbose=False))
    panoptic=dict(name='panoptic_calibration',adapter='panoptic',root=str(audited_root),split='train',
        cameras=8,num_times=4,weight=panoptic_fraction,strides=[1,2,4],dense_kind='camera_only',
        layouts=[dict(cameras=c,num_times=t,weight=w) for c,t,w in [(2,4,1),(4,2,2),(8,1,1)]],
        options=dict(max_long_side=960,strict=True,exclude_videos=excluded,
                     # The data and cache can be relocated independently behind
                     # symlinks. Resolve the recorded cache path itself, not a
                     # new sibling inferred from the data's physical location.
                     cache_dir=str((audit_path.parent/'.panoptic_training_index').resolve()),index_workers=2,verbose=False))
    result['sources'].extend([mv,panoptic])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config',required=True)
    parser.add_argument('--mv-kubric-root',required=True)
    parser.add_argument('--panoptic-root',required=True)
    parser.add_argument('--panoptic-audit',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--mv-fraction',type=float,default=.10)
    parser.add_argument('--panoptic-fraction',type=float,default=.03)
    args=parser.parse_args()
    audit=json.loads(Path(args.panoptic_audit).read_text())
    config=build_mixture(read_data_config(args.base_config),args.mv_kubric_root,args.panoptic_root,
                         audit,mv_fraction=args.mv_fraction,panoptic_fraction=args.panoptic_fraction)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite a registered recipe: {args.output}')
    args.output.write_text(json.dumps(config,indent=2)+'\n')
    args.output.with_suffix('.integration.json').write_text(json.dumps(dict(
        base_config=str(Path(args.base_config).resolve()),panoptic_audit=str(Path(args.panoptic_audit).resolve()),
        activation='proposed_only; current main training and paired experiments are unchanged',
        note='Adding sources changes the sampler contract; activate at an audited training boundary.',
        fractions=dict(existing=1-args.mv_fraction-args.panoptic_fraction,
                       mv_kubric=args.mv_fraction,panoptic_camera_only=args.panoptic_fraction)),indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),sources=len(config['sources']),activated=False)))


if __name__=='__main__':
    main()
