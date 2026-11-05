"""Read only pinned plans/manifests/indexes/receipts, never NPZ or weights."""
import hashlib
import json
from pathlib import Path
import datetime as dt


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def flag(argv, name):
    return argv[argv.index(name)+1] if name in argv else None


def method_name(method, checkpoint=None):
    names = ['ARROW','MoSE3','Point4D-direct-query','VGGT-Omega-1B-512+TrackEverything','UniQuery4R','ReLift+AllTracker+DA3']
    if method in names:
        return method
    if checkpoint and checkpoint.get('sha256') == '2557796c0068dbf33550ea4098d12b5f779cfab1e977decaf8ba13b79feaccb0':
        return 'Ours134400'
    return None


def main(config):
    out = {'captured_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(), 'read_only':True,
           'NPZ_or_weight_payload_read':False, 'payload_hashes_recomputed':False,
           'plans':[], 'manifests':{}, 'producers':[], 'errors':[]}
    manifest_paths = set(config['manifests'])
    index_paths = set(config['indexes'])
    plans = list(config['plans'])
    for path in plans:
        p = Path(path)
        if not p.is_file():
            continue
        try:
            plan = read(p)
            row = {'path':str(p), 'sha256':digest(p), 'jobs':[]}
            for j in plan.get('jobs',[]):
                argv = j.get('predict_argv', [])
                mf = j.get('manifest') or flag(argv, '--manifest')
                predroot = flag(argv, '--output')
                if mf:
                    manifest_paths.add(mf)
                if predroot:
                    index_paths.add(str(Path(predroot)/'predictions.json'))
                checkpoint = j.get('expected_baseline_checkpoint', j.get('expected_checkpoint',j.get('checkpoint',plan.get('checkpoint',{}))))
                method = method_name(j.get('method'), checkpoint)
                if not method:
                    continue
                row['jobs'].append({'id':j['id'], 'method':method, 'dataset':j.get('dataset'),
                     'job_sha256':canonical(j), 'manifest':mf, 'expected_cases':j.get('expected_cases'),
                     'predict_argv':argv, 'score_argv':j.get('score_argv'), 'environment':j.get('environment',{}),
                     'expected_checkpoint_sha256':checkpoint.get('sha256'), 'expected_checkpoint_canonical_sha256':canonical(checkpoint),
                     'checkpoint_artifacts':j.get('checkpoint_artifacts'),
                     'original_job_fields':list(j), 'predictions':str(Path(predroot)/'predictions.json') if predroot else j.get('predictions_path')})
            out['plans'].append(row)
        except Exception as e:
            out['errors'].append({'path':str(p),'phase':'plan','error':str(e)})
    queuepath=Path(config['queue'])
    queue=read(queuepath)
    out['previous_queue']={'path':str(queuepath),'sha256':digest(queuepath),'sealed':queue.get('sealed')}
    for j in queue['jobs']:
        if j['kind'] != 'gpu':continue
        argv=j['argv'];mf=flag(argv,'--manifest');predroot=flag(argv,'--output')
        if mf:manifest_paths.add(mf)
        if predroot:index_paths.add(str(Path(predroot)/'predictions.json'))
    quick=Path('@@BASELINE_ASSETS_ROOT@@/quick_newmodels_20261104')
    for pattern in ['tap_v2/**/manifest*.json','tap/**/manifest*.json','relift/manifest*.json']:
        manifest_paths.update(str(p) for p in quick.glob(pattern))
    manifest_by_sha={}
    for path in sorted(manifest_paths):
        p=Path(path)
        if not p.is_file():continue
        try:
            mf=read(p)
            if 'cases' not in mf:continue
            sha=digest(p);manifest_by_sha.setdefault(sha,str(p))
            cases=[]
            for i,c in enumerate(mf['cases']):
                raw=Path(c['input']);ip=raw if raw.is_absolute() else p.parent/raw
                row={k:c.get(k) for k in ['id','name','subset','input_sha256','frames','queries','height','width','views']}
                row.update(source_index=i,input=str(ip),input_exists=ip.is_file(),
                           input_bytes=ip.stat().st_size if ip.is_file() else None,source_case_sha256=canonical(c))
                cases.append(row)
            out['manifests'][str(p)]={'path':str(p),'sha256':sha,'schema':mf.get('schema'),
                'benchmark':mf.get('benchmark'),'protocol':mf.get('protocol'),'input_condition':mf.get('input_condition'),
                'coordinates':mf.get('coordinates'),'official_sources':mf.get('official_sources'),'cases':cases}
        except Exception as e:out['errors'].append({'path':str(p),'phase':'manifest','error':str(e)})
    for path in sorted(index_paths):
        p=Path(path);statep=p.parent/'run-state.json'
        if not statep.is_file():continue
        try:
            state=read(statep);method=method_name(state.get('method'),state.get('checkpoint'))
            if not method:continue
            statehash=canonical(state)
            normalized={k:v for k,v in state.items() if k!='manifest_sha256'}
            cp=state.get('checkpoint',{})
            row={'prediction_index':str(p),'index_exists':p.is_file(),'method':method,
                 'run_state':str(statep),'run_state_file_sha256':digest(statep),'run_state_canonical_sha256':statehash,
                 'source_manifest_sha256':state['manifest_sha256'],
                 'source_manifest':manifest_by_sha.get(state['manifest_sha256']),
                 'group_signature_sha256':canonical(normalized),
                 'signature_rule':'Entire actual run-state except manifest_sha256; no runtime/source/settings/device fields dropped',
                 'signature_components':{k:canonical(v) for k,v in normalized.items()},
                 'checkpoint_sha256':cp.get('sha256'),'checkpoint_canonical_sha256':canonical(cp),
                 'checkpoint_step':cp.get('step',cp.get('checkpoint_step')),
                 'loader_expected_step':state.get('loader_assertions',{}).get('expected_step'),
                 'inference_settings':state.get('inference_settings'),
                 'runtime_versions':state.get('runtime_versions'),'numerics':state.get('numerics'),
                 'cases':[],'rejections':[]}
            if row['source_manifest'] is None:raise ValueError('No exact producer manifest for run-state SHA')
            mf=read(row['source_manifest']);by_id={c['id']:c for c in mf['cases']}
            index=read(p) if p.is_file() else None
            if index:
                row['prediction_index_sha256']=digest(p)
                if index['run_state_sha256']!=statehash or index['manifest_sha256']!=state['manifest_sha256'] or index['checkpoint']!=cp:
                    raise ValueError('Index/run-state identity differs')
                if [c['id'] for c in index['cases']] != [c['id'] for c in mf['cases']]:
                    raise ValueError('Committed producer index not exact producer manifest order/full coverage')
                records=index['cases']
            else:
                records=[read(r)['record'] for r in sorted(p.parent.glob('*.receipt.json'))]
            seen=set()
            for r in records:
                try:
                    cid=r['id'];c=by_id[cid];pred=(p.parent/r['file']).resolve();receipt=pred.with_suffix('.receipt.json')
                    if cid in seen:raise ValueError('Duplicate producer case ID')
                    seen.add(cid)
                    if not pred.is_relative_to(p.parent.resolve()) or not pred.is_file():raise ValueError('Payload absent/escaped producer root')
                    rec=read(receipt)
                    if rec['record']!=r or rec['run_state_sha256']!=statehash or rec['case_sha256']!=canonical(c):raise ValueError('Receipt case/state/record differs')
                    if r['input_sha256']!=c['input_sha256']:raise ValueError('Input SHA differs')
                    shape=rec['arrays']['tracks_XYZ']['shape']
                    if shape[0]!=c['frames'] or shape[-1]!=3:raise ValueError('Declared native coordinate shape differs')
                    row['cases'].append({'id':cid,'input_sha256':r['input_sha256'],'prediction':str(pred),'prediction_sha256':r['sha256'],
                         'prediction_bytes':pred.stat().st_size,'receipt':str(receipt),'receipt_sha256':digest(receipt),
                         'case_sha256':rec['case_sha256'],'native_shape':shape,'frames':c['frames'],'queries':c['queries']})
                except Exception as e:row['rejections'].append({'id':r.get('id'),'error':str(e)})
            out['producers'].append(row)
        except Exception as e:out['errors'].append({'path':str(p),'phase':'producer','error':str(e)})
    print(json.dumps(out,ensure_ascii=False))
