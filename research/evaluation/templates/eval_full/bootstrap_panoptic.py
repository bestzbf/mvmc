"""Start with the two exact missing Panoptic cases, reusing validated five-case runs."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path('@@EVAL_FULL_ROOT@@')
OLD=ROOT.parent/'eval_10h_20261104'
ROOT.mkdir(parents=True,exist_ok=True)
oldplan=json.loads((OLD/'queue_plan.json').read_text())
byid={j['id']:j for j in oldplan['jobs']}
jobs=[]

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(16<<20):h.update(b)
    return h.hexdigest()
def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def replace(argv,key,value):
    argv=list(argv);argv[argv.index(key)+1]=str(value);return argv

for key in ['arrow_mv','trackeverything_mv']:
    sourcejob=byid['panoptic_'+key+'_predict']
    oldspec=read(OLD/'score_specs'/('panoptic_'+key+'_score.json'))
    originalpath=Path(oldspec['source_manifest']);original=read(originalpath)
    assert sha(originalpath)==oldspec['source_manifest_sha256'] and len(original['cases'])==6
    have={i for s in oldspec['sources'] for i in s['case_ids']}
    missing=[(i,c) for i,c in enumerate(original['cases']) if c['id'] not in have]
    assert len(missing)==1 and missing[0][1]['id']=='@@BENCHMARK_CASE_295F46DA82@@'
    index,case=missing[0];folder=ROOT/'delta'/('panoptic_'+key)
    (folder/'inputs').mkdir(parents=True,exist_ok=True)
    target=folder/'inputs'/f'{index:06d}.npz'
    if not target.exists():
        try:os.link(originalpath.parent/case['input'],target)
        except OSError:shutil.copy2(originalpath.parent/case['input'],target)
    assert sha(target)==case['input_sha256']
    manifest=copy.deepcopy(original);case=copy.deepcopy(case);case['input']=str(target.relative_to(folder));manifest['cases']=[case]
    manifest['diagnostic_subset']={'source_manifest':str(originalpath),'source_manifest_sha256':sha(originalpath),
         'selection':'Exact missing full-coverage case only; previous five cases retained unchanged',
         'selected_source_indices':[index],'formal_score_admitted':False,'all_original_frames_and_queries_preserved':True}
    manifestpath=folder/'manifest.json';save(manifestpath,manifest);prediction=folder/'predictions'
    argv=replace(sourcejob['argv'],'--manifest',manifestpath);argv=replace(argv,'--output',prediction)
    ident='panoptic_'+key+'_fill_predict'
    jobs.append(dict(id=ident,kind='gpu',method=sourcejob['method'],dataset='panoptic',argv=argv,env=sourcejob['env'],
                     timeout_seconds=3600,expected_outputs=[str(prediction/'predictions.json')],case_ids=[case['id']],case_count=1,
                     manifest=str(manifestpath),manifest_sha256=sha(manifestpath),prediction_root=str(prediction),priority=0))
    spec=copy.deepcopy(oldspec);spec['manifest']=str(originalpath);spec['manifest_sha256']=sha(originalpath)
    spec['sources'].append(dict(kind='prediction_index',path=str(prediction/'predictions.json'),manifest=str(manifestpath),
                               manifest_sha256=sha(manifestpath),case_ids=[case['id']]))
    spec['selection_policy']='Complete original frozen6, reuse original5 and only infer missing football'
    specpath=ROOT/'score_specs'/('panoptic_'+key+'_full6.json');save(specpath,spec)
    raw=ROOT/'scores/raw'/('panoptic_'+key+'_full6.json');output=ROOT/'scores'/('panoptic_'+key+'_full6.json')
    engine=OLD/'tools/mv_subset_score.py'
    argvscore=['@@BENCHMARK_ROOT@@/runtime/eval_env/bin/python',str(engine),'--spec',str(specpath),'--output',str(raw)]
    recipe=dict(schema='frozen-full-score-recipe-v1',method=sourcejob['method'],dataset='panoptic',source_manifest=str(originalpath),
                source_manifest_sha256=sha(originalpath),expected_case_ids=[c['id'] for c in original['cases']],expected_cases=6,
                score_argv=argvscore,raw_output=str(raw))
    recipepath=ROOT/'score_recipes'/('panoptic_'+key+'_full6.json');save(recipepath,recipe)
    jobs.append(dict(id='panoptic_'+key+'_full6_score',kind='cpu',method=sourcejob['method'],dataset='panoptic',
                     argv=['@@BENCHMARK_ROOT@@/runtime/eval_env/bin/python',str(ROOT/'score_full.py'),'--recipe',str(recipepath),'--output',str(output)],
                     env={},timeout_seconds=3600,depends_on=[ident],expected_outputs=[str(output)],score_path=str(output),case_count=6))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    check="import sys;sys.path.insert(0,'@@RELEASED_BASELINE_HARNESS_ROOT@@');from multicamera4rc.benchmarks.released_data import load_manifest;m=load_manifest(sys.argv[1],inference_only=True);print('Frozen input contract passed',len(m['cases']))"
    subprocess.run([argv[0],'-c',check,str(manifestpath)],check=True,env=env)

path=ROOT/'queue_plan.json';assert not path.exists(),'Never overwrite an existing full queue'
save(path,dict(schema='full-evaluation-queue-v1',allowed_physical_gpus=[3,4],sealed=False,jobs=jobs,
               completion_policy='Continue through full frozen plans; no inherited10h deadline',source_subsets_reused=True))
print(json.dumps({'bootstrap_jobs':len(jobs),'new_inference_cases':2,'score_targets':{'panoptic':6},'global_deadline':None}))
