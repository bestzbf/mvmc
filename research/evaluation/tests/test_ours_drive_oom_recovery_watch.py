import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('oomwatch',HERE/'ours_drive_oom_recovery_watch.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

class RecoveryGuardTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  (self.root/'scripts').mkdir();(self.root/'planning').mkdir();(self.root/'status').mkdir()
  shutil.copy2(HERE.parent/'full_runner.py', self.root/'full_runner.py')
  for name in ('prepare_missing.py', 'route_ours_drive_lock.py'):
   shutil.copy2(HERE/name, self.root/'scripts'/name)
  cases=[]
  for i in range(50):
   f=self.root/'canonical_inputs'/f'{i:03d}.npz';f.parent.mkdir(exist_ok=True);f.write_bytes(('case'+str(i)).encode())
   cases.append({'id':f'drivetrack/case{i:03d}.npz','input':str(f),'input_sha256':w.sha(f),'frames':i+10,'queries':12,'notes':'unchanged'})
  self.manifest={'schema':'fixture','protocol':{'input_condition':'rgb','release_binding':{'status':'verified_inventory_match','coverage_verified':True}},'cases':cases}
  self.canonical=self.root/'canonical.json';w.save(self.canonical,self.manifest)
  self.state_base={'schema':'released-prediction-run-v1','method':w.METHOD,'runtime_versions':{'torch':'2.11.0+cu128'},'target_time_chunk':8,'checkpoint':{'sha256':'checkpoint','amp':'bf16'}}
  sig=w.canonical(self.state_base)
  self.inventory={'rows':[{'id':w.ROW,'preferred_group_signature_sha256':sig}], 'canonical_panels':{w.PANEL:{'canonical_manifest':str(self.canonical),'canonical_manifest_sha256':w.sha(self.canonical)}}}
  w.save(self.root/'planning/full_inventory.json',self.inventory)
  self.jobs=[]
  for i,key in enumerate(w.ORIGINAL_IDS):
   base=self.root/'prepared'/f'{i:03d}';base.mkdir(parents=True);selected=copy.deepcopy(cases[20+5*i:25+5*i]);mf=base/'manifest.json'
   m=copy.deepcopy(self.manifest);m['cases']=selected;w.save(mf,m)
   out=base/'predictions';out.mkdir();state=copy.deepcopy(self.state_base);state['manifest_sha256']=w.sha(mf);w.save(out/'run-state.json',state)
   native=['python','fixture_native.py','predict','--manifest',str(mf),'--output',str(out),'--amp','bf16','--target-time-chunk','8','--resume','--checkpoint','fixture_checkpoint.pt']
   argv=['python',str(self.root/'scripts/route_ours_drive_lock.py'),'--routing-receipt',str(base/'routing_receipt.json'),'--',*native[2:]]
   job={'id':key,'kind':'gpu','method':w.METHOD,'dataset':w.PANEL,'argv':argv,'env':{},'case_count':5,'case_ids':[c['id'] for c in selected],'expected_outputs':[str(out/'predictions.json')],'native_predict_argv':native,'manifest':str(mf),'manifest_sha256':w.sha(mf)}
   self.jobs.append(job)
  self.plan={'allowed_physical_gpus':[3,4],'sealed':True,'jobs':self.jobs};w.save(self.root/'queue_plan.json',self.plan)
  self.proof={'allocator_settings':'expandable_segments:True','parse_supported':True,'cuda_initialized_before':False,'cuda_initialized_after':False,'CUDA_VISIBLE_DEVICES':'','torch_version':'2.11.0+cu128','source_wrapper_sha256':w.SOURCE_SHA}
  self.proof_path=self.root/'proof.json';w.save(self.proof_path,self.proof)
  self.fail(0)
 def tearDown(self):self.tmp.cleanup()
 def fail(self,index,text='Benchmark failed: CUDA out of memory. Tried to allocate 2.00 GiB.',state='failed'):
  job=self.jobs[index];log=self.root/'logs'/f'{index}.log';log.parent.mkdir(exist_ok=True);log.write_text(text)
  data={'job_id':job['id'],'job_hash':w.digest(job),'state':state,'log':str(log),'finished_at_utc':'2000-01-01T00:00:00Z'}
  w.save(self.root/'status'/(job['id']+'.json'),data)
 def watcher(self):return w.Watcher(self.root,self.proof_path,manifest_validator=lambda p:None)
 def receipt(self,batch,index):
  job=self.jobs[batch];m=w.read(job['manifest']);case=m['cases'][index];out=Path(job['expected_outputs'][0]).parent
  payload=out/f'{index:06d}.npz';payload.write_bytes(b'valid prediction')
  record={'id':case['id'],'file':payload.name,'sha256':w.sha(payload),'input_sha256':case['input_sha256'],'model':w.METHOD}
  state=w.read(out/'run-state.json');receipt={'schema':'released-prediction-case-v1','record':record,'run_state_sha256':w.canonical(state),'case_sha256':w.canonical(case)}
  w.save(out/f'{index:06d}.receipt.json',receipt)
  return payload
 def test_prepare_changes_only_IO_allocator_and_frozen_case_selection(self):
  before=w.sha(self.root/'queue_plan.json');watcher=self.watcher();result=watcher.step(False)
  self.assertEqual(result['recovery_gpu_jobs'],5);self.assertEqual(before,w.sha(self.root/'queue_plan.json'))
  for info in watcher.registry['recoveries'].values():
   for job in info['jobs']:
    self.assertEqual(job['env'],w.ALLOCATOR);self.assertEqual(job['case_count'],1);self.assertEqual(job['physical_gpu_affinity'],[3,4]);self.assertEqual(job['timeout_seconds'],3600)
    self.assertFalse(job['allow_timeout_retry']);self.assertEqual(job['oom_recovery_of'],self.jobs[0]['id'])
    c=w.read(job['manifest'])['cases'][0];old=next(ca for ca in self.manifest['cases'] if ca['id']==c['id'])
    self.assertEqual({k:v for k,v in c.items() if k!='input'},{k:v for k,v in old.items() if k!='input'})
 def test_partial_valid_receipt_is_not_reinferred(self):
  self.receipt(0,0);watcher=self.watcher();watcher.step(False);info=next(iter(watcher.registry['recoveries'].values()))
  self.assertEqual(info['valid_original_case_ids'],[self.jobs[0]['case_ids'][0]])
  self.assertEqual(len(info['jobs']),4);self.assertNotIn(self.jobs[0]['case_ids'][0],[j['case_ids'][0] for j in info['jobs']])
 def test_non_OOM_and_nonterminal_cases_are_not_retried(self):
  self.fail(0,'network error');self.assertEqual(self.watcher().step(False)['recovery_gpu_jobs'],0)
  self.fail(0,state='running');self.assertEqual(self.watcher().step(False)['recovery_gpu_jobs'],0)
 def test_corrupt_receipt_payload_is_rejected(self):
  path=self.receipt(0,0);path.write_bytes(b'corrupt')
  with self.assertRaisesRegex(ValueError,'payload hash'):self.watcher().step(False)
 def test_wrong_model_or_status_signature_is_rejected(self):
  p=Path(self.jobs[0]['expected_outputs'][0]).parent/'run-state.json';d=w.read(p);d['target_time_chunk']=4;w.save(p,d)
  with self.assertRaisesRegex(ValueError,'signature'):self.watcher().step(False)
 def test_bad_allocator_proof_is_rejected(self):
  d=copy.deepcopy(self.proof);d['cuda_initialized_after']=True;w.save(self.proof_path,d)
  with self.assertRaisesRegex(ValueError,'CPU-only'):self.watcher()
 def test_append_is_idempotent_preserves_originals_and_holds_queue_open(self):
  hashes={j['id']:w.digest(j) for j in self.jobs};watcher=self.watcher();watcher.step(True)
  plan=w.read(self.root/'queue_plan.json');self.assertEqual(len(plan['jobs']),11);self.assertFalse(plan['sealed']);self.assertIn(w.HOLD,plan['completion_holds'])
  self.assertEqual({j['id']:w.digest(j) for j in plan['jobs'][:6]},hashes)
  watcher=self.watcher();watcher.step(True);self.assertEqual(len(w.read(self.root/'queue_plan.json')['jobs']),11)
 def test_registered_job_removal_or_mutation_is_rejected(self):
  watcher=self.watcher();watcher.step(True);plan=w.read(self.root/'queue_plan.json');plan['jobs'][-1]['env']={};w.save(self.root/'queue_plan.json',plan)
  with self.assertRaisesRegex(ValueError,'registered recovery'):self.watcher().step(True)
 def score_fixture(self):
  watcher=self.watcher();watcher.step(True)
  for job in [*self.jobs,*[j for v in watcher.registry['recoveries'].values() for j in v['jobs']]]:
   if job['id']==self.jobs[0]['id']:continue
   w.save(self.root/'status'/(job['id']+'.json'),{'job_id':job['id'],'job_hash':w.digest(job),'state':'complete'})
  old={'row_id':w.ROW,'score_path':str(self.root/'old_score.json'),'score_job_id':'old_cpu'}
  new=dict(old,score_path=str(self.root/'new_score.json'),score_job_id='new_cpu')
  w.save(self.root/'full_targets.json',{'rows':[old]})
  intent={'job':{'id':'new_cpu','kind':'cpu','method':w.METHOD,'dataset':w.PANEL,'argv':['python','fixture_score.py'],'depends_on':[],'expected_outputs':[str(self.root/'new_score.json')]},'target_update':new,'target_CAS':{'row_id':w.ROW,'expected_old_row_canonical_sha256':w.canonical(old),'expected_old_score_path':old['score_path'],'expected_old_score_job_id':'old_cpu'}}
  watcher.builder=SimpleNamespace(canonical=w.canonical)
  return watcher,intent
 def test_score_target_CAS_and_append_are_idempotent_and_preserve_failure(self):
  watcher,intent=self.score_fixture();failure=w.read(self.root/'status'/(self.jobs[0]['id']+'.json'))
  watcher.apply_score(intent);first=w.sha(self.root/'full_targets.json');watcher.apply_score(intent)
  self.assertEqual(len(w.read(self.root/'queue_plan.json')['jobs']),12)
  self.assertEqual(first,w.sha(self.root/'full_targets.json'));self.assertTrue(w.read(self.root/'queue_plan.json')['sealed'])
  self.assertEqual(w.read(self.root/'status'/(self.jobs[0]['id']+'.json')),failure)
 def test_score_target_concurrent_generation_is_not_overwritten(self):
  watcher,intent=self.score_fixture();d=w.read(self.root/'full_targets.json');d['rows'][0]['score_path']='another_generation';w.save(self.root/'full_targets.json',d)
  before=w.sha(self.root/'queue_plan.json')
  with self.assertRaisesRegex(ValueError,'CAS failed'):watcher.apply_score(intent)
  self.assertEqual(before,w.sha(self.root/'queue_plan.json'));self.assertEqual(w.read(self.root/'full_targets.json'),d)
 def test_active_source_blocks_score_publication(self):
  watcher,intent=self.score_fixture();job=self.jobs[1];w.save(self.root/'status'/(job['id']+'.json'),{'job_id':job['id'],'job_hash':w.digest(job),'state':'running'})
  before=w.sha(self.root/'queue_plan.json')
  with self.assertRaisesRegex(ValueError,'Source became active'):watcher.apply_score(intent)
  self.assertEqual(before,w.sha(self.root/'queue_plan.json'))

if __name__=='__main__':unittest.main(verbosity=2)
