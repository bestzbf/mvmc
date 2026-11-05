"""CPU-only mocked scheduling/ownership tests. Never invokes CUDA or a model."""
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('fullrunner',Path(__file__).with_name('full_runner.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Process:
    def __init__(self,pid=900001,code=None):self.pid=pid;self.returncode=code;self.signals=[]
    def poll(self):return self.returncode
    def wait(self,timeout=None):
        if self.returncode is None:self.returncode=-15
        return self.returncode
    def send_signal(self,sig):self.signals.append(sig);self.returncode=-sig


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.runners=[]
    def tearDown(self):
        for r in self.runners:
            for a in r.active.values():
                a['stream'].close()
                if a.get('lock'):a['lock'].close()
            r.active.clear();r.singleton.close()
        self.temp.cleanup()
    def job(self,key,kind='cpu',method='A',**extra):
        j={'id':key,'kind':kind,'method':method,'dataset':'test','argv':[sys.executable,'-c','pass'],
           'env':{},'expected_outputs':[],'timeout_seconds':3600}
        if kind=='gpu':
            d=self.root/key;d.mkdir(exist_ok=True);manifest=d/'manifest.json'
            case={'id':key+'/case','input':'input.npz','input_sha256':'b'*64}
            manifest.write_text(json.dumps({'cases':[case]}))
            j.update(argv=[sys.executable,'adapter.py','--manifest',str(manifest),'--resume'],
                     expected_outputs=[str(d/'predictions/predictions.json')],case_count=1)
        j.update(extra);return j
    def runner(self,jobs,sealed=True,status=None):
        plan=self.root/'queue_plan.json';plan.write_text(json.dumps({'allowed_physical_gpus':[3,4],'jobs':jobs,'sealed':sealed}))
        if status:
            (self.root/'status').mkdir(exist_ok=True)
            for s in status:(self.root/'status'/(s['job_id']+'.json')).write_text(json.dumps(s))
        r=m.Runner(plan);self.runners.append(r);return r
    def state(self,j,state='failed',attempt=1):
        return {'job_id':j['id'],'job_hash':m.digest(j),'state':state,'kind':j['kind'],
                'method':j['method'],'dataset':j['dataset'],'attempt_count':attempt}
    def item(self,r,j,attempt=1,termination='job_timeout'):
        s=self.state(j,'running',attempt);s.update(attempt_history=[],receipts_before_attempt=[],
          receipt_origin_ids=[],started_epoch=time.time(),pid_starttime_ticks=100,pid=900001,
          termination_reason=termination)
        a={'job':j,'state':s,'process':Process(code=-15),'stream':io.BytesIO(),'lock':None,
           'start_monotonic':time.monotonic()-2,'term_at':None}
        r.states[j['id']]=s;r.active[j['id']]=a;return a
    def receipt(self,j,valid=True,payload=True):
        d=Path(j['expected_outputs'][0]).parent;d.mkdir(parents=True,exist_ok=True)
        manifest=Path(j['argv'][j['argv'].index('--manifest')+1]);case=json.loads(manifest.read_text())['cases'][0]
        state={'schema':'released-baseline-run-v1','manifest_sha256':m.hashlib.sha256(manifest.read_bytes()).hexdigest(),'method':j['method']}
        (d/'run-state.json').write_text(json.dumps(state))
        record={'id':case['id'],'file':'000000.npz','sha256':'a'*64,'input_sha256':case['input_sha256'],'model':j['method']}
        if payload:(d/'000000.npz').write_bytes(b'payload')
        elif (d/'000000.npz').exists():(d/'000000.npz').unlink()
        receipt={'schema':'released-prediction-case-v1','run_state_sha256':m.canonical(state),'case_sha256':m.canonical(case),'record':record}
        if not valid:receipt={'record':{'id':case['id']}}
        (d/'000000.receipt.json').write_text(json.dumps(receipt));return receipt

    def test_no_global_deadline_after_many_hours(self):
        j=self.job('pending');r=self.runner([j]);self.assertFalse(hasattr(r,'deadline'))
        with patch.object(m.time,'time',return_value=time.time()+100*3600):
            self.assertEqual(r.outcome(),(False,False,'waiting_for_ready_jobs_or_resources'))
    def test_reject_old_window_fields(self):
        r=self.runner([self.job('j')]);p=json.loads(r.plan_path.read_text());p['duration_hours']=10;r.plan_path.write_text(json.dumps(p))
        self.assertRaises(ValueError,r.reload)
    def test_uuid_and_cpu_masks_and_threads(self):
        j=self.job('j');a=m.child_environment(j,3);b=m.child_environment(j,None)
        self.assertEqual(a['CUDA_VISIBLE_DEVICES'],m.GPU_UUIDS[3]);self.assertEqual(b['CUDA_VISIBLE_DEVICES'],'')
        self.assertTrue(all(a[k]=='4' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']))
        self.assertEqual(a['PYTHONUNBUFFERED'],'1')
    def test_foreign_gpu_is_rejected(self):
        def output(argv,**kw):
            if '--query-gpu=uuid' in argv:return m.GPU_UUIDS[4]
            if '--query-compute-apps=gpu_uuid,pid' in argv:return m.GPU_UUIDS[4]+', 31275\n'
            raise AssertionError('Should not dispatch foreign GPU')
        with patch.object(m.subprocess,'check_output',side_effect=output):self.assertFalse(m.available_gpu(4))
    def test_failed_job_does_not_end_other_ready_jobs(self):
        a,b=self.job('a'),self.job('b');r=self.runner([a,b],status=[self.state(a)])
        launched=[]
        with patch.object(r,'start_job',side_effect=lambda j,*_:launched.append(j['id'])):r.admit()
        self.assertEqual(launched,['b']);self.assertFalse(r.outcome()[0])
        r.states['b']=self.state(b,'complete');self.assertEqual(r.outcome(),(True,False,'needs_attention'))
    def test_complete_is_only_success_and_survives_cleanup(self):
        a=self.job('a');r=self.runner([a],status=[self.state(a,'complete')]);r.cleanup()
        x=r.publish();self.assertTrue(x['success']);self.assertEqual(x['overall_status'],'complete')
    def test_active_and_done_hashes_immutable(self):
        a=self.job('a');r=self.runner([a],status=[self.state(a,'complete')]);p=json.loads(r.plan_path.read_text());p['jobs'][0]['argv'].append('changed');r.plan_path.write_text(json.dumps(p))
        self.assertRaises(ValueError,r.reload)
    def test_old_running_refused_without_spawn_or_signal(self):
        a=self.job('a');p=self.root/'queue_plan.json';p.write_text(json.dumps({'allowed_physical_gpus':[3,4],'sealed':True,'jobs':[a]}));(self.root/'status').mkdir();(self.root/'status/a.json').write_text(json.dumps(self.state(a,'running')))
        with patch.object(m.subprocess,'Popen') as spawn,patch.object(m.os,'killpg') as kill:
            self.assertRaises(RuntimeError,m.Runner,p);spawn.assert_not_called();kill.assert_not_called()
    def test_timeout_retry_only_progress_once(self):
        j=self.job('g','gpu');r=self.runner([j]);self.receipt(j);a=self.item(r,j);r.finish_attempt(a,-15)
        self.assertEqual(r.states['g']['state'],'retry_wait');self.assertEqual(len(r.states['g']['attempt_history']),1)
        a=self.item(r,j,attempt=2);r.finish_attempt(a,-15);self.assertEqual(r.states['g']['state'],'interrupted')
    def test_no_progress_or_model_error_never_retries(self):
        j=self.job('g','gpu');r=self.runner([j]);a=self.item(r,j);r.finish_attempt(a,-15);self.assertEqual(r.states['g']['state'],'interrupted')
        self.receipt(j);a=self.item(r,j,attempt=2,termination=None);r.finish_attempt(a,1);self.assertEqual(r.states['g']['state'],'failed')
    def test_fake_or_missing_payload_receipt_not_progress(self):
        j=self.job('g','gpu');self.receipt(j,valid=False);self.assertEqual(m.receipt_snapshot(j)['count'],0)
        self.receipt(j,payload=False);self.assertEqual(m.receipt_snapshot(j)['count'],0)
        self.receipt(j);self.assertEqual(m.receipt_snapshot(j)['count'],1)
        p=Path(j['expected_outputs'][0]).parent/'run-state.json';s=json.loads(p.read_text());s['method']='tampered';p.write_text(json.dumps(s));self.assertEqual(m.receipt_snapshot(j)['count'],0)
    def test_retry_tail_and_retry_dependencies_not_blocked(self):
        retry=self.job('retry','gpu');fresh=self.job('fresh','gpu',method='B');score=self.job('score',depends_on=['retry']);r=self.runner([retry,fresh,score],status=[self.state(retry,'retry_wait')]);launched=[]
        def start(j,index=None,handle=None):launched.append(j['id']);r.last_dispatch[j['method']]=10
        with patch.object(r,'start_job',side_effect=start),patch.object(r,'acquire',return_value=io.BytesIO()),patch.object(m.shutil,'disk_usage',return_value=m.shutil._ntuple_diskusage(100,0,m.MIN_FREE_BYTES+1)):r.admit()
        self.assertEqual(launched[:2],['fresh','retry']);self.assertNotIn('score',r.states)
    def test_method_fairness_and_gpu_affinity(self):
        a=self.job('a','gpu',method='A',physical_gpu_affinity=[3]);b=self.job('b','gpu',method='A');c=self.job('c','gpu',method='B');r=self.runner([a,b,c]);launched=[]
        def start(j,index=None,handle=None):launched.append((j['id'],index));r.last_dispatch[j['method']]=10
        with patch.object(r,'start_job',side_effect=start),patch.object(r,'acquire',return_value=io.BytesIO()),patch.object(m.shutil,'disk_usage',return_value=m.shutil._ntuple_diskusage(100,0,m.MIN_FREE_BYTES+1)):r.admit()
        self.assertEqual(launched,[('a',3),('c',4)])
    def test_low_disk_pauses_gpu_but_cpu_continues(self):
        a=self.job('a','gpu');b=self.job('b');r=self.runner([a,b]);launched=[]
        with patch.object(r,'start_job',side_effect=lambda j,*_:launched.append(j['id'])),patch.object(r,'acquire') as acquire,patch.object(m.shutil,'disk_usage',return_value=m.shutil._ntuple_diskusage(100,99,1)):r.admit()
        self.assertEqual(launched,['b']);acquire.assert_not_called();self.assertEqual(r.outcome()[2],'paused_low_disk')
    def test_gate_registration_and_flock_fd_before_release(self):
        j=self.job('g','gpu');r=self.runner([j]);hold=(self.root/'hold').open('a+');p=Process();events=[]
        def identity(pid):return {'pid':pid,'starttime_ticks':100,'pgid':pid,'sid':pid}
        def writing(state):
            events.append(state['state'])
            if state['state']=='running':self.assertIn('g',r.active)
        with patch.object(m,'process_identity',side_effect=identity),patch.object(m.subprocess,'Popen',return_value=p) as spawn,patch.object(r,'write_state',side_effect=writing),patch.object(m.os,'write',side_effect=lambda fd,b:events.append('gate_release')):
            r.start_job(j,3,hold)
        self.assertEqual(events,['preparing','running','gate_release']);kw=spawn.call_args.kwargs
        self.assertEqual(kw['env']['BASELINE_GPU_LOCK_FD'],str(hold.fileno()));self.assertIn(hold.fileno(),kw['pass_fds']);self.assertTrue(kw['start_new_session'])
    def test_running_save_failure_remains_registered_for_cleanup(self):
        j=self.job('a');r=self.runner([j]);p=Process();identity=lambda pid:{'pid':pid,'starttime_ticks':100,'pgid':pid,'sid':pid}
        def writing(state):
            if state['state']=='running':raise OSError('simulated disk error')
        with patch.object(m,'process_identity',side_effect=identity),patch.object(m.subprocess,'Popen',return_value=p),patch.object(r,'write_state',side_effect=writing),patch.object(m,'owned_group_members',return_value=[]):
            self.assertRaises(OSError,r.start_job,j);self.assertIn('a',r.active)
            with patch.object(r,'signal_owned',side_effect=lambda item,sig:p.signals.append(sig)):r.cleanup()
        self.assertFalse(r.active);self.assertIn(signal.SIGTERM,p.signals)
    def test_signal_refuses_reused_pid(self):
        j=self.job('g','gpu');r=self.runner([j]);a=self.item(r,j)
        with patch.object(m,'process_identity',return_value={'starttime_ticks':999,'pgid':900001,'sid':900001}),patch.object(m.os,'killpg') as kill:
            self.assertRaises(RuntimeError,r.signal_owned,a,signal.SIGTERM);kill.assert_not_called()
    def test_cleanup_io_error_does_not_skip_other_child(self):
        a,b=self.job('a'),self.job('b');r=self.runner([a,b]);ia=self.item(r,a);ib=self.item(r,b);signals=[]
        with patch.object(r,'signal_owned',side_effect=lambda item,sig:signals.append(item['job']['id'])),patch.object(m,'owned_group_members',return_value=[]),patch.object(r,'write_state',side_effect=OSError('disk failure')):
            r.cleanup()
        self.assertEqual(signals,['a','b']);self.assertFalse(r.active);self.assertIsNotNone(r.runner_error)
    def test_native_zero_exit_requires_complete_bound_index(self):
        j=self.job('g','gpu');r=self.runner([j]);receipt=self.receipt(j);d=Path(j['expected_outputs'][0]);d.write_text(json.dumps({'schema':'released-tracking-predictions-v1','run_state_sha256':receipt['run_state_sha256'],'cases':[receipt['record']]}));a=self.item(r,j,termination=None);r.finish_attempt(a,0);self.assertEqual(r.states['g']['state'],'complete')
    def test_closed_gate_never_executes_native(self):
        read_fd,write_fd=os.pipe();os.close(write_fd)
        try:
            with patch.object(m.os,'execvpe') as execute:self.assertRaises(SystemExit,m.gated_child,read_fd,[sys.executable]);execute.assert_not_called()
        finally:
            try:os.close(read_fd)
            except OSError:pass


if __name__=='__main__':unittest.main(verbosity=2)
