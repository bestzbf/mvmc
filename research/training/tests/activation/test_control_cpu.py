"""CPU control contracts; all locks live in private temporary files, no GPUs."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import control_common as common
import observe_old_main as observer
import supervisor


class ControlTests(unittest.TestCase):
    def test_all18_locks_are_exclusive_and_inheritable(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths={i:[str(Path(tmp)/(str(i)+'_'+str(n))) for n in range(3)] for i in common.GPUS}
            for row in paths.values():
                for p in row:Path(p).touch()
            with patch.object(common,'locks',lambda i,u:paths[i]):
                leases=common.Leases().acquire()
                try:
                    self.assertEqual(len(leases.fds),18)
                    program="import os,fcntl,json,sys; pairs=json.loads(sys.argv[1]);\nfor path,fd in pairs:\n fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\n a,b=os.fstat(fd),os.stat(path)\n assert (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)\n with open(path,'rb') as fresh:\n  try:fcntl.flock(fresh,fcntl.LOCK_EX|fcntl.LOCK_NB)\n  except BlockingIOError:pass\n  else:raise AssertionError('foreign descriptor obtained held lease')\nprint('18exclusive/inherited')"
                    pairs=[(p,h.fileno()) for p,h in leases.handles]
                    r=subprocess.run([sys.executable,'-c',program,json.dumps(pairs)],pass_fds=leases.fds,capture_output=True,text=True)
                    self.assertEqual(r.returncode,0,r.stderr)
                finally:leases.close()

    def test_real_kernel_pidfd_handle_signal_zero_only(self):
        import signal
        fd=common.open_pidfd(os.getpid())
        try:signal.pidfd_send_signal(fd,0,None,0)
        finally:os.close(fd)

    def test_pid_reuse_never_signals(self):
        record=dict(pid=123,start='old',uid=os.getuid())
        with patch.object(common,'proc',return_value=dict(pid=123,start='new',uid=os.getuid(),state='S')),patch.object(common,'open_pidfd',side_effect=AssertionError('must not open reused PID')):
            common.send_owned(record,15)

    def test_observer_waits_fresh_save_rescans_after_stop_and_preserves_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';old.mkdir();out=root/'new'
            p200=root/'step_00000200.pt';p300=root/'step_00000300.pt';p200.write_bytes(b'checkpoint200');p300.write_bytes(b'checkpoint300')
            owner=dict(pid=11,ppid=1,start='1',uid=os.getuid());launcher=dict(pid=12,ppid=11,start='2',uid=os.getuid());evaluation=dict(pid=13,ppid=11,start='3',uid=os.getuid())
            spec=dict(old_control=str(old),controller=owner,launcher=launcher,run=str(root),latest_step=100,protected_independent_children=[evaluation],job={'physical_gpus':[0,1,2,5,6]})
            def is_alive(r):return r['pid']==13 or not (old/'PAUSE').exists()
            def saves(run):return [p300] if (old/'PAUSE').exists() else [p200]
            def cp(path,world):return dict(path=str(path),step=int(Path(path).stem.split('_')[1]),world_size=world,training_state={},old_stream_global_consumed_ledger=99)
            with patch.object(observer,'inspect',return_value=deepcopy(spec)),patch.object(observer,'alive',side_effect=is_alive),patch.object(observer,'newest',side_effect=saves),patch.object(observer,'checkpoint',side_effect=cp),patch.object(observer,'descendants',return_value=[launcher]),patch.object(common.os,'kill',side_effect=AssertionError('observer must never signal')):
                receipt=observer.execute(spec,out,poll=0)
            self.assertEqual(receipt['pause_requested_at_step'],200)
            self.assertEqual(receipt['source']['step'],300)
            self.assertTrue((old/'PAUSE').exists())
            self.assertEqual((out/'source_pins'/p300.name).read_bytes(),p300.read_bytes())
            self.assertEqual(receipt['protected_independent_children'],[evaluation])

    def test_checkpoint_schema_and_old_stream_ledger(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'step.pt';state=dict(world_size=5,batch_size=1,gradient_accumulation=1,optimizer_updates=200,dataset_size=101,data_epoch=2,batches_consumed=18,load_balancing={'historical_padding_sample_offset':4})
            data=dict(architecture='multicamera4rc',step=200,model={},config={},rank_rng=[{}]*5,optimizer={'state':{0:{'step':torch.tensor(200.)}},'param_groups':[{'name':'heads'}]},training_state=state)
            torch.save(data,path);value=common.checkpoint(path,5)
            self.assertEqual(value['old_stream_global_consumed_ledger'],4+2*105+18*5)
            data['training_state']['optimizer_updates']=199;torch.save(data,path)
            with self.assertRaises(ValueError):common.checkpoint(path,5)

    def test_owned_refresh_deduplicates_and_foreign_pid_reuse_is_not_owned(self):
        obj=supervisor.Supervisor({'physical_gpus':list(common.GPUS),'control_dir':'/tmp/unexecuted_supervisor_test'})
        identity=dict(pid=123,start='old',uid=os.getuid());obj.child=object();obj.child_identity=identity;obj.known=[identity]
        with patch.object(supervisor,'alive',return_value=True),patch.object(supervisor,'descendants',return_value=[identity]):
            for _ in range(100):obj.refresh_owned()
        self.assertEqual(len(obj.known),1)
        with patch.object(supervisor,'gpu_snapshot',return_value=({},[dict(uuid=common.GPUS[7],pid=123)])),patch.object(supervisor,'alive',return_value=False):
            self.assertEqual(obj.owned_gpu_remains(),[])

    def test_supervisor_two_phase_gate_locations_and_source_handshake(self):
        obj=supervisor.Supervisor({'physical_gpus':list(common.GPUS),'control_dir':'/tmp/unexecuted_supervisor_test'})
        with tempfile.TemporaryDirectory() as tmp:
            proof=Path(tmp)
            self.assertFalse(obj.gate_ready(proof,140400))
            for rank in range(6):
                for folder in ('startup','updates/step_00140401'):
                    p=proof/folder/f'rank_{rank:02d}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
            self.assertTrue(obj.gate_ready(proof,140400))
        self.assertFalse(obj.promoted)

if __name__=='__main__':unittest.main(verbosity=2)
