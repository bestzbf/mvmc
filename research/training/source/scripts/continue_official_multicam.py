"""Admit audited official sources at a main-training boundary, preserving LR.

Only the main controller is replaced; its running launcher/ranks finish their
original segment. Experiments retain their frozen code, data and GPU leases.
"""
from contextlib import ExitStack
from copy import deepcopy
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import traceback

from multicamera4rc.datasets.mixture import read_data_config
from scripts.build_official_multicam_mixture import build_mixture
from scripts.continue_extended_training import next_segment_end, quality_decision
from scripts.continue_quality_recovery import RecoveryController, digest, training_argv, validate_checkpoint
from scripts.continue_syn4d_training import verify_state
from scripts.cutover_native_motion_stereo import proc, write_json
from scripts.migrate_dynamic_validation import (ExpandedDynamicController, build_validation_contract,
    require_validation_contract, require_fixed_training_state)
from scripts.migrate_static_validation import training_ranks, require_training_child


def verify_mixture(old_path, new_path, panoptic_audit):
    old,new=read_data_config(old_path),read_data_config(new_path)
    additions=new['sources'][-2:]
    if [s['adapter'] for s in additions] != ['mv_kubric','panoptic']:
        raise ValueError('mixture must append exactly the two admitted sources')
    expected=build_mixture(old,additions[0]['root'],additions[1]['root'],panoptic_audit)
    # A new run gets its own exclusion log; every sampling/preprocessing choice
    # besides the declared source additions and weights must remain identical.
    expected['invalid_log']=new.get('invalid_log')
    if expected != new:
        raise ValueError('new mixture changed more than the admitted source additions')
    return new


def verify_admission(job):
    for path,sha in job['official_evidence'].items():
        if digest(path)!=sha:
            raise ValueError('official admission evidence changed: '+path)
    loss_only = job.get('admission_kind') == 'cross_camera_only'
    if loss_only:
        if (job['official_data_config'] != job['previous']['data_config']
                or not job.get('cross_camera_loss_config')):
            raise ValueError('loss-only admission must preserve the exact data configuration')
    else:
        proof=json.loads(Path(job['official_admission']).read_text())
        from multicamera4rc.datasets.mv_kubric_inventory import expected_train_ids
        manifest=json.loads(Path(proof['conversion_manifest']).read_text())
        expected=expected_train_ids(manifest)
        if (manifest.get('status')!='complete' or manifest.get('converted_scenes')!=len(expected)
                or manifest.get('official_train_scene_ids')!=expected
                or not proof.get('passed') or proof.get('checked_scenes')!=len(expected)
                or proof.get('mv_loader_cases')!=35 or proof.get('panoptic_captures')!=81
                or proof.get('config_sha256')!=digest(job['official_data_config'])
                or proof.get('conversion_manifest_sha256')!=digest(proof['conversion_manifest'])):
            raise ValueError('official dataset admission is incomplete or stale')
    equivalence=json.loads(Path(job['runtime_equivalence']).read_text())
    if not equivalence.get('passed'):
        raise ValueError('original-model default computation equivalence did not pass')
    if not json.loads(Path(job['tests_proof']).read_text()).get('passed'):
        raise ValueError('integration regression tests did not pass')
    if job.get('cross_camera_loss_config'):
        if not json.loads(Path(job['cross_camera_tests_proof']).read_text()).get('passed'):
            raise ValueError('cross-camera regression tests did not pass')
    if loss_only:
        return read_data_config(job['official_data_config'])
    pan=json.loads(Path(job['panoptic_audit']).read_text())
    return verify_mixture(job['previous']['data_config'],job['official_data_config'],pan)


def cross_camera_loss_config(original, weight):
    """One explicit additive intervention; preserve the final ablation recipe."""
    from multicamera4rc.losses import MultiCamera4RCLoss
    old = MultiCamera4RCLoss(**original).to_config()
    if old.get('cross_camera_3d_weight', 0) != 0:
        raise ValueError('cross-camera loss is already active; refusing a second admission')
    new = MultiCamera4RCLoss(**dict(old, cross_camera_3d_weight=weight)).to_config()
    if new.get('cross_camera_3d_weight', 0) <= 0:
        raise ValueError('cross-camera admission requires a positive weight')
    return new


def require_cross_camera_loss_config(job, original):
    path = job.get('cross_camera_loss_config')
    if not path:
        return None
    new = json.loads(Path(path).read_text())
    if new != cross_camera_loss_config(original, job['cross_camera_3d_weight']):
        raise ValueError('cross-camera admission changed an existing loss setting')
    return new


def owned_main_tree(previous, identity):
    current=proc(identity['pid'])
    if (not current or current['uid']!=os.getuid() or current['state']=='Z'
            or any(current[k]!=identity[k] for k in ('pid','start','cmd'))
            or current['pid']!=previous['pid']):
        raise RuntimeError('recorded main controller identity changed')
    args=shlex.split(current['cmd'])
    # Legacy recovery uses --control-dir; ordinary continuations use --controller.
    recorded=(('--controller' in args and args[args.index('--controller')+1]==str(Path(previous['control_dir'])/'job.json'))
        or ('--control-dir' in args and args[args.index('--control-dir')+1]==previous['control_dir']))
    if (not recorded and 'scripts.intrinsics_structure_trial' in args and '--job' in args
            and previous.get('intrinsics_structure_trial')):
        trial_root=Path(previous['intrinsics_structure_trial']).parent
        recorded=(Path(args[args.index('--job')+1])==trial_root/'job.json'
                  and Path(previous['control_dir']).parent==trial_root)
    if not recorded:
        raise RuntimeError('PID is not the recorded main-training controller')
    rows=[proc(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()]
    tree={current['pid']:current}
    while True:
        added=[r for r in rows if r and r['ppid'] in tree and r['pid'] not in tree]
        if not added:break
        tree.update({r['pid']:r for r in added})
    if any(r['uid']!=os.getuid() for r in tree.values()):
        raise RuntimeError('main tree contains another user')
    return tree


def official_training_argv(job, reference, run, end):
    first=reference['data_config']!=job['official_data_config']
    allowed={job['previous']['data_config'],job['official_data_config']}
    if reference['data_config'] not in allowed:
        raise ValueError('reference is from an unrecorded data configuration')
    argv=training_argv(job,reference['checkpoint'],job['official_data_config'],run,end,reset_cursor=first,
                       loss_config=job.get('cross_camera_loss_config'))
    accumulation = job.get('gradient_accumulation', 1)
    if not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError('gradient accumulation must be a positive integer')
    if accumulation != 1:
        argv += ['--gradient-accumulation', str(accumulation)]
    if set(argv)&{'--override-lr','--lr','--backbone-lr','--lr-schedule','--lr-end-step','--reset-time-projections'}:
        raise ValueError('official-data admission must restore the existing optimizer and LR plan')
    return argv,first


class OfficialDataController(ExpandedDynamicController):
    def __init__(self,path):
        super().__init__(path)
        self.active=Path(self.job['main_active_path'])

    def _check_candidate_state(self,checkpoint,step):
        from multicamera4rc.losses import MultiCamera4RCLoss
        state=validate_checkpoint(checkpoint,step,world_size=4,batch_size=1)
        verify_state(state,self.job['target_step'])
        expected,actual=deepcopy(self._fixed_state()),deepcopy(state)
        # New optional interventions have explicit default fields in receipts;
        # missing historical fields mean those exact defaults, not a new loss.
        for value in (expected,actual):
            value['loss_config']=MultiCamera4RCLoss(**value['loss_config']).to_config()
        activation = self.progress.get('cross_camera_activation_step')
        if self.job.get('cross_camera_loss_config') and activation is not None and step > activation:
            expected['loss_config'] = require_cross_camera_loss_config(self.job, expected['loss_config'])
        require_fixed_training_state(expected,actual)
        if state.get('augmentation')!=self._fixed_state().get('augmentation'):
            raise ValueError('official admission changed augmentation')
        return state

    def verify_handover(self):
        verify_admission(self.job)
        self._evaluation_memory_fraction()
        previous,adopted=self.job['previous'],self.job['adopted_training']
        if self.job.get('max_quality_retries',0)!=0:
            raise ValueError('automatic LR retries are forbidden')
        if build_validation_contract(self.job)!=self.job['validation_contract']:
            raise ValueError('original validation contract changed')
        for path,sha in self.job['evidence_sha256'].items():
            if digest(path)!=sha:raise ValueError('existing training evidence changed: '+path)
        for path,sha in self.job['protected_files'].items():
            if digest(path)!=sha:raise ValueError('frozen experiment configuration changed: '+path)
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip()!=self.job['commit']:
            raise ValueError('integration checkout changed')
        if subprocess.check_output(['git','status','--porcelain'],cwd=self.root,text=True).strip():
            raise ValueError('integration checkout is dirty')
        active=json.loads(self.active.read_text())
        keys=('pid','control_dir','run','phase_target_step','data_config','physical_gpus','selected_checkpoint')
        if any(active[k]!=previous[k] for k in keys):
            raise RuntimeError('main advanced during handover preparation; no process was stopped')
        status=json.loads((Path(previous['control_dir'])/'progress.json').read_text())
        if status['stage']!=f"train_{adopted['end']}_try0" or status.get('child_pid')!=adopted['pid']:
            raise RuntimeError('main is no longer in the recorded ordinary training segment')
        tree=owned_main_tree(previous,self.job['previous_identity'])
        require_training_child(tree.get(adopted['pid']),adopted,previous['pid'])
        if len(training_ranks(tree))!=len(self.job['physical_gpus']):
            raise RuntimeError('main training ranks are missing')
        refs={name:json.loads(Path(self.job[name+'_record']).read_text()) for name in ('initial','anchor','historical')}
        require_validation_contract(self.job['validation_contract'],*refs.values())
        for record in refs.values():self._verify_record(record)
        if (refs['initial']['checkpoint']!=active['selected_checkpoint']
                or refs['initial']['step']!=adopted['begin']
                or status['best']['checkpoint']!=refs['anchor']['checkpoint']):
            raise ValueError('accepted reference or best checkpoint changed')
        decision=quality_decision(refs['initial']['errors'],refs['anchor']['errors'],refs['historical']['errors'],refs['initial']['errors'])
        if decision['action']!='continue':raise ValueError('accepted reference failed the existing quality gate')
        state=validate_checkpoint(refs['initial']['checkpoint'],adopted['begin'],world_size=4,batch_size=1)
        verify_state(state,self.job['target_step'])
        require_cross_camera_loss_config(self.job, state['loss_config'])
        if state['dataset_size']!=read_data_config(self.job['official_data_config'])['epoch_size']:
            raise ValueError('epoch size must remain unchanged')
        self._verified_adopted_state=state
        return refs,status,tree,decision

    def segment(self,reference,best,historical,config,end,**options):
        if self.progress.get('official_activation_failed'):
            return super().segment(reference,best,historical,reference['data_config'],end,**options)
        if options or reference.get('learning_rate_trial') or reference.get('restart_lr'):
            raise ValueError('one-time LR overrides and retries are forbidden')
        verify_admission(self.job)
        self._check_candidate_state(reference['checkpoint'],reference['step'])
        run=self.output/f"continuation_{reference['step']}_{end}_try0"
        if run.exists():raise ValueError('refusing to overwrite a training segment')
        argv,reset=official_training_argv(self.job,reference,run,end)
        admitted_loss = require_cross_camera_loss_config(self.job, self._fixed_state()['loss_config'])
        if admitted_loss:
            self.progress.setdefault('cross_camera_activation_step', reference['step'])
            self.job['loss_config'] = admitted_loss
        self.job.update(run=str(run),selected_checkpoint=reference['checkpoint'],resume_step=reference['step'],
            phase_target_step=end,data_config=self.job['official_data_config'])
        if admitted_loss and reference['step'] == self.progress['cross_camera_activation_step']:
            write_json(self.output/'cross_camera_activation.json',dict(stage='launching',
                activation_step=reference['step'],checkpoint=reference['checkpoint'],
                data_config=self.job['data_config'],reset_data_cursor=reset,
                optimizer_and_rank_rng_restored=True,learning_rate_changed=False,
                lr_plan=self._fixed_state()['lr_plan'],loss_config=admitted_loss))
        if reset:
            write_json(self.output/'official_activation.json',dict(stage='launching',activation_step=reference['step'],
                old_config=reference['data_config'],new_config=self.job['data_config'],
                reset_data_cursor=True,optimizer_and_rank_rng_restored=True,learning_rate_changed=False,
                lr_plan=self._fixed_state()['lr_plan'],fractions=dict(existing=.87,mv_kubric=.10,panoptic=.03),
                cross_camera_3d_weight=(admitted_loss or self._fixed_state()['loss_config']).get('cross_camera_3d_weight',0),
                loss_config=admitted_loss or self._fixed_state()['loss_config']))
        self.run_process(f'train_{end}_try0',argv)
        checkpoint=run/'latest.pt';state=self._check_candidate_state(checkpoint,end)
        record=self.evaluate(checkpoint,f'extended_{end}_try0');record.update(step=end,data_config=self.job['data_config'])
        result=self._accept_or_review(reference,best,historical,record,state)
        write_json(self.output/'official_activation_status.json',dict(stage='training_and_validation_passed',step=end,
            checkpoint=str(checkpoint),learning_rate_changed=False,data_config=self.job['data_config']))
        return result

    def run(self):
        previous_root=Path(self.job['previous']['repo'])
        roots=sorted({previous_root.resolve(),self.root.resolve()})
        with ExitStack() as stack:
            handovers=[]
            for root in roots:
                directory=root/'outputs/training_preflight';directory.mkdir(parents=True,exist_ok=True)
                h=stack.enter_context((directory/'controller_handover.lock').open('a'))
                fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);handovers.append(h)
            old_lock=stack.enter_context((previous_root/'outputs/training_preflight/extended_training.lock').open('a'))
            if self.root.resolve()!=previous_root.resolve():
                new_lock=stack.enter_context((self.root/'outputs/training_preflight/extended_training.lock').open('a'))
                fcntl.flock(new_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                refs=self.adopt(old_lock)
                self.progress['validation_migration']['kind']='official_data_boundary_handover'
                for h in handovers:fcntl.flock(h,fcntl.LOCK_UN)
                current=self.finish_adopted(refs)
                self.remember_best(current);write_json(self.output/'latest_accepted_checkpoint.json',current)
                # GPU checks use a main GPU, after its old segment and original
                # validation finish. A failure continues the accepted old recipe.
                gpu=self.job['environment']['CUDA_VISIBLE_DEVICES'].split(',')[1]
                preflight=self.output/'official_giant_preflight.json'
                command=[self.job['python'],'-u','-m','scripts.preflight_official_multicam',
                    '--config',self.job['official_data_config'],'--checkpoint',current['checkpoint'],
                    '--output',str(preflight),'--gpu-uuid',gpu]
                if self.job.get('cross_camera_loss_config'):
                    command += ['--loss-config', self.job['cross_camera_loss_config']]
                loss_only = self.job.get('admission_kind') == 'cross_camera_only'
                if loss_only:
                    command += ['--existing-only']
                try:
                    RecoveryController.run_process(self,'official_giant_preflight',command,evaluation=True)
                    proof=json.loads(preflight.read_text())
                    has_cross = proof.get('loss_config',{}).get('cross_camera_3d_weight',0)>0
                    expected=len(proof.get('expected_cases',[])) if loss_only else (12 if has_cross else 10)
                    actual={(r['source'],r['cameras'],r['times']) for r in proof['rows']}
                    if (not proof.get('passed') or not expected or len(proof['rows'])!=expected
                            or actual != {tuple(x) for x in proof.get('expected_cases',[])}
                            or proof.get('config_sha256') != digest(self.job['official_data_config'])):
                        raise ValueError('all required new/existing layouts must pass giant backward')
                    if has_cross and not {
                        ('syn4d_v1',8,2),('syn4d_sim',4,4)
                    }.issubset({(r['source'],r['cameras'],r['times']) for r in proof['rows']}):
                        raise ValueError('large native Syn4D cross-camera memory checks are missing')
                except Exception:
                    self.progress['official_activation_failed']=traceback.format_exc()
                    self.job['data_config']=current['data_config']
                    write_json(self.output/'official_activation_status.json',dict(stage='preflight_failed_old_data_continues',
                        error=self.progress['official_activation_failed'],learning_rate_changed=False))
                while current['step']<self.job['target_step']:
                    end=next_segment_end(current['step'],self.job['target_step'],interval=self.job['validation_interval'])
                    current=self.segment(current,self.progress['best'],refs['historical'],self.job['data_config'],end)
                    self.remember_best(current);write_json(self.output/'latest_accepted_checkpoint.json',current)
                self.publish('completed_'+str(self.job['target_step']))
            except Exception:
                self.progress['error']=traceback.format_exc()
                if self.owns_training:self.publish('failed')
                else:write_json(self.control/'progress.json',dict(self.progress,stage='verification_failed'))
                raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--controller',required=True);p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();c=OfficialDataController(a.controller)
    if a.verify_only:
        c.verify_handover();print(json.dumps(dict(passed=True,no_processes_signalled=True)))
    else:c.run()
