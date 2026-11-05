"""Single-GPU paired calibration trials, fixed validation and conditional cutover.

The original training continues while trials run. This process uses one GPU,
serially, with four accumulated microbatches per Adam update. Test benchmarks
are never used for selection. A rejected candidate leaves main training alone.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np

from multicamera4rc.quality import compare_errors, error_panel
from scripts.continue_quality_recovery import training_argv, validate_checkpoint
from scripts.cutover_native_motion_stereo import alive, proc, stop_tree, write_json
from scripts.migrate_dynamic_validation import (SUITES, suite_paths, build_validation_contract,
                                               verify_suite_report, ExpandedDynamicController)


def calibration_groups(entries):
    groups=defaultdict(lambda:defaultdict(list))
    for suite,entry in entries.items():
        for row in entry['records']:
            family=(row['source_name'] if suite=='native_dynamic' else row['source'] if suite=='static'
                    else 'syn4d/'+row['sequence'].split('/')[0] if suite=='syn4d' else suite)
            groups[family][row['sequence']].append(row['metrics'])
    result={}
    for family,scenes in groups.items():
        result[family]={}
        for sequence,rows in scenes.items():
            keys={key for row in rows for key in row if key.startswith('intrinsics_')}
            if 'intrinsics_log_focal_error' not in keys:
                raise ValueError('independent calibration metrics missing')
            result[family][sequence]={key:float(np.mean([row[key] for row in rows if key in row])) for key in keys}
    return dict(result)


def calibration_decision(control,candidate):
    """Fixed thresholds, family-balanced means and paired sequence bootstrap."""
    a,b=calibration_groups(control),calibration_groups(candidate)
    if a.keys()!=b.keys() or any(a[k].keys()!=b[k].keys() for k in a):
        raise ValueError('calibration groups must contain identical sequences')
    rows={};samples=[];bad=[]
    for name in sorted(a):
        pairs=np.array([[a[name][s]['intrinsics_log_focal_error'],b[name][s]['intrinsics_log_focal_error']]
                        for s in sorted(a[name])])
        means=pairs.mean(0);samples.append(pairs)
        rows[name]=dict(sequences=len(pairs),control=float(means[0]),candidate=float(means[1]),
                        ratio=float(means[1]/max(means[0],1e-9)))
        for label,data in (('control_metrics',a),('candidate_metrics',b)):
            keys={key for metrics in data[name].values() for key in metrics}
            rows[name][label]={key:float(np.mean([m[key] for m in data[name].values() if key in m])) for key in keys}
        if means[1]>means[0]*1.10 and means[1]-means[0]>.002:
            bad.append(name)
        p0=rows[name]['control_metrics'].get('intrinsics_principal_error_normalized')
        p1=rows[name]['candidate_metrics'].get('intrinsics_principal_error_normalized')
        if p0 is not None and p1>p0*1.10 and p1-p0>.002:
            bad.append(name+'/principal')
    before=float(np.mean([v['control'] for v in rows.values()]));after=float(np.mean([v['candidate'] for v in rows.values()]))
    rng=np.random.default_rng(20261013);bootstrap=np.zeros((2000,2))
    for pairs in samples:
        indices=rng.integers(0,len(pairs),size=(2000,len(pairs)))
        bootstrap+=pairs[indices].mean(1)/len(samples)
    ratios=bootstrap[:,1]/np.maximum(bootstrap[:,0],1e-9)
    ci=np.quantile(ratios,[.025,.975]).tolist()
    physical=compare_errors(error_panel(control),error_panel(candidate))
    syn_names=('depth_abs_rel','base_epe_normalized','trajectory_epe_temporal_normalized','displacement_epe_dynamic_normalized')
    syn_ratios={k:candidate['syn4d']['mean'][k]/max(control['syn4d']['mean'][k],1e-9)
                for k in syn_names if k in control['syn4d']['mean']}
    visibility=[]
    for suite in SUITES:
        for old,new in zip(control[suite]['records'],candidate[suite]['records']):
            if 'visibility_f1' in old['metrics']:
                visibility.append(new['metrics']['visibility_f1']-old['metrics']['visibility_f1'])
    visibility_change=float(np.mean(visibility)) if visibility else 0.
    ratio=after/max(before,1e-9)
    passed=(ratio<=.98 and ci[1]<1. and not bad and physical['score_ratio']<=1.01
            and physical['worst_metric_ratio']<=1.05 and all(v<=1.05 for v in syn_ratios.values())
            and visibility_change>=-.01)
    return dict(passed=bool(passed),macro_log_focal_control=before,macro_log_focal_candidate=after,
                macro_log_focal_ratio=ratio,paired_sequence_bootstrap_95pct=ci,
                by_family=rows,regressed_families=bad,physical=physical,syn4d_metric_ratios=syn_ratios,
                mean_visibility_f1_change=visibility_change,
                thresholds=dict(focal_ratio_max=.98,bootstrap_upper_max=1.,physical_macro_max=1.01,
                                physical_worst_max=1.05,syn4d_worst_max=1.05,visibility_drop_max=.01))


def checkpoint_health(job,path,loss_config,augmentation,*,step=None):
    state=validate_checkpoint(path,job['end'] if step is None else step,world_size=1,batch_size=1)
    if (state['lr_plan']!=job['source_lr_plan'] or state['loss_config']!=loss_config
            or state.get('augmentation')!=augmentation or state.get('gradient_accumulation')!=4
            or state.get('effective_batch_size')!=4 or state.get('sampler_kind')!='distributed_v1'):
        raise ValueError('paired trial changed its LR, loss, augmentation or effective batch contract')
    return state


def trial_milestones(job):
    steps=job.get('milestones',[job['end']])
    if (not steps or steps!=sorted(set(steps)) or steps[-1]!=job['end']
            or any(not isinstance(step,int) or step<=job['begin'] for step in steps)
            or job.get('control_end',job['end'])>steps[0]):
        raise ValueError('trial milestones must increase from the shared starting checkpoint')
    return steps


class Trial:
    def __init__(self,path):
        self.path=Path(path);self.job=json.loads(self.path.read_text())
        self.root=Path(self.job['repo']);self.output=Path(self.job['output_root'])
        self.progress=dict(stage='starting',phases=[],physical_gpus=self.job['physical_gpus'])
        if len(self.job['physical_gpus'])!=1 or self.job['gradient_accumulation']!=4:
            raise ValueError('this trial is restricted to one physical GPU and effective batch four')
        trial_milestones(self.job)
        self.environment=os.environ|self.job['environment']
        from multicamera4rc.launch import gpu_snapshot
        devices,_=gpu_snapshot();self.uuid=devices[self.job['physical_gpus'][0]].uuid

    def publish(self,stage,**fields):
        self.progress.update(stage=stage,updated_at=time.time(),**fields)
        write_json(self.output/'progress.json',self.progress)

    def process(self,name,command,*,evaluation=False):
        env=dict(self.environment)
        if evaluation:env['CUDA_VISIBLE_DEVICES']=self.uuid
        log=self.output/(name+'.log');began=time.time()
        self.publish(name,child_log=str(log),phase_started_at=began)
        with log.open('w') as handle:
            child=subprocess.Popen(command,cwd=self.root,env=env,stdout=handle,stderr=subprocess.STDOUT)
            self.publish(name,child_pid=child.pid)
            result=child.wait()
        self.progress.pop('child_pid',None)
        self.progress['phases'].append(dict(name=name,returncode=result,log=str(log),
                                           started_at=began,seconds=time.time()-began))
        self.publish(name)
        if result:raise RuntimeError(f'{name} failed ({result}); see {log}')

    def wait_control(self):
        recorded=self.job['control_training'];checkpoint=Path(recorded['run'])/'latest.pt'
        self.publish('train_control_single',child_pid=recorded['pid'],child_log=recorded['log'])
        while alive(recorded['identity']):time.sleep(10)
        self.progress.pop('child_pid',None)
        if not checkpoint.is_file():raise RuntimeError('control training exited without an atomic checkpoint')
        return checkpoint

    def train(self,label,*,resume=None,end=None):
        end=self.job['end'] if end is None else end
        name=label+f'_{end}' if 'milestones' in self.job else label
        run=self.output/'arms'/name
        if run.exists():raise RuntimeError('refusing to overwrite a trial')
        config=self.job['roi_config'] if label=='log_focal_roi' else self.job['data_config']
        command=training_argv(self.job,resume or self.job['checkpoint'],config,run,end,
                              loss_config=None if label=='control' else self.job['loss_config'])
        command[command.index('--save-every')+1]=str(self.job.get('save_every',100))
        command+=['--gradient-accumulation','4']
        if '--override-lr' in command or '--reset-data-cursor' in command:raise ValueError('trial must restore LR and cursor')
        self.process(name+'_train',command)
        return run/'latest.pt'

    def evaluate(self,path,label,*,step=None):
        reports={};entries={}
        record=dict(label=label,checkpoint=str(path),step=self.job['end'] if step is None else step,
                    validation_contract=self.job['validation_contract'],static_contract=self.job['static_contract'])
        if build_validation_contract(self.job)!=self.job['validation_contract']:
            raise ValueError('validation evidence changed')
        for suite in SUITES:
            plan,config=suite_paths(self.job,suite);directory=self.output/'validation'/label/suite
            directory.mkdir(parents=True,exist_ok=True)
            if suite=='native_dynamic':
                module='scripts.evaluate_native_dynamic';args=['--data-config',config,'--plan',plan,'--memory-fraction','.70']
            elif suite=='static':
                copied=directory/'fixed_plan.json';shutil.copyfile(plan,copied)
                module='scripts.evaluate_static_multiview';args=['--train-config',config,'--plan',str(copied),'--memory-fraction','.70']
            else:
                module='multicamera4rc.evaluate';args=['--data-config',config,'--plan',plan,'--cpu-threads','4']
            self.process('validate_'+label+'_'+suite,[self.job['python'],'-u','-m',module,*args,
                        '--output',str(directory),'--checkpoints',label+'='+str(path)],evaluation=True)
            report_path=directory/'evaluation.json';report=json.loads(report_path.read_text())
            if report.get('stage')!='passed':raise ValueError('incomplete evaluation')
            entries[suite]=verify_suite_report(suite,report,record,json.loads(Path(plan).read_text()),self.job['validation_contract'])
            reports[suite]=str(report_path)
        record.update(errors=error_panel(entries),reports=reports,intrinsics=calibration_groups(entries))
        write_json(self.output/'validation'/label/'quality.json',record)
        return record,entries

    def final_benchmark(self,results):
        """Measure the frozen final arms after held-out selection is recorded."""
        if 'pstudio' not in self.job:return
        from scripts.evaluate_pstudio_calibration import compare_reports
        reports={}
        for label in ('control','log_focal','log_focal_roi'):
            output=self.output/'pstudio'/label
            self.process('pstudio_'+label,[self.job['python'],'-u','-m',
                'scripts.evaluate_pstudio_calibration','--config',self.job['pstudio'],
                '--checkpoint',results[label][0]['checkpoint'],'--output',str(output)],evaluation=True)
            report=json.loads((output/'results.json').read_text())
            if report['stage']!='passed':raise ValueError('incomplete final PStudio benchmark')
            reports[label]=report
        write_json(self.output/'pstudio'/'comparison.json',compare_reports(reports))

    def run(self):
        import torch
        payload=torch.load(self.job['checkpoint'],map_location='cpu',mmap=True,weights_only=True)
        legacy=payload['training_state']['loss_config'];del payload
        proposed=json.loads(Path(self.job['loss_config']).read_text())
        audit=json.loads((self.output/'roi_loader_gt_audit.json').read_text())
        if not audit.get('passed') or audit.get('sources')!=25 or not audit.get('loader_integration'):
            raise ValueError('real ROI loader/GT audit did not pass')
        augmentation=json.loads(Path(self.job['roi_config']).read_text())['augmentation']
        control=self.wait_control();control_end=self.job.get('control_end',self.job['end'])
        checkpoint_health(self.job,control,legacy,None,step=control_end)
        paths=dict(control=control,log_focal=self.job['checkpoint'],log_focal_roi=self.job['checkpoint'])
        history=[]
        for end in trial_milestones(self.job):
            results={}
            for label in ('control','log_focal','log_focal_roi'):
                self.publish('milestone',milestone_step=end,arm=label)
                if label!='control' or end!=control_end:
                    paths[label]=self.train(label,resume=paths[label],end=end)
                checkpoint_health(self.job,paths[label],legacy if label=='control' else proposed,
                                  augmentation if label=='log_focal_roi' else None,step=end)
                name=label+f'_{end}' if 'milestones' in self.job else label
                results[label]=self.evaluate(paths[label],name,step=end)
                results[label][0]['arm']=label
                if label!='control':
                    decision=calibration_decision(results['control'][1],results[label][1])
                    write_json(self.output/(name+'_decision.json'),decision)
            decisions={label:calibration_decision(results['control'][1],result[1])
                       for label,result in results.items() if label!='control'}
            history.append(dict(step=end,updates=end-self.job['begin'],comparisons=decisions))
            write_json(self.output/'milestone_results.json',history)
        eligible=[label for label,value in decisions.items() if value['passed']]
        selected=min(eligible,key=lambda label:decisions[label]['macro_log_focal_candidate']) if eligible else None
        decision=dict(selected=selected,comparisons=decisions,milestones=history,lr_changed=False,
                      begin=self.job['begin'],end=self.job['end'],test_used_for_selection=False,validation_clips=sum(
                      len(value['records']) for value in results['control'][1].values()))
        write_json(self.output/'decision.json',decision)
        self.final_benchmark(results)
        if selected:
            self.publish('candidate_passed',selected=selected)
            adopt_and_continue(self,results[selected][0])
        else:
            self.publish('no_candidate_passed_main_unchanged',decision=str(self.output/'decision.json'))


def owned_tree(identity):
    current=proc(identity['pid'])
    if (not current or current['start']!=identity['start'] or current['cmd']!=identity['cmd']
            or current['uid']!=os.getuid() or current['state']=='Z'):
        raise RuntimeError('main controller identity changed; leave it untouched')
    records=[proc(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()]
    tree={current['pid']:current}
    while True:
        children=[p for p in records if p and p['ppid'] in tree and p['pid'] not in tree]
        if not children:break
        tree.update({p['pid']:p for p in children})
    if any(p['uid']!=os.getuid() for p in tree.values()):raise RuntimeError('foreign process in main tree')
    return tree


class CalibratedContinuation(ExpandedDynamicController):
    def publish(self,stage):
        super().publish(stage)
        own_active=self.root/'outputs/training_preflight/active_training.json'
        if self.owns_training and own_active!=self.active:
            write_json(own_active,self.job)

    def _check_candidate_state(self,checkpoint,step):
        if str(checkpoint)==self.job['initial_checkpoint']:
            return validate_checkpoint(checkpoint,step,world_size=1,batch_size=1)
        state=super()._check_candidate_state(checkpoint,step)
        if state.get('augmentation')!=self.job['admitted_augmentation'] or state.get('gradient_accumulation',1)!=1:
            raise ValueError('four-GPU continuation changed the admitted augmentation or accumulation')
        return state

    def evaluate(self,checkpoint,label):
        record=super().evaluate(checkpoint,label)
        entries={suite:json.loads(Path(path).read_text())['checkpoints'][label] for suite,path in record['reports'].items()}
        record['intrinsics']=calibration_groups(entries)
        write_json(self.output/'validation'/label/'quality.json',record)
        return record


def adopt_and_continue(trial,record):
    """Adopt only if no newer main checkpoint would be discarded."""
    import torch
    active=Path(trial.job['main_active_path']);old=json.loads(active.read_text())
    old_control=Path(old['control_dir']);old_output=Path(old['output_root'])
    previous=json.loads((old_output/'latest_accepted_checkpoint.json').read_text())
    if old['pid']!=trial.job['main_identity']['pid'] or previous['step']!=trial.job['begin']:
        trial.publish('passed_requires_rebase_main_advanced');return
    newer=list(Path(old['run']).glob('step_*.pt'))
    if any(int(path.stem.split('_')[-1])>record['step'] for path in newer):
        trial.publish('passed_requires_rebase_newer_main_checkpoint');return
    migration=compare_errors(previous['errors'],record['errors'])
    if migration['score_ratio']>1.01 or migration['worst_metric_ratio']>1.05:
        trial.publish('passed_control_but_failed_main_anchor',comparison=migration);return
    payload=torch.load(record['checkpoint'],map_location='cpu',mmap=True,weights_only=True)
    state=payload['training_state'];del payload
    if state['lr_plan']!=trial.job['source_lr_plan'] or state['batches_consumed']%4:
        raise ValueError('candidate cannot restore main LR and sample cursor')
    config=trial.job['roi_config'] if record.get('arm',record['label'])=='log_focal_roi' else trial.job['data_config']
    from multicamera4rc.datasets.mixture import read_data_config
    before,after=read_data_config(trial.job['data_config']),read_data_config(config)
    after.pop('augmentation',None)
    if before!=after:raise ValueError('augmentation admission changed source sampling')
    progress=json.loads((old_control/'progress.json').read_text())
    historical=json.loads(Path(old['historical_record']).read_text())
    control=trial.output/'continuation_control';output=trial.output/'continued_training'
    control.mkdir();output.mkdir()
    (trial.root/'outputs/training_preflight').mkdir(parents=True,exist_ok=True)
    job=deepcopy(old);job.pop('audited_data_config_replacement',None)
    job.update(repo=str(trial.root),environment=dict(old['environment'],PYTHONPATH=str(trial.root)),
        control_dir=str(control),output_root=str(output),run=str(output),pid=os.getpid(),
        initial_checkpoint=record['checkpoint'],initial_step=record['step'],resume_step=record['step'],
        selected_checkpoint=record['checkpoint'],data_config=config,admitted_augmentation=state.get('augmentation'),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=trial.root,text=True).strip(),
        calibration_trial=str(trial.output/'decision.json'),max_quality_retries=0,
        log=str(trial.output/'supervisor.log'))
    write_json(control/'job.json',job)
    controller=CalibratedContinuation(control/'job.json');controller.active=active
    controller.progress=deepcopy(progress)
    for key in ('child_pid','child_log','error'):controller.progress.pop(key,None)
    expected=dict(state,world_size=4,gradient_accumulation=1,effective_batch_size=4)
    controller.progress.update(adopted_expected_state=expected,restored_training_state=expected,
                               resumed_training_state=state,intrinsics_admission=str(trial.output/'decision.json'))
    record=dict(record,data_config=config)
    proof=dict(passed=True,selected=record,previous=previous,learning_rate_changed=False,
               old_world_size=1,new_world_size=4,effective_batch_size=4,reset_data_cursor=False,
               source_sampling_unchanged=True,main_anchor_comparison=migration)
    write_json(output/'admission.json',proof)
    # All proof, candidate and continuation configuration exist before stopping
    # only the recorded same-UID main controller and its descendants.
    with (Path(old['repo'])/'outputs/training_preflight/controller_handover.lock').open('a') as handover:
        fcntl.flock(handover,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if json.loads(active.read_text())['pid']!=old['pid']:raise RuntimeError('main changed before adoption')
        if json.loads((old_output/'latest_accepted_checkpoint.json').read_text())['step']!=trial.job['begin']:
            trial.publish('passed_requires_rebase_main_advanced');return
        if any(int(path.stem.split('_')[-1])>record['step'] for path in Path(old['run']).glob('step_*.pt')):
            trial.publish('passed_requires_rebase_newer_main_checkpoint');return
        tree=owned_tree(trial.job['main_identity']);stop_tree(tree)
        write_json(old_control/'transition.json',dict(stage='intrinsics_candidate_adopted',new_control_dir=str(control),
                                                     checkpoint=record['checkpoint'],stopped_pids=list(tree)))
        controller.owns_training=True;controller.publish('intrinsics_candidate_adopted')
    from scripts.continue_extended_training import next_segment_end
    with ExitStack() as stack:
        for root in sorted({Path(old['repo']),trial.root}):
            lock=stack.enter_context((root/'outputs/training_preflight/extended_training.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(output/'latest_accepted_checkpoint.json',record)
        write_json(output/'latest_evaluated_checkpoint.json',record)
        current=record
        try:
            while current['step']<job['target_step']:
                end=next_segment_end(current['step'],job['target_step'],interval=job['validation_interval'])
                current=controller.segment(current,controller.progress['best'],historical,config,end)
                controller.remember_best(current)
                write_json(output/'latest_accepted_checkpoint.json',current)
            controller.publish('completed_'+str(job['target_step']))
        except Exception:
            controller.progress['error']=traceback.format_exc();controller.publish('failed');raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True)
    args=parser.parse_args();trial=Trial(args.job)
    with (trial.output/'trial.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:trial.run()
        except Exception:
            trial.publish('failed',error=traceback.format_exc());raise


if __name__=='__main__':main()
