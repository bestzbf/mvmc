#!/usr/bin/env python3
"""Inspect or root-authorized PAUSE at a NEW completed100-step old5 save.

Default sends no signals. Explicit finish flag may complete only the captured
training subtree AFTER the original controller exits; evaluation is excluded.
Independent validation sessions are recorded and preserved. No GPU work.
"""
import argparse
import os
from pathlib import Path
import time
from control_common import alive,atomic,checkpoint,descendants,pinned,proc,read,sha,stop_records


def newest(run):
    return sorted((p for p in Path(run).glob('step_*.pt') if not p.is_symlink()),key=lambda p:int(p.stem.split('_')[1]))
def inspect(control):
    control=Path(control);job=read(control/'job.json');progress=read(control/'progress.json')
    owner=proc(job['pid']);child=proc(progress['child_pid'])
    if (not owner or owner['uid']!=os.getuid() or '--controller '+str(control/'job.json') not in owner['cmd']
            or not child or child['ppid']!=owner['pid'] or child['uid']!=os.getuid()
            or child['sid']!=child['pid'] or 'multicamera4rc.launch' not in child['cmd']):raise RuntimeError('Old controller/launcher identity mismatch')
    if job['physical_gpus']!=[0,1,2,5,6] or job.get('allow_shared'):raise ValueError('Unexpected old5 layout')
    files=newest(job['run'])
    if not files:raise FileNotFoundError('No complete numbered old checkpoint')
    latest=checkpoint(files[-1],5)
    protected=[p for p in descendants(owner) if p['ppid']==owner['pid'] and p['pid']!=child['pid']]
    return dict(old_control=str(control),controller=owner,launcher=child,run=job['run'],latest_checkpoint=latest['path'],latest_step=latest['step'],last_observed_update=progress.get('last_observed_update'),protected_independent_children=protected,job=job)
def execute(spec,output,poll=1.,finish_paused_owned_tree=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    current=inspect(spec['old_control'])
    for key in ('controller','launcher'):
        if current[key]['pid']!=spec[key]['pid'] or current[key]['start']!=spec[key]['start']:raise RuntimeError('Old process generation changed')
    pause=Path(spec['old_control'])/'PAUSE'
    if pause.exists():raise FileExistsError('Do not overwrite an existing PAUSE request')
    atomic(output/'old_job_snapshot.json',current['job'])
    atomic(output/'observer_status.json',dict(stage='waiting_new_complete100_checkpoint',baseline_step=current['latest_step']))
    baseline=max(spec['latest_step'],current['latest_step'])
    while True:
        if not alive(spec['controller']) or not alive(spec['launcher']):raise RuntimeError('Old process exited before authorized pause')
        files=[p for p in newest(spec['run']) if int(p.stem.split('_')[1])>baseline and int(p.stem.split('_')[1])%100==0]
        candidate=None
        for path in reversed(files):
            try:candidate=checkpoint(path,5);break
            except (ValueError,EOFError,RuntimeError,OSError):continue
        if candidate is None:time.sleep(poll);continue
        # Check identities again immediately before the only old-control write.
        if not alive(spec['controller']) or not alive(spec['launcher']):raise RuntimeError('Old identity changed before PAUSE')
        training=descendants(spec['launcher'])
        protected=list(spec['protected_independent_children'])
        protected.extend(p for p in descendants(spec['controller']) if p['ppid']==spec['controller']['pid'] and p['pid']!=spec['launcher']['pid'])
        with pause.open('x') as stream:
            stream.write('six_gpu_joint_activation_v22; new completed checkpoint '+str(candidate['step'])+'\n');stream.flush();os.fsync(stream.fileno())
        atomic(output/'pause_requested.json',dict(observed_checkpoint=candidate,controller=spec['controller'],launcher=spec['launcher'],training_tree=training,protected_independent_children=spec['protected_independent_children']))
        deadline=time.monotonic()+180
        finished_owned=False
        while alive(spec['controller']) or any(alive(p) for p in training):
            if finish_paused_owned_tree and not alive(spec['controller']) and any(alive(p) for p in training) and not finished_owned:
                atomic(output/'old_controller_cleanup_incomplete.json',dict(old_controller_exited=True,remaining_owned=[p for p in training if alive(p)],protected_evaluation=protected))
                stop_records(training,protected=protected)
                finished_owned=True
                atomic(output/'observer_owned_cleanup_complete.json',dict(old_controller_exited=True,pidfd_owned_training_cleanup_complete=True,training_tree=training,protected_evaluation=protected))
            if time.monotonic()>deadline:raise RuntimeError('Old owned processes remain after PAUSE; no six-card launch authorized')
            time.sleep(.5)
        atomic(output/'source_processes_stopped.json',dict(stage='processes_stopped',controller=spec['controller'],training_tree=training,all_owned_training_exited=True,protected_independent_children=[dict(identity=p,still_alive=alive(p)) for p in spec['protected_independent_children']]))
        # Rescan AFTER shutdown; the stop can race a later valid native save.
        final=None
        for path in reversed(newest(spec['run'])):
            try:final=checkpoint(path,5);break
            except (ValueError,EOFError,RuntimeError,OSError):continue
        if final is None or final['step']<candidate['step']:raise RuntimeError('Final then-latest save disappeared')
        pin_dir=output/'source_pins';pin_dir.mkdir(exist_ok=True)
        pin=pin_dir/Path(final['path']).name
        os.link(final['path'],pin)  # no copy/overwrite; source old5 remains intact
        if (os.stat(pin).st_dev,os.stat(pin).st_ino)!=(os.stat(final['path']).st_dev,os.stat(final['path']).st_ino):raise RuntimeError('Source hardpin inode mismatch')
        digest=sha(pin);final['original_path']=final['path'];final['path']=str(pin);final['sha256']=digest
        receipt=dict(stage='complete',new_data_stream_reset=True,old_permutation_continuity_claim=False,source=final,pause_requested_at_step=candidate['step'],old_processes_exited=True,source_processes_stopped=str(output/'source_processes_stopped.json'),old_job_snapshot=str(output/'old_job_snapshot.json'),protected_independent_children=spec['protected_independent_children'],rollback_note='Retain exact original5 pin. Root must prove unique starter, all failed new6 owned processes exited and release locks before restarting a derived original03a5 job; never restart initial139006 or138900.')
        atomic(output/'source_stopped.json',receipt)
        atomic(output/'observer_status.json',dict(stage='source_stopped_thenlatest_hardpinned',step=final['step'],source_sha256=digest))
        return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('inspect','observe_pause'));p.add_argument('--old-control',type=Path);p.add_argument('--snapshot',type=Path);p.add_argument('--snapshot-sha256');p.add_argument('--finish-paused-owned-tree',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.command=='inspect':
        if a.old_control is None:p.error('--old-control required')
        value=inspect(a.old_control);atomic(a.output,value)
        print({'controller_pid':value['controller']['pid'],'launcher_pid':value['launcher']['pid'],'latest_step':value['latest_step'],'snapshot':str(a.output)});return
    if a.snapshot is None or a.snapshot_sha256 is None:p.error('Pinned inspect snapshot required')
    spec=pinned(a.snapshot,a.snapshot_sha256);result=execute(spec,a.output,finish_paused_owned_tree=a.finish_paused_owned_tree);print({'stage':result['stage'],'step':result['source']['step'],'source':result['source']['path']})
if __name__=='__main__':main()
