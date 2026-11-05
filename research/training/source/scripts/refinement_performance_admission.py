"""Run frozen speed trials inside the owning main controller at a safe boundary.

The controller keeps its GPU leases throughout testing and resumption. Trial
updates never replace main weights. Rejection returns the same source state.
"""
from pathlib import Path
import traceback

from scripts.admit_throughput import Admission
from scripts.continue_quality_recovery import training_argv, validate_checkpoint
from scripts.cutover_native_motion_stereo import write_json
from scripts.migrate_dynamic_validation import require_fixed_training_state
from scripts.query_pair_main_admission import continuation_health, require_cursor_progress, verify_files, read


def admit_at_checkpoint(controller,checkpoint,step):
    spec=controller.job['performance_admission']
    if step+5>controller.job['target_step']:return checkpoint,step
    verify_files(spec['proof_hashes'])
    proof=read(spec['cpu_proof'])
    if not proof['passed'] or proof['commit']!=controller.job['commit']:
        raise ValueError('performance code lacks a passing exact-commit CPU proof')
    trial=object.__new__(Admission)
    trial.root=controller.root
    trial.out=Path(spec['output'])/f'step_{step}'
    trial.out.mkdir(parents=True,exist_ok=False)
    trial.job=dict(boundary_step=step,costs=spec['costs'])
    trial.previous=dict(controller.job)
    status=dict(source_checkpoint=checkpoint,source_step=step)
    def publish(stage=None,**values):
        if stage:status['stage']=stage
        status.update(values)
        write_json(trial.out/'progress.json',status)
        controller.progress['performance_trial']=dict(status,output=str(trial.out))
        controller.publish('performance_admission')
    trial.write=publish
    controller.progress['performance_admission_attempted']=True
    try:
        flags=trial.experiments(checkpoint)
        migrated,config=trial.migrate(checkpoint,flags)
        run=trial.out/'production_preflight'
        argv=training_argv(dict(controller.job,allow_shared=False),migrated,controller.job['data_config'],run,step+5)
        log=trial.command('production_preflight',argv+['--gradient-accumulation','1'])
        health=continuation_health(log,step,step+5,4)
        before=validate_checkpoint(migrated,step,world_size=4,batch_size=1)
        after=validate_checkpoint(run/'latest.pt',step+5,world_size=4,batch_size=1)
        require_fixed_training_state(before,after)
        require_cursor_progress(before,after,5)
        if before['load_balancing']!=after['load_balancing']:
            raise ValueError('production sampler state changed')
        write_json(trial.out/'production_preflight_health.json',health)
        selected=str(run/'latest.pt')
        controller.job['model_config']=config
        controller.progress.update(performance_status='accepted',latest_checkpoint=selected,step=step+5)
        publish('accepted',accepted=True,selected_checkpoint=selected,selected_step=step+5)
        write_json(controller.output/'latest_training_checkpoint.json',dict(checkpoint=selected,step=step+5))
        return selected,step+5
    except Exception:
        controller.progress['performance_status']='rejected_resume_original'
        publish('rejected',accepted=False,error=traceback.format_exc(),resume_checkpoint=checkpoint,resume_step=step)
        return checkpoint,step
