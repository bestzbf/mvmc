"""Opt-in observational six-rank startup/update admission, with a CPU parser.

No tensor/gradient/state/RNG is modified. Full exact admission is checked before
the first native update. Only the requested first updates incur finite scans.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

SCHEMA='six-rank-execution-admission-v22'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def canonical(value):return json.loads(json.dumps(value,allow_nan=False))


def publish(path,value):
    """Whole tiny proof files appear atomically; a fresh attempt never overwrites."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    try:os.link(tmp,path)
    except OSError:
        # NFS can report an error after creating the link. Reconcile own inode.
        if not path.exists() or (path.stat().st_dev,path.stat().st_ino)!=(tmp.stat().st_dev,tmp.stat().st_ino):raise
    tmp.unlink()


def equal_tree(actual,expected,path='state',counts=None):
    import torch
    if counts is None:counts={'tensors':0,'elements':0}
    if isinstance(expected,torch.Tensor):
        if not isinstance(actual,torch.Tensor) or actual.shape!=expected.shape or actual.dtype!=expected.dtype:
            raise ValueError('Restore shape/dtype mismatch: '+path)
        a,b=actual.detach().reshape(-1),expected.detach().reshape(-1)
        for start in range(0,a.numel(),1<<20):
            if not torch.equal(a[start:start+(1<<20)].cpu(),b[start:start+(1<<20)].cpu()):
                raise ValueError('Restore tensor value mismatch: '+path)
        counts['tensors']+=1;counts['elements']+=a.numel()
    elif isinstance(expected,dict):
        if not isinstance(actual,dict) or actual.keys()!=expected.keys():raise ValueError('Restore mapping mismatch: '+path)
        for key in expected:equal_tree(actual[key],expected[key],path+'/'+str(key),counts)
    elif isinstance(expected,(list,tuple)):
        if type(actual)!=type(expected) or len(actual)!=len(expected):raise ValueError('Restore sequence mismatch: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):equal_tree(a,b,path+'/'+str(i),counts)
    elif type(actual)!=type(expected) or actual!=expected:raise ValueError('Restore metadata mismatch: '+path)
    return counts


def optimizer_proof(model,optimizer,resume):
    from multicamera4rc.optim import _legacy_names
    saved=resume['optimizer'];groups=saved['param_groups']
    names=([g['param_names'] for g in groups] if all('param_names' in g for g in groups)
           else _legacy_names(model,groups,resume['model']))
    expected={name:saved['state'].get(key,{}) for group,nn in zip(groups,names)
              for name,key in zip(nn,group['params'])}
    source_groups={g['name']:g for g in groups};actual_groups=optimizer.state_dict()['param_groups']
    counts={'tensors':0,'elements':0};existing=absent=0
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:continue
        if name not in expected:raise ValueError('Missing named source parameter: '+name)
        old=expected[name];actual=optimizer.state.get(parameter,{})
        equal_tree(actual,old,'Adam/'+name,counts)
        if old:existing+=1
        else:absent+=1
    for group in actual_groups:
        source=source_groups.get(group['name'],source_groups['heads'] if group['name']=='geometry' else None)
        if source is None:raise ValueError('Unsupported restored optimizer group')
        for key,value in source.items():
            if key not in ('name','params','param_names'):equal_tree(group[key],value,'group/'+group['name']+'/'+key)
    return dict(all_named_Adam_values_exact=True,existing_states_exact=existing,
        absent_states_not_created=absent,source_hyperparameters_exact=True,**counts)


def read_spec(path,sha):
    if digest(path)!=sha:raise ValueError('Six-rank gate spec digest mismatch')
    spec=json.loads(Path(path).read_text())
    if (spec.get('schema')!=SCHEMA or spec.get('world_size')!=6 or spec.get('batch_size')!=1
            or spec.get('gradient_accumulation')!=1 or spec.get('phase') not in ('boot','resume','continuous')
            or spec.get('required_update_count')!=1 or len(spec.get('ordered_gpu_uuids',[]))!=6):
        raise ValueError('Require the reviewed six-rank/batch1/accum1 one-update spec')
    if spec['expect_data_reset']!=(spec['phase']=='boot'):raise ValueError('Data reset is allowed exactly once at boot')
    return spec


def clean_FA4_errors(receipt):
    """Checkpoint's exact controlled-stop is counted separately, never hidden."""
    raw=receipt.get('execution_errors');expected=receipt.get('expected_checkpoint_early_stops')
    unexpected=receipt.get('unexpected_execution_errors');types=receipt.get('exception_types')
    if any(type(value) is not int or value<0 for value in (raw,expected,unexpected)):
        return False
    if not isinstance(types,dict) or any(type(value) is not int or value<0 for value in types.values()):return False
    return (unexpected==0 and raw==expected+unexpected and sum(types.values())==raw
            and types.get('torch.utils.checkpoint._StopRecomputationError',0)==expected
            and all(key=='torch.utils.checkpoint._StopRecomputationError' or value==0 for key,value in types.items()))


class SixRankProof:
    def __init__(self,*,args,context,resume,model,optimizer,lr_plan,criterion,scaler,
                 execution,data_epoch,batches_consumed,balance,dataset_size):
        self.root=os.environ.get('DFRT_SIX_RANK_PROOF_ROOT');self.enabled=bool(self.root)
        if not self.enabled:return
        import torch
        from multicamera4rc.distributed import capture_rng
        from multicamera4rc.losses import MultiCamera4RCLoss
        from multicamera4rc.schedule import LearningRatePlan,migrate_geometry_plan
        self.root=Path(self.root);self.context=context;self.rank=context.rank
        self.spec_sha=os.environ['DFRT_SIX_RANK_GATE_SPEC_SHA256']
        self.spec=read_spec(os.environ['DFRT_SIX_RANK_GATE_SPEC'],self.spec_sha)
        self.model,self.optimizer,self.plan=model,optimizer,lr_plan
        self.first=int(self.spec['checkpoint']['step']);self.written=0
        started=time.monotonic();row=self.base();error=None
        try:
            if resume is None or resume['step']!=self.first or context.world_size!=6 or not 0<=self.rank<6:
                raise ValueError('Six-rank source checkpoint/layout mismatch')
            if context.local_rank!=self.rank or context.device.index!=self.rank:
                raise ValueError('Single-node physical rank mapping differs')
            runtime_uuid=str(torch.cuda.get_device_properties(context.device).uuid)
            expected_uuid=self.spec['ordered_gpu_uuids'][self.rank]
            if runtime_uuid.removeprefix('GPU-')!=expected_uuid.removeprefix('GPU-'):
                raise ValueError('Rank physical GPU UUID differs')
            if os.environ.get('DFRT_FUTURE_DEPLOYMENT_SHA256')!=self.spec['deployment_sha256']:
                raise ValueError('Rank deployment identity differs')
            if (Path(args.resume).resolve()!=Path(self.spec['checkpoint']['path']).resolve()
                    or bool(args.migrate_data_config)!=self.spec['expect_data_reset']):
                raise ValueError('Checkpoint path or reset-once flag differs')
            # Full source SHA is checked by publisher/coordinator before spawn;
            # each rank binds it and proves actual loaded tensor values below.
            for option in ('data_config','loss_config'):
                item=self.spec[option]
                selected=getattr(args,option)
                if option=='loss_config' and selected is None and not self.spec['expect_data_reset']:
                    selected=item['path']
                if selected is None or digest(selected)!=item['sha256']:
                    raise ValueError('Rank pinned recipe file changed: '+option)
            if (args.batch_size,args.gradient_accumulation)!=(1,1):raise ValueError('Effective batch policy changed')
            counts=equal_tree(dict(model.state_dict()),dict(resume['model']),'model')
            if canonical(model.config.to_dict())!=canonical(resume['config']):
                raise ValueError('Native model configuration changed at restore')
            adam=optimizer_proof(model,optimizer,resume)
            saved=resume['training_state'];old_names=[g['name'] for g in resume['optimizer']['param_groups']]
            same_data=saved.get('data_config_sha256')==self.spec['data_config']['sha256']
            if same_data==self.spec['expect_data_reset']:
                raise ValueError('Boot/reset versus ordinary-resume data fingerprint differs')
            expected_plan=LearningRatePlan(**saved['lr_plan'])
            if 'geometry' not in old_names:
                expected_plan=migrate_geometry_plan(expected_plan,old_names,[g['name'] for g in optimizer.param_groups],
                    start_step=self.first,**self.spec['geometry_lr_options'])
            if canonical(lr_plan.to_config())!=canonical(expected_plan.to_config()):raise ValueError('LR phase or geometry warmup anchor changed')
            expected_loss=MultiCamera4RCLoss(**json.loads(Path(self.spec['loss_config']['path']).read_text())).to_config()
            if canonical(criterion.to_config())!=canonical(expected_loss):raise ValueError('Loss recipe changed')
            equal_tree(capture_rng(context.device),resume['rank_rng'][self.rank],'rank_rng')
            expected_cursor=(self.spec.get('expected_start_cursor',{'data_epoch':0,'batches_consumed':0})
                             if self.spec['expect_data_reset'] else
                             {key:saved[key] for key in ('data_epoch','batches_consumed')})
            actual_cursor={'data_epoch':data_epoch,'batches_consumed':batches_consumed}
            if actual_cursor!=expected_cursor:raise ValueError('Native data cursor differs')
            if saved.get('optimizer_updates')!=self.first:raise ValueError('Optimizer logical-step boundary differs')
            if not balance or balance['world_size']!=6 or balance['dataset_size']!=dataset_size:
                raise ValueError('Native balancing layout differs')
            if self.spec['expect_data_reset'] and (balance['start_epoch']!=0 or balance['start_batch']!=0):
                raise ValueError('New-stream balancing was not reset once')
            if scaler.is_enabled():raise ValueError('Expected native BF16 disabled GradScaler')
            equal_tree(scaler.state_dict(),resume.get('grad_scaler',{}),'GradScaler')
            execution.assert_checkpoint_safe()
            row.update(passed=True,physical_gpu_uuid=expected_uuid,model_values_exact=True,model=counts,
                optimizer=adam,LR_phase_exact=True,loss_recipe_exact=True,rank_RNG_exact=True,
                native_cursor_exact=True,cursor=actual_cursor,optimizer_updates=self.first,
                logical_step=self.first,lr_plan=lr_plan.to_config(),native_config_restored=True,
                model_config_exact=True,grad_scaler_exact=True)
        except Exception as exc:
            error=exc;row.update(passed=False,error={'type':type(exc).__name__,'message':str(exc)})
        row['seconds']=time.monotonic()-started
        publish(self.root/'startup'/f'rank_{self.rank:02d}.json',row)
        all_passed=context.all_true(error is None)
        if not all_passed:raise RuntimeError('At least one rank failed strict startup admission') from error

    def base(self):
        return {'schema':SCHEMA,'phase':self.spec['phase'],'rank':self.rank,'world_size':6,
                'gate_spec_sha256':self.spec_sha,'checkpoint_sha256':self.spec['checkpoint']['sha256'],
                'deployment_sha256':self.spec['deployment_sha256']}

    def after_update(self,*,step_before,skipped,norm,execution,targets,need_visibility,source_count):
        if not self.enabled or self.written>=self.spec['required_update_count']:return
        import torch
        if step_before!=self.first+self.written:raise ValueError('First-update logical-step mismatch')
        receipt=execution.last_receipt or {};fa4=receipt.get('fa4',{});kv=receipt.get('prepared_kv',{})
        geometry_only=('displacement_valid' in targets and not targets['displacement_valid'].any()
            and not targets.get('sparse_displacement_valid',torch.zeros((),device=self.context.device)).any()
            and not targets.get('sparse_trajectory_valid',torch.zeros((),device=self.context.device)).any()
            and not need_visibility and not targets.get('static_mask',torch.zeros((),device=self.context.device)).any())
        calls=receipt.get('actual_prepare_memory_calls',0)
        reason=('prepared_memory_used' if calls>0 else 'geometry_only' if geometry_only else
                'one_source' if source_count<=1 else 'native_8192MiB_budget_fallback')
        grouped={};floating=0
        tensors=list(self.model.state_dict().values())+[v for state in self.optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor)]
        for tensor in tensors:
            if tensor.is_floating_point():
                grouped.setdefault(str(tensor.device),[]).append(torch.isfinite(tensor).all());floating+=tensor.numel()
        finite=all(bool(torch.stack(flags).all()) for flags in grouped.values())
        rates=[g['lr'] for g in self.optimizer.param_groups]
        expected_rates=[base*self.plan.factor(step_before)*self.plan.group_factor(name,step_before)
                        for base,name in zip(self.plan.base_lrs,self.plan.group_names)]
        passed=(finite and not skipped and math.isfinite(float(norm)) and fa4.get('fa4_calls',0)>0
                and fa4.get('restored_original_sdpa') is True and clean_FA4_errors(fa4)
                and rates==expected_rates
                and kv.get('restored_original_config_object') is True
                and kv.get('parameters_buffers_and_keys_unchanged') is True)
        row=self.base();row.update(passed=passed,logical_step_before=step_before,step_after=step_before+1,
            model_Adam_all_floating_values_finite=finite,floating_values_checked=floating,
            optimizer_step_skipped=bool(skipped),grad_norm=float(norm),actual_route_receipt=receipt,
            prepared_KV_route=reason,source_count=int(source_count),
            LR_values=rates,LR_values_expected=expected_rates,LR_values_exact=rates==expected_rates)
        publish(self.root/'updates'/f'step_{step_before+1:08d}'/f'rank_{self.rank:02d}.json',row)
        self.written+=1
        if not self.context.all_true(passed):raise RuntimeError('Rank first-update finite/actual-route admission failed')


def assess_proofs(root,spec,spec_sha):
    errors=[];startup=[];updates=[]
    def read(path,rank):
        if not path.is_file():errors.append('Missing proof '+str(path));return {}
        row=json.loads(path.read_text())
        for key,value in [('schema',SCHEMA),('rank',rank),('world_size',6),('phase',spec['phase']),
            ('gate_spec_sha256',spec_sha),('checkpoint_sha256',spec['checkpoint']['sha256']),
            ('deployment_sha256',spec['deployment_sha256'])]:
            if row.get(key)!=value:errors.append(str(path)+': identity '+key)
        if row.get('passed') is not True:errors.append(str(path)+': failed')
        return row
    for rank in range(6):
        a=read(Path(root)/'startup'/f'rank_{rank:02d}.json',rank);startup.append(a)
        for key in ('model_values_exact','LR_phase_exact','loss_recipe_exact','rank_RNG_exact','native_cursor_exact','native_config_restored','model_config_exact','grad_scaler_exact'):
            if a.get(key) is not True:errors.append(f'rank{rank}: missing strict '+key)
        if a.get('optimizer',{}).get('all_named_Adam_values_exact') is not True:errors.append(f'rank{rank}: Adam not exact')
        if a.get('model',{}).get('elements',0)<=0 or a.get('optimizer',{}).get('existing_states_exact',0)<=0:
            errors.append(f'rank{rank}: empty strict-state scan')
        if a.get('physical_gpu_uuid')!=spec['ordered_gpu_uuids'][rank]:errors.append(f'rank{rank}: physical UUID')
        for offset in range(spec['required_update_count']):
            step=spec['checkpoint']['step']+offset+1
            b=read(Path(root)/'updates'/f'step_{step:08d}'/f'rank_{rank:02d}.json',rank);updates.append(b)
            if b.get('step_after')!=step or b.get('logical_step_before')!=step-1:errors.append(f'rank{rank}: logical update')
            if not b.get('model_Adam_all_floating_values_finite') or b.get('optimizer_step_skipped') is not False:
                errors.append(f'rank{rank}: finite/skipped update')
            receipt=b.get('actual_route_receipt',{})
            if receipt.get('fa4',{}).get('fa4_calls',0)<=0:errors.append(f'rank{rank}: no actual FA4')
            if receipt.get('fa4',{}).get('restored_original_sdpa') is not True or not clean_FA4_errors(receipt.get('fa4',{})):
                errors.append(f'rank{rank}: FA4 cleanup/error')
            if (receipt.get('prepared_kv',{}).get('restored_original_config_object') is not True
                    or receipt.get('prepared_kv',{}).get('parameters_buffers_and_keys_unchanged') is not True):
                errors.append(f'rank{rank}: KV cleanup/inventory')
            if b.get('floating_values_checked',0)<=0 or not math.isfinite(b.get('grad_norm',float('nan'))):
                errors.append(f'rank{rank}: empty/nonfinite full-state scan')
            if b.get('LR_values_exact') is not True or b.get('LR_values')!=b.get('LR_values_expected'):
                errors.append(f'rank{rank}: next-update LR differs')
            if not receipt.get('metadata_stats'):errors.append(f'rank{rank}: missing metadata observation')
            if b.get('prepared_KV_route') not in ('prepared_memory_used','geometry_only','one_source','native_8192MiB_budget_fallback'):
                errors.append(f'rank{rank}: unexplained KV fallback')
    return {'passed':not errors,'errors':errors,'startup_all6_exact':not errors,
            'routes_all6_verified':not errors,'startup_records':len(startup),'update_records':len(updates),
            'gate_spec_sha256':spec_sha,'proof_scope_no_statistical_quality_claim':True,
            'phase':spec['phase'],'checkpoint':spec['checkpoint']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--proof-root',type=Path,required=True)
    p.add_argument('--gate-spec',type=Path,required=True);p.add_argument('--gate-spec-sha256',required=True)
    p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    spec=read_spec(args.gate_spec,args.gate_spec_sha256)
    result=assess_proofs(args.proof_root,spec,args.gate_spec_sha256)
    publish(args.report,result);print(json.dumps(result));raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
