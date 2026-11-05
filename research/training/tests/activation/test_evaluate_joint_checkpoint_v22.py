"""CPU contract/queue tests only; no actual GT scoring or GPU launch."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

import evaluate_joint_checkpoint_v22 as evaluate

TEMPLATE=Path(os.environ['EVAL_V22_TEMPLATE'])


def fixture(job):
    names=['backbone.pretrained.weight','track_head.weight','refiner.weight','head.weight']
    model={name:torch.arange(6.).reshape(2,3)+i for i,name in enumerate(names)}
    groups=[];adam={}
    for index,(group,name) in enumerate(zip(evaluate.GROUPS,names)):
        groups.append(dict(name=group,params=[index],param_names=[name],lr=0.,weight_decay=.05,
            betas=(.9,.999),eps=1e-8,amsgrad=False,foreach=None,fused=None,capturable=False))
        adam[index]=dict(step=torch.tensor(140501.),exp_avg=torch.full((2,3),.1+index),
            exp_avg_sq=torch.full((2,3),.2+index))
    source=Path(job['repo'])/'multicamera4rc/schedule.py'
    spec=importlib.util.spec_from_file_location('_test_eval_schedule22',source)
    module=importlib.util.module_from_spec(spec);import sys;sys.modules[spec.name]=module;spec.loader.exec_module(module)
    plan=module.migrate_geometry_plan(module.LearningRatePlan(**job['expected_state']['lr_plan']),
        evaluate.GROUPS[:3],evaluate.GROUPS,start_step=140500,multiplier=2.,warmup_steps=100)
    from types import SimpleNamespace
    plan.apply(SimpleNamespace(param_groups=groups),140500)
    state=deepcopy(job['expected_state']);state.update(world_size=6,effective_batch_size=6,
        batch_size=1,gradient_accumulation=1,data_epoch=0,batches_consumed=1,optimizer_updates=140501,
        data_config_sha256=job['admitted_training_data_sha256'],geometry_lr_options=dict(multiplier=2.,warmup_steps=100),
        data_migration=dict(step=140500,previous_sha256=None,new_sha256=job['admitted_training_data_sha256'],
            cursor_policy='explicit_reset_once',model_optimizer_rng_lr_preserved=True),lr_plan=plan.to_config())
    state['load_balancing'].update(world_size=6,start_epoch=0,start_batch=0,historical_padding_sample_offset=0)
    state['loss_config'].update(depth_objective_options=dict(relative_weight=.05,log_bias_weight=.02,temporal_scale_weight=.01),
        depth_gradient_scales=[1,2,4])
    state['execution_optimizations']=dict(identity=dict(mode='fa4_metadata_prepared_kv',source_base_commit=evaluate.COMMIT,
        implementation_sha256=job['admitted_execution_implementation_sha256'],
        runtime={'deployment_sha256':job['eval_runtime_manifest_sha256']},resume_policy={'saved_mode':'native'},process_layout={'RANK':'0'}),
        last_completed_update=dict(aborted=False,fa4=dict(restored_original_sdpa=True),prepared_kv=dict(restored_original_config_object=True)))
    return dict(architecture='multicamera4rc',format_version=2,step=140501,model=model,
        config=job['model_config'],training_state=state,optimizer=dict(state=adam,param_groups=groups),
        rank_rng=[{} for _ in range(6)],grad_scaler={})


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.root=Path(tempfile.mkdtemp(prefix='eval22_test_'))
        self.job=evaluate.read(TEMPLATE);self.value=fixture(self.job)
        self.cp=self.root/'step.pt';torch.save(self.value,self.cp)
        self.template=self.root/'template.json';self.template.write_text(json.dumps(self.job))
    def tearDown(self):pass # outer harness removes owned NFS fixtures after mmap process exit
    def test_fourgroup_native_checkpoint_admission_and_ordinary_resume_policy(self):
        job=deepcopy(self.job);job['science_contract']=evaluate.make_contract(self.value,job)
        evaluate.check_contract(self.value,job,140501)
        value=deepcopy(self.value)
        value['training_state']['execution_optimizations']['identity']['resume_policy']={'saved_mode':'fa4_metadata_prepared_kv'}
        evaluate.check_contract(value,job,140501)
        identity=evaluate.file_identity(self.cp)
        job.update(control_dir=str(self.root/'check'),validation_checkpoint=str(self.cp),validation_step=140501,
            validation_label='refinement_140501',checkpoint_identity=identity)
        path=self.root/'job.json';path.write_text(json.dumps(job))
        controller=evaluate.worker_class(job)(path)
        controller.check(self.cp,140501)
        self.assertFalse(torch.cuda.is_initialized())
    def test_scientific_mutations_are_rejected(self):
        job=deepcopy(self.job);job['science_contract']=evaluate.make_contract(self.value,job)
        def changed_data(v):v['training_state']['data_config_sha256']='0'*64
        def changed_loss(v):v['training_state']['loss_config']['depth_objective_options']['relative_weight']=0.
        def changed_phase(v):v['training_state']['lr_plan']['start_step']=140500
        def changed_mode(v):v['training_state']['execution_optimizations']['identity']['mode']='native'
        def changed_config(v):v['config']['reuse_decoder_kv']=True
        def three_groups(v):v['optimizer']['param_groups'].pop()
        def wrong_rng(v):v['rank_rng'].pop()
        def wrong_lr(v):v['optimizer']['param_groups'][3]['lr']*=2
        for mutate in (changed_data,changed_loss,changed_phase,changed_mode,changed_config,three_groups,wrong_rng,wrong_lr):
            value=deepcopy(self.value);mutate(value)
            with self.assertRaises(ValueError,msg=mutate.__name__):evaluate.check_contract(value,job,140501)
    def test_exact_gpu_pool_and_all_original_GT_contracts(self):
        evaluate.validate_pool(self.job)
        for pool in ([4],[3,7],[0,1,2,3,4,5,6,7]):
            job=deepcopy(self.job);job['validation_gpus']=pool
            with self.assertRaises(ValueError):evaluate.validate_pool(job)
        import sys;sys.path.insert(0,self.job['repo'])
        from scripts.migrate_dynamic_validation import build_validation_contract
        self.assertEqual(build_validation_contract(self.job),self.job['validation_contract'])
        self.assertEqual(set(self.job['validation_contract']['suites']),{'kubric','native_dynamic','static','variable','syn4d'})
    def test_queue_bind_once_pin_deduplicate_and_result_collection(self):
        queue=evaluate.EvaluationQueue(self.template,self.root/'control')
        queue.bind_checkpoint(self.cp,evaluate.digest(self.cp))
        row=queue.enqueue(self.cp,140501);self.assertEqual(queue.enqueue(self.cp,140501),row)
        self.assertEqual(len(queue.pending()),1);self.assertIn(row['checkpoint'],queue.protected_paths())
        with self.assertRaises(ValueError):queue.bind_checkpoint(self.cp,'0'*64)
        class Fake:
            pid=os.getpid()
            def poll(self):return None
        with patch.object(evaluate.subprocess,'Popen',return_value=Fake()) as popen:
            queue.poll();self.assertEqual(popen.call_count,1)
            command=popen.call_args.args[0];environment=popen.call_args.kwargs['env']
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'],'');self.assertIn('--controller',command)
            queue.poll();self.assertEqual(popen.call_count,1)
        entry=evaluate.read(Path(row['job']).parent/'entry.json');job=evaluate.read(row['job'])
        result=dict(checkpoint_identity=row['checkpoint_identity'],quality_passed=True,comparison={},
            record=dict(step=140501,checkpoint=row['checkpoint'],validation_contract=job['validation_contract']))
        evaluate.atomic_json(Path(row['job']).parent/'result.json',result)
        self.assertEqual(len(queue.poll()),1);self.assertEqual(queue.pending(),[])
        queue.lock.close()
    def test_failed_or_wrong_identity_result_surfaces(self):
        queue=evaluate.EvaluationQueue(self.template,self.root/'control')
        queue.bind_checkpoint(self.cp,evaluate.digest(self.cp));row=queue.enqueue(self.cp,140501)
        job=evaluate.read(row['job'])
        result=dict(checkpoint_identity={'wrong':True},quality_passed=True,record=dict(step=140501,
            checkpoint=row['checkpoint'],validation_contract=job['validation_contract']))
        evaluate.atomic_json(Path(row['job']).parent/'result.json',result)
        with self.assertRaises(ValueError):queue.poll()
        result['checkpoint_identity']=row['checkpoint_identity'];result['quality_passed']=False
        evaluate.atomic_json(Path(row['job']).parent/'result.json',result)
        with self.assertRaises(RuntimeError):queue.poll()
        queue.lock.close()


if __name__=='__main__':unittest.main(verbosity=2)
