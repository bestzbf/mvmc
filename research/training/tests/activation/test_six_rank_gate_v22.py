"""CPU parser/strict equality regressions; fixture records are NOT GPU evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
import torch
from multicamera4rc.six_rank_gate_v22 import assess_proofs,equal_tree,optimizer_proof,publish,SCHEMA


class ProofTests(unittest.TestCase):
    def setUp(self):
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise RuntimeError('CPU-only tests')
    def test_all_tensor_values_dtype_and_metadata_are_strict(self):
        value={'tensor':torch.arange(12).reshape(3,4),'step':3,'state':[torch.ones(())]}
        self.assertEqual(equal_tree(deepcopy(value),value)['elements'],13)
        changed=deepcopy(value);changed['tensor'][2,3]+=1
        with self.assertRaises(ValueError):equal_tree(changed,value)
        changed=deepcopy(value);changed['state'][0]=changed['state'][0].double()
        with self.assertRaises(ValueError):equal_tree(changed,value)
        with self.assertRaises(ValueError):equal_tree({**value,'step':4},value)
    def fixtures(self,root):
        spec={'phase':'boot','checkpoint':{'sha256':'a'*64,'step':140300},'deployment_sha256':'b'*64,
              'required_update_count':1,'ordered_gpu_uuids':['GPU-'+str(i) for i in range(6)]}
        for rank in range(6):
            common=dict(schema=SCHEMA,phase='boot',rank=rank,world_size=6,gate_spec_sha256='c'*64,
                checkpoint_sha256='a'*64,deployment_sha256='b'*64,passed=True)
            startup={**common,'physical_gpu_uuid':spec['ordered_gpu_uuids'][rank],
                'optimizer':{'all_named_Adam_values_exact':True,'existing_states_exact':3},'model':{'elements':10}}
            for key in ('model_values_exact','LR_phase_exact','loss_recipe_exact','rank_RNG_exact','native_cursor_exact','native_config_restored','model_config_exact','grad_scaler_exact'):
                startup[key]=True
            update={**common,'logical_step_before':140300,'step_after':140301,'optimizer_step_skipped':False,
                'model_Adam_all_floating_values_finite':True,'floating_values_checked':10,'grad_norm':.5,
                'LR_values_exact':True,'LR_values':[.01,.1,.2,.1],'LR_values_expected':[.01,.1,.2,.1],
                'prepared_KV_route':'geometry_only','actual_route_receipt':{'fa4':{'fa4_calls':2,
                'restored_original_sdpa':True,'execution_errors':0,'expected_checkpoint_early_stops':0,
                'unexpected_execution_errors':0,'exception_types':{}},'metadata_stats':{'hits':0},
                'prepared_kv':{'restored_original_config_object':True,'parameters_buffers_and_keys_unchanged':True}}}
            publish(root/'startup'/f'rank_{rank:02d}.json',startup)
            publish(root/'updates'/'step_00140301'/f'rank_{rank:02d}.json',update)
        return spec
    def test_six_records_required_no_rank0_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=self.fixtures(root)
            self.assertTrue(assess_proofs(root,spec,'c'*64)['passed'])
            (root/'startup'/'rank_05.json').unlink()
            self.assertFalse(assess_proofs(root,spec,'c'*64)['passed'])
    def test_actual_route_and_restore_failures_are_not_flags_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=self.fixtures(root);path=root/'updates'/'step_00140301'/'rank_04.json'
            row=json.loads(path.read_text());row['actual_route_receipt']['fa4']['fa4_calls']=0
            path.write_text(json.dumps(row));self.assertFalse(assess_proofs(root,spec,'c'*64)['passed'])
    def test_existing_verdict_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gate.json';publish(path,{'first':True})
            with self.assertRaises(OSError):publish(path,{'replacement':True})
            self.assertEqual(json.loads(path.read_text()),{'first':True})
    def test_actual_native069_named_Adam_proof_detects_moment_corruption(self):
        from multicamera4rc.optim import optimizer_groups,load_optimizer_state
        model=torch.nn.Module();model.backbone=torch.nn.Module()
        model.backbone.pretrained=torch.nn.Linear(3,3)
        model.head=torch.nn.Linear(3,3);model.motion=torch.nn.Linear(3,3)
        model.trajectory_refiner=torch.nn.Linear(3,3)
        old=torch.optim.AdamW(optimizer_groups(model,.01,.1,.2))
        for name,parameter in model.named_parameters():
            if name!='motion.bias':parameter.grad=torch.ones_like(parameter)
        old.step();old.zero_grad(set_to_none=True)
        source={'model':deepcopy(model.state_dict()),'optimizer':deepcopy(old.state_dict())}
        migrated=torch.optim.AdamW(optimizer_groups(model,.01,.1,.2,geometry_lr=.1))
        load_optimizer_state(migrated,source['optimizer'],model,model_state=source['model'])
        # An independent immutable reference is necessary for this CPU fixture:
        # Torch's CPU loader may alias source tensors, unlike real GPU moments.
        evidence=deepcopy(source)
        row=optimizer_proof(model,migrated,evidence)
        self.assertEqual(row['existing_states_exact'],7);self.assertEqual(row['absent_states_not_created'],1)
        migrated.state[model.head.weight]['exp_avg'][0,0]+=.01
        with self.assertRaises(ValueError):optimizer_proof(model,migrated,evidence)
    def test_only_exact_checkpoint_control_exception_is_allowed(self):
        from torch.utils.checkpoint import _StopRecomputationError
        from multicamera4rc.execution_helpers.fa4_execution import record_execution_exception
        from multicamera4rc.six_rank_gate_v22 import clean_FA4_errors
        receipt={'execution_errors':0,'expected_checkpoint_early_stops':0,
                 'unexpected_execution_errors':0,'exception_types':{}}
        record_execution_exception(receipt,_StopRecomputationError())
        self.assertTrue(clean_FA4_errors(receipt));self.assertEqual(receipt['execution_errors'],1)
        class DifferentStop(_StopRecomputationError):pass
        record_execution_exception(receipt,DifferentStop())
        self.assertFalse(clean_FA4_errors(receipt))
        self.assertEqual(receipt['unexpected_execution_errors'],1)


if __name__=='__main__':unittest.main(verbosity=2)
