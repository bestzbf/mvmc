"""CPU lifetime/accumulation/exception tests; fake FA4 is not kernel evidence."""
from contextlib import contextmanager
from dataclasses import dataclass,replace
import unittest
import torch
from torch import nn
from training_execution_future import TrainingExecution,execution_scope,validate_resume_mode,validate_resume_identity


@dataclass(frozen=True)
class Config:
    reuse_decoder_kv:bool=False


class Decoder(nn.Linear):
    def prepare_memory(self,value):return value


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.config=Config();self.motion_decoder=Decoder(3,3)
    def forward(self,value):return self.motion_decoder(self.motion_decoder.prepare_memory(value))


def contexts(model,events,active):
    class Meta:
        def __init__(self):self.stats={'steps':0,'hits':0}
        def new_step(self):self.stats['steps']+=1
    @contextmanager
    def metadata():
        events.append('metadata_enter');yield Meta();events.append('metadata_exit')
    @contextmanager
    def fa4():
        active[0]=True;events.append('fa4_enter')
        receipt={'fa4_calls':0,'native_calls':0}
        try:yield receipt
        finally:active[0]=False;receipt['restored_original_sdpa']=True;events.append('fa4_exit')
    @contextmanager
    def kv():
        original=model.config;model.config=replace(original,reuse_decoder_kv=True);events.append('kv_enter')
        receipt={'cache_limit_mb':8192}
        try:yield receipt
        finally:model.config=original;receipt['restored_original_config_object']=True;events.append('kv_exit')
    return {'metadata':metadata,'fa4':fa4,'kv':kv}


class LifecycleTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def run_updates(self,enabled):
        torch.manual_seed(1901);model=Model();opt=torch.optim.AdamW(model.parameters());events=[];active=[False]
        grad_context=[];model.motion_decoder.weight.register_hook(lambda grad:grad_context.append(active[0]))
        @execution_scope
        def body():
            controller=TrainingExecution(model,'fa4_metadata_prepared_kv' if enabled else 'native',
                _test_contexts=contexts(model,events,active))
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                for micro in range(2):
                    controller.before_microbatch(step)
                    (model(torch.ones(4,3)*(micro+1)).square().mean()/2).backward()
                self.assertEqual(active[0],enabled)
                opt.step();controller.finish_update();controller.assert_checkpoint_safe()
                self.assertFalse(model.config.reuse_decoder_kv);self.assertFalse(active[0])
            return controller
        controller=body()
        return model,opt,controller,events,grad_context
    def test_accumulation_and_native_default_equivalence(self):
        a,oa,ca,ea,ga=self.run_updates(False);b,ob,cb,eb,gb=self.run_updates(True)
        self.assertFalse(ea);self.assertEqual(ga,[False]*4);self.assertEqual(gb,[True]*4)
        for p,q in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(p,q))
        self.assertEqual(cb.last_receipt['microbatches'],2)
        self.assertEqual(cb.last_receipt['metadata_stats']['steps'],2)
        self.assertEqual(cb.last_receipt['actual_prepare_memory_calls'],2)
        self.assertNotIn('prepare_memory',b.motion_decoder.__dict__)
        self.assertEqual(eb.count('fa4_enter'),2);self.assertEqual(eb.count('fa4_exit'),2)
    def test_exception_partial_update_restores_before_propagating(self):
        model=Model();events=[];active=[False];old=model.config
        @execution_scope
        def body():
            controller=TrainingExecution(model,'fa4_metadata_prepared_kv',_test_contexts=contexts(model,events,active))
            controller.before_microbatch(0)
            with self.assertRaises(RuntimeError):controller.assert_checkpoint_safe()
            raise ValueError('original failure')
        with self.assertRaisesRegex(ValueError,'original failure'):body()
        self.assertFalse(active[0]);self.assertIs(model.config,old)
        self.assertNotIn('prepare_memory',model.motion_decoder.__dict__)
    def test_mode_resume_is_explicit_and_never_silent_disable(self):
        self.assertTrue(validate_resume_mode({},'fa4_metadata_prepared_kv')['explicit_initial_opt_in'])
        saved={'identity':{'mode':'fa4_metadata_prepared_kv'}}
        with self.assertRaises(ValueError):validate_resume_mode(saved,'native')
        self.assertEqual(validate_resume_mode(saved,'fa4_metadata_prepared_kv')['requested_mode'],saved['identity']['mode'])
    def test_resume_implementation_and_runtime_are_pinned(self):
        current={'mode':'fa4_metadata_prepared_kv','implementation_sha256':{'helper':'sha'},
                 'runtime':{'torch_version':str(torch.__version__)},'process_layout':{'RANK':'0'}}
        saved={'identity':current}
        validate_resume_identity(saved,{**current,'process_layout':{'RANK':'5'}})
        validate_resume_identity({},current)
        for key in ('implementation_sha256','runtime'):
            with self.assertRaises(ValueError):validate_resume_identity(saved,{**current,key:{'changed':'bad'}})


if __name__=='__main__':unittest.main(verbosity=2)
