"""Optional execution scopes for an isolated069 trainer copy; native by default.

No optimizer, loss, DDP synchronization, LR, precision or checkpoint is replaced.
The decorator guarantees scope cleanup if a native training path raises.
"""
from contextlib import ExitStack
from contextvars import ContextVar
from functools import wraps
import hashlib
import os
from pathlib import Path
import sys

_CONTROLLERS = ContextVar('dfrt_future_execution_controllers', default=None)
MODES = ('native', 'fa4_metadata_prepared_kv')


def validate_resume_mode(saved, requested):
    previous = (saved or {}).get('identity', {}).get('mode', 'native')
    if previous not in MODES or requested not in MODES:
        raise ValueError('Unknown saved/requested execution mode')
    if previous != 'native' and requested != previous:
        raise ValueError('Resume must explicitly retain its saved execution mode; no silent default-native fallback')
    return {'saved_mode':previous,'requested_mode':requested,
            'explicit_initial_opt_in':previous=='native' and requested!='native'}


def validate_resume_identity(saved, current):
    if (saved or {}).get('identity', {}).get('mode','native')=='native':
        return
    prior=saved['identity']
    for key in ('implementation_sha256','runtime'):
        if not prior.get(key) or prior[key]!=current.get(key):
            raise ValueError('Saved execution '+key+' differs; ordinary resume requires the same pinned policy')


def execution_scope(function):
    @wraps(function)
    def scoped(*args, **kwargs):
        token = _CONTROLLERS.set([])
        try:
            return function(*args, **kwargs)
        finally:
            controllers = _CONTROLLERS.get()
            try:
                for controller in reversed(controllers):
                    controller.close(aborted=True)
            finally:
                _CONTROLLERS.reset(token)
    return scoped


class TrainingExecution:
    def __init__(self, model, mode='native', *, _test_contexts=None):
        if mode not in MODES:
            raise ValueError('Unknown optional training execution mode')
        self.model, self.mode = model, mode
        self.enabled = mode != 'native'
        self.stack = None
        self.logical_step = None
        self.microbatches = 0
        self.last_receipt = None
        self.factories = _test_contexts
        self.identity = {'mode': mode, 'source_base_commit': '069b1c28ce4763e2ac6a4f06dbc5b73c9ed3dc55',
                         'default_disabled': True, 'SAC_or_streaming': False}
        controllers = _CONTROLLERS.get()
        if controllers is None:
            raise RuntimeError('TrainingExecution requires the execution_scope cleanup decorator')
        controllers.append(self)
        if self.enabled:
            self.original_config = model.config
            if self.factories is None:
                expected = Path('/data8/zbf/training_optimization_20261104/private_envs/fa4_torch210_v1')
                if (Path(sys.prefix).resolve()!=expected
                        or not getattr(sys.modules.get('flash_attn'),'__fa4_private_parent__',False)
                        or not os.environ.get('DFRT_FUTURE_DEPLOYMENT_SHA256')):
                    raise ValueError('Execution opt-in requires the reviewed independent private bootstrap')
                from .execution_helpers.metadata_attention import metadata_attention_cache
                from .execution_helpers.fa4_execution import fa4_attention
                from .execution_helpers.decoder_kv_cache import decoder_kv_cache
                self.factories = {'metadata': metadata_attention_cache,
                    'fa4': fa4_attention, 'kv': lambda: decoder_kv_cache(model)}
                helpers = Path(__file__).parent/'execution_helpers'
                paths = [Path(__file__)] + [helpers/name for name in
                    ('metadata_attention.py','fa4_execution.py','decoder_kv_cache.py')]
                self.identity['implementation_sha256'] = {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
                self.identity['runtime'] = {'python':sys.executable,'prefix':sys.prefix,
                    'deployment_sha256':os.environ['DFRT_FUTURE_DEPLOYMENT_SHA256'],
                    'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'OpenEXR_file':getattr(sys.modules.get('OpenEXR'),'__file__',None)}
                self.identity['process_layout'] = {'LOCAL_RANK':os.environ.get('LOCAL_RANK'),
                    'RANK':os.environ.get('RANK'),'WORLD_SIZE':os.environ.get('WORLD_SIZE')}
                torch=sys.modules['torch'];interface=sys.modules['flash_attn.cute.interface']
                self.identity['runtime'].update(torch_version=str(torch.__version__),cuda_build=str(torch.version.cuda))
                for name,path in (('torch_C',torch._C.__file__),('fa4_interface',interface.__file__),
                                  ('OpenEXR',sys.modules['OpenEXR'].__file__),('Imath',sys.modules['Imath'].__file__)):
                    self.identity['runtime'][name+'_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            else:
                self.identity['injected_CPU_unit_contexts'] = True

    def before_microbatch(self, step):
        if not self.enabled:
            return
        if self.stack is None:
            stack = ExitStack()
            try:
                self.metadata = stack.enter_context(self.factories['metadata']())
                self.fa4 = stack.enter_context(self.factories['fa4']())
                self.kv = stack.enter_context(self.factories['kv']())
                owner = self.model.motion_decoder
                original_prepare = owner.prepare_memory
                had_prepare = 'prepare_memory' in owner.__dict__
                self.prepare_calls = 0
                def counted(*args, **kwargs):
                    self.prepare_calls += 1
                    return original_prepare(*args, **kwargs)
                def restore_prepare():
                    if had_prepare: owner.prepare_memory = original_prepare
                    else: del owner.prepare_memory
                owner.prepare_memory = counted
                stack.callback(restore_prepare)
            except BaseException:
                stack.close()
                raise
            self.stack, self.logical_step, self.microbatches = stack, step, 0
        elif self.logical_step != step:
            raise RuntimeError('An optimizer update escaped its execution scope')
        # Each microbatch has its own metadata lifetime, while all execution
        # scopes remain installed across every microbatch and its backward.
        self.metadata.new_step()
        self.microbatches += 1

    def finish_update(self):
        if not self.enabled:
            return None
        if self.stack is None:
            raise RuntimeError('Optimizer update completed without its execution scope')
        return self.close(aborted=False)

    def close(self, *, aborted):
        if self.stack is None:
            return self.last_receipt
        stack, self.stack = self.stack, None
        try:
            stack.close()
        finally:
            self.last_receipt = {'mode':self.mode,'logical_step_before_update':self.logical_step,
                'microbatches':self.microbatches,'aborted':bool(aborted),
                'metadata_stats':dict(self.metadata.stats),
                'fa4':{key:self.fa4.get(key) for key in ('fa4_calls','native_calls',
                    'native_reasons','execution_errors','expected_checkpoint_early_stops',
                    'unexpected_execution_errors','exception_types','return_contracts','restored_original_sdpa')},
                'prepared_kv':{key:self.kv.get(key) for key in ('cache_limit_mb',
                    'restored_original_config_object','parameters_buffers_and_keys_unchanged')},
                'actual_prepare_memory_calls':self.prepare_calls,
                'metadata_cleared_before_each_complete_microbatch_forward_backward':True,
                'native_DDP_reductions_accumulation_and_optimizer_unchanged':True}
            self.logical_step = None
        self.assert_checkpoint_safe()
        return self.last_receipt

    def assert_checkpoint_safe(self):
        if not self.enabled:
            return
        if self.stack is not None:
            raise RuntimeError('Native checkpoint attempted inside a partial optimizer update')
        if self.model.config is not self.original_config or self.model.config.reuse_decoder_kv:
            raise RuntimeError('Native checkpoint configuration was not restored')

    def checkpoint_metadata(self):
        self.assert_checkpoint_safe()
        return {'identity':dict(self.identity),'last_completed_update':self.last_receipt}
