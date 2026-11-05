"""Scoped use of the existing native differentiable decoder K/V cache only."""
from contextlib import contextmanager
from dataclasses import is_dataclass, replace


def _inventory(model):
    return (tuple((name, id(value)) for name, value in model.named_parameters()),
            tuple((name, id(value)) for name, value in model.named_buffers()),
            tuple(model.state_dict()))


@contextmanager
def decoder_kv_cache(model):
    """Enable existing same-decode reuse; retain its original 8192MiB fallback.

    No projection/memory/attention function is replaced, no Tensor is detached,
    and no cache persists across optimizer updates. Existing native code decides
    whether each input fits the unchanged limit. Keep this scope around full
    forward/backward/checkpoint recomputation. No combined reuse candidate here.
    """
    original = model.config
    if not is_dataclass(original):
        raise TypeError('Require the frozen native dataclass model config')
    before = original.to_dict()
    if (before.get('reuse_decoder_kv') is not False
            or before.get('reuse_shared_computation') is not False
            or before.get('decoder_kv_cache_limit_mb') != 8192):
        raise ValueError('Require native KV=False, shared=False and unchanged 8192MiB cap')
    if any(block.cross_detach_kv for block in model.motion_decoder.cross_blocks):
        raise ValueError('This training candidate requires native differentiable K/V')
    inventory = _inventory(model)
    active = replace(original, reuse_decoder_kv=True)
    during = active.to_dict()
    changed = [key for key in before if before[key] != during[key]]
    if changed != ['reuse_decoder_kv']:
        raise RuntimeError('Candidate changes more than the existing KV reuse boolean')
    receipt = {'variant': 'decoder_kv_cache', 'before_config': before,
        'active_config': during, 'changed_config_keys': changed,
        'cache_limit_mb': 8192, 'reuse_shared_computation': False,
        'scope': 'one native decode/autograd graph; never across optimizer updates',
        'native_budget_and_one_source_fallback_retained': True,
        'projection_attention_dropout_and_rng_implementations_unchanged': True,
        'cache_activation_is_input_dependent': True, 'production_adoption': False}
    model.config = active
    try:
        yield receipt
    finally:
        model.config = original
        receipt['restored_original_config_object'] = model.config is original
        receipt['after_config'] = model.config.to_dict()
        receipt['parameters_buffers_and_keys_unchanged'] = _inventory(model) == inventory
        if receipt['after_config'] != before or not receipt['parameters_buffers_and_keys_unchanged']:
            raise RuntimeError('Decoder KV context did not restore the original execution state')
