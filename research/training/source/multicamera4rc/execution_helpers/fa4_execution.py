"""Private-worker FA4 dispatch, preserving SDPA autocast and camera gradients.

Install the private namespace bootstrap before entering this context. Keep it
around BOTH forward and backward/checkpoint recomputation. Main training must
never import this experiment or inherit its temporary F.scaled_dot_product patch.
"""
from collections import Counter
from contextlib import contextmanager
import threading
import sys

import torch
import torch.nn.functional as F


_PATCH_LOCK = threading.RLock()
_LOW = (torch.float16, torch.bfloat16)
_CASTABLE = (torch.float32, torch.float16, torch.bfloat16)


def _output_tensor(result):
    """Pinned FA4 4.0.0b37 returns (out, lse), even return_lse=False.

    Selecting the original out keeps its FlashAttnFuncBackward connection.
    A Tensor return remains supported for existing fake/reference API tests.
    """
    if isinstance(result, torch.Tensor):
        return result, 'tensor_compatibility'
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], torch.Tensor):
        return result[0], 'tuple_output_lse'
    raise TypeError('Expected official FA4 (output Tensor, lse) tuple or a Tensor compatibility return')


def _eligible(q, k, v, mask, dropout, causal, gqa):
    if not all(t.is_cuda for t in (q, k, v)):
        return 'not_cuda'
    if mask is not None:
        return 'non_null_mask'
    if dropout != 0:
        return 'dropout'
    # SDPA and FA4 non-square causal alignment differ; native handles all causal.
    if causal:
        return 'causal'
    if gqa:
        return 'gqa_not_admitted'
    if torch.are_deterministic_algorithms_enabled():
        return 'deterministic_policy_not_admitted'
    if any(t.ndim != 4 for t in (q, k, v)):
        return 'rank'
    if not (q.device == k.device == v.device):
        return 'device_mismatch'
    if not (q.shape[:2] == k.shape[:2] == v.shape[:2]):
        return 'batch_or_heads'
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        return 'qkv_shape'
    if q.shape[-2] == 0 or k.shape[-2] == 0:
        return 'empty_sequence'
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        return 'last_stride'
    if any(d < 8 or d > 128 or d % 8 for d in (q.shape[-1], v.shape[-1])):
        return 'head_dimension'
    if any(t.dtype not in _CASTABLE for t in (q, k, v)):
        return 'dtype_not_autocastable'
    return None


def record_execution_exception(receipt, error):
    """Count exactly Torch's checkpoint control-flow stop, then re-raise upstream."""
    from torch.utils.checkpoint import _StopRecomputationError
    receipt['execution_errors'] += 1
    name = type(error).__module__ + '.' + type(error).__qualname__
    types = receipt.setdefault('exception_types', {})
    types[name] = types.get(name, 0) + 1
    key = ('expected_checkpoint_early_stops' if type(error) is _StopRecomputationError
           else 'unexpected_execution_errors')
    receipt[key] += 1


@contextmanager
def fa4_attention(flash_func=None, *, sample_limit=32):
    """Route supported calls to FA4; every other call retains original SDPA.

    CUDA autocast is an operator policy. Replacing F.sdpa bypasses that policy,
    so the dispatch explicitly casts eligible fp32/fp16/bf16 inputs to its active
    AMP dtype. Tensor.to preserves CastBackward; nothing is detached. Without
    autocast, any fp32 input falls back unchanged (e.g. the fp32 refiner).
    Unexpected FA4 execution errors surface; they are never concealed by fallback.
    """
    if flash_func is None:
        if not getattr(sys.modules.get('flash_attn'), '__fa4_private_parent__', False):
            raise RuntimeError('Activate the private FA4 namespace before importing its API')
        from flash_attn.cute import flash_attn_func
        flash_func = flash_attn_func
    receipt = {'variant': 'fa4', 'fa4_calls': 0, 'native_calls': 0,
               'native_reasons': {}, 'samples': [], 'execution_errors': 0,
               'expected_checkpoint_early_stops': 0, 'unexpected_execution_errors': 0, 'exception_types': {},
               'return_contracts': {},
               'scope': 'forward_backward_and_checkpoint_recomputation',
               'mask_bias_or_gradient_detached': False}
    reasons = Counter()
    with _PATCH_LOCK:
        original = F.scaled_dot_product_attention

        def dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, *, scale=None, enable_gqa=False):
            reason = _eligible(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa)
            amp = torch.is_autocast_enabled('cuda')
            target = torch.get_autocast_dtype('cuda') if amp else None
            if reason is None:
                if scale is not None and not isinstance(scale, (float, int)):
                    reason = 'scale_type'
                elif amp and target not in _LOW:
                    reason = 'unsupported_amp_dtype'
                elif not amp and not (query.dtype == key.dtype == value.dtype and query.dtype in _LOW):
                    reason = 'fp32_or_mixed_without_autocast'
            if reason is not None:
                receipt['native_calls'] += 1
                reasons[reason] += 1
                return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
            raw_dtypes = [str(t.dtype) for t in (query, key, value)]
            if amp:
                query, key, value = (t.to(dtype=target) for t in (query, key, value))
            # Use the exact caller's original scale for camera-feature Q/K;
            # never replace an explicit scale by augmented_width**(-0.5).
            effective_scale = query.shape[-1]**(-.5) if scale is None else scale
            if len(receipt['samples']) < sample_limit:
                receipt['samples'].append({
                    'query_shape_bhsd': list(query.shape), 'key_shape_bhsd': list(key.shape),
                    'value_shape_bhsd': list(value.shape), 'raw_dtypes': raw_dtypes,
                    'execution_dtypes': [str(t.dtype) for t in (query, key, value)],
                    'cuda_autocast_enabled': amp, 'softmax_scale': effective_scale,
                    'caller_scale_was_explicit': scale is not None})
            receipt['fa4_calls'] += 1
            try:
                result = flash_func(query.transpose(1, 2), key.transpose(1, 2),
                                    value.transpose(1, 2), softmax_scale=effective_scale,
                                    causal=False)
                output, contract = _output_tensor(result)
                receipt['return_contracts'][contract] = receipt['return_contracts'].get(contract, 0) + 1
                return output.transpose(1, 2)
            except BaseException as error:
                record_execution_exception(receipt, error)
                raise

        F.scaled_dot_product_attention = dispatch
        try:
            yield receipt
        finally:
            # Scoped monkeypatch is confined to this isolated worker process.
            # Reject concurrent foreign replacement instead of clobbering it.
            if F.scaled_dot_product_attention is not dispatch:
                raise RuntimeError('Foreign SDPA patch replaced private FA4 dispatch')
            F.scaled_dot_product_attention = original
            receipt['native_reasons'] = dict(reasons)
            receipt['restored_original_sdpa'] = F.scaled_dot_product_attention is original
