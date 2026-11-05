"""Step-scoped integer/bool relation metadata, with native float math untouched.

This isolated candidate modifies process-local function bindings only. Encoder
layers allocate new token metadata, so exact-object caching does NOT promise
cross-layer encoder reuse. Motion rounds and checkpoint replay can reuse it.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import inspect
import sys

_ACTIVE = False


class RelationMetadataCache:
    def __init__(self):
        self.entries = {}
        self.stats = {"hits": 0, "misses": 0, "uncacheable": 0, "steps": 0}

    def new_step(self):
        """Call before every complete forward/backward, never between them."""
        self.entries.clear()
        self.stats["steps"] += 1

    @staticmethod
    def _token(tensor):
        if tensor is None:
            return ("none",)
        import torch
        allowed = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if tensor.dtype not in allowed or tensor.layout != torch.strided or tensor.requires_grad:
            return None
        try:
            version = tensor._version
        except RuntimeError:  # Inference tensors do not expose version counters.
            return None
        return (id(tensor), tensor.untyped_storage().data_ptr(), version,
            tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(),
            str(tensor.device), tensor.dtype, tensor.layout)

    def _memo(self, kind, tensors, extra, compute):
        tokens = tuple(self._token(t) for t in tensors)
        if any(token is None for token in tokens):
            self.stats["uncacheable"] += 1
            return compute()
        key = (kind, tokens, extra)
        if key in self.entries:
            self.stats["hits"] += 1
            return self.entries[key][1]
        self.stats["misses"] += 1
        result = compute()
        # Strong input references prevent object/storage identity reuse. Alias
        # writes increment the shared version counter and therefore miss.
        self.entries[key] = (tuple(tensors), result)
        return result

    def features(self, query_camera, key_camera):
        import torch
        import torch.nn.functional as F
        def compute():
            labels = torch.cat([query_camera, key_camera], dim=1)
            cameras, inverse = torch.unique(labels, return_inverse=True)
            return labels, cameras, F.one_hot(inverse, num_classes=len(cameras))
        return self._memo("features", (query_camera,key_camera), None, compute)

    def single_camera(self, key_camera, key_valid):
        import torch
        def compute():
            valid = key_valid if key_valid is not None else torch.ones_like(key_camera,dtype=torch.bool)
            reference = key_camera.gather(1, valid.long().argmax(-1,keepdim=True))
            return bool(((key_camera == reference) | ~valid).all())
        return self._memo("single_camera", (key_camera,key_valid), None, compute)

    def all_true(self, valid):
        return self._memo("all_true", (valid,), None, lambda:bool(valid.all()))

    def safe_keys(self, native_function, valid, num_keys):
        return self._memo("safe_keys", (valid,), num_keys, lambda:native_function(valid,num_keys))


def _native_namespace(native, cache):
    """Fail closed if the pinned native metadata expressions have changed."""
    source = inspect.getsource(native)
    replacements = (
        ("    labels = torch.cat([query_camera, key_camera], dim=1)\n"
         "    cameras, inverse = torch.unique(labels, return_inverse=True)\n"
         "    hot = F.one_hot(inverse, num_classes=len(cameras)).to(q.dtype)",
         "    labels, cameras, hot_integer = _metadata_cache.features(query_camera, key_camera)\n"
         "    hot = hot_integer.to(q.dtype)", 1),
        ("    valid_keys = key_valid if key_valid is not None else torch.ones_like(key_camera, dtype=torch.bool)\n"
         "    reference = key_camera.gather(1, valid_keys.long().argmax(-1, keepdim=True))\n"
         "    if ((key_camera == reference) | ~valid_keys).all():",
         "    if _metadata_cache.single_camera(key_camera, key_valid):", 1),
        ("safe_valid.all()", "_metadata_cache.all_true(safe_valid)", 3),
        ("key_valid.all()", "_metadata_cache.all_true(key_valid)", 1),
    )
    transformed = source
    for old,new,count in replacements:
        if transformed.count(old) != count:
            raise ValueError("Frozen native attention metadata source changed; refuse transformation")
        transformed = transformed.replace(old,new)
    namespace = dict(vars(native), _metadata_cache=cache)
    exec(compile(transformed,str(native.__file__)+"[metadata-only]","exec"),namespace)
    original_safe = native._safe_key_valid
    namespace["_safe_key_valid"] = lambda valid,num_keys: cache.safe_keys(original_safe,valid,num_keys)
    return namespace, hashlib.sha256(source.encode()).hexdigest()


def _bindings(function):
    # Include already-imported DINO aliases, plus modules imported during scope.
    for name,module in tuple(sys.modules.items()):
        if name.startswith("multicamera4rc") and module is not None:
            for attribute,value in tuple(vars(module).items()):
                if value is function:
                    yield module,attribute


@contextmanager
def metadata_attention_cache():
    """One step, including backward/checkpoint replay; never cache float state.

    Re-enter per update, or call cache.new_step() before each next forward.
    Generated functions preserve native branches, QKV, alpha math, padding,
    explicit scale, dropout, masks and checkpoint calls. No value hashing or
    CUDA-to-CPU camera comparison is used.
    """
    global _ACTIVE
    if _ACTIVE:
        raise ValueError("Do not nest metadata attention caches")
    import multicamera4rc.attention as native
    cache = RelationMetadataCache()
    namespace, digest = _native_namespace(native,cache)
    cache.feature_attention = namespace["_feature_relation_attention"]
    cache.relation_attention = namespace["relation_attention"]
    cache.receipt = {"native_attention_source_sha256":digest,"cached_data":"integer/bool metadata only",
        "exact_object_and_version_keys":True,"cross_layer_encoder_hits_claimed":False,
        "learned_features_or_weights_cached":False,"production_adoption":False}
    original, replacement = native.relation_attention, cache.relation_attention
    _ACTIVE = True
    try:
        for module,attribute in tuple(_bindings(original)):
            setattr(module,attribute,replacement)
        cache.new_step()
        yield cache
    finally:
        for module,attribute in tuple(_bindings(replacement)):
            setattr(module,attribute,original)
        _ACTIVE = False
        cache.entries.clear()
        cache.receipt["restored_exactly"] = native.relation_attention is original
        if not cache.receipt["restored_exactly"]:
            raise RuntimeError("Native relation attention binding failed to restore")
