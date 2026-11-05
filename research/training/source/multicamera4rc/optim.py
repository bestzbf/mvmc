"""Stable named parameter grouping and lossless, checked Adam migrations."""

from collections.abc import Mapping


def _group_name(name, separate_geometry=False):
    if name.startswith('trajectory_refiner.'):
        return 'refinement'
    if name.startswith('backbone.pretrained.') and not name.endswith('camera_relation_weight'):
        return 'backbone'
    if separate_geometry and name.startswith(('head.', 'cam_dec.')):
        return 'geometry'
    return 'heads'


def _named_parameters(model, separate_geometry=False):
    """Deduplicate tied parameters, refusing aliases with different LR owners."""
    parameters, owners = [], {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        owner = _group_name(name, separate_geometry)
        previous = owners.get(id(parameter))
        if previous is not None:
            if previous[1] != owner:
                raise ValueError(f'shared parameter crosses optimizer groups: {previous[0]} / {name}')
            continue
        owners[id(parameter)] = (name, owner)
        parameters.append((name, parameter))
    return parameters


def parameter_groups(model, *, separate_geometry=False):
    grouped = {name: [] for name in ('backbone', 'heads', 'refinement', 'geometry')}
    for name, parameter in _named_parameters(model, separate_geometry):
        grouped[_group_name(name, separate_geometry)].append((name, parameter))
    result = [('backbone', grouped['backbone']), ('heads', grouped['heads'])]
    if grouped['refinement']:
        result.append(('refinement', grouped['refinement']))
    if separate_geometry:
        if not grouped['geometry']:
            raise ValueError('separate geometry LR requires trainable head./cam_dec. parameters')
        result.append(('geometry', grouped['geometry']))
    return result


def optimizer_groups(model, backbone_lr, head_lr, refinement_lr, *, geometry_lr=None):
    rates = dict(backbone=backbone_lr, heads=head_lr, refinement=refinement_lr, geometry=geometry_lr)
    return [dict(params=[p for _, p in entries], param_names=[n for n, _ in entries],
                 lr=rates[name], name=name)
            for name, entries in parameter_groups(model, separate_geometry=geometry_lr is not None)]


def _validate_adam_state(state, parameter, name, *, amsgrad=False):
    """Validate moments before torch's positional loader can mutate optimizer state."""
    import torch
    if not isinstance(state, Mapping):
        raise ValueError(f'Adam state must be a mapping: {name}')
    if not state:
        return
    if not {'step', 'exp_avg', 'exp_avg_sq'} <= set(state):
        raise ValueError(f'incomplete Adam moments: {name}')
    if amsgrad and 'max_exp_avg_sq' not in state:
        raise ValueError(f'incomplete AMSGrad moments: {name}')
    if set(state) - {'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'}:
        raise ValueError(f'unsupported Adam state fields: {name}')
    for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
        if key not in state:
            continue
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
            raise ValueError(f'Adam moment shape mismatch: {name}/{key}')
        if not value.is_floating_point():
            raise ValueError(f'Adam moment must be floating point: {name}/{key}')
    step = state['step']
    if isinstance(step, torch.Tensor):
        if step.numel() != 1:
            raise ValueError(f'Adam step must be scalar: {name}')
        step = step.item()
    import math
    if not isinstance(step, (int, float)) or not math.isfinite(step) or step < 0:
        raise ValueError(f'invalid Adam step: {name}')


def _legacy_names(model, saved_groups, model_state):
    """Reconstruct the audited legacy helper's order, never match by shape alone.

    Callers must explicitly opt into this known legacy grouping contract. The
    checkpoint model's ordered names, group names, canonical sequential IDs,
    shapes and complete trainable inventory must all corroborate it.
    """
    if not isinstance(model_state, Mapping):
        raise ValueError('legacy optimizer requires the checkpoint model state as ordering evidence')
    expected = parameter_groups(model)
    if [group.get('name') for group in saved_groups] != [name for name, _ in expected]:
        raise ValueError('legacy optimizer group names/order do not match the known contract')
    parameters = _named_parameters(model)
    names = [name for name, _ in parameters]
    name_set = set(names)
    if [name for name in model_state if name in name_set] != names:
        raise ValueError('legacy checkpoint model parameter order cannot be established')
    for name, parameter in parameters:
        if name not in model_state or getattr(model_state[name], 'shape', None) != parameter.shape:
            raise ValueError(f'legacy checkpoint model shape mismatch: {name}')
    flat_ids = [key for group in saved_groups for key in group['params']]
    if flat_ids != list(range(len(names))):
        raise ValueError('legacy optimizer IDs do not follow deterministic serialized order')
    result = []
    for group, (_, entries) in zip(saved_groups, expected):
        if len(group['params']) != len(entries):
            raise ValueError('legacy optimizer parameter count changed; mapping cannot be proved')
        result.append([name for name, _ in entries])
    return result


def load_optimizer_state(optimizer, state, model, *, model_state=None, allow_legacy=False):
    """Restore Adam by parameter name, optionally splitting geometry from heads.

    Preserve every existing moment and source group's hyperparameters. Splitting
    geometry initially inherits the old heads LR; its multiplier belongs to the
    existing LR plan, not to a freshly restarted schedule. New checkpoints carry
    parameter names inside optimizer groups and need no legacy reconstruction.
    """
    if not isinstance(state, Mapping) or set(state) != {'state', 'param_groups'}:
        raise ValueError('expected a standard Adam optimizer state dictionary')
    saved = state['param_groups']
    if not isinstance(saved, list) or not saved or not isinstance(state['state'], Mapping):
        raise ValueError('invalid Adam optimizer state')
    group_names = [group.get('name') for group in saved]
    if any(not isinstance(name, str) for name in group_names) or len(set(group_names)) != len(group_names):
        raise ValueError('optimizer groups require unique, explicit names')
    if set(group_names) != {name for name, _ in parameter_groups(model, separate_geometry='geometry' in group_names)}:
        raise ValueError('unsupported saved optimizer group layout')
    has_names = ['param_names' in group for group in saved]
    if any(has_names) and not all(has_names):
        raise ValueError('partially named optimizer cannot prove parameter identity')
    if all(has_names):
        saved_names = [group['param_names'] for group in saved]
    elif allow_legacy:
        saved_names = _legacy_names(model, saved, model_state)
    else:
        raise ValueError('unnamed legacy optimizer requires explicit allow_legacy and model ordering evidence')
    current = optimizer.state_dict()
    separate = any(group.get('name') == 'geometry' for group in optimizer.param_groups)
    expected = parameter_groups(model, separate_geometry=separate)
    if len(current['param_groups']) != len(expected):
        raise ValueError('current optimizer group layout differs from the model')
    parameters = dict(_named_parameters(model, separate))
    source_ids, source_owners, source_by_group = {}, {}, {}
    for group, names in zip(saved, saved_names):
        if not isinstance(names, (list, tuple)) or len(names) != len(group['params']):
            raise ValueError('optimizer parameter names/count mismatch')
        source_by_group[group['name']] = group
        for name, key in zip(names, group['params']):
            if not isinstance(name, str) or name in source_ids or key in source_owners:
                raise ValueError('duplicate/invalid optimizer parameter identity')
            if name not in parameters:
                raise ValueError(f'optimizer contains unknown or frozen parameter: {name}')
            owner = _group_name(name, 'geometry' in group_names)
            if group['name'] != owner:
                raise ValueError(f'optimizer group does not own named parameter: {name}')
            source_ids[name] = key
            source_owners[key] = name
            _validate_adam_state(state['state'].get(key, {}), parameters[name], name,
                                 amsgrad=group.get('amsgrad', False))
    if set(source_ids) != set(parameters) or set(state['state']) - set(source_owners):
        raise ValueError('optimizer parameter/state inventory is incomplete or contains foreign entries')
    new_groups, new_state = [], {}
    for current_group, (name, entries) in zip(current['param_groups'], expected):
        names = [n for n, _ in entries]
        if current_group.get('name') != name or current_group.get('param_names') != names:
            raise ValueError('current optimizer does not preserve canonical parameter names')
        source_name = 'heads' if name == 'geometry' and name not in source_by_group else name
        if source_name not in source_by_group:
            raise ValueError(f'unsupported optimizer group migration: {name}')
        source = source_by_group[source_name]
        if any(old_name not in saved_names[group_names.index(source_name)] for old_name in names):
            raise ValueError(f'optimizer parameter ownership changed for group {name}')
        updated = dict(source, name=name, params=current_group['params'], param_names=names)
        new_groups.append(updated)
        for parameter_name, new_id in zip(names, current_group['params']):
            old_id = source_ids[parameter_name]
            if old_id in state['state']:
                new_state[new_id] = state['state'][old_id]
    # All proof/shape checks above precede the sole mutation.
    optimizer.load_state_dict(dict(state=new_state, param_groups=new_groups))
    return dict(old_group_names=group_names, new_group_names=[name for name, _ in expected],
                parameter_count=len(parameters), restored_states=len(new_state),
                geometry_split='geometry' not in group_names and separate,
                legacy_order_reconstructed=not all(has_names))
