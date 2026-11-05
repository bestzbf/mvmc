#!/usr/bin/env python3
"""CPU-only immutable5->6 checkpoint publication. Never writes a live pointer.

Publication authority is COMMITTED.json plus a successful verify command.
The frozen v14 payload lineage remains a planning record (checkpoint_written=False).
A separate new_data_stream_reset branch explicitly archives, then ends the old
stream; its only valid first use is native069 --migrate-data-config with the bound
new configuration. Neither branch changes model/Adam/LR/loss/retained RNG values.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import uuid

import expand_checkpoint_v14 as expand

SCHEMA = 'immutable-five-to-six-checkpoint-v21'
MODES = ('zero_skip_old_stream', 'new_data_stream_reset')
EXPECTED_EXPAND_SHA = '3975ebec67cc0eb8f8406e020adff94edd9029e8d063ee763db2d67d3af80c29'


def sha(path):
    return expand.sha256_file(path)


def identity(path):
    value = Path(path).lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('Require a regular non-symlink file: ' + str(path))
    return dict(device=value.st_dev, inode=value.st_ino, bytes=value.st_size,
                mtime_ns=value.st_mtime_ns)


def load_pinned(path, digest, *, mmap=True):
    import torch
    path = Path(path).absolute()
    before = identity(path)
    if sha(path) != digest:
        raise ValueError('Pinned SHA mismatch: ' + str(path))
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=mmap)
    if identity(path) != before:
        raise ValueError('Pinned file changed during load')
    return payload, before


def source_data_contract(path, digest):
    value = dict(path=str(Path(path).absolute()), sha256=digest, identity=identity(path))
    require_source_data(value)
    return value


def require_source_data(value):
    if value is not None and (identity(value['path']) != value['identity']
            or sha(value['path']) != value['sha256']):
        raise ValueError('External admitted original data configuration changed')


def new_stream_plan(state, target_data_sha256, source_data=None):
    expand._layout(state)
    previous = state.get('data_config_sha256')
    require_source_data(source_data)
    if previous is None:
        if source_data is None:
            raise ValueError('Legacy checkpoint requires external admitted original data SHA256')
        previous = source_data['sha256']
        origin = 'external_admitted_config_legacy_checkpoint_missing_fingerprint'
    else:
        origin = 'checkpoint_training_state'
        if source_data is not None and previous != source_data['sha256']:
            raise ValueError('Saved and externally admitted source data SHA disagree')
    if (not isinstance(previous, str) or len(previous) != 64
            or any(c not in '0123456789abcdef' for c in previous)):
        raise ValueError('Invalid archived original data SHA256')
    if (not isinstance(target_data_sha256, str) or len(target_data_sha256) != 64
            or any(c not in '0123456789abcdef' for c in target_data_sha256)
            or target_data_sha256 == previous):
        raise ValueError('New stream requires a distinct explicit target data SHA256')
    balance = deepcopy(state['load_balancing'])
    balance.update(world_size=6, start_epoch=0, start_batch=0,
                   historical_padding_sample_offset=0)
    return dict(schema='explicit-new-data-stream-reset-v21', old_world_size=5,
        world_size=6, data_epoch=0, batches_consumed=0, load_balancing=balance,
        reset_data_cursor=True, old_stream_continuity_claimed=False,
        source_global_samples_consumed=expand.consumed_samples(state),
        new_stream_global_samples_consumed=0,
        source_data_sha256=previous, source_data_sha256_origin=origin,
        external_source_data_contract=deepcopy(source_data),
        target_data_sha256=target_data_sha256,
        required_first_native_flag='--migrate-data-config')


def plan_for(source, mode, target_data_sha256=None, source_data=None):
    state = expand.validate_source_payload(source)
    if mode == 'zero_skip_old_stream':
        if target_data_sha256 is not None:
            raise ValueError('Old-stream mode cannot bind a new data stream')
        return expand.plan_expansion(state)
    if mode == 'new_data_stream_reset':
        return new_stream_plan(state, target_data_sha256, source_data)
    raise ValueError('Explicit expansion mode is required')


def transformed(source, envelope, source_sha256, mode, target_data_sha256=None, source_data=None):
    plan = plan_for(source, mode, target_data_sha256, source_data)
    if mode == 'zero_skip_old_stream':
        return expand.transform_payload(source, envelope, source_sha256=source_sha256)
    expand.validate_new_rank_rng(envelope, source, source_sha256)
    result = dict(source)
    result['training_state'] = expand.migrated_training_state(source['training_state'], plan)
    result['rank_rng'] = list(source['rank_rng']) + [deepcopy(envelope['state'])]
    result[expand.LINEAGE] = dict(schema=plan['schema'], mode=mode,
        source_checkpoint_sha256=source_sha256, source_step=source['step'],
        physical_gpus=[0, 1, 2, 5, 6, 7], plan=deepcopy(plan),
        old_training_state=deepcopy(source['training_state']),
        old_stream_consumed_sample_ledger=expand.consumed_samples(source['training_state']),
        new_rank_rng_provenance=deepcopy(envelope['provenance']),
        recipe_migration='pending_native069_explicit_reset_once', checkpoint_written=False)
    verify_payload(source, result, envelope, source_sha256, mode, target_data_sha256, source_data)
    return result


def verify_payload(source, result, envelope, source_sha256, mode, target_data_sha256=None, source_data=None):
    plan = plan_for(source, mode, target_data_sha256, source_data)
    if mode == 'zero_skip_old_stream':
        return expand.verify_transformed_payload(source, result, envelope,
                                                 source_sha256=source_sha256)
    expand.validate_new_rank_rng(envelope, source, source_sha256)
    if result.keys() != source.keys() | {expand.LINEAGE}:
        raise ValueError('Unexpected top-level mutation')
    expand._equal(expand.migrated_training_state(source['training_state'], plan),
                  result['training_state'], 'new_stream_training_state')
    expand._equal(source['rank_rng'], result['rank_rng'][:5], 'retained5_rng')
    if len(result['rank_rng']) != 6:
        raise ValueError('Require six explicit RNG states')
    expand._equal(envelope['state'], result['rank_rng'][5], 'new_rank5_rng')
    for key in source:
        if key not in ('training_state', 'rank_rng'):
            expand._equal(source[key], result[key], key)
    expected = dict(schema=plan['schema'], mode=mode,
        source_checkpoint_sha256=source_sha256, source_step=source['step'],
        physical_gpus=[0, 1, 2, 5, 6, 7], plan=deepcopy(plan),
        old_training_state=deepcopy(source['training_state']),
        old_stream_consumed_sample_ledger=expand.consumed_samples(source['training_state']),
        new_rank_rng_provenance=deepcopy(envelope['provenance']),
        recipe_migration='pending_native069_explicit_reset_once', checkpoint_written=False)
    expand._equal(expected, result[expand.LINEAGE], 'new_stream_lineage')
    return plan


def check_rng_proof(proof, envelope, envelope_sha256):
    checks = proof.get('checks', {})
    if (proof.get('completed') is not True or
            any(checks.get(key) is not True for key in (
                'initialized_state_exact', 'random_draws_exact', 'advanced_state_exact',
                'original_process_rng_restored_exact')) or
            proof.get('envelope', {}).get('sha256') != envelope_sha256):
        raise ValueError('Require exact guarded native CUDA RNG roundtrip receipt')
    expand._equal(proof.get('provenance'), envelope['provenance'], 'RNG_proof_provenance')
    for key in ('native_capture_restore', 'cuda_state_generated_by_real_cuda_generator',
                'restore_random_draws_exact', 'guard_verified'):
        if envelope['provenance'].get(key) is not True:
            raise ValueError('Missing genuine native CUDA RNG proof: ' + key)
    if envelope['provenance'].get('test_fixture_only'):
        raise ValueError('Synthetic CPU fixture is forbidden in publication')


def tensor_inventory(value):
    import torch
    if isinstance(value, torch.Tensor):
        return dict(tensors=1, elements=value.numel(), bytes=value.numel()*value.element_size())
    result = dict(tensors=0, elements=0, bytes=0)
    children = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else ()
    for child in children:
        for key, count in tensor_inventory(child).items():
            result[key] += count
    return result


def verify_file(source_path, source_sha256, rng_path, rng_sha256, candidate_path,
                mode, target_data_sha256=None, proof_path=None, proof_sha256=None, source_data=None):
    import torch
    source, source_id = load_pinned(source_path, source_sha256)
    envelope, rng_id = load_pinned(rng_path, rng_sha256)
    candidate_id = identity(candidate_path)
    candidate = torch.load(candidate_path, map_location='cpu', mmap=True, weights_only=True)
    if proof_path is not None:
        proof_id = identity(proof_path)
        if sha(proof_path) != proof_sha256:
            raise ValueError('RNG proof SHA mismatch')
        check_rng_proof(json.loads(Path(proof_path).read_text()), envelope, rng_sha256)
        if identity(proof_path) != proof_id:
            raise ValueError('RNG proof changed')
    plan = verify_payload(source, candidate, envelope, source_sha256, mode, target_data_sha256, source_data)
    digest = sha(candidate_path)
    require_source_data(source_data)
    if (identity(source_path) != source_id or identity(rng_path) != rng_id
            or identity(candidate_path) != candidate_id or torch.cuda.is_initialized()):
        raise ValueError('File identity changed or CPU verifier initialized CUDA')
    return dict(exact_all_values_verified=True, independent_fresh_load=True,
        cuda_initialized=False, source_identity=source_id, candidate_identity=candidate_id,
        checkpoint_sha256=digest, plan=plan, source_step=source['step'],
        model=tensor_inventory(source['model']), optimizer=tensor_inventory(source['optimizer']))


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def link_exact(source, destination, expected_sha256):
    """Reconcile NFS's 'operation succeeded but RPC returned error' ambiguity."""
    before = identity(source)
    try:
        os.link(source, destination, follow_symlinks=False)
        reconciled = False
    except OSError:
        try:
            after = identity(destination)
        except (OSError, ValueError):
            raise
        if ((after['device'], after['inode']) != (before['device'], before['inode'])
                or sha(destination) != expected_sha256):
            raise FileExistsError('Different or unverified destination; never overwrite: '+str(destination))
        reconciled = True
    after = identity(destination)
    if ((after['device'], after['inode']) != (before['device'], before['inode'])
            or sha(destination) != expected_sha256):
        raise ValueError('Publication did not retain exact staged inode/hash')
    return reconciled


def write_exclusive(path, writer):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        writer(stream)
        stream.flush()
        os.fchmod(stream.fileno(), 0o444)
        os.fsync(stream.fileno())
    return identity(path)


def verify_bundle(bundle, expected_marker_sha256=None):
    bundle = Path(bundle)
    marker_path = bundle/'COMMITTED.json'
    marker_id = identity(marker_path)
    if expected_marker_sha256 is not None and sha(marker_path) != expected_marker_sha256:
        raise ValueError('Commit marker differs from the pinned publication receipt')
    marker = json.loads(marker_path.read_text())
    if marker.get('schema') != SCHEMA or marker.get('checkpoint_filename') != 'checkpoint.pt':
        raise ValueError('Unknown immutable bundle marker')
    checkpoint = bundle/'checkpoint.pt'
    if sha(checkpoint) != marker['checkpoint_sha256']:
        raise ValueError('Committed checkpoint hash mismatch')
    result = verify_file(marker['source'], marker['source_sha256'], marker['new_rank_rng'],
        marker['new_rank_rng_sha256'], checkpoint, marker['mode'], marker['target_data_sha256'],
        marker['rng_proof'], marker['rng_proof_sha256'], marker.get('source_data'))
    if identity(marker_path) != marker_id:
        raise ValueError('Commit marker changed during independent verification')
    # Strict: unsupported NFS directory fsync is an error, never claimed durable.
    fsync_directory(bundle);fsync_directory(bundle.parent)
    result.update(bundle=str(bundle.absolute()), committed=True,
        directory_fsync_completed=True, marker_sha256=sha(marker_path))
    return result


def publish(source_path, source_sha256, rng_path, rng_sha256, proof_path, proof_sha256,
            bundle, mode, target_data_sha256=None, *, source_data=None, _test_independent_verifier=None):
    import torch
    source, source_id = load_pinned(source_path, source_sha256)
    envelope, _ = load_pinned(rng_path, rng_sha256)
    proof_id = identity(proof_path)
    if sha(proof_path) != proof_sha256:
        raise ValueError('RNG proof SHA mismatch')
    check_rng_proof(json.loads(Path(proof_path).read_text()), envelope, rng_sha256)
    result = transformed(source, envelope, source_sha256, mode, target_data_sha256, source_data)
    bundle = Path(bundle).absolute()
    bundle.mkdir(mode=0o700)  # Require a new unique directory; never reuse/overwrite.
    fsync_directory(bundle.parent);fsync_directory(bundle)
    stage = bundle/('.checkpoint.'+uuid.uuid4().hex+'.tmp')
    write_exclusive(stage, lambda stream:torch.save(result, stream))
    if _test_independent_verifier is not None:
        verification = _test_independent_verifier(stage)
    else:
        command = [sys.executable, '-B', str(Path(__file__).resolve()), 'verify-candidate',
            '--source', str(Path(source_path).absolute()), '--source-sha256', source_sha256,
            '--new-rank-rng', str(Path(rng_path).absolute()), '--new-rank-rng-sha256', rng_sha256,
            '--rng-proof', str(Path(proof_path).absolute()), '--rng-proof-sha256', proof_sha256,
            '--candidate', str(stage), '--mode', mode]
        if target_data_sha256:command += ['--target-data-sha256', target_data_sha256]
        if source_data:command += ['--source-data-config', source_data['path'], '--source-data-sha256', source_data['sha256']]
        completed = subprocess.run(command, check=True, capture_output=True, text=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
        verification = json.loads(completed.stdout)
    if (verification.get('exact_all_values_verified') is not True
            or verification.get('independent_fresh_load') is not True
            or verification['source_identity'] != source_id
            or identity(source_path) != source_id or identity(proof_path) != proof_id):
        raise ValueError('Independent verifier or immutable source/proof identity failed')
    checkpoint = bundle/'checkpoint.pt'
    require_source_data(source_data)
    checkpoint_reconciled = link_exact(stage, checkpoint, verification['checkpoint_sha256'])
    fsync_directory(bundle)
    marker = dict(schema=SCHEMA, checkpoint_filename='checkpoint.pt',
        checkpoint_sha256=verification['checkpoint_sha256'],
        source=str(Path(source_path).absolute()), source_sha256=source_sha256,
        source_identity=source_id, source_step=source['step'],
        new_rank_rng=str(Path(rng_path).absolute()), new_rank_rng_sha256=rng_sha256,
        rng_proof=str(Path(proof_path).absolute()), rng_proof_sha256=proof_sha256,
        mode=mode, target_data_sha256=target_data_sha256, source_data=source_data,
        requires_native_data_migration=mode=='new_data_stream_reset',
        first_native_required_flag='--migrate-data-config' if mode=='new_data_stream_reset' else None,
        plan=verification['plan'], independent_verification=verification,
        payload_checkpoint_written_flag_is_frozen_planning_lineage=True,
        durability_claim='Successful publish/verify return also requires final directory fsync; marker alone is visibility evidence')
    marker_bytes = (json.dumps(marker,indent=2,allow_nan=False)+'\n').encode()
    marker_stage = bundle/('.commit.'+uuid.uuid4().hex+'.tmp')
    write_exclusive(marker_stage, lambda stream:stream.write(marker_bytes))
    marker_reconciled = link_exact(marker_stage, bundle/'COMMITTED.json', hashlib.sha256(marker_bytes).hexdigest())
    fsync_directory(bundle);fsync_directory(bundle.parent)
    # Remove only our temporary links after both final links are verified/fsynced.
    stage.unlink();marker_stage.unlink();fsync_directory(bundle)
    return dict(committed=True,bundle=str(bundle),checkpoint=str(checkpoint),
        checkpoint_sha256=verification['checkpoint_sha256'],mode=mode,
        marker=str(bundle/'COMMITTED.json'),marker_sha256=sha(bundle/'COMMITTED.json'),
        source_identity=source_id,directory_fsync_completed=True,
        nfs_ambiguous_link_reconciled=dict(checkpoint=checkpoint_reconciled,marker=marker_reconciled),
        independent_verification=verification,live_pointer_modified=False)


def main(argv=None):
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Require explicit CUDA_VISIBLE_DEVICES empty; CPU publisher only')
    helper = Path(expand.__file__)
    if sha(helper) != EXPECTED_EXPAND_SHA:
        raise ValueError('Frozen expansion helper changed')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('dry-run','publish','verify-candidate','verify'))
    parser.add_argument('--source',type=Path);parser.add_argument('--source-sha256')
    parser.add_argument('--new-rank-rng',type=Path);parser.add_argument('--new-rank-rng-sha256')
    parser.add_argument('--rng-proof',type=Path);parser.add_argument('--rng-proof-sha256')
    parser.add_argument('--bundle',type=Path);parser.add_argument('--candidate',type=Path)
    parser.add_argument('--marker-sha256')
    parser.add_argument('--mode',choices=MODES);parser.add_argument('--target-data-sha256')
    parser.add_argument('--source-data-config',type=Path);parser.add_argument('--source-data-sha256')
    args = parser.parse_args(argv)
    source_data = (source_data_contract(args.source_data_config, args.source_data_sha256)
                   if args.source_data_config is not None else None)
    if args.command=='verify':
        if not args.marker_sha256:
            raise ValueError('verify requires the pinned --marker-sha256 from publication')
        output=verify_bundle(args.bundle,args.marker_sha256)
    elif args.command=='dry-run':
        payload,pinned_id=load_pinned(args.source,args.source_sha256)
        state=expand.validate_source_payload(payload)
        plan=plan_for(payload,args.mode,args.target_data_sha256,source_data)
        output=dict(schema=SCHEMA,checkpoint_written=False,source_identity=pinned_id,
            source_step=payload['step'],mode=args.mode,plan=plan,
            source_model=tensor_inventory(payload['model']),
            source_optimizer=tensor_inventory(payload['optimizer']),
            real_newrank_rng_verified=False,source_unchanged=identity(args.source)==pinned_id,
            cuda_initialized=torch.cuda.is_initialized())
    elif args.command=='verify-candidate':
        output=verify_file(args.source,args.source_sha256,args.new_rank_rng,args.new_rank_rng_sha256,
            args.candidate,args.mode,args.target_data_sha256,args.rng_proof,args.rng_proof_sha256,source_data)
    else:
        if not all((args.source,args.source_sha256,args.new_rank_rng,args.new_rank_rng_sha256,
                    args.rng_proof,args.rng_proof_sha256,args.bundle,args.mode)):
            raise ValueError('Publication requires every pinned source/RNG/proof/bundle/mode argument')
        output=publish(args.source,args.source_sha256,args.new_rank_rng,args.new_rank_rng_sha256,
            args.rng_proof,args.rng_proof_sha256,args.bundle,args.mode,args.target_data_sha256,source_data=source_data)
    print(json.dumps(output,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
