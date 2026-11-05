"""Prepare, verify, or explicitly execute a controller-only Kinoptic handover.

Preparation never schedules execution. The running launcher/ranks finish their
original segment and unchanged full validation before new data is activated.
Only the data traversal resets; the original checkpoint and LR phase are used.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from multicamera4rc.datasets.mixture import read_data_config
from multicamera4rc.datasets.adapters.panoptic import heldout_sequence_reason
from multicamera4rc.datasets.sampling import source_quotas
from multicamera4rc.quality import compare_errors, error_panel
from scripts.continue_refinement_training import RefinementController
from scripts.cutover_native_motion_stereo import alive, proc, write_json
from scripts.intrinsics_trial import owned_tree
from scripts.migrate_static_validation import require_training_child, training_ranks
from scripts.migrate_dynamic_validation import (SUITES, build_validation_contract,
    ExpandedDynamicController, verify_suite_report)
from scripts.query_pair_main_admission import (clean_runtime, continuation_health,
    require_cursor_progress, verify_files)
from scripts.qualify_kinoptic_dynamic import _runtime_sources
from scripts.pidfd_compat import pidfd_open, pidfd_send_signal, probe_pidfd_support


CRITICAL_RUNTIME_FILES = (
    'multicamera4rc/train.py', 'multicamera4rc/model.py', 'multicamera4rc/losses.py',
    'multicamera4rc/joint_geometry.py', 'multicamera4rc/cross_camera.py',
    'multicamera4rc/query_pair.py', 'multicamera4rc/schedule.py',
    'multicamera4rc/checkpoint.py', 'multicamera4rc/config.py', 'multicamera4rc/distributed.py',
    'scripts/continue_refinement_training.py', 'scripts/continue_quality_recovery.py',
)
GT_SOURCE_FILES = (
    'multicamera4rc/datasets/adapters/panoptic_kinoptic.py',
    'multicamera4rc/datasets/adapters/kinoptic_geometry.py',
    'multicamera4rc/datasets/adapters/kinoptic_stereo_validation.py',
    'multicamera4rc/datasets/adapters/kinoptic_rgb_validation.py',
    'multicamera4rc/datasets/adapters/panoptic.py',
    'multicamera4rc/datasets/adapters/base.py',
    'multicamera4rc/datasets/bridge.py', 'multicamera4rc/datasets/partial_dense.py',
    'multicamera4rc/datasets/dense.py', 'multicamera4rc/datasets/augmentation.py',
    'multicamera4rc/datasets/mixture.py', 'multicamera4rc/datasets/registry.py',
    'scripts/audit_kinoptic_dynamic.py', 'scripts/audit_kinoptic_hd_sync.py',
    'scripts/build_kinoptic_mixture.py',
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transition_contract(original, candidate):
    """Preserve actual source weights; report integer-quota rounding explicitly."""
    if ({k:v for k,v in original.items() if k != 'sources'}
            != {k:v for k,v in candidate.items() if k != 'sources'}
            or original.get('supervision') != 'dense'
            or original.get('num_queries', 0) != 0
            or original.get('sampling') != 'balanced_coverage'):
        raise ValueError('only Panoptic sources may change; dense full-pixel settings are fixed')
    before, after = original['sources'], candidate['sources']
    if any(isinstance(s.get('weight', 1.), bool)
            or not math.isfinite(float(s.get('weight', 1.))) or float(s.get('weight', 1.)) <= 0
            for s in before+after):
        raise ValueError('all source weights must be finite and positive')
    indices = [i for i,s in enumerate(before) if s['adapter'] == 'panoptic']
    if len(indices) != 1 or any(s['adapter'] == 'panoptic_kinoptic' for s in before):
        raise ValueError('expected exactly one original camera-only Panoptic source')
    i = indices[0]
    family = [s for s in after if s['adapter'] in ('panoptic', 'panoptic_kinoptic')]
    if (not any(s['adapter'] == 'panoptic_kinoptic' for s in family)
            or len(family) not in (1, 2) or after[:i] != before[:i]
            or after[i+len(family):] != before[i+1:]
            or after[i:i+len(family)] != family):
        raise ValueError('non-Panoptic source contents or source ordering changed')
    old = before[i]
    weight = float(old.get('weight', 1.))
    if not math.isclose(sum(float(s.get('weight', 1.)) for s in family), weight,
                        rel_tol=0, abs_tol=1e-14):
        raise ValueError('original Panoptic total weight changed')
    covered = set()
    for source in family:
        if any(source.get(k) != old.get(k) for k in ('split', 'cameras', 'num_times', 'strides', 'layouts')):
            raise ValueError('original Panoptic camera/time layout changed')
        names = source.get('sequences')
        if (not isinstance(names, list) or not names or len(set(names)) != len(names)
                or any(not isinstance(name, str) for name in names) or covered & set(names)):
            raise ValueError('Panoptic recipes need nonempty disjoint capture whitelists')
        covered.update(names)
        if source['adapter'] == 'panoptic':
            if (source.get('dense_kind') != 'camera_only' or source['root'] != old['root']
                    or source.get('options', {}) != old.get('options', {})):
                raise ValueError('fallback must preserve the original camera-only reader/options')
        elif source.get('dense_kind') != 'geometry':
            raise ValueError('Kinoptic supplies measured geometry, not material-point motion GT')
        else:
            old_options, options = old.get('options', {}), source.get('options', {})
            mutable = {'cache_dir','exclude_sequences','exclude_videos'}
            if any(options.get(k) != value for k,value in old_options.items() if k not in mutable):
                raise ValueError('original HD image/calibration reader options changed')
            if not set(options.get('exclude_videos', [])) <= set(old_options.get('exclude_videos', [])):
                raise ValueError('new unreviewed HD video exclusions were introduced')
    excluded = set(old.get('options', {}).get('exclude_sequences') or [])
    expected = (set(old['sequences']) if 'sequences' in old else
        {p.name for p in Path(old['root']).iterdir() if p.is_dir()
         and (p/f'calibration_{p.name}.json').is_file()})
    expected = {name for name in expected-excluded if not heldout_sequence_reason(
        name, old.get('options', {}).get('sequence_aliases'))}
    if covered != expected:
        raise ValueError('original Panoptic capture restrictions or complete fallback coverage changed')
    old_q = source_quotas([s.get('weight', 1.) for s in before], original['epoch_size'])
    new_q = source_quotas([s.get('weight', 1.) for s in after], candidate['epoch_size'])
    changes = []
    for oi, source in enumerate(before):
        if oi == i:
            continue
        ni = oi if oi < i else oi+len(family)-1
        a, b = sum(old_q[:oi]), sum(new_q[:ni])
        if (a, old_q[oi]) != (b, new_q[ni]):
            changes.append(dict(source=source.get('name', source['adapter']),
                old_interval=[a,a+old_q[oi]], new_interval=[b,b+new_q[ni]],
                old_quota=old_q[oi], new_quota=new_q[ni]))
    return dict(passed=True, intentional_data_migration=True,
        data_cursor_policy='reset_once_after_original_validation',
        non_panoptic_sources_unchanged=True, non_panoptic_weights_unchanged=True,
        old_panoptic_weight=weight, old_panoptic_quota=old_q[i],
        new_panoptic_quotas=new_q[i:i+len(family)],
        integer_quota_changes=changes, reset_model_optimizer_lr_scaler_rng=False,
        new_sources=[dict(adapter=s['adapter'], captures=len(s['sequences']), weight=s['weight']) for s in family])


def gt_gate(path, base_config, candidate_config, runtime):
    """The old one-timepoint/file audit is deliberately insufficient."""
    proof = read(path)
    flags = ('passed', 'multi_time_audit_passed', 'independent_rgb_roi_check_passed',
             'target_view_occlusion_check_passed')
    if (proof.get('kind') != 'panoptic_kinoptic_dynamic_gt_admission'
            or proof.get('schema_version') != 1 or proof.get('failures') != []
            or any(proof.get(flag) is not True for flag in flags)
            or proof.get('base_config_sha256') != sha(base_config)
            or proof.get('candidate_config_sha256') != sha(candidate_config)):
        raise ValueError('a current multi-time independent RGB/ROI/occlusion GT admission is required')
    measured_sources = [s for s in read_data_config(candidate_config)['sources']
                        if s['adapter'] == 'panoptic_kinoptic']
    capture_lists = [source.get('sequences') for source in measured_sources]
    if any(not isinstance(names, list) or not names
           or any(not isinstance(name, str) or not name for name in names)
           or len(set(names)) != len(names) for names in capture_lists):
        raise ValueError('every geometry source requires an explicit capture whitelist')
    captures = {name for names in capture_lists for name in names}
    qualified = proof.get('qualified_captures')
    if (not isinstance(qualified, list) or not qualified
            or any(not isinstance(name, str) or not name for name in qualified)
            or len(set(qualified)) != len(qualified) or not captures <= set(qualified)):
        raise ValueError('every geometry capture must appear in proof.qualified_captures')
    reader_options = proof.get('reader_options', {})
    def guarded(options):
        return (options.get('occlusion_guard') is True
                and options.get('rgb_verification') == 'stereo'
                and options.get('rgb_edge_guard') is True)
    if (not measured_sources or not isinstance(reader_options, dict) or not guarded(reader_options)
            or any(not guarded(source.get('options', {})) for source in measured_sources)):
        raise ValueError('candidate and audited reader_options must explicitly enable occlusion_guard, stereo RGB verification and rgb_edge_guard')
    sources = proof.get('source_sha256', {})
    if not all(str(Path(runtime)/name) in sources for name in GT_SOURCE_FILES):
        raise ValueError('GT audit must bind the actual reader and dense-geometry source files')
    artifacts = proof.get('artifacts_sha256', {})
    if not artifacts:
        raise ValueError('GT admission has no immutable independent audit artifacts')
    verify_files(sources)
    verify_files(artifacts)
    # Follow the audit's transitive policy (all 49 files in this runtime), not
    # merely the minimum entrypoints listed above. Added GT helpers must not
    # become an unaudited dependency by omission from a hand-written proof.
    current_sources = {str((Path(runtime)/name).resolve()): expected
                       for name,expected in _runtime_sources(Path(runtime)).items()}
    if any(sources.get(name) != expected for name,expected in current_sources.items()):
        raise ValueError('GT proof must freeze every current runtime GT pipeline source')
    reports = proof.get('dynamic_audit_reports')
    if (not isinstance(reports, list) or not reports or len(set(reports)) != len(reports)
            or any(not isinstance(p, str) or p not in artifacts for p in reports)):
        raise ValueError('current dynamic audit reports must be bound by artifacts_sha256')
    measured_frames = {name: set() for name in captures}
    for report_path in reports:
        report = read(report_path)
        if (report.get('kind') != 'kinoptic_dynamic_human_region_diagnostics_v1'
                or report.get('status') != 'completed'
                or report.get('source_hashes_unchanged') is not True
                or not guarded(report.get('adapter_options', {}))):
            raise ValueError('the actual completed dynamic audit must use the admitted stereo/occlusion policy')
        reported_sources = report.get('source_sha256', {})
        normalized_sources = {str((Path(runtime)/name).resolve()): expected
                              for name, expected in reported_sources.items()}
        if not all(str(Path(runtime)/name) in normalized_sources for name in GT_SOURCE_FILES):
            raise ValueError('dynamic audit must freeze every reader, stereo filter and GT pipeline source')
        if normalized_sources != current_sources:
            raise ValueError('dynamic audit must freeze the complete current GT pipeline source map')
        for full_path, expected in normalized_sources.items():
            if sources.get(full_path) != expected:
                raise ValueError('dynamic audit source identity differs from GT admission: '+full_path)
        for scene in report.get('scenes', []):
            capture = scene.get('scene')
            if capture not in measured_frames:
                continue
            for sample in scene.get('samples', []):
                frame, cameras = sample.get('frame'), sample.get('camera_names')
                if (sample.get('status') == 'measured'
                        and type(frame) is int and frame >= 0
                        and isinstance(cameras, list) and len(cameras) in (2, 4)
                        and all(isinstance(name, str) and name for name in cameras)
                        and len(set(cameras)) == len(cameras)):
                    measured_frames[capture].add(frame)
    missing = sorted(name for name, frames in measured_frames.items() if len(frames) < 2)
    if missing:
        raise ValueError('dynamic admission requires measured body-region evidence at multiple times '
                         'with explicit C2/C4 cameras for every geometry capture: '+', '.join(missing))
    return dict(sources, **artifacts)


def same_identity(actual, expected):
    return bool(actual and all(actual.get(k) == expected.get(k) for k in ('pid', 'start', 'uid', 'cmd')))


def require_refinement_controller(identity, previous):
    args = shlex.split(identity['cmd'])
    def argument(flag):
        if args.count(flag) != 1:
            return None
        index = args.index(flag)
        return args[index+1] if index+1 < len(args) else None
    if (argument('-m') != 'scripts.continue_refinement_training'
            or argument('--controller') != str(Path(previous['control_dir'])/'job.json')
            or '-c' in args or ('python' in previous and args[0] != previous['python'])):
        raise ValueError('process is not the recorded refinement controller command')


def require_controller_signal_contract(identity, *, proc_root=Path('/proc')):
    """Establish default TERM and an already detached, orphaned session.

    The production controller starts its launcher with ordinary Popen, with no
    parent-death signal or forwarding handler. Requiring its observed daemon
    layout also avoids creating a newly orphaned stopped process group during
    the STOP/TERM/CONT handover. No process is signalled by this check.
    """
    directory = Path(proc_root)/str(identity['pid'])
    try:
        stat = (directory/'stat').read_text().rsplit(')', 1)[1].split()
        status = dict(line.split(':', 1) for line in (directory/'status').read_text().splitlines()
                      if ':' in line)
        ppid, pgrp, session, tty = map(int, stat[1:5])
        masks = {key: int(status[key].strip(), 16) for key in ('SigBlk','SigIgn','SigCgt')}
        start = stat[19]
    except (OSError, KeyError, ValueError, IndexError) as error:
        raise ValueError('cannot establish controller signal/session contract') from error
    if start != identity['start'] or directory.stat().st_uid != identity['uid']:
        raise ValueError('controller identity changed during signal/session inspection')
    if any(mask & (1 << (signal.SIGTERM-1)) for mask in masks.values()):
        raise ValueError('controller SIGTERM must use its default unblocked disposition')
    if ppid != 1 or pgrp != identity['pid'] or session != identity['pid'] or tty != 0:
        raise ValueError('controller must retain its detached session with PPID 1 and no controlling tty')
    return dict(pid=identity['pid'], start=start, ppid=ppid, pgrp=pgrp, session=session,
                controlling_tty=tty, signal_masks={key: f'{value:016x}' for key,value in masks.items()},
                term_default_unblocked=True, signal_scope='controller_pidfd_only')


def require_inherited_training_contract(job):
    """Controller metadata may move; all inherited training settings stay exact."""
    previous = job['previous']
    operational = {'pid','repo','commit','control_dir','environment','protected_files','stage',
                   'inherited_baseline','inherited_best'}
    changed = [key for key,value in previous.items()
               if key not in operational and job.get(key) != value]
    if changed:
        raise ValueError('inherited training/validation/LR settings changed: '+', '.join(sorted(changed)))
    old_environment, environment = (deepcopy(value.get('environment', {})) for value in (previous,job))
    old_environment.pop('PYTHONPATH', None)
    if environment.pop('PYTHONPATH', None) != job['repo'] or environment != old_environment:
        raise ValueError('only PYTHONPATH may change in the inherited training environment')
    if any(job.get('protected_files', {}).get(path) != expected
           for path,expected in previous.get('protected_files', {}).items()):
        raise ValueError('original protected training/validation artifacts must remain frozen')


def retire_controller_only(identity, verify_frozen, *, timeout=10.):
    """The only signal target is one identity-checked controller pidfd."""
    if (identity['uid'] != os.getuid() or not alive(identity)
            or not same_identity(proc(identity['pid']), identity)):
        raise ValueError('controller identity changed')
    require_controller_signal_contract(identity)
    fd = pidfd_open(identity['pid'])
    frozen = retired = False
    try:
        if not same_identity(proc(identity['pid']), identity):
            raise ValueError('controller identity changed while acquiring its pidfd')
        pidfd_send_signal(fd, signal.SIGSTOP)
        frozen = True
        deadline = time.monotonic()+timeout
        while True:
            current = proc(identity['pid'])
            if not same_identity(current, identity):
                raise ValueError('controller exited before frozen verification')
            if current['state'] in ('T', 't'):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('controller did not stop for identity verification')
            time.sleep(.01)
        signal_contract = require_controller_signal_contract(identity)
        verify_frozen()
        pidfd_send_signal(fd, signal.SIGTERM)
        pidfd_send_signal(fd, signal.SIGCONT)
        frozen = False
        deadline = time.monotonic()+timeout
        while alive(identity):
            if time.monotonic() > deadline:
                raise TimeoutError('controller did not exit; no child was signalled')
            time.sleep(.05)
        retired = True
        return dict(signalled_pid=identity['pid'], signal_scope='controller_pidfd_only',
                    child_processes_signalled=False, training_restarted=False,
                    signal_contract=signal_contract)
    finally:
        if frozen and not retired:
            pidfd_send_signal(fd, signal.SIGCONT)
        os.close(fd)


def verify_ready(job):
    if job.get('preparation', {}).get('blocked_reasons'):
        raise ValueError('this pending draft is not executable; prepare again with completed proofs')
    if job.get('output_root') != job['previous'].get('output_root'):
        raise ValueError('original training output_root and checkpoint retention must be preserved')
    require_inherited_training_contract(job)
    clean_runtime(job['repo'], job['commit'])
    clean_runtime(job['previous']['repo'], job['previous']['commit'])
    verify_files(job['protected_files'])
    test = read(job['tests_proof'])
    if test.get('passed') is not True or test.get('commit') != job['commit']:
        raise ValueError('the exact clean admission commit must pass its tests')
    gt_gate(job['gt_proof'], job['previous']['data_config'], job['candidate_data_config'], job['repo'])
    contract = transition_contract(read_data_config(job['previous']['data_config']),
                                   read_data_config(job['candidate_data_config']))
    if contract != job['data_transition']:
        raise ValueError('the prepared data transition changed')
    for name, expected in job['critical_runtime_sha256'].items():
        if sha(Path(job['repo'])/name) != expected or sha(Path(job['previous']['repo'])/name) != expected:
            raise ValueError('training, model, loss, scheduler or controller code changed: '+name)


class KinopticAdmissionController(RefinementController):
    """Reuse the production loop and change data only after adopted validation."""
    def __init__(self, path):
        super().__init__(path)
        self._adoption_pending = True
        self._handover_handles = []

    def _cached_objective_record(self, checkpoint, label):
        """Verify the inherited objective boundary without launching any GPU job."""
        admission = self.job['objective_admission']
        if (label != admission['validation_label']
                or label not in self.job.get('reuse_validation_labels', [])
                or Path(checkpoint).resolve() != Path(admission['validation_checkpoint']).resolve()):
            raise ValueError('objective prologue requires its original explicitly reusable cached validation')
        contract = self.job['validation_contract']
        if build_validation_contract(self.job) != contract:
            raise ValueError('fixed validation contract changed')
        record = dict(checkpoint=str(checkpoint), label=label, step=admission['step'],
                      validation_contract=deepcopy(contract), reports={})
        entries, frozen = {}, {}
        plans = self._plans()
        for suite in SUITES:
            path = self.output/'validation'/label/suite/'evaluation.json'
            if not path.is_file():
                raise ValueError('objective prologue needs all five completed cached panels: '+suite)
            frozen[str(path)] = sha(path)
            report = read(path)
            if report.get('stage') != 'passed' or label not in report.get('checkpoints', {}):
                raise ValueError('objective prologue cannot reuse an incomplete or differently labelled panel: '+suite)
            entries[suite] = verify_suite_report(suite, report, record, plans[suite], contract)
            record['reports'][suite] = str(path)
        record['errors'] = error_panel(entries)
        if ExpandedDynamicController._verify_record(self, record) != admission['step']:
            raise ValueError('objective cached panels disagree with the original checkpoint step')
        verify_files(frozen)
        return record

    def evaluate(self, checkpoint, label):
        admission = self.job.get('objective_admission')
        if self._adoption_pending and admission and label == admission['validation_label']:
            record = self._cached_objective_record(checkpoint, label)
            self.progress['reused_validation_suites'] = list(SUITES)
            write_json(self.control/'objective_validation_cache_reuse.json', record)
            self.publish('reused_original_objective_validation')
            return record
        return super().evaluate(checkpoint, label)

    def verify_handover(self):
        pidfd_capability = probe_pidfd_support()  # Own pidfd + signal 0 only, including --verify-only.
        verify_ready(self.job)
        previous = self.job['previous']
        active = read(self.active)
        keys = ('pid','repo','commit','control_dir','run','phase_target_step','data_config',
                'physical_gpus','selected_checkpoint','model_config','loss_config','lr_plan')
        if any(active.get(k) != previous.get(k) for k in keys):
            raise ValueError('main advanced or changed; prepare a fresh handover')
        identity = self.job['previous_identity']
        if not same_identity(proc(identity['pid']), identity):
            raise ValueError('previous controller identity changed')
        require_refinement_controller(identity, previous)
        signal_contract = require_controller_signal_contract(identity)
        objective = self.job.get('objective_admission')
        if (objective and self.job.get('continuation_step', self.job.get('initial_step')) == objective['step']):
            self._cached_objective_record(objective['validation_checkpoint'], objective['validation_label'])
        tree = owned_tree(identity)
        adopted = self.job['adopted_training']
        require_training_child(tree.get(adopted['pid']), adopted, previous['pid'])
        ranks = training_ranks(tree)
        if (len(ranks) != len(previous['physical_gpus']) or not all(alive(rank) for rank in ranks)
                or any(record['uid'] != os.getuid() for record in tree.values())):
            raise ValueError('recorded main ranks or process ownership changed')
        status = read(Path(previous['control_dir'])/'progress.json')
        if status.get('stage') != f"train_{adopted['end']}" or status.get('child_pid') != adopted['pid']:
            raise ValueError('handover needs the recorded ordinary refinement segment')
        return dict(tree=tree, ranks=ranks, progress=status, signal_contract=signal_contract,
                    pidfd_capability=pidfd_capability)

    def run_explicit_handover(self):
        self.verify_handover()  # All gates precede dispatch and every signal.
        with ExitStack() as stack:
            for root in sorted({Path(self.job['previous']['repo']), self.root}):
                directory = root/'outputs/training_preflight'
                directory.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context((directory/'controller_handover.lock').open('a'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._handover_handles.append(handle)
            verified = self.verify_handover()
            self._preserved_ranks = deepcopy(verified['ranks'])
            receipt = retire_controller_only(self.job['previous_identity'], self.verify_handover)
            receipt.update(original_launcher=self.job['adopted_training'],
                           preserved_rank_identities=self._preserved_ranks,
                           rank_identity_after_retirement=[dict(pid=rank['pid'],
                               same_process=same_identity(proc(rank['pid']), rank))
                               for rank in self._preserved_ranks])
            write_json(self.control/'controller_handover_receipt.json', receipt)
            # The unchanged parent acquires canonical locks and runs its loop.
            # Hooks below adopt existing children rather than launch a duplicate.
            super().run()

    def wait_for_initial_training_gpus(self, stack):
        from multicamera4rc.launch import gpu_snapshot
        devices, _ = gpu_snapshot()
        for index, uuid in zip(self.job['physical_gpus'], self.job['environment']['CUDA_VISIBLE_DEVICES'].split(',')):
            if devices[index].uuid != uuid:
                raise ValueError('GPU identity changed during controller adoption')
            lease = self.device_lease(index, uuid)
            if lease is None:
                raise RuntimeError('original GPU lease was acquired by another scheduler')
            stack.enter_context(lease)
        self.owns_training = True
        self.publish('adopted_train_'+str(self.job['adopted_training']['end']))
        for handle in self._handover_handles:
            fcntl.flock(handle, fcntl.LOCK_UN)
        self._handover_handles.clear()
        return devices

    def wait_for_selected_training_gpus(self):
        if not self._adoption_pending:
            return super().wait_for_selected_training_gpus()

    def maybe_expand(self, stack):
        if not self._adoption_pending:
            return super().maybe_expand(stack)

    def check(self, checkpoint, step):
        state = super().check(checkpoint, step)
        activation = self.job.get('data_activation')
        if activation and (str(checkpoint), step) == (activation['checkpoint'], activation['step']):
            # No checkpoint bytes are edited. This is the explicitly reset
            # traversal origin used only by the first segment's cursor proof.
            state = deepcopy(state)
            state.update(data_epoch=0, batches_consumed=0)
        return state

    def maybe_admit_performance(self, current, step):
        if not self._adoption_pending:
            if self.job.get('data_activation') and step > self.job['data_activation']['step']:
                self.job['performance_args'] = [x for x in self.job.get('performance_args', [])
                                                if x != '--reset-data-cursor']
            return super().maybe_admit_performance(current, step)
        adopted = self.job['adopted_training']
        deadline = time.monotonic()+48*3600
        while alive(adopted):
            if time.monotonic() > deadline:
                raise TimeoutError('the unchanged original training segment did not finish')
            self.publish('adopted_train_'+str(adopted['end']))
            time.sleep(10)
        checkpoint = str(Path(adopted['run'])/'latest.pt')
        before = super().check(adopted['checkpoint'], adopted['begin'])
        after = super().check(checkpoint, adopted['end'])
        require_cursor_progress(before, after, adopted['end']-adopted['begin'])
        health = continuation_health(adopted['log'], adopted['begin'], adopted['end'],
                                     len(self.job['physical_gpus']))
        admission_output = Path(self.job['admission_output_root'])
        write_json(admission_output/'adopted_training_health.json', health)
        self.progress.update(step=adopted['end'], latest_checkpoint=checkpoint)
        record = self.evaluate(checkpoint, f"refinement_{adopted['end']}_before_kinoptic")
        comparison = compare_errors(self.progress['baseline']['errors'], record['errors'])
        if comparison['score_ratio'] > 1.5:
            raise RuntimeError('original fixed validation failed; Kinoptic remains inactive')
        verify_ready(self.job)
        self.remember_best(record)
        activation = dict(stage='post_validation', step=adopted['end'], checkpoint=checkpoint,
            old_data_config=self.job['data_config'], new_data_config=self.job['candidate_data_config'],
            old_data_epoch=after['data_epoch'], old_batches_consumed=after['batches_consumed'],
            new_data_epoch=0, new_batches_consumed=0, reset_data_cursor=True,
            global_step_preserved=True, optimizer_scaler_rank_rng_restored_from_same_checkpoint=True,
            lr_plan_unchanged=True, loss_config_unchanged=True,
            validation_record=record, comparison=comparison, data_transition=self.job['data_transition'])
        write_json(admission_output/'data_activation.json', activation)
        self.job.update(data_activation=activation, data_config=self.job['candidate_data_config'])
        self.job['performance_args'] = list(self.job.get('performance_args', []))+['--reset-data-cursor']
        self._adoption_pending = False
        self.publish('kinoptic_ready_after_original_validation')
        return checkpoint, adopted['end']


def prepare(active_path, runtime, commit, candidate_config, gt_proof, tests_proof,
            output, *, allow_pending=False):
    """Create an inert draft/job. No controller is dispatched by this function."""
    active_path, runtime = Path(active_path).resolve(), Path(runtime).resolve()
    previous = read(active_path)
    verify_files(previous['protected_files'])
    candidate_config, gt_proof, tests_proof = map(lambda p: Path(p).resolve(),
                                                 (candidate_config, gt_proof, tests_proof))
    contract = transition_contract(read_data_config(previous['data_config']), read_data_config(candidate_config))
    output = Path(output).resolve()
    forbidden = [Path(previous[k]).resolve() for k in ('repo','run','control_dir','output_root')]+[runtime]
    sources = (read_data_config(previous['data_config'])['sources']
               + read_data_config(candidate_config)['sources'])
    forbidden += [Path(s['root']).resolve() for s in sources]
    for source in sources:
        if source['adapter'] in ('panoptic','panoptic_kinoptic'):
            root = Path(source['root'])
            if root.is_dir():
                forbidden += [path.resolve() for path in root.iterdir() if path.is_dir()]
    if any(output == path or path in output.parents for path in forbidden):
        raise ValueError('preparation output must be separate from active runtime, runs and data')
    identity = proc(previous['pid'])
    if not identity or identity['uid'] != os.getuid():
        raise ValueError('main controller is not an owned live process')
    require_refinement_controller(identity, previous)
    status = read(Path(previous['control_dir'])/'progress.json')
    child = proc(status.get('child_pid', -1))
    if not child:
        raise ValueError('prepare during an ordinary running refinement training segment')
    if status.get('stage') != f"train_{previous['phase_target_step']}":
        raise ValueError('preparation requires an ordinary running refinement phase')
    if not status.get('baseline'):
        raise ValueError('original fixed-panel validation baseline is required')
    adopted = dict(child, run=previous['run'], checkpoint=previous['selected_checkpoint'],
        begin=previous['resume_step'], end=previous['phase_target_step'],
        data_config=previous['data_config'], log=status['child_log'])
    require_training_child(child, adopted, previous['pid'])
    critical = {}
    for name in CRITICAL_RUNTIME_FILES:
        old_sha, new_sha = sha(Path(previous['repo'])/name), sha(runtime/name)
        if old_sha != new_sha:
            raise ValueError('training/model/loss/controller code differs: '+name)
        critical[name] = old_sha
    protected = deepcopy(previous['protected_files'])
    protected[str(candidate_config)] = sha(candidate_config)
    blocked = []
    for label, check in (
        ('clean_runtime', lambda: clean_runtime(runtime, commit)),
        ('current_dynamic_gt', lambda: gt_gate(gt_proof, previous['data_config'], candidate_config, runtime)),
        ('exact_commit_tests', lambda: read(tests_proof))):
        try:
            value = check()
            if label == 'current_dynamic_gt':
                protected.update(value)
                protected[str(gt_proof)] = sha(gt_proof)
            if label == 'exact_commit_tests':
                if value.get('passed') is not True or value.get('commit') != commit:
                    raise ValueError('exact commit test receipt did not pass')
                protected[str(tests_proof)] = sha(tests_proof)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            if not allow_pending:
                raise
            blocked.append(dict(gate=label, error=str(error)))
    job = deepcopy(previous)
    job.update(repo=str(runtime), commit=commit,
        environment=dict(previous['environment'], PYTHONPATH=str(runtime)),
        control_dir=str(output/'control'), admission_output_root=str(output),
        previous=deepcopy(previous), previous_identity=identity, adopted_training=adopted,
        candidate_data_config=str(candidate_config), gt_proof=str(gt_proof), tests_proof=str(tests_proof),
        critical_runtime_sha256=critical, protected_files=protected, data_transition=contract,
        inherited_baseline=deepcopy(status.get('baseline', previous.get('inherited_baseline'))),
        inherited_best=deepcopy(status.get('best', previous.get('inherited_best'))),
        preparation=dict(execution_scheduled=False, signals_sent=False,
                         blocked_reasons=blocked, prepared_unix_seconds=time.time()))
    output.mkdir(parents=True, exist_ok=False)
    (output/'control').mkdir()
    write_json(output/'control/job.json', job)
    write_json(output/'data_transition.json', contract)
    return output/'control/job.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--controller', type=Path)
    parser.add_argument('--active', type=Path)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--commit')
    parser.add_argument('--candidate-config', type=Path)
    parser.add_argument('--gt-proof', type=Path)
    parser.add_argument('--tests-proof', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-pending', action='store_true')
    dispatch = parser.add_mutually_exclusive_group()
    dispatch.add_argument('--verify-only', action='store_true')
    dispatch.add_argument('--execute-controller-handoff', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare:
        required = ('active','runtime','commit','candidate_config','gt_proof','tests_proof','output')
        if args.execute_controller_handoff or any(getattr(args, name) is None for name in required):
            parser.error('preparation requires all input paths and cannot dispatch execution')
        path = prepare(args.active, args.runtime, args.commit, args.candidate_config,
                       args.gt_proof, args.tests_proof, args.output, allow_pending=args.allow_pending)
        print(json.dumps(dict(job=str(path), execution_scheduled=False,
                              preparation=read(path)['preparation'])))
    else:
        controller = KinopticAdmissionController(args.controller)
        if args.execute_controller_handoff:
            controller.run_explicit_handover()
        else:
            result = controller.verify_handover()
            print(json.dumps(dict(passed=True, signals_sent=False,
                                  preserved_rank_pids=[r['pid'] for r in result['ranks']])))


if __name__ == '__main__':
    main()
