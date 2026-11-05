"""Replace fixed validation panels without restarting an active training segment.

Every reference is measured on the same admitted plans. This controller keeps
the current cosine LR plan, loss, mixture, optimizer and sampler; a failed hard
quality gate is retained for review, never converted into an automatic LR retry.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shlex
import time

from multicamera4rc.metrics import METRICS_VERSION
from multicamera4rc.quality import compare_errors, error_panel
from scripts.continue_extended_training import continuation_argv, quality_decision
from scripts.continue_quality_recovery import digest, training_argv, validate_checkpoint
from scripts.continue_syn4d_training import verify_state
from scripts.cutover_native_motion_stereo import alive, write_json
from scripts.migrate_static_validation import (
    ExpandedStaticController, checkpoint_entry, plan_digest, verify_static_cases,
)
from scripts.resize_training import distributed_health


SUITES = ('kubric', 'native_dynamic', 'static', 'variable', 'syn4d')
PANEL_SUITES = SUITES[:-1]


def suite_paths(job, suite):
    if suite == 'syn4d':
        return job['syn4d_eval_plan'], job['syn4d_eval_config']
    prefix = 'native' if suite == 'native_dynamic' else suite
    return job['evaluation'][prefix+'_plan'], job['evaluation'][prefix+'_config']


def build_validation_contract(job):
    suites = {}
    for suite in SUITES:
        plan_path, config_path = suite_paths(job, suite)
        plan = json.loads(Path(plan_path).read_text())
        if not plan.get('cases'):
            raise ValueError('empty admitted validation suite: '+suite)
        suites[suite] = dict(plan_sha256=plan_digest(plan), data_config_sha256=digest(config_path))
    return dict(metrics_version=METRICS_VERSION, suites=suites)


def require_validation_contract(contract, *records):
    if any(record.get('validation_contract') != contract for record in records):
        raise ValueError('refusing comparison across validation contracts')


def _flat(values):
    if len(values) == 1 and isinstance(values[0], list):
        return values[0]
    return values


def verify_suite_report(suite, report, record, plan, contract):
    """Verify checkpoint identity, hashes and every planned observation case."""
    expected = contract['suites'][suite]
    if contract['metrics_version'] != METRICS_VERSION or plan_digest(plan) != expected['plan_sha256']:
        raise ValueError('validation contract does not identify the plan: '+suite)
    if report.get('plan_sha256') != expected['plan_sha256']:
        raise ValueError('validation report plan changed: '+suite)
    # Static is unchanged and its historical reports predate config hashes;
    # its train config and generated manifests remain locked by admission proof.
    config_hash = report.get('data_config_sha256')
    if (config_hash is None and suite != 'static') or (
            config_hash is not None and config_hash != expected['data_config_sha256']):
        raise ValueError('validation report config changed: '+suite)
    entry = checkpoint_entry(report, record)
    if 'step' in record and entry.get('step') != record['step']:
        raise ValueError('validation report checkpoint step changed: '+suite)
    rows = entry.get('records', [])
    if entry.get('clips') != len(plan['cases']) or len(rows) != len(plan['cases']):
        raise ValueError('validation report has missing or extra cases: '+suite)
    if suite == 'static':
        verify_static_cases(report, record, plan)
        return entry
    if suite == 'native_dynamic':
        expected_ids, actual_ids = [], []
        for case in plan['cases']:
            expected_ids.append((case['source'], case['source_name'], case['sequence'], case['index']))
        for row in rows:
            actual_ids.append((row['source'], row['source_name'], row['sequence'], row['index']))
        if len(set(expected_ids)) != len(expected_ids) or sorted(expected_ids) != sorted(actual_ids):
            raise ValueError('native validation cases differ from the fixed plan')
        by_id = dict(zip(actual_ids, rows))
        for identity, case in zip(expected_ids, plan['cases']):
            row = by_id[identity]
            times = _flat(row['time_indices'])
            cameras = _flat(row['camera_ids'])
            planned_times = case.get('expected_time_indices', case.get('time_indices'))
            planned_cameras = case.get('expected_camera_ids', case.get('camera_ids'))
            if planned_times is None or planned_cameras is None:
                raise ValueError('native plan must freeze actual times and cameras')
            if times != _flat(planned_times) or cameras != _flat(planned_cameras):
                raise ValueError('native validation window or camera changed')
            if len(times) != len(cameras) or row['input_shape'][1] != len(times):
                raise ValueError('native validation observation count changed')
        return entry
    if suite not in ('kubric', 'variable', 'syn4d'):
        raise ValueError('unsupported validation suite: '+suite)
    indexed = {row['case']: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(range(len(plan['cases']))):
        raise ValueError('renderer report has missing or duplicate case indices')
    for index, case in enumerate(plan['cases']):
        row = indexed[index]
        layout = plan['layouts'][case['layout']]
        cameras, times = _flat(row['camera_ids']), _flat(row['time_indices'])
        if row['sequence'] != case['sequence'] or row['layout'] != layout:
            raise ValueError('renderer validation scene or layout changed')
        if (row.get('sampling_seed') != case['sampling_seed']
                or row.get('planned_times', row.get('times')) != case['times']):
            raise ValueError('renderer validation sampling identity changed')
        C, T = layout
        if suite == 'syn4d' and plan.get('format_version', 1) >= 3 and any(
                key not in case for key in ('expected_time_indices', 'expected_camera_ids')):
            raise ValueError('Syn4D plan must distinguish positions from physical observations')
        expected_times = case.get('expected_time_indices', case['times']*C)
        if ('expected_camera_ids' in case and cameras != case['expected_camera_ids']):
            raise ValueError('renderer validation physical cameras changed')
        if (len(case['times']) != T or times != expected_times or len(times) != C*T or len(cameras) != C*T
                or len(set(cameras)) != C or any(cameras.count(c) != T for c in set(cameras))):
            raise ValueError('renderer validation actual observations changed')
    return entry


def rebase_validation_record(record, replacement_reports, plans, contract):
    """Replace measurements, never compare a new panel with old-panel errors."""
    original = {}
    for suite in PANEL_SUITES:
        report = json.loads(Path(record['reports'][suite]).read_text())
        original[suite] = checkpoint_entry(report, record)
    if error_panel(original) != record['errors']:
        raise ValueError('cached quality panel differs from its source reports')
    entries, paths = {}, dict(record['reports'])
    for suite in SUITES:
        path = replacement_reports.get(suite, paths.get(suite))
        if path is None:
            raise ValueError('missing measured validation suite: '+suite)
        report = json.loads(Path(path).read_text())
        entries[suite] = verify_suite_report(suite, report, record, plans[suite], contract)
        paths[suite] = str(path)
    if len({entry['step'] for entry in entries.values()}) != 1:
        raise ValueError('validation suites disagree about the checkpoint step')
    panel = error_panel(entries)
    if set(panel) != set(record['errors']):
        raise ValueError('validation migration changed the quality metric set')
    for suite in PANEL_SUITES:
        if suite not in replacement_reports and entries[suite] != original[suite]:
            raise ValueError('unchanged validation suite changed measurements: '+suite)
    result = deepcopy(record)
    result.update(errors=panel, reports=paths, validation_contract=deepcopy(contract))
    result.pop('learning_rate_trial', None)
    result.pop('restart_lr', None)
    return result


def require_fixed_training_state(expected, actual):
    if any(actual.get(key) != expected.get(key) for key in (
            'loss_config', 'lr_plan', 'dense_target_version', 'kubric_background_policy', 'world_size', 'batch_size',
            'dataset_size', 'seed', 'source_count', 'max_source_time_pairs', 'prefer_motion_sources', 'supervision')):
        raise ValueError('continuation changed the admitted LR, loss or training contract')


def verify_data_config_replacement(job):
    """Permit only audited per-pixel Replica GT masks, preserving index order."""
    replacement = job.get('audited_data_config_replacement')
    if not replacement:
        return None
    from multicamera4rc.datasets.mixture import read_data_config
    for name in ('old', 'new'):
        if digest(replacement[name+'_config']) != replacement[name+'_config_sha256']:
            raise ValueError('audited training config changed: '+name)
    proof_path = replacement['sampling_equivalence_proof']
    proof = json.loads(Path(proof_path).read_text())
    if (not proof.get('passed') or proof.get('sampling_equivalent') is not True
            or any(proof.get(name+'_config_sha256') != replacement[name+'_config_sha256'] for name in ('old', 'new'))):
        raise ValueError('training data replacement lacks sampling equivalence proof')
    if job.get('evidence_sha256', {}).get(str(proof_path)) != digest(proof_path):
        raise ValueError('sampling equivalence proof is not immutable admission evidence')
    old, new = (deepcopy(read_data_config(replacement[name+'_config'])) for name in ('old', 'new'))
    if len(old['sources']) != len(new['sources']):
        raise ValueError('GT mask migration changed the source count')
    changed = False
    for before, after in zip(old['sources'], new['sources']):
        if before.get('adapter') == after.get('adapter') == 'dynamic_replica':
            keys = ('depth_validity_mask_root', 'depth_validity_manifest_sha256')
            previous = {key: before.setdefault('options', {}).pop(key, None) for key in keys}
            current = {key: after.setdefault('options', {}).pop(key, None) for key in keys}
            if previous != current:
                fingerprint = current['depth_validity_manifest_sha256']
                if (not current['depth_validity_mask_root'] or not isinstance(fingerprint, str)
                        or len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint)):
                    raise ValueError('GT mask migration requires the validity mask and pinned manifest SHA256')
                changed = True
    if old != new or not changed:
        raise ValueError('GT mask migration changed sampling or other training settings')
    return replacement


class ExpandedDynamicController(ExpandedStaticController):
    def _evaluation_memory_fraction(self):
        value = self.job.get('evaluation_memory_fraction', .20)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError('evaluation_memory_fraction must be finite and lie in (0, 1]')
        return float(value)

    def run_process(self, name, command, *, evaluation=False):
        if not evaluation:
            return super().run_process(name, command, evaluation=False)
        fraction = self._evaluation_memory_fraction()
        command = list(command)
        wrapper = ('import runpy,sys,torch; torch.cuda.set_per_process_memory_fraction('
                   +repr(fraction)+',0); runpy.run_module(sys.argv.pop(1),run_name="__main__")')
        if '-c' in command:
            index = command.index('-c')
            old_wrapper = command[index+1]
            if ('torch.cuda.set_per_process_memory_fraction(' not in old_wrapper
                    or 'runpy.run_module(sys.argv.pop(1),run_name="__main__")' not in old_wrapper):
                raise ValueError('unrecognized validation wrapper; refusing to replace its program')
            command[index+1] = wrapper
        elif '-m' in command:
            index = command.index('-m')
            command[index:index+1] = ['-c', wrapper]
        else:
            raise ValueError('validation memory cap requires a known Python module command')
        if '--memory-fraction' in command:
            index = command.index('--memory-fraction')
            if index+1 >= len(command):
                raise ValueError('validation memory-fraction option has no value')
            command[index+1] = str(fraction)
        return super().run_process(name, command, evaluation=True)

    def publish(self, stage):
        if stage == 'failed' and self.progress.get('failed_quality_needs_review'):
            stage = 'failed_quality_needs_review'
        return super().publish(stage)

    def _plans(self):
        return {suite: json.loads(Path(suite_paths(self.job, suite)[0]).read_text()) for suite in SUITES}

    def _verify_record(self, record):
        contract = self.job['validation_contract']
        require_validation_contract(contract, record)
        steps, entries = set(), {}
        for suite, plan in self._plans().items():
            report = json.loads(Path(record['reports'][suite]).read_text())
            entry = verify_suite_report(suite, report, record, plan, contract)
            entries[suite] = entry
            steps.add(entry['step'])
        if len(steps) != 1:
            raise ValueError('validation suites disagree about the checkpoint step')
        if error_panel(entries) != record['errors']:
            raise ValueError('cached quality panel differs from admitted source reports')
        return steps.pop()

    def _fixed_state(self):
        return self.progress['adopted_expected_state']

    def _check_candidate_state(self, checkpoint, step):
        state = validate_checkpoint(checkpoint, step, world_size=len(self.job['physical_gpus']),
                                    batch_size=self.job.get('batch_size_per_gpu', 1))
        verify_state(state, self.job['target_step'])
        require_fixed_training_state(self._fixed_state(), state)
        return state

    def _cumulative_diagnostic(self, record):
        path = self.job.get('motion_anchor_record')
        if not path:
            return None
        anchor = json.loads(Path(path).read_text())
        require_validation_contract(self.job['validation_contract'], anchor, record)
        diagnostic = dict(anchor_checkpoint=anchor['checkpoint'], anchor_step=anchor['step'],
                          candidate_checkpoint=record['checkpoint'], candidate_step=record['step'],
                          comparison=compare_errors(anchor['errors'], record['errors']), diagnostic_only=True)
        self.progress['motion_anchor_diagnostic'] = diagnostic
        write_json(self.output/'motion_anchor_diagnostic.json', diagnostic)
        return diagnostic

    def evaluate(self, checkpoint, label):
        if build_validation_contract(self.job) != self.job['validation_contract']:
            raise ValueError('admitted validation plan or config changed on disk')
        record = super().evaluate(checkpoint, label)
        record['validation_contract'] = deepcopy(self.job['validation_contract'])
        record['step'] = self._verify_record(record)
        write_json(self.output/'validation'/label/'quality.json', record)
        return record

    def remember_best(self, record):
        require_validation_contract(self.job['validation_contract'], record, self.progress['best'])
        return super().remember_best(record)

    def verify_handover(self):
        self._evaluation_memory_fraction()
        if self.job.get('learning_rate_trial') or self.job.get('max_quality_retries', 0) != 0:
            raise ValueError('validation migration must not schedule an LR trial or reduction retry')
        if build_validation_contract(self.job) != self.job['validation_contract']:
            raise ValueError('admitted validation plan or config changed on disk')
        refs, status, tree, decision = super().verify_handover()
        adopted, previous = self.job['adopted_training'], self.job['previous']
        if (status['stage'] != f"train_{adopted['end']}_try0"
                or previous['resume_step'] != adopted['begin']
                or previous['phase_target_step'] != adopted['end']
                or refs['initial']['step'] != adopted['begin']):
            raise RuntimeError('handover requires the recorded ordinary try0 training interval')
        if any(record.get('learning_rate_trial') or record.get('restart_lr') for record in refs.values()):
            raise ValueError('reference retains a one-time LR override')
        if not (self.job['data_config'] == previous['data_config'] == adopted['data_config']
                == refs['initial']['data_config']):
            raise ValueError('validation-only migration cannot change training data')
        replacement = verify_data_config_replacement(self.job)
        if replacement and replacement['old_config'] != adopted['data_config']:
            raise ValueError('GT mask replacement does not identify the active training mixture')
        if digest(self.job['evaluation']['static_config']) != digest(previous['evaluation']['static_config']):
            raise ValueError('dynamic validation migration cannot change the retained static config')
        args = shlex.split(tree[adopted['pid']]['cmd'])
        if ('--override-lr' in args or '--reset-data-cursor' in args or '--resume' not in args
                or args[args.index('--resume')+1] != refs['initial']['checkpoint']):
            raise ValueError('active segment does not preserve the accepted checkpoint state')
        for record in refs.values():
            self._verify_record(record)
            if len(record['errors']) != self.job.get('expected_metric_count', 24):
                raise ValueError('admitted quality metric count changed')
        proof_path = self.job['migration_proof']
        proof = json.loads(Path(proof_path).read_text())
        if not proof.get('passed') or proof.get('validation_contract') != self.job['validation_contract']:
            raise ValueError('validation split and leakage migration proof failed')
        if str(proof_path) not in self.job['evidence_sha256']:
            raise ValueError('validation migration proof is not immutable admission evidence')
        if self.job.get('motion_anchor_record'):
            anchor = json.loads(Path(self.job['motion_anchor_record']).read_text())
            self._verify_record(anchor)
        state = validate_checkpoint(refs['initial']['checkpoint'], adopted['begin'],
                                    world_size=len(self.job['physical_gpus']),
                                    batch_size=self.job.get('batch_size_per_gpu', 1))
        verify_state(state, self.job['target_step'])
        if 'adopted_expected_state' in self.job:
            require_fixed_training_state(self.job['adopted_expected_state'], state)
        self._verified_adopted_state = state
        return refs, status, tree, decision

    def adopt(self, lock):
        refs = super().adopt(lock)
        self.progress['adopted_expected_state'] = deepcopy(self._verified_adopted_state)
        self.progress['validation_migration'].update(validation_contract=self.job['validation_contract'],
            kind='dynamic_fixed_panels', learning_rate_changed=False, automatic_lr_retries=False)
        if self.job.get('audited_data_config_replacement'):
            self.progress['audited_data_config_replacement'] = dict(
                self.job['audited_data_config_replacement'], stage='pending_original_segment_completion',
                activation_step=self.job['adopted_training']['end'], reset_data_cursor=False)
        proof_path = self.output/'handover_proof.json'
        proof = json.loads(proof_path.read_text())
        proof.update(validation_contract=self.job['validation_contract'], learning_rate_changed=False,
                     admitted_lr_plan=self._fixed_state()['lr_plan'])
        write_json(proof_path, proof)
        self.publish('train_'+str(self.job['adopted_training']['end'])+'_adopted')
        return refs

    def _accept_or_review(self, reference, best, historical, record, state, *, adopted=False):
        require_validation_contract(self.job['validation_contract'], reference, best, historical, record)
        decision = quality_decision(reference['errors'], best['errors'], historical['errors'], record['errors'])
        decision.update(step=record['step'], retry=0, checkpoint=record['checkpoint'],
                        rollback_checkpoint=reference['checkpoint'], adopted=adopted,
                        automatic_lr_retries=False, motion_anchor_diagnostic=self._cumulative_diagnostic(record))
        if decision['action'] != 'continue':
            decision['original_action'] = decision['action']
            decision['action'] = 'failed_quality_needs_review'
            self.progress['failed_quality_needs_review'] = True
        self.progress['decisions'].append(decision)
        write_json(self.output/'latest_evaluated_checkpoint.json', record)
        self.publish('quality_checked')
        if decision['action'] != 'continue':
            self.publish('failed_quality_needs_review')
            raise RuntimeError('quality gate requires review; checkpoints retained and LR plan unchanged')
        self.progress['resumed_training_state'] = state
        return record

    def finish_adopted(self, refs):
        adopted = self.job['adopted_training']
        deadline = time.monotonic()+self.job.get('adopted_wait_hours', 24)*3600
        while alive(adopted):
            if time.monotonic() > deadline:
                raise TimeoutError('adopted segment exceeded its original completion window')
            time.sleep(5)
        checkpoint = Path(adopted['run'])/'latest.pt'
        state = self._check_candidate_state(checkpoint, adopted['end'])
        proof = distributed_health(adopted['log'], adopted['begin'], adopted['end'], len(self.job['physical_gpus']))
        write_json(self.output/'adopted_training_proof.json', proof)
        self.progress.pop('child_pid', None)
        record = self.evaluate(checkpoint, 'extended_'+str(adopted['end'])+'_adopted')
        record.update(step=adopted['end'], data_config=adopted['data_config'])
        return self._accept_or_review(refs['initial'], self.progress['best'], refs['historical'],
                                      record, state, adopted=True)

    def segment(self, reference, best, historical, config, end, *, start_retry=0):
        if start_retry or reference.get('learning_rate_trial') or reference.get('restart_lr'):
            raise ValueError('fixed-LR continuation does not permit retries or one-time LR overrides')
        require_validation_contract(self.job['validation_contract'], reference, best, historical)
        if config != self.job['data_config'] or config != reference['data_config']:
            raise ValueError('fixed-LR continuation cannot reset the data cursor')
        replacement = verify_data_config_replacement(self.job)
        replace_data = bool(replacement and config == replacement['old_config'])
        if replace_data:
            if reference['step'] < self.job['adopted_training']['end']:
                raise ValueError('GT mask replacement cannot interrupt the adopted segment')
            config = replacement['new_config']
        self._check_candidate_state(reference['checkpoint'], reference['step'])
        run = self.output/f"continuation_{reference['step']}_{end}_try0"
        if run.exists():
            raise RuntimeError('refusing to overwrite an existing training segment: '+str(run))
        self.job.update(run=str(run), selected_checkpoint=reference['checkpoint'],
                        resume_step=reference['step'], phase_target_step=end, data_config=config)
        # train._resume_data_cursor accepts an equal-sized sampling pool. The
        # generic continuation helper resets on any path change, so bypass that
        # path heuristic only for this proved label-only change.
        argv = (training_argv(self.job, reference['checkpoint'], config, run, end, reset_cursor=False)
                if replace_data else continuation_argv(self.job, reference, config, run, end))
        if '--override-lr' in argv or '--reset-data-cursor' in argv:
            raise ValueError('continuation unexpectedly overrides LR or data cursor')
        if replace_data:
            self.progress['audited_data_config_replacement'] = dict(replacement,
                stage='active', activation_step=reference['step'], reset_data_cursor=False)
            write_json(self.output/'audited_data_config_replacement.json',
                       self.progress['audited_data_config_replacement'])
        self.run_process(f'train_{end}_try0', argv)
        checkpoint = run/'latest.pt'
        state = self._check_candidate_state(checkpoint, end)
        record = self.evaluate(checkpoint, f'extended_{end}_try0')
        record.update(step=end, data_config=config)
        return self._accept_or_review(reference, best, historical, record, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    controller = ExpandedDynamicController(args.controller)
    if args.verify_only:
        refs, status, tree, decision = controller.verify_handover()
        print(json.dumps(dict(passed=True, reference_labels=[r['label'] for r in refs.values()],
                              old_stage=status['stage'], processes=len(tree), baseline_decision=decision)))
    else:
        controller.run()
