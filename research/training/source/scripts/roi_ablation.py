"""One idle GPU, four matched arms; no experimental weights enter main training.

At dispatch, pin the latest accepted main checkpoint. Preserve its optimizer,
LR schedule and sampler cursor in every arm. Compare the two individual fixes
against the unchanged crop treatment, and require improvement over a separate
current-main control before producing an admission recipe. PStudio is a final
held-out report, never a selection input. Main training keeps advancing; an
older experimental checkpoint must never replace its newer weights.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np

from multicamera4rc.datasets.mixture import read_data_config
from multicamera4rc.launch import gpu_snapshot
from scripts.continue_quality_recovery import training_argv
from scripts.cutover_native_motion_stereo import write_json
from scripts.intrinsics_trial import Trial, calibration_decision, checkpoint_health
from scripts.migrate_dynamic_validation import build_validation_contract

ARMS = ('control', 'roi_control', 'roi_principal', 'roi_geometry')


def arm_settings(legacy_loss, log_focal_loss, original, cropped):
    """Only two single-variable comparisons: each candidate vs roi_control."""
    if legacy_loss.get('motion_confidence_targets') != ['trajectory', 'displacement', 'velocity']:
        raise ValueError('source motion-confidence contract changed; re-audit the proposed ablation')
    proposed = dict(legacy_loss, intrinsics_loss_mode='log_focal', focal_loss_weight=.5,
                    principal_loss_weight=.5)
    if log_focal_loss != proposed:
        raise ValueError('crop baseline must change only the already audited intrinsics loss')
    geometry = dict(proposed, motion_confidence_targets=['displacement', 'velocity'])
    return dict(
        control=dict(data_config=original, loss=legacy_loss, principal_point_mode='image'),
        roi_control=dict(data_config=cropped, loss=proposed, principal_point_mode='image'),
        roi_principal=dict(data_config=cropped, loss=proposed, principal_point_mode='crop_original'),
        roi_geometry=dict(data_config=cropped, loss=geometry, principal_point_mode='image'))


def available_gpu(devices, processes, candidates, protected):
    occupied = {p.gpu_uuid for p in processes}
    return next((i for i in candidates if i not in protected and i in devices
                 and devices[i].uuid not in occupied and devices[i].memory_used_mib < 1024), None)


def paired_task_groups(entries, task):
    """Sequence-balanced metrics; use actual C>1 observations for multiview."""
    groups = defaultdict(lambda: defaultdict(list))
    for suite, entry in entries.items():
        for row in entry['records']:
            source = (row['source_name'] if suite == 'native_dynamic' else row['source']
                      if suite == 'static' else row['sequence'].split('/')[0]
                      if suite == 'syn4d' else suite)
            if task == 'dynamic':
                if suite != 'native_dynamic':
                    continue
                metrics = ('native_trajectory_epe_temporal_normalized',
                           'native_displacement_epe_temporal_normalized') if source == 'pointodyssey' else (
                           'trajectory_epe_temporal_normalized', 'displacement_epe_dynamic_normalized')
            elif task == 'multicamera':
                cameras = row.get('actual_cameras', row.get('layout', [0])[0])
                if cameras < 2:
                    continue
                metrics = ('depth_abs_rel', 'base_epe_normalized') if suite == 'static' else (
                    'base_epe_normalized', 'trajectory_epe_temporal_normalized',
                    'displacement_epe_dynamic_normalized')
            else:
                raise ValueError('unknown task panel')
            sequence = row.get('physical_family', row['sequence'])
            for metric in metrics:
                value = row['metrics'][metric]
                if not math.isfinite(value) or value < 0:
                    raise ValueError('invalid task metric')
                groups[f'{suite}/{source}/{metric}'][sequence].append(value)
    if not groups:
        raise ValueError(f'empty {task} panel')
    return {name: {seq: float(np.mean(values)) for seq, values in scenes.items()}
            for name, scenes in groups.items()}


def paired_task_comparison(control, candidate, task):
    a, b = paired_task_groups(control, task), paired_task_groups(candidate, task)
    if a.keys() != b.keys() or any(a[k].keys() != b[k].keys() for k in a):
        raise ValueError('task panels must contain identical sequences')
    rng = np.random.default_rng(20261014)
    log_bootstrap = np.zeros(2000)
    ratios, draws = {}, {}
    for name in sorted(a):
        pairs = np.array([[a[name][s], b[name][s]] for s in sorted(a[name])])
        means = pairs.mean(0)
        ratios[name] = float(max(means[1], 1e-9)/max(means[0], 1e-9))
        # Reuse resampled physical sequences across metrics of a family.
        # Independent metric resampling would understate uncertainty.
        key = (name.rsplit('/', 1)[0], tuple(sorted(a[name])))
        if key not in draws:
            draws[key] = rng.integers(0, len(pairs), size=(2000, len(pairs)))
        indices = draws[key]
        means = pairs[indices].mean(1)
        log_bootstrap += np.log(np.maximum(means[:, 1], 1e-9)/np.maximum(means[:, 0], 1e-9))/len(a)
    ratio = math.exp(float(np.log(list(ratios.values())).mean()))
    ci = np.quantile(np.exp(log_bootstrap), [.025, .975]).tolist()
    return dict(passed=ratio <= .995 and ci[1] < 1 and max(ratios.values()) <= 1.05,
                score_ratio=ratio, paired_sequence_bootstrap_95pct=ci, metric_ratios=ratios,
                thresholds=dict(macro_ratio_max=.995, bootstrap_upper_max=1., worst_ratio_max=1.05))


def ablation_decision(control, candidate):
    camera = calibration_decision(control, candidate)
    dynamic = paired_task_comparison(control, candidate, 'dynamic')
    multicamera = paired_task_comparison(control, candidate, 'multicamera')
    return dict(passed=all(v['passed'] for v in (camera, dynamic, multicamera)),
                calibration=camera, dynamic=dynamic, multicamera=multicamera)


class ROITrial(Trial):
    def __init__(self, path):
        # GPU assignment happens only after a stable, completely idle interval.
        self.path = Path(path)
        self.job = json.loads(self.path.read_text())
        self.root, self.output = Path(self.job['repo']), Path(self.job['output_root'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.progress = dict(stage='starting', phases=[], physical_gpus=[], max_concurrent_gpus=1)
        self.environment = os.environ | self.job['environment']
        self.lease = None

    def check_code(self):
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.root, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                        cwd=self.root, text=True).strip()
        if head != self.job['commit'] or dirty:
            raise ValueError('trial source changed after registration')

    def acquire_gpu(self):
        stable = None
        while True:
            self.check_code()
            devices, processes = gpu_snapshot()
            active = json.loads(Path(self.job['main_active_path']).read_text())
            protected = set(self.job['protected_gpus']) | set(active.get('physical_gpus', []))
            index = available_gpu(devices, processes, self.job['gpu_candidates'], protected)
            now = time.monotonic()
            if index is None:
                stable = None
            elif stable is None or stable[0] != index:
                stable = (index, now)
            elif now-stable[1] >= self.job.get('idle_seconds', 60):
                lease = (Path('/tmp')/f'multicamera4rc_{devices[index].uuid}.lock').open('a')
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close(); stable = None
                else:
                    self.lease = lease
                    self.uuid = devices[index].uuid
                    self.job['physical_gpus'] = [index]
                    self.progress['physical_gpus'] = [index]
                    self.publish('gpu_acquired', gpu_uuid=self.uuid)
                    return
            self.publish('waiting_for_one_idle_gpu', candidate=index, protected_gpus=sorted(protected))
            time.sleep(20)

    def process(self, name, command, *, evaluation=False):
        self.check_code()
        # Recheck every phase, including evaluation, so another user's job
        # arriving between children is never treated as permission to share.
        while True:
            devices, processes = gpu_snapshot()
            if available_gpu(devices, processes, self.job['physical_gpus'], []) is not None:
                break
            self.publish('waiting_for_assigned_gpu', pending_phase=name)
            time.sleep(20)
        return super().process(name, command, evaluation=evaluation)

    def prepare(self):
        import torch
        manifest = self.output/'frozen_job.json'
        if manifest.exists():
            frozen = json.loads(manifest.read_text())
            gpu = self.job['physical_gpus']
            self.job.update(frozen, physical_gpus=gpu)
            self.settings = self.job['arms']
            return
        active = json.loads(Path(self.job['main_active_path']).read_text())
        accepted = json.loads((Path(active['output_root'])/'latest_accepted_checkpoint.json').read_text())
        path = accepted['checkpoint']
        payload = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        state, cfg = payload['training_state'], payload['config']
        begin = int(payload['step'])
        if (begin != accepted['step'] or not payload['optimizer']['state']
                or cfg.get('principal_point_mode', 'image') != 'image'
                or state.get('augmentation') is not None
                or state['world_size'] != 4 or state['batch_size'] != 1
                or state.get('gradient_accumulation', 1) != 1
                or state['dataset_size'] % 4
                or state.get('sampler_kind', 'distributed_v1') != 'distributed_v1' or len(payload['rank_rng']) != 4):
            raise ValueError('accepted main checkpoint no longer matches the audited control contract')
        config = read_data_config(active['data_config'])
        if config.get('augmentation') is not None:
            raise ValueError('main augmentation changed')
        proposed = json.loads(Path(self.job['log_focal_loss']).read_text())
        configs = self.output/'configs'; configs.mkdir(exist_ok=True)
        data = configs/'original.json'; cropped = configs/'roi.json'
        write_json(data, config)
        crop_config = dict(config, augmentation=dict(probability=.35, min_crop_fraction=.75, max_crop_fraction=1.))
        write_json(cropped, crop_config)
        self.settings = arm_settings(state['loss_config'], proposed, str(data), str(cropped))
        for label, setting in self.settings.items():
            loss = configs/(label+'_loss.json'); write_json(loss, setting['loss'])
            setting['loss_config'] = str(loss)
            setting['augmentation'] = None if label == 'control' else crop_config['augmentation']
        self.job.update(checkpoint=path, begin=begin, end=begin+1000, milestones=[begin+500, begin+1000],
                        source_lr_plan=state['lr_plan'], source_training_state=state, arms=self.settings,
                        data_config=str(data), roi_config=str(cropped), allow_shared=False,
                        gradient_accumulation=4, global_batch_size=4, main_at_start=accepted,
                        checkpoint_bytes=Path(path).stat().st_size)
        if build_validation_contract(self.job) != self.job['validation_contract']:
            raise ValueError('frozen validation inputs changed')
        self.job['config_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in configs.iterdir() if p.is_file()}
        write_json(manifest, self.job)
        del payload

    def check_trial_checkpoint(self, path, label, end):
        import torch
        setting = self.settings[label]
        state = checkpoint_health(self.job, path, setting['loss'], setting['augmentation'], step=end)
        data = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        expected = dict(self.job['source_training_state'])
        if data['config'].get('principal_point_mode', 'image') != setting['principal_point_mode']:
            raise ValueError('checkpoint principal semantics changed')
        # Same optimizer-update count is insufficient if the data cursor moved.
        total = (expected['data_epoch']*expected['dataset_size']
                 + expected['batches_consumed']*expected['world_size'] + 4*(end-self.job['begin']))
        consumed = state['data_epoch']*state['dataset_size']+state['batches_consumed']
        if state['dataset_size'] != expected['dataset_size'] or consumed != total:
            raise ValueError('trial did not consume the matched number of training clips')

    def train(self, label, *, resume, end):
        setting = self.settings[label]
        run = self.output/'arms'/f'{label}_{end}'
        # A completed milestone can be resumed without repeating any updates.
        if (run/'latest.pt').exists():
            import torch
            payload = torch.load(run/'latest.pt', map_location='cpu', mmap=True, weights_only=True)
            step = payload['step']; del payload
            if step == end:
                self.check_trial_checkpoint(run/'latest.pt', label, end)
                return run/'latest.pt'
            resume = run/'latest.pt'
        if run.exists():
            # Keep interrupted logs/checkpoints; do not overwrite their evidence.
            run = run.with_name(run.name+f'_resume_{int(time.time())}')
        command = training_argv(self.job, resume, setting['data_config'], run, end,
                                loss_config=setting['loss_config'])
        command[command.index('--save-every')+1] = '500'
        command += ['--gradient-accumulation', '4', '--principal-point-mode', setting['principal_point_mode']]
        if '--override-lr' in command or '--reset-data-cursor' in command or '--allow-shared' in command:
            raise ValueError('ablation must preserve LR/cursor and use an idle GPU')
        self.process(f'{label}_{end}_train', command)
        self.check_trial_checkpoint(run/'latest.pt', label, end)
        return run/'latest.pt'

    def final_benchmark(self, results):
        from scripts.evaluate_pstudio_calibration import compare_reports
        reports = {}
        for label in ARMS:
            output = self.output/'pstudio'/label
            self.process('pstudio_'+label, [self.job['python'], '-u', '-m',
                'scripts.evaluate_pstudio_calibration', '--config', self.job['pstudio'],
                '--checkpoint', results[label][0]['checkpoint'], '--output', str(output), '--resume'], evaluation=True)
            reports[label] = json.loads((output/'results.json').read_text())
        write_json(self.output/'pstudio/comparison.json', compare_reports(reports))

    def run(self):
        self.acquire_gpu()
        self.prepare()
        for path, digest in self.job['config_hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('frozen arm config changed')
        paths = {label: self.job['checkpoint'] for label in ARMS}
        history = []
        for end in self.job['milestones']:
            results = {}
            for label in ARMS:
                self.publish('milestone', milestone_step=end, arm=label, begin=self.job['begin'], end=self.job['end'])
                paths[label] = self.train(label, resume=paths[label], end=end)
                results[label] = self.evaluate(paths[label], f'{label}_{end}', step=end)
            comparisons = {label: dict(
                versus_main=ablation_decision(results['control'][1], results[label][1]),
                versus_crop=ablation_decision(results['roi_control'][1], results[label][1]))
                for label in ('roi_principal', 'roi_geometry')}
            history.append(dict(step=end, updates=end-self.job['begin'], comparisons=comparisons))
            write_json(self.output/'milestone_results.json', history)
        # No selection at 500 updates. Do not combine two changes that have
        # only been tested separately, or use the held-out PStudio test to tune.
        eligible = [k for k, v in comparisons.items() if v['versus_main']['passed'] and v['versus_crop']['passed']]
        selected = min(eligible, key=lambda k: comparisons[k]['versus_main']['calibration']['macro_log_focal_candidate']) if eligible else None
        decision = dict(selected=selected, comparisons=comparisons, milestones=history,
                        begin=self.job['begin'], end=self.job['end'], lr_changed=False,
                        test_used_for_selection=False, main_training_modified=False,
                        validation_clips=sum(len(v['records']) for v in results['control'][1].values()))
        write_json(self.output/'decision.json', decision)
        self.final_benchmark(results)
        if selected:
            # This is a validated recipe, NOT permission to discard newer main
            # optimizer updates. Integrate at a validated main boundary using
            # that boundary's current weights/optimizer and the admitted config.
            write_json(self.output/'admission_recipe.json', dict(
                arm=selected, settings=self.settings[selected], evidence=str(self.output/'decision.json'),
                lr_plan=self.job['source_lr_plan'], use_latest_main_weights=True,
                experimental_checkpoint_must_not_replace_newer_main=True,
                required_next_step='validate_current_main_boundary_then_apply_recipe'))
            self.publish('passed_recipe_ready_for_main_boundary', selected=selected)
        else:
            self.publish('no_candidate_passed_main_unchanged', decision=str(self.output/'decision.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    trial = ROITrial(args.job)
    with (trial.output/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            trial.run()
        except Exception:
            trial.publish('failed', error=traceback.format_exc())
            raise
        finally:
            if trial.lease is not None:
                trial.lease.close()


if __name__ == '__main__':
    main()
