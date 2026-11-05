"""Queued, one-GPU, matched trials of two independent intrinsic interventions.

Wait for the entire ROI trial (including held-out reporting) before taking any
GPU. Start all three arms from the latest accepted main checkpoint at dispatch.
Keep LR/Adam/data cursor, all loss weights and augmentation fixed. Admission
requires calibration AND dynamic AND multiview gains, with no material losses.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

from multicamera4rc.config import MultiCamera4RCConfig
from multicamera4rc.datasets.mixture import read_data_config
from scripts.continue_quality_recovery import training_argv
from scripts.cutover_native_motion_stereo import write_json
from scripts.intrinsics_trial import checkpoint_health
from scripts.migrate_dynamic_validation import build_validation_contract
from scripts.roi_ablation import ROITrial, ablation_decision

ARMS = ('control', 'gradient_isolation', 'shared_intrinsics')
TERMINAL_PREDECESSOR_STAGES = {'no_candidate_passed_main_unchanged',
                             'passed_recipe_ready_for_main_boundary'}


def source_sampler_kind(state):
    # Historical DDP checkpoints predate this metadata field. Training used
    # DistributedSampler whenever world_size>1; single-rank legacy used random.
    return state.get('sampler_kind','distributed_v1' if state.get('world_size',1)>1 else 'random_v1')


def arm_settings(loss, config, data_path):
    if loss.get('absolute_intrinsics_gradient', 'full') != 'full':
        raise ValueError('main already has gradient isolation; this contrast is no longer valid')
    if config.get('intrinsics_sharing', 'none') != 'none':
        raise ValueError('main already shares intrinsics; this contrast is no longer valid')
    base = dict(loss=deepcopy(loss), data_config=data_path,
                principal_point_mode=config.get('principal_point_mode', 'image'), intrinsics_sharing='none')
    return dict(control=deepcopy(base),
        gradient_isolation=dict(deepcopy(base), loss=dict(deepcopy(loss), absolute_intrinsics_gradient='isolated')),
        shared_intrinsics=dict(deepcopy(base), intrinsics_sharing='camera'))


def predecessor_complete(root):
    root = Path(root)
    required = ['progress.json', 'decision.json', 'pstudio/comparison.json']
    if not all((root/name).is_file() for name in required):
        return False
    progress, decision, benchmark = (json.loads((root/name).read_text()) for name in required)
    if progress.get('stage') not in TERMINAL_PREDECESSOR_STAGES:
        return False
    if (decision['end']-decision['begin'] != 1000 or len(decision['milestones']) != 2
            or benchmark.get('stage') != 'passed' or benchmark.get('clips') != 954
            or benchmark.get('sequences') != 106 or decision.get('test_used_for_selection')):
        raise ValueError('predecessor is marked done without its complete frozen protocol')
    with (root/'controller.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def check_sharing_audit(report):
    if (report.get('stage') != 'passed' or report.get('failures')
            or report.get('timeline_truncated') or report.get('gt_used_for_model_input') is not False
            or report.get('audited_clips', 0) < 50):
        raise ValueError('sharing requires a complete fixed-intrinsics GT audit')


def pstudio_admission_guard(comparison, selected):
    """A predeclared veto for regression, never rank arms using the benchmark."""
    if (comparison.get('stage') != 'passed' or comparison.get('clips') != 954
            or comparison.get('sequences') != 106):
        raise ValueError('PStudio admission requires the complete matched benchmark')
    control, candidate = comparison['means']['control'], comparison['means'][selected]
    keys = ('intrinsics_log_focal_error', 'source_intrinsics_log_focal_error', 'epe_m',
            'intrinsics_principal_error_normalized', 'source_intrinsics_principal_error_normalized', 'apd3d')
    if any(not math.isfinite(values[k]) or values[k] < 0 for values in (control, candidate) for k in keys):
        raise ValueError('invalid PStudio admission metrics')
    ratios = {k:candidate[k]/max(control[k], 1e-9) for k in keys[:3]}
    principal_bad = [k for k in keys[3:5] if candidate[k] > 1.10*control[k]
                     and candidate[k]-control[k] > .002]
    apd_change = candidate['apd3d']-control['apd3d']
    return dict(passed=max(ratios.values()) <= 1.05 and not principal_bad and apd_change >= -.005,
        error_ratios=ratios, regressed_principal_metrics=principal_bad, apd3d_change=apd_change,
        thresholds=dict(error_ratio_max=1.05, apd3d_drop_max=.005,
                        principal_relative_max=1.10, principal_absolute_allowance=.002),
        test_used_for_candidate_ranking=False, test_used_for_admission=True)


class StructureTrial(ROITrial):
    def wait_predecessor(self):
        while not predecessor_complete(self.job['predecessor_root']):
            self.check_code()
            path = Path(self.job['predecessor_root'])/'progress.json'
            prior = json.loads(path.read_text()) if path.exists() else {}
            self.publish('waiting_for_roi_trial_completion', predecessor_stage=prior.get('stage'),
                         predecessor_root=self.job['predecessor_root'], physical_gpus=[])
            time.sleep(30)

    def prepare(self):
        import torch
        manifest = self.output/'frozen_job.json'
        if manifest.exists():
            frozen = json.loads(manifest.read_text())
            self.job.update(frozen, physical_gpus=self.job['physical_gpus'])
            self.settings = self.job['arms']
            return
        active = json.loads(Path(self.job['main_active_path']).read_text())
        accepted = json.loads((Path(active['output_root'])/'latest_accepted_checkpoint.json').read_text())
        payload = torch.load(accepted['checkpoint'], map_location='cpu', mmap=True, weights_only=True)
        state, cfg, begin = payload['training_state'], payload['config'], int(payload['step'])
        if (begin != accepted['step'] or not payload['optimizer']['state']
                or state['world_size'] != 4 or state['batch_size'] != 1
                or state.get('gradient_accumulation', 1) != 1 or state['dataset_size'] % 4
                or source_sampler_kind(state) != 'distributed_v1' or len(payload['rank_rng']) != 4):
            raise ValueError('latest main no longer matches the matched effective-batch-four contract')
        config = read_data_config(active['data_config'])
        if config.get('augmentation') != state.get('augmentation') or config['image_mode'] != 'native':
            raise ValueError('native preprocessing/augmentation changed without a new audit')
        if build_validation_contract(self.job) != self.job['validation_contract']:
            raise ValueError('validation inputs changed')
        audit = json.loads(Path(self.job['sharing_audit']).read_text())
        check_sharing_audit(audit)
        if (read_data_config(audit['data_config']) != config
                or audit.get('data_config_sha256') != hashlib.sha256(Path(active['data_config']).read_bytes()).hexdigest()):
            raise ValueError('audit does not cover the actual main data configuration')
        configs = self.output/'configs'; configs.mkdir(exist_ok=True)
        data = configs/'unchanged_mixture.json'; write_json(data, config)
        self.settings = arm_settings(state['loss_config'], cfg, str(data))
        for name, setting in self.settings.items():
            path = configs/(name+'_loss.json'); write_json(path, setting['loss'])
            setting.update(loss_config=str(path), augmentation=state.get('augmentation'))
        required_space = 8*Path(accepted['checkpoint']).stat().st_size
        if shutil.disk_usage(self.output).free < required_space:
            raise RuntimeError('insufficient checkpoint space for paired arms and boundary admission')
        self.job.update(checkpoint=accepted['checkpoint'], begin=begin, end=begin+1000,
            milestones=[begin+500, begin+1000], source_lr_plan=state['lr_plan'],
            source_training_state=state, source_model_config=cfg, arms=self.settings,
            data_config=str(data), allow_shared=False, gradient_accumulation=4, global_batch_size=4,
            main_at_start=accepted, main_config_at_start=active,
            checkpoint_bytes=Path(accepted['checkpoint']).stat().st_size)
        files = list(configs.iterdir())+[Path(self.job['sharing_audit'])]
        self.job['config_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        write_json(manifest, self.job)

    def check_trial_checkpoint(self, path, label, end):
        import torch
        setting = self.settings[label]
        state = checkpoint_health(self.job, path, setting['loss'], setting['augmentation'], step=end)
        payload = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        expected_cfg = dict(self.job['source_model_config'], intrinsics_sharing=setting['intrinsics_sharing'])
        if MultiCamera4RCConfig.from_dict(payload['config']) != MultiCamera4RCConfig.from_dict(expected_cfg):
            raise ValueError('model configuration changed beyond the single declared intervention')
        source = self.job['source_training_state']
        expected = source['data_epoch']*source['dataset_size']+source['batches_consumed']*4+4*(end-self.job['begin'])
        actual = state['data_epoch']*state['dataset_size']+state['batches_consumed']
        if expected != actual or state['dataset_size'] != source['dataset_size']:
            raise ValueError('trial sampling cursor differs from matched control')

    def train(self, label, *, resume, end):
        import torch
        setting = self.settings[label]
        run = self.output/'arms'/f'{label}_{end}'
        if (run/'latest.pt').exists():
            payload = torch.load(run/'latest.pt', map_location='cpu', mmap=True, weights_only=True)
            step = payload['step']; del payload
            if step == end:
                self.check_trial_checkpoint(run/'latest.pt', label, end)
                return run/'latest.pt'
            resume = run/'latest.pt'
        if run.exists():
            run = run.with_name(run.name+f'_resume_{int(time.time())}')
        command = training_argv(self.job, resume, setting['data_config'], run, end,
                                loss_config=setting['loss_config'])
        command[command.index('--save-every')+1] = '500'
        command += ['--gradient-accumulation', '4', '--intrinsics-sharing', setting['intrinsics_sharing']]
        if set(command) & {'--override-lr','--reset-data-cursor','--allow-shared','--lr-schedule'}:
            raise ValueError('trial must preserve LR/cursor and use only one idle GPU')
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
        self.wait_predecessor()
        # The audit is CPU-only and complete before reserving a GPU.
        check_sharing_audit(json.loads(Path(self.job['sharing_audit']).read_text()))
        self.acquire_gpu()
        self.prepare()
        for path, digest in self.job['config_hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('frozen trial evidence changed')
        if self.job.get('cuda_preflight', True):
            self.process('cuda_structure_preflight', [self.job['python'], '-m', 'pytest',
                '-q', 'tests/multicamera4rc/test_intrinsics_structure.py', '-k', 'cuda_full_path'], evaluation=True)
        paths = {name:self.job['checkpoint'] for name in ARMS}
        history = []
        for end in self.job['milestones']:
            results = {}
            for label in ARMS:
                self.publish('milestone', arm=label, milestone_step=end, begin=self.job['begin'], end=self.job['end'])
                paths[label] = self.train(label, resume=paths[label], end=end)
                results[label] = self.evaluate(paths[label], f'{label}_{end}', step=end)
            comparisons = {name:ablation_decision(results['control'][1], results[name][1]) for name in ARMS[1:]}
            history.append(dict(step=end, updates=end-self.job['begin'], comparisons=comparisons))
            write_json(self.output/'milestone_results.json', history)
        eligible = [name for name in ARMS[1:] if comparisons[name]['passed']]
        selected = min(eligible, key=lambda n:comparisons[n]['calibration']['macro_log_focal_candidate']) if eligible else None
        decision = dict(selected=selected, comparisons=comparisons, milestones=history,
            begin=self.job['begin'], end=self.job['end'], lr_changed=False, test_used_for_selection=False,
            combined_changes_tested=False, main_training_modified=False,
            validation_clips=sum(len(v['records']) for v in results['control'][1].values()))
        write_json(self.output/'decision.json', decision)
        self.final_benchmark(results)
        if selected:
            # The test cannot switch to a different candidate or tune a loss.
            # A regression can only veto automatic admission of the frozen arm.
            benchmark = json.loads((self.output/'pstudio/comparison.json').read_text())
            test_guard = pstudio_admission_guard(benchmark, selected)
            write_json(self.output/'pstudio/admission_guard.json', dict(test_guard, selected=selected))
            if not test_guard['passed']:
                self.publish('validation_passed_pstudio_regressed_main_unchanged', selected=selected)
                return
            recipe = dict(arm=selected, settings=self.settings[selected], evidence=str(self.output/'decision.json'),
                lr_plan=self.job['source_lr_plan'], use_latest_main_weights=True,
                experimental_checkpoint_must_not_replace_newer_main=True,
                pstudio_admission_guard=str(self.output/'pstudio/admission_guard.json'))
            write_json(self.output/'admission_recipe.json', recipe)
            if self.job.get('auto_adopt', False):
                from scripts.intrinsics_structure_admission import adopt_at_main_boundary
                adopt_at_main_boundary(self, recipe)
            else:
                self.publish('passed_recipe_ready_for_main_boundary', selected=selected)
        else:
            self.publish('no_candidate_passed_main_unchanged')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    trial = StructureTrial(args.job)
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
