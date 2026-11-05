"""Isolated one-GPU matched query-view-consistency trial at unchanged LR.

Both arms use the same synchronized source-pair policy, dense GT, checkpoint,
Adam moments, sampler cursor and RNG. Only the auxiliary weight differs.
Main and existing frozen experiments continue. A successful result writes an
admission recipe for the latest main boundary; old trial weights never replace
newer main weights. No learning-rate retries or test-set selection.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np

from multicamera4rc.config import MultiCamera4RCConfig
from multicamera4rc.datasets.mixture import read_data_config
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.quality import compare_errors, error_panel
from scripts.cutover_native_motion_stereo import write_json
from scripts.intrinsics_trial import checkpoint_health, calibration_groups
from scripts.migrate_dynamic_validation import build_validation_contract
from scripts.roi_ablation import ROITrial, paired_task_comparison


def arm_losses(base, weight=.01):
    if base.get('query_pair_weight', 0) or base.get('query_pair_source_sampling', False):
        raise ValueError('query pair consistency/policy is already enabled in main')
    return {name: MultiCamera4RCLoss(**dict(deepcopy(base), query_pair_weight=w,
                query_pair_source_sampling=True)).to_config() for name, w in (('control', 0.), ('candidate', weight))}


def pair_panel(entries):
    groups = defaultdict(lambda: defaultdict(list))
    masks = []
    for suite, entry in entries.items():
        for i, row in enumerate(entry['records']):
            family = (row['source_name'] if suite == 'native_dynamic' else row['source'] if suite == 'static'
                      else row['sequence'].split('/')[0] if suite == 'syn4d' else suite)
            seq = row.get('physical_family', row['sequence'])
            metrics = row['metrics']
            masks.append((suite, i, seq, metrics.get('query_pair_points', 0), metrics.get('query_pair_dynamic_temporal_points', 0)))
            for key in ('query_pair_corrected_epe', 'query_pair_dynamic_temporal_epe', 'cross_camera_3d_temporal_epe'):
                if key not in metrics or metrics.get('query_pair_points', 0) <= 0:
                    continue
                if key == 'query_pair_dynamic_temporal_epe' and metrics.get('query_pair_dynamic_temporal_points', 0) <= 0:
                    continue
                if key == 'cross_camera_3d_temporal_epe' and metrics.get('query_pair_temporal_points', 0) <= 0:
                    continue
                value = metrics[key]
                if not math.isfinite(value) or value < 0: raise ValueError('invalid pair metric')
                groups[family+'/'+key][seq].append(value)
    means = {name: float(np.mean([np.mean(v) for v in scenes.values()])) for name, scenes in groups.items()}
    return means, masks


def trial_decision(control, candidate):
    before, masks_a = pair_panel(control); after, masks_b = pair_panel(candidate)
    if masks_a != masks_b or before.keys() != after.keys() or not before:
        raise ValueError('pair comparison requires identical nonempty GT correspondence panels')
    physical = compare_errors(error_panel(control), error_panel(candidate))
    dynamic = paired_task_comparison(control, candidate, 'dynamic')
    multicamera = paired_task_comparison(control, candidate, 'multicamera')
    ratios = {k: max(after[k], 1e-9)/max(v, 1e-9) for k, v in before.items()}
    cross = [v for k, v in ratios.items() if k.endswith('/cross_camera_3d_temporal_epe')]
    agreement = [v for k, v in ratios.items() if k.endswith('/query_pair_corrected_epe')]
    dynamic_pairs = [v for k, v in ratios.items() if k.endswith('/query_pair_dynamic_temporal_epe')]
    macro = lambda values: math.exp(float(np.log(values).mean())) if values else float('inf')
    camera_a, camera_b = calibration_groups(control), calibration_groups(candidate)
    bad_camera = []
    if camera_a.keys() != camera_b.keys(): raise ValueError('different calibration families')
    for family in camera_a:
        if camera_a[family].keys() != camera_b[family].keys(): raise ValueError('different calibration clips')
        for key in ('intrinsics_log_focal_error', 'intrinsics_principal_error_normalized'):
            a = np.mean([v[key] for v in camera_a[family].values()])
            b = np.mean([v[key] for v in camera_b[family].values()])
            if b > 1.05*a and b-a > .002: bad_camera.append(family+'/'+key)
    passed = (physical['score_ratio'] <= 1.01 and physical['worst_metric_ratio'] <= 1.05
        and dynamic['score_ratio'] <= 1.01 and max(dynamic['metric_ratios'].values()) <= 1.05
        and multicamera['score_ratio'] <= 1.01 and max(multicamera['metric_ratios'].values()) <= 1.05
        and macro(cross) <= .99 and macro(agreement) <= .99 and macro(dynamic_pairs) <= .99
        and max(ratios.values()) <= 1.05 and not bad_camera)
    return dict(passed=bool(passed), physical=physical, dynamic=dynamic, multicamera=multicamera,
        pair_metric_ratios=ratios, cross_camera_ratio=macro(cross), agreement_ratio=macro(agreement),
        dynamic_pair_ratio=macro(dynamic_pairs), regressed_calibration=bad_camera,
        thresholds=dict(physical_macro_max=1.01, worst_metric_max=1.05, pair_macro_max=.99),
        gt_correspondence_panels_identical=True, test_used_for_selection=False)


class QueryPairTrial(ROITrial):
    def check_code(self):
        super().check_code()
        for path, digest in self.job.get('protected_files', {}).items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('independent protected experiment changed; recheck scheduling')

    def prepare(self):
        import torch
        frozen = self.output/'frozen_job.json'
        if frozen.exists():
            gpu = self.job['physical_gpus']
            self.job.update(json.loads(frozen.read_text()), physical_gpus=gpu)
            self.settings = self.job['arms']; return
        active = json.loads(Path(self.job['main_active_path']).read_text())
        accepted = json.loads((Path(active['output_root'])/'latest_accepted_checkpoint.json').read_text())
        source = Path(accepted['checkpoint'])
        checkpoint = self.output/'inputs'/'accepted.pt'; checkpoint.parent.mkdir(exist_ok=True)
        os.link(source, checkpoint)  # atomic-save training never mutates this inode
        payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        state = payload['training_state']; cfg = payload['config']
        if (payload['step'] != accepted['step'] or state['batch_size'] != 1
            or state.get('effective_batch_size', state['world_size']*state.get('gradient_accumulation', 1)) != 4
            or not payload['optimizer']['state'] or state.get('sampler_kind', 'distributed_v1') != 'distributed_v1'):
            raise ValueError('accepted checkpoint must preserve Adam and effective batch four')
        data = self.output/'configs'/'data.json'; data.parent.mkdir(exist_ok=True)
        write_json(data, read_data_config(active['data_config']))
        losses = arm_losses(state['loss_config'], self.job.get('weight', .01))
        self.settings = {}
        for name, loss in losses.items():
            path = data.parent/(name+'_loss.json'); write_json(path, loss)
            self.settings[name] = dict(loss=loss, loss_config=str(path), data_config=str(data),
                augmentation=state.get('augmentation'), principal_point_mode=cfg.get('principal_point_mode', 'image'))
        begin = int(payload['step'])
        self.job.update(checkpoint=str(checkpoint), begin=begin, end=begin+1000,
            milestones=[begin+500, begin+1000], arms=self.settings,
            data_config=str(data), allow_shared=False, gradient_accumulation=4, global_batch_size=4,
            source_training_state=state, source_lr_plan=state['lr_plan'], source_model_config=cfg,
            evaluation=active['evaluation'], syn4d_eval_plan=active['syn4d_eval_plan'],
            syn4d_eval_config=active['syn4d_eval_config'], static_contract=accepted['static_contract'],
            main_at_start=accepted, main_config_at_start=active)
        self.job['validation_contract'] = build_validation_contract(self.job)
        self.job['config_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in data.parent.iterdir()}
        write_json(frozen, self.job); del payload

    def check_trial_checkpoint(self, path, label, end):
        import torch
        setting = self.settings[label]
        state = checkpoint_health(self.job, path, setting['loss'], setting['augmentation'], step=end)
        payload = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        if MultiCamera4RCConfig.from_dict(payload['config']) != MultiCamera4RCConfig.from_dict(self.job['source_model_config']):
            raise ValueError('paired objective trial changed architecture')
        before = self.job['source_training_state']
        expected = (before['data_epoch']*before['dataset_size']+before['batches_consumed']*before['world_size']
                    +4*(end-self.job['begin']))
        actual = state['data_epoch']*state['dataset_size']+state['batches_consumed']
        if expected != actual or state['dataset_size'] != before['dataset_size']:
            raise ValueError('trial changed its matched data cursor')

    def run(self):
        proof = json.loads(Path(self.job['gt_audit']).read_text())
        if proof.get('stage') != 'passed' or proof.get('failures'):
            raise ValueError('production correspondence GT audit must pass before GPU use')
        required = {'kubric_variable', 'mv_kubric_official', 'syn4d_v1', 'syn4d_sim', 'vkitti2_stereo'}
        if not required.issubset({r['source'] for r in proof['cases'] if r['metrics']['query_pair_dynamic_temporal_points'] > 0}):
            raise ValueError('dynamic multi-camera GT coverage is incomplete')
        if self.job.get('mixed_gt_audit'):
            while True:
                self.check_code()
                path = Path(self.job['mixed_gt_audit'])
                report = json.loads(path.read_text()) if path.exists() else {}
                if report.get('failures') or report.get('stage') == 'failed':
                    raise ValueError('mixed-source GT audit failed; main unchanged')
                if report.get('stage') == 'passed':
                    if len(report['cases']) != self.job['expected_mixed_audit_cases']:
                        raise ValueError('mixed-source audit coverage is incomplete')
                    break
                self.publish('waiting_for_mixed_gt_audit', completed_cases=len(report.get('cases', [])))
                time.sleep(20)
        self.acquire_gpu(); self.prepare()
        for path, digest in self.job['config_hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest: raise ValueError('trial config changed')
        preflight = self.output/'cuda_preflight.json'
        if not preflight.exists() or not json.loads(preflight.read_text()).get('passed'):
            command = [self.job['python'], '-u', '-m', 'scripts.preflight_query_pair',
                '--config', self.job['data_config'], '--checkpoint', self.job['checkpoint'],
                '--loss-config', self.settings['candidate']['loss_config'], '--output', str(preflight), '--gpu-uuid', self.uuid]
            self.process('cuda_preflight', command, evaluation=True)
        if not json.loads(preflight.read_text()).get('passed'): raise ValueError('giant preflight did not pass')
        paths = {name: self.job['checkpoint'] for name in self.settings}; history = []
        for end in self.job['milestones']:
            results = {}
            for name in ('control', 'candidate'):
                self.publish('milestone', arm=name, milestone_step=end, begin=self.job['begin'], end=self.job['end'])
                paths[name] = self.train(name, resume=paths[name], end=end)
                results[name] = self.evaluate(paths[name], f'{name}_{end}', step=end)
            comparison = trial_decision(results['control'][1], results['candidate'][1])
            history.append(dict(step=end, updates=end-self.job['begin'], comparison=comparison))
            write_json(self.output/'milestone_results.json', history)
        decision = dict(selected='candidate' if comparison['passed'] else None, comparison=comparison,
            begin=self.job['begin'], end=self.job['end'], milestones=history, lr_changed=False,
            main_training_modified=False, test_used_for_selection=False)
        write_json(self.output/'decision.json', decision)
        if comparison['passed']:
            write_json(self.output/'admission_recipe.json', dict(arm='query_pair',
                loss=self.settings['candidate']['loss'], source_loss=self.job['source_training_state']['loss_config'],
                use_latest_main_weights=True, restore_optimizer_rng_cursor=True, lr_changed=False,
                experimental_checkpoint_must_not_replace_newer_main=True, evidence=str(self.output/'decision.json'),
                required_next_step='apply_at_latest_accepted_main_boundary_after_config_migration_check'))
            self.publish('passed_recipe_ready_for_main_boundary')
        else: self.publish('candidate_rejected_main_unchanged')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--job', required=True)
    args = parser.parse_args(); trial = QueryPairTrial(args.job)
    with (trial.output/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        trial.publish('starting', pid=os.getpid())
        try: trial.run()
        except Exception:
            trial.publish('failed_main_unchanged', error=traceback.format_exc()); raise
        finally:
            if trial.lease is not None: trial.lease.close()


if __name__ == '__main__': main()
