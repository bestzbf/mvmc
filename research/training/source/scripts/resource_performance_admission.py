"""Measured execution-only optimization on the main controller's leased GPUs.

Frozen native checkpoints initialize every speed arm. Experimental updates are
discarded; only five subsequent native production updates can advance main.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import math
import json
import statistics
import subprocess
import threading
import time
import traceback

from multicamera4rc.checkpoint import config_from_checkpoint
from scripts.admit_throughput import Admission, training_rows
from scripts.continue_quality_recovery import training_argv, validate_checkpoint
from scripts.continue_resource_training import execution, cursor_progress, training_health
from scripts.cutover_native_motion_stereo import write_json
from scripts.migrate_dynamic_validation import require_fixed_training_state
from scripts.query_pair_main_admission import read, verify_files


def equivalence_argv(job, checkpoint, output, world):
    # Match the native launcher. This machine's hostname resolves to another
    # interface; torchrun --standalone stalls before spawning any workers.
    return [job['python'], '-u', '-m', 'torch.distributed.run', '--nnodes=1',
        '--rdzv-backend=c10d', '--rdzv-endpoint=127.0.0.1:0', '--local-addr=127.0.0.1',
        '--nproc-per-node=' + str(world), '-m', 'scripts.validate_throughput_equivalence',
        '--checkpoint', checkpoint, '--data-config', job['data_config'], '--output', str(output)]


class PowerSamples:
    """Read-only observations; sampling failures cannot affect an optimizer."""
    def __init__(self, output, gpus, log, begin):
        self.output, self.gpus, self.log, self.begin = Path(output), gpus, Path(log), begin
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with self.output.open('w') as stream:
            while not self.stop.is_set():
                try:
                    result = subprocess.check_output(['nvidia-smi', '--id=' + ','.join(map(str, self.gpus)),
                        '--query-gpu=index,utilization.gpu,power.draw,memory.used',
                        '--format=csv,noheader,nounits'], text=True, timeout=5)
                    rows = training_rows(self.log) if self.log.exists() else []
                    last = rows[-1]['step'] if rows else None
                    sample = dict(time=time.time(), last_step=last,
                        measured=last is not None and self.begin + 8 <= last < self.begin + 24,
                        gpus=[[float(v.strip()) for v in line.split(',')] for line in result.strip().splitlines()])
                    stream.write(json.dumps(sample) + '\n')
                    stream.flush()
                except Exception as error:
                    stream.write(json.dumps(dict(time=time.time(), error=str(error))) + '\n')
                    stream.flush()
                self.stop.wait(2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(timeout=8)


def speed_comparison(reference, candidate, world, window=8):
    count = 3 * window
    if len(reference) != count or len(candidate) != count:
        raise ValueError('speed arms require one warmup and two measured windows')
    for rows in (reference, candidate):
        for r in rows:
            if (r['world_size'] != world or r['effective_batch_size'] != world
                    or r['gradient_accumulation'] != 1 or len(r['rank_batches']) != world
                    or r['supervision'] != 'dense' or r['optimizer_step_skipped']
                    or not math.isfinite(r['loss'])):
                raise ValueError('speed arm changed execution or supervision contract')
    for a, b in zip(reference, candidate):
        if any(a.get(k) != b.get(k) for k in ('step', 'lr', 'backbone_lr', 'refinement_lr')):
            raise ValueError('speed arm changed optimizer update or learning rates')
    for start in range(0, count, window):
        multisets = [Counter((name, index) for r in rows[start:start + window]
            for batch in r['rank_batches'] for name, index in zip(batch['dataset'], batch['sample_index']))
            for rows in (reference, candidate)]
        if multisets[0] != multisets[1]:
            raise ValueError('speed windows changed sample multisets')
    times = [[r['step_timing']['seconds'] for r in rows[window:]] for rows in (reference, candidate)]
    if any(not math.isfinite(t) or t <= 0 for ts in times for t in ts):
        raise ValueError('invalid step timing')
    old, new = times
    ratios = [sum(new[i:i + window]) / sum(old[i:i + window]) for i in range(0, 2 * window, window)]
    ratio = sum(new) / sum(old)
    return dict(reference_seconds=statistics.mean(old), candidate_seconds=statistics.mean(new),
        reference_samples_per_second=world / statistics.mean(old),
        candidate_samples_per_second=world / statistics.mean(new), speedup=1 / ratio,
        window_ratios=ratios, measured_updates=2 * window, identical_sample_multisets=True,
        passed=ratio <= .95 and max(ratios) <= 1.02)


def balance_state(state, costs, world, *, reset=False):
    return dict(version=1, window_batches=8, start_epoch=0 if reset else state['data_epoch'],
        start_batch=0 if reset else state['batches_consumed'], world_size=world,
        dataset_size=state['dataset_size'], source_count=state['source_count'],
        max_source_time_pairs=state['max_source_time_pairs'], image_hw=deepcopy(costs))


def save_execution_checkpoint(source, destination, *, shared, kv, balance, reset=False):
    import torch
    payload = torch.load(source, mmap=True, map_location='cpu', weights_only=True)
    payload['config'] = replace(config_from_checkpoint(payload),
        reuse_shared_computation=shared, reuse_decoder_kv=kv).to_dict()
    state = deepcopy(payload['training_state'])
    state['load_balancing'] = balance
    if reset:
        state.update(data_epoch=0, batches_consumed=0)
    payload['training_state'] = state
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(destination)
    return payload['config'], state


class ResourceAdmission(Admission):
    def __init__(self, controller, checkpoint, step):
        self.controller = controller
        self.root = controller.root
        self.previous = deepcopy(controller.job)
        self.spec = controller.job['performance_admission']
        self.out = Path(self.spec['output']) / f'step_{step}'
        self.out.mkdir(parents=True, exist_ok=False)
        self.world, accumulation, batch = execution(self.previous)
        if accumulation != 1 or batch != self.world:
            raise ValueError('this admission requires one sample per rank')
        self.job = dict(boundary_step=step, costs=self.spec['costs'])
        self.state = dict(source_checkpoint=checkpoint, source_step=step, world_size=self.world)
        self.replaced = False

    def write(self, stage=None, **values):
        if stage:
            self.state['stage'] = stage
        self.state.update(values)
        write_json(self.out / 'progress.json', self.state)
        self.controller.progress['performance_trial'] = dict(self.state, output=str(self.out))
        self.controller.publish('performance_admission')

    def bench(self, name, checkpoint, flags):
        begin, end = self.boundary_step, self.boundary_step + 24
        run = self.out / 'benchmarks' / name
        argv = training_argv(self.previous, checkpoint, self.previous['data_config'], run, end,
            loss_config=self.previous['loss_config_path'])
        # No detailed CUDA synchronization in speed arms: measure production's
        # actual end-to-end execution, with the same diagnostics and full GT.
        argv += ['--gradient-accumulation', '1', *flags]
        with PowerSamples(self.out / (name + '_power.jsonl'), self.previous['physical_gpus'],
                          self.out / ('bench_' + name + '.log'), begin):
            log = self.command('bench_' + name, argv, timeout=7200)
        training_health(log, begin, end, self.world, 1, self.world)
        before = validate_checkpoint(checkpoint, begin, world_size=self.world, batch_size=1)
        after = validate_checkpoint(run / 'latest.pt', end, world_size=self.world, batch_size=1)
        require_fixed_training_state(before, after)
        cursor_progress(before, after, 24, self.world)
        return training_rows(log)

    def experiments(self, checkpoint):
        equiv = self.out / 'equivalence'
        self.command('equivalence', equivalence_argv(self.previous, checkpoint, equiv, self.world), timeout=7200)
        reports = [read(equiv / f'rank_{r}.json') for r in range(self.world)]
        if any(r['checkpoint'] != checkpoint or len(r['records']) != 2 for r in reports):
            raise ValueError('native equivalence coverage or checkpoint changed')
        if not all(any(case.get('decoded_motion') for case in r['records']) for r in reports):
            raise ValueError('each numerical-validation worker must exercise the actual motion decoder')
        modes = set.intersection(*(set(r['passed_modes']) for r in reports))
        reference = self.bench('reference', checkpoint, [])
        shared = 'shared' in modes
        flags = ['--reuse-shared-computation'] if shared else []
        balanced = ['--balance-window', '8', '--balance-costs', self.job['costs']]
        name = 'shared_balanced' if shared else 'balanced'
        candidates = {name: (flags, self.bench(name, checkpoint, flags + balanced))}
        if 'shared_kv' in modes:
            kv_flags = ['--reuse-shared-computation', '--reuse-decoder-kv']
            try:
                candidates['shared_kv_balanced'] = (kv_flags,
                    self.bench('shared_kv_balanced', checkpoint, kv_flags + balanced))
            except Exception:
                self.write(kv_rejection=traceback.format_exc())
        comparisons = {name: speed_comparison(reference, rows, self.world)
                       for name, (_, rows) in candidates.items()}
        eligible = [name for name, result in comparisons.items() if result['passed']]
        result = dict(numerically_passed_modes=sorted(modes), comparisons=comparisons,
            selected_mode=min(eligible, key=lambda name: comparisons[name]['candidate_seconds']) if eligible else None)
        write_json(self.out / 'speed_comparison.json', result)
        if not eligible:
            raise ValueError('no optimization passed the measured five-percent speed gate')
        self.write('speed_gate_passed', result=result)
        return candidates[result['selected_mode']][0]


def admit(controller, checkpoint, step):
    spec = controller.job['performance_admission']
    verify_files(spec['proof_hashes'])
    proof = read(spec['cpu_proof'])
    if not proof['passed'] or proof['commit'] != controller.job['commit']:
        raise ValueError('performance runtime lacks exact-commit tests')
    trial = ResourceAdmission(controller, checkpoint, step)
    controller.job['performance_admission_attempted'] = True
    original_state = controller.check(checkpoint, step)
    original_config = deepcopy(controller.job['model_config'])
    original_expected = deepcopy(controller.job['expected_state'])
    try:
        flags = trial.experiments(checkpoint)
        migrated = trial.out / f'execution_{step}.pt'
        config, state = save_execution_checkpoint(checkpoint, migrated,
            shared='--reuse-shared-computation' in flags, kv='--reuse-decoder-kv' in flags,
            balance=balance_state(original_state, read(spec['costs']), trial.world))
        controller.job.update(model_config=config, expected_state=state)
        controller.check(str(migrated), step)
        run = trial.out / 'production_preflight'
        argv = training_argv(controller.job, migrated, controller.job['data_config'], run, step + 5,
            loss_config=controller.job['loss_config_path']) + ['--gradient-accumulation', '1']
        log = trial.command('production_preflight', argv, timeout=7200)
        health = training_health(log, step, step + 5, trial.world, 1, trial.world)
        candidate = str(run / 'latest.pt')
        after = controller.check(candidate, step + 5)
        cursor_progress(original_state, after, 5, trial.world)
        write_json(trial.out / 'production_preflight_health.json', health)
        receipt = dict(passed=True, source_checkpoint=checkpoint, selected_checkpoint=candidate,
            source_step=step, step=step + 5, model_config=config, load_balancing=state['load_balancing'],
            lr_plan_unchanged=state['lr_plan'] == original_state['lr_plan'],
            experimental_weights_adopted=False, health=health)
        write_json(trial.out / 'accepted.json', receipt)
        controller.progress.update(performance_status='accepted', latest_checkpoint=candidate, step=step + 5)
        write_json(controller.output / 'latest_training_checkpoint.json', dict(checkpoint=candidate, step=step + 5))
        trial.write('accepted', receipt=receipt)
        return candidate, step + 5
    except Exception:
        controller.job.update(model_config=original_config, expected_state=original_expected)
        controller.progress['performance_status'] = 'rejected_resume_original'
        trial.write('rejected', accepted=False, error=traceback.format_exc(),
                    resume_checkpoint=checkpoint, resume_step=step)
        return checkpoint, step


def reanchor_data(controller, checkpoint, step):
    """Preserve the previously admitted data reset with a new balanced stream."""
    from multicamera4rc.datasets.mixture import create_dataset
    before = controller.check(checkpoint, step)
    dataset = create_dataset(controller.job['data_config'])
    state = dict(before, dataset_size=len(dataset))
    costs = read(controller.job['performance_admission']['costs'])
    names = {source['display_name'] for source in dataset.sources}
    if not names <= costs.keys():
        raise ValueError('pending dataset is absent from frozen execution-cost catalogue')
    if state['dataset_size'] != before['dataset_size']:
        raise ValueError('pending mixture size changed beyond the audited reset')
    config = controller.job['model_config']
    path = controller.control / f'balanced_data_reset_{step}.pt'
    _, after = save_execution_checkpoint(checkpoint, path,
        shared=config.get('reuse_shared_computation', False), kv=config.get('reuse_decoder_kv', False),
        balance=balance_state(state, costs, len(controller.job['physical_gpus']), reset=True), reset=True)
    controller.job['expected_state'] = after
    controller.job.pop('reset_data_cursor_once', None)
    controller.job['data_activation'].update(balanced_reset_checkpoint=str(path),
        reset_completed_in_checkpoint=True, load_balancing=after['load_balancing'])
    write_json(controller.control / 'data_activation.json', controller.job['data_activation'])
    controller.publish('balanced_data_cursor_reanchored')
    return str(path)
