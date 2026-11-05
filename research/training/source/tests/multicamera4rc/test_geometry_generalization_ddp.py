"""Two real Gloo ranks train every generalization option together under BF16."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import h5py
import pytest
import torch
from torch import distributed as dist

from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.schedule import LearningRatePlan
from scripts.build_monocular_generalization_config import build_config
from test_geometry_generalization_integration import command, native_config


_WORKER = r'''
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from torch import distributed as dist
from multicamera4rc import MultiCamera4RC, train
from multicamera4rc.datasets import augmentation

records = dict(writes=[], geometry_predictions=[], gradient_steps=[], photo=[], objective_metrics=[], used_rgb=[])
original_ddp = train.DistributedDataParallel
def ddp(*args, **kwargs):
    value = original_ddp(*args, **kwargs)
    records['model'] = value.module.model
    return value
train.DistributedDataParallel = ddp

original_encode = MultiCamera4RC.encode
def encode(self, *args, **kwargs):
    value = original_encode(self, *args, **kwargs)
    records['geometry_predictions'].append(dict(
        finite=all(torch.isfinite(v).all().item() for v in value.geometry.values() if torch.is_tensor(v)),
        autocast=torch.is_autocast_enabled('cpu'), dtype=str(torch.get_autocast_dtype('cpu'))))
    return value
MultiCamera4RC.encode = encode

original_forward = train._TrainingStep.forward
def forward(self, *args, **kwargs):
    records['used_rgb'].append(hashlib.sha256(args[0].detach().contiguous().cpu().numpy().tobytes()).hexdigest())
    loss, metrics = original_forward(self, *args, **kwargs)
    records['objective_metrics'].append({key:float(value) for key,value in metrics.items()
                                          if key.startswith('depth_objective_')})
    return loss, metrics
train._TrainingStep.forward = forward

original_step = train._optimizer_step
def optimizer_step(loss, model, optimizer, scaler, clip_grad):
    result = original_step(loss, model, optimizer, scaler, clip_grad)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    geometry = next(group for group in optimizer.param_groups if group['name'] == 'geometry')
    records['gradient_steps'].append(dict(
        finite=all(torch.isfinite(value).all().item() for value in gradients),
        tensors=len(gradients), norm=float(result[0]), skipped=bool(result[1]),
        geometry_gradient=any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                              for parameter in geometry['params'])))
    return result
train._optimizer_step = optimizer_step

original_photo = augmentation.photometric_sample
def photo(*args, **kwargs):
    sample = original_photo(*args, **kwargs)
    images = sample['images'].detach().contiguous().cpu()
    records['photo'].append(dict(info=sample['photometric_augmentation'],
        images_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest()))
    return sample
augmentation.photometric_sample = photo

original_save = train.save_checkpoint
def save(path, *args, **kwargs):
    records['writes'].append(Path(path).name)
    return original_save(path, *args, **kwargs)
train.save_checkpoint = save

checkpoint = train.main(sys.argv[1:])
model = records.pop('model')
digest = hashlib.sha256()
for name, value in sorted(model.state_dict().items()):
    digest.update(name.encode())
    digest.update(value.detach().contiguous().cpu().numpy().tobytes())
records.update(digest=digest.hexdigest(),
    model_finite=all(torch.isfinite(value).all().item() for value in model.state_dict().values()),
    group_initialized_after_main=dist.is_initialized(), rank=int(os.environ['RANK']))
Path(checkpoint).parent.joinpath('rank_'+str(records['rank'])+'.json').write_text(json.dumps(records,allow_nan=False))
'''


def _torchrun(tmp_path, config, output, steps, *, resume=None, loss_config=None):
    worker = tmp_path/'generalization_ddp_worker.py'
    worker.write_text(textwrap.dedent(_WORKER))
    argv = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1',
        '--rdzv-backend=c10d', '--rdzv-endpoint=127.0.0.1:0', '--local-addr=127.0.0.1',
        '--nproc_per_node=2', str(worker), *command(config, output, steps),
        '--amp', '--amp-dtype', 'bfloat16', '--save-every', '100']
    if resume is None:
        argv += ['--geometry-lr-multiplier', '2', '--geometry-lr-warmup-steps', '4',
                 '--lr-schedule', 'cosine', '--lr-end-step', '20', '--loss-config', str(loss_config)]
    else:
        # This intentionally carries no geometry/loss/LR migration flags.
        argv += ['--resume', str(resume)]
    env = os.environ.copy()
    for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(name, None)
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
    repo = Path(__file__).resolve().parents[2]
    env['PYTHONPATH'] = str(repo)+os.pathsep+env.get('PYTHONPATH', '')
    result = subprocess.run(argv, cwd=repo, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    logs = []
    for line in result.stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and 'step' in row:
            logs.append(row)
    records = [json.loads((output/f'rank_{rank}.json').read_text()) for rank in range(2)]
    return records, logs


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='Gloo is unavailable')
def test_all_options_two_rank_bf16_save_resume_preserves_models_and_geometry_anchor(tmp_path):
    base = native_config(tmp_path)
    original = json.loads(base.read_text())
    # Make every GT-defined depth bucket nonempty so all newly registered
    # diagnostics pass through the real distributed reducer, not just loss.
    with h5py.File(original['sources'][0]['root'], 'r+') as handle:
        depths = handle['train/scene/depths']
        pattern = depths[0].copy()
        pattern[:, :14], pattern[:, 14:28], pattern[:, 28:] = .25, 2., 8.
        for index in range(len(depths)):
            depths[index] = pattern*(1+.01*index)
    candidate, _ = build_config(base, return_receipt=True)
    candidate['photometric_augmentation'].update(probability=1., grayscale_probability=1., blur_probability=1.)
    config = tmp_path/'all-data.json'
    config.write_text(json.dumps(candidate))
    loss = MultiCamera4RCLoss.dense().to_config()
    loss.update(depth_gradient_scales=[1, 2, 4], depth_gradient_confidence=False,
        depth_objective_options=dict(relative_weight=.05, log_bias_weight=.02,
                                     temporal_scale_weight=.01, diagnostics=True))
    loss_path = tmp_path/'all-loss.json'
    loss_path.write_text(json.dumps(loss))
    first_output, resumed_output, direct_output = (tmp_path/name for name in ('first', 'resumed', 'direct'))
    first, first_logs = _torchrun(tmp_path, config, first_output, 2, loss_config=loss_path)
    saved = torch.load(first_output/'latest.pt', weights_only=True)
    resumed, resumed_logs = _torchrun(tmp_path, config, resumed_output, 3, resume=first_output/'latest.pt')
    restored = torch.load(resumed_output/'latest.pt', weights_only=True)
    direct, direct_logs = _torchrun(tmp_path, config, direct_output, 3, loss_config=loss_path)
    assert [row['step'] for row in first_logs] == [1, 2]
    assert [row['step'] for row in resumed_logs] == [3]
    assert [row['step'] for row in direct_logs] == [1, 2, 3]
    all_logs = first_logs+resumed_logs+direct_logs
    required = {'depth_objective_relative', 'depth_objective_log_bias', 'depth_objective_temporal_scale',
                'depth_objective_observations', 'depth_objective_temporal_pairs',
                'depth_objective_near_pixels', 'depth_objective_mid_pixels', 'depth_objective_far_pixels',
                'depth_objective_near_relative_error', 'depth_objective_mid_relative_error',
                'depth_objective_far_relative_error'}
    for row in all_logs:
        assert row['world_size'] == 2 and row['effective_batch_size'] == 2
        assert row['amp_dtype'] == 'bfloat16' and not row['optimizer_step_skipped']
        assert len(row['rank_batches']) == 2 and required <= row.keys()
        assert all(math.isfinite(float(row[name])) for name in required | {'loss', 'grad_norm'})
        assert row['depth_objective_temporal_pairs'] > 0
        for bucket in ('near', 'mid', 'far'):
            assert row[f'depth_objective_{bucket}_pixels'] > 0
    for output, records in ((first_output, first), (resumed_output, resumed), (direct_output, direct)):
        assert len({row['digest'] for row in records}) == 1
        assert records[0]['writes'] == ['latest.pt'] and records[1]['writes'] == []
        assert [path.name for path in output.glob('*.pt')] == ['latest.pt']
        for row in records:
            assert row['model_finite'] and not row['group_initialized_after_main']
            assert row['geometry_predictions'] and all(item['finite'] and item['autocast']
                and item['dtype'] == 'torch.bfloat16' for item in row['geometry_predictions'])
            assert all(item['finite'] and item['tensors'] > 0 and item['geometry_gradient']
                and math.isfinite(item['norm']) and not item['skipped'] for item in row['gradient_steps'])
            assert all(item['info']['applied'] and item['info']['grayscale'] and item['info']['blur']
                       for item in row['photo'])
            assert all(required <= item.keys() for item in row['objective_metrics'])
    state, restored_state = saved['training_state'], restored['training_state']
    assert state['amp_dtype'] == restored_state['amp_dtype'] == 'bfloat16'
    assert state['world_size'] == restored_state['world_size'] == 2
    assert state['optimizer_updates'] == 2 and restored_state['optimizer_updates'] == 3
    assert len(saved['rank_rng']) == len(restored['rank_rng']) == 2
    assert state['lr_plan'] == restored_state['lr_plan']
    assert state['geometry_lr_options'] == restored_state['geometry_lr_options']
    assert state['photometric_augmentation'] == restored_state['photometric_augmentation']
    assert state['loss_config'] == restored_state['loss_config']
    policy = restored_state['lr_plan']['group_multipliers']['geometry']
    assert policy == dict(start_step=0, multiplier=2., warmup_steps=4)
    plan = LearningRatePlan(**restored_state['lr_plan'])
    group = plan.group_names.index('geometry')
    assert resumed_logs[0]['geometry_lr'] == pytest.approx(
        plan.base_lrs[group]*plan.factor(2)*plan.group_factor('geometry', 2))
    # Exact direct-vs-resumed equality exercises rank RNG, Adam moments, photo
    # sample seeds and the absolute geometry warmup anchor at the same time.
    assert {row['digest'] for row in resumed+direct} == {direct[0]['digest']}
    for rank in range(2):
        assert first[rank]['used_rgb']+resumed[rank]['used_rgb'] == direct[rank]['used_rgb']
        # The ordinary loader may prepare one next batch before the loop sees
        # the target step. Compare only the batches consumed by real forwards.
        assert (first[rank]['photo'][:len(first[rank]['used_rgb'])]
                +resumed[rank]['photo'][:len(resumed[rank]['used_rgb'])]
                == direct[rank]['photo'][:len(direct[rank]['used_rgb'])])
    direct_checkpoint = torch.load(direct_output/'latest.pt', weights_only=True)
    assert restored['training_state'] == direct_checkpoint['training_state']
    assert restored['optimizer']['param_groups'] == direct_checkpoint['optimizer']['param_groups']
    for key, values in restored['optimizer']['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(value, direct_checkpoint['optimizer']['state'][key][name], rtol=0, atol=0)
