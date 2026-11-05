"""CPU preparation preserves checkpoints and emits a single explicit migration."""
from copy import deepcopy
import errno
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

from multicamera4rc import MultiCamera4RC, MultiCamera4RCConfig
from multicamera4rc.checkpoint import save_checkpoint
from multicamera4rc.launch import build_parser
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.optim import optimizer_groups
from scripts.prepare_geometry_generalization import main, prepare


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(tmp_path, *, world=2):
    dataset = tmp_path/'training_assets'
    dataset.mkdir()
    manifest = tmp_path/'admitted.json'
    manifest.write_text(json.dumps(dict(root=str(dataset), sequences={'existing': dict(valid_frames=list(range(8)))})))
    config = dict(image_mode='native', supervision='dense', sampling='balanced_coverage',
        epoch_size=20, num_times=4, num_queries=0, seed=49,
        invalid_log='original_invalid.jsonl', kubric_background_policy='objects_ground',
        sources=[dict(name='existing_geometry', adapter='mugen_h5', root='training_assets',
                      manifest='admitted.json', split='train', cameras=1, num_times=4,
                      weight=1., dense_kind='geometry', options=dict(dataset_name='scannet'))])
    base = tmp_path/'base.json'
    base.write_text(json.dumps(config))
    model = MultiCamera4RC(MultiCamera4RCConfig.tiny(decoder_memory='target_time'))
    optimizer = torch.optim.AdamW(optimizer_groups(model, 1e-5, 1e-4, 1e-4))
    for group in optimizer.param_groups:
        group['params'][0].grad = torch.ones_like(group['params'][0])
    optimizer.step()
    loss = MultiCamera4RCLoss.dense().to_config()
    loss['joint_geometry_options'] = dict(source_anchor_weight=.1)
    state = dict(world_size=world, batch_size=1, gradient_accumulation=2,
        source_count=4, max_source_time_pairs=64, prefer_motion_sources=True,
        amp_dtype='bfloat16', loss_config=loss, seed=49,
        data_config_sha256=sha(base),
        lr_plan=dict(mode='cosine', start_step=0, end_step=200000,
                     base_lrs=[1e-5, 1e-4]), optimizer_updates=10)
    checkpoint = tmp_path/'step_00000010.pt'
    save_checkpoint(checkpoint, model, optimizer, step=10, training_state=state)
    return checkpoint, base, state


def test_combined_preparation_preserves_checkpoint_and_full_legacy_contract(tmp_path):
    checkpoint, base, state = inputs(tmp_path)
    before_cp, before_base = sha(checkpoint), sha(base)
    output = tmp_path/'candidate'
    receipt = prepare(checkpoint, base, output, '0,5', python=sys.executable)
    assert sha(checkpoint) == before_cp and sha(base) == before_base
    assert receipt['checkpoint']['pinned']
    assert Path(receipt['checkpoint']['selected']).stat().st_ino == checkpoint.stat().st_ino
    assert receipt['inherited_lr_plan'] == state['lr_plan']
    loss = json.loads((output/'loss.json').read_text())
    expected = deepcopy(state['loss_config'])
    expected['depth_gradient_scales'] = [1, 2, 4]
    expected['depth_objective_options'] = dict(relative_weight=.05, log_bias_weight=.02,
        temporal_scale_weight=.01, diagnostics=True)
    assert loss == expected
    mixture = json.loads((output/'mixture.json').read_text())
    assert mixture['sources'][0]['root'] == str(tmp_path/'training_assets')
    assert mixture['sources'][0]['manifest'] == str(tmp_path/'admitted.json')
    assert mixture['sources'][0]['num_times'] == 4
    assert mixture['invalid_log'] == str(output/'training_invalid_samples.jsonl')
    assert mixture['photometric_augmentation']['probability'] == .8
    assert receipt['data_changes']['time_layouts_unchanged']
    metadata = next(row for row in receipt['external_assets'] if row['field'] == 'manifest')
    assert metadata['sha256'] == sha(tmp_path/'admitted.json')
    continuation = json.loads((output/'continuation.json').read_text())
    args = build_parser().parse_args(continuation['argv'][3:])
    assert args.resume == receipt['checkpoint']['selected']
    assert args.geometry_lr_multiplier == 2 and args.geometry_lr_warmup_steps == 100
    assert args.gradient_accumulation == 2 and args.batch_size == 1
    assert args.source_count == 4 and args.max_source_time_pairs == 64
    assert args.prefer_motion_sources and args.decoder_memory == 'target_time'
    assert args.migrate_data_config and not args.override_lr and args.lr_schedule is None
    assert not (output/'run').exists() and not receipt['training_started']
    assert not receipt['gpu_probe_executed'] and not receipt['quality_improvement_validated']
    resume = continuation['ordinary_resume_argv_template']
    assert '--migrate-data-config' not in resume
    assert '--geometry-lr-multiplier' not in resume
    assert '--geometry-lr-warmup-steps' not in resume
    assert sys.executable == continuation['argv'][0]  # Keep a venv's Python path.


def test_world_count_and_nonempty_output_fail_before_writing(tmp_path):
    checkpoint, base, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match='world_size'):
        prepare(checkpoint, base, tmp_path/'wrong_count', '0')
    assert not (tmp_path/'wrong_count').exists()
    output = tmp_path/'nonempty'
    output.mkdir()
    marker = output/'keep.txt'
    marker.write_text('untouched')
    with pytest.raises(ValueError, match='empty directory'):
        prepare(checkpoint, base, output, '0,1')
    assert marker.read_text() == 'untouched'


def test_cross_filesystem_never_silently_copies_large_checkpoint(tmp_path, monkeypatch):
    checkpoint, base, _ = inputs(tmp_path)
    def no_link(*args):
        raise OSError(errno.EXDEV, 'cross-device link')
    monkeypatch.setattr('scripts.prepare_geometry_generalization.os.link', no_link)
    output = tmp_path/'not_pinned'
    result = prepare(checkpoint, base, output, [0, 1])
    assert not result['checkpoint']['pinned']
    assert result['checkpoint']['selected'] == str(checkpoint)
    assert not (output/checkpoint.name).exists()
    copied = tmp_path/'explicit_copy'
    result2 = prepare(checkpoint, base, copied, [0, 1], copy_checkpoint=True)
    assert result2['checkpoint']['pin_mode'] == 'explicit_copy'
    assert sha(copied/checkpoint.name) == sha(checkpoint)
    assert (copied/checkpoint.name).stat().st_ino != checkpoint.stat().st_ino


def test_asset_missing_and_output_inside_dataset_are_rejected(tmp_path):
    checkpoint, base, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match='read-only dataset roots'):
        prepare(checkpoint, base, tmp_path/'training_assets/candidate', '0,1')
    (tmp_path/'admitted.json').unlink()
    with pytest.raises(FileNotFoundError, match='external training asset'):
        prepare(checkpoint, base, tmp_path/'missing_asset', '0,1')
    assert not (tmp_path/'missing_asset').exists()


def test_explicit_long_context_fails_without_real_adapter_index(tmp_path):
    checkpoint, base, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match='admitted frame pool'):
        prepare(checkpoint, base, tmp_path/'long', '0,1', max_monocular_times=48)
    assert not (tmp_path/'long').exists()


def test_cli_total_target_is_not_additional_steps(tmp_path, capsys):
    checkpoint, base, _ = inputs(tmp_path, world=1)
    capsys.readouterr()  # Discard the tiny model fixture's initialization message.
    output = tmp_path/'cli'
    main(['--checkpoint', str(checkpoint), '--base-data-config', str(base),
          '--output', str(output), '--gpus', '4', '--steps', '100'])
    result = json.loads(capsys.readouterr().out)
    assert result['checkpoint']['step'] == 10
    command = json.loads((output/'continuation.json').read_text())['argv']
    assert command[command.index('--steps')+1] == '100'
    with pytest.raises(ValueError, match='exceed'):
        prepare(checkpoint, base, tmp_path/'finished', [4], steps=10)


def test_no_optimizer_or_missing_saved_runtime_are_not_guessed(tmp_path):
    checkpoint, base, _ = inputs(tmp_path)
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    data['training_state'].pop('source_count')
    invalid = tmp_path/'missing_runtime.pt'
    torch.save(data, invalid)
    with pytest.raises(ValueError, match='source_count'):
        prepare(invalid, base, tmp_path/'bad_runtime', '0,1')
    data.pop('optimizer')
    torch.save(data, invalid)
    with pytest.raises(ValueError, match='model\\+optimizer'):
        prepare(invalid, base, tmp_path/'no_optimizer', '0,1')


def test_candidate_checkpoint_cannot_repeat_c1_boost_and_cursor_migration(tmp_path):
    checkpoint, base, _ = inputs(tmp_path)
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    data['training_state']['geometry_lr_options'] = dict(multiplier=2., start_step=10, warmup_steps=100)
    already_migrated = tmp_path/'candidate_step.pt'
    torch.save(data, already_migrated)
    with pytest.raises(ValueError, match='ordinary resume'):
        prepare(already_migrated, base, tmp_path/'repeat', '0,1')
    assert not (tmp_path/'repeat').exists()


def test_wrong_base_config_is_not_silently_substituted(tmp_path):
    checkpoint, base, _ = inputs(tmp_path)
    changed = json.loads(base.read_text())
    changed['seed'] += 1
    base.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='data_config_sha256'):
        prepare(checkpoint, base, tmp_path/'wrong_base', '0,1')
    assert not (tmp_path/'wrong_base').exists()
