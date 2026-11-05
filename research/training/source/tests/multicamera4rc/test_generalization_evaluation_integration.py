"""Real CPU evaluation keeps frozen scores separate from new diagnostics."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from multicamera4rc import MultiCamera4RC, MultiCamera4RCConfig
from multicamera4rc.checkpoint import save_checkpoint
from scripts.evaluate_monocular import main
from scripts.geometry_diagnostics import (depth_diagnostics, file_sha256,
                                         prefix_clip, verify_prediction)
from scripts.monocular_protocol import geometry_metrics, load_clip
from test_monocular_benchmark import make_scannet


@pytest.fixture
def evaluation_case(tmp_path):
    root = make_scannet(tmp_path, frames=8)
    checkpoint = tmp_path/'tiny.pt'
    torch.manual_seed(41)
    model = MultiCamera4RC(MultiCamera4RCConfig.tiny(gradient_checkpointing=False))
    save_checkpoint(checkpoint, model, step=126900)
    common = ['--dataset', 'scannet', '--roots', str(root),
        '--plan', str(tmp_path/'frozen-plan.json'), '--num-frames', '4',
        '--frame-stride', '2', '--eval-size', '28', '--long-side', '42',
        '--device', 'cpu', '--cpu-threads', '1']
    return dict(root=root, checkpoint=checkpoint, common=common, temporary=tmp_path)


def _run(case, output, *extra):
    args = case['common']+['--output', str(output)]
    audit = main(args+['--audit-only'])
    assert audit['passed'] and len(audit['cases']) == 1
    return main(args+['--checkpoint', str(case['checkpoint']), *extra])


def test_real_evaluation_archives_exact_arrays_and_diagnostics_preserve_frozen_scores(evaluation_case):
    case = evaluation_case
    plain = _run(case, case['temporary']/'plain', '--no-diagnostics', '--no-save-predictions')
    output = case['temporary']/'diagnostic'
    result = _run(case, output, '--context-lengths', '2', '4')
    assert result['stage'] == 'passed' and result['checkpoint_step'] == 126900
    assert result['mean_metrics'] == plain['mean_metrics']
    assert result['protocol'] == plain['protocol']
    row = result['records'][0]
    assert row['metrics'] == plain['records'][0]['metrics']
    assert row['frame_indices'] == [0, 2, 4, 6] and row['actual_times'] == 4
    assert row['peak_cuda_bytes'] is None
    archive = output/row['prediction_file']
    assert row['prediction_sha256'] == file_sha256(archive)
    verify_prediction(archive, row['prediction_sha256'], row['frame_indices'])
    with np.load(archive, allow_pickle=False) as arrays:
        assert set(arrays) == {'depth', 'K', 'c2w', 'frame_indices'}
        assert arrays['depth'].shape == (4, 28, 28)
        assert arrays['K'].shape == (4, 3, 3) and arrays['c2w'].shape == (4, 4, 4)
        np.testing.assert_array_equal(arrays['frame_indices'], [0, 2, 4, 6])
        prediction = tuple(torch.from_numpy(arrays[name].copy()) for name in ('depth', 'K', 'c2w'))
    clip = load_clip(dict(dataset='scannet', root=str(case['root']), scene=row['scene'],
                          frame_indices=row['frame_indices']), eval_size=28, long_side=42)
    original = tuple(tensor.clone() for tensor in prediction)
    metrics_before = geometry_metrics(*prediction, clip, dataset='scannet')
    diagnostic = depth_diagnostics(*prediction, clip, dataset='scannet')
    assert diagnostic['kind'] == 'gt_assisted_diagnostic_not_official_score'
    assert geometry_metrics(*prediction, clip, dataset='scannet') == metrics_before == row['metrics']
    for before, after in zip(original, prediction):
        torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_context_comparison_scores_same_prefix_and_reuses_full_context_forward(evaluation_case, monkeypatch):
    case = evaluation_case
    original_encode = MultiCamera4RC.encode
    calls = []

    def encode(self, images, camera_id, time_idx, *args, **kwargs):
        calls.append(dict(frames=images.shape[1], labels=time_idx.detach().cpu().tolist()))
        return original_encode(self, images, camera_id, time_idx, *args, **kwargs)

    monkeypatch.setattr(MultiCamera4RC, 'encode', encode)
    output = case['temporary']/'contexts'
    result = _run(case, output, '--context-lengths', '2', '4')
    assert calls == [dict(frames=4, labels=[[0, 2, 4, 6]]), dict(frames=2, labels=[[0, 2]])]
    row = result['records'][0]
    context = row['context_diagnostics']
    assert context['kind'] == 'matched_scoring_frames_not_official_score'
    assert [record['input_frames'] for record in context['records']] == [2, 4]
    for record in context['records']:
        assert record['scored_frames'] == 2 and record['frame_indices'] == [0, 2]
        assert record['metrics']['pose_num_pairs'] == 1
    with np.load(output/row['prediction_file'], allow_pickle=False) as arrays:
        full_prefix = tuple(torch.from_numpy(arrays[name][:2].copy()) for name in ('depth', 'K', 'c2w'))
    clip = load_clip(dict(dataset='scannet', root=str(case['root']), scene=row['scene'],
                          frame_indices=row['frame_indices']), eval_size=28, long_side=42)
    expected = geometry_metrics(*full_prefix, prefix_clip(clip, 2), dataset='scannet')
    assert context['records'][1]['metrics'] == expected
    # Once the exact complete row exists, resume verifies it without invoking
    # either the full-clip or additional context encoder forwards.
    calls.clear()

    def forbidden_encode(*args, **kwargs):
        raise AssertionError('resume must not execute another model forward')

    monkeypatch.setattr(MultiCamera4RC, 'encode', forbidden_encode)
    resumed = main(case['common']+['--output', str(output), '--checkpoint', str(case['checkpoint']),
        '--context-lengths', '2', '4', '--resume-results'])
    assert resumed['stage'] == 'passed' and resumed['records'] == result['records']
    assert resumed['mean_metrics'] == result['mean_metrics']
    assert not calls


@pytest.mark.parametrize('tamper', ['bytes', 'frame_identity'])
def test_resume_rejects_changed_archive_before_forward(evaluation_case, monkeypatch, tamper):
    case = evaluation_case
    output = case['temporary']/'tampered'
    result = _run(case, output, '--context-lengths', '2', '4')
    report_file = output/'evaluation.json'
    row = result['records'][0]
    archive = output/row['prediction_file']
    if tamper == 'bytes':
        archive.write_bytes(archive.read_bytes()+b'changed')
        expected = 'missing or changed'
    else:
        with np.load(archive, allow_pickle=False) as arrays:
            payload = {key: arrays[key].copy() for key in arrays}
        payload['frame_indices'] += 100
        np.savez_compressed(archive, **payload)
        # Even an updated byte receipt must not conceal a different frame set.
        result['records'][0]['prediction_sha256'] = file_sha256(archive)
        report_file.write_text(json.dumps(result))
        expected = 'frame identity changed'
    frozen_report = report_file.read_bytes()
    monkeypatch.setattr(MultiCamera4RC, 'encode', lambda *args, **kwargs:
        (_ for _ in ()).throw(AssertionError('tampered archive must be rejected before inference')))
    with pytest.raises(ValueError, match=expected):
        main(case['common']+['--output', str(output), '--checkpoint', str(case['checkpoint']),
            '--context-lengths', '2', '4', '--resume-results'])
    assert report_file.read_bytes() == frozen_report


def test_legacy_rows_without_prediction_arrays_cannot_masquerade_as_new_results(evaluation_case, monkeypatch):
    case = evaluation_case
    output = case['temporary']/'legacy'
    result = _run(case, output, '--context-lengths', '2', '4')
    legacy = deepcopy(result)
    for row in legacy['records']:
        (output/row.pop('prediction_file')).unlink()
        row.pop('prediction_sha256')
    # Retain the matching settings at top level: merely backfilling those
    # fields into an old metric-only JSON must not certify missing arrays.
    report = output/'evaluation.json'
    report.write_text(json.dumps(legacy))
    monkeypatch.setattr(MultiCamera4RC, 'encode', lambda *args, **kwargs:
        (_ for _ in ()).throw(AssertionError('old metrics must not bypass archive verification')))
    with pytest.raises(ValueError, match='no archived predictions'):
        main(case['common']+['--output', str(output), '--checkpoint', str(case['checkpoint']),
            '--context-lengths', '2', '4', '--resume-results'])


@pytest.mark.parametrize('tamper', ['official_metric', 'frozen_frames', 'missing_diagnostics', 'missing_context'])
def test_resume_rejects_edited_result_identity_or_metrics(evaluation_case, monkeypatch, tamper):
    case = evaluation_case
    output = case['temporary']/'edited-json'
    result = _run(case, output, '--context-lengths', '2', '4')
    row = result['records'][0]
    if tamper == 'official_metric':
        row['metrics']['depth_S_abs_rel'] += .1
        expected = 'metrics disagree'
    elif tamper == 'frozen_frames':
        row['frame_indices'] = list(reversed(row['frame_indices']))
        expected = 'differs from frozen plan'
    elif tamper == 'missing_diagnostics':
        row.pop('diagnostics')
        expected = 'missing requested diagnostics'
    else:
        row['context_diagnostics']['records'].pop()
        expected = 'missing matched-frame context diagnostics'
    report = output/'evaluation.json'
    report.write_text(json.dumps(result))
    before = report.read_bytes()
    monkeypatch.setattr(MultiCamera4RC, 'encode', lambda *args, **kwargs:
        (_ for _ in ()).throw(AssertionError('edited results must be rejected without model forward')))
    with pytest.raises(ValueError, match=expected):
        main(case['common']+['--output', str(output), '--checkpoint', str(case['checkpoint']),
            '--context-lengths', '2', '4', '--resume-results'])
    assert report.read_bytes() == before
