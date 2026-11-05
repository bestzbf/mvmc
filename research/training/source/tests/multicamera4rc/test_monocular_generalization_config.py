"""Training exposure candidates preserve source, GT and held-out boundaries."""
from copy import deepcopy
import json

import pytest

from multicamera4rc.datasets.mixture import read_data_config
from scripts.build_monocular_generalization_config import build_config, main, write_config
from test_native_loaders import _write_mugen


def base(tmp_path):
    return dict(image_mode='native', supervision='dense', sampling='balanced_coverage',
        num_queries=0, num_times=4, epoch_size=1000, seed=49,
        augmentation=dict(probability=.35, min_crop_fraction=.75, max_crop_fraction=1.),
        sources=[
            dict(name='variable', adapter='kubric_multicam', root=str(tmp_path/'kubric'),
                split='train', cameras=4, num_times=8, weight=.5, dense_kind='kubric',
                sequences=['already_admitted'], layouts=[dict(cameras=1, num_times=8, weight=.2),
                    dict(cameras=2, num_times=6, weight=.3), dict(cameras=4, num_times=4, weight=.5)]),
            dict(name='mono', adapter='mugen_h5', root=str(tmp_path/'mono.h5'), split='train',
                cameras=1, num_times=4, weight=.2, dense_kind='geometry',
                options=dict(dataset_name='pointodyssey', generate_static_tracks=False)),
            dict(name='stereo', adapter='vkitti2_stereo', root=str(tmp_path/'stereo'),
                split='train', cameras=2, num_times=4, weight=.2, dense_kind='vkitti_scene_flow'),
            dict(name='calibration', adapter='panoptic', root=str(tmp_path/'panoptic'),
                split='train', cameras=4, num_times=4, weight=.1, dense_kind='camera_only')])


def test_mass_non_c1_ratios_and_every_gt_contract_are_preserved(tmp_path):
    original = base(tmp_path)
    untouched = deepcopy(original)
    result, receipt = build_config(original, return_receipt=True)
    assert original == untouched
    assert sum(source['weight'] for source in result['sources']) == pytest.approx(1.)
    assert result['sources'][2]['weight']/result['sources'][3]['weight'] == pytest.approx(.2/.1)
    layouts = result['sources'][0]['layouts']
    assert sum(row['weight'] for row in layouts) == pytest.approx(1.)
    assert layouts[1]['weight']/layouts[2]['weight'] == pytest.approx(.3/.5)
    assert layouts[0]['weight']/layouts[1]['weight'] == pytest.approx(.2*1.25/.3)
    # Compare actual configured source*layout masses across mixed-C and pure
    # sources. All non-C1 atoms retain their global relative probabilities.
    scale = result['sources'][2]['weight']/original['sources'][2]['weight']
    for index, layout in enumerate(layouts):
        old_mass = original['sources'][0]['weight']*original['sources'][0]['layouts'][index]['weight']
        new_mass = result['sources'][0]['weight']*layout['weight']
        assert new_mass/old_mass == pytest.approx(scale*(1.25 if layout['cameras'] == 1 else 1.))
    assert result['sources'][1]['weight']/original['sources'][1]['weight'] == pytest.approx(scale*1.25)
    for old, new in zip(original['sources'], result['sources']):
        old = deepcopy(old); new = deepcopy(new)
        old.pop('weight'); new.pop('weight')
        if old.get('layouts'):
            for rows in (old['layouts'], new['layouts']):
                for row in rows:
                    row.pop('weight')
        assert old == new
    assert result['augmentation'] == original['augmentation']
    assert receipt['time_layouts_unchanged'] and receipt['camera_time_layouts_added'] == 0
    assert receipt['sources_added'] == 0 and receipt['motion_gt_recipes_unchanged']
    assert receipt['configured_reliable_c1_clip_mass_after'] > receipt['configured_reliable_c1_clip_mass_before']
    assert not receipt['training_started'] and not receipt['benchmark_sources_added']


@pytest.mark.parametrize('split', ['val', 'valid', 'test', 'all'])
def test_validation_test_or_all_splits_cannot_be_reinterpreted_as_training(tmp_path, split):
    config = base(tmp_path)
    config['sources'][1]['split'] = split
    with pytest.raises(ValueError, match='training splits'):
        build_config(config)


def test_sintel_and_unsupported_c1_geometry_are_never_added(tmp_path):
    config = base(tmp_path)
    config['sources'][1]['options']['dataset_name'] = 'mpi_sintel'
    with pytest.raises(ValueError, match='Sintel'):
        build_config(config)
    config = base(tmp_path)
    config['sources'] = [config['sources'][3]]
    config['sources'][0]['cameras'] = 1
    with pytest.raises(ValueError, match='no existing reliable C1'):
        build_config(config)
    for value in (.5, 2., float('inf'), True):
        with pytest.raises(ValueError):
            build_config(base(tmp_path), c1_multiplier=value)


def test_default_never_adds_t48_and_unproved_long_times_fail(tmp_path):
    config = base(tmp_path)
    result = build_config(config)
    assert max(row['num_times'] for row in result['sources'][0]['layouts']) == 8
    with pytest.raises(ValueError, match='manifest'):
        build_config(config, max_monocular_times=48)


def test_explicit_long_layout_uses_actual_admitted_frame_pool_and_adapter(tmp_path):
    path = tmp_path/'motion.h5'
    _write_mugen(path)
    manifest = tmp_path/'admitted.json'
    manifest.write_text(json.dumps(dict(root=str(path), sequences={'moving':dict(valid_frames=list(range(11)))})))
    config = base(tmp_path)
    config['sources'] = [dict(name='existing_mono', adapter='mugen_h5', root=str(path),
        split='train', cameras=1, num_times=3, weight=1., dense_kind='geometry',
        manifest=str(manifest), sequences=['moving'], strides=[1],
        options=dict(dataset_name='pointodyssey', verbose=False, train_only_val_ratio=0.))]
    result, receipt = build_config(config, max_monocular_times=8, return_receipt=True)
    assert result['sources'][0]['layouts'] == [dict(cameras=1, num_times=3, weight=.8),
                                             dict(cameras=1, num_times=8, weight=.2)]
    assert receipt['long_layout_preflights'][0]['adapter_index_preflight']
    assert receipt['long_layout_preflights'][0]['min_admitted_frames'] == 11
    assert not receipt['time_layouts_unchanged']
    with pytest.raises(ValueError, match='cannot prove'):
        build_config(config, max_monocular_times=48)
    payload = json.loads(manifest.read_text())
    payload['sequences']['moving']['valid_frames'][-1] = 100
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='frame exceeds'):
        build_config(config, max_monocular_times=8)


def test_exclusive_output_resolves_paths_and_cli_records_no_time_change(tmp_path, capsys):
    config = base(tmp_path)
    config['sources'][1]['root'] = 'relative.h5'
    file = tmp_path/'base.json'
    file.write_text(json.dumps(config))
    saved = file.read_bytes()
    output = tmp_path/'candidate.json'
    main(['--base-config', str(file), '--output', str(output)])
    receipt = json.loads(capsys.readouterr().out)
    assert receipt['training_started'] is False and receipt['time_layouts_unchanged'] is True
    assert read_data_config(output)['sources'][1]['root'] == str(tmp_path/'relative.h5')
    assert file.read_bytes() == saved
    with pytest.raises(FileExistsError):
        write_config(file, output)
    with pytest.raises(ValueError, match='outside'):
        write_config(file, file)
    with pytest.raises(ValueError, match='outside'):
        write_config(file, tmp_path/'kubric'/'forbidden.json')
