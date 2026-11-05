"""Clip-shared appearance changes never alter calibrated geometry or GT."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from multicamera4rc.datasets.augmentation import (DEFAULT_PHOTOMETRIC, crop_sample,
    photometric_sample, principal_canvas_for_sample, validate_photometric_augmentation)
from multicamera4rc.datasets.mixture import collate_adapter_samples, create_dataset, read_data_config
from test_native_loaders import _write_mugen


def _sample(dtype=torch.float32):
    pixels = torch.linspace(0, 1, 3*32*42).reshape(1, 3, 32, 42).repeat(4, 1, 1, 1)
    pixels[..., 29:, :] = 0
    if dtype == torch.uint8:
        pixels = (pixels*255).round().to(dtype)
    return dict(clip_id='rgb', images=pixels, camera_id=torch.tensor([0, 0, 1, 1]),
        time_idx=torch.tensor([0, 1, 0, 1]), query_time_idx=torch.tensor([0, 1]),
        observation_valid=torch.ones(4, dtype=torch.bool), targets=dict(
            K=torch.eye(3).repeat(4, 1, 1), c2w=torch.eye(4).repeat(4, 1, 1),
            depth=torch.ones(4, 32, 42), depth_valid=torch.ones(4, 32, 42, dtype=torch.bool),
            flow=torch.randn(4, 32, 42, 2), mask=torch.zeros(4, 32, 42, dtype=torch.bool)))


@pytest.mark.parametrize('dtype', [torch.float32, torch.uint8])
def test_photometric_is_seeded_clip_shared_and_preserves_all_gt(dtype):
    sample = _sample(dtype)
    before = deepcopy(sample)
    options = dict(DEFAULT_PHOTOMETRIC, probability=1., blur_probability=1.)
    torch.manual_seed(10)
    state = torch.get_rng_state().clone()
    np.random.seed(13)
    numpy_state = np.random.get_state()
    first = photometric_sample(sample, options, 493, [[29, 42]]*4)
    torch.manual_seed(99)
    second = photometric_sample(sample, options, 493, [[29, 42]]*4)
    assert first['images'].shape == sample['images'].shape and first['images'].dtype == dtype
    torch.testing.assert_close(first['images'], second['images'], atol=0, rtol=0)
    assert not torch.equal(first['images'], sample['images'])
    for image in first['images'][1:]:
        torch.testing.assert_close(image, first['images'][0], atol=0, rtol=0)
    torch.testing.assert_close(first['images'][..., 29:, :], sample['images'][..., 29:, :], atol=0, rtol=0)
    assert first['targets'] is sample['targets']
    for key in sample['targets']:
        assert first['targets'][key] is sample['targets'][key]
        torch.testing.assert_close(sample['targets'][key], before['targets'][key], atol=0, rtol=0)
    # Neither the operation nor its local NumPy generator touches global RNGs.
    torch.set_rng_state(state)
    photometric_sample(sample, options, 493, [[29, 42]]*4)
    assert torch.equal(torch.get_rng_state(), state)
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert first['photometric_augmentation'] == second['photometric_augmentation']


def test_grayscale_blur_and_disabled_options_have_exact_contracts():
    sample = _sample()
    identity = dict(probability=1., brightness=0., contrast=0., saturation=0.,
                    grayscale_probability=0., blur_probability=0.)
    same = photometric_sample(sample, identity, 9)
    torch.testing.assert_close(same['images'], sample['images'], atol=0, rtol=0)
    assert photometric_sample(sample, None, 9) is sample
    skipped = photometric_sample(sample, dict(identity, probability=0.), 9)
    assert skipped['images'] is sample['images'] and not skipped['photometric_augmentation']['applied']
    gray = photometric_sample(sample, dict(identity, grayscale_probability=1.), 9)
    torch.testing.assert_close(gray['images'][:, 0], gray['images'][:, 1], atol=0, rtol=0)
    torch.testing.assert_close(gray['images'][:, 1], gray['images'][:, 2], atol=0, rtol=0)
    sample['images'].zero_()
    sample['images'][:, :, 14, 18] = 1.
    blurred = photometric_sample(sample, dict(identity, blur_probability=1., blur_sigma=[1., 1.]), 9)
    assert 0 < blurred['images'][0, 0, 14, 18] < 1
    assert blurred['images'][0, 0, 13, 18] > 0
    torch.testing.assert_close(blurred['images'].sum(), sample['images'].sum(), atol=1e-5, rtol=0)


def test_crop_principal_metadata_and_invalid_slots_are_unchanged():
    sample = _sample()
    cropped = crop_sample(sample, [[1, 2, 28, 28]]*4)
    cropped['observation_valid'][-1] = False
    before = principal_canvas_for_sample(cropped).clone()
    result = photometric_sample(cropped, dict(DEFAULT_PHOTOMETRIC, probability=1.,
        blur_probability=1.), 44, [[28, 28]]*4)
    assert result['augmentation'] is cropped['augmentation']
    torch.testing.assert_close(principal_canvas_for_sample(result), before, atol=0, rtol=0)
    torch.testing.assert_close(result['images'][-1], cropped['images'][-1], atol=0, rtol=0)
    batch = collate_adapter_samples([result])
    assert batch['photometric_augmentation'] == [result['photometric_augmentation']]


@pytest.mark.parametrize('options', [dict(unknown=1), dict(probability=True), dict(contrast=float('nan')),
    dict(brightness=-.1), dict(blur_sigma=[1., .1]), dict(blur_sigma=[0., 1.]),
    dict(blur_kernel_size=4), dict(blur_kernel_size=True), dict(blur_kernel_size=33)])
def test_invalid_appearance_contracts_are_rejected(options):
    with pytest.raises(ValueError):
        validate_photometric_augmentation(options)


def test_native_dataset_reconstructs_same_sample_and_saves_normalized_contract(tmp_path):
    path = tmp_path/'data.h5'
    _write_mugen(path)
    config = dict(image_mode='native', supervision='dense', sampling='balanced_coverage',
        num_times=3, num_queries=0, epoch_size=4, seed=49,
        sources=[dict(adapter='mugen_h5', root=str(path), split='train', cameras=1,
            dense_kind='geometry', options=dict(dataset_name='pointodyssey', verbose=False,
                                                train_only_val_ratio=0.))],
        photometric_augmentation=dict(probability=1., grayscale_probability=1.))
    file = tmp_path/'train.json'
    file.write_text(json.dumps(config))
    first = create_dataset(file)
    first.set_epoch(2)
    saved = read_data_config(file)
    resumed = create_dataset(saved)
    resumed.set_epoch(2)
    a, b = first[1], resumed[1]
    assert first.photometric_augmentation == validate_photometric_augmentation(config['photometric_augmentation'])
    assert a['photometric_augmentation'] == b['photometric_augmentation']
    torch.testing.assert_close(a['images'], b['images'], atol=0, rtol=0)
    for key in a['targets']:
        torch.testing.assert_close(a['targets'][key], b['targets'][key], atol=0, rtol=0)
    validation = deepcopy(config)
    validation['sources'][0]['split'] = 'val'
    with pytest.raises(ValueError, match='training splits'):
        create_dataset(validation)
