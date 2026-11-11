"""Project renaming must preserve predictions and reject unsupported models."""

import pytest
import torch

from mvmc import MVMC, MVMCConfig
from multid4rt import MultiD4RT
from multid4rt.checkpoint import config_from_checkpoint, save_checkpoint


@pytest.mark.parametrize("scope", ["target_time", "all_observations"])
def test_mvmc_loads_legacy_checkpoint_without_changing_predictions(tmp_path, scope):
    torch.set_num_threads(1)
    model = MVMC(MVMCConfig.tiny(decoder_memory=scope, gradient_checkpointing=False)).eval()
    checkpoint = tmp_path / "legacy.pt"
    save_checkpoint(checkpoint, model)
    renamed = MVMC.from_checkpoint(checkpoint).eval()
    legacy = MultiD4RT.from_checkpoint(checkpoint).eval()
    images = torch.rand(1, 4, 3, 28, 28)
    cameras = torch.tensor([[0, 0, 1, 1]])
    times = torch.tensor([[0, 5, 0, 5]])
    with torch.inference_mode():
        expected = model(images, cameras, times, source_obs_idx=[0])
        for restored in (renamed, legacy):
            assert restored.config.decoder_memory == scope
            actual = restored(images, cameras, times, source_obs_idx=[0])
            for key in ("trajectory", "displacement", "visibility", "depth", "K", "c2w"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_refiner_config_is_rejected_with_research_route():
    config = MVMCConfig.tiny().to_dict()
    config["refinement_dim"] = 128
    with pytest.raises(ValueError, match="TrajectoryRefiner.*research/training/source"):
        config_from_checkpoint({"config": config})


@pytest.mark.parametrize("strict", [False, True])
def test_refiner_weights_cannot_be_silently_dropped_even_with_config_override(strict):
    config = MVMCConfig.tiny()
    model = MVMC(config)
    state = {"module." + key: value for key, value in model.state_dict().items()}
    state["module.trajectory_refiner.delta_head.weight"] = torch.zeros(3, 128)
    checkpoint = {"architecture": "multicamera4rc", "model": state}
    with pytest.raises(ValueError, match="TrajectoryRefiner weights"):
        MVMC.from_checkpoint(checkpoint, config=config, strict_base=strict)
