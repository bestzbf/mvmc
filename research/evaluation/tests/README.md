# Evaluation regression tests

These tests cover the full runner, manifest preparation, TAP/MV/World scoring, model adapters, MoSE3 observation, Ours DriveTrack lock routing and recovery, and full-result reporting. `test_paths.json` records the source fingerprints and the paths each file occupies in a temporary deployment. The test runner applies the same source bindings as the research runtime.

From the repository root, install the project test and visualization extras and run the CPU suite.

```sh
python -m pip install -e '.[test,viz]'
python research/evaluation/run_tests.py
```

The cases that use original author preprocessing require the relevant pinned source checkout from [the dependency manifest](../third_party_versions.json). The runner checks the revision before importing it. A full run uses compatible PyTorch and torchvision installations, plus `imageio` and `tqdm` for the DA3 processor. Initialize the DA3 submodule at the revision recorded by the ReLift checkout.

```sh
git -C /work/upstream/ReLift submodule update --init third_party/Depth-Anything-3
python research/evaluation/run_tests.py \
  --checkout arrow=/work/upstream/arrow \
  --checkout mose3=/work/upstream/MoSE3 \
  --checkout point4d=/work/upstream/Point4D \
  --checkout relift=/work/upstream/ReLift \
  --checkout uniquery4r=/work/upstream/UniQuery4R
```

When an optional checkout is absent, its native-source cases are reported as skipped. The other contract tests still run. Use `-- -k FILTER` to pass a selection to pytest. The fixtures use synthetic data and CPU execution. Native GPU runs and comparisons against historical scores require the corresponding deployment assets.
