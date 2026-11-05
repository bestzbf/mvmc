# Current six-rank training source

The running ranks use the `069b1c28ce4763e2ac6a4f06dbc5b73c9ed3dc55` checkout as their working directory. The private bootstrap prepends the deployed `future_trainer_code_v22/source` tree to `sys.path`. That tree supplies the effective Python source. Compared with the clean checkout, the deployment replaces `multicamera4rc/train.py` and adds six files for the execution helpers, execution controller and six-rank gate.

The export verified all 1,076 files pinned by the original deployment manifest. `source/` contains 288 effective source and configuration files, including the runtime package, adapter code, transitively referenced controller scripts and vendor license notices. `base_overrides/multicamera4rc/train.py` holds the unmodified baseline training module for native checkpoint validation. `provenance.json` records the original file hashes and role-relative source paths, the seven-file overlay, and the difference between the complete deployment and this filtered export.

## Launch chain and dependencies

The active entry point is `supervisor/supervisor_resume_v23.py`. It imports `control_common.py`, `native_update_audit_v23.py` and `scripts.async_training_state` from the baseline checkout. It launches the private Python wrapper, which loads `future_private_bootstrap.py` and runs `multicamera4rc.launch`. The launcher calls `torch.distributed.run`, which starts six `multicamera4rc.train` ranks through the same wrapper. The bootstrap adds the explicit `fa4_metadata_prepared_kv` execution mode. The supervisor checks the per-rank proof with `multicamera4rc/six_rank_gate_v22.py`.

The production supervisor command is `python -u -B supervisor_resume_v23.py serve --lease-self --plan "$TRAINING_PLAN" --plan-sha256 "$TRAINING_PLAN_SHA256"` with the original pinned plan and exclusive leases. Use `check_plan` in place of `serve` and omit `--lease-self` for read-only plan verification.

The active resume launches on physical GPUs `0,1,2,5,6,7`, from step `140501` toward `200000`, saving every 100 steps. It uses the `giant` preset, `bfloat16`, batch size and gradient accumulation of 1, four sources, at most 64 source-time pairs, two workers, four CPU threads, seed 49, and `target_time` decoder memory. The launcher uses `--prefer-motion-sources --no-query-chunking --no-reuse-decoder-kv --no-reuse-shared-computation`. The 28-source recipe comes from `optional_long.json` and the loss recipe from `latest_merged_loss.json`.

The supervisor loads `evaluate_joint_checkpoint_v22.py` dynamically for training-side checkpoint validation on physical GPUs `3,4`. This module uses `scripts.independent_validation`, `multicamera4rc/schedule.py`, the native Git checkout, and external fixed validation plans. Its template captures the relevant configuration and bindings. Runtime proof receipts, checkpoints, data mounts, data manifests, masks, caches, and validation outputs are supplied separately.

Python is `3.10.20`. The bootstrap binds PyTorch `2.10.0+cu128` and CUDA `12.8` from the shared environment, `flash_attn_4` `4.0.0b37` from the private environment, and OpenEXR/Imath from the data environment. `runtime_packages.json` records the corresponding installed distribution metadata. `source/pyproject.toml` lists the source package's dependencies.

## Materialize the runtime bindings

Use the repository's `research/materialize.py` with the template directory, a filled copy of its `bindings.example.json`, and a fresh destination. The destination must be outside the source snapshot. Use absolute ASCII paths without spaces, quotes or shell metacharacters for the bindings. If the checkout path falls outside that format, copy `source/` to a suitable runtime path. The `binding_locations.json` files identify the fields served by each binding.

1. Choose the effective source tree. For the original private environment path, use `source/` and keep the example value for `TRAINING_EXECUTION_SOURCE_SHA256`. To use a different private environment path, copy `source/` to a fresh directory, render `templates/execution_overlay/` with the resolved absolute `PRIVATE_ENV`, and place the generated `multicamera4rc/training_execution_future.py` at the same relative path in that copy. The template changes the expected environment path in the existing guard. Keep the audited `source/` copy intact.
2. Render `templates/runtime/` to an isolated runtime directory using the same resolved `PRIVATE_ENV`. Bind the shared and data sites, the rendered bootstrap, and the planned deployment JSON. Before rendering `templates/inputs/`, calculate the SHA-256 of the bootstrap, wrapper and selected `multicamera4rc/training_execution_future.py`. Set `RUNTIME_BOOTSTRAP_SHA256`, `RUNTIME_WRAPPER_SHA256` and `TRAINING_EXECUTION_SOURCE_SHA256` to those hashes. Set `SOURCE_ROOT` to the selected source tree and supply the GPU UUIDs and dataset bindings. Verify that `deployment.json` pins the selected controller's exact bytes before running the CPU preflight below.
3. For checkpoint validation and supervised launch, render `templates/supervisor_code/`, `templates/checkpoint_validation_config/`, `templates/activation_config/` and `templates/supervisor_plan/` in that order. Supply the SHA-256 values of the files produced by each preceding stage and the trusted external proof records. Run the control scripts from a directory that also exposes the rendered `control_common.py` to Python.

The live checkpoint and evaluator require the original deployment, input hashes, and Git identity. The filtered export has a new deployment digest, and the optional controller template changes the execution implementation digest. An exact production resume requires the original bound deployment and artifacts. The production controller also requires its pinned publisher and gate records before it can adopt a checkpoint. Historical repository configuration files retain their original defaults. Use the bound `optional_long.json` and the exported loss recipe for the active configuration.

## CPU validation

The base training regression suite and the activation, checkpoint and evaluator contract tests are described in [the training test guide](tests/activation/README.md).

Run the deployment's existing four CPU lifecycle tests with a Python environment that provides PyTorch. From this directory:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/source/multicamera4rc:$PWD/source" \
  python -B tests/test_training_execution_future.py
```

For a materialized private runtime, the existing bootstrap also provides a CPU preflight. Set `TRAINING_WRAPPER`, `TRAINING_DEPLOYMENT` and `TRAINING_EXR` to the executable wrapper, its deployment JSON and an external sample EXR file. This checks the pinned paths, the OpenEXR decode, the runtime import origins and checkpoint metadata serialization with CUDA hidden.

```bash
deployment_sha=$(sha256sum "$TRAINING_DEPLOYMENT" | cut -d ' ' -f 1)
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  DFRT_TRAIN_EXECUTION=fa4_metadata_prepared_kv \
  DFRT_FUTURE_DEPLOYMENT_SHA256="$deployment_sha" \
  "$TRAINING_WRAPPER" --preflight --exr "$TRAINING_EXR"
```
