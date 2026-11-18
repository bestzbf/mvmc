# Training source and historical six-rank controller snapshot

For a new experiment, start with the [training source guide](source/README.md) and the [repository map](../../docs/project-guide.md). To load a trained research checkpoint for inference, see the [research inference guide](../../docs/research-inference.md). The setup below describes a historical deployment, not a live training status.

At the original export, the running ranks used the `069b1c28ce4763e2ac6a4f06dbc5b73c9ed3dc55` checkout as their working directory. That private bootstrap prepended the deployed `future_trainer_code_v22/source` tree to `sys.path`. That tree supplies the effective Python source. Compared with the clean checkout, the deployment replaces `multicamera4rc/train.py` and adds six files for the execution helpers, execution controller and six-rank gate.

The export verified all 1,076 files pinned by the original deployment manifest. `source/` contains 288 effective source and configuration files, including the runtime package, adapter code, transitively referenced controller scripts and vendor license notices. `base_overrides/multicamera4rc/train.py` holds the unmodified baseline training module for native checkpoint validation. `provenance.json` records the original file hashes and role-relative source paths, the seven-file overlay, and the difference between the complete deployment and this filtered export.

The core `model.py`, `config.py`, `refinement.py`, `train.py` and `launch.py` matched the original deployment at export. The subsequent source changes below are not covered by that historical byte-for-byte claim. This does not certify the whole runtime or make the exported v23 supervisor the latest recovery controller. A subsequent recovery used `reliability_hardened_v3/resume_latest.py`, outside this snapshot. See [model/version boundaries and verified hashes](../../docs/training-architecture.md). Identify an active recipe from its running command, checkpoint, and bound configuration.

## Supervision repair source update (2026-11-15)

`source/` now also includes the deployed calibrated Ego-Exo4D/Aria adapters and the following opt-in continuation features. Historical controller templates, `provenance.json`, and original dataset/loss recipes remain snapshots; reproduce a continuation using its own checkpointed recipe and deployment manifest.

- `dense_kind="geometry_tracks"` admits audited `mugen_h5` depth/cameras together with precomputed sparse trajectories. It requires real track masks, retains unknown temporal regions as unknown, and does not densify sparse trajectories. Dataset production and exclusion audits are external inputs.
- `reprojection_options` supervises predicted 3D trajectories projected through predicted cameras against independently supported GT image correspondences. It excludes source self-projection and uses GT visibility or measured depth witnesses. Predicted confidence/visibility cannot remove residuals. A continuation may use `weight=0.05`, `huber_delta=0.01` in normalized image coordinates, and a 2048-point budget per view.
- `focal_ratio_loss_weight=0.5`, with `intrinsics_loss_mode="log_focal"`, adds `abs(log(fx/fy)-log(fx_gt/fy_gt))` only on calibrated observations. This learns the calibrated ratio rather than assuming square focal lengths for every dataset. The existing absolute focal loss remains active.
- `geometry_source_min_pixels=64` prioritizes supported depth observations for partial geometry samples; motion-labelled sources retain precedence. No depth-validity or occlusion tolerance is relaxed.
- `rewarm_group_plan` creates a named, checkpointed LR continuation. The tested geometry ramp is 1.5 times the boundary LR over 500 updates, with other groups, Adam moments and the original global cosine phase preserved. Increasing the target step alone does not create this ramp.
- Legacy resize utilities now transform pixel centres and intrinsic principal points together using `(x + 0.5) * scale - 0.5`. The current native loader already used this convention.

These changes do not create dense human-surface temporal GT. Ego-Exo4D supplies its existing audited sparse joint displacement supervision; adjacent depth maps alone do not identify the same moving surface point.

`source/scripts/depth_alignment_audit.py` exposes a separate least-squares alignment diagnostic. It preserves the GT-valid denominator and fixed log floor. It does not replace the frozen official scorer, remove failed predictions, or derive a clipping floor from evaluation GT. Visibility thresholds and content-canvas intrinsics likewise require separately labelled inference experiments.

Targeted tests cover GT masks, focal-ratio gradients, pixel-centre geometry, two-rank training/save/resume, and sparse-motion retention. B200 real-sample probes additionally cover Waymo, Ego-Exo4D, Kinoptic and Kubric with the full model and Adam state. These are numerical/execution checks, not evidence of benchmark improvement after training.

## Launch chain and dependencies

The entry point at export time was `supervisor/supervisor_resume_v23.py`. It imports `control_common.py`, `native_update_audit_v23.py` and `scripts.async_training_state` from the baseline checkout. It launches the private Python wrapper, which loads `future_private_bootstrap.py` and runs `multicamera4rc.launch`. The launcher calls `torch.distributed.run`, which starts six `multicamera4rc.train` ranks through the same wrapper. The bootstrap adds the explicit `fa4_metadata_prepared_kv` execution mode. The supervisor checks the per-rank proof with `multicamera4rc/six_rank_gate_v22.py`.

The exported supervisor command is `python -u -B supervisor_resume_v23.py serve --lease-self --plan "$TRAINING_PLAN" --plan-sha256 "$TRAINING_PLAN_SHA256"` with the original pinned plan and exclusive leases. Use `check_plan` in place of `serve` and omit `--lease-self` for read-only plan verification.

The exported run launched on physical GPUs `0,1,2,5,6,7`, from step `140501` toward `200000`, saving every 100 steps. The later recovery resumed from `175300` with the same decoder-memory mode and core source; its recovery controller is not part of this snapshot. It uses the `giant` preset, `bfloat16`, batch size and gradient accumulation of 1, four sources, at most 64 source-time pairs, two workers, four CPU threads, seed 49, and `target_time` decoder memory. The launcher uses `--prefer-motion-sources --no-query-chunking --no-reuse-decoder-kv --no-reuse-shared-computation`. The 28-source recipe comes from `optional_long.json` and the loss recipe from `latest_merged_loss.json`.

The supervisor loads `evaluate_joint_checkpoint_v22.py` dynamically for training-side checkpoint validation on physical GPUs `3,4`. This module uses `scripts.independent_validation`, `multicamera4rc/schedule.py`, the native Git checkout, and external fixed validation plans. Its template captures the relevant configuration and bindings. Runtime proof receipts, checkpoints, data mounts, data manifests, masks, caches, and validation outputs are supplied separately.

Python is `3.10.20`. The bootstrap binds PyTorch `2.10.0+cu128` and CUDA `12.8` from the shared environment, `flash_attn_4` `4.0.0b37` from the private environment, and OpenEXR/Imath from the data environment. `runtime_packages.json` records the corresponding installed distribution metadata. `source/pyproject.toml` lists the source package's dependencies.

## Stereo4D sparse-track integration (2026-11-15)

The `stereo4d` adapter uses `dense_kind="camera_tracks"` inside the existing dense mixture, while retaining sparse trajectory, displacement and velocity targets. Released left-rectified RGB and `camera2world` poses share the published track coordinate system. Timestamps are microseconds; velocity uses original frame indices and rejects nonuniform frame intervals. The 60-degree rectified camera uses pixel-centre principal points `(255.5, 255.5)` at 512 pixels. Rectification is not applied twice.

Supply an audited train-only JSONL inventory through `options.inventory`; each row contains `clip`, `split`, `frames`, `npz`, and `left_video`. Exclude held-out video IDs before building this inventory. The adapter checks RGB frame count, finite recorded trajectories, and RGB forward/backward correspondence for source queries. A missing track record remains invalid. The labels are published **pseudo trajectories**, not guaranteed sensor ground truth. Pose translation and sparse XYZ/displacement are normalized by the same scene scale.

The binocular option (`cameras=2`, adapter `stereo=True`) requires `right_video` in each admitted inventory row. It uses the official fixed 0.063 m pseudo-metric baseline: rectified rotations coincide and the right centre is `t_left + 0.063 * R_left[:, 0]`. This is not a recovered per-device measured calibration. Rectification rotations have already been applied to the RGB and must not be applied again. Right source queries require independent SGBM correspondence, forward/backward LK checks, and at most 1.5 pixels residual against the projected published trajectory. Unverified queries are masked; insufficient support rejects the sample.

Neither mode fabricates dense depth, normals or occlusion labels. The visibility-valid mask is false and the loss tolerates an intentionally disabled visibility head. The mixed continuation retains the total 10% Stereo4D allocation, split evenly between monocular and binocular sources, while retaining all other source weights. This is a sampling proportion, not a claim about the proportion of valid supervised points. Binocular inventory membership additionally requires both videos; monocular sampling still covers all admitted left videos.

Admission checks used 10 independent real-clip geometry audits, 32 sampled loading/mask checks, and five full-model B200 optimizer updates with an intermediate model/Adam/LR save and reload. These tests establish pipeline operation and finite gradients; they do not establish convergence or an evaluation gain. The production checkpoint, inventory digest and migration receipt remain external deployment artifacts.

## Kinoptic measured depth and separate human-motion labels

The opt-in `kinoptic_human` adapter uses fixed two-camera, two-time packs with `dense_kind="kinoptic_human"`. Its manifest pins each pack hash, the original training-capture whitelist and an auxiliary capture split. Production configuration should also pin `options.manifest_sha256`; unadmitted candidate packs require an explicit test-only `allow_candidate=true`.

The three supervision families remain separate:

- Measured Kinect depth retains the original calibration, timing, sensor agreement, occlusion and RGB verification filters. Selecting supported neighbouring HD camera pairs changes sampling, never fills depth holes. Any measured coverage gain must be reported as conditional on the accepted pairs.
- `joint_motion_*` contains official joint-centre displacement with measured-surface proximity checks in both cameras and at both times. It is an auxiliary motion target, not absolute human-surface XYZ or visibility ground truth.
- `pseudo_motion_*` contains RGB-correspondence/depth surface estimates checked with a third camera. These are pseudo labels, not dense temporal GT. They do not populate `dense_pair_valid`, sparse GT trajectories, or visibility targets. Shared anchor-camera errors remain possible despite third-view consistency.

`auxiliary_motion_options` enables independently logged joint and pseudo losses. The initial candidate uses weights `0.1` and `0.02`, respectively, with Huber delta `0.01` in scene-normalized units. Pseudo residuals inside a positive, depth-dependent uncertainty band receive zero gradient. This band is a conservative heuristic, not a calibrated accuracy guarantee. Displacements and uncertainty use the same scene normalization; crop operations update query locations and masks. All options default to disabled for older recipes.

Real-sample admission and optimizer smoke tests establish valid loading, finite updates and gradient connectivity. They do not establish convergence, evaluation gains or the availability of true dense human-surface motion GT. Deployment manifests and dated admission receipts determine whether an optional source is active.

## Materialize the runtime bindings

Use the repository's `research/materialize.py` with the template directory, a filled copy of its `bindings.example.json`, and a fresh destination. The destination must be outside the source snapshot. Use absolute ASCII paths without spaces, quotes or shell metacharacters for the bindings. If the checkout path falls outside that format, copy `source/` to a suitable runtime path. The `binding_locations.json` files identify the fields served by each binding.

1. Choose the effective source tree. For the original private environment path, use `source/` and keep the example value for `TRAINING_EXECUTION_SOURCE_SHA256`. To use a different private environment path, copy `source/` to a fresh directory, render `templates/execution_overlay/` with the resolved absolute `PRIVATE_ENV`, and place the generated `multicamera4rc/training_execution_future.py` at the same relative path in that copy. The template changes the expected environment path in the existing guard. Keep the audited `source/` copy intact.
2. Render `templates/runtime/` to an isolated runtime directory using the same resolved `PRIVATE_ENV`. Bind the shared and data sites, the rendered bootstrap, and the planned deployment JSON. Before rendering `templates/inputs/`, calculate the SHA-256 of the bootstrap, wrapper and selected `multicamera4rc/training_execution_future.py`. Set `RUNTIME_BOOTSTRAP_SHA256`, `RUNTIME_WRAPPER_SHA256` and `TRAINING_EXECUTION_SOURCE_SHA256` to those hashes. Set `SOURCE_ROOT` to the selected source tree and supply the GPU UUIDs and dataset bindings. Verify that `deployment.json` pins the selected controller's exact bytes before running the CPU preflight below.
3. For checkpoint validation and supervised launch, render `templates/supervisor_code/`, `templates/checkpoint_validation_config/`, `templates/activation_config/` and `templates/supervisor_plan/` in that order. Supply the SHA-256 values of the files produced by each preceding stage and the trusted external proof records. Run the control scripts from a directory that also exposes the rendered `control_common.py` to Python.

The checkpoint and evaluator for this deployment require its original deployment, input hashes, and Git identity. The filtered export has a new deployment digest, and the optional controller template changes the execution implementation digest. An exact resume requires the original bound deployment and artifacts. The corresponding controller also requires its pinned publisher and gate records before it can adopt a checkpoint. Repository templates retain their original defaults; refer to this deployment's bound `optional_long.json` and loss recipe for its effective configuration.

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
