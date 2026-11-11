# Changelog

## v0.5.2 — 2026-11-11

- Rename the project and distribution to mvmc; add MVMC/MVMCConfig and mvmc CLI entry points while retaining the multid4rt API and existing checkpoint identities.
- Distinguish all-observation defaults from target-time K/V used by the released 60k checkpoint and active research training.
- Document the research refiner, visibility ordering, core-source hash verification and exported-controller snapshot boundary.
- Reject unsupported refiner checkpoints in the lightweight inference loader with an explicit route to the research implementation.

## v0.5.1 — 2026-10-09

- Add curated real RGB reconstruction examples using the trained giant checkpoint at step 56,000, with original inputs, offline viewers, PLY, tracks and scene flow.
- Sample display pixels uniformly without replacement using a fixed seed to avoid aliasing from flat row-major strides, retaining exact point budgets and source identity.
- Verify native CUDA inference, unchanged network/checkpoint behavior and offline browser controls; preserve input sizes and shared camera/frame labels.

## v0.5.0 — 2026-10-09

- Add one inference/visualization command for synchronized multi-camera clips, static multiview images and monocular videos/sequences.
- Preserve image aspect ratio, shared frame labels, manifest reference order and raw-pixel query transforms; exclude padding from point exports.
- Export an offline interactive HTML viewer, binary PLY, 3D tracks and scene-flow arrays, with prediction reuse.
- Add optional visualization dependencies and packaged viewer assets; retain the existing network and native checkpoint behavior.

## v0.4.0 — 2026-10-09

- Rename the project, Python package, model and config to MultiD4RT, multid4rt, MultiD4RT and MultiD4RTConfig.
- Focus the release on the model, RGB manifest inference and downstream point/flow APIs.
- Keep checkpoint parameter keys and accept earlier native architecture markers.
- Simplify README and retain detailed model/inference documentation.

Earlier tagged releases remain available in repository history.
