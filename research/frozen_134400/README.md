# Frozen step-134400 evaluation model

`model_source/` contains the model import closure from source commit `dd45a94604b20970b8d89cd177d614c3698fa8a9` and the original vendored model package with its license notices. The original Git tree is `d407a307313c9be838cfb75502371201f223b145`. The `multicamera4rc` package tree is `7e70902c3ff4ae0b6ee69bbf209fa7fc09ba55d7`. `git_source_blobs.tsv` records the original blob IDs and modes for the exported files. The 53 file hashes in `source_manifest.json` were checked against the original Git objects and the frozen evaluation checkout.

The step-134400 checkpoint and benchmark inputs are separate runtime dependencies. Historical Git-verified evaluation requires a separate clean checkout of the original commit and the original harness. Keep this model and the current training model in separate Python processes because both use the `multicamera4rc` namespace.

## Verify the export

```sh
python "$FROZEN/portable_identity.py" "$FROZEN/source_manifest.json"
```

Use absolute paths for `FROZEN` and `OURS_HARNESS`. Apply the identity overlay to a freshly materialized Ours harness. Its unmodified `multicamera4rc/benchmarks/refined_current.py` must have the original SHA-256 listed in `verification.json`. That file has no path placeholders.

```sh
git -C "$OURS_HARNESS" apply --check "$FROZEN/refined_current_exported_snapshot.patch"
git -C "$OURS_HARNESS" apply "$FROZEN/refined_current_exported_snapshot.patch"
cp "$FROZEN/portable_identity.py" "$OURS_HARNESS/multicamera4rc/benchmarks/portable_identity.py"
```

For an exported-model run, set `MULTID4RT_EXPORTED_MODEL_MANIFEST="$FROZEN/source_manifest.json"` in the evaluation subprocess environment. Use `--model-code-root "$FROZEN/model_source"` and `--expected-model-commit dd45a94604b20970b8d89cd177d614c3698fa8a9` with the normal harness options. Start a fresh process and write to a new prediction root. The runtime records the exported snapshot identity and verifies every listed file before it loads the model. Existing production prediction receipts retain their original Git-checkout identity. Leave the environment variable unset for the original Git-verified mode.

The overlay adds source verification only. The model, checkpoint loading, inference and scorer code come from their original sources. Keep the snapshot source directory free of extra files, including bytecode caches.
