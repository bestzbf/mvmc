# Evaluation runtime source snapshot

This snapshot records the deployed evaluation entry points, two benchmark harness versions, six model adapters, the streamed Ours wrapper, the score consumers, recovery helpers, and report generation. The source hashes identify the actual deployed bytes. Templates carry path and device bindings outside source control.

Copy `bindings.example.json` to a private bindings file and fill in the deployment values. Run the repository's `research/materialize.py` with this directory, that bindings file, and a new output directory. Set each role root to its corresponding materialized directory. Supply the pinned external checkouts from `third_party_versions.json` and the Python environments, weights and benchmark inputs separately. Bind the Ours frozen model root to its pinned Git checkout when using the original Git provenance guard.

`baseline_assets/eval_full_20261105/full_runner.py` consumes a private queue plan. The preparation and score-job builders produce runtime manifests and recipes from supplied sources. Existing output indices, checkpoint hashes and receipt signatures continue to be checked by the original evaluation code. Using a different installation path changes the materialized file hashes. Generate manifests and receipts for that installation. Restoring the original private bindings reproduces the original source hashes.

The two harness directories are separate source roots. Choose the one declared by the job. Do not combine their modules into a single Python package root. The archived 10-hour scorer helpers are retained as score dependencies. Production plans, model payloads, case identifiers, scores and logs are supplied as private runtime assets.

After materializing, set `EVAL_FULL_ROOT` to the bound full evaluation directory and `EVAL_PYTHON` to the evaluation environment's Python. Build and validate the plan against the local inputs before starting the runner. The reporting command reads the results from that runtime.

```sh
"$EVAL_PYTHON" "$EVAL_FULL_ROOT/full_runner.py" --plan "$EVAL_FULL_ROOT/queue_plan.json"
"$EVAL_PYTHON" "$EVAL_FULL_ROOT/scripts/report_full.py" --root "$EVAL_FULL_ROOT" --once
```

For a new evaluation using the repository's exported step-134400 model, apply the [snapshot identity overlay](../frozen_134400/README.md) before constructing the plan. The archived recovery controller also requires the matching allocator proof, original job statuses and source records for the run it recovers.

`OURS_STREAM_COMPONENT_ROOT` points to the materialized `benchmarks/evaluation` subtree for the four dynamically loaded streaming modules. `BENCHMARK_ROOT` selects the benchmark assets and the shared GPU lock directory. The runtime tree preserves the relative sibling names used by the score builders, including `eval_10h_20261104` and `adapter_cpu_tests`.
