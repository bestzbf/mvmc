# CPU regressions for the six-rank training deployment

The base training commit `069b1c28ce4763e2ac6a4f06dbc5b73c9ed3dc55` passed 95 CPU cases on the original server, including a CPU distributed test. Its ten exercised test modules are in `source/tests/multicamera4rc/`. The ScanNet and MUGEN fixture helpers generate small inputs on demand. Use the base `train.py` from `base_overrides/` for that suite.

| Activation test | Original server cases | Coverage |
| --- | ---: | --- |
| v22 coordinator | 7 | Exclusive leases, process identity, checkpoint bookkeeping and activation gates |
| v23 update audit | 5 | Nested step metadata, true duplicates and gaps, partial writes and malformed updates |
| v22 six-rank gate | 6 | Rank coverage, named Adam values and FA4 error handling |
| v22 checkpoint admission | 5 | Scientific checkpoint checks, all five validation contract entries and queue behavior |
| v14 checkpoint expansion | 8 | CPU serialization, sampler alignment and optimizer state preservation |
| v21 checkpoint publisher | 12 | Independent verification, atomic publication and failure behavior |

`provenance.json` identifies the frozen source by role and SHA-256. The v23 audit uses a generated six-rank update and a nested metadata record to exercise the same parser failure without the production log. The checkpoint admission tests use synthetic suite plans and a toy checkpoint while keeping the original admission assertions. These fixtures test the contracts on CPU. Actual model quality and GPU execution require the original evaluation runs and data.

## Run the base training suite

Use a Python environment with PyTorch 2.10 and the project's `data` and `test` dependencies. Linux with PyTorch Gloo and a working `pidfd` implementation covers all suites. From the repository root, prepare an isolated copy for the CPU tests.

```bash
TRAINING="$PWD/research/training"
RUN_ROOT=$(mktemp -d /tmp/multid4rt-cpu-tests.XXXXXX)
cp -R "$TRAINING/source" "$RUN_ROOT/source"
cp "$TRAINING/base_overrides/multicamera4rc/train.py" "$RUN_ROOT/source/multicamera4rc/train.py"
mkdir "$RUN_ROOT/temp"
export TMPDIR="$RUN_ROOT/temp" CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$RUN_ROOT/source"
(cd "$RUN_ROOT/source" && python -B -m pytest -q -p no:cacheprovider tests/multicamera4rc \
  --ignore=tests/multicamera4rc/test_monocular_benchmark.py \
  --ignore=tests/multicamera4rc/test_native_loaders.py)
```

## Run the activation and checkpoint suites

Keep the environment from the previous commands. Materialize the already-uploaded control code templates using synthetic GPU identities and local lock paths. The evaluator's lock binding is a filename prefix ending in `.gpu`.

```bash
python - "$TRAINING" "$RUN_ROOT" <<'PY'
import json
from pathlib import Path
import sys

training, root = map(Path, sys.argv[1:])
example = training / "templates/supervisor_code/bindings.example.json"
keys = json.loads(example.read_text())["bindings"]
values = {key: str(root / key.lower()) for key in keys}
for index, key in enumerate(key for key in keys if key.endswith("_UUID")):
    values[key] = f"GPU-00000000-0000-0000-0000-{index:012d}"
values["EVALUATION_GPU_LOCK_DIR"] = str(root / ".gpu")
(root / "supervisor-bindings.json").write_text(json.dumps({"bindings": values}))
PY
python -B research/materialize.py "$TRAINING/templates/supervisor_code" \
  "$RUN_ROOT/supervisor-bindings.json" "$RUN_ROOT/supervisor_code"
export PYTHONPATH="$RUN_ROOT/source:$TRAINING/supervisor:$RUN_ROOT/supervisor_code"
export EXPANSION_NATIVE_ROOT="$RUN_ROOT/source"
ACTIVATION="$TRAINING/tests/activation"
export EVAL_V22_TEMPLATE="$RUN_ROOT/evaluation_fixture/job.json"
python -B "$ACTIVATION/make_checkpoint_validation_fixture.py" \
  "$RUN_ROOT/evaluation_fixture" "$RUN_ROOT/source" "$RUN_ROOT/.gpu"
for test in test_control_cpu.py test_native_update_audit_v23.py test_six_rank_gate_v22.py \
  test_evaluate_joint_checkpoint_v22.py publisher/test_expand_checkpoint_v14.py \
  publisher/test_publish_expanded_checkpoint_v21.py; do
  python -B "$ACTIVATION/$test" || exit 1
done
```

The coordinator exercises the Linux `pidfd` API by sending signal zero to its own test process. Its other cases use isolated or mocked process state. Remove the scratch directory after the suites exit to release the toy checkpoints, lock files and generated plans.
