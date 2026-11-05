"""Generate a tiny five-suite input contract for the CPU admission tests."""

import argparse
import json
from pathlib import Path
import sys

import evaluate_joint_checkpoint_v22 as evaluate
from scripts.migrate_dynamic_validation import SUITES, build_validation_contract


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return str(path)


def create_fixture(root, source, lock_prefix):
    root.mkdir()
    schedule = source / "multicamera4rc/schedule.py"
    manifest = write_json(root / "runtime.json", {
        "source_base_commit": evaluate.COMMIT,
        "candidate_source": str(source),
        "files": {str(schedule): evaluate.digest(schedule)},
    })
    job = {
        "repo": str(source),
        "python": sys.executable,
        "commit": evaluate.COMMIT,
        "output_root": str(root / "unused-output"),
        "main_active_path": str(root / "unused-active.json"),
        "physical_gpus": evaluate.TRAIN_GPUS,
        "validation_gpus": evaluate.EVAL_GPUS,
        "validation_gpu_uuids": {str(i): f"synthetic-gpu-{i}" for i in evaluate.EVAL_GPUS},
        "cooperative_gpu_locks": {
            str(i): [f"{lock_prefix}{i}_single_inference.lock"] for i in evaluate.EVAL_GPUS
        },
        "allow_shared": False,
        "environment": {},
        "admitted_training_data_sha256": "a" * 64,
        "admitted_execution_implementation_sha256": {"synthetic.py": "b" * 64},
        "eval_runtime_manifest": manifest,
        "eval_runtime_manifest_sha256": evaluate.digest(manifest),
        "eval_shim_sha256": evaluate.digest(evaluate.__file__),
        "expected_state": {
            "supervision": "dense",
            "amp_dtype": "bfloat16",
            "loss_config": {},
            "load_balancing": {},
            "lr_plan": {
                "mode": "cosine",
                "start_step": 95000,
                "end_step": 200000,
                "base_lrs": [1e-5, 1e-4, 1e-4],
            },
        },
        "model_config": {
            "reuse_decoder_kv": False,
            "reuse_shared_computation": False,
            "decoder_kv_cache_limit_mb": 8192,
        },
        "evaluation": {},
    }
    for suite in SUITES:
        config = write_json(root / f"{suite}_config.json", {"fixture_suite": suite})
        plan = write_json(root / f"{suite}_plan.json", {"cases": [{"id": f"synthetic-{suite}"}]})
        if suite == "syn4d":
            job.update(syn4d_eval_config=config, syn4d_eval_plan=plan)
        else:
            prefix = "native" if suite == "native_dynamic" else suite
            job["evaluation"].update({f"{prefix}_config": config, f"{prefix}_plan": plan})
    job["validation_contract"] = build_validation_contract(job)
    return write_json(root / "job.json", job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("lock_prefix")
    args = parser.parse_args()
    print(create_fixture(args.destination.resolve(), args.source.resolve(strict=True), args.lock_prefix))
