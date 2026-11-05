#!/usr/bin/env python3
"""Released benchmarks with external model activation before core imports.

Reuse the established refined_current bootstrap; this script never invokes the
96k reference runner, training, or stale benchmark modules from the model tree.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "multicamera4rc/benchmarks"


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("command", nargs="?")
    early.add_argument("--model-code-root", type=Path)
    early.add_argument("--expected-model-commit")
    args, _ = early.parse_known_args(arguments)
    runtime = None
    if (args.command == "predict" and args.model_code_root is not None
            and "--help" not in arguments and "-h" not in arguments):
        spec = importlib.util.spec_from_file_location("_released_model_bootstrap", BENCHMARK_ROOT / "refined_current.py")
        bootstrap = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = bootstrap
        spec.loader.exec_module(bootstrap)
        runtime = bootstrap.activate_external_model(args.model_code_root, BENCHMARK_ROOT,
                                                    expected_revision=args.expected_model_commit)
    else:
        sys.path.insert(0, str(ROOT))
    entry = importlib.import_module("multicamera4rc.benchmarks.released")
    return entry.main(arguments, model_runtime_manifest=runtime)


if __name__ == "__main__":
    main()
