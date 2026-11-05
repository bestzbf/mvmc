#!/usr/bin/env python3
"""WorldTrack scorer bridge for native dense grids and explicit partial diagnostics.

Only the wrapper's RGB-size preflight changes. Author evaluator sampling,
alignment, metric bodies and per-dataset aggregation execute unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_native_grid_wrapper(code):
    sys.path.insert(0, str(code))
    import multicamera4rc.benchmarks.released_official as original
    import multicamera4rc.benchmarks.released_runner as runner
    source_path = Path(original.__file__)
    old = '            finite_tracks(pred, (len(images), images[0].height, images[0].width, 3))'
    new = '''            if (pred.ndim != 4 or pred.shape[0] != len(images)
                    or pred.shape[-1] != 3 or min(pred.shape[1:3]) < 1):
                raise ValueError("WorldTrack prediction must be [T,Hpred,Wpred,3]")
            finite_tracks(pred, pred.shape)'''
    source = source_path.read_text()
    if source.count(old) != 1:
        raise ValueError("Frozen WorldTrack wrapper shape assertion changed")
    modified = source.replace(old, new)
    namespace = {"__name__": original.__name__, "__package__": original.__package__,
                 "__file__": str(source_path)}
    exec(compile(modified, str(source_path), "exec"), namespace)
    runner.OfficialWorldTrack = namespace["OfficialWorldTrack"]
    return runner, {"original_wrapper": str(source_path), "original_wrapper_sha256": sha(source_path),
        "adapted_wrapper_sha256": hashlib.sha256(modified.encode()).hexdigest(),
        "adapter_sha256": sha(__file__),
        "change": "Finite native dense grid accepted at wrapper boundary; official evaluator unchanged"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-code-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--official-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-unverified-release", action="store_true",
                   help="Only explicitly labeled partial diagnostics; never a formal200-case result")
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    if manifest.get("benchmark") != "worldtrack":
        raise ValueError("Native-grid scoring requires WorldTrack")
    binding = manifest.get("protocol", {}).get("release_binding", {})
    if a.allow_unverified_release and binding.get("coverage_verified"):
        raise ValueError("A diagnostic must explicitly clear full-release coverage in its own manifest")
    runner, record = load_native_grid_wrapper(a.benchmark_code_root.resolve())
    runner.score_predictions(a.manifest, a.predictions, a.official_root, a.output,
                             allow_unverified_release=a.allow_unverified_release)
    result = json.loads(a.output.read_text())
    result["scorer_wrapper_adapter"] = record
    runner.write_json(a.output, result)
    print(a.output)


if __name__ == "__main__":
    main()
