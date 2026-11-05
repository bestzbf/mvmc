"""Run the evaluation regression tests against a temporary CPU-only deployment."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


EVALUATION = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALUATION.parent))
from materialize import materialize, relative_path  # noqa: E402

CHECKOUT_VARIABLES = {
    "arrow": "ARROW_REPO",
    "mose3": "MOSE3_REPO",
    "point4d": "POINT4D_REPO",
    "relift": "RELIFT_REPO",
    "uniquery4r": "UNIQUERY4R_REPO",
}


def pinned_checkout(name, value):
    path = Path(value).resolve(strict=True)
    versions = json.loads((EVALUATION / "third_party_versions.json").read_text())
    expected = next(row["commit"] for row in versions["author_checkouts"] if row["name"] == name)
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    root = subprocess.check_output(["git", "-C", str(path), "rev-parse", "--show-toplevel"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "-uno"], text=True)
    if revision != expected or dirty or Path(root).resolve() != path:
        raise ValueError(f"Use a clean {name} checkout at {expected}")
    return path


def stage_tests(runtime):
    tests = EVALUATION / "tests"
    entries = json.loads((tests / "test_paths.json").read_text())["files"]
    selected = []
    for entry in entries:
        source = tests / relative_path(entry["source"])
        target = runtime / relative_path(entry["output_path"])
        if target.exists():
            raise ValueError(f"Test path overlaps another runtime file: {entry['output_path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if entry.get("collect", True):
            selected.append(str(target))
    if not selected:
        raise ValueError("No evaluation regression tests are registered")
    return selected


def run_tests(checkouts, pytest_args):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    python_paths = []
    for name, variable in CHECKOUT_VARIABLES.items():
        value = checkouts.get(name) or environment.get(variable)
        if value:
            checkout = pinned_checkout(name, value)
            environment[variable] = str(checkout)
            package_root = checkout / "src" if (checkout / "src" / name).is_dir() else checkout
            python_paths.append(str(package_root))
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    if python_paths:
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    with TemporaryDirectory(prefix="multid4rt-evaluation-tests-") as directory:
        workspace = Path(directory)
        bindings = json.loads((EVALUATION / "bindings.example.json").read_text())
        bindings["bindings"].update({
            "BENCHMARK_CASE_295F46DA82": "panoptic/fixture_missing.npz",
            "BENCHMARK_CASE_4BC7D2E490": "adt_mini/fixture_ineligible.npz",
            "GPU3_UUID": "GPU-00000000-0000-0000-0000-000000000003",
            "GPU4_UUID": "GPU-00000000-0000-0000-0000-000000000004",
        })
        bindings_path = workspace / "bindings.json"
        bindings_path.write_text(json.dumps(bindings), encoding="utf-8")
        runtime = workspace / "runtime"
        materialize(EVALUATION, bindings_path, runtime)
        tests = stage_tests(runtime)
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *pytest_args, *tests],
            cwd=runtime,
            env=environment,
            check=False,
        ).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", action="append", default=[], metavar="NAME=PATH",
                        help="Pinned upstream checkout, repeat once per optional author dependency")
    args, pytest_args = parser.parse_known_args()
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    try:
        checkouts = {}
        for binding in args.checkout:
            name, separator, path = binding.partition("=")
            if name not in CHECKOUT_VARIABLES or not separator or not path or name in checkouts:
                raise ValueError(f"Invalid or duplicate checkout binding {binding!r}")
            checkouts[name] = path
        return run_tests(checkouts, pytest_args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
