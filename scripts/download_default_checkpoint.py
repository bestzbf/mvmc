#!/usr/bin/env python3
"""Download and verify the ModelScope mvmc default inference release."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile


REPO = "binfengzou/MultiD4RT"
WEIGHT = "mvmc_giant_step195500.pt"
RUNTIME = "mvmc_inference_dd45a946.zip"
RELEASE = {
    WEIGHT: "2d76af9a0ded8b90f93954d1c730bd85a874e8d99bcae43149b7d2088cd4c5ea",
    RUNTIME: "2ccbae7e1468b6468e9a8615548fbbe8f63611b376b05755edc029632056cc75",
}


def verify(path, expected):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            sha.update(block)
    if sha.hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")


def unpack_runtime(archive, target):
    with zipfile.ZipFile(archive) as bundle:
        digest = json.loads(bundle.read("SOURCE.json"))["files"]
        if target.exists():
            # Reuse only an unmodified runtime, including its own digest file.
            if (target / "SOURCE.json").read_bytes() != bundle.read("SOURCE.json"):
                raise ValueError(f"Refusing to replace an existing runtime: {target}")
            for name, value in digest.items():
                verify(target / name, value)
            expected = set(digest) | {"SOURCE.json"}
            present = {p.relative_to(target).as_posix() for p in target.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts}
            if present != expected:
                raise ValueError(f"Unexpected files in {target}; use a new output directory")
            return
        # Only unpack the archive pinned above and publish an extracted tree
        # after verifying its contents. No checkpoint or code is executed here.
        with tempfile.TemporaryDirectory(prefix=".runtime-", dir=target.parent) as directory:
            candidate = Path(directory) / "verified"
            bundle.extractall(candidate)
            for name, value in digest.items():
                verify(candidate / name, value)
            candidate.rename(target)


def write_demo(directory):
    """Write a 2-camera, 2-time, RGB-only smoke input using just the stdlib."""
    directory.mkdir(exist_ok=True)
    observations = []
    width, height = 42, 28
    for camera in range(2):
        for time in (0, 5):
            image = f"camera_{camera}_frame_{time}.ppm"
            pixels = bytes(
                channel
                for y in range(height)
                for x in range(width)
                for channel in ((6*x + 15*time + 8*camera) % 256, (9*y + 20*camera) % 256,
                                (13*(x+y)) % 256)
            )
            (directory / image).write_bytes(f"P6\n{width} {height}\n255\n".encode() + pixels)
            observations.append(dict(image=image, camera_id=camera, time_idx=time))
    manifest = directory / "observations.json"
    manifest.write_text(json.dumps({"clips": [{"id": "default_weight_smoke", "observations": observations}]},
                                   indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("weights/mvmc-default"))
    args = parser.parse_args()
    try:
        from modelscope.hub.file_download import model_file_download
    except ImportError:
        parser.error("Install the downloader first: python -m pip install 'modelscope>=1.35.1'")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    def download(name):
        return Path(model_file_download(REPO, file_path=name, revision="master", local_dir=str(output)))

    config = json.loads(download("configuration.json").read_text())["model"]
    if (config["checkpoint"], config["runtime_archive"], config["step"]) != (WEIGHT, RUNTIME, 195500):
        raise ValueError("The ModelScope default changed; use the matching mvmc checkout.")
    runtime = download(RUNTIME)
    verify(runtime, RELEASE[RUNTIME])
    weight = download(WEIGHT)
    verify(weight, RELEASE[WEIGHT])
    unpack_runtime(runtime, output / "runtime")
    manifest = write_demo(output / "demo")
    print(json.dumps(dict(checkpoint=str(weight), runtime=str(output / "runtime"),
                          demo_manifest=str(manifest), sha256=RELEASE[WEIGHT]), indent=2))


if __name__ == "__main__":
    main()
