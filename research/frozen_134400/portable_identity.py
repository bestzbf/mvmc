"""Validate the frozen step-134400 model export without emulating a Git checkout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


SOURCE_COMMIT = "dd45a94604b20970b8d89cd177d614c3698fa8a9"
SOURCE_TREE = "d407a307313c9be838cfb75502371201f223b145"
SOURCE_PACKAGE_TREE = "7e70902c3ff4ae0b6ee69bbf209fa7fc09ba55d7"
MANIFEST_SHA256 = "3008aacd4a077aa345c763d20cb0e2ebc826e23e846db81a17f98269dfb04ac6"
MANIFEST_ENV = "MULTID4RT_EXPORTED_MODEL_MANIFEST"


def build_exported_model_runtime_manifest(model_code_root, manifest_path, *, expected_revision=None):
    """Verify all export paths and bytes, returning a distinct runtime identity."""
    if expected_revision is not None and expected_revision != SOURCE_COMMIT:
        raise ValueError(f"Expected snapshot source commit {SOURCE_COMMIT}, got {expected_revision}")
    manifest = Path(manifest_path)
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("Snapshot manifest must be a regular file")
    manifest = manifest.resolve(strict=True)
    manifest_bytes = manifest.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Snapshot manifest does not match the committed SHA-256")
    identity = json.loads(manifest_bytes)
    if (
        identity.get("schema") != "multid4rt-exported-model-source-v1"
        or identity.get("source_commit") != SOURCE_COMMIT
        or identity.get("source_tree") != SOURCE_TREE
        or identity.get("source_package_tree") != SOURCE_PACKAGE_TREE
    ):
        raise ValueError("Snapshot manifest has incompatible source provenance")
    root = Path(model_code_root)
    if root.is_symlink() or root.resolve(strict=True) != manifest.parent / "model_source":
        raise ValueError("Snapshot model root must be the manifest's sibling model_source directory")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Snapshot model_source must be a directory")

    expected = identity["files_sha256"]
    directories = {
        parent.as_posix()
        for name in expected
        for parent in Path(name).parents
        if parent != Path(".")
    }
    seen = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Symlink in snapshot: {relative}")
        if path.is_dir() and relative in directories:
            continue
        if relative not in expected or not path.is_file():
            raise ValueError(f"Unexpected snapshot path: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[relative]:
            raise ValueError(f"Snapshot source SHA-256 mismatch: {relative}")
        seen.add(relative)
    if seen != set(expected):
        raise ValueError(f"Missing snapshot source: {sorted(set(expected) - seen)[0]}")

    # A second runtime verification must see the same path inventory.
    sys.dont_write_bytecode = True
    return {
        "manifest_version": 1,
        "model_code_root": str(root),
        "model_package": str(root / "multicamera4rc"),
        "revision": SOURCE_COMMIT,
        "tracked_worktree": "not_applicable_verified_export",
        "files_sha256": expected,
        "source_identity_mode": "exported_snapshot_v1",
        "source_tree": SOURCE_TREE,
        "source_package_tree": SOURCE_PACKAGE_TREE,
        "export_manifest_sha256": MANIFEST_SHA256,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_path", type=Path)
    options = parser.parse_args()
    runtime = build_exported_model_runtime_manifest(
        options.manifest_path.parent / "model_source",
        options.manifest_path,
        expected_revision=SOURCE_COMMIT,
    )
    print(f"{runtime['source_identity_mode']} {runtime['revision']}: verified {len(runtime['files_sha256'])} files")
