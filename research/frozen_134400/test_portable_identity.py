"""Tests for the exported model source identity, with no weights or GPU."""
from pathlib import Path
import shutil

import pytest

from portable_identity import (
    SOURCE_COMMIT,
    build_exported_model_runtime_manifest,
)


EXPORT = Path(__file__).resolve().parent


@pytest.fixture
def copied_export(tmp_path):
    manifest = tmp_path / "source_manifest.json"
    root = tmp_path / "model_source"
    shutil.copy2(EXPORT / manifest.name, manifest)
    shutil.copytree(EXPORT / root.name, root)
    return root, manifest


def test_exact_export(copied_export):
    root, manifest = copied_export
    verified = build_exported_model_runtime_manifest(root, manifest, expected_revision=SOURCE_COMMIT)
    assert verified["source_identity_mode"] == "exported_snapshot_v1"
    assert verified["tracked_worktree"] == "not_applicable_verified_export"
    assert verified["revision"] == SOURCE_COMMIT
    assert len(verified["files_sha256"]) == 53
    assert verified == build_exported_model_runtime_manifest(root, manifest)


@pytest.mark.parametrize("mode", ["edited", "missing", "extra", "bytecode", "symlink"])
def test_rejects_changed_file_inventory(copied_export, mode):
    root, manifest = copied_export
    model = root / "multicamera4rc/model.py"
    if mode == "edited":
        model.write_bytes(model.read_bytes() + b"\n")
    elif mode == "missing":
        model.unlink()
    elif mode == "extra":
        (root / "torch.py").touch()
    elif mode == "bytecode":
        cache = model.parent / "__pycache__"
        cache.mkdir()
        (cache / "model.cpython-312.pyc").touch()
    else:
        content = root.parent / "same_source.py"
        model.rename(content)
        model.symlink_to(content)
    with pytest.raises(ValueError):
        build_exported_model_runtime_manifest(root, manifest)


def test_rejects_changed_manifest_even_with_valid_json(copied_export):
    root, manifest = copied_export
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="committed SHA-256"):
        build_exported_model_runtime_manifest(root, manifest)


def test_rejects_expected_different_commit(copied_export):
    root, manifest = copied_export
    with pytest.raises(ValueError, match="Expected snapshot source commit"):
        build_exported_model_runtime_manifest(root, manifest, expected_revision="0" * 40)


def test_rejects_exact_files_at_wrong_root(copied_export):
    root, manifest = copied_export
    wrong = shutil.copytree(root, root.parent / "other_model_source")
    with pytest.raises(ValueError, match="sibling model_source"):
        build_exported_model_runtime_manifest(wrong, manifest)
