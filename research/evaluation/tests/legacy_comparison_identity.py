"""Identity verifier extracted from the pinned legacy comparison runner.

Original source SHA-256
757697ee83de9ffcc53a99690601268fff54420a7dd0f1afe60bd3a85cf18f78
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("Nonfinite JSON token: " + value)))

class FileHashes:
    """One full hash per unchanged inode/stat inside one queue execution."""
    def __init__(self):
        self.cache = {}
        self.lock = threading.RLock()

    def sha256(self, path):
        path = Path(path).resolve(strict=True)
        before = path.stat()
        signature = (before.st_dev, before.st_ino, before.st_size,
                     before.st_mtime_ns, before.st_ctime_ns)
        with self.lock:
            if self.cache.get(path, (None, None))[0] == signature:
                return self.cache[path][1]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
                digest.update(block)
        after = path.stat()
        require(signature == (after.st_dev, after.st_ino, after.st_size,
                              after.st_mtime_ns, after.st_ctime_ns),
                "File changed during checksum: " + str(path))
        value = digest.hexdigest()
        with self.lock:
            self.cache[path] = (signature, value)
        return value

def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root), "Artifact escaped its declared directory")
    return path

def valid_digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None

def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()

def baseline_artifacts(value):
    """Find declared real files without confusing a composed identity hash for a file."""
    artifacts = []
    if isinstance(value, dict):
        filename = value.get("path", value.get("checkpoint_path"))
        digest = value.get("sha256", value.get("checkpoint_sha256"))
        if isinstance(filename, str) and valid_digest(digest):
            artifacts.append({"path": filename, "sha256": digest})
        for child in value.values():
            artifacts.extend(baseline_artifacts(child))
    elif isinstance(value, list):
        for child in value:
            artifacts.extend(baseline_artifacts(child))
    return artifacts

def verify_baseline_sources(value, hashes):
    if not isinstance(value, dict):
        return
    if "source_root" in value:
        root = Path(value["source_root"]).resolve(strict=True)
        source = value.get("source", {})
        expected_revision = value.get("source_commit", source.get("revision"))
        require(expected_revision, "Baseline source lacks its pinned revision")
        archive = root / "code-provenance.json"
        if source.get("verification") == "clean official git archive file manifest":
            declaration = read_json(archive)
            files = declaration.get("files", {})
            require(declaration.get("code_revision") == expected_revision and files and
                    len(files) == source.get("verified_native_files"),
                    "Author archive source manifest identity changed")
            for relative, digest in files.items():
                require(valid_digest(digest) and hashes.sha256(contained(root, relative)) == digest,
                        "Author archive source bytes changed: " + relative)
        else:
            revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            require(revision == expected_revision, "Baseline source revision changed")
            require(not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1",
                                                 "--untracked-files=no"]), "Baseline source has tracked edits")
        for relative, digest in value.get("source_sha256", {}).items():
            require(hashes.sha256(contained(root, relative)) == digest,
                    "Baseline source bytes changed: " + relative)
    # OmniX stores two repository manifests under source.omnix and source.da3,
    # and records actual installed extension/preload bytes separately.
    if "root" in value and "revision" in value and "files_sha256" in value:
        root = Path(value["root"]).resolve(strict=True)
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        require(revision == value["revision"] and value.get("tracked_worktree") == "clean",
                "Author repository source revision changed")
        require(not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1",
                                             "--untracked-files=no"]), "Author repository has tracked edits")
        require(value["files_sha256"], "Author repository source hashes are empty")
        for relative, digest in value["files_sha256"].items():
            require(valid_digest(digest) and hashes.sha256(contained(root, relative)) == digest,
                    "Author repository source bytes changed: " + relative)
    for child in value.values():
        if isinstance(child, dict):
            verify_baseline_sources(child, hashes)

def verify_checkpoint_file(job, hashes):
    if job["identity_kind"] == "native":
        cp = job["checkpoint"]
        require(hashes.sha256(cp["path"]) == cp["sha256"], "Checkpoint bytes differ from plan")
        artifacts = job.get("checkpoint_artifacts", [])
    else:
        declaration = job.get("expected_baseline_checkpoint", {})
        artifacts = baseline_artifacts(declaration) + job.get("checkpoint_artifacts", [])
        verify_baseline_sources(declaration, hashes)
    for artifact in artifacts:
        require(valid_digest(artifact.get("sha256")) and
                hashes.sha256(artifact["path"]) == artifact["sha256"],
                "Auxiliary baseline/model checkpoint bytes differ")

def verify_identity(job, checkpoint, hashes):
    require(isinstance(checkpoint, dict), "Missing real checkpoint declaration")
    verify_checkpoint_file(job, hashes)
    if job["identity_kind"] == "baseline":
        if "expected_baseline_checkpoint" in job:
            require(checkpoint == job["expected_baseline_checkpoint"], "Baseline declaration differs from plan")
        if "expected_baseline_checkpoint_sha256" in job:
            require(canonical_sha256(checkpoint) == job["expected_baseline_checkpoint_sha256"],
                    "Baseline declaration canonical hash differs")
        require(not checkpoint.get("native_loading") and
                checkpoint.get("native_multicamera4rc_loading") is not True,
                "Baseline must not masquerade as the native trained model")
        require(checkpoint.get("method") == job["method"], "Wrong baseline method declaration")
        artifacts = baseline_artifacts(checkpoint)
        require(artifacts, "Baseline declaration lacks real component checkpoints")
        for artifact in artifacts:
            require(hashes.sha256(artifact["path"]) == artifact["sha256"],
                    "Declared baseline component checkpoint changed")
        verify_baseline_sources(checkpoint, hashes)
        return
    expected = job["checkpoint"]
    require(checkpoint.get("sha256") == expected["sha256"] and
            Path(checkpoint.get("path", "")).resolve() == Path(expected["path"]).resolve(),
            "Prediction checkpoint identity differs from frozen plan")
    loading = checkpoint.get("native_loading", {})
    require(loading.get("checkpoint_step") == expected["step"], "Wrong native checkpoint step")
    coverage = loading.get("state_coverage", {})
    require(coverage.get("keys_exact") is True and coverage.get("shapes_exact") is True
            and coverage.get("refiner_configured") is True and coverage.get("refiner_tensors") == 42,
            "Native full state/refiner coverage is incomplete")
    count = coverage.get("parameter_tensors")
    expected_count = expected.get("parameter_tensors", 1148)
    require(type(count) is int and count == expected_count and
            coverage.get("checkpoint_tensors") == count, "Incomplete native tensor count")
    report = loading.get("load_report", {})
    require(report.get("matched") == count and report.get("missing") == [] and
            report.get("missing_base") == [] and report.get("unexpected") == [],
            "Native strict load did not cover every saved tensor")
    require(loading.get("audit_only") is False, "Inference cannot reuse CPU audit-only evidence")
    config = loading.get("config_coverage", {})
    require(config.get("all_init_fields_saved") is True and
            config.get("no_unknown_config_fields") is True and
            config.get("saved_restored_values_equal") is True,
            "Native saved configuration coverage is incomplete")
    root = Path(expected["model_code_root"]).resolve(strict=True)
    runtime = loading.get("model_runtime", {})
    require(runtime.get("revision") == expected["model_commit"] and
            runtime.get("tracked_worktree") == "clean" and
            Path(runtime.get("model_code_root", "")).resolve() == root and
            Path(loading.get("model_code_root", "")).resolve() == root,
            "Native source root/commit differs")
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    require(actual == expected["model_commit"], "Native model checkout revision changed")
    require(not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1",
                                         "--untracked-files=no"]),
            "Native model checkout has modified tracked files")
    files = runtime.get("files_sha256")
    require(isinstance(files, dict) and files, "Native runtime lacks source byte bindings")
    for relative, digest in files.items():
        require(valid_digest(digest) and hashes.sha256(contained(root, relative)) == digest,
                "Native model source bytes changed: " + relative)
