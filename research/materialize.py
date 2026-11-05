"""Bind a verified research runtime template to a local deployment layout."""

import argparse
import hashlib
import json
import re
import shutil
import stat
from pathlib import Path, PurePosixPath


TOKEN = re.compile(rb"@@([A-Z][A-Z0-9_]*)@@")
SHA256_LITERAL = re.compile(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
LITERAL = re.compile(r"[A-Za-z0-9_./,:@%+=-]+")
GENERATED_MANIFEST = "runtime-source-manifest.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"Invalid manifest path {value!r}")
    return path


def link_source_pins(files, entries):
    """Resolve explicitly recorded source-to-source digests after binding paths."""
    resolved = {}
    visiting = set()

    def resolve(path):
        if path in resolved:
            return resolved[path]
        if path in visiting:
            raise ValueError(f"Cyclic source digest binding at {path}")
        visiting.add(path)
        contents, mode = files[path]
        replacements = {}
        literals = SHA256_LITERAL.findall(contents)
        for original, linked_path in entries[path].get("linked_source_sha256", {}).items():
            dependency = relative_path(linked_path)
            if dependency not in files or entries[dependency].get("original_sha256") != original:
                raise ValueError(f"Original source digest mismatch for {path} -> {dependency}")
            literal = original.encode("ascii")
            if literals.count(literal) != 1:
                raise ValueError(f"Expected exactly one original digest literal in {path}")
            replacements[literal] = digest(resolve(dependency)[0]).encode("ascii")
        if replacements:
            contents = SHA256_LITERAL.sub(lambda match: replacements.get(match[0], match[0]), contents)
        visiting.remove(path)
        resolved[path] = (contents, mode)
        return resolved[path]

    return {path: resolve(path) for path in files}


def prepare_runtime(template_dir, bindings_file, destination):
    root = template_dir.resolve(strict=True)
    manifest_bytes = (root / "template_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    bindings = json.loads(bindings_file.read_text(encoding="utf-8"))["bindings"]
    if not isinstance(bindings, dict):
        raise ValueError("The bindings field must be a JSON object")

    files = {}
    entries = {}
    bindings_used = {}
    missing = set()
    for entry in manifest["files"]:
        source = relative_path(entry["template_path"])
        target = relative_path(entry.get("output_path", str(source)))
        if str(target) == GENERATED_MANIFEST or target in files:
            raise ValueError(f"Duplicate or reserved output path {target}")
        source_file = (root / source).resolve(strict=True)
        if not source_file.is_relative_to(root) or not source_file.is_file():
            raise ValueError(f"Template leaves the source directory: {source}")
        contents = source_file.read_bytes()
        if digest(contents) != entry["template_sha256"]:
            raise ValueError(f"Template digest differs from the manifest: {source}")

        keys = {value.decode("ascii") for value in TOKEN.findall(contents)}
        replacements = {}
        for key in sorted(keys):
            if key not in bindings:
                missing.add(key)
                continue
            value = bindings[key]
            if isinstance(value, str):
                value = value.replace("${RUNTIME_ROOT}", str(destination))
            if not isinstance(value, str) or not LITERAL.fullmatch(value) or "@@" in value:
                raise ValueError(f"Binding {key} must be a nonempty, shell-safe literal")
            replacements[key.encode("ascii")] = value.encode("ascii")
        rendered = TOKEN.sub(lambda match: replacements.get(match[1], match[0]), contents)
        mode = entry.get("mode", stat.S_IMODE(source_file.stat().st_mode))
        if isinstance(mode, str):
            mode = int(mode, 8)
        files[target] = (rendered, 0o755 if mode & 0o111 else 0o644)
        entries[target] = entry
        bindings_used[target] = sorted(keys)
    if missing:
        raise ValueError("Missing bindings: " + ", ".join(sorted(missing)))
    if not files:
        raise ValueError("Template manifest contains no files")

    files = link_source_pins(files, entries)
    provenance = []
    for path, (contents, _) in files.items():
        entry = entries[path]
        record = {
            "path": str(path),
            "template_sha256": entry["template_sha256"],
            "original_sha256": entry.get("original_sha256"),
            "materialized_sha256": digest(contents),
            "bindings": bindings_used[path],
        }
        links = entry.get("linked_source_sha256", {})
        if links:
            record["linked_source_sha256"] = {
                dependency: digest(files[relative_path(dependency)][0])
                for dependency in links.values()
            }
        provenance.append(record)
    record = {
        "schema": "multid4rt-materialized-runtime-v1",
        "template_manifest_sha256": digest(manifest_bytes),
        "files": provenance,
    }
    files[PurePosixPath(GENERATED_MANIFEST)] = (
        (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        0o644,
    )
    return files


def materialize(template_dir, bindings_file, output):
    destination = output.resolve()
    files = prepare_runtime(template_dir, bindings_file, destination)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for path, (contents, mode) in files.items():
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
            target.chmod(mode)
    except Exception:
        shutil.rmtree(destination)
        raise
    return len(files) - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template_dir", type=Path)
    parser.add_argument("bindings", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        count = materialize(args.template_dir, args.bindings, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Materialized {count} source files in {args.output}")


if __name__ == "__main__":
    main()
