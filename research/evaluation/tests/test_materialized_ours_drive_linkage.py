"""Check the published source pins and streaming dependency paths after relocation."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import CodeType


SCRIPTS = Path(__file__).resolve().parent
RUNTIME = SCRIPTS.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materialized_route_and_streamed_dependency_identity(tmp_path):
    file = SCRIPTS / "route_ours_drive_lock.py"
    spec = importlib.util.spec_from_file_location("materialized_ours_route", file)
    route = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(route)

    record = json.loads((RUNTIME / "runtime-source-manifest.json").read_text())
    entries = {row["path"]: row for row in record["files"]}
    route_entry = entries[file.relative_to(RUNTIME).as_posix()]
    stream_path = route.SOURCE.relative_to(RUNTIME).as_posix()
    assert sha(route.SOURCE) == route.SOURCE_SHA == entries[stream_path]["materialized_sha256"]
    assert sha(file) == route_entry["materialized_sha256"]
    assert route_entry["linked_source_sha256"] == {stream_path: route.SOURCE_SHA}
    assert route.SOURCE_SHA != entries[stream_path]["original_sha256"]

    wrapper = compile(route.SOURCE.read_bytes(), str(route.SOURCE), "exec")
    main = next(c for c in wrapper.co_consts if isinstance(c, CodeType) and c.co_name == "main")
    changed, proof = route.redirect_one_status_constant(main, tmp_path / "new_status.json")
    assert proof["old_status"] == route.LEGACY_STATUS
    assert changed.co_code == main.co_code

    helpers = {
        "eval_exact_v2_20261102/visibility.py",
        "eval_exact_v9_fixed20_20261102/memory_exact.py",
        "eval_streaming_candidate_20261102/model.py",
        "eval_streaming_candidate_20261102/streaming.py",
    }
    for helper in helpers:
        relative = "benchmarks/evaluation/" + helper
        path = RUNTIME / relative
        identity = entries[relative]
        assert sha(path) == identity["materialized_sha256"] == identity["original_sha256"]
        assert str(path) in main.co_consts or (
            str(path.parent) in main.co_consts and path.name in main.co_consts
        )
