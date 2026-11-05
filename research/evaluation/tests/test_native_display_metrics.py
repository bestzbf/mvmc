"""Native display metrics from synthetic official-score metadata."""
import copy
import importlib.util
from pathlib import Path

import pytest


FILE = Path(__file__).with_name("native_display_metrics.py")
spec = importlib.util.spec_from_file_location("display_metrics", FILE)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def tap_score():
    ids = [f"drivetrack/fixture_{i:02}.npz" for i in range(50)]
    offsets = {
        "average_pts_within_thresh": 2,
        "average_jaccard": 1,
        "occlusion_accuracy": 32,
    }
    per_case = [
        {
            "id": case_id,
            "metrics": {"median": {key: (i + offset) / 128 for key, offset in offsets.items()}},
        }
        for i, case_id in enumerate(ids)
    ]
    return {
        "sample_ids": ids,
        "per_case": per_case,
        "metrics": {"drivetrack": {"median": {
            key: (24.5 + offset) / 128 for key, offset in offsets.items()
        }}},
    }


def mv_score():
    values = {
        "average_pts_within_thresh": 64.5,
        "average_jaccard": 21.25,
        "occlusion_accuracy": 87.75,
        "ate_visible": 14.25,
        "mte_visible": 11.5,
        "fde_visible": 17.0,
        "survival": 86.0,
    }
    headline = {f"3dpt/model__{key}__any": value for key, value in values.items()}
    headline.update({
        "3dpt/model__average_pts_within_thresh__dynamic": 1.0,
        "3dpt/model__average_pts_within_thresh__static": 99.0,
        "3dpt/model__average_pts_within_thresh__n/v": 98.0,
    })
    return {
        "sample_ids": ["panoptic/fixture_a.npz"],
        "protocol": {"mode": "query_anchor_sim3"},
        "metrics": {"panoptic": headline},
    }


def test_tap_units_and_unchanged_official_mean():
    score = tap_score()
    raw = score["metrics"]["drivetrack"]["median"]
    out = metrics.extract_native_metrics(score, "tapvid3d", "drivetrack_minival")
    assert out["APD_percent"]["mean"] == raw["average_pts_within_thresh"] * 100
    assert out["AJ_percent"]["mean"] == raw["average_jaccard"] * 100
    assert out["OA_percent"]["mean"] == raw["occlusion_accuracy"] * 100
    assert all(v["direction"] == "max" and v["std_unit"] == "percentage points" for v in out.values())


def test_mv_any_units_error_direction_and_excluded_diagnostics():
    out = metrics.extract_native_metrics(mv_score(), "mvtracker", "panoptic")
    assert out["APD_percent"]["mean"] == 64.5
    assert out["ATE_native"]["mean"] == 14.25
    assert out["ATE_native"]["direction"] == "min"
    assert out["ATE_native"]["unit"] == "cm"
    assert set(out) == {
        "APD_percent", "AJ_percent", "OA_percent", "ATE_native",
        "MTE_native", "FDE_native", "Survival_percent",
    }


def test_world_keeps_official_mean_without_inventing_std_or_disabled_metrics():
    ids = [f"adt_mini/fixture_{i:02}.npz" for i in range(50)]
    score = {
        "sample_ids": ids,
        "scored_sample_ids": ids[:39] + ids[40:],
        "metrics": {"adt_mini": {
            "avg_pts_global": 0.82,
            "avg_pts_sim3_closed": 0.92,
            "epe_global": 0.4,
            "epe_sim3_closed": 0.2,
            "epe_sim3": 0,
        }},
    }
    out = metrics.extract_native_metrics(score, "worldtrack", "trajectory_adt_mini")
    global_apd = out["APD_global_percent"]
    assert global_apd["mean"] == 82.0
    assert global_apd["sample_std"] is None
    assert global_apd["valid_N"] == 49
    assert global_apd["scope_ids"] == ids
    assert global_apd["scored_scope_ids"] == score["scored_sample_ids"]
    assert len(out) == 4 and out["EPE_closed_m"]["direction"] == "min"


def test_exact20_cached_scalar_subset_regression():
    score = tap_score()
    out = metrics.extract_native_metrics(score, "tapvid3d", "drivetrack_first20", score["sample_ids"][:20])
    assert out["APD_percent"]["mean"] == 11.5 / 128 * 100
    assert out["AJ_percent"]["mean"] == 10.5 / 128 * 100
    assert out["OA_percent"]["mean"] == 41.5 / 128 * 100
    assert all(v["source_full_aggregate_regression_pass"] and v["subset_of_source"] for v in out.values())


def test_subset_rejects_wrong_order_missing_id_and_changed_source_scalar():
    score = tap_score()
    selected = score["sample_ids"][:20]
    for ids in (selected[::-1], selected + ["missing"]):
        with pytest.raises(ValueError):
            metrics.extract_native_metrics(score, "tapvid3d", "drivetrack_first20", ids)
    bad = copy.deepcopy(score)
    bad["per_case"][0]["metrics"]["median"]["average_pts_within_thresh"] += 0.1
    with pytest.raises(ValueError, match="full-mean regression"):
        metrics.extract_native_metrics(bad, "tapvid3d", "drivetrack_first20", selected)


def test_unsupported_visibility_is_omitted_and_real_zero_remains_zero():
    score = mv_score()
    score["metric_support"] = {"native_visibility": False, "AJ": False, "OA": False}
    score["metrics"]["panoptic"]["3dpt/model__average_pts_within_thresh__any"] = 0.0
    out = metrics.extract_native_metrics(score, "mvtracker", "panoptic")
    assert "AJ_percent" not in out and "OA_percent" not in out
    assert out["APD_percent"]["mean"] == 0.0
