import json

import pytest

from src.router.policy_taxonomy_utils import (
    apply_runtime_profile,
    build_policy_graph,
    deep_merge_dicts,
    load_crop_part_compatibility,
    load_taxonomy,
    policy_enabled,
    resolve_requested_profile,
)


def test_deep_merge_dicts_does_not_mutate_inputs():
    base = {"a": {"x": 1}, "b": 2}
    override = {"a": {"y": 3}, "c": 4}
    merged = deep_merge_dicts(base, override)

    assert merged == {"a": {"x": 1, "y": 3}, "b": 2, "c": 4}
    assert base == {"a": {"x": 1}, "b": 2}
    assert override == {"a": {"y": 3}, "c": 4}


def test_build_policy_graph_applies_overrides():
    vlm_config = {"policy_graph": {"open_set_gate": {"enabled": False}}}
    graph = build_policy_graph(vlm_config)
    assert graph["open_set_gate"]["enabled"] is False
    assert graph["roi_filter"]["enabled"] is True


def test_policy_enabled_parses_string_booleans():
    graph = {"open_set_gate": {"enabled": "false"}}
    assert policy_enabled(graph, "open_set_gate", default=True) is False


def test_resolve_requested_profile_prefers_env(monkeypatch):
    monkeypatch.setenv("AADS_ULORA_VLM_PROFILE", "fast")
    assert resolve_requested_profile({"profile": "balanced"}) == "fast"


def test_apply_runtime_profile_returns_changed_config():
    base = {
        "profile": "balanced",
        "profiles": {
            "fast": {
                "policy_graph": {
                    "crop_evidence": {"enabled": False},
                }
            }
        },
    }
    patched, active, changed = apply_runtime_profile(base, "fast")
    assert changed is True
    assert active == "fast"
    assert patched["profile"] == "fast"
    assert patched["policy_graph"]["crop_evidence"]["enabled"] is False


def test_apply_runtime_profile_handles_missing_profile():
    base = {"profiles": {"fast": {"x": 1}}}
    patched, active, changed = apply_runtime_profile(base, "balanced", suppress_warning=True)
    assert changed is False
    assert active is None
    assert patched == base


def test_load_taxonomy_and_compatibility_from_file(tmp_path):
    taxonomy = {
        "crops": ["tomato"],
        "common_weeds": ["weed_a"],
        "ornamentals": ["rose"],
        "parts": {"core": ["leaf"], "extended": ["fruit"]},
        "crop_part_compatibility": {"Tomato": ["Leaf", "Whole", "Entire Plant", "Fruit"], "bad": "skip"},
    }
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(json.dumps(taxonomy), encoding="utf-8")

    crops, parts = load_taxonomy(str(taxonomy_path))
    compatibility = load_crop_part_compatibility(str(taxonomy_path))

    assert crops == ["tomato", "weed_a", "rose"]
    assert parts == ["leaf", "fruit"]
    assert compatibility == {"tomato": ["leaf", "whole plant", "fruit"]}


def test_load_taxonomy_prefers_cwd_then_repo_root(tmp_path, monkeypatch):
    cwd_taxonomy = {
        "crops": ["alpha"],
        "parts": ["leaf"],
        "crop_part_compatibility": {"alpha": ["leaf"]},
    }
    cwd_path = tmp_path / "taxonomy.json"
    cwd_path.write_text(json.dumps(cwd_taxonomy), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    crops, parts = load_taxonomy("taxonomy.json")
    repo_crops, repo_parts = load_taxonomy("config/plant_taxonomy.json")

    assert crops == ["alpha"]
    assert parts == ["leaf"]
    assert "tomato" in repo_crops
    assert "leaf" in repo_parts


def test_load_taxonomy_raises_when_file_is_missing(tmp_path):
    missing_path = tmp_path / "missing_taxonomy.json"

    with pytest.raises(FileNotFoundError, match="Taxonomy file not found"):
        load_taxonomy(str(missing_path))
