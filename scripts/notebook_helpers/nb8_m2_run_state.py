"""Notebook 8 M2 run-state helpers.

This module keeps the Notebook 8 M2 cell focused on orchestration while preserving
the visible notebook variable names and run-state JSON contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

RUN_STATE_OPERATOR_OVERRIDE_NAMES = (
    "M2_RUN_PROBLEM_ONLY_DEMO",
    "M2_RUN_FULL_DEMO",
    "M2_OPEN_WORLD_ONLY",
    "M2_REFRESH_HANDOFF_CACHE",
    "M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION",
    "M2_BATCH_SIZE",
    "M2_ADAPTER_BATCH_SIZE",
    "M2_DEMO_MANIFEST",
    "M2_HANDOFF_CACHE",
    "M2_PROBLEM_ONLY_MANIFEST",
    "M2_PROBLEM_ONLY_CALIBRATION_MANIFEST",
    "M2_PROBLEM_ONLY_COMPARISON_BASELINE",
    "M2_COMPARISON_BASELINE",
    "M2_RUN_OPEN_WORLD_ROUTER_VALIDATION",
    "M2_OPEN_WORLD_BASELINE_SUMMARY",
    "M2_OPEN_WORLD_PROTOTYPE_ARTIFACT_DIR",
    "M2_PROTOTYPE_CURATION_ROOT",
)

RUN_STATE_BOOL_KEYS = {
    "M2_RUN_FULL_DEMO": "m2_run_full_demo",
    "M2_OPEN_WORLD_ONLY": "m2_open_world_only",
    "M2_REFRESH_HANDOFF_CACHE": "m2_refresh_handoff_cache",
    "M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION": "m2_reuse_existing_prototype_calibration",
    "M2_RUN_OPEN_WORLD_ROUTER_VALIDATION": "m2_run_open_world_router_validation",
}

RUN_STATE_INT_KEYS = {
    "M2_BATCH_SIZE": "m2_batch_size",
    "M2_ADAPTER_BATCH_SIZE": "m2_adapter_batch_size",
}

RUN_STATE_STR_KEYS = {
    "M2_DEMO_MANIFEST": "m2_demo_manifest",
    "M2_HANDOFF_CACHE": "m2_handoff_cache",
    "M2_PROBLEM_ONLY_MANIFEST": "m2_problem_only_manifest",
    "M2_PROBLEM_ONLY_CALIBRATION_MANIFEST": "m2_problem_only_calibration_manifest",
    "M2_PROBLEM_ONLY_COMPARISON_BASELINE": "m2_problem_only_comparison_baseline",
    "M2_COMPARISON_BASELINE": "m2_comparison_baseline",
    "M2_OPEN_WORLD_BASELINE_SUMMARY": "m2_open_world_baseline_summary",
    "M2_OPEN_WORLD_PROTOTYPE_ARTIFACT_DIR": "m2_open_world_prototype_artifact_dir",
}


def coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return fallback


def load_m2_run_state_config(path: str | Path, *, root: Path | None = None, print_fn: Callable[..., None] = print) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (root or Path.cwd()) / config_path
    if not config_path.is_file():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print_fn(f"[M2] Failed to read run-state config {config_path}: {exc}")
        return {}
    return payload if isinstance(payload, dict) else {}


def operator_override_names(initial_global_names: set[str]) -> set[str]:
    return {name for name in RUN_STATE_OPERATOR_OVERRIDE_NAMES if name in initial_global_names}


def has_m2_run_state_operator_override(name: str, *, force_run_state: bool, operator_overrides: set[str]) -> bool:
    return (not force_run_state) and name in operator_overrides


def apply_m2_run_state(
    settings: Mapping[str, Any],
    run_state: Mapping[str, Any],
    *,
    force_run_state: bool,
    operator_overrides: set[str],
) -> dict[str, Any]:
    updated = dict(settings)

    def overridden(name: str) -> bool:
        return has_m2_run_state_operator_override(
            name,
            force_run_state=force_run_state,
            operator_overrides=operator_overrides,
        )

    if not overridden("M2_RUN_PROBLEM_ONLY_DEMO"):
        mode = str(run_state.get("mode") or "").strip().lower()
        if mode in {"problem_only", "problem-only", "problem"}:
            updated["M2_RUN_PROBLEM_ONLY_DEMO"] = True
        elif mode in {"full", "full_manifest", "full-manifest"}:
            updated["M2_RUN_PROBLEM_ONLY_DEMO"] = False
        elif mode in {"open_world_only", "open-world-only", "open_world", "open-world"}:
            updated["M2_RUN_PROBLEM_ONLY_DEMO"] = False
            updated["M2_RUN_FULL_DEMO"] = False
            updated["M2_OPEN_WORLD_ONLY"] = True
            updated["M2_RUN_OPEN_WORLD_ROUTER_VALIDATION"] = True
        updated["M2_RUN_PROBLEM_ONLY_DEMO"] = coerce_bool(
            run_state.get("m2_run_problem_only_demo"),
            bool(updated["M2_RUN_PROBLEM_ONLY_DEMO"]),
        )
    for setting_name, run_state_name in RUN_STATE_BOOL_KEYS.items():
        if not overridden(setting_name):
            updated[setting_name] = coerce_bool(run_state.get(run_state_name), bool(updated[setting_name]))
    for setting_name, run_state_name in RUN_STATE_INT_KEYS.items():
        if not overridden(setting_name):
            updated[setting_name] = int(run_state.get(run_state_name) or updated[setting_name])
    for setting_name, run_state_name in RUN_STATE_STR_KEYS.items():
        if not overridden(setting_name):
            updated[setting_name] = str(run_state.get(run_state_name) or updated[setting_name])
    return updated


def format_m2_run_state_message(
    settings: Mapping[str, Any],
    *,
    force_run_state: bool,
    operator_overrides: set[str],
) -> str:
    mode = (
        "open_world_only"
        if settings["M2_OPEN_WORLD_ONLY"]
        else "problem_only"
        if settings["M2_RUN_PROBLEM_ONLY_DEMO"]
        else "full"
    )
    return (
        "[M2] Applied run-state config: "
        f"mode={mode}, "
        f"refresh_handoff_cache={settings['M2_REFRESH_HANDOFF_CACHE']}, "
        f"batch={settings['M2_BATCH_SIZE']}/{settings['M2_ADAPTER_BATCH_SIZE']}, "
        f"open_world_gate={settings['M2_RUN_OPEN_WORLD_ROUTER_VALIDATION']}, "
        f"force_run_state={force_run_state}, "
        f"operator_overrides={sorted(operator_overrides)}"
    )


def validate_balanced_manifest_request(settings: Mapping[str, Any], run_state: Mapping[str, Any]) -> None:
    expected_balanced_manifest = str(run_state.get("m2_balanced_demo_manifest") or "").strip()
    next_track = str(run_state.get("m2_next_recommended_track") or "").strip()
    if (
        next_track == "run_balanced_664_notebook8_demo"
        and settings["M2_RUN_FULL_DEMO"]
        and not settings["M2_RUN_PROBLEM_ONLY_DEMO"]
        and not settings["M2_OPEN_WORLD_ONLY"]
        and expected_balanced_manifest
        and settings["M2_DEMO_MANIFEST"] != expected_balanced_manifest
    ):
        raise RuntimeError(
            "[M2] Refusing to run stale Notebook 8 customer-demo manifest while run-state requests "
            f"the balanced 664-row manifest. Expected M2_DEMO_MANIFEST={expected_balanced_manifest!r}, "
            f"got {settings['M2_DEMO_MANIFEST']!r}. Pull latest/restart the Colab runtime or set M2_FORCE_RUN_STATE=True."
        )
