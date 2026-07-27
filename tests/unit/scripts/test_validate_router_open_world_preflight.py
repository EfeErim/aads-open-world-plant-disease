import json
from pathlib import Path

import scripts.validate_router_open_world_preflight as preflight


def _write_open_world_manifest(path: Path, image_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                (
                    "image_id,source,expected_target,expected_crop,expected_part,"
                    "expected_behavior,ood_slice,origin_url,notes"
                ),
                (
                    f"n000,staged_external:{image_path.as_posix()},unknown_crop,unknown,unknown,"
                    "abstain,unsupported_crop,http://example.test/o0,ok"
                ),
            ]
        ),
        encoding="utf-8",
    )


def _write_notebook(path: Path, *, open_world_enabled: bool, open_world_only: bool = False) -> None:
    value = "True" if open_world_enabled else "False"
    open_world_only_value = "True" if open_world_only else "False"
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [
                            f"M2_OPEN_WORLD_ONLY = {open_world_only_value}\n",
                            f"M2_RUN_OPEN_WORLD_ROUTER_VALIDATION = {value}\n",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_router_open_world_preflight_passes_for_prepared_launch(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(preflight, "REPO_ROOT", tmp_path)
    run_state = tmp_path / "run_state.json"
    baseline = tmp_path / "baseline.json"
    supported_manifest = tmp_path / "supported.csv"
    open_world_manifest = tmp_path / "open_world.csv"
    open_world_summary = tmp_path / "open_world_summary.json"
    notebook = tmp_path / "notebook.ipynb"
    artifact_dir = tmp_path / "open_world_artifacts"
    image = tmp_path / "ow.jpg"

    image.write_bytes(b"not-a-real-image-but-hashable")
    artifact_dir.mkdir()
    for artifact_name in ("prototype_bank.json", "taxonomy_registry.json", "router_prototype_calibration.json"):
        (artifact_dir / artifact_name).write_text("{}", encoding="utf-8")
    baseline.write_text(json.dumps({"runner_elapsed_seconds": 10.0, "summary": {"total": 5}}), encoding="utf-8")
    run_state.write_text(
        json.dumps(
            {
                "mode": "open_world_only",
                "m2_run_full_demo": False,
                "m2_open_world_only": True,
                "m2_run_problem_only_demo": False,
                "m2_run_open_world_router_validation": True,
                "m2_open_world_baseline_summary": baseline.name,
                "m2_open_world_prototype_artifact_dir": artifact_dir.name,
            }
        ),
        encoding="utf-8",
    )
    supported_manifest.write_text("image_id,source,expected_target\ns000,http://example.test/a,tomato__leaf\n", encoding="utf-8")
    _write_open_world_manifest(open_world_manifest, image)
    open_world_summary.write_text(json.dumps({"row_count": 1, "duplicate_hash_count": 0}), encoding="utf-8")
    _write_notebook(notebook, open_world_enabled=True, open_world_only=True)

    exit_code = preflight.main(
        [
            "--run-state",
            str(run_state),
            "--notebook",
            str(notebook),
            "--supported-manifest",
            str(supported_manifest),
            "--open-world-manifest",
            str(open_world_manifest),
            "--open-world-summary",
            str(open_world_summary),
            "--min-open-world-rows",
            "1",
            "--fail-on-invalid",
        ]
    )

    assert exit_code == 0


def test_router_open_world_preflight_fails_when_open_world_run_state_is_disabled(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(preflight, "REPO_ROOT", tmp_path)
    run_state = tmp_path / "run_state.json"
    baseline = tmp_path / "baseline.json"
    supported_manifest = tmp_path / "supported.csv"
    open_world_manifest = tmp_path / "open_world.csv"
    open_world_summary = tmp_path / "open_world_summary.json"
    notebook = tmp_path / "notebook.ipynb"
    image = tmp_path / "ow.jpg"

    image.write_bytes(b"hashable")
    baseline.write_text(json.dumps({"runner_elapsed_seconds": 10.0, "summary": {"total": 5}}), encoding="utf-8")
    run_state.write_text(
        json.dumps(
            {
                "mode": "full",
                "m2_run_problem_only_demo": False,
                "m2_run_open_world_router_validation": False,
                "m2_open_world_baseline_summary": baseline.name,
            }
        ),
        encoding="utf-8",
    )
    supported_manifest.write_text("image_id,source,expected_target\ns000,http://example.test/a,tomato__leaf\n", encoding="utf-8")
    _write_open_world_manifest(open_world_manifest, image)
    open_world_summary.write_text(json.dumps({"row_count": 1, "duplicate_hash_count": 0}), encoding="utf-8")
    _write_notebook(notebook, open_world_enabled=True)

    report = preflight.build_report(
        preflight.build_parser().parse_args(
            [
                "--run-state",
                str(run_state),
                "--notebook",
                str(notebook),
                "--supported-manifest",
                str(supported_manifest),
                "--open-world-manifest",
                str(open_world_manifest),
                "--open-world-summary",
                str(open_world_summary),
                "--min-open-world-rows",
                "1",
            ]
        )
    )

    assert report["status"] == "fail"
    assert "run_state_open_world_enabled" in {check["name"] for check in report["failed_checks"]}
