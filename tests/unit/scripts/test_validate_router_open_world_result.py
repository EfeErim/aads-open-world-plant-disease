import csv
import json
from pathlib import Path

from scripts.validate_router_open_world_result import validate_result_dir


def _write_csv(path: Path, rows: list[dict[str, str]] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows or [])


def _write_result_dir(root: Path, *, status: str = "pass", negative_rows: int = 300, coverage: float = 0.85) -> None:
    for relative in (
        "supported_balanced_run.json",
        "supported_balanced_analysis.json",
        "open_world_run.json",
        "open_world_analysis.json",
    ):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("{}", encoding="utf-8")
    for relative in (
        "supported_balanced_run.md",
        "supported_balanced_analysis.md",
        "open_world_run.md",
        "open_world_analysis.md",
        "router_open_world_readiness.md",
    ):
        (root / relative).write_text("# report\n", encoding="utf-8")
    for relative in (
        "failures/wrong_supported_target_handoffs.csv",
        "failures/negative_false_accepts.csv",
        "failures/wrong_part_false_accepts.csv",
    ):
        _write_csv(root / relative)
    (root / "provenance").mkdir()
    (root / "provenance" / "supported_manifest.csv").write_text("image_id\n", encoding="utf-8")
    (root / "provenance" / "open_world_manifest.csv").write_text("image_id\n", encoding="utf-8")
    (root / "provenance" / "baseline_summary.json").write_text("{}", encoding="utf-8")
    (root / "provenance" / "prototype_bank.json").write_text("{}", encoding="utf-8")
    (root / "provenance" / "taxonomy_registry.json").write_text("{}", encoding="utf-8")
    (root / "provenance" / "router_prototype_calibration.json").write_text("{}", encoding="utf-8")
    readiness = {
        "status": status,
        "checks": {
            "open_world_min_rows": True,
            "manifest_valid": True,
            "wrong_supported_target_handoffs_zero": True,
            "negative_false_accepts_zero": True,
            "wrong_part_false_accepts_zero": True,
            "supported_route_coverage_min": True,
            "latency_not_regressed": True,
            "runner_exit_codes_zero": True,
        },
        "open_world": {
            "negative_row_count": negative_rows,
            "negative_false_accept_count": 0,
            "wrong_part_false_accept_count": 0,
        },
        "supported": {
            "route_coverage": coverage,
            "wrong_supported_target_handoff_count": 0,
        },
        "latency": {
            "candidate_p95_latency_ms": 100.0,
            "baseline_p95_latency_ms": 100.0,
        },
        "runner_exit_codes": {"supported": 0, "open_world": 0},
        "copied_provenance": {
            "supported_manifest": str(root / "provenance" / "supported_manifest.csv"),
            "open_world_manifest": str(root / "provenance" / "open_world_manifest.csv"),
            "baseline_summary": str(root / "provenance" / "baseline_summary.json"),
            "prototype_bank": str(root / "provenance" / "prototype_bank.json"),
            "taxonomy_registry": str(root / "provenance" / "taxonomy_registry.json"),
            "prototype_calibration_report": str(root / "provenance" / "router_prototype_calibration.json"),
        },
    }
    (root / "router_open_world_readiness.json").write_text(json.dumps(readiness), encoding="utf-8")


def test_validate_router_open_world_result_accepts_complete_passing_folder(tmp_path: Path):
    _write_result_dir(tmp_path)

    report = validate_result_dir(tmp_path)

    assert report["status"] == "pass"
    assert report["issue_count"] == 0


def test_validate_router_open_world_result_rejects_failure_rows(tmp_path: Path):
    _write_result_dir(tmp_path)
    _write_csv(tmp_path / "failures" / "negative_false_accepts.csv", [{"image_id": "n1", "reason": "accept"}])

    report = validate_result_dir(tmp_path)

    assert report["status"] == "fail"
    assert any(issue["code"] == "failure_csv_not_empty" for issue in report["issues"])


def test_validate_router_open_world_result_rejects_missing_required_file(tmp_path: Path):
    _write_result_dir(tmp_path)
    (tmp_path / "open_world_run.json").unlink()

    report = validate_result_dir(tmp_path)

    assert report["status"] == "fail"
    assert any(issue["code"] == "missing_required_file" for issue in report["issues"])


def test_validate_router_open_world_result_rejects_missing_latency(tmp_path: Path):
    _write_result_dir(tmp_path)
    readiness_path = tmp_path / "router_open_world_readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["latency"]["baseline_p95_latency_ms"] = None
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")

    report = validate_result_dir(tmp_path)

    assert report["status"] == "fail"
    assert any(issue["code"] == "baseline_latency_missing" for issue in report["issues"])


def test_validate_router_open_world_result_rejects_missing_production_provenance(tmp_path: Path):
    _write_result_dir(tmp_path)
    (tmp_path / "provenance" / "prototype_bank.json").unlink()

    report = validate_result_dir(tmp_path)

    assert report["status"] == "fail"
    assert any(issue["code"] == "missing_production_provenance" for issue in report["issues"])
