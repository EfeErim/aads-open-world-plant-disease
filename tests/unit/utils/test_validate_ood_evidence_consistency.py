import json
from pathlib import Path

from scripts.validate_ood_evidence_consistency import collect_readiness_files, validate_readiness_file


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_ready_real_ood_passes_with_sample_count(tmp_path: Path):
    readiness_path = tmp_path / "runs" / "tomato_leaf" / "production_readiness.json"
    _write_json(
        readiness_path,
        {
            "status": "ready",
            "passed": True,
            "ood_evidence_source": "real_ood_split",
            "missing_deployment_requirements": [],
            "ood_evidence": {"metrics": {"ood_samples": 12}},
        },
    )

    assert validate_readiness_file(readiness_path, min_real_ood_images=10) == []


def test_ready_fails_when_real_ood_sample_count_missing(tmp_path: Path):
    readiness_path = tmp_path / "runs" / "tomato_leaf" / "production_readiness.json"
    _write_json(
        readiness_path,
        {
            "status": "ready",
            "passed": True,
            "ood_evidence_source": "real_ood_split",
            "missing_deployment_requirements": [],
            "ood_evidence": {"metrics": {"ood_auroc": 0.95}},
        },
    )

    issues = validate_readiness_file(readiness_path, min_real_ood_images=10)

    assert [issue.code for issue in issues] == ["real_ood_sample_count_missing"]


def test_fallback_provisional_requires_benchmark_summary(tmp_path: Path):
    readiness_path = tmp_path / "runs" / "tomato_leaf" / "production_readiness.json"
    _write_json(
        readiness_path,
        {
            "status": "provisional",
            "passed": False,
            "ood_evidence_source": "held_out_benchmark",
            "missing_deployment_requirements": ["real_ood_evidence"],
        },
    )

    issues = validate_readiness_file(readiness_path, min_real_ood_images=10)

    assert [issue.code for issue in issues] == ["fallback_summary_missing"]


def test_fallback_provisional_passes_with_benchmark_summary(tmp_path: Path):
    readiness_path = tmp_path / "runs" / "tomato_leaf" / "production_readiness.json"
    _write_json(
        readiness_path,
        {
            "status": "provisional",
            "passed": False,
            "ood_evidence_source": "held_out_benchmark",
            "missing_deployment_requirements": ["real_ood_evidence"],
        },
    )
    _write_json(readiness_path.parent / "ood_benchmark" / "summary.json", {"status": "completed", "passed": True})

    assert validate_readiness_file(readiness_path, min_real_ood_images=10) == []


def test_collect_readiness_files_skips_run_index(tmp_path: Path):
    runs_root = tmp_path / "runs"
    valid = runs_root / "tomato_leaf" / "production_readiness.json"
    _write_json(valid, {"status": "not_ready"})
    _write_json(runs_root / "_index" / "production_readiness.json", {"status": "ready"})

    assert collect_readiness_files(runs_root) == [valid]
