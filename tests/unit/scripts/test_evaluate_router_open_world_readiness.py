import json
from pathlib import Path

from scripts.evaluate_router_open_world_readiness import (
    evaluate_readiness,
    main,
    validate_open_world_manifest,
)


def _supported_row(
    image_id: str,
    *,
    expected_target: str = "tomato__leaf",
    actual_status: str = "success",
    predicted_crop: str = "tomato",
    predicted_part: str = "leaf",
):
    expected_crop, expected_part = expected_target.split("__", 1)
    return {
        "image_id": image_id,
        "expected_target": expected_target,
        "expected_crop": expected_crop,
        "expected_part": expected_part,
        "actual_status": actual_status,
        "predicted_crop": predicted_crop,
        "predicted_part": predicted_part,
        "predicted_disease": "healthy",
    }


def _negative_row(
    image_id: str,
    *,
    actual_status: str = "router_uncertain",
    expected_target: str = "unknown_crop",
    expected_crop: str = "unknown",
    expected_part: str = "unknown",
    predicted_crop: str = "",
    predicted_part: str = "",
    ood_slice: str = "non_plant",
):
    return {
        "image_id": image_id,
        "expected_target": expected_target,
        "expected_crop": expected_crop,
        "expected_part": expected_part,
        "expected_behavior": "abstain or review",
        "ood_slice": ood_slice,
        "actual_status": actual_status,
        "predicted_crop": predicted_crop,
        "predicted_part": predicted_part,
        "predicted_disease": "healthy" if actual_status == "success" else "",
    }


def _payload(rows):
    return {"rows": rows, "summary": {"total": len(rows)}}


def test_readiness_passes_for_zero_false_accepts_and_enough_supported_coverage():
    report = evaluate_readiness(
        supported_payload=_payload([_supported_row(f"s{i:03d}") for i in range(8)] + [
            _supported_row("s_review_1", actual_status="router_uncertain", predicted_crop="", predicted_part=""),
            _supported_row("s_review_2", actual_status="router_uncertain", predicted_crop="", predicted_part=""),
        ]),
        open_world_payload=_payload([_negative_row(f"n{i:03d}") for i in range(300)]),
        min_open_world_rows=300,
        baseline_p95_latency_ms=100.0,
        candidate_p95_latency_ms=120.0,
        require_latency_baseline=True,
    )

    assert report["status"] == "pass"
    assert report["supported"]["route_coverage"] == 0.8
    assert report["open_world"]["negative_false_accept_count"] == 0
    assert 0.009 < report["open_world"]["zero_failure_95_upper_bound"] < 0.011
    assert report["checks"]["latency_not_regressed"] is True


def test_readiness_fails_on_one_negative_false_accept():
    report = evaluate_readiness(
        supported_payload=_payload([_supported_row(f"s{i:03d}") for i in range(10)]),
        open_world_payload=_payload([_negative_row(f"n{i:03d}") for i in range(299)] + [
            _negative_row("n_accept", actual_status="success", predicted_crop="tomato", predicted_part="leaf"),
        ]),
        min_open_world_rows=300,
        baseline_p95_latency_ms=100.0,
        candidate_p95_latency_ms=100.0,
        require_latency_baseline=True,
    )

    assert report["status"] == "fail"
    assert report["checks"]["negative_false_accepts_zero"] is False
    assert report["open_world"]["negative_false_accept_count"] == 1
    assert report["failures"]["negative_false_accepts"][0]["image_id"] == "n_accept"


def test_readiness_fails_on_wrong_supported_target_handoff():
    report = evaluate_readiness(
        supported_payload=_payload(
            [_supported_row(f"s{i:03d}") for i in range(9)]
            + [_supported_row("s_wrong", predicted_crop="grape", predicted_part="leaf")]
        ),
        open_world_payload=_payload([_negative_row(f"n{i:03d}") for i in range(300)]),
        baseline_p95_latency_ms=100.0,
        candidate_p95_latency_ms=100.0,
        require_latency_baseline=True,
    )

    assert report["status"] == "fail"
    assert report["checks"]["wrong_supported_target_handoffs_zero"] is False
    assert report["supported"]["wrong_supported_target_handoff_count"] == 1


def test_readiness_fails_when_supported_route_coverage_below_floor():
    report = evaluate_readiness(
        supported_payload=_payload(
            [_supported_row(f"s{i:03d}") for i in range(7)]
            + [
                _supported_row("s_review_1", actual_status="router_uncertain", predicted_crop="", predicted_part=""),
                _supported_row("s_review_2", actual_status="router_uncertain", predicted_crop="", predicted_part=""),
                _supported_row("s_review_3", actual_status="router_uncertain", predicted_crop="", predicted_part=""),
            ]
        ),
        open_world_payload=_payload([_negative_row(f"n{i:03d}") for i in range(300)]),
        baseline_p95_latency_ms=100.0,
        candidate_p95_latency_ms=100.0,
        require_latency_baseline=True,
    )

    assert report["status"] == "fail"
    assert report["checks"]["supported_route_coverage_min"] is False
    assert report["supported"]["route_coverage"] == 0.7


def test_readiness_fails_on_latency_regression_when_required():
    report = evaluate_readiness(
        supported_payload=_payload([_supported_row(f"s{i:03d}") for i in range(10)]),
        open_world_payload=_payload([_negative_row(f"n{i:03d}") for i in range(300)]),
        baseline_p95_latency_ms=100.0,
        candidate_p95_latency_ms=126.0,
        require_latency_baseline=True,
    )

    assert report["status"] == "fail"
    assert report["checks"]["latency_not_regressed"] is False
    assert report["latency"]["limit_p95_latency_ms"] == 125.0


def test_readiness_derives_candidate_latency_from_elapsed_time_when_p95_missing():
    report = evaluate_readiness(
        supported_payload=_payload([_supported_row(f"s{i:03d}") for i in range(10)]),
        open_world_payload={
            "elapsed_seconds": 30.0,
            "summary": {"total": 300},
            "rows": [_negative_row(f"n{i:03d}") for i in range(300)],
        },
        baseline_p95_latency_ms=200.0,
        require_latency_baseline=True,
    )

    assert report["status"] == "pass"
    assert report["latency"]["candidate_p95_latency_ms"] == 100.0


def test_manifest_audit_rejects_missing_provenance_and_duplicate_local_hashes(tmp_path: Path):
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_a.write_bytes(b"same")
    image_b.write_bytes(b"same")
    manifest = tmp_path / "open_world.csv"
    manifest.write_text(
        "\n".join(
            [
                (
                    "image_id,source,expected_target,expected_crop,expected_part,"
                    "expected_behavior,ood_slice,origin_url,notes"
                ),
                (
                    f"n001,staged_external:{image_a},unknown_crop,unknown,unknown,"
                    "abstain,non_plant,http://example.test/a,ok"
                ),
                (
                    f"n002,staged_external:{image_b},unknown_crop,unknown,unknown,"
                    "abstain,non_plant,http://example.test/b,"
                ),
            ]
        ),
        encoding="utf-8",
    )

    audit = validate_open_world_manifest(manifest, repo_root=tmp_path, min_rows=2)

    assert audit["status"] == "fail"
    assert any(issue["code"] == "missing_provenance_notes" for issue in audit["issues"])
    assert any(issue["code"] == "duplicate_local_sha256" for issue in audit["issues"])


def test_manifest_audit_rejects_disjoint_root_hash_overlap(tmp_path: Path):
    image = tmp_path / "open.jpg"
    overlap = tmp_path / "prior" / "open.jpg"
    overlap.parent.mkdir()
    image.write_bytes(b"same")
    overlap.write_bytes(b"same")
    manifest = tmp_path / "open_world.csv"
    manifest.write_text(
        "\n".join(
            [
                (
                    "image_id,source,expected_target,expected_crop,expected_part,"
                    "expected_behavior,ood_slice,origin_url,notes"
                ),
                (
                    f"n001,staged_external:{image},unknown_crop,unknown,unknown,"
                    "abstain,non_plant,http://example.test/a,ok"
                ),
            ]
        ),
        encoding="utf-8",
    )

    audit = validate_open_world_manifest(
        manifest,
        repo_root=tmp_path,
        min_rows=1,
        disjoint_roots=[tmp_path / "prior"],
    )

    assert audit["status"] == "fail"
    assert audit["disjoint_overlap_count"] == 1
    assert any(issue["code"] == "disjoint_sha256_overlap" for issue in audit["issues"])


def test_cli_writes_readiness_outputs(tmp_path: Path):
    supported = tmp_path / "supported.json"
    open_world = tmp_path / "open_world.json"
    output = tmp_path / "router_open_world_readiness.json"
    markdown = tmp_path / "router_open_world_readiness.md"
    failures = tmp_path / "failures"
    supported.write_text(json.dumps(_payload([_supported_row(f"s{i:03d}") for i in range(10)])), encoding="utf-8")
    open_world.write_text(json.dumps(_payload([_negative_row(f"n{i:03d}") for i in range(300)])), encoding="utf-8")

    exit_code = main(
        [
            "--supported-report",
            str(supported),
            "--open-world-report",
            str(open_world),
            "--baseline-p95-latency-ms",
            "100",
            "--candidate-p95-latency-ms",
            "100",
            "--require-latency-baseline",
            "--output",
            str(output),
            "--markdown-output",
            str(markdown),
            "--failure-dir",
            str(failures),
            "--fail-on-not-ready",
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
    assert "# Router Open-World Readiness" in markdown.read_text(encoding="utf-8")
    assert (failures / "negative_false_accepts.csv").is_file()
