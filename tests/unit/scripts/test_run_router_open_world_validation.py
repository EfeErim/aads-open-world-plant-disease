import json
from pathlib import Path

import scripts.run_router_open_world_validation as runner


def _supported_rows():
    return [
        {
            "image_id": f"s{i:03d}",
            "expected_target": "tomato__leaf",
            "expected_crop": "tomato",
            "expected_part": "leaf",
            "actual_status": "success",
            "predicted_crop": "tomato",
            "predicted_part": "leaf",
        }
        for i in range(10)
    ]


def _open_world_rows():
    return [
        {
            "image_id": f"n{i:03d}",
            "expected_target": "unknown_crop",
            "expected_crop": "unknown",
            "expected_part": "unknown",
            "expected_behavior": "open-world negative; abstain or review expected",
            "ood_slice": "unsupported_crop",
            "actual_status": "router_uncertain",
            "predicted_crop": "",
            "predicted_part": "",
        }
        for i in range(3)
    ]


def test_router_open_world_validation_writes_combined_readiness(tmp_path: Path, monkeypatch):
    supported_manifest = tmp_path / "supported.csv"
    open_world_manifest = tmp_path / "open_world.csv"
    supported_manifest.write_text("image_id,source,expected_target\n", encoding="utf-8")
    open_world_manifest.write_text(
        "\n".join(
            [
                (
                    "image_id,source,expected_target,expected_crop,expected_part,"
                    "expected_behavior,ood_slice,origin_url,notes"
                ),
                (
                    "n000,http://example.test/0.jpg,unknown_crop,unknown,unknown,"
                    "abstain,unsupported_crop,http://example.test/o0,ok"
                ),
                (
                    "n001,http://example.test/1.jpg,unknown_crop,unknown,unknown,"
                    "abstain,unsupported_crop,http://example.test/o1,ok"
                ),
                (
                    "n002,http://example.test/2.jpg,unknown_crop,unknown,unknown,"
                    "abstain,unsupported_crop,http://example.test/o2,ok"
                ),
            ]
        ),
        encoding="utf-8",
    )

    def fake_run_demo_command(*, output_json, output_markdown, analysis_json, analysis_markdown, manifest, **_kwargs):
        rows = _open_world_rows() if manifest == open_world_manifest else _supported_rows()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps({"rows": rows}), encoding="utf-8")
        output_markdown.write_text("# report\n", encoding="utf-8")
        analysis_json.write_text("{}", encoding="utf-8")
        analysis_markdown.write_text("# analysis\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "_run_demo_command", fake_run_demo_command)

    exit_code = runner.main(
        [
            "--run-id",
            "unit",
            "--output-root",
            str(tmp_path / "results"),
            "--supported-manifest",
            str(supported_manifest),
            "--open-world-manifest",
            str(open_world_manifest),
            "--min-open-world-rows",
            "3",
            "--baseline-p95-latency-ms",
            "100",
            "--candidate-p95-latency-ms",
            "100",
            "--require-latency-baseline",
            "--fail-on-not-ready",
        ]
    )

    readiness = json.loads(
        (tmp_path / "results" / "unit" / "router_open_world_readiness.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert readiness["status"] == "pass"
    assert readiness["runner_exit_codes"] == {"supported": 0, "open_world": 0}
    assert (tmp_path / "results" / "unit" / "failures" / "negative_false_accepts.csv").is_file()


def test_router_open_world_validation_resolves_artifacts_and_latency_baseline(tmp_path: Path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    for name in runner.PROTOTYPE_ARTIFACT_FILES.values():
        (artifact_dir / name).write_text("{}", encoding="utf-8")
    baseline = tmp_path / "baseline_summary.json"
    baseline.write_text(
        json.dumps({"runner_elapsed_seconds": 20.0, "summary": {"total": 10}}),
        encoding="utf-8",
    )
    supported_manifest = tmp_path / "supported.csv"
    open_world_manifest = tmp_path / "open_world.csv"
    supported_manifest.write_text("image_id,source,expected_target\n", encoding="utf-8")
    open_world_manifest.write_text(
        "\n".join(
            [
                (
                    "image_id,source,expected_target,expected_crop,expected_part,"
                    "expected_behavior,ood_slice,origin_url,notes"
                ),
                (
                    "n000,http://example.test/0.jpg,unknown_crop,unknown,unknown,"
                    "abstain,unsupported_crop,http://example.test/o0,ok"
                ),
            ]
        ),
        encoding="utf-8",
    )
    seen_args = []

    def fake_run_demo_command(*, output_json, output_markdown, analysis_json, analysis_markdown, manifest, args):
        seen_args.append(args)
        rows = _open_world_rows()[:1] if manifest == open_world_manifest else _supported_rows()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps({"rows": rows}), encoding="utf-8")
        output_markdown.write_text("# report\n", encoding="utf-8")
        analysis_json.write_text("{}", encoding="utf-8")
        analysis_markdown.write_text("# analysis\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "_run_demo_command", fake_run_demo_command)

    exit_code = runner.main(
        [
            "--run-id",
            "unit_artifacts",
            "--output-root",
            str(tmp_path / "results"),
            "--supported-manifest",
            str(supported_manifest),
            "--open-world-manifest",
            str(open_world_manifest),
            "--min-open-world-rows",
            "1",
            "--baseline-summary",
            str(baseline),
            "--candidate-p95-latency-ms",
            "2000",
            "--require-latency-baseline",
            "--prototype-artifact-dir",
            str(artifact_dir),
            "--enable-prototype-reconciler",
            "--fail-on-not-ready",
        ]
    )

    readiness = json.loads(
        (tmp_path / "results" / "unit_artifacts" / "router_open_world_readiness.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert readiness["latency"]["baseline_p95_latency_ms"] == 2000.0
    assert readiness["resolved_inputs"]["prototype_bank"].endswith("prototype_bank.json")
    assert readiness["copied_provenance"]["prototype_bank"].endswith("provenance/prototype_bank.json")
    assert (tmp_path / "results" / "unit_artifacts" / "provenance" / "open_world_manifest.csv").is_file()
    assert seen_args[0].resolved_taxonomy_registry == artifact_dir / "taxonomy_registry.json"


def test_router_open_world_validation_writes_failed_readiness_when_reports_are_missing(
    tmp_path: Path,
    monkeypatch,
):
    supported_manifest = tmp_path / "supported.csv"
    open_world_manifest = tmp_path / "open_world.csv"
    supported_manifest.write_text("image_id,source,expected_target\n", encoding="utf-8")
    open_world_manifest.write_text(
        "\n".join(
            [
                (
                    "image_id,source,expected_target,expected_crop,expected_part,"
                    "expected_behavior,ood_slice,origin_url,notes"
                ),
                (
                    "n000,http://example.test/0.jpg,unknown_crop,unknown,unknown,"
                    "abstain,unsupported_crop,http://example.test/o0,ok"
                ),
            ]
        ),
        encoding="utf-8",
    )

    def fake_run_demo_command(**_kwargs):
        return 1

    monkeypatch.setattr(runner, "_run_demo_command", fake_run_demo_command)

    exit_code = runner.main(
        [
            "--run-id",
            "unit_missing_reports",
            "--output-root",
            str(tmp_path / "results"),
            "--supported-manifest",
            str(supported_manifest),
            "--open-world-manifest",
            str(open_world_manifest),
            "--min-open-world-rows",
            "1",
            "--baseline-p95-latency-ms",
            "100",
            "--candidate-p95-latency-ms",
            "100",
            "--fail-on-not-ready",
        ]
    )

    result_dir = tmp_path / "results" / "unit_missing_reports"
    readiness = json.loads((result_dir / "router_open_world_readiness.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert readiness["status"] == "fail"
    assert readiness["checks"]["supported_report_written"] is False
    assert readiness["checks"]["open_world_report_written"] is False
    assert "supported_report_missing" in readiness["warnings"]
    assert "open_world_report_missing" in readiness["warnings"]
    assert (result_dir / "router_open_world_readiness.md").is_file()
    assert (result_dir / "failures" / "negative_false_accepts.csv").is_file()
