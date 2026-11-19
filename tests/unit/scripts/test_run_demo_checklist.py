import json
from pathlib import Path

from PIL import Image

from scripts.run_demo_checklist import (
    ChecklistRow,
    _handoff_cache_key,
    _run_official_batch_rows,
    build_analysis_summary,
    parse_checklist_rows,
    parse_manifest_rows,
    summarize_results,
)


def test_parse_checklist_rows_reads_demo_table(tmp_path: Path):
    checklist = tmp_path / "demo.md"
    checklist.write_text(
        "\n".join(
            [
                "| image_id | source | expected_target | expected_behavior | actual_status | predicted_crop | "
                "predicted_part | predicted_disease | confidence_or_ood | pass_fail | failure_bucket | notes |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|",
                "| demo_xxx | internet | tomato__leaf | template row |  |  |  |  |  |  |  | ignore |",
                "",
                "## M1 Candidate Checklist",
                "",
                "| image_id | source | expected_target | expected_behavior | actual_status | predicted_crop | "
                "predicted_part | predicted_disease | confidence_or_ood | pass_fail | failure_bucket | notes |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|",
                "| demo_001 | local_test_pool:data/x | tomato__leaf | known disease |  |  |  |  |  |  |  | note |",
                "",
                "## Pass Criteria",
            ]
        ),
        encoding="utf-8",
    )

    rows = parse_checklist_rows(checklist)

    assert len(rows) == 1
    assert rows[0].image_id == "demo_001"
    assert rows[0].expected_target == "tomato__leaf"
    assert rows[0].notes == "note"


def test_parse_manifest_rows_reads_extra_csv(tmp_path: Path):
    manifest = tmp_path / "extra.csv"
    manifest.write_text(
        "\n".join(
            [
                "image_id,source,expected_target,expected_crop,expected_part,expected_class,expected_behavior,notes,origin_url",
                (
                    "demo_049,staged_external:.runtime_tmp/x.jpg,tomato__leaf,tomato,leaf,"
                    "domates_late_blight_yaprak,\"answer or review, no crash\",internet,http://example.test"
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = parse_manifest_rows(manifest)

    assert len(rows) == 1
    assert rows[0].image_id == "demo_049"
    assert rows[0].expected_target == "tomato__leaf"
    assert rows[0].expected_crop == "tomato"
    assert rows[0].expected_part == "leaf"
    assert rows[0].expected_class == "domates_late_blight_yaprak"
    assert rows[0].notes == "internet"


def test_official_batch_rows_reuses_batch_router_results(tmp_path: Path, monkeypatch):
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    rows = [
        ChecklistRow(
            image_id="demo_001",
            source="staged_external:a.jpg",
            expected_target="tomato__leaf",
            expected_behavior="answer",
            notes="",
            expected_crop="tomato",
            expected_part="leaf",
            expected_class="healthy",
        ),
        ChecklistRow(
            image_id="demo_002",
            source="staged_external:b.jpg",
            expected_target="grape__fruit",
            expected_behavior="answer",
            notes="",
            expected_crop="grape",
            expected_part="fruit",
            expected_class="healthy",
        ),
    ]
    batch_calls = []
    adapter_calls = []

    def fake_batch(image_paths, **_kwargs):
        batch_calls.append([Path(path).name for path in image_paths])
        return [
            {"status": "ok", "crop": "tomato", "part": "leaf", "router": {}},
            {"status": "ok", "crop": "grape", "part": "fruit", "router": {}},
        ]

    def fake_adapter(image_path, *, router_result, **_kwargs):
        adapter_calls.append((Path(image_path).name, router_result["crop"], router_result["part"]))
        return {
            "status": "success",
            "crop": router_result["crop"],
            "part": router_result["part"],
            "diagnosis": "healthy",
            "confidence": 0.9,
            "router_handoff": {"prototype_reconciliation": {}},
        }

    monkeypatch.setattr("scripts.run_demo_checklist.run_router_inference_batch", fake_batch)
    monkeypatch.setattr("scripts.run_demo_checklist.run_auto_router_adapter_prediction", fake_adapter)

    output_rows = _run_official_batch_rows(
        rows,
        repo_root=tmp_path,
        config_env="colab",
        device="cuda",
        adapter_root=tmp_path / "runs",
        batch_size=2,
    )

    assert batch_calls == [["a.jpg", "b.jpg"]]
    assert adapter_calls == [("a.jpg", "tomato", "leaf"), ("b.jpg", "grape", "fruit")]
    assert [row["pass_fail"] for row in output_rows] == ["pass", "pass"]


def test_official_batch_rows_blocks_cached_handoff_for_expected_unsupported_part(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "grape.jpg"
    Image.new("RGB", (16, 16), color="green").save(image_path)
    rows = [
        ChecklistRow(
            image_id="demo_044",
            source="staged_external:grape.jpg",
            expected_target="grape__unknown_part",
            expected_behavior="unsupported part",
            notes="mixed grape fruit/leaf should review",
            expected_crop="grape",
            expected_part="",
            expected_class="unsupported_part",
        )
    ]

    def unexpected_router(*_args, **_kwargs):
        raise AssertionError("router batch should not run on full handoff-cache hit")

    def unexpected_adapter(*_args, **_kwargs):
        raise AssertionError("adapter path must not run for expected unsupported-part rows")

    monkeypatch.setattr("scripts.run_demo_checklist.run_router_inference_batch", unexpected_router)
    monkeypatch.setattr("scripts.run_demo_checklist.run_auto_router_adapter_prediction", unexpected_adapter)

    cache_key = _handoff_cache_key(
        row=rows[0],
        image_path=image_path,
        config_env="colab",
        device="cuda",
        enable_prototype_reconciler=True,
        prototype_bank_path=None,
        taxonomy_registry_path=None,
        prototype_min_similarity=None,
        prototype_min_margin=None,
        prototype_min_negative_gap=None,
        prototype_target_policies=None,
        expected_target_id=rows[0].expected_target,
        expected_class_label=rows[0].expected_class,
    )
    cache = {
        "schema_version": "m2_router_prototype_handoff_cache.v1",
        "entries": {
            cache_key: {
                "image_id": "demo_044",
                "image": str(image_path),
                "handoff": {
                    "adapter_allowed": True,
                    "status": "ok",
                    "crop": "grape",
                    "part": "leaf",
                    "message": "",
                    "router_confidence": 1.0,
                    "router": {},
                    "rejection_status": None,
                    "rejection_message": None,
                    "prototype_reconciliation": {
                        "enabled": True,
                        "vlm_crop": "grape",
                        "vlm_part": "leaf",
                        "prototype_crop": "grape",
                        "prototype_part": "leaf",
                        "prototype_target": "grape__leaf",
                        "reconciled_crop": "grape",
                        "reconciled_part": "leaf",
                        "reconcile_decision": "accept_router",
                        "reason": "router_and_prototype_agree",
                    },
                },
            }
        },
        "stats": {},
    }

    output_rows = _run_official_batch_rows(
        rows,
        repo_root=tmp_path,
        config_env="colab",
        device="cuda",
        adapter_root=tmp_path / "runs",
        batch_size=2,
        enable_prototype_reconciler=True,
        handoff_cache=cache,
    )

    assert output_rows[0]["actual_status"] == "router_uncertain"
    assert output_rows[0]["predicted_crop"] == "grape"
    assert output_rows[0]["predicted_part"] == "leaf"
    assert output_rows[0]["predicted_disease"] is None
    assert output_rows[0]["pass_fail"] == "pass"
    assert output_rows[0]["reconcile_reason"] == "router_and_prototype_agree"
    assert cache["stats"]["hits"] == 1


def test_summarize_results_counts_buckets_and_targets():
    summary = summarize_results(
        [
            {
                "actual_status": "success",
                "pass_fail": "pass",
                "expected_target": "tomato__leaf",
                "failure_bucket": "",
            },
            {
                "actual_status": "router_unavailable",
                "pass_fail": "fail",
                "expected_target": "tomato__leaf",
                "failure_bucket": "dependency_access",
            },
        ]
    )

    assert json.loads(json.dumps(summary)) == {
        "total": 2,
        "passed": 1,
        "failed": 1,
        "answered": 1,
        "abstained_or_reviewed": 1,
        "asset_ready": 0,
        "failure_buckets": {"dependency_access": 1},
        "per_target": {"tomato__leaf": {"total": 2, "pass": 1, "fail": 1}},
    }


def test_build_analysis_summary_does_not_count_blocked_negative_handoff_as_false_accept():
    analysis = build_analysis_summary(
        [
            {
                "image_id": "demo_blocked",
                "actual_status": "router_uncertain",
                "pass_fail": "pass",
                "expected_target": "unknown_crop",
                "expected_crop": "unknown_crop",
                "expected_part": "unknown",
                "expected_class": "",
                "predicted_crop": "tomato",
                "predicted_part": "leaf",
                "predicted_disease": None,
                "reconcile_decision": "use_prototype",
            },
            {
                "image_id": "demo_accepted",
                "actual_status": "success",
                "pass_fail": "fail",
                "expected_target": "unknown_crop",
                "expected_crop": "unknown_crop",
                "expected_part": "unknown",
                "expected_class": "",
                "predicted_crop": "tomato",
                "predicted_part": "leaf",
                "predicted_disease": "domates_early_blight_yaprak",
                "reconcile_decision": "use_prototype",
            },
        ]
    )

    assert analysis["negative_false_accepts"]["count"] == 1
    assert analysis["negative_false_accepts"]["image_ids"] == ["demo_accepted"]
