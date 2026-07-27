import json
from pathlib import Path

from PIL import Image

from scripts.build_m2_supported_disease_manifest import build_rows, is_healthy_class
from scripts.run_demo_checklist import (
    ChecklistRow,
    _handoff_cache_key,
    _path_fingerprint,
    _run_official_batch_rows,
    build_analysis_summary,
    build_parser,
    classify_failure,
    evaluate_pass,
    format_elapsed_seconds,
    parse_checklist_rows,
    parse_manifest_rows,
    resolve_image_path,
    resolve_prototype_thresholds_from_calibration,
    summarize_results,
    write_markdown_report,
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


def test_resolve_image_path_uses_first_sorted_local_image(tmp_path: Path):
    image_dir = tmp_path / "data" / "x"
    image_dir.mkdir(parents=True)
    (image_dir / "b.png").write_bytes(b"b")
    (image_dir / "a.jpg").write_bytes(b"a")

    image, status = resolve_image_path("local_test_pool:data/x", tmp_path)

    assert status == "ok"
    assert image == image_dir / "a.jpg"


def test_resolve_image_path_matches_ascii_source_to_unicode_folder(tmp_path: Path):
    image_dir = tmp_path / "data" / "test" / "üzüm_sağlıklı_meyve"
    image_dir.mkdir(parents=True)
    (image_dir / "image.png").write_bytes(b"x")

    image, status = resolve_image_path("local_test_pool:data/test/uzum_saglikli_meyve", tmp_path)

    assert status == "ok"
    assert image == image_dir / "image.png"


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


def test_parse_manifest_rows_infers_expected_fields_from_existing_manifest_shape(tmp_path: Path):
    manifest = tmp_path / "extra.csv"
    manifest.write_text(
        "\n".join(
            [
                "image_id,source,expected_target,expected_behavior,notes,original_source",
                (
                    "demo_049,staged_external:.runtime_tmp/x.jpg,tomato__fruit,"
                    "\"answer or review, no crash\",internet,"
                    "local_test_pool:data/prepared_runtime_datasets/tomato__fruit/test/domates_late_blight_meyve/x.jpg"
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = parse_manifest_rows(manifest)

    assert rows[0].expected_crop == "tomato"
    assert rows[0].expected_part == "fruit"
    assert rows[0].expected_class == "domates_late_blight_meyve"


def test_parse_manifest_rows_does_not_infer_expected_class_from_external_url_parent(tmp_path: Path):
    manifest = tmp_path / "extra.csv"
    manifest.write_text(
        "\n".join(
            [
                "image_id,source,expected_target,expected_crop,expected_part,expected_class,"
                "expected_behavior,notes,original_source,disease_class",
                (
                    "demo_525,staged_external:docs/demo_assets/m2_full_image_set/images/demo_525_tomato_fruit.jpg,"
                    "tomato__fruit,tomato,fruit,,"
                    "external supported crop/part image; disease answer or review expected,"
                    "external iNaturalist enrichment,https://www.inaturalist.org/observations/366091936,"
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = parse_manifest_rows(manifest)

    assert rows[0].expected_class == ""


def test_evaluate_classless_supported_probe_accepts_review_or_correct_target_answer():
    row = ChecklistRow(
        image_id="demo_525",
        source="staged_external:docs/demo_assets/m2_full_image_set/images/demo_525_tomato_fruit.jpg",
        expected_target="tomato__fruit",
        expected_behavior="external supported crop/part image; disease answer or review expected",
        notes="",
        expected_crop="tomato",
        expected_part="fruit",
        expected_class="",
    )

    assert evaluate_pass(row, {"status": "router_uncertain", "crop": None, "part": None}, asset_status="ok") == "pass"
    assert evaluate_pass(
        row,
        {"status": "success", "crop": "tomato", "part": "fruit", "diagnosis": "domates_saglikli_meyve"},
        asset_status="ok",
    ) == "pass"
    assert evaluate_pass(
        row,
        {"status": "success", "crop": "tomato", "part": "leaf", "diagnosis": "domates_late_blight_yaprak"},
        asset_status="ok",
    ) == "fail"


def test_parser_supports_manifest_only_runs():
    args = build_parser().parse_args(["--no-checklist", "--extra-manifest", "manifest.csv"])

    assert args.no_checklist is True
    assert [str(path) for path in args.extra_manifest] == ["manifest.csv"]


def test_parser_supports_official_batch_size():
    args = build_parser().parse_args(
        ["--no-checklist", "--extra-manifest", "manifest.csv", "--batch-size", "4", "--adapter-batch-size", "8"]
    )

    assert args.batch_size == 4
    assert args.adapter_batch_size == 8


def test_path_fingerprint_uses_repo_relative_paths_for_repo_files():
    fingerprint = _path_fingerprint(Path("scripts/run_demo_checklist.py"))

    assert fingerprint["exists"] is True
    assert fingerprint["path"] == "scripts/run_demo_checklist.py"


def test_format_elapsed_seconds_uses_human_readable_units():
    assert format_elapsed_seconds(4.4) == "4s"
    assert format_elapsed_seconds(65.2) == "1m 5s"
    assert format_elapsed_seconds(3725.0) == "1h 2m 5s"


def _minimal_report() -> dict:
    return {
        "started_at": "2026-06-23T10:00:00+00:00",
        "finished_at": "2026-06-23T10:01:05+00:00",
        "elapsed_seconds": 65.2,
        "elapsed_human": "1m 5s",
        "generated_at": "2026-06-23T10:01:05+00:00",
        "checklist": "docs/demo_checklist.md",
        "device": "cuda",
        "adapter_root": "runs",
        "mode": "official",
        "summary": {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "answered": 1,
            "abstained_or_reviewed": 0,
            "asset_ready": 1,
            "failure_buckets": {},
        },
        "rows": [
            {
                "image_id": "demo_001",
                "actual_status": "success",
                "pass_fail": "pass",
                "failure_bucket": "",
                "predicted_crop": "tomato",
                "predicted_part": "leaf",
                "predicted_disease": "healthy",
                "message": "",
            }
        ],
    }


def test_markdown_report_includes_run_timing(tmp_path: Path):
    output = tmp_path / "report.md"
    write_markdown_report(_minimal_report(), output)

    text = output.read_text(encoding="utf-8")
    assert "- started_at: `2026-06-23T10:00:00+00:00`" in text
    assert "- finished_at: `2026-06-23T10:01:05+00:00`" in text
    assert "- elapsed: `1m 5s` (65.200s)" in text

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


def test_official_batch_rows_can_batch_adapter_predictions(tmp_path: Path, monkeypatch):
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    Image.new("RGB", (16, 16), color="green").save(image_a)
    Image.new("RGB", (16, 16), color="green").save(image_b)
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
            expected_target="tomato__leaf",
            expected_behavior="answer",
            notes="",
            expected_crop="tomato",
            expected_part="leaf",
            expected_class="healthy",
        ),
    ]
    router_calls = []
    adapter_batch_sizes = []

    def fake_batch(image_paths, **_kwargs):
        router_calls.append([Path(path).name for path in image_paths])
        return [
            {"status": "ok", "crop": "tomato", "part": "leaf", "router": {}},
            {"status": "ok", "crop": "tomato", "part": "leaf", "router": {}},
        ]

    class FakeAdapter:
        def predict_batch_with_ood(self, images):
            adapter_batch_sizes.append(int(images.shape[0]))
            return [
                {
                    "status": "success",
                    "disease": {"class_index": 0, "name": "healthy", "confidence": 0.9},
                    "ood_analysis": {
                        "score_method": "ensemble",
                        "primary_score": 0.1,
                        "decision_threshold": 0.5,
                        "is_ood": False,
                        "calibration_version": 1,
                    },
                }
                for _ in range(int(images.shape[0]))
            ]

    class FakeRuntime:
        input_guard_enabled = False
        target_size = 8

        def load_adapter(self, crop, *, part_name=None):
            assert (crop, part_name) == ("tomato", "leaf")
            return FakeAdapter()

        def _coerce_image(self, image):
            return Image.open(image).convert("RGB")

    class FakeWorkflow:
        def __init__(self, **_kwargs):
            self.runtime = FakeRuntime()

    def unexpected_single_adapter(*_args, **_kwargs):
        raise AssertionError("single-row adapter path should not run")

    monkeypatch.setattr("scripts.run_demo_checklist.run_router_inference_batch", fake_batch)
    monkeypatch.setattr("scripts.run_demo_checklist.run_auto_router_adapter_prediction", unexpected_single_adapter)
    monkeypatch.setattr("scripts.run_demo_checklist.InferenceWorkflow", FakeWorkflow)

    output_rows = _run_official_batch_rows(
        rows,
        repo_root=tmp_path,
        config_env="colab",
        device="cuda",
        adapter_root=tmp_path / "runs",
        batch_size=2,
        adapter_batch_size=8,
    )

    assert router_calls == [["a.jpg", "b.jpg"]]
    assert adapter_batch_sizes == [2]
    assert [row["pass_fail"] for row in output_rows] == ["pass", "pass"]


def test_official_batch_rows_skips_router_on_handoff_cache_hit(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "a.jpg"
    Image.new("RGB", (16, 16), color="green").save(image_path)
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
        )
    ]

    def unexpected_router(*_args, **_kwargs):
        raise AssertionError("router batch should not run on full handoff-cache hit")

    def fake_adapter(image_path, *, handoff_result, **_kwargs):
        assert handoff_result["crop"] == "tomato"
        return {
            "status": "success",
            "crop": "tomato",
            "part": "leaf",
            "diagnosis": "healthy",
            "confidence": 0.9,
            "router_handoff": {"prototype_reconciliation": handoff_result["prototype_reconciliation"]},
        }

    monkeypatch.setattr("scripts.run_demo_checklist.run_router_inference_batch", unexpected_router)
    monkeypatch.setattr("scripts.run_demo_checklist.run_auto_router_adapter_prediction", fake_adapter)

    cache_key = _handoff_cache_key(
        row=rows[0],
        image_path=image_path,
        config_env="colab",
        device="cuda",
        enable_prototype_reconciler=False,
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
                "image_id": "demo_001",
                "image": str(image_path),
                "handoff": {
                    "adapter_allowed": True,
                    "status": "ok",
                    "crop": "tomato",
                    "part": "leaf",
                    "message": "",
                    "router_confidence": 1.0,
                    "router": {},
                    "rejection_status": None,
                    "rejection_message": None,
                    "prototype_reconciliation": {
                        "enabled": False,
                        "vlm_crop": "tomato",
                        "vlm_part": "leaf",
                        "reconciled_crop": "tomato",
                        "reconciled_part": "leaf",
                        "reconcile_decision": "disabled",
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
        handoff_cache=cache,
    )
    assert output_rows[0]["pass_fail"] == "pass"
    assert cache["stats"]["hits"] == 1


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


def test_official_batch_rows_blocks_classless_probe_handoff_target_mismatch(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "tomato_leaf.jpg"
    Image.new("RGB", (16, 16), color="green").save(image_path)
    rows = [
        ChecklistRow(
            image_id="demo_535",
            source="staged_external:tomato_leaf.jpg",
            expected_target="tomato__leaf",
            expected_behavior="external supported crop/part image; disease answer or review expected",
            notes="external iNaturalist enrichment",
            expected_crop="tomato",
            expected_part="leaf",
            expected_class="",
        )
    ]

    def unexpected_router(*_args, **_kwargs):
        raise AssertionError("router batch should not run on full handoff-cache hit")

    def unexpected_adapter(*_args, **_kwargs):
        raise AssertionError("adapter path must not run for mismatched classless probe handoff")

    class FakeWorkflow:
        class Runtime:
            input_guard_enabled = False

        runtime = Runtime()

    monkeypatch.setattr("scripts.run_demo_checklist.run_router_inference_batch", unexpected_router)
    monkeypatch.setattr("scripts.run_demo_checklist.run_auto_router_adapter_prediction", unexpected_adapter)
    monkeypatch.setattr("scripts.run_demo_checklist.InferenceWorkflow", lambda **_kwargs: FakeWorkflow())

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
                "image_id": "demo_535",
                "image": str(image_path),
                "handoff": {
                    "adapter_allowed": True,
                    "status": "ok",
                    "crop": "tomato",
                    "part": "fruit",
                    "message": "",
                    "router_confidence": 1.0,
                    "router": {},
                    "rejection_status": None,
                    "rejection_message": None,
                    "prototype_reconciliation": {
                        "enabled": True,
                        "vlm_crop": "tomato",
                        "vlm_part": "fruit",
                        "prototype_crop": "tomato",
                        "prototype_part": "fruit",
                        "prototype_target": "tomato__fruit",
                        "reconciled_crop": "tomato",
                        "reconciled_part": "fruit",
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
        adapter_batch_size=8,
        enable_prototype_reconciler=True,
        handoff_cache=cache,
    )

    assert output_rows[0]["actual_status"] == "router_uncertain"
    assert output_rows[0]["predicted_crop"] == "tomato"
    assert output_rows[0]["predicted_part"] == "fruit"
    assert output_rows[0]["predicted_disease"] is None
    assert output_rows[0]["pass_fail"] == "pass"
    assert output_rows[0]["reconcile_reason"] == "router_and_prototype_agree"
    assert cache["stats"]["hits"] == 1


def test_resolve_prototype_thresholds_from_calibration_uses_selected_policy(tmp_path: Path):
    report_path = tmp_path / "calibration.json"
    report_path.write_text(
        json.dumps(
            {
                "selected_policy": {
                    "min_similarity": 0.4,
                    "min_margin": 0.08,
                    "min_negative_gap": 0.02,
                    "precision": 0.95,
                    "coverage": 0.7,
                },
                "target_policies": {"tomato__leaf": {"status": "target_specific"}},
            }
        ),
        encoding="utf-8",
    )

    min_similarity, min_margin, min_negative_gap, report, target_policies = (
        resolve_prototype_thresholds_from_calibration(
            report_path,
            min_similarity=None,
            min_margin=None,
        )
    )

    assert min_similarity == 0.4
    assert min_margin == 0.08
    assert min_negative_gap == 0.02
    assert report["policy_selected"] is True
    assert report["selected_policy"]["precision"] == 0.95
    assert target_policies["tomato__leaf"]["status"] == "target_specific"


def test_resolve_prototype_thresholds_preserves_explicit_values(tmp_path: Path):
    report_path = tmp_path / "calibration.json"
    report_path.write_text(json.dumps({"selected_policy": {"min_similarity": 0.4, "min_margin": 0.08}}), encoding="utf-8")

    min_similarity, min_margin, min_negative_gap, _report, target_policies = resolve_prototype_thresholds_from_calibration(
        report_path,
        min_similarity=0.6,
        min_margin=0.1,
        min_negative_gap=0.03,
    )

    assert min_similarity == 0.6
    assert min_margin == 0.1
    assert min_negative_gap == 0.03
    assert target_policies == {}


def test_resolve_prototype_thresholds_preserves_explicit_negative_gap_without_report():
    min_similarity, min_margin, min_negative_gap, report, target_policies = resolve_prototype_thresholds_from_calibration(
        None,
        min_similarity=0.6,
        min_margin=0.1,
        min_negative_gap=0.04,
    )

    assert min_similarity == 0.6
    assert min_margin == 0.1
    assert min_negative_gap == 0.04
    assert report == {"enabled": False}
    assert target_policies == {}


def test_supported_disease_manifest_excludes_healthy_classes(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    disease_dir = dataset_root / "tomato__leaf" / "test" / "domates_late_blight_yaprak"
    healthy_dir = dataset_root / "tomato__leaf" / "test" / "domates_sağlıklı_yaprak"
    disease_dir.mkdir(parents=True)
    healthy_dir.mkdir(parents=True)
    (disease_dir / "image.png").write_bytes(b"x")
    (healthy_dir / "image.png").write_bytes(b"x")

    for idx in range(1, 12):
        (disease_dir / f"image_{idx:02d}.png").write_bytes(b"x")

    rows = build_rows(dataset_root, start_id=145, images_per_class=10)

    assert is_healthy_class("domates_sağlıklı_yaprak")
    assert len(rows) == 10
    assert rows[0]["image_id"] == "demo_145"
    assert rows[0]["expected_target"] == "tomato__leaf"
    assert rows[0]["expected_crop"] == "tomato"
    assert rows[0]["expected_part"] == "leaf"
    assert rows[0]["expected_class"] == "domates_late_blight_yaprak"
    assert rows[0]["disease_class"] == "domates_late_blight_yaprak"
    assert rows[0]["source"].endswith("image.png")
    assert rows[-1]["source"].endswith("image_09.png")


def test_classify_failure_marks_gated_model_as_dependency_access():
    bucket = classify_failure(
        {
            "status": "router_unavailable",
            "message": "Strict VLM model loading failed: gated repo 401 Client Error",
        },
        asset_status="ok",
    )

    assert bucket == "dependency_access"


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


def test_build_analysis_summary_separates_router_and_adapter_failures():
    analysis = build_analysis_summary(
        [
            {
                "image_id": "demo_001",
                "actual_status": "success",
                "pass_fail": "pass",
                "expected_target": "tomato__fruit",
                "expected_crop": "tomato",
                "expected_part": "fruit",
                "expected_class": "domates_late_blight_meyve",
                "predicted_crop": "tomato",
                "predicted_part": "fruit",
                "predicted_disease": "domates_late_blight_meyve",
                "failure_bucket": "",
            },
            {
                "image_id": "demo_002",
                "actual_status": "success",
                "pass_fail": "fail",
                "expected_target": "tomato__fruit",
                "expected_crop": "tomato",
                "expected_part": "fruit",
                "expected_class": "domates_late_blight_meyve",
                "predicted_crop": "tomato",
                "predicted_part": "fruit",
                "predicted_disease": "domates_late_blight_yaprak",
                "failure_bucket": "",
            },
            {
                "image_id": "demo_003",
                "actual_status": "adapter_unavailable",
                "pass_fail": "fail",
                "expected_target": "tomato__fruit",
                "expected_crop": "tomato",
                "expected_part": "fruit",
                "expected_class": "",
                "predicted_crop": "potato",
                "predicted_part": "leaf",
                "predicted_disease": None,
                "failure_bucket": "adapter_loading",
            },
            {
                "image_id": "demo_004",
                "actual_status": "success",
                "pass_fail": "pass",
                "expected_target": "apricot__fruit",
                "expected_crop": "apricot",
                "expected_part": "fruit",
                "expected_class": "kayısıda_yaprak_delen_cil_hastalığı_meyve_128",
                "predicted_crop": "apricot",
                "predicted_part": "fruit",
                "predicted_disease": "kayısıda_yaprak_delen_cil_hastalığı_meyve_128",
                "failure_bucket": "",
            },
            {
                "image_id": "demo_005",
                "actual_status": "router_uncertain",
                "pass_fail": "pass",
                "expected_target": "grape__fruit",
                "expected_crop": "grape",
                "expected_part": "fruit",
                "expected_class": "",
                "expected_behavior": "external supported crop/part image; disease answer or review expected",
                "predicted_crop": None,
                "predicted_part": None,
                "predicted_disease": None,
                "failure_bucket": "",
            },
        ]
    )

    assert analysis["router_crop_correctness"] == {"correct": 3, "incorrect": 2, "not_applicable": 0}
    assert analysis["router_part_correctness"] == {"correct": 3, "incorrect": 2, "not_applicable": 0}
    assert analysis["normalized_disease_class_correctness"] == {
        "correct": 2,
        "incorrect": 1,
        "not_applicable": 2,
    }
    assert analysis["classless_supported_probes"] == {
        "total": 1,
        "answered": 0,
        "answered_target_correct": 0,
        "answered_target_incorrect": 0,
        "reviewed_or_abstained": 1,
        "failed": 0,
    }
    assert analysis["adapter_unavailable"] == {"wrong_router": 1, "missing_adapter": 0, "unknown": 0}
    assert analysis["opposite_part_disease_labels"]["image_ids"] == ["demo_002"]


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
