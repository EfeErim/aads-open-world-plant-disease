import json
import os
from pathlib import Path

from PIL import Image

from scripts import colab_roi_ablation as roi_ablation


class FakeWorkflow:
    calls = []

    def __init__(self, *, environment=None, device="cuda", adapter_root=None, status_callback=None):
        self.environment = environment
        self.device = device
        self.adapter_root = adapter_root
        self.status_callback = status_callback

    def predict(self, image, *, crop_hint=None, part_hint=None, return_ood=True, trust_crop_hint=False):
        self.calls.append(
            {
                "image": image,
                "crop_hint": crop_hint,
                "part_hint": part_hint,
                "return_ood": return_ood,
                "trust_crop_hint": trust_crop_hint,
                "environment": self.environment,
                "device": self.device,
                "adapter_root": self.adapter_root,
            }
        )
        return {
            "status": "success",
            "crop": crop_hint or "tomato",
            "part": part_hint or "leaf",
            "diagnosis": "healthy",
            "confidence": 0.88,
            "router_confidence": 0.93,
            "router": {
                "status": "trusted_hint_skipped",
                "primary_detection": {
                    "crop": crop_hint or "tomato",
                    "part": part_hint or "leaf",
                    "crop_confidence": 1.0,
                },
            },
            "ood_analysis": {
                "is_ood": False,
                "primary_score": 0.12,
            },
        }


class DualViewWorkflow:
    calls = []

    def __init__(self, *, environment=None, device="cuda", adapter_root=None, status_callback=None):
        self.environment = environment
        self.device = device
        self.adapter_root = adapter_root
        self.status_callback = status_callback

    def predict(self, image, *, crop_hint=None, part_hint=None, return_ood=True, trust_crop_hint=False):
        is_roi = hasattr(image, "size")
        payload = {
            "status": "success",
            "crop": crop_hint or "tomato",
            "part": part_hint or "fruit",
            "diagnosis": "roi_label" if is_roi else "full_label",
            "confidence": 0.95 if is_roi else 0.70,
            "router": {
                "status": "trusted_hint_skipped",
                "primary_detection": {
                    "crop": crop_hint or "tomato",
                    "part": part_hint or "fruit",
                    "crop_confidence": 1.0,
                },
            },
            "ood_analysis": {
                "is_ood": False,
                "primary_score": 0.10,
            },
        }
        self.calls.append({"image": image, "payload": payload})
        return payload


class GroundingDinoMock:
    calls = []
    detections = []
    status = "ok"
    error = ""

    @classmethod
    def reset(cls, detections=None, *, status="ok", error=""):
        cls.calls = []
        cls.detections = list(detections or [])
        cls.status = status
        cls.error = error

    @classmethod
    def run(cls, image, **kwargs):
        cls.calls.append({"image": image, **kwargs})
        payload = {"detections": list(cls.detections), "candidate_count": len(cls.detections), "status": cls.status}
        if cls.error:
            payload["error"] = cls.error
        return payload


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 60), color=(20, 120, 40)).save(path)
    return path


def _router_result(*, status="ok", crop="tomato", part="leaf", bbox=None, detections=None):
    primary = {
        "crop": crop,
        "part": part,
        "crop_confidence": 0.93,
        "part_confidence": 0.85,
        "bbox": bbox,
    }
    detection_rows = list(detections) if detections is not None else [primary]
    return {
        "status": status,
        "crop": crop,
        "part": part,
        "router_confidence": 0.93,
        "router": {
            "status": status,
            "primary_detection": primary,
        },
        "router_details": {
            "status": status,
            "primary_detection": primary,
            "detections": detection_rows,
        },
    }


def _write_report(path: Path, *, accuracy: float, macro_f1: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"summary": {"sample_count": 3, "comparable_count": 3, "accuracy": accuracy, "macro_f1": macro_f1}}),
        encoding="utf-8",
    )
    return path


def test_prepare_primary_roi_sanitizes_valid_and_invalid_bbox():
    image = Image.new("RGB", (100, 80), color="green")

    roi, bbox, area_ratio = roi_ablation.prepare_primary_roi(image, [-5, 10, 50, 90])
    assert roi is not None
    assert bbox == [0.0, 10.0, 50.0, 80.0]
    assert area_ratio > 0.0

    roi, bbox, area_ratio = roi_ablation.prepare_primary_roi(image, [10, 10, 10, 30])
    assert roi is None
    assert bbox is None
    assert area_ratio == 0.0


def test_classify_roi_quality_marks_missing_small_large_and_ok():
    assert roi_ablation.classify_roi_quality(bbox=None, area_ratio=0.0) == "roi_missing"
    assert roi_ablation.classify_roi_quality(bbox=[0, 0, 1, 1], area_ratio=0.01) == "roi_too_small"
    assert roi_ablation.classify_roi_quality(bbox=[0, 0, 99, 99], area_ratio=0.95) == "roi_too_large"
    assert roi_ablation.classify_roi_quality(bbox=[0, 0, 50, 40], area_ratio=0.25) == "roi_ok"


def test_tokenized_git_remote_url_uses_github_token(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "secret-token")

    push_url = roi_ablation._tokenized_git_remote_url("https://github.com/EfeErim/aads-open-world-plant-disease.git")

    assert push_url == "https://x-access-token:secret-token@github.com/EfeErim/aads-open-world-plant-disease.git"


def test_grounding_dino_component_loader_reuses_session_cache(monkeypatch):
    calls = {"processor": 0, "model": 0}
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    class FakeProcessor:
        pass

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_processor_from_pretrained(*args, **kwargs):
        calls["processor"] += 1
        return FakeProcessor()

    def fake_model_from_pretrained(*args, **kwargs):
        calls["model"] += 1
        return FakeModel()

    roi_ablation.clear_grounding_dino_cache()
    monkeypatch.setattr(AutoProcessor, "from_pretrained", fake_processor_from_pretrained)
    monkeypatch.setattr(AutoModelForZeroShotObjectDetection, "from_pretrained", fake_model_from_pretrained)

    first = roi_ablation._load_grounding_dino_components(model_id="fake-model", device="cpu", status_printer=None)
    second = roi_ablation._load_grounding_dino_components(model_id="fake-model", device="cpu", status_printer=None)

    assert first[0] is second[0]
    assert first[1] is second[1]
    assert calls == {"processor": 1, "model": 1}


def test_grounding_dino_prompt_normalization_matches_transformers_contract():
    prompts = roi_ablation._normalize_grounding_prompts([" Tomato Fruit ", "fruit on tomato plant.", "", None])

    assert prompts == ("tomato fruit.", "fruit on tomato plant.")


def test_grounding_dino_prompts_are_built_from_crop_and_part():
    prompts = roi_ablation.build_grounding_dino_prompts("Grape", "Leaf")

    assert prompts[:4] == ("grape leaf.", "a grape leaf.", "grape leafs.", "leaf on grape plant.")
    assert "tomato fruit." not in prompts


def test_discover_dual_view_targets_resolves_all_matching_dataset_adapter_pairs(tmp_path: Path):
    for dataset_key in ("tomato__fruit", "grape__leaf"):
        _write_image(tmp_path / "data" / "prepared_runtime_datasets" / dataset_key / "test" / "class_a" / "sample.jpg")
    (tmp_path / "runs" / "tomato" / "fruit" / "old" / "outputs" / "colab_notebook_training").mkdir(parents=True)
    latest = tmp_path / "runs" / "tomato" / "fruit" / "new" / "outputs" / "colab_notebook_training"
    latest.mkdir(parents=True)
    old_run = tmp_path / "runs" / "tomato" / "fruit" / "old"
    new_run = tmp_path / "runs" / "tomato" / "fruit" / "new"
    os.utime(old_run, (1_700_000_000, 1_700_000_000))
    os.utime(new_run, (1_700_000_100, 1_700_000_100))
    os.utime(old_run / "outputs" / "colab_notebook_training", (1_700_000_000, 1_700_000_000))
    os.utime(latest, (1_700_000_100, 1_700_000_100))
    (tmp_path / "runs" / "grape" / "leaf" / "only" / "outputs" / "colab_notebook_training").mkdir(parents=True)

    targets = roi_ablation.discover_dual_view_targets(
        tmp_path,
        dataset_root="data/prepared_runtime_datasets",
    )

    assert [target["dataset_key"] for target in targets] == ["grape__leaf", "tomato__fruit"]
    tomato = next(target for target in targets if target["dataset_key"] == "tomato__fruit")
    assert tomato["adapter_root"] == latest
    assert tomato["adapter_crop"] == "tomato"
    assert tomato["adapter_part"] == "fruit"
    assert "tomato fruit." in tomato["grounding_dino_prompts"]


def test_run_dual_view_inference_targets_writes_aggregate_report(tmp_path: Path):
    _write_image(tmp_path / "data" / "prepared_runtime_datasets" / "tomato__fruit" / "test" / "class_a" / "sample.jpg")
    adapter_root = tmp_path / "runs" / "tomato" / "fruit" / "run" / "outputs" / "colab_notebook_training"
    adapter_root.mkdir(parents=True)
    calls = []

    def fake_folder_runner(image_dir, **kwargs):
        calls.append({"image_dir": image_dir, **kwargs})
        row = {
            "ablation_name": "dual_view_inference",
            "image_path": str(Path(image_dir) / "class_a" / "sample.jpg"),
            "expected_label": "class_a",
            "crop": kwargs["adapter_crop"],
            "part": kwargs["adapter_part"],
            "diagnosis": "class_a",
            "status": "success",
        }
        return {
            "summary": {"sample_count": 1, "comparable_count": 1, "accuracy": 1.0, "macro_f1": 1.0},
            "rows": [row],
        }

    report = roi_ablation.run_dual_view_inference_targets(
        repo_root=tmp_path,
        dataset_root="data/prepared_runtime_datasets",
        output_dir=tmp_path / "reports",
        status_printer=None,
        folder_runner=fake_folder_runner,
    )

    assert len(calls) == 1
    assert calls[0]["adapter_crop"] == "tomato"
    assert calls[0]["adapter_part"] == "fruit"
    assert calls[0]["grounding_dino_prompts"][0] == "tomato fruit."
    assert report["summary"]["accuracy"] == 1.0
    assert report["failure_analysis"]["schema_version"] == "v1_dual_view_evidence_failure_analysis"
    assert report["rows"][0]["dataset_key"] == "tomato__fruit"
    assert (tmp_path / "reports" / "multi_target_report.json").is_file()


def test_primary_roi_ablation_skips_without_fallback_when_bbox_missing(tmp_path: Path):
    FakeWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "leaf.jpg")

    rows = roi_ablation.run_ablation_image(
        image_path,
        ablation_name="primary_roi_inference",
        workflow_factory=FakeWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(bbox=None),
    )

    assert rows[0]["status"] == "roi_missing"
    assert rows[0]["input_view"] == "router_primary_roi"
    assert FakeWorkflow.calls == []


def test_hybrid_ablation_falls_back_to_full_image_when_bbox_missing(tmp_path: Path):
    FakeWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "leaf.jpg")

    rows = roi_ablation.run_ablation_image(
        image_path,
        ablation_name="hybrid_roi_fallback",
        workflow_factory=FakeWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(bbox=None),
    )

    assert rows[0]["status"] == "fallback_full_image"
    assert rows[0]["input_view"] == "fallback_full_image"
    assert FakeWorkflow.calls[0]["image"] == image_path
    assert FakeWorkflow.calls[0]["crop_hint"] == "tomato"
    assert FakeWorkflow.calls[0]["part_hint"] == "leaf"
    assert FakeWorkflow.calls[0]["trust_crop_hint"] is True


def test_full_image_baseline_uses_fixed_adapter_without_router(tmp_path: Path):
    FakeWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "fruit.jpg")

    def fail_router(*args, **kwargs):
        raise AssertionError("full-image fixed-adapter baseline should not call the router")

    rows = roi_ablation.run_ablation_image(
        image_path,
        ablation_name="full_image_baseline",
        adapter_crop="tomato",
        adapter_part="fruit",
        workflow_factory=FakeWorkflow,
        router_runner=fail_router,
    )

    assert rows[0]["status"] == "success"
    assert rows[0]["crop"] == "tomato"
    assert rows[0]["part"] == "fruit"
    assert FakeWorkflow.calls[0]["crop_hint"] == "tomato"
    assert FakeWorkflow.calls[0]["part_hint"] == "fruit"
    assert FakeWorkflow.calls[0]["trust_crop_hint"] is True


def test_fixed_adapter_roi_ignores_wrong_router_crop_for_adapter_selection(tmp_path: Path):
    FakeWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "fruit.jpg")

    rows = roi_ablation.run_ablation_image(
        image_path,
        ablation_name="primary_roi_inference",
        adapter_crop="tomato",
        adapter_part="fruit",
        workflow_factory=FakeWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(crop="tomato", part="fruit", bbox=[10, 10, 50, 40]),
    )

    assert rows[0]["status"] == "success"
    assert rows[0]["crop"] == "tomato"
    assert rows[0]["part"] == "fruit"
    assert rows[0]["router_status"] == "ok"
    assert FakeWorkflow.calls[0]["crop_hint"] == "tomato"
    assert FakeWorkflow.calls[0]["part_hint"] == "fruit"


def test_target_aware_roi_selects_matching_detection_instead_of_wrong_primary(tmp_path: Path):
    FakeWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "fruit.jpg")

    rows = roi_ablation.run_ablation_image(
        image_path,
        ablation_name="primary_roi_inference",
        adapter_crop="tomato",
        adapter_part="fruit",
        workflow_factory=FakeWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(
            crop="eggplant",
            part="unknown",
            bbox=[1, 1, 20, 20],
            detections=[
                {
                    "crop": "eggplant",
                    "part": "unknown",
                    "crop_confidence": 0.95,
                    "part_confidence": 0.0,
                    "bbox": [1, 1, 20, 20],
                },
                {
                    "crop": "tomato",
                    "part": "fruit",
                    "crop_confidence": 0.62,
                    "part_confidence": 0.71,
                    "bbox": [10, 10, 50, 40],
                },
            ],
        ),
    )

    assert rows[0]["status"] == "success"
    assert rows[0]["bbox"] == [10.0, 10.0, 50.0, 40.0]
    assert rows[0]["selected_detection_source"] == "router_detection"
    assert rows[0]["target_detection_source"] == "router_detection"
    assert rows[0]["target_detection_found"] is True
    assert FakeWorkflow.calls[0]["crop_hint"] == "tomato"
    assert FakeWorkflow.calls[0]["part_hint"] == "fruit"


def test_target_aware_roi_skips_wrong_primary_when_target_detection_missing(tmp_path: Path):
    FakeWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "fruit.jpg")

    rows = roi_ablation.run_ablation_image(
        image_path,
        ablation_name="primary_roi_inference",
        adapter_crop="tomato",
        adapter_part="fruit",
        workflow_factory=FakeWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(crop="eggplant", part="unknown", bbox=[10, 10, 50, 40]),
        target_roi_backend="router_detections",
    )

    assert rows[0]["status"] == "roi_missing"
    assert rows[0]["bbox"] is None
    assert rows[0]["selected_detection_source"] == "target_detection_missing"
    assert rows[0]["target_detection_found"] is False
    assert FakeWorkflow.calls == []


def test_target_aware_roi_uses_grounding_dino_when_router_target_missing(tmp_path: Path):
    FakeWorkflow.calls.clear()
    GroundingDinoMock.reset(
        detections=[
            {
                "crop_confidence": 0.77,
                "part_confidence": 0.77,
                "bbox": [10, 10, 50, 40],
                "prompt": "tomato fruit",
            }
        ]
    )
    image_path = _write_image(tmp_path / "fruit.jpg")

    rows = roi_ablation.run_ablation_image(
        image_path,
        ablation_name="primary_roi_inference",
        adapter_crop="tomato",
        adapter_part="fruit",
        workflow_factory=FakeWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(crop="eggplant", part="unknown", bbox=[1, 1, 20, 20]),
        grounding_dino_runner=GroundingDinoMock.run,
    )

    assert rows[0]["status"] == "success"
    assert rows[0]["bbox"] == [10.0, 10.0, 50.0, 40.0]
    assert rows[0]["selected_detection_source"] == "grounding_dino"
    assert rows[0]["target_detection_source"] == "grounding_dino"
    assert rows[0]["target_detection_found"] is True
    assert rows[0]["target_prompt"] == "tomato fruit"
    assert rows[0]["target_detection_confidence"] == 0.77
    assert rows[0]["grounding_dino_candidate_count"] == 1
    assert rows[0]["grounding_dino_status"] == "ok"
    assert rows[0]["grounding_dino_error"] == ""
    assert len(GroundingDinoMock.calls) == 1
    assert FakeWorkflow.calls[0]["crop_hint"] == "tomato"


def test_target_aware_roi_skips_low_quality_grounding_dino_bbox(tmp_path: Path):
    FakeWorkflow.calls.clear()
    GroundingDinoMock.reset(
        detections=[
            {
                "crop_confidence": 0.88,
                "part_confidence": 0.88,
                "bbox": [0, 0, 79, 59],
                "prompt": "tomato",
            }
        ]
    )
    image_path = _write_image(tmp_path / "fruit.jpg")

    rows = roi_ablation.run_ablation_image(
        image_path,
        ablation_name="primary_roi_inference",
        adapter_crop="tomato",
        adapter_part="fruit",
        workflow_factory=FakeWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(crop="eggplant", part="unknown", bbox=[1, 1, 20, 20]),
        grounding_dino_runner=GroundingDinoMock.run,
    )

    assert rows[0]["status"] == "roi_too_large"
    assert rows[0]["target_detection_found"] is True
    assert rows[0]["selected_detection_source"] == "grounding_dino"
    assert rows[0]["bbox"] == [0.0, 0.0, 79.0, 59.0]
    assert FakeWorkflow.calls == []


def test_part_unknown_skips_adapter_load(tmp_path: Path):
    FakeWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "leaf.jpg")

    rows = roi_ablation.run_ablation_image(
        image_path,
        ablation_name="hybrid_roi_fallback",
        workflow_factory=FakeWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(part="unknown", bbox=[5, 5, 40, 40]),
    )

    assert rows[0]["status"] == "adapter_skipped"
    assert rows[0]["part"] == "unknown"
    assert FakeWorkflow.calls == []


def test_roi_prediction_uses_cropped_pil_image_and_same_adapter_metadata(tmp_path: Path):
    FakeWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "leaf.jpg")

    rows = roi_ablation.run_ablation_image(
        image_path,
        ablation_name="primary_roi_inference",
        workflow_factory=FakeWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(bbox=[10, 10, 50, 40]),
        adapter_root=Path("models/adapters"),
        device="cpu",
    )

    assert rows[0]["status"] == "success"
    assert rows[0]["input_view"] == "router_primary_roi"
    assert rows[0]["bbox"] == [10.0, 10.0, 50.0, 40.0]
    assert FakeWorkflow.calls[0]["image"].size[0] < 80
    assert FakeWorkflow.calls[0]["crop_hint"] == "tomato"
    assert FakeWorkflow.calls[0]["part_hint"] == "leaf"
    assert FakeWorkflow.calls[0]["adapter_root"] == Path("models/adapters")


def test_build_mixed_full_roi_dataset_writes_full_and_roi_training_view(tmp_path: Path):
    source_root = tmp_path / "source"
    dataset = source_root / "tomato__fruit"
    _write_image(dataset / "continual" / "healthy" / "train_a.jpg")
    _write_image(dataset / "val" / "healthy" / "val_a.jpg")
    _write_image(dataset / "test" / "healthy" / "test_a.jpg")
    _write_image(dataset / "ood" / "field" / "ood_a.jpg")

    def strict_router_runner(
        image_path,
        *,
        config_env="colab",
        device="cuda",
        include_adapter_target=True,
        status_printer=None,
    ):
        assert image_path.name == "train_a.jpg"
        assert config_env == "colab"
        assert device == "cuda"
        assert include_adapter_target is False
        assert status_printer is None
        return _router_result(crop="tomato", part="fruit", bbox=[10, 10, 50, 40])

    manifest = roi_ablation.build_mixed_full_roi_dataset(
        source_root,
        dataset_key="tomato__fruit",
        output_root=tmp_path / "mixed",
        router_runner=strict_router_runner,
        status_printer=None,
    )

    target = tmp_path / "mixed" / "tomato__fruit"
    assert len(list((target / "continual" / "healthy").glob("full__*.jpg"))) == 1
    assert len(list((target / "continual" / "healthy").glob("roi__*.jpg"))) == 1
    assert len(list((target / "val" / "healthy").glob("full__*.jpg"))) == 1
    assert not list((target / "val" / "healthy").glob("roi__*.jpg"))
    assert (target / "ood" / "field" / "ood_a.jpg").exists()
    assert manifest["splits"]["continual"]["full_images"] == 1
    assert manifest["splits"]["continual"]["roi_images"] == 1


def test_build_roi_only_dataset_skips_full_images_and_missing_roi(tmp_path: Path):
    source_root = tmp_path / "source"
    dataset = source_root / "tomato__fruit"
    _write_image(dataset / "continual" / "healthy" / "train_a.jpg")
    _write_image(dataset / "continual" / "healthy" / "train_b.jpg")
    _write_image(dataset / "val" / "healthy" / "val_a.jpg")
    _write_image(dataset / "test" / "healthy" / "test_a.jpg")

    def router_runner(image_path, **kwargs):
        if image_path.name == "train_b.jpg":
            return _router_result(crop="tomato", part="fruit", bbox=None)
        return _router_result(crop="tomato", part="fruit", bbox=[10, 10, 50, 40])

    manifest = roi_ablation.build_roi_only_dataset(
        source_root,
        dataset_key="tomato__fruit",
        output_root=tmp_path / "roi_only",
        router_runner=router_runner,
        status_printer=None,
    )

    target = tmp_path / "roi_only" / "tomato__fruit"
    assert not list((target / "continual" / "healthy").glob("full__*.jpg"))
    assert len(list((target / "continual" / "healthy").glob("roi__*.jpg"))) == 1
    assert len(list((target / "val" / "healthy").glob("roi__*.jpg"))) == 1
    assert len(list((target / "test" / "healthy").glob("roi__*.jpg"))) == 1
    assert manifest["splits"]["continual"]["full_images"] == 0
    assert manifest["splits"]["continual"]["roi_images"] == 1
    assert manifest["splits"]["continual"]["roi_missing"] == 1


def test_roi_quality_audit_reports_router_mismatch_and_missing_roi(tmp_path: Path):
    image_path = _write_image(tmp_path / "class_a" / "sample.jpg")

    row = roi_ablation.audit_roi_quality_image(
        image_path,
        expected_label="class_a",
        expected_crop="tomato",
        expected_part="fruit",
        router_runner=lambda *args, **kwargs: _router_result(crop="eggplant", part="unknown", bbox=None),
        status_printer=None,
    )
    summary = roi_ablation.summarize_roi_quality_rows([row])

    assert row["status"] == "roi_missing"
    assert row["crop_matches_expected"] is False
    assert row["part_matches_expected"] is False
    assert summary["roi_missing_rate"] == 1.0
    assert summary["crop_mismatch_rate"] == 1.0


def test_dual_view_inference_keeps_full_image_primary_without_review_for_roi_conflict_alone(tmp_path: Path):
    DualViewWorkflow.calls.clear()
    GroundingDinoMock.reset()
    image_path = _write_image(tmp_path / "class_a" / "sample.jpg")

    row = roi_ablation.run_dual_view_inference_image(
        image_path,
        expected_label="roi_label",
        adapter_crop="tomato",
        adapter_part="fruit",
        workflow_factory=DualViewWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(crop="tomato", part="fruit", bbox=[10, 10, 50, 40]),
        grounding_dino_runner=GroundingDinoMock.run,
        status_printer=None,
        device="cpu",
    )

    assert row["semantic_roi_match"] is True
    assert row["roi_eligible"] is True
    assert row["decision_policy"] == "full_image_primary_with_roi_evidence"
    assert row["selected_view"] == "full_image"
    assert row["final_view"] == "full_image"
    assert row["diagnosis"] == "full_label"
    assert row["full_diagnosis"] == "full_label"
    assert row["roi_diagnosis"] == "roi_label"
    assert row["roi_evidence_status"] == "conflicts_with_full"
    assert row["requires_review"] is False
    assert row["review_reasons"] == ""
    assert row["dual_view_disagreement"] is True
    assert len(DualViewWorkflow.calls) == 2
    assert GroundingDinoMock.calls == []


def test_dual_view_inference_flags_roi_conflict_when_full_confidence_is_low(tmp_path: Path):
    DualViewWorkflow.calls.clear()
    GroundingDinoMock.reset()
    image_path = _write_image(tmp_path / "class_a" / "sample.jpg")

    row = roi_ablation.run_dual_view_inference_image(
        image_path,
        expected_label="roi_label",
        adapter_crop="tomato",
        adapter_part="fruit",
        workflow_factory=DualViewWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(crop="tomato", part="fruit", bbox=[10, 10, 50, 40]),
        grounding_dino_runner=GroundingDinoMock.run,
        full_confidence_review_threshold=0.80,
        status_printer=None,
        device="cpu",
    )

    assert row["selected_view"] == "full_image"
    assert row["roi_evidence_status"] == "conflicts_with_full"
    assert row["requires_review"] is True
    assert row["review_reasons"] == "low_full_confidence;roi_conflict;roi_confidence_leads"


def test_dual_view_inference_falls_back_to_full_when_roi_missing(tmp_path: Path):
    DualViewWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "class_a" / "sample.jpg")

    row = roi_ablation.run_dual_view_inference_image(
        image_path,
        expected_label="full_label",
        adapter_crop="tomato",
        adapter_part="fruit",
        workflow_factory=DualViewWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(crop="tomato", part="fruit", bbox=None),
        status_printer=None,
        device="cpu",
    )

    assert row["status"] == "success"
    assert row["selected_view"] == "full_image"
    assert row["roi_evidence_status"] == "roi_missing"
    assert row["requires_review"] is False
    assert row["review_reasons"] == ""
    assert row["diagnosis"] == "full_label"
    assert row["roi_diagnosis"] is None
    assert len(DualViewWorkflow.calls) == 1


def test_dual_view_inference_rejects_semantic_mismatch_roi(tmp_path: Path):
    DualViewWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "class_a" / "sample.jpg")

    row = roi_ablation.run_dual_view_inference_image(
        image_path,
        expected_label="full_label",
        adapter_crop="tomato",
        adapter_part="fruit",
        workflow_factory=DualViewWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(crop="eggplant", part="unknown", bbox=[10, 10, 50, 40]),
        target_roi_backend="router_detections",
        status_printer=None,
        device="cpu",
    )

    assert row["status"] == "success"
    assert row["semantic_roi_match"] is False
    assert row["roi_eligible"] is False
    assert row["roi_evidence_status"] == "target_detection_missing"
    assert row["requires_review"] is True
    assert row["review_reasons"] == "target_detection_missing"
    assert row["selected_detection_source"] == "target_detection_missing"
    assert row["target_detection_found"] is False
    assert row["selected_view"] == "full_image"
    assert row["diagnosis"] == "full_label"
    assert row["roi_diagnosis"] is None
    assert len(DualViewWorkflow.calls) == 1


def test_dual_view_inference_uses_grounding_dino_when_router_target_missing(tmp_path: Path):
    DualViewWorkflow.calls.clear()
    GroundingDinoMock.reset(
        detections=[
            {
                "crop_confidence": 0.81,
                "part_confidence": 0.81,
                "bbox": [10, 10, 50, 40],
                "prompt": "tomato fruit",
            }
        ]
    )
    image_path = _write_image(tmp_path / "class_a" / "sample.jpg")

    row = roi_ablation.run_dual_view_inference_image(
        image_path,
        expected_label="roi_label",
        adapter_crop="tomato",
        adapter_part="fruit",
        workflow_factory=DualViewWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(crop="eggplant", part="unknown", bbox=[1, 1, 20, 20]),
        grounding_dino_runner=GroundingDinoMock.run,
        status_printer=None,
        device="cpu",
    )

    assert row["target_detection_found"] is True
    assert row["target_detection_source"] == "grounding_dino"
    assert row["selected_detection_source"] == "grounding_dino"
    assert row["target_prompt"] == "tomato fruit"
    assert row["target_detection_confidence"] == 0.81
    assert row["grounding_dino_candidate_count"] == 1
    assert row["grounding_dino_status"] == "ok"
    assert row["grounding_dino_error"] == ""
    assert row["selected_view"] == "full_image"
    assert row["diagnosis"] == "full_label"
    assert row["roi_evidence_status"] == "conflicts_with_full"
    assert row["requires_review"] is False
    assert row["review_reasons"] == ""
    assert len(GroundingDinoMock.calls) == 1
    assert len(DualViewWorkflow.calls) == 2


def test_dual_view_inference_falls_back_when_grounding_dino_returns_no_candidates(tmp_path: Path):
    DualViewWorkflow.calls.clear()
    GroundingDinoMock.reset(detections=[], status="no_candidates")
    image_path = _write_image(tmp_path / "class_a" / "sample.jpg")

    row = roi_ablation.run_dual_view_inference_image(
        image_path,
        expected_label="full_label",
        adapter_crop="tomato",
        adapter_part="fruit",
        workflow_factory=DualViewWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(crop="eggplant", part="unknown", bbox=[1, 1, 20, 20]),
        grounding_dino_runner=GroundingDinoMock.run,
        status_printer=None,
        device="cpu",
    )

    assert row["status"] == "success"
    assert row["target_detection_found"] is False
    assert row["grounding_dino_candidate_count"] == 0
    assert row["grounding_dino_status"] == "no_candidates"
    assert row["roi_evidence_status"] == "target_detection_missing"
    assert row["requires_review"] is True
    assert row["selected_view"] == "full_image"
    assert row["roi_diagnosis"] is None
    assert len(GroundingDinoMock.calls) == 1
    assert len(DualViewWorkflow.calls) == 1


def test_dual_view_inference_reports_grounding_dino_error_status(tmp_path: Path):
    DualViewWorkflow.calls.clear()
    GroundingDinoMock.reset(detections=[], status="error", error="bad processor call")
    image_path = _write_image(tmp_path / "class_a" / "sample.jpg")

    row = roi_ablation.run_dual_view_inference_image(
        image_path,
        expected_label="full_label",
        adapter_crop="tomato",
        adapter_part="fruit",
        workflow_factory=DualViewWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(crop="eggplant", part="unknown", bbox=[1, 1, 20, 20]),
        grounding_dino_runner=GroundingDinoMock.run,
        status_printer=None,
        device="cpu",
    )

    assert row["status"] == "success"
    assert row["target_detection_found"] is False
    assert row["grounding_dino_status"] == "error"
    assert row["grounding_dino_error"] == "bad processor call"
    assert row["roi_evidence_status"] == "target_detection_missing"
    assert row["requires_review"] is True
    assert row["review_reasons"] == "target_detection_missing;grounding_dino_error"
    assert row["selected_view"] == "full_image"


def test_dual_view_inference_uses_target_detection_when_primary_is_wrong(tmp_path: Path):
    DualViewWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "class_a" / "sample.jpg")

    row = roi_ablation.run_dual_view_inference_image(
        image_path,
        expected_label="roi_label",
        adapter_crop="tomato",
        adapter_part="fruit",
        workflow_factory=DualViewWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(
            crop="eggplant",
            part="unknown",
            bbox=[1, 1, 20, 20],
            detections=[
                {
                    "crop": "eggplant",
                    "part": "unknown",
                    "crop_confidence": 0.95,
                    "part_confidence": 0.0,
                    "bbox": [1, 1, 20, 20],
                },
                {
                    "crop": "tomato",
                    "part": "fruit",
                    "crop_confidence": 0.62,
                    "part_confidence": 0.71,
                    "bbox": [10, 10, 50, 40],
                },
            ],
        ),
        status_printer=None,
        device="cpu",
    )

    assert row["router_primary_crop"] == "eggplant"
    assert row["router_crop"] == "tomato"
    assert row["router_part"] == "fruit"
    assert row["bbox"] == [10.0, 10.0, 50.0, 40.0]
    assert row["semantic_roi_match"] is True
    assert row["target_detection_found"] is True
    assert row["selected_detection_source"] == "router_detection"
    assert row["target_detection_source"] == "router_detection"
    assert row["selected_view"] == "full_image"
    assert row["diagnosis"] == "full_label"
    assert row["roi_evidence_status"] == "conflicts_with_full"
    assert row["requires_review"] is False
    assert row["review_reasons"] == ""
    assert len(DualViewWorkflow.calls) == 2


def test_dual_view_training_gate_blocks_until_required_reports_exist(tmp_path: Path):
    payload = roi_ablation.evaluate_dual_view_training_gate(repo_root=tmp_path)

    assert payload["status"] == "blocked_until_dual_view_inference_results"
    assert payload["gate_passed"] is False
    assert set(payload["missing_reports"]) == {"full_image_baseline", "dual_view_inference"}
    assert (tmp_path / "docs" / "ablation_results" / "dual_view_trained_adapter" / "dual_view_training_gate.json").is_file()


def test_dual_view_training_gate_skips_when_dual_view_underperforms(tmp_path: Path):
    baseline = _write_report(tmp_path / "baseline.json", accuracy=0.90, macro_f1=0.88)
    dual_view = _write_report(tmp_path / "dual_view.json", accuracy=0.82, macro_f1=0.80)

    payload = roi_ablation.evaluate_dual_view_training_gate(
        repo_root=tmp_path,
        baseline_report=baseline,
        dual_view_report=dual_view,
    )

    assert payload["status"] == "skipped_by_gate"
    assert payload["gate_passed"] is False
    assert payload["deltas"]["accuracy"] < 0
    assert payload["deltas"]["macro_f1"] < 0


def test_dual_view_training_gate_passes_when_dual_view_improves(tmp_path: Path):
    baseline = _write_report(tmp_path / "baseline.json", accuracy=0.90, macro_f1=0.88)
    dual_view = _write_report(tmp_path / "dual_view.json", accuracy=0.91, macro_f1=0.87)

    payload = roi_ablation.evaluate_dual_view_training_gate(
        repo_root=tmp_path,
        baseline_report=baseline,
        dual_view_report=dual_view,
    )

    assert payload["status"] == "ready_for_paired_dual_view_training"
    assert payload["gate_passed"] is True
    assert payload["deltas"]["accuracy"] > 0
