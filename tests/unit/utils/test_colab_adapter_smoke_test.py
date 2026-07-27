import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from scripts import colab_adapter_smoke_test as smoke
from tests.utils.test_helpers import make_image


class _FakeAdapter:
    def __init__(self, crop_name: str, device: str = "cpu"):
        self.crop_name = crop_name
        self.device = device
        self.loaded_paths: list[str] = []

    def load_adapter(self, adapter_dir: str) -> None:
        self.loaded_paths.append(adapter_dir)

    def get_summary(self):
        return {
            "crop_name": self.crop_name,
            "engine": "continual_sd_lora",
            "schema_version": "v6",
            "is_trained": True,
            "num_classes": 2,
            "class_to_idx": {"healthy": 0, "blight": 1},
            "ood_calibration_version": 3,
        }

    def predict_with_ood(self, image):
        return {
            "status": "success",
            "disease": {"class_index": 1, "name": "blight", "confidence": 0.91},
            "ood_analysis": {
                "is_ood": False,
                "score_method": "ensemble",
                "primary_score": 0.2,
                "decision_threshold": 0.8,
                "candidate_scores": {"ensemble": 0.2},
                "candidate_thresholds": {"ensemble": 0.8},
                "calibration_version": 3,
            },
        }


class _ScriptedViewAdapter(_FakeAdapter):
    def __init__(self, crop_name: str, payloads_by_marker: dict[float, dict], device: str = "cpu"):
        super().__init__(crop_name, device=device)
        self.payloads_by_marker = payloads_by_marker

    def predict_with_ood(self, image):
        marker = round(float(image[0, 0, 0].item()), 2)
        return dict(self.payloads_by_marker[marker])


class _FakeAttentionConfig:
    image_size = 16
    patch_size = 8
    num_register_tokens = 1
    _attn_implementation = "sdpa"


class _FakeAttentionModel:
    def __init__(self):
        self.config = _FakeAttentionConfig()

    def __call__(self, pixel_values, *, output_attentions: bool, output_hidden_states: bool):
        assert output_attentions is True
        assert output_hidden_states is True
        assert self.config._attn_implementation == "eager"
        attention = torch.zeros((1, 2, 6, 6), dtype=torch.float32, device=pixel_values.device)
        attention[:, :, 0, 2:] = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32, device=pixel_values.device)
        return SimpleNamespace(attentions=(attention,))


class _FakeAttentionAdapter(_FakeAdapter):
    @property
    def trainer(self):
        return SimpleNamespace(
            adapter_model=_FakeAttentionModel(),
            device=torch.device("cpu"),
            set_eval_mode=lambda: None,
        )


def _write_adapter_export(root: Path) -> Path:
    asset_dir = root / "continual_sd_lora_adapter"
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "adapter_meta.json").write_text(
        """
        {
          "schema_version": "v6",
          "engine": "continual_sd_lora",
          "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
          "fusion": {"layers": [2, 5, 8, 11], "output_dim": 768, "dropout": 0.1, "gating": "softmax"},
          "class_to_idx": {"healthy": 0, "blight": 1},
          "ood_calibration": {"version": 3},
          "target_modules_resolved": ["encoder.layer.0.attention.q_proj"],
          "adapter_runtime": {"adapter_wrapped": true}
        }
        """.strip(),
        encoding="utf-8",
    )
    return asset_dir


def _write_adapter_export_with_crop_info(root: Path, crop_name: str = "tomato") -> Path:
    asset_dir = _write_adapter_export(root)
    (asset_dir / "crop_info.json").write_text(
        f'{{"crop": "{crop_name}", "run_id": "run_local", "export_source": "notebook_2"}}',
        encoding="utf-8",
    )
    return asset_dir


def _write_current_drive_adapter_export(root: Path, crop_name: str = "tomato") -> Path:
    asset_dir = root / "artifacts" / "adapter_export" / "continual_sd_lora_adapter"
    asset_dir.mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "crop_info.json").write_text(
        f'{{"crop": "{crop_name}", "run_id": "{root.name}"}}',
        encoding="utf-8",
    )
    (asset_dir / "adapter_meta.json").write_text(
        """
        {
          "schema_version": "v6",
          "engine": "continual_sd_lora",
          "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
          "fusion": {"layers": [2, 5, 8, 11], "output_dim": 768, "dropout": 0.1, "gating": "softmax"},
          "class_to_idx": {"healthy": 0, "blight": 1},
          "ood_calibration": {"version": 3},
          "target_modules_resolved": ["encoder.layer.0.attention.q_proj"],
          "adapter_runtime": {"adapter_wrapped": true}
        }
        """.strip(),
        encoding="utf-8",
    )
    return asset_dir


def _write_production_readiness(root: Path, crop_name: str = "tomato") -> None:
    readiness_path = root / "production_readiness.json"
    readiness_path.parent.mkdir(parents=True, exist_ok=True)
    readiness_path.write_text(
        f'{{"status": "ready", "passed": true, "context": {{"crop_name": "{crop_name}"}}}}',
        encoding="utf-8",
    )


def _write_training_summary(root: Path, crop_name: str = "tomato") -> None:
    summary_path = root / "training" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        f'{{"run_id": "run_local", "crop_name": "{crop_name}"}}',
        encoding="utf-8",
    )


def _write_adapter_export_with_prefixed_classes(root: Path) -> Path:
    asset_dir = root / "continual_sd_lora_adapter"
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "adapter_meta.json").write_text(
        """
        {
          "schema_version": "v6",
          "engine": "continual_sd_lora",
          "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
          "fusion": {"layers": [2, 5, 8, 11], "output_dim": 768, "dropout": 0.1, "gating": "softmax"},
          "class_to_idx": {
            "healthy": 0,
            "tomato_early_blight_leaf": 1,
            "tomato_late_blight_leaf": 2
          },
          "ood_calibration": {"version": 3},
          "target_modules_resolved": ["encoder.layer.0.attention.q_proj"],
          "adapter_runtime": {"adapter_wrapped": true}
        }
        """.strip(),
        encoding="utf-8",
    )
    return asset_dir


def test_cli_writes_skipped_report_when_no_adapters(tmp_path: Path):
    output = tmp_path / "adapter_smoke.json"

    exit_code = smoke.main([
        "--adapter-root",
        str(tmp_path / "models" / "adapters"),
        "--output",
        str(output),
        "--strict",
    ])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "skipped"
    assert report["adapter_count"] == 0


def test_cli_validates_adapter_metadata_contract(tmp_path: Path):
    adapter_root = tmp_path / "models" / "adapters" / "tomato" / "leaf"
    asset_dir = _write_adapter_export(adapter_root)
    output = tmp_path / "adapter_smoke.json"

    exit_code = smoke.main([
        "--adapter-root",
        str(tmp_path / "models" / "adapters"),
        "--output",
        str(output),
        "--strict",
    ])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "pass"
    assert report["adapter_count"] == 1
    assert report["adapters"][0]["adapter_dir"] == str(asset_dir)
    assert report["adapters"][0]["crop_name"] == "tomato"
    assert report["adapters"][0]["part_name"] == "leaf"


def test_cli_fails_invalid_adapter_metadata_contract(tmp_path: Path):
    asset_dir = tmp_path / "models" / "adapters" / "tomato" / "leaf" / "continual_sd_lora_adapter"
    asset_dir.mkdir(parents=True)
    (asset_dir / "adapter_meta.json").write_text(
        '{"schema_version": "v6", "engine": "continual_sd_lora"}',
        encoding="utf-8",
    )
    output = tmp_path / "adapter_smoke.json"

    exit_code = smoke.main([
        "--adapter-root",
        str(tmp_path / "models" / "adapters"),
        "--output",
        str(output),
    ])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["status"] == "fail"
    assert report["fail_count"] == 1
    assert "adapter_meta.class_to_idx" in report["adapters"][0]["errors"][0]


def _scripted_view_tensor(_image, *, target_size: int, view_name: str):
    marker_map = {
        "full_resize": 0.11,
        "resize_pad": 0.22,
        "center_crop": 0.33,
    }
    return torch.full((3, target_size, target_size), marker_map[view_name], dtype=torch.float32)


def _make_payload(
    *,
    predicted_class: str = "blight",
    confidence: float = 0.91,
    is_ood: bool = False,
    primary_score: float = 0.2,
    decision_threshold: float = 0.8,
    extra_ood: dict | None = None,
):
    ood_analysis = {
        "is_ood": is_ood,
        "primary_score": primary_score,
        "decision_threshold": decision_threshold,
        "calibration_version": 3,
    }
    if extra_ood:
        ood_analysis.update(extra_ood)
    return {
        "status": "success",
        "disease": {"class_index": 1, "name": predicted_class, "confidence": confidence},
        "ood_analysis": ood_analysis,
    }



def test_load_adapter_summary_accepts_parent_export_dir(monkeypatch, tmp_path: Path):
    export_root = tmp_path / "adapter_export"
    asset_dir = _write_adapter_export(export_root)
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary("tomato", adapter_dir=export_root, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["backbone_model_name"] == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert summary["class_names"] == ["healthy", "blight"]
    assert summary["target_modules_resolved"] == ["encoder.layer.0.attention.q_proj"]



def test_load_adapter_summary_accepts_asset_dir(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary("tomato", adapter_dir=asset_dir, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["class_count"] == 2



def test_load_adapter_summary_accepts_adapter_meta_file(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    meta_path = asset_dir / "adapter_meta.json"
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary("tomato", adapter_dir=meta_path, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)



def test_load_adapter_summary_accepts_current_drive_export_dir_and_infers_crop(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "telemetry" / "run_789"
    asset_dir = _write_current_drive_adapter_export(run_dir, crop_name="tomato")
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary(None, adapter_dir=run_dir / "artifacts" / "adapter_export", device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["crop_name"] == "tomato"



def test_load_adapter_summary_accepts_telemetry_run_dir_and_infers_crop_from_readiness(
    monkeypatch, tmp_path: Path
):
    run_dir = tmp_path / "telemetry" / "run_456"
    asset_dir = _write_adapter_export(run_dir / "artifacts" / "adapter_export")
    _write_production_readiness(run_dir / "artifacts", crop_name="tomato")
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary(None, adapter_dir=run_dir, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["crop_name"] == "tomato"



def test_load_adapter_summary_accepts_telemetry_artifacts_dir_and_infers_crop_from_summary(
    monkeypatch, tmp_path: Path
):
    artifacts_dir = tmp_path / "telemetry" / "run_654" / "artifacts"
    asset_dir = _write_adapter_export(artifacts_dir / "adapter_export")
    _write_training_summary(artifacts_dir, crop_name="tomato")
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary(None, adapter_dir=artifacts_dir, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["crop_name"] == "tomato"



def test_load_adapter_summary_infers_crop_from_local_export_crop_info(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export_with_crop_info(
        tmp_path / "outputs" / "colab_notebook_training",
        crop_name="tomato",
    )
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary(None, adapter_dir=asset_dir, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["crop_name"] == "tomato"



def test_load_adapter_summary_infers_crop_from_local_export_artifacts(monkeypatch, tmp_path: Path):
    export_root = tmp_path / "outputs" / "colab_notebook_training"
    asset_dir = _write_adapter_export(export_root)
    _write_production_readiness(export_root / "artifacts", crop_name="tomato")
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary(None, adapter_dir=export_root, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["crop_name"] == "tomato"


def test_load_adapter_summary_resolves_crop_part_local_export_root(monkeypatch, tmp_path: Path):
    export_root = tmp_path / "outputs" / "colab_notebook_training"
    asset_dir = _write_adapter_export(export_root / "tomato" / "leaf")
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary(
        "tomato",
        adapter_dir=export_root,
        part_name="leaf",
        device="cpu",
    )

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["crop_name"] == "tomato"



def test_load_adapter_summary_infers_crop_from_adapter_meta_classes(monkeypatch, tmp_path: Path):
    export_root = tmp_path / "outputs" / "adapter_export"
    asset_dir = _write_adapter_export_with_prefixed_classes(export_root)
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary(None, adapter_dir=asset_dir, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["crop_name"] == "tomato"



def test_predict_single_image_returns_notebook_payload(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    image_path = tmp_path / "leaf.png"
    make_image(image_path, size=(8, 8), color="green")

    def zero_preprocess(_image, target_size=224):
        return torch.zeros(3, target_size, target_size)

    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))
    monkeypatch.setattr(smoke, "preprocess_image", zero_preprocess)
    monkeypatch.setattr(smoke, "_target_size", lambda _env: 224)

    result = smoke.predict_single_image(image_path, "tomato", adapter_dir=asset_dir, device="cpu")

    assert result["image_name"] == "leaf.png"
    assert result["predicted_class"] == "blight"
    assert result["confidence"] == 0.91
    assert result["is_ood"] is False
    assert result["raw_payload"]["status"] == "success"
    assert "views" not in result


def test_predict_single_image_can_attach_occlusion_visualization(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    image_path = tmp_path / "leaf.png"
    make_image(image_path, size=(12, 10), color="green")

    def zero_preprocess(_image, target_size=224):
        return torch.zeros(3, target_size, target_size)

    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))
    monkeypatch.setattr(smoke, "preprocess_image", zero_preprocess)
    monkeypatch.setattr(smoke, "_target_size", lambda _env: 16)

    result = smoke.predict_single_image(
        image_path,
        "tomato",
        adapter_dir=asset_dir,
        device="cpu",
        explain_prediction=True,
        explanation_grid_size=3,
    )

    visualization = result["visualization"]
    assert visualization["method"] == "occlusion_sensitivity"
    assert visualization["view_name"] == "full_resize"
    assert visualization["target_class_name"] == "blight"
    assert visualization["grid_size"] == 3
    assert len(visualization["heatmap"]) == 3
    assert len(visualization["heatmap"][0]) == 3

    images = smoke.build_prediction_visualization_images(image_path, result)
    assert set(images) == {"model_view", "heatmap_overlay", "occlusion_overlay"}
    assert images["model_view"].size == images["occlusion_overlay"].size


def test_predict_single_image_can_attach_attention_visualization(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    image_path = tmp_path / "leaf.png"
    make_image(image_path, size=(12, 10), color="green")

    def zero_preprocess(_image, target_size=224):
        return torch.zeros(3, target_size, target_size)

    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAttentionAdapter(crop_name, device))
    monkeypatch.setattr(smoke, "preprocess_image", zero_preprocess)
    monkeypatch.setattr(smoke, "_target_size", lambda _env: 16)

    result = smoke.predict_single_image(
        image_path,
        "tomato",
        adapter_dir=asset_dir,
        device="cpu",
        explain_prediction=True,
        explanation_method="attention_map",
    )

    visualization = result["visualization"]
    assert visualization["status"] == "success"
    assert visualization["method"] == "attention_map"
    assert visualization["register_tokens"] == 1
    assert visualization["patch_tokens"] == 4
    assert visualization["grid_size"] == 2
    assert visualization["heatmap"] == [[0.0, 0.333333], [0.666667, 1.0]]

    images = smoke.build_prediction_visualization_images(image_path, result)
    assert set(images) == {"model_view", "heatmap_overlay", "occlusion_overlay"}
    assert images["model_view"].size == images["heatmap_overlay"].size



def test_predict_single_image_infers_crop_from_drive_export(monkeypatch, tmp_path: Path):
    asset_dir = _write_current_drive_adapter_export(tmp_path / "telemetry" / "run_123", crop_name="tomato")
    image_path = tmp_path / "leaf.png"
    make_image(image_path, size=(8, 8), color="green")

    def zero_preprocess(_image, target_size=224):
        return torch.zeros(3, target_size, target_size)

    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))
    monkeypatch.setattr(smoke, "preprocess_image", zero_preprocess)
    monkeypatch.setattr(smoke, "_target_size", lambda _env: 224)

    result = smoke.predict_single_image(image_path, None, adapter_dir=asset_dir, device="cpu")

    assert result["predicted_class"] == "blight"
    assert result["adapter_dir"] == str(asset_dir)



def test_prepare_view_tensor_resize_pad_and_center_crop_return_target_sized_tensors(tmp_path: Path):
    image_path = tmp_path / "leaf_rect.png"
    make_image(image_path, size=(40, 20), color="green")

    with Image.open(image_path) as image:
        resize_pad_tensor = smoke._prepare_view_tensor(image, target_size=32, view_name="resize_pad")
    with Image.open(image_path) as image:
        center_crop_tensor = smoke._prepare_view_tensor(image, target_size=32, view_name="center_crop")

    assert tuple(resize_pad_tensor.shape) == (3, 32, 32)
    assert tuple(center_crop_tensor.shape) == (3, 32, 32)



def test_predict_single_image_robust_mode_returns_ordered_views_and_full_resize_top_level(
    monkeypatch, tmp_path: Path
):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    image_path = tmp_path / "leaf.png"
    make_image(image_path, size=(8, 8), color="green")
    payloads = {
        0.11: _make_payload(
            extra_ood={
                "candidate_scores": {"ensemble": 0.2, "energy": 0.15},
                "candidate_thresholds": {"ensemble": 0.8, "energy": 0.75},
                "sure_confidence_reject": True,
                "sure_semantic_ood": False,
                "conformal_set": ["blight", "healthy"],
                "conformal_set_size": 2,
            }
        ),
        0.22: _make_payload(confidence=0.89),
        0.33: _make_payload(confidence=0.87),
    }

    monkeypatch.setattr(
        smoke,
        "_build_adapter",
        lambda crop_name, device: _ScriptedViewAdapter(crop_name, payloads, device),
    )
    monkeypatch.setattr(smoke, "_prepare_view_tensor", _scripted_view_tensor)
    monkeypatch.setattr(smoke, "_target_size", lambda _env: 16)

    result = smoke.predict_single_image(
        image_path,
        "tomato",
        adapter_dir=asset_dir,
        device="cpu",
        enable_robust_smoke=True,
        robust_views=("full_resize", "resize_pad", "center_crop"),
    )

    assert [row["view_name"] for row in result["views"]] == ["full_resize", "resize_pad", "center_crop"]
    assert result["predicted_class"] == "blight"
    assert result["confidence"] == 0.91
    assert result["raw_payload"]["status"] == "success"
    assert result["view_consistency"]["primary_view"] == "full_resize"
    assert result["view_consistency"]["stable"] is True
    assert result["uncertainty_diagnostics"]["candidate_scores"] == {"ensemble": 0.2, "energy": 0.15}
    assert result["uncertainty_diagnostics"]["candidate_thresholds"] == {"ensemble": 0.8, "energy": 0.75}
    assert result["uncertainty_diagnostics"]["sure_confidence_reject"] is True
    assert result["uncertainty_diagnostics"]["sure_semantic_ood"] is False
    assert result["uncertainty_diagnostics"]["conformal_set"] == ["blight", "healthy"]
    assert result["uncertainty_diagnostics"]["conformal_set_size"] == 2
    assert result["uncertainty_diagnostics"]["warning_codes"] == [
        "confidence_not_calibrated",
        "sure_confidence_reject",
        "conformal_set_wide",
    ]



def test_predict_single_image_robust_mode_flags_view_class_disagreement_and_keeps_full_resize_top_level(
    monkeypatch, tmp_path: Path
):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    image_path = tmp_path / "leaf.png"
    make_image(image_path, size=(8, 8), color="green")
    payloads = {
        0.11: _make_payload(predicted_class="blight", confidence=0.91),
        0.22: _make_payload(predicted_class="healthy", confidence=0.90),
        0.33: _make_payload(predicted_class="blight", confidence=0.89),
    }

    monkeypatch.setattr(
        smoke,
        "_build_adapter",
        lambda crop_name, device: _ScriptedViewAdapter(crop_name, payloads, device),
    )
    monkeypatch.setattr(smoke, "_prepare_view_tensor", _scripted_view_tensor)
    monkeypatch.setattr(smoke, "_target_size", lambda _env: 16)

    result = smoke.predict_single_image(
        image_path,
        "tomato",
        adapter_dir=asset_dir,
        device="cpu",
        enable_robust_smoke=True,
    )

    assert result["predicted_class"] == "blight"
    assert "view_class_disagreement" in result["view_consistency"]["warning_codes"]
    assert result["view_consistency"]["stable"] is False
    assert "view_instability" in result["uncertainty_diagnostics"]["warning_codes"]



def test_predict_single_image_robust_mode_flags_view_ood_disagreement(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    image_path = tmp_path / "leaf.png"
    make_image(image_path, size=(8, 8), color="green")
    payloads = {
        0.11: _make_payload(is_ood=False),
        0.22: _make_payload(is_ood=True),
        0.33: _make_payload(is_ood=False),
    }

    monkeypatch.setattr(
        smoke,
        "_build_adapter",
        lambda crop_name, device: _ScriptedViewAdapter(crop_name, payloads, device),
    )
    monkeypatch.setattr(smoke, "_prepare_view_tensor", _scripted_view_tensor)
    monkeypatch.setattr(smoke, "_target_size", lambda _env: 16)

    result = smoke.predict_single_image(
        image_path,
        "tomato",
        adapter_dir=asset_dir,
        device="cpu",
        enable_robust_smoke=True,
    )

    assert "view_ood_disagreement" in result["view_consistency"]["warning_codes"]



def test_predict_single_image_robust_mode_flags_large_confidence_spread(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    image_path = tmp_path / "leaf.png"
    make_image(image_path, size=(8, 8), color="green")
    payloads = {
        0.11: _make_payload(confidence=0.95),
        0.22: _make_payload(confidence=0.60),
        0.33: _make_payload(confidence=0.89),
    }

    monkeypatch.setattr(
        smoke,
        "_build_adapter",
        lambda crop_name, device: _ScriptedViewAdapter(crop_name, payloads, device),
    )
    monkeypatch.setattr(smoke, "_prepare_view_tensor", _scripted_view_tensor)
    monkeypatch.setattr(smoke, "_target_size", lambda _env: 16)

    result = smoke.predict_single_image(
        image_path,
        "tomato",
        adapter_dir=asset_dir,
        device="cpu",
        enable_robust_smoke=True,
    )

    assert "view_confidence_spread_high" in result["view_consistency"]["warning_codes"]
    assert result["view_consistency"]["confidence_spread"] == 0.35



def test_predict_single_image_robust_mode_keeps_error_row_when_primary_view_fails(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    image_path = tmp_path / "leaf.png"
    Image.new("RGB", (8, 8), color="green").save(image_path)

    def _prepare_with_primary_failure(_image, *, target_size: int, view_name: str):
        if view_name == "full_resize":
            raise ValueError("forced failure on primary view")
        return torch.zeros(3, target_size, target_size)

    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))
    monkeypatch.setattr(smoke, "_prepare_view_tensor", _prepare_with_primary_failure)
    monkeypatch.setattr(smoke, "_target_size", lambda _env: 16)

    result = smoke.predict_single_image(
        image_path,
        "tomato",
        adapter_dir=asset_dir,
        device="cpu",
        enable_robust_smoke=True,
    )

    assert result["status"] == "error"
    assert "forced failure on primary view" in result["error"]
    assert result["view_consistency"]["failed_views"] == ["full_resize"]
    assert result["uncertainty_diagnostics"]["status"] == "error"
    assert result["uncertainty_diagnostics"]["error"] == result["error"]
    assert "prediction_error" in result["uncertainty_diagnostics"]["warning_codes"]
    assert "view_instability" in result["uncertainty_diagnostics"]["warning_codes"]


def test_predict_image_folder_skips_non_images_and_records_errors(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    make_image(image_dir / "ok.png", size=(8, 8), color="green")
    (image_dir / "broken.png").write_text("not an image", encoding="utf-8")
    (image_dir / "notes.txt").write_text("skip me", encoding="utf-8")

    def zero_preprocess(_image, target_size=224):
        return torch.zeros(3, target_size, target_size)

    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))
    monkeypatch.setattr(smoke, "preprocess_image", zero_preprocess)
    monkeypatch.setattr(smoke, "_target_size", lambda _env: 224)

    rows = smoke.predict_image_folder(image_dir, "tomato", adapter_dir=asset_dir, device="cpu")

    assert len(rows) == 2
    assert {row["image_name"] for row in rows} == {"ok.png", "broken.png"}
    ok_row = next(row for row in rows if row["image_name"] == "ok.png")
    broken_row = next(row for row in rows if row["image_name"] == "broken.png")
    assert ok_row["predicted_class"] == "blight"
    assert ok_row["error"] == ""
    assert broken_row["status"] == "error"
    assert broken_row["error"]



def test_load_adapter_summary_raises_for_invalid_crop_metadata_json(monkeypatch, tmp_path: Path):
    export_root = tmp_path / "outputs" / "colab_notebook_training"
    asset_dir = _write_adapter_export(export_root)
    (export_root / "artifacts").mkdir(parents=True, exist_ok=True)
    (export_root / "artifacts" / "crop_info.json").write_text("{not-valid-json", encoding="utf-8")
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    try:
        smoke.load_adapter_summary(None, adapter_dir=asset_dir, device="cpu")
    except ValueError as exc:
        assert "Failed to parse JSON metadata" in str(exc)
        assert "crop_info.json" in str(exc)
    else:
        raise AssertionError("Expected load_adapter_summary to fail on invalid crop metadata.")


def test_discover_adapter_candidates_reads_current_drive_exports(tmp_path: Path):
    drive_root = tmp_path / "drive_root"
    asset_dir = _write_current_drive_adapter_export(drive_root / "telemetry" / "run_654", crop_name="tomato")

    candidates = smoke.discover_adapter_candidates([drive_root], crop_name="tomato")

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["adapter_dir"] == str(asset_dir)
    assert candidate["crop_name"] == "tomato"
    assert candidate["run_id"] == "run_654"



def test_discover_adapter_candidates_marks_metadata_warning_for_invalid_crop_metadata(tmp_path: Path):
    project_root = tmp_path / "project"
    export_root = project_root / "outputs" / "colab_notebook_training"
    asset_dir = _write_adapter_export(export_root)
    (export_root / "artifacts").mkdir(parents=True, exist_ok=True)
    (export_root / "artifacts" / "crop_info.json").write_text("{not-valid-json", encoding="utf-8")

    candidates = smoke.discover_adapter_candidates([project_root], crop_name=None)

    assert len(candidates) == 1
    assert candidates[0]["adapter_dir"] == str(asset_dir)
    assert candidates[0]["metadata_error"]
    assert "metadata-warning" in candidates[0]["display_name"]


def test_discover_adapter_candidates_collapses_same_run_mirrors_preferring_repo_output(tmp_path: Path):
    project_root = tmp_path / "project"
    run_id = "run_777"
    run_root = project_root / "runs" / run_id
    preferred_asset_dir = _write_adapter_export_with_crop_info(
        run_root / "outputs" / "colab_notebook_training",
        crop_name="tomato",
    )
    _write_current_drive_adapter_export(run_root / "telemetry", crop_name="tomato")
    _write_current_drive_adapter_export(run_root / "checkpoint_state", crop_name="tomato")

    drive_root = tmp_path / "drive_root"
    _write_current_drive_adapter_export(drive_root / "telemetry" / run_id, crop_name="tomato")

    candidates = smoke.discover_adapter_candidates([project_root, drive_root], crop_name=None)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["adapter_dir"] == str(preferred_asset_dir)
    assert candidate["crop_name"] == "tomato"
    assert candidate["run_id"] == run_id



def test_discover_adapter_candidates_keeps_nested_crop_part_runs_separate(tmp_path: Path):
    project_root = tmp_path / "project"
    first_run = "tomato_fruit_2026-04-21_20-07-45"
    second_run = "tomato_fruit_2026-04-22_09-13-11"
    first_asset_dir = _write_adapter_export_with_crop_info(
        project_root
        / "runs"
        / "tomato"
        / "fruit"
        / first_run
        / "outputs"
        / "colab_notebook_training"
        / "tomato"
        / "fruit",
        crop_name="tomato",
    )
    second_asset_dir = _write_adapter_export_with_crop_info(
        project_root
        / "runs"
        / "tomato"
        / "fruit"
        / second_run
        / "outputs"
        / "colab_notebook_training"
        / "tomato"
        / "fruit",
        crop_name="tomato",
    )

    candidates = smoke.discover_adapter_candidates([project_root], crop_name=None)

    assert len(candidates) == 2
    candidates_by_run = {candidate["run_id"]: candidate for candidate in candidates}
    assert set(candidates_by_run) == {first_run, second_run}
    assert candidates_by_run[first_run]["adapter_dir"] == str(first_asset_dir)
    assert candidates_by_run[second_run]["adapter_dir"] == str(second_asset_dir)
    assert "run=tomato_fruit_2026-04-22_09-13-11" in candidates_by_run[second_run]["display_name"]


def test_discover_adapter_candidates_keeps_same_run_crop_parts_separate(tmp_path: Path):
    project_root = tmp_path / "project"
    run_id = "tomato_2026-04-22_10-00-00"
    leaf_asset_dir = _write_adapter_export_with_crop_info(
        project_root
        / "runs"
        / "tomato"
        / "leaf"
        / run_id
        / "outputs"
        / "colab_notebook_training"
        / "tomato"
        / "leaf",
        crop_name="tomato",
    )
    fruit_asset_dir = _write_adapter_export_with_crop_info(
        project_root
        / "runs"
        / "tomato"
        / "fruit"
        / run_id
        / "outputs"
        / "colab_notebook_training"
        / "tomato"
        / "fruit",
        crop_name="tomato",
    )

    candidates = smoke.discover_adapter_candidates([project_root], crop_name=None)

    candidates_by_part = {candidate["part_name"]: candidate for candidate in candidates}
    assert set(candidates_by_part) == {"leaf", "fruit"}
    assert candidates_by_part["leaf"]["adapter_dir"] == str(leaf_asset_dir)
    assert candidates_by_part["fruit"]["adapter_dir"] == str(fruit_asset_dir)


def test_discover_adapter_candidates_skips_run_mirrors_with_mismatched_crop_part(tmp_path: Path):
    project_root = tmp_path / "project"
    run_id = "grape_leaf_2026-05-07_13-23-38"
    run_root = project_root / "runs" / "grape" / "leaf" / run_id
    valid_asset_dir = _write_adapter_export_with_crop_info(
        run_root
        / "telemetry"
        / "artifacts"
        / "adapter_export"
        / "grape"
        / "leaf",
        crop_name="grape",
    )
    _write_adapter_export_with_crop_info(
        run_root
        / "outputs"
        / "colab_notebook_training"
        / "grape"
        / "fruit",
        crop_name="grape",
    )

    candidates = smoke.discover_adapter_candidates([project_root], crop_name=None)

    assert len(candidates) == 1
    assert candidates[0]["adapter_dir"] == str(valid_asset_dir)
    assert candidates[0]["crop_name"] == "grape"
    assert candidates[0]["part_name"] == "leaf"



def test_discover_adapter_candidates_scans_project_root_and_skips_cache_dirs(tmp_path: Path):
    project_root = tmp_path / "project"
    asset_dir = _write_adapter_export_with_crop_info(
        project_root / "outputs" / "colab_notebook_training",
        crop_name="tomato",
    )
    _write_adapter_export(project_root / ".venv" / "ignored_export")

    candidates = smoke.discover_adapter_candidates([project_root], crop_name=None)

    assert len(candidates) == 1
    assert candidates[0]["adapter_dir"] == str(asset_dir)
    assert candidates[0]["crop_name"] == "tomato"



def test_discover_adapter_candidates_infers_crop_from_local_artifacts(tmp_path: Path):
    project_root = tmp_path / "project"
    export_root = project_root / "outputs" / "colab_notebook_training"
    asset_dir = _write_adapter_export(export_root)
    _write_training_summary(export_root / "artifacts", crop_name="tomato")

    candidates = smoke.discover_adapter_candidates([project_root], crop_name=None)

    assert len(candidates) == 1
    assert candidates[0]["adapter_dir"] == str(asset_dir)
    assert candidates[0]["crop_name"] == "tomato"



def test_discover_adapter_candidates_infers_crop_from_adapter_meta_classes(tmp_path: Path):
    project_root = tmp_path / "project"
    asset_dir = _write_adapter_export_with_prefixed_classes(project_root / "adapter_export")

    candidates = smoke.discover_adapter_candidates([project_root], crop_name=None)

    assert len(candidates) == 1
    assert candidates[0]["adapter_dir"] == str(asset_dir)
    assert candidates[0]["crop_name"] == "tomato"



def test_load_adapter_summary_accepts_crop_dir_as_adapter_root(monkeypatch, tmp_path: Path):
    crop_dir = tmp_path / "models" / "adapters" / "tomato"
    asset_dir = _write_adapter_export(crop_dir)
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary("tomato", adapter_root=crop_dir, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)



def test_load_adapter_summary_accepts_crop_dir_as_adapter_root_without_crop_name(
    monkeypatch, tmp_path: Path
):
    crop_dir = tmp_path / "models" / "adapters" / "tomato"
    asset_dir = _write_adapter_export(crop_dir)
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary(None, adapter_root=crop_dir, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["crop_name"] == "tomato"



def test_discover_adapter_candidates_skips_redundant_descendant_roots(tmp_path: Path, monkeypatch):
    scan_roots: list[Path] = []
    (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outputs" / "nested").mkdir(parents=True, exist_ok=True)

    def _recording_iter(root: Path):
        scan_roots.append(root)
        return iter(())

    monkeypatch.setattr(smoke, "_iter_adapter_meta_paths", _recording_iter)

    candidates = smoke.discover_adapter_candidates(
        search_roots=[tmp_path, tmp_path / "outputs", tmp_path / "outputs" / "nested"],
    )

    assert candidates == []
    assert scan_roots == [tmp_path]


def test_discover_adapter_candidates_skips_redundant_descendant_roots_even_when_child_comes_first(
    tmp_path: Path, monkeypatch
):
    scan_roots: list[Path] = []
    parent = tmp_path / "outputs"
    child = parent / "nested"
    child.mkdir(parents=True, exist_ok=True)

    def _recording_iter(root: Path):
        scan_roots.append(root)
        return iter(())

    monkeypatch.setattr(smoke, "_iter_adapter_meta_paths", _recording_iter)

    candidates = smoke.discover_adapter_candidates(
        search_roots=[child, parent],
    )

    assert candidates == []
    assert scan_roots == [parent]
