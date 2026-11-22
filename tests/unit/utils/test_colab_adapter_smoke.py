import json
from pathlib import Path

import torch

from src.colab import adapter_smoke as smoke_impl
from src.colab import adapter_smoke_cli as smoke
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
    monkeypatch.setattr(smoke_impl, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke_impl.load_adapter_summary("tomato", adapter_dir=export_root, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["backbone_model_name"] == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert summary["class_names"] == ["healthy", "blight"]
    assert summary["target_modules_resolved"] == ["encoder.layer.0.attention.q_proj"]


def test_load_adapter_summary_accepts_telemetry_run_dir_and_infers_crop_from_readiness(
    monkeypatch, tmp_path: Path
):
    run_dir = tmp_path / "telemetry" / "run_456"
    asset_dir = _write_adapter_export(run_dir / "artifacts" / "adapter_export")
    _write_production_readiness(run_dir / "artifacts", crop_name="tomato")
    monkeypatch.setattr(smoke_impl, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke_impl.load_adapter_summary(None, adapter_dir=run_dir, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["crop_name"] == "tomato"


def test_predict_single_image_returns_notebook_payload(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    image_path = tmp_path / "leaf.png"
    make_image(image_path, size=(8, 8), color="green")

    def zero_preprocess(_image, target_size=224):
        return torch.zeros(3, target_size, target_size)

    monkeypatch.setattr(smoke_impl, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))
    monkeypatch.setattr(smoke_impl, "preprocess_image", zero_preprocess)
    monkeypatch.setattr(smoke_impl, "_target_size", lambda _env: 224)

    result = smoke_impl.predict_single_image(image_path, "tomato", adapter_dir=asset_dir, device="cpu")

    assert result["image_name"] == "leaf.png"
    assert result["predicted_class"] == "blight"
    assert result["confidence"] == 0.91
    assert result["is_ood"] is False
    assert result["raw_payload"]["status"] == "success"
    assert "views" not in result


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
        smoke_impl,
        "_build_adapter",
        lambda crop_name, device: _ScriptedViewAdapter(crop_name, payloads, device),
    )
    monkeypatch.setattr(smoke_impl, "_prepare_view_tensor", _scripted_view_tensor)
    monkeypatch.setattr(smoke_impl, "_target_size", lambda _env: 16)

    result = smoke_impl.predict_single_image(
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


def test_predict_image_folder_skips_non_images_and_records_errors(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    make_image(image_dir / "ok.png", size=(8, 8), color="green")
    (image_dir / "broken.png").write_text("not an image", encoding="utf-8")
    (image_dir / "notes.txt").write_text("skip me", encoding="utf-8")

    def zero_preprocess(_image, target_size=224):
        return torch.zeros(3, target_size, target_size)

    monkeypatch.setattr(smoke_impl, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))
    monkeypatch.setattr(smoke_impl, "preprocess_image", zero_preprocess)
    monkeypatch.setattr(smoke_impl, "_target_size", lambda _env: 224)

    rows = smoke.predict_image_folder(image_dir, "tomato", adapter_dir=asset_dir, device="cpu")

    assert len(rows) == 2
    assert {row["image_name"] for row in rows} == {"ok.png", "broken.png"}
    ok_row = next(row for row in rows if row["image_name"] == "ok.png")
    broken_row = next(row for row in rows if row["image_name"] == "broken.png")
    assert ok_row["predicted_class"] == "blight"
    assert ok_row["error"] == ""
    assert broken_row["status"] == "error"
    assert broken_row["error"]


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
