from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer
from src.training.validation import (
    evaluate_model,
    evaluate_model_with_artifact_metrics,
    evaluate_model_with_predictions,
)


class IdentityModule(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class FakeOODDetector:
    sure_enabled = True
    conformal_enabled = True
    conformal_qhat = 0.1
    calibration_version = 1

    @staticmethod
    def calibration_issue():
        return None

    def score(self, features, logits, predicted_labels=None):
        del predicted_labels
        first_value = float(logits[0, 0].item())
        if first_value >= 5.0:
            ensemble = torch.tensor([0.1, 0.2], dtype=torch.float32, device=features.device)
            semantic = torch.tensor([False, False], dtype=torch.bool, device=features.device)
            confidence = torch.tensor([False, False], dtype=torch.bool, device=features.device)
        else:
            ensemble = torch.tensor([0.9, 1.1], dtype=torch.float32, device=features.device)
            semantic = torch.tensor([True, True], dtype=torch.bool, device=features.device)
            confidence = torch.tensor([True, True], dtype=torch.bool, device=features.device)
        return {
            "primary_score_method": "ensemble",
            "primary_score": ensemble,
            "candidate_scores": {"ensemble": ensemble},
            "sure_semantic_ood": semantic,
            "sure_confidence_reject": confidence,
        }

    def build_conformal_set(self, features, logits, idx_to_class):
        del features
        predicted = int(torch.argmax(logits).item())
        return [idx_to_class[predicted]]


class FakeOODLoader(list):
    def __init__(self, batches, image_paths):
        super().__init__(batches)
        self.dataset = type(
            "FakeOODDataset",
            (),
            {
                "image_paths": list(image_paths),
                "split": "ood",
            },
        )()


def test_evaluate_model_reports_expected_metrics():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.class_to_idx = {"healthy": 0, "disease_a": 1}
    trainer.adapter_model = IdentityModule()
    trainer.classifier = nn.Linear(4, 2)
    trainer.fusion = IdentityModule()
    trainer.forward_logits = lambda images: torch.tensor([[2.0, 0.0], [1.0, 0.0]])  # type: ignore[assignment]

    report = evaluate_model(
        trainer,
        [{"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)}],
    )

    assert report is not None
    assert report.val_loss >= 0.0
    assert report.val_accuracy == 0.5
    assert report.macro_precision == pytest.approx(0.25)
    assert report.macro_recall == pytest.approx(0.5)
    assert "healthy" in report.per_class_accuracy
    assert len(report.worst_classes) >= 1


def test_evaluate_model_with_predictions_returns_labels_and_predictions():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.class_to_idx = {"healthy": 0, "disease_a": 1}
    trainer.adapter_model = IdentityModule()
    trainer.classifier = nn.Linear(4, 2)
    trainer.fusion = IdentityModule()
    trainer.forward_logits = lambda images: torch.tensor([[2.0, 0.0], [1.0, 0.0]])  # type: ignore[assignment]

    result = evaluate_model_with_predictions(
        trainer,
        [{"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)}],
    )

    assert result is not None
    report, y_true, y_pred = result
    assert report.val_accuracy == 0.5
    assert y_true == [0, 1]
    assert y_pred == [0, 0]


def test_evaluate_model_with_artifact_metrics_collects_optional_ood_and_conformal_fields():
    trainer = type("FakeTrainer", (), {})()
    trainer.device = "cpu"
    trainer.class_to_idx = {"healthy": 0, "disease_a": 1}
    trainer.adapter_model = IdentityModule()
    trainer.classifier = IdentityModule()
    trainer.fusion = IdentityModule()
    trainer.ood_detector = FakeOODDetector()
    trainer.set_eval_mode = lambda: None
    trainer.encode = lambda images: images.float()
    trainer.forward_logits = lambda images: images.float()

    id_loader = [
        {"images": torch.tensor([[5.0, 1.0], [1.0, 5.0]]), "labels": torch.tensor([0, 1], dtype=torch.long)}
    ]
    ood_loader = [
        {"images": torch.tensor([[0.2, 0.1], [0.1, 0.2]]), "labels": torch.tensor([-1, -1], dtype=torch.long)}
    ]

    result = evaluate_model_with_artifact_metrics(trainer, id_loader, ood_loader=ood_loader)

    assert result is not None
    assert result.report.val_accuracy == 1.0
    assert result.ood_labels == [0, 0, 1, 1]
    assert result.ood_scores == pytest.approx([0.1, 0.2, 0.9, 1.1])
    assert result.ood_primary_score_method == "ensemble"
    assert result.ood_scores_by_method["ensemble"] == pytest.approx([0.1, 0.2, 0.9, 1.1])
    assert result.sure_ds_f1 == pytest.approx(1.0)
    assert result.conformal_empirical_coverage == pytest.approx(1.0)
    assert result.conformal_avg_set_size == pytest.approx(1.0)


def test_evaluate_model_with_artifact_metrics_keeps_ood_fields_empty_without_ood_loader():
    trainer = type("FakeTrainer", (), {})()
    trainer.device = "cpu"
    trainer.class_to_idx = {"healthy": 0, "disease_a": 1}
    trainer.adapter_model = IdentityModule()
    trainer.classifier = IdentityModule()
    trainer.fusion = IdentityModule()
    trainer.ood_detector = FakeOODDetector()
    trainer.set_eval_mode = lambda: None
    trainer.encode = lambda images: images.float()
    trainer.forward_logits = lambda images: images.float()

    id_loader = [
        {"images": torch.tensor([[5.0, 1.0], [1.0, 5.0]]), "labels": torch.tensor([0, 1], dtype=torch.long)}
    ]

    result = evaluate_model_with_artifact_metrics(trainer, id_loader)

    assert result is not None
    assert result.ood_labels is None
    assert result.ood_scores is None
    assert result.sure_ds_f1 is None
    assert result.conformal_empirical_coverage == pytest.approx(1.0)


def test_evaluate_model_with_artifact_metrics_builds_ood_type_breakdown():
    trainer = type("FakeTrainer", (), {})()
    trainer.device = "cpu"
    trainer.class_to_idx = {"healthy": 0, "disease_a": 1}
    trainer.adapter_model = IdentityModule()
    trainer.classifier = IdentityModule()
    trainer.fusion = IdentityModule()
    trainer.ood_detector = FakeOODDetector()
    trainer.set_eval_mode = lambda: None
    trainer.encode = lambda images: images.float()
    trainer.forward_logits = lambda images: images.float()

    id_loader = [{"images": torch.tensor([[5.0, 1.0], [1.0, 5.0]]), "labels": torch.tensor([0, 1], dtype=torch.long)}]
    ood_loader = FakeOODLoader(
        [{"images": torch.tensor([[0.2, 0.1], [0.1, 0.2]]), "labels": torch.tensor([-1, -1], dtype=torch.long)}],
        [
            Path("runtime/tomato/ood/blur/sample1.jpg"),
            Path("runtime/tomato/ood/non_plant/sample2.jpg"),
        ],
    )

    result = evaluate_model_with_artifact_metrics(trainer, id_loader, ood_loader=ood_loader)

    assert result is not None
    assert result.ood_type_breakdown["blur"]["sample_count"] == 1
    assert result.ood_type_breakdown["non_plant"]["sample_count"] == 1
    assert result.context["ood_types"] == ["blur", "non_plant"]


def test_evaluate_model_with_artifact_metrics_collects_prediction_rows_with_paths():
    trainer = type("FakeTrainer", (), {})()
    trainer.device = "cpu"
    trainer.class_to_idx = {"healthy": 0, "disease_a": 1}
    trainer.adapter_model = IdentityModule()
    trainer.classifier = IdentityModule()
    trainer.fusion = IdentityModule()
    trainer.ood_detector = FakeOODDetector()
    trainer.set_eval_mode = lambda: None
    trainer.encode = lambda images: images.float()
    trainer.forward_logits = lambda images: images.float()

    id_loader = FakeOODLoader(
        [{"images": torch.tensor([[5.0, 1.0], [1.0, 5.0]]), "labels": torch.tensor([0, 1], dtype=torch.long)}],
        [
            Path("runtime/tomato/val/healthy/sample1.jpg"),
            Path("runtime/tomato/val/disease_a/sample2.jpg"),
        ],
    )
    id_loader.dataset.split = "val"
    ood_loader = FakeOODLoader(
        [{"images": torch.tensor([[0.2, 0.1], [0.1, 0.2]]), "labels": torch.tensor([-1, -1], dtype=torch.long)}],
        [
            Path("runtime/tomato/ood/blur/sample3.jpg"),
            Path("runtime/tomato/ood/non_plant/sample4.jpg"),
        ],
    )

    result = evaluate_model_with_artifact_metrics(trainer, id_loader, ood_loader=ood_loader)

    assert result is not None
    assert len(result.prediction_rows) == 4
    assert result.prediction_rows[0]["image_path"].endswith("runtime/tomato/val/healthy/sample1.jpg")
    assert result.prediction_rows[0]["true_label"] == "healthy"
    assert result.prediction_rows[0]["pred_label"] == "healthy"
    assert result.prediction_rows[0]["sample_origin"] == "in_distribution"
    assert result.prediction_rows[2]["sample_origin"] == "ood"
    assert result.prediction_rows[2]["ood_type"] == "blur"
    assert result.prediction_rows[2]["true_label"] == ""


def test_evaluate_model_keeps_validation_loss_unweighted(monkeypatch):
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
        extra={"class_balance": {"weights_by_class": {"healthy": 0.8, "disease_a": 1.2}}},
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.class_to_idx = {"healthy": 0, "disease_a": 1}
    trainer._refresh_class_index_cache()
    trainer.adapter_model = IdentityModule()
    trainer.classifier = nn.Linear(4, 2)
    trainer.fusion = IdentityModule()
    trainer.forward_logits = lambda images: torch.tensor([[2.0, 0.0], [1.0, 0.0]])  # type: ignore[assignment]
    recorded = {}

    def fake_cross_entropy(logits, labels, weight=None, label_smoothing=0.0):
        recorded["weight"] = weight
        return torch.tensor(0.5)

    monkeypatch.setattr("src.training.validation.torch.nn.functional.cross_entropy", fake_cross_entropy)

    report = evaluate_model(
        trainer,
        [{"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)}],
    )

    assert report is not None
    assert recorded["weight"] is None
