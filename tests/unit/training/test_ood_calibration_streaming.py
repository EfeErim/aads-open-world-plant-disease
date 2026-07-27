import pytest
import torch
import torch.nn as nn

from src.ood.continual_ood import ContinualOODDetector
from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer


class _IdentityModule(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def _build_trainer() -> ContinualSDLoRATrainer:
    trainer = ContinualSDLoRATrainer(
        ContinualSDLoRAConfig(
            backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
            target_modules_strategy="all_linear_transformer",
            fusion_layers=[2],
            fusion_output_dim=4,
            device="cpu",
        )
    )
    trainer.class_to_idx = {"healthy": 0, "disease_a": 1}
    trainer.adapter_model = _IdentityModule()
    trainer.fusion = _IdentityModule()
    trainer.classifier = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        trainer.classifier.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
    trainer.encode = lambda images: images.to(torch.float32)  # type: ignore[assignment]
    return trainer


def _make_loader():
    return [
        {
            "images": torch.tensor(
                [
                    [1.00, 0.10, 0.0, 0.0],
                    [1.10, 0.20, 0.0, 0.0],
                    [0.10, 1.00, 0.0, 0.0],
                    [0.20, 1.10, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        },
        {
            "images": torch.tensor(
                [
                    [0.95, 0.05, 0.0, 0.0],
                    [1.05, 0.15, 0.0, 0.0],
                    [0.05, 0.95, 0.0, 0.0],
                    [0.15, 1.05, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        },
    ]


def test_streamed_calibration_matches_materialized_reference_within_tolerance():
    trainer = _build_trainer()
    loader = _make_loader()

    all_features = torch.cat([batch["images"] for batch in loader], dim=0)
    all_labels = torch.cat([batch["labels"] for batch in loader], dim=0)
    all_logits = trainer.classifier(all_features)

    cfg = trainer.config
    reference = ContinualOODDetector(
        threshold_factor=cfg.ood_threshold_factor,
        energy_temperature=cfg.energy_temperature,
        energy_temperature_mode=cfg.energy_temperature_mode,
        energy_temperature_range=(cfg.energy_temperature_min, cfg.energy_temperature_max),
        energy_temperature_steps=cfg.energy_temperature_steps,
        radial_l2_enabled=cfg.radial_l2_enabled,
        radial_beta_range=(cfg.radial_beta_min, cfg.radial_beta_max),
        radial_beta_steps=cfg.radial_beta_steps,
        sure_enabled=cfg.sure_enabled,
        sure_semantic_percentile=cfg.sure_semantic_percentile,
        sure_confidence_percentile=cfg.sure_confidence_percentile,
        conformal_enabled=cfg.conformal_enabled,
        conformal_alpha=cfg.conformal_alpha,
    )
    reference.calibrate(features=all_features, logits=all_logits, labels=all_labels)

    calibration = trainer.calibrate_ood(loader)

    assert int(calibration["num_classes"]) == 2
    assert set(trainer.ood_detector.class_stats.keys()) == set(reference.class_stats.keys())

    for class_id, reference_stats in reference.class_stats.items():
        actual_stats = trainer.ood_detector.class_stats[class_id]
        assert torch.allclose(actual_stats.mean, reference_stats.mean, atol=1e-4, rtol=1e-4)
        assert torch.allclose(actual_stats.var, reference_stats.var, atol=1e-4, rtol=1e-4)
        assert actual_stats.mahalanobis_mu == pytest.approx(reference_stats.mahalanobis_mu, abs=1e-4)
        assert actual_stats.mahalanobis_sigma == pytest.approx(reference_stats.mahalanobis_sigma, abs=1e-4)
        assert actual_stats.energy_mu == pytest.approx(reference_stats.energy_mu, abs=1e-4)
        assert actual_stats.energy_sigma == pytest.approx(reference_stats.energy_sigma, abs=1e-4)
        assert actual_stats.threshold == pytest.approx(reference_stats.threshold, abs=1e-4)

    score_features = all_features[:4]
    score_logits = trainer.classifier(score_features)
    predicted_labels = all_labels[:4]
    actual_score = trainer.ood_detector.score(score_features, score_logits, predicted_labels=predicted_labels)
    reference_score = reference.score(score_features, score_logits, predicted_labels=predicted_labels)

    for key in ("mahalanobis_z", "energy_z", "primary_score", "decision_threshold"):
        assert torch.allclose(actual_score[key], reference_score[key], atol=1e-4, rtol=1e-4)
    assert torch.equal(actual_score["is_ood"], reference_score["is_ood"])


def test_one_shot_loader_falls_back_to_materialized_calibration(monkeypatch):
    trainer = _build_trainer()
    called = {"count": 0}

    def fake_calibrate(features, logits, labels):
        called["count"] += 1
        assert features.shape[0] == logits.shape[0] == labels.shape[0]
        return {"num_classes": 2.0, "calibration_version": 1.0}

    monkeypatch.setattr(trainer.ood_detector, "calibrate", fake_calibrate)

    class OneShotLoader:
        def __init__(self, batches):
            self._iterator = iter(batches)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._iterator)

    result = trainer.calibrate_ood(OneShotLoader(_make_loader()))

    assert called["count"] == 1
    assert int(result["num_classes"]) == 2


def test_reiterable_loader_with_extended_ood_uses_materialized_calibration(monkeypatch):
    trainer = _build_trainer()
    called = {"count": 0}

    def fake_calibrate(features, logits, labels):
        called["count"] += 1
        assert features.shape[0] == logits.shape[0] == labels.shape[0]
        return {"num_classes": 2.0, "calibration_version": 1.0}

    monkeypatch.setattr(trainer.ood_detector, "calibrate", fake_calibrate)

    result = trainer.calibrate_ood(_make_loader())

    assert called["count"] == 1
    assert int(result["num_classes"]) == 2


def test_streamed_calibration_raises_for_empty_loader():
    trainer = _build_trainer()

    with pytest.raises(ValueError, match="empty loader"):
        trainer.calibrate_ood([])
