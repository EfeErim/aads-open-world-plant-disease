"""Integration tests for the kept Colab training surfaces."""

from pathlib import Path

import torch

from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.core.config_manager import ConfigurationManager
from src.training.continual_sd_lora import ContinualSDLoRAConfig
from src.training.session import ContinualTrainingSession
from src.training.types import TrainingCheckpointPayload


def test_colab_config_contains_minimal_contract():
    manager = ConfigurationManager(config_dir="config", environment="colab")
    cfg = manager.load_all_configs()

    assert {"training", "router", "colab", "inference"} <= set(cfg.keys())
    assert "ood" not in cfg
    continual = cfg["training"]["continual"]
    assert continual["backbone"]["model_name"] == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert "ood" in continual


def test_continual_config_rejects_low_bit_payload():
    try:
        ContinualSDLoRAConfig.from_training_config(
            {
                "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
                "adapter": {"target_modules_strategy": "all_linear_transformer", "lora_r": 4, "lora_alpha": 8},
                "fusion": {"layers": [2, 5, 8, 11]},
                "load_in_4bit": True,
            }
        )
        assert False, "low-bit payload should be rejected"
    except ValueError:
        assert True


def test_adapter_metadata_roundtrip_without_model_download(monkeypatch, tmp_path):
    from src.adapter import independent_crop_adapter as adapter_module

    class FakeTrainerConfig:
        @classmethod
        def from_training_config(cls, _payload):
            return cls()

    class FakeTrainer:
        def __init__(self, config):
            self.config = type(
                "Cfg",
                (),
                {
                    "backbone_model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                    "fusion_layers": [2, 5, 8, 11],
                    "fusion_output_dim": 768,
                    "fusion_dropout": 0.1,
                    "fusion_gating": "softmax",
                },
            )()
            self.class_to_idx = {}
            self.target_modules_resolved = ["transformer.block.0.linear"]
            self.ood_detector = type(
                "OOD",
                (),
                {
                    "calibration_version": 1,
                    "class_stats": {0: object()},
                    "calibration_issue": lambda self: None,
                },
            )()
            self.current_epoch = 0

        def initialize_engine(self, class_to_idx=None):
            self.class_to_idx = dict(class_to_idx or {})

        def add_classes(self, names):
            for name in names:
                if name not in self.class_to_idx:
                    self.class_to_idx[name] = len(self.class_to_idx)
            return dict(self.class_to_idx)

        def snapshot_training_state(self):
            return TrainingCheckpointPayload(
                schema_version="v6_training_checkpoint",
                created_at="2026-03-07T00:00:00Z",
                trainer_config={},
                class_to_idx=dict(self.class_to_idx),
                target_modules_resolved=list(self.target_modules_resolved),
                model_state={"adapter_model": {}, "classifier": {}, "fusion": {}},
                optimizer_state={},
                ood_state={},
                rng_state={},
                current_epoch=int(self.current_epoch),
            )

        def restore_training_state(self, payload):
            checkpoint = (
                payload
                if isinstance(payload, TrainingCheckpointPayload)
                else TrainingCheckpointPayload.from_dict(payload)
            )
            self.class_to_idx = dict(checkpoint.class_to_idx)
            return checkpoint

        def calibrate_ood(self, loader):
            return {"num_classes": float(len(self.class_to_idx))}

        def predict_with_ood(self, image):
            return {
                "status": "success",
                "disease": {"class_index": 0, "name": "healthy", "confidence": 0.9},
                "ood_analysis": {
                    "score_method": "ensemble",
                "primary_score": 0.2,
                "decision_threshold": 0.8,
                "candidate_scores": {"ensemble": 0.2},
                "candidate_thresholds": {"ensemble": 0.8},
                    "is_ood": False,
                    "calibration_version": 1,
                },
            }

        def save_adapter(self, output_dir):
            root = Path(output_dir) / "continual_sd_lora_adapter"
            root.mkdir(parents=True, exist_ok=True)
            return root

        def load_adapter(self, adapter_dir):
            return {}

    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    adapter.initialize_engine(class_names=["healthy"])
    adapter.add_classes(["disease_a"])
    session = adapter.build_training_session(
        train_loader=[{"images": torch.zeros(1, 3, 224, 224), "labels": torch.zeros(1, dtype=torch.long)}]
    )
    assert isinstance(session, ContinualTrainingSession)

    save_dir = tmp_path / "model"
    adapter.save_adapter(str(save_dir))

    reloaded = IndependentCropAdapter(crop_name="tomato", device="cpu")
    reloaded.load_adapter(str(save_dir / "continual_sd_lora_adapter"))
    assert reloaded.is_trained is True


def test_training_checkpoint_roundtrip_without_model_download(monkeypatch, tmp_path):
    from src.adapter import independent_crop_adapter as adapter_module

    class FakeTrainerConfig:
        @classmethod
        def from_training_config(cls, _payload):
            return cls()

    class FakeTrainer:
        def __init__(self, config):
            self.config = type("Cfg", (), {"num_epochs": 2, "as_contract_dict": lambda self: {}})()
            self.class_to_idx = {}
            self.target_modules_resolved = ["transformer.block.0.linear"]
            self.ood_detector = type(
                "OOD",
                (),
                {
                    "calibration_version": 0,
                    "class_stats": {},
                    "calibration_issue": lambda self: "OOD detector has no calibrated class statistics.",
                },
            )()
            self.current_epoch = 0

        def initialize_engine(self, class_to_idx=None):
            self.class_to_idx = dict(class_to_idx or {})

        def snapshot_training_state(self):
            return TrainingCheckpointPayload(
                schema_version="v6_training_checkpoint",
                created_at="2026-03-07T00:00:00Z",
                trainer_config={},
                class_to_idx=dict(self.class_to_idx),
                target_modules_resolved=list(self.target_modules_resolved),
                model_state={"adapter_model": {}, "classifier": {}, "fusion": {}},
                optimizer_state={},
                ood_state={},
                rng_state={},
                current_epoch=int(self.current_epoch),
            )

        def restore_training_state(self, payload):
            checkpoint = (
                payload
                if isinstance(payload, TrainingCheckpointPayload)
                else TrainingCheckpointPayload.from_dict(payload)
            )
            self.class_to_idx = dict(checkpoint.class_to_idx)
            self.current_epoch = int(checkpoint.current_epoch)
            return checkpoint

    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    adapter.initialize_engine(class_names=["healthy", "disease_a"])
    checkpoint_dir = adapter.save_training_checkpoint(
        str(tmp_path / "checkpoint"),
        session_state={
            "run_id": "run_42",
            "progress_state": {"epoch": 2, "global_step": 11},
            "history": {"train_loss": [0.3, 0.2]},
        },
        run_id="run_42",
    )

    reloaded = IndependentCropAdapter(crop_name="tomato", device="cpu")
    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))
    payload = reloaded.load_training_checkpoint(str(checkpoint_dir))

    assert payload["run_id"] == "run_42"
    assert payload["progress_state"]["global_step"] == 11
    assert payload["history"]["train_loss"] == [0.3, 0.2]
