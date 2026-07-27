import json
from pathlib import Path

import pytest
import torch

from src.adapter import independent_crop_adapter as adapter_module
from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.training.session import ContinualTrainingSession
from src.training.types import TrainingCheckpointPayload


class FakeOOD:
    def __init__(self):
        self.calibration_version = 2
        self.class_stats = {}

    def calibration_issue(self):
        if not self.class_stats:
            return "OOD detector has no calibrated class statistics."
        if self.calibration_version <= 0:
            return "OOD detector calibration version is unset."
        return None


class FakeCalibrationStats:
    def __init__(self):
        self.mean = torch.tensor([0.1, 0.2])
        self.var = torch.tensor([1.0, 1.5])
        self.mahalanobis_mu = 0.3
        self.mahalanobis_sigma = 0.4
        self.energy_mu = 0.5
        self.energy_sigma = 0.6
        self.threshold = 0.7
        self.sure_semantic_threshold = 0.8
        self.sure_confidence_threshold = 0.9


class FakeConfig:
    backbone_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    fusion_layers = [2, 5, 8, 11]
    fusion_output_dim = 768
    fusion_dropout = 0.1
    fusion_gating = "softmax"
    num_epochs = 3

    def as_contract_dict(self):
        return {"backbone": {"model_name": self.backbone_model_name}}


class FakeTrainerConfig:
    @classmethod
    def from_training_config(cls, _payload):
        return cls()


class FakeTrainer:
    def __init__(self, config):
        self.config = FakeConfig()
        self.class_to_idx = {}
        self.target_modules_resolved = ["transformer.block.0.linear"]
        self.ood_detector = FakeOOD()
        self.current_epoch = 0

    def initialize_engine(self, class_to_idx=None):
        self.class_to_idx = dict(class_to_idx or {})

    def add_classes(self, new_classes):
        for name in new_classes:
            if name not in self.class_to_idx:
                self.class_to_idx[name] = len(self.class_to_idx)
        return dict(self.class_to_idx)

    def snapshot_training_state(self):
        return TrainingCheckpointPayload(
            schema_version="v6_training_checkpoint",
            created_at="2026-03-07T00:00:00Z",
            trainer_config=self.config.as_contract_dict(),
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

    def calibrate_ood(self, loader):
        self.ood_detector.calibration_version += 1
        self.ood_detector.class_stats = {0: object()}
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
                "calibration_version": self.ood_detector.calibration_version,
            },
        }

    def save_adapter(self, output_dir):
        root = Path(output_dir) / "continual_sd_lora_adapter"
        root.mkdir(parents=True, exist_ok=True)
        (root / "classifier.pth").write_bytes(b"")
        (root / "fusion.pth").write_bytes(b"")
        return root

    def load_adapter(self, adapter_dir):
        meta = json.loads((Path(adapter_dir) / "adapter_meta.json").read_text(encoding="utf-8"))
        self.class_to_idx = dict(meta.get("class_to_idx", {}))
        return meta


def test_adapter_lifecycle_surface(monkeypatch):
    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    initialized = adapter.initialize_engine(class_names=["healthy"])
    assert initialized["status"] == "initialized"

    added = adapter.add_classes(["disease_a"])
    assert added["num_classes"] == 2

    pred = adapter.predict_with_ood(torch.zeros(3, 224, 224))
    assert {"score_method", "primary_score", "decision_threshold", "is_ood", "calibration_version"} <= set(
        pred["ood_analysis"].keys()
    )


def test_adapter_builds_training_session(monkeypatch):
    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    adapter.initialize_engine(class_names=["healthy"])

    session = adapter.build_training_session(
        train_loader=[{"images": torch.zeros(1, 3, 224, 224), "labels": torch.zeros(1, dtype=torch.long)}],
        observers=[],
        run_id="run_1",
    )

    assert isinstance(session, ContinualTrainingSession)
    assert adapter.trainer.class_to_idx == {"healthy": 0}


def test_adapter_builds_training_session_without_num_epochs_on_trainer_config(monkeypatch):
    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    adapter.initialize_engine(class_names=["healthy"])
    adapter.trainer.config = type("Cfg", (), {})()

    session = adapter.build_training_session(
        train_loader=[{"images": torch.zeros(1, 3, 224, 224), "labels": torch.zeros(1, dtype=torch.long)}],
    )

    assert isinstance(session, ContinualTrainingSession)
    assert session.num_epochs == 1


def test_adapter_save_load_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    adapter.initialize_engine(class_names=["healthy", "disease_a"])
    adapter.calibrate_ood([{"images": torch.zeros(1, 3, 224, 224), "labels": torch.zeros(1, dtype=torch.long)}])

    save_dir = tmp_path / "model_dir"
    adapter.save_adapter(str(save_dir))

    loaded = IndependentCropAdapter(crop_name="tomato", device="cpu")
    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))
    loaded.load_adapter(str(save_dir / "continual_sd_lora_adapter"))

    assert loaded.class_to_idx
    assert loaded.crop_name == "tomato"
    assert loaded.is_trained is True


def test_adapter_metadata_contains_required_contract_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    adapter.initialize_engine(class_names=["healthy", "disease_a"])
    adapter.calibrate_ood([{"images": torch.zeros(1, 3, 224, 224), "labels": torch.zeros(1, dtype=torch.long)}])

    save_dir = tmp_path / "model_dir"
    adapter.save_adapter(str(save_dir))

    meta_path = save_dir / "continual_sd_lora_adapter" / "adapter_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    required = {
        "schema_version",
        "engine",
        "crop_name",
        "backbone",
        "fusion",
        "class_to_idx",
        "ood_calibration",
        "target_modules_resolved",
    }
    assert required <= set(meta.keys())


def test_save_adapter_auto_calibrates_from_training_session_loader(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    adapter.initialize_engine(class_names=["healthy"])
    train_loader = [{"images": torch.zeros(1, 3, 224, 224), "labels": torch.zeros(1, dtype=torch.long)}]
    adapter.build_training_session(train_loader=train_loader)

    save_dir = tmp_path / "auto_calibrated_model"
    adapter.save_adapter(str(save_dir))

    assert adapter.ood_calibration_version == 3


def test_adapter_metadata_golden_contract_with_exportable_ood_state(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    adapter.initialize_engine(class_names=["healthy"])
    adapter.trainer.ood_detector.class_stats = {0: FakeCalibrationStats()}
    adapter.trainer.ood_detector.calibration_version = 5

    save_dir = tmp_path / "golden_model"
    adapter.save_adapter(str(save_dir))
    meta = json.loads((save_dir / "continual_sd_lora_adapter" / "adapter_meta.json").read_text(encoding="utf-8"))

    assert meta["schema_version"] == "v6"
    assert meta["engine"] == "continual_sd_lora"
    assert meta["crop_name"] == "tomato"
    assert meta["class_to_idx"] == {"healthy": 0}
    assert meta["ood_calibration"]["version"] == 5
    assert meta["ood_state"]["primary_score_method"] == "ensemble"
    assert meta["ood_state"]["knn_k"] == 10
    assert meta["ood_state"]["class_stats"]["0"]["mean"] == [0.10000000149011612, 0.20000000298023224]
    assert meta["ood_state"]["class_stats"]["0"]["threshold"] == 0.7


def test_save_adapter_merges_export_metadata_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    adapter.initialize_engine(class_names=["healthy"])
    adapter.trainer.ood_detector.class_stats = {0: FakeCalibrationStats()}
    adapter.trainer.ood_detector.calibration_version = 4
    adapter.set_export_metadata(
        ood_calibration={
            "source_split": "val",
            "source_loader_size": 8,
            "primary_score_method": "energy",
            "selection_source": "real_ood_split",
        },
        adapter_runtime={"best_state_restored": True},
    )

    save_dir = tmp_path / "merged_metadata_model"
    adapter.save_adapter(str(save_dir))
    meta = json.loads((save_dir / "continual_sd_lora_adapter" / "adapter_meta.json").read_text(encoding="utf-8"))

    assert meta["ood_calibration"]["version"] == 4
    assert meta["ood_calibration"]["source_split"] == "val"
    assert meta["ood_calibration"]["source_loader_size"] == 8
    assert meta["ood_calibration"]["primary_score_method"] == "energy"
    assert meta["adapter_runtime"]["best_state_restored"] is True


def test_save_adapter_raises_without_loader_when_ood_uncalibrated(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    adapter.initialize_engine(class_names=["healthy"])

    with pytest.raises(RuntimeError, match="No calibration loader is available for automatic export calibration"):
        adapter.save_adapter(str(tmp_path / "missing_loader"))


def test_adapter_training_checkpoint_passthrough(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    adapter.initialize_engine(class_names=["healthy"])

    payload = adapter.save_training_checkpoint(
        str(tmp_path / "ckpt"),
        session_state={
            "run_id": "run_1",
            "progress_state": {"epoch": 1, "global_step": 10},
            "history": {"train_loss": [0.1]},
        },
        run_id="run_1",
    )
    assert (payload / "checkpoint_meta.json").exists()

    loaded = adapter.load_training_checkpoint(str(payload))
    assert loaded["progress_state"]["global_step"] == 10
    assert loaded["history"]["train_loss"] == [0.1]


def test_load_adapter_rejects_crop_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    adapter.initialize_engine(class_names=["healthy"])
    adapter.calibrate_ood([{"images": torch.zeros(1, 3, 224, 224), "labels": torch.zeros(1, dtype=torch.long)}])

    save_dir = tmp_path / "mismatched_model"
    adapter.save_adapter(str(save_dir))

    loaded = IndependentCropAdapter(crop_name="potato", device="cpu")
    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))

    with pytest.raises(ValueError, match="Adapter crop mismatch"):
        loaded.load_adapter(str(save_dir / "continual_sd_lora_adapter"))
