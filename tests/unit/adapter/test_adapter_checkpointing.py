from src.adapter import independent_crop_adapter as adapter_module
from src.adapter.checkpointing import normalize_trainer_config
from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.training.types import TrainingCheckpointPayload


class FakeTrainerConfig:
    @classmethod
    def from_training_config(cls, _payload):
        return cls()


class FakeTrainer:
    def __init__(self, config):
        self.config = config
        self.class_to_idx = {"healthy": 0}
        self.target_modules_resolved = []
        self.ood_detector = type("OOD", (), {"calibration_version": 0})()
        self.current_epoch = 0

    def initialize_engine(self, class_to_idx=None):
        self.class_to_idx = dict(class_to_idx or {})

    def snapshot_training_state(self):
        return TrainingCheckpointPayload(
            schema_version="v6_training_checkpoint",
            created_at="2026-03-07T00:00:00Z",
            trainer_config={},
            class_to_idx=dict(self.class_to_idx),
            target_modules_resolved=[],
            model_state={"adapter_model": {}, "classifier": {}, "fusion": {}},
            optimizer_state={},
            ood_state={},
            rng_state={},
            current_epoch=0,
        )

    def restore_training_state(self, payload):
        checkpoint = (
            payload
            if isinstance(payload, TrainingCheckpointPayload)
            else TrainingCheckpointPayload.from_dict(payload)
        )
        self.class_to_idx = dict(checkpoint.class_to_idx)
        return checkpoint


def test_adapter_checkpoint_save_load(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    adapter.initialize_engine(class_names=["healthy"])

    saved = adapter.save_training_checkpoint(
        str(tmp_path / "run"),
        session_state={
            "run_id": "run_123",
            "progress_state": {"epoch": 2, "global_step": 42},
            "history": {"train_loss": [0.3, 0.2]},
            "best_metric_state": {"best_metric_name": "val_loss", "best_metric_value": 0.2},
        },
        run_id="run_123",
    )
    assert (saved / "checkpoint_meta.json").exists()

    loaded = adapter.load_training_checkpoint(str(saved))
    assert loaded["run_id"] == "run_123"
    assert loaded["progress_state"]["global_step"] == 42
    assert loaded["best_metric_state"]["best_metric_name"] == "val_loss"


def test_normalize_trainer_config_infers_cross_entropy_for_legacy_ber_export():
    normalized = normalize_trainer_config(
        {"ood": {"ber_enabled": True}},
        model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        device="cpu",
    )

    assert normalized["ood"]["ber_enabled"] is True
    assert normalized["optimization"]["loss_name"] == "cross_entropy"


def test_normalize_trainer_config_preserves_explicit_ber_loss_name():
    normalized = normalize_trainer_config(
        {
            "ood": {"ber_enabled": True},
            "optimization": {"loss_name": "logitnorm"},
        },
        model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        device="cpu",
    )

    assert normalized["ood"]["ber_enabled"] is True
    assert normalized["optimization"]["loss_name"] == "logitnorm"
