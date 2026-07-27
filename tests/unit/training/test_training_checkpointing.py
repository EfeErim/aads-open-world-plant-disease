import torch
import torch.nn as nn

from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer
from src.training.session import ContinualTrainingSession


class IdentityModule(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def _build_minimal_trainer() -> tuple[ContinualSDLoRATrainer, nn.Parameter]:
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
        num_epochs=2,
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.class_to_idx = {"healthy": 0}
    trainer.adapter_model = IdentityModule()
    trainer.classifier = IdentityModule()
    trainer.fusion = IdentityModule()
    trainable = nn.Parameter(torch.tensor([1.0], requires_grad=True))
    trainer.optimizer = torch.optim.SGD([trainable], lr=0.1)
    trainer.training_step = lambda _batch: (trainable ** 2).sum()  # type: ignore[assignment]
    return trainer, trainable


def _make_loader(batch_count: int) -> list[dict[str, torch.Tensor]]:
    return [
        {"images": torch.zeros(1, 3, 8, 8), "labels": torch.zeros(1, dtype=torch.long)}
        for _ in range(batch_count)
    ]


def test_training_snapshot_roundtrip_preserves_current_epoch():
    trainer, _ = _build_minimal_trainer()
    trainer.current_epoch = 1
    trainer.optimizer_steps = 3
    trainer.best_metric_state = {"best_metric_name": "val_loss", "best_metric_value": 0.2}

    snapshot = trainer.snapshot_training_state()

    resumed, _ = _build_minimal_trainer()
    payload = resumed.restore_training_state(snapshot)

    assert payload.current_epoch == 1
    assert resumed.current_epoch == 1
    assert resumed.class_to_idx == {"healthy": 0}
    assert payload.optimizer_steps == 3
    assert resumed.best_metric_state["best_metric_name"] == "val_loss"
    assert snapshot.config_hash


def test_training_snapshot_roundtrip_preserves_scheduler_state():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
        scheduler_name="linear",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.class_to_idx = {"healthy": 0}
    trainer.adapter_model = IdentityModule()
    trainer.classifier = IdentityModule()
    trainer.fusion = IdentityModule()
    trainable = nn.Parameter(torch.tensor([1.0], requires_grad=True))
    trainer.optimizer = torch.optim.SGD([trainable], lr=0.1)
    trainer.configure_training_plan(total_batches=2, num_epochs=1)
    trainer._ensure_scheduler()
    assert trainer.scheduler is not None
    trainer.scheduler.step()

    snapshot = trainer.snapshot_training_state()

    resumed = ContinualSDLoRATrainer(cfg)
    resumed.class_to_idx = {"healthy": 0}
    resumed.adapter_model = IdentityModule()
    resumed.classifier = IdentityModule()
    resumed.fusion = IdentityModule()
    resumed.optimizer = torch.optim.SGD([nn.Parameter(torch.tensor([1.0], requires_grad=True))], lr=0.1)
    resumed.configure_training_plan(total_batches=2, num_epochs=1)
    payload = resumed.restore_training_state(snapshot)

    assert payload.scheduler_state is not None
    assert resumed.scheduler is not None


def test_session_snapshot_contains_progress_and_history():
    trainer, _ = _build_minimal_trainer()
    session = ContinualTrainingSession(trainer, _make_loader(2), 1, run_id="run_ckpt")
    history = session.run()

    payload = session.snapshot_state()

    assert payload["run_id"] == "run_ckpt"
    assert payload["progress_state"]["global_step"] == history.global_step
    assert payload["history"]["train_loss"] == history.train_loss
