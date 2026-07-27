import pytest
import torch
import torch.nn as nn

from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer
from src.training.session import ContinualTrainingSession


class IdentityModule(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def _build_minimal_trainer(
    *,
    num_epochs: int = 2,
    grad_accumulation_steps: int = 1,
) -> tuple[ContinualSDLoRATrainer, nn.Parameter]:
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
        num_epochs=num_epochs,
        grad_accumulation_steps=grad_accumulation_steps,
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.class_to_idx = {"healthy": 0}
    trainer.adapter_model = IdentityModule()
    trainer.classifier = nn.Linear(1, 1, bias=False)
    trainer.fusion = IdentityModule()
    trainer.classifier.weight.data.fill_(1.0)
    trainable = trainer.classifier.weight
    trainer.optimizer = torch.optim.SGD([trainable], lr=0.1)
    trainer.training_step = lambda _batch: (trainable ** 2).sum()  # type: ignore[assignment]
    trainer.forward_logits = lambda images: torch.zeros(images.shape[0], 2)  # type: ignore[assignment]
    return trainer, trainable


def _make_loader(batch_count: int) -> list[dict[str, torch.Tensor]]:
    return [
        {"images": torch.zeros(1, 3, 8, 8), "labels": torch.zeros(1, dtype=torch.long)}
        for _ in range(batch_count)
    ]


def test_session_run_emits_observer_events_and_history():
    trainer, _ = _build_minimal_trainer()
    events = []
    session = ContinualTrainingSession(
        trainer,
        _make_loader(2),
        1,
        val_loader=_make_loader(1),
        observers=[events.append],
        run_id="run_1",
    )

    history = session.run()

    assert len(history.train_loss) == 1
    assert len(history.val_loss) == 1
    assert history.global_step == 2
    event_types = [event["event_type"] for event in events]
    assert event_types.count("batch_end") == 2
    assert event_types.count("epoch_end") == 1


def test_session_sets_preferred_ood_calibration_loader_on_trainer():
    trainer, _ = _build_minimal_trainer()
    train_loader = _make_loader(2)
    val_loader = _make_loader(1)

    _ = ContinualTrainingSession(
        trainer,
        train_loader,
        1,
        val_loader=val_loader,
    )

    assert trainer._ood_calibration_loader is val_loader


def test_session_resume_continues_global_step_and_resume_epoch():
    trainer, _ = _build_minimal_trainer()
    session = ContinualTrainingSession(
        trainer,
        _make_loader(2),
        2,
        resume_state={
            "progress_state": {"epoch": 1, "global_step": 7, "elapsed_sec": 2.5},
            "history": {
                "train_loss": [0.3],
                "val_loss": [],
                "val_accuracy": [],
                "macro_f1": [],
                "weighted_f1": [],
                "balanced_accuracy": [],
                "generalization_gap": [],
                "per_class_accuracy": [],
                "worst_classes": [],
            },
        },
    )

    history = session.run()

    assert history.global_step >= 9
    assert history.resume_start_epoch == 1
    assert len(history.train_loss) >= 2


def test_session_honors_stop_policy():
    trainer, _ = _build_minimal_trainer()
    stop_flag = {"value": False}
    events = []

    def observer(event):
        events.append(event)
        if event["event_type"] == "batch_end":
            stop_flag["value"] = True

    session = ContinualTrainingSession(
        trainer,
        _make_loader(2),
        2,
        observers=[observer],
        stop_policy=lambda: stop_flag["value"],
    )

    history = session.run()

    assert history.stopped_early is True
    event_types = [event["event_type"] for event in events]
    assert event_types.count("batch_end") == 1
    assert "stop_requested" in event_types


def test_session_batch_event_exposes_loss_and_checkpoint_request():
    trainer, _ = _build_minimal_trainer()
    events = []
    session = ContinualTrainingSession(
        trainer,
        _make_loader(2),
        1,
        observers=[events.append],
        checkpoint_every_n_steps=1,
    )

    session.run()

    batch_events = [event for event in events if event["event_type"] == "batch_end"]
    assert batch_events
    assert "loss" in batch_events[0]["payload"]
    assert "batch_loss" not in batch_events[0]["payload"]

    checkpoint_events = [event for event in events if event["event_type"] == "checkpoint_requested"]
    assert checkpoint_events
    assert checkpoint_events[0]["payload"]["reason"] == "batch_interval"


def test_session_snapshot_includes_best_metric_state():
    trainer, _ = _build_minimal_trainer()
    session = ContinualTrainingSession(
        trainer,
        _make_loader(1),
        1,
        val_loader=_make_loader(1),
    )

    session.run()
    snapshot = session.snapshot_state()

    assert "best_metric_state" in snapshot


def test_session_can_skip_intermediate_validation_epochs():
    trainer, _ = _build_minimal_trainer(num_epochs=3)
    events = []
    session = ContinualTrainingSession(
        trainer,
        _make_loader(1),
        3,
        val_loader=_make_loader(1),
        observers=[events.append],
        validation_every_n_epochs=2,
    )

    history = session.run()

    validation_events = [event for event in events if event["event_type"] == "validation_end"]
    epoch_events = [event for event in events if event["event_type"] == "epoch_end"]
    assert len(validation_events) == 2
    assert len(history.val_loss) == 2
    assert epoch_events[0]["payload"]["validation_skipped"] is True
    assert epoch_events[1]["payload"]["validation_skipped"] is False
    assert epoch_events[2]["payload"]["validation_skipped"] is False


def test_session_emits_exception_checkpoint_request():
    trainer, _ = _build_minimal_trainer()
    trainer.train_batch = lambda _batch: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[assignment]
    events = []
    session = ContinualTrainingSession(
        trainer,
        _make_loader(1),
        1,
        observers=[events.append],
        checkpoint_on_exception=True,
    )

    with pytest.raises(RuntimeError, match="boom"):
        session.run()

    event_types = [event["event_type"] for event in events]
    assert "training_aborted" in event_types
    checkpoint_events = [event for event in events if event["event_type"] == "checkpoint_requested"]
    assert checkpoint_events
    assert checkpoint_events[-1]["payload"]["reason"] == "exception"


def test_session_flushes_final_accumulation_step():
    trainer, trainable = _build_minimal_trainer(num_epochs=1, grad_accumulation_steps=4)
    session = ContinualTrainingSession(
        trainer,
        _make_loader(2),
        1,
    )

    history = session.run()

    assert history.optimizer_steps == 1
    assert trainable.item() != pytest.approx(1.0)


def test_train_batch_computes_grad_norm_once_when_step_is_applied():
    trainer, _ = _build_minimal_trainer(num_epochs=1, grad_accumulation_steps=1)
    calls = []

    trainer._reported_grad_norm = lambda *, gradients_unscaled: calls.append("reported") or 0.1  # type: ignore[assignment]
    trainer._compute_grad_norm = lambda: calls.append("compute") or 0.2  # type: ignore[assignment]

    stats = trainer.train_batch(_make_loader(1)[0])

    assert stats.optimizer_step_applied is True
    assert calls == ["compute"]


def test_train_batch_skips_grad_norm_scan_on_non_step_microbatch():
    trainer, _ = _build_minimal_trainer(num_epochs=1, grad_accumulation_steps=2)
    calls = []

    trainer._compute_grad_norm = lambda: calls.append("compute") or 0.2  # type: ignore[assignment]

    first_stats = trainer.train_batch(_make_loader(1)[0])
    second_stats = trainer.train_batch(_make_loader(1)[0])

    assert first_stats.optimizer_step_applied is False
    assert first_stats.grad_norm == pytest.approx(0.0)
    assert second_stats.optimizer_step_applied is True
    assert second_stats.grad_norm == pytest.approx(0.2)
    assert calls == ["compute"]


def test_session_resume_matches_uninterrupted_training_mid_epoch():
    uninterrupted_trainer, uninterrupted_param = _build_minimal_trainer(num_epochs=1, grad_accumulation_steps=2)
    uninterrupted_session = ContinualTrainingSession(uninterrupted_trainer, _make_loader(3), 1)
    uninterrupted_history = uninterrupted_session.run()

    resume_trainer, _ = _build_minimal_trainer(num_epochs=1, grad_accumulation_steps=2)
    captured = {}
    stop_flag = {"value": False}

    holder = {"session": None}

    def observer(event):
        if event["event_type"] != "batch_end" or int(event["payload"]["global_step"]) != 1:
            return
        captured["session_state"] = holder["session"].snapshot_state()
        captured["trainer_state"] = resume_trainer.snapshot_training_state()
        stop_flag["value"] = True

    interrupted_session = ContinualTrainingSession(
        resume_trainer,
        _make_loader(3),
        1,
        observers=[observer],
        stop_policy=lambda: stop_flag["value"],
    )
    holder["session"] = interrupted_session
    interrupted_session.run()

    resumed_trainer, resumed_param = _build_minimal_trainer(num_epochs=1, grad_accumulation_steps=2)
    resumed_trainer.restore_training_state(captured["trainer_state"])
    resumed_session = ContinualTrainingSession(
        resumed_trainer,
        _make_loader(3),
        1,
        resume_state=captured["session_state"],
    )
    resumed_history = resumed_session.run()

    assert resumed_history.optimizer_steps == uninterrupted_history.optimizer_steps
    assert resumed_history.global_step == uninterrupted_history.global_step
    assert resumed_param.item() == pytest.approx(uninterrupted_param.item())
