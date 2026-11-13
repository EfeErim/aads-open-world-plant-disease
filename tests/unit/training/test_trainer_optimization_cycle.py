"""Characterization tests for the trainer's optimizer cycle.

They pin observable behaviour (optimizer steps, accumulation, scheduler learning rates, gradient clipping,
optimizer state across class expansion) through the trainer's public methods only.
"""

import pytest
import torch
import torch.nn as nn

from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer


def _trainer(**overrides) -> ContinualSDLoRATrainer:
    torch.manual_seed(0)
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
        learning_rate=0.1,
        weight_decay=0.0,
        **{"grad_accumulation_steps": 1, **overrides},
    )
    trainer = ContinualSDLoRATrainer(cfg)
    # No backbone download in unit tests: stand in small modules for the initialized engine.
    trainer._is_initialized = True
    trainer.adapter_model = nn.Linear(4, 4)
    trainer.fusion = nn.Linear(4, 4)
    trainer.classifier = nn.Linear(4, 2)
    trainer.class_to_idx = {"a": 0, "b": 1}
    trainer.forward_logits = lambda images, **_kwargs: trainer.classifier(  # type: ignore[assignment]
        trainer.fusion(trainer.adapter_model(images))
    )
    return trainer


def _batch() -> dict:
    return {"images": torch.randn(3, 4), "labels": torch.tensor([0, 1, 0])}


def _lr(trainer: ContinualSDLoRATrainer) -> float:
    assert trainer.optimizer is not None
    return float(trainer.optimizer.param_groups[0]["lr"])


def test_gradient_accumulation_steps_every_n_batches_and_flushes_the_remainder():
    trainer = _trainer(grad_accumulation_steps=2, scheduler_name="none")
    trainer.setup_optimizer()

    first = trainer.train_batch(_batch())
    assert first.optimizer_step_applied is False
    assert first.accumulation_step == 1
    assert trainer.has_pending_gradients() is True
    assert trainer.optimizer_steps == 0

    second = trainer.train_batch(_batch())
    assert second.optimizer_step_applied is True
    assert second.accumulation_step == 2
    assert second.optimizer_steps == 1
    assert second.grad_norm > 0.0
    assert trainer.has_pending_gradients() is False
    assert trainer.flush_pending_gradients() is None

    trainer.train_batch(_batch())
    flushed = trainer.flush_pending_gradients()
    assert flushed is not None and flushed > 0.0
    assert trainer.optimizer_steps == 2
    assert trainer.has_pending_gradients() is False


def test_non_step_microbatch_reports_the_last_measured_grad_norm():
    trainer = _trainer(grad_accumulation_steps=2, scheduler_name="none")
    trainer.setup_optimizer()
    trainer.train_batch(_batch())
    stepped = trainer.train_batch(_batch())

    microbatch = trainer.train_batch(_batch())

    assert microbatch.optimizer_step_applied is False
    assert microbatch.grad_norm == pytest.approx(stepped.grad_norm)


def test_train_batch_sets_up_the_optimizer_on_first_use():
    trainer = _trainer(scheduler_name="none")
    assert trainer.optimizer is None

    stats = trainer.train_batch(_batch())

    assert trainer.optimizer is not None
    assert stats.optimizer_step_applied is True
    assert stats.lr == pytest.approx(0.1)


def test_linear_scheduler_warms_up_then_decays_to_the_min_lr_per_optimizer_step():
    trainer = _trainer(
        scheduler_name="linear",
        scheduler_warmup_ratio=0.25,
        scheduler_min_lr=0.01,
        scheduler_step_on="batch",
    )
    trainer.configure_training_plan(total_batches=4, num_epochs=2)
    trainer.setup_optimizer()

    observed = [_lr(trainer)]
    for _ in range(8):
        trainer.train_batch(_batch())
        observed.append(_lr(trainer))

    # 8 optimizer steps planned, 2 warm-up steps, linear decay to min_lr / lr = 0.1.
    expected = [0.05, 0.1, 0.1, 0.085, 0.07, 0.055, 0.04, 0.025, 0.01]
    assert observed == pytest.approx(expected)


def test_cosine_scheduler_planned_per_epoch_ignores_batch_count():
    trainer = _trainer(scheduler_name="cosine", scheduler_min_lr=0.0, scheduler_step_on="epoch")
    trainer.configure_training_plan(total_batches=50, num_epochs=4)
    trainer.setup_optimizer()

    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert trainer.scheduler.T_max == 4
    trainer.train_batch(_batch())
    assert _lr(trainer) == pytest.approx(0.1)  # epoch-stepped schedulers do not move per batch


def test_scheduler_name_none_keeps_a_constant_learning_rate():
    trainer = _trainer(scheduler_name="none")
    trainer.configure_training_plan(total_batches=2, num_epochs=1)
    trainer.setup_optimizer()

    assert trainer.scheduler is None
    trainer.train_batch(_batch())
    assert _lr(trainer) == pytest.approx(0.1)


def test_max_grad_norm_clips_trainable_gradients_and_reports_the_clipped_norm():
    trainer = _trainer(scheduler_name="none", max_grad_norm=1e-3)
    trainer.setup_optimizer()

    stats = trainer.train_batch(_batch())

    assert stats.optimizer_step_applied is True
    assert 0.0 < stats.grad_norm <= 1e-3 * (1 + 1e-4)


def test_setup_optimizer_requires_an_initialized_engine():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)

    with pytest.raises(RuntimeError, match="initialize_engine"):
        trainer.setup_optimizer()


def test_add_classes_keeps_optimizer_hyperparameters_and_scheduler_position():
    trainer = _trainer(scheduler_name="linear", scheduler_warmup_ratio=0.0, scheduler_min_lr=0.0)
    trainer.configure_training_plan(total_batches=4, num_epochs=1)
    trainer.setup_optimizer()
    trainer.train_batch(_batch())
    adapter_weight = trainer.adapter_model.weight
    state_before = {k: v.clone() for k, v in trainer.optimizer.state[adapter_weight].items() if torch.is_tensor(v)}

    trainer.add_classes(["c"])

    assert trainer.classifier.out_features == 3
    # The scheduler keeps its position. Its learning rate is deliberately not pinned: rebuilding the LambdaLR
    # currently resets the group LR to the schedule's step-0 value until the next step, an open question.
    assert trainer.scheduler.last_epoch == 1
    state_after = trainer.optimizer.state[adapter_weight]
    for key, value in state_before.items():
        assert torch.equal(state_after[key], value)
    assert trainer.has_pending_gradients() is False
