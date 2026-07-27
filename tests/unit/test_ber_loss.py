"""Unit tests for BERLoss (Bi-directional Energy Regularization)."""

from __future__ import annotations

import torch

from src.training.ber_loss import BERLoss


class TestBERLoss:
    def test_basic_forward(self):
        loss_fn = BERLoss(lambda_old=0.1, lambda_new=0.1, num_old_classes=2)
        logits = torch.randn(8, 4, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        total_loss, components = loss_fn(logits, labels)
        assert total_loss.requires_grad
        assert "ce" in components
        assert "ber_old" in components
        assert "ber_new" in components
        assert total_loss.item() >= components["ce"]

    def test_first_task_no_old_classes(self):
        """With num_old_classes=0, all classes are 'new' — ber_old should be 0."""
        loss_fn = BERLoss(lambda_old=0.1, lambda_new=0.1, num_old_classes=0)
        logits = torch.randn(8, 3, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        _, components = loss_fn(logits, labels)
        assert components["ber_old"] == 0.0

    def test_all_old_classes(self):
        """When all classes are old, ber_new should be 0."""
        loss_fn = BERLoss(lambda_old=0.1, lambda_new=0.1, num_old_classes=5)
        logits = torch.randn(8, 5, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
        _, components = loss_fn(logits, labels)
        assert components["ber_new"] == 0.0

    def test_gradient_flows(self):
        loss_fn = BERLoss(lambda_old=0.1, lambda_new=0.1, num_old_classes=2)
        logits = torch.randn(8, 4, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        total_loss, _ = loss_fn(logits, labels)
        total_loss.backward()
        assert logits.grad is not None
        assert logits.grad.norm().item() > 0

    def test_zero_lambdas_equals_ce(self):
        loss_fn = BERLoss(lambda_old=0.0, lambda_new=0.0, num_old_classes=2)
        logits = torch.randn(8, 4, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        total_loss, components = loss_fn(logits, labels)
        assert components["ber_old"] == 0.0
        assert components["ber_new"] == 0.0
        assert abs(total_loss.item() - components["ce"]) < 1e-6

    def test_label_smoothing(self):
        loss_fn = BERLoss(lambda_old=0.1, lambda_new=0.1, num_old_classes=1)
        logits = torch.randn(4, 3, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 0])
        total_loss, _ = loss_fn(logits, labels, label_smoothing=0.1)
        assert total_loss.item() > 0

    def test_ema_margins_update(self):
        loss_fn = BERLoss(lambda_old=0.1, lambda_new=0.1, num_old_classes=2)
        logits = torch.randn(8, 4)
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        loss_fn(logits, labels)
        assert loss_fn._margins_initialized
        # Second pass should update via EMA
        loss_fn(logits, labels)
        # Margins may change (momentum blending)
        assert isinstance(loss_fn._margin_old.item(), float)

    def test_single_sample_per_partition(self):
        loss_fn = BERLoss(lambda_old=0.1, lambda_new=0.1, num_old_classes=1)
        logits = torch.randn(2, 2, requires_grad=True)
        labels = torch.tensor([0, 1])  # 1 old, 1 new
        total_loss, components = loss_fn(logits, labels)
        assert total_loss.item() > 0
