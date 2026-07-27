"""Bi-directional Energy Regularization (BER) loss for Class-Incremental Learning.

BER counteracts the bias toward newly added classes and OOD samples that
develops during sequential task learning.  It applies energy regularization
jointly to both old and new classes via independent margin-based penalties.

Reference: OpenCIL benchmark — BER for robust class-incremental OOD detection.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BERLoss(nn.Module):
    """Composable loss wrapper: CrossEntropy + Bi-directional Energy Regularization.

    Energy margins are maintained as exponential moving averages of the
    per-partition energy statistics, updated each forward pass.

    Args:
        lambda_old: Regularization weight for old-class energy penalty.
        lambda_new: Regularization weight for new-class energy penalty.
        num_old_classes: Number of classes known before the current incremental task.
            Set to 0 for the very first task (BER terms deactivate gracefully).
        ema_momentum: Smoothing factor for margin EMA updates.
    """

    def __init__(
        self,
        lambda_old: float = 0.1,
        lambda_new: float = 0.1,
        num_old_classes: int = 0,
        ema_momentum: float = 0.9,
        warmup_steps: int = 50,
    ) -> None:
        super().__init__()
        self.lambda_old = float(lambda_old)
        self.lambda_new = float(lambda_new)
        self.num_old_classes = int(num_old_classes)
        self.ema_momentum = float(ema_momentum)
        self.warmup_steps = int(max(0, warmup_steps))

        # Running energy margins (will be initialized on first forward pass)
        self._margin_old: torch.Tensor
        self._margin_new: torch.Tensor
        self.register_buffer("_margin_old", torch.tensor(0.0))
        self.register_buffer("_margin_new", torch.tensor(0.0))
        self._margins_initialized = False
        self._forward_steps = 0

    @staticmethod
    def _energy(logits: torch.Tensor) -> torch.Tensor:
        """Free energy: E(x) = −logsumexp(logits)."""
        return -torch.logsumexp(logits, dim=1)

    def _update_margins(self, energy_old: torch.Tensor, energy_new: torch.Tensor) -> None:
        """Update running EMA margins from batch energy statistics."""
        momentum = self.ema_momentum

        if energy_old.numel() > 0:
            batch_margin_old = energy_old.mean().detach()
            if self._margins_initialized:
                self._margin_old = momentum * self._margin_old + (1 - momentum) * batch_margin_old
            else:
                self._margin_old = batch_margin_old.clone()

        if energy_new.numel() > 0:
            batch_margin_new = energy_new.mean().detach()
            if self._margins_initialized:
                self._margin_new = momentum * self._margin_new + (1 - momentum) * batch_margin_new
            else:
                self._margin_new = batch_margin_new.clone()

        if energy_old.numel() > 0 or energy_new.numel() > 0:
            self._margins_initialized = True

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_smoothing: float = 0.0,
        class_weight: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute CE + BER loss.

        Args:
            logits: Model output logits ``[N, C]``.
            labels: Ground-truth labels ``[N]``.
            label_smoothing: Label smoothing factor for CE.
            class_weight: Optional normalized class weights for CE.

        Returns:
            ``(total_loss, {"ce": ..., "ber_old": ..., "ber_new": ...})``
        """
        ce_loss = F.cross_entropy(logits, labels, weight=class_weight, label_smoothing=label_smoothing)

        energies = self._energy(logits)
        old_mask = labels < self.num_old_classes
        new_mask = labels >= self.num_old_classes

        energy_old = energies[old_mask]
        energy_new = energies[new_mask]

        # Update EMA margins
        self._update_margins(energy_old, energy_new)

        # BER is only meaningful once we have incremental classes and warmed-up margins.
        apply_ber = bool(
            self.num_old_classes > 0
            and self._margins_initialized
            and self._forward_steps >= self.warmup_steps
        )

        # BER old-class term: penalize energy drifting above margin
        # (prevents old-class representations from becoming overconfident / collapsing)
        ber_old_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if apply_ber and energy_old.numel() > 0 and self.lambda_old > 0:
            margin_old = self._margin_old.detach()
            ber_old_loss = self.lambda_old * torch.relu(energy_old - margin_old).pow(2).mean()

        # BER new-class term: penalize energy being too low
        # (prevents new classes from dominating the energy landscape)
        ber_new_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if apply_ber and energy_new.numel() > 0 and self.lambda_new > 0:
            margin_new = self._margin_new.detach()
            ber_new_loss = self.lambda_new * torch.relu(margin_new - energy_new).pow(2).mean()

        total_loss = ce_loss + ber_old_loss + ber_new_loss
        self._forward_steps += 1

        return total_loss, {
            "ce": float(ce_loss.detach().item()),
            "ber_old": float(ber_old_loss.detach().item()),
            "ber_new": float(ber_new_loss.detach().item()),
        }
