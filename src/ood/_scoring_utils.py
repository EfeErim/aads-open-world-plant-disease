"""Shared OOD scoring helpers used across detector and calibration paths."""

from __future__ import annotations

from typing import Final, Iterable

import torch

MAHALANOBIS_WEIGHT: Final[float] = 0.6
ENERGY_WEIGHT: Final[float] = 0.4
_MIN_TEMPERATURE: Final[float] = 1e-3


def safe_std(value: torch.Tensor) -> torch.Tensor:
    return value.clamp_min(1e-6)


def normalize_temperature(value: float) -> float:
    return max(float(value), _MIN_TEMPERATURE)


def temperature_scale_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return logits / normalize_temperature(float(temperature))


def energy_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    resolved_temperature = normalize_temperature(float(temperature))
    scaled_logits = logits / resolved_temperature
    return -resolved_temperature * torch.logsumexp(scaled_logits, dim=1)


def select_temperature_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    candidate_temperatures: Iterable[float],
) -> tuple[float, float]:
    resolved_labels = labels.reshape(-1).to(dtype=torch.long, device=logits.device)
    if logits.ndim != 2 or logits.shape[0] != resolved_labels.shape[0] or logits.shape[0] <= 0:
        raise ValueError("Temperature calibration requires logits shaped [N, C] with matching non-empty labels.")

    best_temperature = 1.0
    best_nll = float("inf")
    last_error: RuntimeError | None = None
    for candidate in candidate_temperatures:
        temperature = normalize_temperature(float(candidate))
        try:
            loss = torch.nn.functional.cross_entropy(
                temperature_scale_logits(logits, temperature),
                resolved_labels,
            )
        except RuntimeError as exc:
            last_error = exc
            continue
        loss_value = float(loss.item())
        if loss_value < best_nll:
            best_temperature = temperature
            best_nll = loss_value
    if best_nll == float("inf"):
        if last_error is not None:
            raise RuntimeError("Temperature calibration failed for every candidate temperature.") from last_error
        raise ValueError("Temperature calibration requires at least one valid candidate temperature.")
    return best_temperature, best_nll


def mahalanobis_distance(
    features: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
) -> torch.Tensor:
    normalized = ((features - mean) ** 2) / var
    return normalized.sum(dim=-1).sqrt()


def ensemble_z_score(
    distances: torch.Tensor,
    energies: torch.Tensor,
    *,
    mahalanobis_mu: float,
    mahalanobis_sigma: float,
    energy_mu: float,
    energy_sigma: float,
) -> torch.Tensor:
    mahalanobis_z = (distances - float(mahalanobis_mu)) / max(float(mahalanobis_sigma), 1e-6)
    energy_z = (energies - float(energy_mu)) / max(float(energy_sigma), 1e-6)
    return (MAHALANOBIS_WEIGHT * mahalanobis_z) + (ENERGY_WEIGHT * energy_z)


def distribution_threshold(values: torch.Tensor, threshold_factor: float) -> float:
    std = float(safe_std(values.std(unbiased=False)).item())
    return float(values.mean().item() + (float(threshold_factor) * std))


def ensemble_threshold(ensemble_scores: torch.Tensor, threshold_factor: float) -> float:
    return distribution_threshold(ensemble_scores, threshold_factor)
