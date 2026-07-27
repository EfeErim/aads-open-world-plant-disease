#!/usr/bin/env python3
"""Continual-learning OOD detector using a multi-score runtime stack.

The detector keeps the Mahalanobis + energy ensemble path, but also
calibrates raw energy and per-class kNN distance scores from the same feature
materialization. The repo's SURE+/DS-F1-inspired double scoring remains
ensemble-based, while conformal prediction now supports threshold, APS, and
RAPS modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.ood._scoring_utils import (
    distribution_threshold,
    energy_from_logits,
    ensemble_z_score,
    mahalanobis_distance,
    normalize_temperature,
    safe_std,
    select_temperature_from_logits,
)
from src.ood.conformal_prediction import (
    build_prediction_set,
    calibrate_conformal_qhat,
    calibrate_prediction_set_qhat,
    compute_nonconformity_scores,
    describe_conformal_method,
    normalize_conformal_method,
)
from src.ood.radial_normalization import auto_tune_beta, radial_l2_normalize
from src.ood.sure_scoring import (
    calibrate_sure_thresholds,
    compute_confidence_score,
    compute_semantic_score,
)

SUPPORTED_OOD_SCORE_METHODS = ("ensemble", "energy", "knn")
SUPPORTED_KNN_BACKENDS = ("auto", "cdist", "chunked", "faiss")
DEFAULT_KNN_K = 10
DEFAULT_KNN_BANK_CAP = 256
DEFAULT_KNN_CHUNK_SIZE = 2048
_FLOAT_EPS = 1e-6

try:
    import faiss as _faiss
except ImportError:  # pragma: no cover - optional acceleration
    _faiss = None


def normalize_knn_backend(value: Any) -> str:
    resolved = str(value or "auto").strip().lower()
    if resolved not in SUPPORTED_KNN_BACKENDS:
        raise ValueError("knn_backend must be one of: " + ", ".join(SUPPORTED_KNN_BACKENDS) + ".")
    return resolved


_BASE_SCORE_FIELDS = (
    "mahalanobis_z",
    "energy_z",
    "energy_score",
    "energy_threshold",
    "knn_distance",
    "knn_threshold",
    "primary_score",
    "decision_threshold",
    "is_ood",
)
_SURE_SCORE_FIELDS = (
    "sure_semantic_score",
    "sure_confidence_score",
    "sure_semantic_ood",
    "sure_confidence_reject",
)


def normalize_primary_score_method(value: Any) -> str:
    resolved = str(value or "ensemble").strip().lower()
    if resolved not in SUPPORTED_OOD_SCORE_METHODS:
        raise ValueError(
            "primary_score_method must be one of: "
            + ", ".join(SUPPORTED_OOD_SCORE_METHODS)
            + "."
        )
    return resolved


@dataclass
class ClassCalibration:
    mean: torch.Tensor
    var: torch.Tensor
    mahalanobis_mu: float
    mahalanobis_sigma: float
    energy_mu: float
    energy_sigma: float
    threshold: float
    energy_threshold: float = 0.0
    knn_distance_mu: float = 0.0
    knn_distance_sigma: float = 1.0
    knn_threshold: float = 0.0
    knn_bank: Optional[torch.Tensor] = None
    knn_k: int = DEFAULT_KNN_K
    sure_semantic_threshold: float = 0.0
    sure_confidence_threshold: float = 0.0


class ContinualOODDetector:
    """OOD detector with configurable primary score method.

    All score methods are calibrated together:
    - ``ensemble``: weighted Mahalanobis z + energy z
    - ``energy``: raw energy score thresholded per predicted class
    - ``knn``: mean distance to the class-local k nearest neighbors
    """

    def __init__(
        self,
        threshold_factor: float = 2.0,
        *,
        primary_score_method: str = "ensemble",
        knn_k: int = DEFAULT_KNN_K,
        knn_bank_cap: int = DEFAULT_KNN_BANK_CAP,
        knn_backend: str = "auto",
        knn_chunk_size: int = DEFAULT_KNN_CHUNK_SIZE,
        radial_l2_enabled: bool = False,
        radial_beta_range: Tuple[float, float] = (0.5, 2.0),
        radial_beta_steps: int = 16,
        sure_enabled: bool = False,
        sure_semantic_percentile: float = 95.0,
        sure_confidence_percentile: float = 90.0,
        conformal_enabled: bool = False,
        conformal_alpha: float = 0.05,
        conformal_method: str = "threshold",
        conformal_raps_lambda: float = 0.0,
        conformal_raps_k_reg: int = 1,
        energy_temperature: float = 1.0,
        energy_temperature_mode: str = "fixed",
        energy_temperature_range: Tuple[float, float] = (0.5, 3.0),
        energy_temperature_steps: int = 16,
        react_enabled: bool = False,
        react_percentile: float = 0.99,
        react_apply_during_calibration: bool = True,
        react_apply_during_inference: bool = True,
        score_threshold_overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        self.threshold_factor = float(threshold_factor)
        self.primary_score_method = normalize_primary_score_method(primary_score_method)
        self.knn_k = int(knn_k)
        self.knn_bank_cap = int(knn_bank_cap)
        self.knn_backend = normalize_knn_backend(knn_backend)
        self.knn_chunk_size = int(knn_chunk_size)
        self.class_stats: Dict[int, ClassCalibration] = {}
        self.calibration_version = 0

        self.radial_l2_enabled = bool(radial_l2_enabled)
        self.radial_beta_range = (float(radial_beta_range[0]), float(radial_beta_range[1]))
        self.radial_beta_steps = int(radial_beta_steps)
        self.radial_beta: Optional[float] = None

        self.sure_enabled = bool(sure_enabled)
        self.sure_semantic_percentile = float(sure_semantic_percentile)
        self.sure_confidence_percentile = float(sure_confidence_percentile)

        self.conformal_enabled = bool(conformal_enabled)
        self.conformal_alpha = float(conformal_alpha)
        self.conformal_method = normalize_conformal_method(conformal_method)
        self.conformal_raps_lambda = float(conformal_raps_lambda)
        self.conformal_raps_k_reg = int(conformal_raps_k_reg)
        self.conformal_qhat: Optional[float] = None

        self.energy_temperature = normalize_temperature(float(energy_temperature))
        self.energy_temperature_mode = str(energy_temperature_mode or "fixed").strip().lower()
        self.energy_temperature_range = (
            normalize_temperature(float(energy_temperature_range[0])),
            normalize_temperature(float(energy_temperature_range[1])),
        )
        self.energy_temperature_steps = int(energy_temperature_steps)
        self.react_enabled = bool(react_enabled)
        self.react_percentile = float(react_percentile)
        self.react_apply_during_calibration = bool(react_apply_during_calibration)
        self.react_apply_during_inference = bool(react_apply_during_inference)
        self.react_threshold: Optional[float] = None
        self.score_threshold_overrides: Dict[str, float] = {}
        for method_name, threshold in dict(score_threshold_overrides or {}).items():
            self.set_score_threshold_override(method_name, threshold)

    def set_score_threshold_override(self, method_name: Any, threshold: Any) -> None:
        resolved_method = normalize_primary_score_method(method_name)
        self.score_threshold_overrides[resolved_method] = float(threshold)

    @staticmethod
    def _safe_std(value: torch.Tensor) -> torch.Tensor:
        return safe_std(value)

    def _energy(self, logits: torch.Tensor) -> torch.Tensor:
        return energy_from_logits(logits, temperature=self.energy_temperature)

    @staticmethod
    def _calibrated_class_ids(labels: torch.Tensor) -> List[int]:
        return [int(class_id) for class_id in torch.unique(labels).tolist()]

    @staticmethod
    def _build_calibration_summary(
        *,
        class_count: int,
        calibration_version: int,
        primary_score_method: str,
        radial_beta: Optional[float],
        conformal_qhat: Optional[float],
        knn_k: int,
        knn_bank_cap: int,
        knn_backend: str,
        knn_chunk_size: int,
        conformal_method: str,
        energy_temperature: float,
        react_enabled: bool,
        react_percentile: float,
        react_threshold: Optional[float],
    ) -> Dict[str, float | str]:
        result: Dict[str, float | str] = {
            "num_classes": float(class_count),
            "calibration_version": float(calibration_version),
            "primary_score_method": str(primary_score_method),
            "knn_k": float(knn_k),
            "knn_bank_cap": float(knn_bank_cap),
            "knn_backend": str(knn_backend),
            "knn_chunk_size": float(knn_chunk_size),
            "conformal_method": str(conformal_method),
            "conformal_method_description": describe_conformal_method(conformal_method),
            "energy_temperature": float(energy_temperature),
            "react_enabled": bool(react_enabled),
            "react_percentile": float(react_percentile),
        }
        if react_threshold is not None:
            result["react_threshold"] = float(react_threshold)
        if radial_beta is not None:
            result["radial_beta"] = float(radial_beta)
        if conformal_qhat is not None:
            result["conformal_qhat"] = float(conformal_qhat)
        return result

    @staticmethod
    def _stats_on_input_device(
        stats: ClassCalibration,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_dtype = reference.dtype if torch.is_floating_point(reference) else torch.float32
        mean = stats.mean.to(device=reference.device, dtype=target_dtype)
        var = stats.var.to(device=reference.device, dtype=target_dtype)
        return mean, var

    @staticmethod
    def _knn_bank_on_input_device(stats: ClassCalibration, reference: torch.Tensor) -> Optional[torch.Tensor]:
        if stats.knn_bank is None or stats.knn_bank.numel() <= 0:
            return None
        target_dtype = reference.dtype if torch.is_floating_point(reference) else torch.float32
        return stats.knn_bank.to(device=reference.device, dtype=target_dtype)

    @staticmethod
    def _default_score_result(primary_score_method: str) -> Dict[str, Any]:
        return {
            "mahalanobis_z": 0.0,
            "energy_z": 0.0,
            "energy_score": 0.0,
            "energy_threshold": float("inf"),
            "knn_distance": 0.0,
            "knn_threshold": float("inf"),
            "primary_score": 0.0,
            "decision_threshold": float("inf"),
            "is_ood": False,
            "score_method": primary_score_method,
            "candidate_scores": {name: 0.0 for name in SUPPORTED_OOD_SCORE_METHODS},
            "candidate_thresholds": {name: float("inf") for name in SUPPORTED_OOD_SCORE_METHODS},
            "sure_semantic_score": None,
            "sure_confidence_score": None,
            "sure_semantic_ood": None,
            "sure_confidence_reject": None,
        }

    def _default_score_batch(self, *, count: int, device: torch.device) -> Dict[str, Any]:
        base = self._default_score_result(self.primary_score_method)
        float_fields = {
            "mahalanobis_z",
            "energy_z",
            "energy_score",
            "energy_threshold",
            "knn_distance",
            "knn_threshold",
            "primary_score",
            "decision_threshold",
        }
        result: Dict[str, Any] = {}
        for field, value in base.items():
            if field in {"score_method", "candidate_scores", "candidate_thresholds"}:
                continue
            if field == "is_ood":
                result[field] = torch.full((count,), bool(value), dtype=torch.bool, device=device)
                continue
            if field in _SURE_SCORE_FIELDS:
                if field.endswith("_ood") or field.endswith("_reject"):
                    result[field] = torch.zeros(count, dtype=torch.bool, device=device)
                else:
                    result[field] = torch.zeros(count, dtype=torch.float32, device=device)
                continue
            if field in float_fields:
                result[field] = torch.full((count,), float(value), dtype=torch.float32, device=device)
        result["score_method"] = self.primary_score_method
        result["candidate_scores"] = {
            name: torch.zeros(count, dtype=torch.float32, device=device)
            for name in SUPPORTED_OOD_SCORE_METHODS
        }
        result["candidate_thresholds"] = {
            name: torch.full((count,), float("inf"), dtype=torch.float32, device=device)
            for name in SUPPORTED_OOD_SCORE_METHODS
        }
        return result

    @staticmethod
    def _set_sure_thresholds(
        class_stats: Dict[int, ClassCalibration],
        *,
        semantic_threshold: float,
        confidence_threshold: float,
    ) -> None:
        for stats in class_stats.values():
            stats.sure_semantic_threshold = semantic_threshold
            stats.sure_confidence_threshold = confidence_threshold

    def _maybe_normalize(self, features: torch.Tensor) -> torch.Tensor:
        if self.radial_l2_enabled and self.radial_beta is not None:
            return radial_l2_normalize(features, self.radial_beta)
        return features

    def fit_react_threshold(self, features: torch.Tensor) -> Optional[float]:
        if not self.react_enabled or features.numel() <= 0:
            self.react_threshold = None
            return None
        clipped_percentile = min(1.0, max(_FLOAT_EPS, float(self.react_percentile)))
        threshold = float(torch.quantile(features.detach().to(dtype=torch.float32), clipped_percentile).item())
        self.react_threshold = threshold
        return threshold

    def _maybe_reactify(
        self,
        features: torch.Tensor,
        *,
        apply: bool,
    ) -> torch.Tensor:
        if not self.react_enabled or not apply or self.react_threshold is None:
            return features
        return torch.clamp(features, max=float(self.react_threshold))

    def apply_inference_adjustments(self, features: torch.Tensor) -> torch.Tensor:
        return self._maybe_reactify(features, apply=self.react_apply_during_inference)

    def _class_distances(self, features: torch.Tensor, stats: ClassCalibration) -> torch.Tensor:
        mean, var = self._stats_on_input_device(stats, features)
        return mahalanobis_distance(features, mean, var)

    def _class_ensemble_scores(
        self,
        features: torch.Tensor,
        energies: torch.Tensor,
        stats: ClassCalibration,
    ) -> torch.Tensor:
        distances = self._class_distances(features, stats)
        return ensemble_z_score(
            distances,
            energies,
            mahalanobis_mu=stats.mahalanobis_mu,
            mahalanobis_sigma=stats.mahalanobis_sigma,
            energy_mu=stats.energy_mu,
            energy_sigma=stats.energy_sigma,
        )

    @staticmethod
    def _subsample_evenly(features: torch.Tensor, limit: int) -> torch.Tensor:
        if int(features.shape[0]) <= int(limit):
            return features.detach().clone()
        indices = torch.div(
            torch.arange(int(limit), device=features.device) * int(features.shape[0]),
            int(limit),
            rounding_mode="floor",
        )
        return features.index_select(0, indices.to(dtype=torch.long)).detach().clone()

    def _build_knn_bank(self, class_features: torch.Tensor) -> torch.Tensor:
        return self._subsample_evenly(class_features, self.knn_bank_cap)

    @staticmethod
    def _resolve_neighbor_count(bank_size: int, k: int, *, exclude_self: bool) -> int:
        if exclude_self:
            return min(max(1, int(k)), max(1, int(bank_size) - 1))
        return min(max(1, int(k)), int(bank_size))

    @staticmethod
    def _pairwise_l2(query: torch.Tensor, ref_bank: torch.Tensor) -> torch.Tensor:
        query_norm = (query * query).sum(dim=1, keepdim=True)
        ref_norm = (ref_bank * ref_bank).sum(dim=1).unsqueeze(0)
        distances_sq = (query_norm + ref_norm - (2.0 * query @ ref_bank.transpose(0, 1))).clamp_min(0.0)
        return distances_sq.sqrt()

    @classmethod
    def _chunked_knn_distances(
        cls,
        query: torch.Tensor,
        ref_bank: torch.Tensor,
        *,
        neighbor_count: int,
        chunk_size: int,
        exclude_self: bool,
    ) -> torch.Tensor:
        topk_values: Optional[torch.Tensor] = None
        resolved_chunk_size = max(1, int(chunk_size))
        for start in range(0, int(ref_bank.shape[0]), resolved_chunk_size):
            distances = cls._pairwise_l2(query, ref_bank[start:start + resolved_chunk_size])
            if exclude_self:
                distances = distances.clone()
                zero_mask = distances <= _FLOAT_EPS
                if bool(zero_mask.any().item()):
                    distances[zero_mask] = float("inf")
            if topk_values is None:
                first_k = min(neighbor_count, int(distances.shape[1]))
                topk_values = torch.topk(distances, k=first_k, largest=False, dim=1).values
            else:
                merged = torch.cat([topk_values, distances], dim=1)
                topk_values = torch.topk(merged, k=neighbor_count, largest=False, dim=1).values
        if topk_values is None:
            return torch.zeros(query.shape[0], dtype=query.dtype, device=query.device)
        return topk_values.mean(dim=1)

    @staticmethod
    def _faiss_knn_distances(
        query: torch.Tensor,
        ref_bank: torch.Tensor,
        *,
        neighbor_count: int,
        exclude_self: bool,
    ) -> torch.Tensor:
        if _faiss is None:  # pragma: no cover - optional dependency
            raise RuntimeError("faiss backend requested but faiss is not installed.")
        index = _faiss.IndexFlatL2(int(ref_bank.shape[1]))
        bank_np = ref_bank.detach().cpu().numpy().astype("float32", copy=False)
        query_np = query.detach().cpu().numpy().astype("float32", copy=False)
        index.add(bank_np)
        search_k = min(int(ref_bank.shape[0]), neighbor_count + (1 if exclude_self else 0))
        distances_sq, _ = index.search(query_np, search_k)
        distances = torch.from_numpy(distances_sq).to(device=query.device, dtype=query.dtype).clamp_min(0.0).sqrt()
        if exclude_self and distances.shape[1] > neighbor_count:
            distances = distances[:, 1:]
        return distances[:, :neighbor_count].mean(dim=1)

    def _resolve_knn_backend(self, bank: torch.Tensor) -> str:
        if self.knn_backend != "auto":
            return self.knn_backend
        if _faiss is not None and bank.device.type == "cpu" and int(bank.shape[0]) >= 4096:
            return "faiss"
        return "chunked" if int(bank.shape[0]) >= max(self.knn_chunk_size, 512) else "cdist"

    def _knn_distances_from_bank(
        self,
        features: torch.Tensor,
        bank: Optional[torch.Tensor],
        *,
        k: int,
        exclude_self: bool = False,
    ) -> torch.Tensor:
        batch_size = int(features.shape[0])
        if bank is None or bank.numel() <= 0 or batch_size <= 0:
            return torch.zeros(batch_size, dtype=features.dtype, device=features.device)
        if int(bank.shape[0]) <= 1 and exclude_self:
            return torch.zeros(batch_size, dtype=features.dtype, device=features.device)

        query = features.to(dtype=torch.float32)
        ref_bank = bank.to(device=features.device, dtype=torch.float32)
        neighbor_count = self._resolve_neighbor_count(int(ref_bank.shape[0]), int(k), exclude_self=exclude_self)
        backend = self._resolve_knn_backend(ref_bank)

        if backend == "faiss":
            nearest = self._faiss_knn_distances(
                query,
                ref_bank,
                neighbor_count=neighbor_count,
                exclude_self=exclude_self,
            )
            return nearest.to(device=features.device, dtype=features.dtype)

        if backend == "chunked":
            nearest = self._chunked_knn_distances(
                query,
                ref_bank,
                neighbor_count=neighbor_count,
                chunk_size=self.knn_chunk_size,
                exclude_self=exclude_self,
            )
            return nearest.to(device=features.device, dtype=features.dtype)

        distances = torch.cdist(query, ref_bank, p=2)
        if exclude_self:
            distances = distances.clone()
            zero_mask = distances <= _FLOAT_EPS
            if bool(zero_mask.any().item()):
                distances[zero_mask] = float("inf")
        nearest = torch.topk(distances, k=neighbor_count, largest=False, dim=1).values
        return nearest.mean(dim=1).to(device=features.device, dtype=features.dtype)

    def _class_knn_distances(
        self,
        features: torch.Tensor,
        stats: ClassCalibration,
        *,
        exclude_self: bool = False,
    ) -> torch.Tensor:
        bank = self._knn_bank_on_input_device(stats, features)
        return self._knn_distances_from_bank(
            features,
            bank,
            k=int(stats.knn_k),
            exclude_self=exclude_self,
        )

    def _score_class_batch(self, class_id: int, features: torch.Tensor, logits: torch.Tensor) -> Dict[str, Any]:
        batch_size = int(features.shape[0])
        if batch_size <= 0:
            return self._default_score_batch(count=0, device=features.device)
        if class_id not in self.class_stats:
            return self._default_score_batch(count=batch_size, device=features.device)

        score_features = self._maybe_normalize(
            self._maybe_reactify(features, apply=self.react_apply_during_inference)
        )
        stats = self.class_stats[class_id]
        distance = self._class_distances(score_features, stats).to(dtype=torch.float32)
        energy = self._energy(logits).to(dtype=torch.float32)
        ensemble = self._class_ensemble_scores(score_features, energy, stats).to(dtype=torch.float32)
        knn_distance = self._class_knn_distances(score_features, stats).to(dtype=torch.float32)

        mahalanobis_z = (distance - float(stats.mahalanobis_mu)) / max(float(stats.mahalanobis_sigma), _FLOAT_EPS)
        energy_z = (energy - float(stats.energy_mu)) / max(float(stats.energy_sigma), _FLOAT_EPS)
        candidate_scores = {
            "ensemble": ensemble,
            "energy": energy,
            "knn": knn_distance,
        }
        candidate_thresholds = {
            "ensemble": torch.full((batch_size,), float(stats.threshold), dtype=torch.float32, device=features.device),
            "energy": torch.full(
                (batch_size,),
                float(stats.energy_threshold),
                dtype=torch.float32,
                device=features.device,
            ),
            "knn": torch.full((batch_size,), float(stats.knn_threshold), dtype=torch.float32, device=features.device),
        }
        for method_name, threshold in self.score_threshold_overrides.items():
            if method_name in candidate_thresholds:
                candidate_thresholds[method_name] = torch.full(
                    (batch_size,),
                    float(threshold),
                    dtype=torch.float32,
                    device=features.device,
                )
        primary_score = candidate_scores[self.primary_score_method]
        decision_threshold = candidate_thresholds[self.primary_score_method]
        is_ood = primary_score > decision_threshold

        result: Dict[str, Any] = {
            "mahalanobis_z": mahalanobis_z,
            "energy_z": energy_z,
            "energy_score": candidate_scores["energy"],
            "energy_threshold": candidate_thresholds["energy"],
            "knn_distance": candidate_scores["knn"],
            "knn_threshold": candidate_thresholds["knn"],
            "primary_score": primary_score,
            "decision_threshold": decision_threshold,
            "is_ood": is_ood,
            "score_method": self.primary_score_method,
            "candidate_scores": candidate_scores,
            "candidate_thresholds": candidate_thresholds,
        }

        if self.sure_enabled:
            confidence_score = compute_confidence_score(logits).to(dtype=torch.float32)
            sure_semantic_ood = candidate_scores["ensemble"] > float(stats.sure_semantic_threshold)
            sure_confidence_reject = confidence_score > float(stats.sure_confidence_threshold)
            result["sure_semantic_score"] = candidate_scores["ensemble"]
            result["sure_confidence_score"] = confidence_score
            result["sure_semantic_ood"] = sure_semantic_ood
            result["sure_confidence_reject"] = sure_confidence_reject
        else:
            result["sure_semantic_score"] = torch.zeros(batch_size, dtype=torch.float32, device=features.device)
            result["sure_confidence_score"] = torch.zeros(batch_size, dtype=torch.float32, device=features.device)
            result["sure_semantic_ood"] = torch.zeros(batch_size, dtype=torch.bool, device=features.device)
            result["sure_confidence_reject"] = torch.zeros(batch_size, dtype=torch.bool, device=features.device)
        return result

    def calibration_issue(self) -> Optional[str]:
        if not self.class_stats:
            return "OOD detector has no calibrated class statistics."
        if self.calibration_version <= 0:
            return "OOD detector calibration version is unset."
        if self.primary_score_method == "knn":
            missing_banks = [
                str(class_id)
                for class_id, stats in self.class_stats.items()
                if stats.knn_bank is None or stats.knn_bank.numel() <= 0
            ]
            if missing_banks:
                return "OOD detector is missing kNN calibration banks for configured primary_score_method='knn'."
        return None

    def assert_calibrated(self, *, operation: str) -> None:
        issue = self.calibration_issue()
        if issue is None:
            return
        raise RuntimeError(f"{issue} Run calibrate_ood(...) before {operation}.")

    def _maybe_calibrate_energy_temperature(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if self.energy_temperature_mode != "auto":
            self.energy_temperature = normalize_temperature(self.energy_temperature)
            return
        min_temp, max_temp = self.energy_temperature_range
        if max_temp < min_temp:
            min_temp, max_temp = max_temp, min_temp
        steps = max(2, int(self.energy_temperature_steps))
        candidates = torch.linspace(min_temp, max_temp, steps=steps, dtype=torch.float32).tolist()
        best_temperature, _best_nll = select_temperature_from_logits(logits, labels, candidate_temperatures=candidates)
        self.energy_temperature = normalize_temperature(best_temperature)

    def calibrate(self, features: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        if features.ndim != 2:
            raise ValueError("features must be [N, D]")
        if logits.ndim != 2:
            raise ValueError("logits must be [N, C]")
        if labels.ndim != 1:
            labels = labels.reshape(-1)

        self._maybe_calibrate_energy_temperature(logits, labels)
        if self.react_enabled:
            self.fit_react_threshold(features)
            features = self._maybe_reactify(features, apply=self.react_apply_during_calibration)

        if self.radial_l2_enabled:
            self.radial_beta = auto_tune_beta(
                features,
                labels,
                beta_range=self.radial_beta_range,
                beta_steps=self.radial_beta_steps,
            )
            features = radial_l2_normalize(features, self.radial_beta)

        self.class_stats = {}
        energies = self._energy(logits)
        class_ids = self._calibrated_class_ids(labels)

        for class_id in class_ids:
            mask = labels == class_id
            class_features = features[mask]
            class_energy = energies[mask]
            if class_features.numel() == 0:
                continue

            mean = class_features.mean(dim=0)
            var = class_features.var(dim=0, unbiased=False).clamp_min(_FLOAT_EPS)

            distances = mahalanobis_distance(class_features, mean, var)
            mahalanobis_mu = float(distances.mean().item())
            mahalanobis_sigma = float(self._safe_std(distances.std(unbiased=False)).item())

            energy_mu = float(class_energy.mean().item())
            energy_sigma = float(self._safe_std(class_energy.std(unbiased=False)).item())

            ensemble_scores = ensemble_z_score(
                distances,
                class_energy,
                mahalanobis_mu=mahalanobis_mu,
                mahalanobis_sigma=mahalanobis_sigma,
                energy_mu=energy_mu,
                energy_sigma=energy_sigma,
            )
            ensemble_threshold_value = distribution_threshold(ensemble_scores, self.threshold_factor)
            energy_threshold_value = distribution_threshold(class_energy, self.threshold_factor)

            knn_bank = self._build_knn_bank(class_features)
            knn_distances = self._knn_distances_from_bank(
                class_features,
                knn_bank,
                k=self.knn_k,
                exclude_self=True,
            )
            knn_distance_mu = float(knn_distances.mean().item())
            knn_distance_sigma = float(self._safe_std(knn_distances.std(unbiased=False)).item())
            knn_threshold_value = distribution_threshold(knn_distances, self.threshold_factor)

            self.class_stats[int(class_id)] = ClassCalibration(
                mean=mean,
                var=var,
                mahalanobis_mu=mahalanobis_mu,
                mahalanobis_sigma=mahalanobis_sigma,
                energy_mu=energy_mu,
                energy_sigma=energy_sigma,
                threshold=ensemble_threshold_value,
                energy_threshold=energy_threshold_value,
                knn_distance_mu=knn_distance_mu,
                knn_distance_sigma=knn_distance_sigma,
                knn_threshold=knn_threshold_value,
                knn_bank=knn_bank,
                knn_k=self.knn_k,
            )

        if self.sure_enabled and self.class_stats:
            all_ensemble_scores = []
            all_confidence = []
            for class_id in class_ids:
                mask = labels == class_id
                if class_id not in self.class_stats:
                    continue
                stats = self.class_stats[class_id]
                class_features = features[mask]
                class_logits = logits[mask]
                class_energy = energies[mask]

                ensemble_scores = self._class_ensemble_scores(class_features, class_energy, stats)
                all_ensemble_scores.append(ensemble_scores)
                all_confidence.append(compute_confidence_score(class_logits))

            if all_ensemble_scores:
                cat_semantic = compute_semantic_score(torch.cat(all_ensemble_scores))
                cat_confidence = torch.cat(all_confidence)
                sem_thresh, conf_thresh = calibrate_sure_thresholds(
                    cat_semantic,
                    cat_confidence,
                    semantic_percentile=self.sure_semantic_percentile,
                    confidence_percentile=self.sure_confidence_percentile,
                )
                self._set_sure_thresholds(
                    self.class_stats,
                    semantic_threshold=sem_thresh,
                    confidence_threshold=conf_thresh,
                )

        if self.conformal_enabled and self.class_stats:
            if self.conformal_method == "threshold":
                all_ensemble_scores = []
                all_class_thresholds = []
                for class_id in class_ids:
                    mask = labels == class_id
                    if class_id not in self.class_stats:
                        continue
                    stats = self.class_stats[class_id]
                    class_features = features[mask]
                    class_energy = energies[mask]

                    ensemble_scores = self._class_ensemble_scores(class_features, class_energy, stats)
                    all_ensemble_scores.append(ensemble_scores)
                    all_class_thresholds.append(torch.full_like(ensemble_scores, stats.threshold))

                if all_ensemble_scores:
                    cat_scores = torch.cat(all_ensemble_scores)
                    cat_thresholds = torch.cat(all_class_thresholds)
                    nc_scores = compute_nonconformity_scores(cat_scores, cat_thresholds)
                    self.conformal_qhat = calibrate_conformal_qhat(nc_scores, alpha=self.conformal_alpha)
            else:
                self.conformal_qhat = calibrate_prediction_set_qhat(
                    logits,
                    labels,
                    alpha=self.conformal_alpha,
                    method=self.conformal_method,
                    raps_lambda=self.conformal_raps_lambda,
                    raps_k_reg=self.conformal_raps_k_reg,
                )

        self.calibration_version += 1
        return self._build_calibration_summary(
            class_count=len(self.class_stats),
            calibration_version=self.calibration_version,
            primary_score_method=self.primary_score_method,
            radial_beta=self.radial_beta,
            conformal_qhat=self.conformal_qhat,
            knn_k=self.knn_k,
            knn_bank_cap=self.knn_bank_cap,
            knn_backend=self.knn_backend,
            knn_chunk_size=self.knn_chunk_size,
            conformal_method=self.conformal_method,
            energy_temperature=self.energy_temperature,
            react_enabled=self.react_enabled,
            react_percentile=self.react_percentile,
            react_threshold=self.react_threshold,
        )

    def _score_class(self, class_id: int, feature: torch.Tensor, logit: torch.Tensor) -> Dict[str, Any]:
        score_feature = feature.unsqueeze(0) if feature.ndim == 1 else feature
        score_logit = logit.unsqueeze(0) if logit.ndim == 1 else logit
        batch_result = self._score_class_batch(class_id, score_feature, score_logit)

        result: Dict[str, Any] = {
            "score_method": batch_result["score_method"],
            "candidate_scores": {
                name: float(values[0].item())
                for name, values in batch_result["candidate_scores"].items()
            },
            "candidate_thresholds": {
                name: float(values[0].item())
                for name, values in batch_result["candidate_thresholds"].items()
            },
        }
        for field in _BASE_SCORE_FIELDS:
            values = batch_result[field]
            result[field] = bool(values[0].item()) if field == "is_ood" else float(values[0].item())
        if self.sure_enabled:
            result["sure_semantic_score"] = float(batch_result["sure_semantic_score"][0].item())
            result["sure_confidence_score"] = float(batch_result["sure_confidence_score"][0].item())
            result["sure_semantic_ood"] = bool(batch_result["sure_semantic_ood"][0].item())
            result["sure_confidence_reject"] = bool(batch_result["sure_confidence_reject"][0].item())
        else:
            result["sure_semantic_score"] = None
            result["sure_confidence_score"] = None
            result["sure_semantic_ood"] = None
            result["sure_confidence_reject"] = None
        return result

    def build_conformal_set(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        idx_to_class: Dict[int, str],
    ) -> List[str]:
        if not self.conformal_enabled or self.conformal_qhat is None:
            return []

        if features.ndim == 1:
            feat = self._maybe_normalize(
                self._maybe_reactify(features.unsqueeze(0), apply=self.react_apply_during_inference)
            ).squeeze(0)
        else:
            feat = self._maybe_normalize(
                self._maybe_reactify(features, apply=self.react_apply_during_inference)
            )

        return build_prediction_set(
            features=feat,
            logits=logits,
            qhat=self.conformal_qhat,
            class_stats=self.class_stats,
            idx_to_class=idx_to_class,
            method=self.conformal_method,
            raps_lambda=self.conformal_raps_lambda,
            raps_k_reg=self.conformal_raps_k_reg,
            energy_temperature=self.energy_temperature,
        )

    def score(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        predicted_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if features.ndim != 2 or logits.ndim != 2:
            raise ValueError("features/logits must be batched tensors [N, D] and [N, C]")

        if predicted_labels is None:
            predicted_labels = torch.argmax(logits, dim=1)
        predicted_labels = predicted_labels.reshape(-1)
        batch_size = int(features.size(0))
        float_outputs = {
            "mahalanobis_z": torch.empty(batch_size, dtype=torch.float32, device=features.device),
            "energy_z": torch.empty(batch_size, dtype=torch.float32, device=features.device),
            "energy_score": torch.empty(batch_size, dtype=torch.float32, device=features.device),
            "energy_threshold": torch.empty(batch_size, dtype=torch.float32, device=features.device),
            "knn_distance": torch.empty(batch_size, dtype=torch.float32, device=features.device),
            "knn_threshold": torch.empty(batch_size, dtype=torch.float32, device=features.device),
            "primary_score": torch.empty(batch_size, dtype=torch.float32, device=features.device),
            "decision_threshold": torch.empty(batch_size, dtype=torch.float32, device=features.device),
        }
        bool_outputs = {
            "is_ood": torch.empty(batch_size, dtype=torch.bool, device=features.device),
        }
        candidate_scores = {
            name: torch.empty(batch_size, dtype=torch.float32, device=features.device)
            for name in SUPPORTED_OOD_SCORE_METHODS
        }
        candidate_thresholds = {
            name: torch.empty(batch_size, dtype=torch.float32, device=features.device)
            for name in SUPPORTED_OOD_SCORE_METHODS
        }
        if self.sure_enabled:
            sure_float_outputs = {
                "sure_semantic_score": torch.empty(batch_size, dtype=torch.float32, device=features.device),
                "sure_confidence_score": torch.empty(batch_size, dtype=torch.float32, device=features.device),
            }
            sure_bool_outputs = {
                "sure_semantic_ood": torch.empty(batch_size, dtype=torch.bool, device=features.device),
                "sure_confidence_reject": torch.empty(batch_size, dtype=torch.bool, device=features.device),
            }
        else:
            sure_float_outputs = {}
            sure_bool_outputs = {}

        for class_id in torch.unique(predicted_labels):
            mask = predicted_labels == class_id
            indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
            if int(indices.numel()) <= 0:
                continue
            batch_result = self._score_class_batch(
                int(class_id.item()),
                features.index_select(0, indices),
                logits.index_select(0, indices),
            )
            for field, storage in float_outputs.items():
                storage.index_copy_(0, indices, batch_result[field])
            for field, storage in bool_outputs.items():
                storage.index_copy_(0, indices, batch_result[field])
            for method_name, storage in candidate_scores.items():
                storage.index_copy_(0, indices, batch_result["candidate_scores"][method_name])
            for method_name, storage in candidate_thresholds.items():
                storage.index_copy_(0, indices, batch_result["candidate_thresholds"][method_name])
            if self.sure_enabled:
                for field, storage in sure_float_outputs.items():
                    storage.index_copy_(0, indices, batch_result[field])
                for field, storage in sure_bool_outputs.items():
                    storage.index_copy_(0, indices, batch_result[field])

        output: Dict[str, Any] = {
            **float_outputs,
            **bool_outputs,
            "calibration_version": torch.full(
                (features.size(0),),
                self.calibration_version,
                dtype=torch.long,
                device=features.device,
            ),
            "primary_score_method": self.primary_score_method,
            "candidate_scores": candidate_scores,
            "candidate_thresholds": candidate_thresholds,
        }
        if self.sure_enabled:
            output.update(sure_float_outputs)
            output.update(sure_bool_outputs)
        if self.radial_beta is not None:
            output["radial_beta"] = self.radial_beta
        return output
