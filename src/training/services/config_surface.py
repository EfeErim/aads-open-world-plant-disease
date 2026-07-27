"""Normalization helpers for the public continual-training config contract."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.shared.json_utils import deep_merge
from src.training.services.ood_score_selection import normalize_requested_primary_score_method

DEFAULT_BACKBONE_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_FUSION_LAYERS = [2, 5, 8, 11]
DEFAULT_FUSION_OUTPUT_DIM = 768
DEFAULT_DEVICE = "cuda"
VALID_AUGMENTATION_POLICIES = {"none", "basic", "randaugment", "augmix"}


def _build_default_continual_surface(*, model_name: str, device: Any) -> Dict[str, Any]:
    return {
        "backbone": {"model_name": str(model_name)},
        "adapter": {
            "target_modules_strategy": "all_linear_transformer",
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
        },
        "fusion": {
            "layers": list(DEFAULT_FUSION_LAYERS),
            "output_dim": DEFAULT_FUSION_OUTPUT_DIM,
            "dropout": 0.1,
            "gating": "softmax",
        },
        "ood": {
            "threshold_factor": 3.0,
            "primary_score_method": "auto",
            "ber_enabled": False,
            "ber_lambda_old": 0.1,
            "ber_lambda_new": 0.1,
            "ber_warmup_steps": 50,
            "energy_temperature_mode": "auto",
            "energy_temperature": 1.0,
            "energy_temperature_range": [0.5, 3.0],
            "energy_temperature_steps": 16,
            "react_enabled": False,
            "react_percentile": 0.99,
            "react_apply_during_calibration": True,
            "react_apply_during_inference": True,
            "radial_l2_enabled": True,
            "radial_beta_range": [0.5, 2.0],
            "radial_beta_steps": 16,
            "real_split_enabled": True,
            "real_split_dev_fraction": 0.4,
            "real_split_min_per_slice": 2,
            "real_split_min_total": 30,
            "real_split_manifest_name": "ood_split_manifest.json",
            "enforce_oe_disjoint": True,
            "real_dev_selection_enabled": True,
            "real_dev_target_fpr": 0.05,
            "knn_backend": "auto",
            "knn_chunk_size": 2048,
            "sure_enabled": True,
            "sure_semantic_percentile": 90.0,
            "sure_confidence_percentile": 97.0,
            "conformal_enabled": True,
            "conformal_alpha": 0.05,
            "conformal_method": "raps",
            "conformal_raps_lambda": 0.2,
            "conformal_raps_k_reg": 1,
            "oe_enabled": True,
            "oe_loss_weight": 0.5,
            "oe_target": "uniform",
            "oe_root": "",
        },
        "classifier_rebalance": {
            "enabled": False,
            "epochs": 3,
            "learning_rate": 5e-5,
            "weight_decay": 0.0,
            "sampler": "weighted",
            "objective": "logit_adjusted_cross_entropy",
            "logit_adjustment_tau": 1.0,
        },
        "class_balance": {
            "allow_sampler_and_loss_rebalance": False,
        },
        "optimization": {
            "grad_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "mixed_precision": "auto",
            "label_smoothing": 0.0,
            "loss_name": "logitnorm",
            "logitnorm_tau": 1.0,
            "scheduler": {
                "name": "cosine",
                "warmup_ratio": 0.1,
                "min_lr": 1e-6,
                "step_on": "batch",
            },
        },
        "data": {
            "sampler": "auto",
            "loader_error_policy": "tolerant",
            "target_size": 224,
            "augmentation_policy": "randaugment",
            "randaugment_num_ops": 2,
            "randaugment_magnitude": 7,
            "augmix_severity": 3,
            "augmix_width": 3,
            "augmix_depth": -1,
            "augmix_alpha": 1.0,
            "allow_under_min_training": False,
            "cache_size": 1000,
            "validate_images_on_init": False,
        },
        "early_stopping": {
            "enabled": True,
            "metric": "val_loss",
            "mode": "min",
            "patience": 5,
            "min_delta": 0.0,
        },
        "evaluation": {
            "best_metric": "val_loss",
            "emit_ood_gate": True,
            "require_ood_for_gate": True,
            "ood_benchmark_min_classes": 3,
            "min_in_distribution_samples": 30,
            "min_ood_samples": 30,
            "min_ood_samples_per_type": 5,
            "gate_auxiliary_ood_diagnostics": False,
        },
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "num_epochs": 10,
        "batch_size": 8,
        "device": str(device),
        "strict_model_loading": False,
        "seed": 42,
        "deterministic": True,
    }


def normalize_continual_training_config(
    training_continual: Optional[Dict[str, Any]],
    *,
    model_name: str = DEFAULT_BACKBONE_MODEL_NAME,
    device: Any = DEFAULT_DEVICE,
) -> Dict[str, Any]:
    normalized = deep_merge(
        _build_default_continual_surface(model_name=model_name, device=device),
        dict(training_continual or {}),
    )

    backbone = normalized.setdefault("backbone", {})
    adapter = normalized.setdefault("adapter", {})
    fusion = normalized.setdefault("fusion", {})
    ood = normalized.setdefault("ood", {})
    classifier_rebalance = normalized.setdefault("classifier_rebalance", {})
    class_balance = normalized.setdefault("class_balance", {})
    optimization = normalized.setdefault("optimization", {})
    scheduler = optimization.setdefault("scheduler", {})
    data = normalized.setdefault("data", {})
    early_stopping = normalized.setdefault("early_stopping", {})
    evaluation = normalized.setdefault("evaluation", {})

    backbone["model_name"] = str(backbone.get("model_name", model_name))

    adapter["target_modules_strategy"] = str(adapter.get("target_modules_strategy", "all_linear_transformer"))
    adapter["lora_r"] = int(adapter.get("lora_r", 16))
    adapter["lora_alpha"] = int(adapter.get("lora_alpha", 16))
    adapter["lora_dropout"] = float(adapter.get("lora_dropout", 0.1))

    raw_layers = fusion.get("layers", DEFAULT_FUSION_LAYERS)
    fusion["layers"] = [int(value) for value in list(raw_layers or DEFAULT_FUSION_LAYERS)]
    fusion["output_dim"] = int(fusion.get("output_dim", DEFAULT_FUSION_OUTPUT_DIM))
    fusion["dropout"] = float(fusion.get("dropout", 0.1))
    fusion["gating"] = str(fusion.get("gating", "softmax"))

    normalized["learning_rate"] = float(normalized.get("learning_rate", 1e-4))
    normalized["weight_decay"] = float(normalized.get("weight_decay", 0.01))
    normalized["num_epochs"] = int(normalized.get("num_epochs", 10))
    normalized["batch_size"] = int(normalized.get("batch_size", 8))
    normalized["device"] = str(normalized.get("device", device))
    normalized["strict_model_loading"] = bool(normalized.get("strict_model_loading", False))
    normalized["seed"] = int(normalized.get("seed", 42))
    normalized["deterministic"] = bool(normalized.get("deterministic", False))

    ood["threshold_factor"] = float(ood.get("threshold_factor", 3.0))
    ood["primary_score_method"] = normalize_requested_primary_score_method(ood.get("primary_score_method", "auto"))
    ood["ber_enabled"] = bool(ood.get("ber_enabled", False))
    ood["ber_lambda_old"] = float(ood.get("ber_lambda_old", 0.1))
    ood["ber_lambda_new"] = float(ood.get("ber_lambda_new", 0.1))
    ood["ber_warmup_steps"] = int(ood.get("ber_warmup_steps", 50))
    ood["energy_temperature_mode"] = str(ood.get("energy_temperature_mode", "auto"))
    ood["energy_temperature"] = float(ood.get("energy_temperature", 1.0))
    raw_energy_range = list(ood.get("energy_temperature_range", [0.5, 3.0]))
    if len(raw_energy_range) < 2:
        raw_energy_range = [0.5, 3.0]
    ood["energy_temperature_range"] = [float(raw_energy_range[0]), float(raw_energy_range[1])]
    ood["energy_temperature_steps"] = int(ood.get("energy_temperature_steps", 16))
    ood["react_enabled"] = bool(ood.get("react_enabled", False))
    ood["react_percentile"] = float(ood.get("react_percentile", 0.99))
    if not 0.0 < ood["react_percentile"] <= 1.0:
        raise ValueError("training.continual.ood.react_percentile must be in (0, 1].")
    ood["react_apply_during_calibration"] = bool(ood.get("react_apply_during_calibration", True))
    ood["react_apply_during_inference"] = bool(ood.get("react_apply_during_inference", True))
    ood["radial_l2_enabled"] = bool(ood.get("radial_l2_enabled", True))
    raw_beta_range = list(ood.get("radial_beta_range", [0.5, 2.0]))
    if len(raw_beta_range) < 2:
        raw_beta_range = [0.5, 2.0]
    ood["radial_beta_range"] = [float(raw_beta_range[0]), float(raw_beta_range[1])]
    ood["radial_beta_steps"] = int(ood.get("radial_beta_steps", 16))
    ood["real_split_enabled"] = bool(ood.get("real_split_enabled", True))
    ood["real_split_dev_fraction"] = max(0.05, min(0.95, float(ood.get("real_split_dev_fraction", 0.4))))
    ood["real_split_min_per_slice"] = max(2, int(ood.get("real_split_min_per_slice", 2)))
    ood["real_split_min_total"] = max(0, int(ood.get("real_split_min_total", 30)))
    ood["real_split_manifest_name"] = str(
        ood.get("real_split_manifest_name", "ood_split_manifest.json") or "ood_split_manifest.json"
    )
    ood["enforce_oe_disjoint"] = bool(ood.get("enforce_oe_disjoint", True))
    ood["real_dev_selection_enabled"] = bool(ood.get("real_dev_selection_enabled", True))
    ood["real_dev_target_fpr"] = max(0.0, min(1.0, float(ood.get("real_dev_target_fpr", 0.05))))
    ood["knn_backend"] = str(ood.get("knn_backend", "auto"))
    ood["knn_chunk_size"] = int(ood.get("knn_chunk_size", 2048))
    ood["sure_enabled"] = bool(ood.get("sure_enabled", True))
    ood["sure_semantic_percentile"] = float(ood.get("sure_semantic_percentile", 90.0))
    ood["sure_confidence_percentile"] = float(ood.get("sure_confidence_percentile", 97.0))
    ood["conformal_enabled"] = bool(ood.get("conformal_enabled", True))
    ood["conformal_alpha"] = float(ood.get("conformal_alpha", 0.05))
    ood["conformal_method"] = str(ood.get("conformal_method", "raps"))
    ood["conformal_raps_lambda"] = float(ood.get("conformal_raps_lambda", 0.2))
    ood["conformal_raps_k_reg"] = int(ood.get("conformal_raps_k_reg", 1))
    ood["oe_enabled"] = bool(ood.get("oe_enabled", True))
    ood["oe_loss_weight"] = float(ood.get("oe_loss_weight", 0.5))
    if ood["oe_loss_weight"] < 0.0:
        raise ValueError("training.continual.ood.oe_loss_weight must be non-negative.")
    ood["oe_target"] = str(ood.get("oe_target", "uniform")).strip().lower()
    if ood["oe_target"] not in {"uniform"}:
        raise ValueError("training.continual.ood.oe_target must be 'uniform'.")
    ood["oe_root"] = str(ood.get("oe_root", "") or "")

    classifier_rebalance["enabled"] = bool(classifier_rebalance.get("enabled", False))
    classifier_rebalance["epochs"] = int(classifier_rebalance.get("epochs", 3))
    if classifier_rebalance["epochs"] < 1:
        raise ValueError("training.continual.classifier_rebalance.epochs must be at least 1.")
    classifier_rebalance["learning_rate"] = float(classifier_rebalance.get("learning_rate", 5e-5))
    if classifier_rebalance["learning_rate"] <= 0.0:
        raise ValueError("training.continual.classifier_rebalance.learning_rate must be positive.")
    classifier_rebalance["weight_decay"] = float(classifier_rebalance.get("weight_decay", 0.0))
    classifier_rebalance["sampler"] = str(classifier_rebalance.get("sampler", "weighted")).strip().lower()
    if classifier_rebalance["sampler"] not in {"weighted", "shuffle", "auto"}:
        raise ValueError(
            "training.continual.classifier_rebalance.sampler must be one of: auto, shuffle, weighted."
        )
    classifier_rebalance["objective"] = str(
        classifier_rebalance.get("objective", "logit_adjusted_cross_entropy")
    ).strip().lower()
    if classifier_rebalance["objective"] not in {"cross_entropy", "logit_adjusted_cross_entropy"}:
        raise ValueError(
            "training.continual.classifier_rebalance.objective must be one of: "
            "cross_entropy, logit_adjusted_cross_entropy."
        )
    classifier_rebalance["logit_adjustment_tau"] = float(
        classifier_rebalance.get("logit_adjustment_tau", 1.0)
    )
    if classifier_rebalance["logit_adjustment_tau"] < 0.0:
        raise ValueError("training.continual.classifier_rebalance.logit_adjustment_tau must be non-negative.")

    class_balance["allow_sampler_and_loss_rebalance"] = bool(
        class_balance.get("allow_sampler_and_loss_rebalance", False)
    )

    optimization["grad_accumulation_steps"] = int(optimization.get("grad_accumulation_steps", 4))
    optimization["max_grad_norm"] = float(optimization.get("max_grad_norm", 1.0))
    optimization["mixed_precision"] = str(optimization.get("mixed_precision", "auto"))
    optimization["label_smoothing"] = float(optimization.get("label_smoothing", 0.0))
    optimization["loss_name"] = str(optimization.get("loss_name", "logitnorm"))
    optimization["logitnorm_tau"] = float(optimization.get("logitnorm_tau", 1.0))
    scheduler["name"] = str(scheduler.get("name", "cosine"))
    scheduler["warmup_ratio"] = float(scheduler.get("warmup_ratio", 0.1))
    scheduler["min_lr"] = float(scheduler.get("min_lr", 1e-6))
    scheduler["step_on"] = str(scheduler.get("step_on", "batch"))

    data["sampler"] = str(data.get("sampler", "auto"))
    data["loader_error_policy"] = str(data.get("loader_error_policy", "tolerant"))
    data["target_size"] = int(data.get("target_size", 224))
    data["augmentation_policy"] = str(data.get("augmentation_policy", "randaugment")).strip().lower()
    if data["augmentation_policy"] not in VALID_AUGMENTATION_POLICIES:
        raise ValueError(
            "training.continual.data.augmentation_policy must be one of: "
            + ", ".join(sorted(VALID_AUGMENTATION_POLICIES))
        )
    data["randaugment_num_ops"] = int(data.get("randaugment_num_ops", 2))
    if data["randaugment_num_ops"] < 1:
        raise ValueError("training.continual.data.randaugment_num_ops must be at least 1.")
    data["randaugment_magnitude"] = int(data.get("randaugment_magnitude", 7))
    if not 0 <= data["randaugment_magnitude"] <= 30:
        raise ValueError("training.continual.data.randaugment_magnitude must be between 0 and 30.")
    data["augmix_severity"] = int(data.get("augmix_severity", 3))
    if not 1 <= data["augmix_severity"] <= 10:
        raise ValueError("training.continual.data.augmix_severity must be between 1 and 10.")
    data["augmix_width"] = int(data.get("augmix_width", 3))
    if data["augmix_width"] < 1:
        raise ValueError("training.continual.data.augmix_width must be at least 1.")
    data["augmix_depth"] = int(data.get("augmix_depth", -1))
    if data["augmix_depth"] == 0 or data["augmix_depth"] < -1:
        raise ValueError("training.continual.data.augmix_depth must be -1 or a positive integer.")
    data["augmix_alpha"] = float(data.get("augmix_alpha", 1.0))
    if data["augmix_alpha"] <= 0.0:
        raise ValueError("training.continual.data.augmix_alpha must be positive.")
    data["allow_under_min_training"] = bool(data.get("allow_under_min_training", False))
    data["cache_size"] = int(data.get("cache_size", 1000))
    data["validate_images_on_init"] = bool(data.get("validate_images_on_init", False))

    evaluation_best_metric = str(evaluation.get("best_metric", early_stopping.get("metric", "val_loss")))
    evaluation["best_metric"] = evaluation_best_metric
    evaluation["emit_ood_gate"] = bool(evaluation.get("emit_ood_gate", True))
    evaluation["require_ood_for_gate"] = bool(evaluation.get("require_ood_for_gate", True))
    evaluation["ood_benchmark_min_classes"] = int(evaluation.get("ood_benchmark_min_classes", 3))
    evaluation["min_in_distribution_samples"] = int(evaluation.get("min_in_distribution_samples", 30))
    evaluation["min_ood_samples"] = int(evaluation.get("min_ood_samples", 30))
    evaluation["min_ood_samples_per_type"] = int(evaluation.get("min_ood_samples_per_type", 5))
    evaluation["gate_auxiliary_ood_diagnostics"] = bool(
        evaluation.get("gate_auxiliary_ood_diagnostics", False)
    )
    evaluation.pop("ood_fallback_strategy", None)
    evaluation.pop("ood_benchmark_auto_run", None)

    early_metric = str(early_stopping.get("metric", evaluation_best_metric))
    inferred_mode = "min" if early_metric in {"val_loss", "generalization_gap"} else "max"
    early_stopping["enabled"] = bool(early_stopping.get("enabled", True))
    early_stopping["metric"] = early_metric
    early_stopping["mode"] = str(early_stopping.get("mode", inferred_mode))
    early_stopping["patience"] = int(early_stopping.get("patience", 5))
    early_stopping["min_delta"] = float(early_stopping.get("min_delta", 0.0))

    return normalized


def extract_continual_training_config(
    config: Optional[Dict[str, Any]],
    *,
    model_name: str = DEFAULT_BACKBONE_MODEL_NAME,
    device: Any = DEFAULT_DEVICE,
) -> Dict[str, Any]:
    payload = dict(config or {})
    if not payload:
        source: Dict[str, Any] = {}
    else:
        training = payload.get("training")
        if isinstance(training, dict) and isinstance(training.get("continual"), dict):
            source = dict(training.get("continual", {}))
        elif any(
            key in payload
            for key in ("backbone", "adapter", "fusion", "ood", "optimization", "data", "early_stopping", "evaluation")
        ):
            source = payload
        else:
            raise ValueError(
                "Continual training config must be provided under training.continual or as a canonical continual block."
            )
    return normalize_continual_training_config(source, model_name=model_name, device=device)
