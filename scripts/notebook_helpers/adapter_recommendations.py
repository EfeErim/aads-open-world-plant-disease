"""Shared adapter recommendations for Notebook 2 and Notebook 6."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

ADAPTER_RECS: dict[str, dict[str, Any]] = {
    "grape__fruit": {
        "crop": "grape", "part": "fruit",
        "ood": "grape__fruit/ood",
        "oe": "grape__fruit/oe",
        "oe_enabled": True, "oe_w": 0.24, "allow_under_min": False,
        "defaults": {"EPOCHS": 32, "BATCH_SIZE": 48, "LEARNING_RATE": 1e-4, "LORA_R": 24, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.16, "OOD_FACTOR": 3.0, "LABEL_SMOOTHING": 0.10, "OE_LOSS_WEIGHT": 0.24, "REACT_ENABLED": True, "REACT_PERCENTILE": 0.985, "FUSION_DROPOUT": 0.14, "CLASSIFIER_REBALANCE_ENABLED": True, "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU": 1.20},
    },
    "grape__leaf": {
        "crop": "grape", "part": "leaf",
        "ood": "grape__leaf/ood",
        "oe": "grape__leaf/oe",
        "oe_enabled": True, "oe_w": 0.24, "allow_under_min": False,
        "defaults": {"EPOCHS": 30, "BATCH_SIZE": 56, "LEARNING_RATE": 1e-4, "LORA_R": 24, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.14, "OOD_FACTOR": 3.5, "LABEL_SMOOTHING": 0.10, "OE_LOSS_WEIGHT": 0.24, "REACT_ENABLED": True, "REACT_PERCENTILE": 0.985, "FUSION_DROPOUT": 0.12, "CLASSIFIER_REBALANCE_ENABLED": True, "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU": 1.15},
    },
    "strawberry__fruit": {
        "crop": "strawberry", "part": "fruit",
        "ood": "strawberry__fruit/ood",
        "oe": "strawberry__fruit/oe",
        "oe_enabled": True, "oe_w": 0.12, "allow_under_min": False,
        "defaults": {"EPOCHS": 38, "BATCH_SIZE": 40, "LEARNING_RATE": 5e-5, "LORA_R": 20, "LORA_ALPHA": 26, "LORA_DROPOUT": 0.13, "OOD_FACTOR": 2.0, "LABEL_SMOOTHING": 0.05, "OE_LOSS_WEIGHT": 0.12, "REACT_ENABLED": True, "REACT_PERCENTILE": 0.99, "FUSION_DROPOUT": 0.12, "CLASSIFIER_REBALANCE_ENABLED": True, "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU": 1.10},
    },
    "strawberry__leaf": {
        "crop": "strawberry", "part": "leaf",
        "ood": "strawberry__leaf/ood",
        "oe": "strawberry__leaf/oe",
        "oe_enabled": True, "oe_w": 0.30, "allow_under_min": False,
        "defaults": {"EPOCHS": 24, "BATCH_SIZE": 88, "LEARNING_RATE": 1.2e-4, "LORA_R": 24, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.125, "OOD_FACTOR": 4.0, "LABEL_SMOOTHING": 0.10, "EARLY_STOPPING_PATIENCE": 5, "OE_LOSS_WEIGHT": 0.30, "REACT_ENABLED": True, "REACT_PERCENTILE": 0.99, "FUSION_DROPOUT": 0.10, "CLASSIFIER_REBALANCE_ENABLED": False},
    },
    "tomato__fruit": {
        "crop": "tomato", "part": "fruit",
        "ood": "tomato__fruit/ood",
        "oe": "tomato__fruit/oe",
        "oe_enabled": True, "oe_w": 0.25, "allow_under_min": False,
        "defaults": {"EPOCHS": 30, "BATCH_SIZE": 64, "LEARNING_RATE": 6e-5, "LORA_R": 24, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.125, "OOD_FACTOR": 3.0, "LABEL_SMOOTHING": 0.10, "OE_LOSS_WEIGHT": 0.25, "REACT_ENABLED": True, "REACT_PERCENTILE": 0.985, "FUSION_DROPOUT": 0.12, "CLASSIFIER_REBALANCE_ENABLED": True, "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU": 1.15},
    },
    "tomato__leaf": {
        "crop": "tomato", "part": "leaf",
        "ood": "tomato__leaf/ood",
        "oe": "tomato__leaf/oe",
        "oe_enabled": True, "oe_w": 0.25, "allow_under_min": False,
        "defaults": {"EPOCHS": 26, "BATCH_SIZE": 88, "LEARNING_RATE": 1.1e-4, "LORA_R": 32, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.125, "OOD_FACTOR": 3.5, "LABEL_SMOOTHING": 0.10, "OE_LOSS_WEIGHT": 0.25, "REACT_ENABLED": True, "REACT_PERCENTILE": 0.985, "FUSION_DROPOUT": 0.10, "CLASSIFIER_REBALANCE_ENABLED": True, "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU": 1.10},
    },
    "apricot__fruit": {
        "crop": "apricot", "part": "fruit",
        "ood": "apricot__fruit/ood",
        "oe": "apricot__fruit/oe",
        "oe_enabled": True, "oe_w": 0.30, "allow_under_min": False,
        "defaults": {"EPOCHS": 36, "BATCH_SIZE": 40, "LEARNING_RATE": 1e-4, "LORA_R": 20, "LORA_ALPHA": 20, "LORA_DROPOUT": 0.20, "OOD_FACTOR": 4.5, "LABEL_SMOOTHING": 0.12, "OE_LOSS_WEIGHT": 0.30, "REACT_ENABLED": True, "REACT_PERCENTILE": 0.985, "FUSION_DROPOUT": 0.12, "CLASSIFIER_REBALANCE_ENABLED": True, "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU": 1.10},
    },
    "apricot__leaf": {
        "crop": "apricot", "part": "leaf",
        "ood": "apricot__leaf/ood",
        "oe": "apricot__leaf/oe",
        "oe_enabled": True, "oe_w": 0.30, "allow_under_min": False,
        "defaults": {"EPOCHS": 38, "BATCH_SIZE": 52, "LEARNING_RATE": 1.2e-4, "LORA_R": 26, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.125, "OOD_FACTOR": 4.5, "LABEL_SMOOTHING": 0.10, "OE_LOSS_WEIGHT": 0.30, "REACT_ENABLED": True, "REACT_PERCENTILE": 0.985, "FUSION_DROPOUT": 0.10, "CLASSIFIER_REBALANCE_ENABLED": True, "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU": 1.10},
    },
}


def get_adapter_recs() -> dict[str, dict[str, Any]]:
    """Return a copy so notebook cells can mutate defaults safely."""
    return deepcopy(ADAPTER_RECS)
