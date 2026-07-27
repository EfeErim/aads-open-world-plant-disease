"""Checkpoint and metadata helpers for the independent crop adapter surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

from src.shared.contracts import AdapterMetadata
from src.shared.json_utils import read_json_dict, write_json
from src.training.services.config_surface import extract_continual_training_config
from src.training.services.serialization import serialize_ood_state

TrainerFactory = Callable[[Dict[str, Any]], Any]
CheckpointPayloadFactory = Callable[[], type[Any]]


def _extract_continual_source(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = dict(payload or {})
    training = data.get("training")
    if isinstance(training, dict) and isinstance(training.get("continual"), dict):
        return dict(training.get("continual", {}))
    return data


def _has_explicit_loss_name(payload: Optional[Dict[str, Any]]) -> bool:
    source = _extract_continual_source(payload)
    optimization = source.get("optimization")
    return isinstance(optimization, dict) and "loss_name" in optimization


def _apply_legacy_ber_loss_compatibility(
    normalized: Dict[str, Any],
    *,
    source_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Keep old BER adapter exports loadable when loss metadata was absent."""
    ood = normalized.get("ood")
    optimization = normalized.get("optimization")
    if not isinstance(ood, dict) or not isinstance(optimization, dict):
        return normalized
    if not bool(ood.get("ber_enabled", False)):
        return normalized
    if _has_explicit_loss_name(source_config):
        return normalized

    optimization["loss_name"] = "cross_entropy"
    return normalized


def resolve_training_checkpoint_root(checkpoint_dir: str | Path) -> Path:
    root = Path(checkpoint_dir)
    if root.is_dir() and (root / "training_checkpoint").exists():
        return root / "training_checkpoint"
    return root


def normalize_trainer_config(
    trainer_config: Optional[Dict[str, Any]],
    *,
    model_name: str,
    device: Any,
    backbone: Optional[Dict[str, Any]] = None,
    fusion: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized = extract_continual_training_config(
        trainer_config,
        model_name=str(model_name),
        device=device,
    )
    normalized = _apply_legacy_ber_loss_compatibility(
        normalized,
        source_config=trainer_config,
    )
    if isinstance(backbone, dict) and backbone:
        normalized["backbone"] = {**dict(normalized.get("backbone", {})), **dict(backbone)}
    if isinstance(fusion, dict) and fusion:
        normalized["fusion"] = {**dict(normalized.get("fusion", {})), **dict(fusion)}
    normalized["device"] = str(normalized.get("device", str(device)))
    return normalized


def save_training_checkpoint(
    *,
    trainer: Any,
    checkpoint_dir: str,
    session_state: Optional[Dict[str, Any]] = None,
    run_id: str = "",
) -> Path:
    """Persist trainer and session state for fault-tolerant notebook runs."""
    root = Path(checkpoint_dir)
    checkpoint_dir_path = root / "training_checkpoint"
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    trainer_payload = trainer.snapshot_training_state()
    torch.save(trainer_payload.to_dict(), checkpoint_dir_path / "training_checkpoint.pt")

    session_payload = dict(session_state or {})
    meta = {
        "schema_version": trainer_payload.schema_version,
        "created_at": trainer_payload.created_at,
        "run_id": str(run_id),
        "current_epoch": int(trainer_payload.current_epoch),
        "global_step": int(dict(session_payload.get("progress_state", {})).get("global_step", 0)),
        "epoch": int(dict(session_payload.get("progress_state", {})).get("epoch", 0)),
    }
    write_json(checkpoint_dir_path / "session_state.json", session_payload)
    write_json(checkpoint_dir_path / "checkpoint_meta.json", meta)
    return checkpoint_dir_path


def load_training_checkpoint(
    *,
    checkpoint_dir: str,
    device: Any,
    trainer: Any,
    trainer_factory: TrainerFactory,
    checkpoint_payload_factory: CheckpointPayloadFactory,
    model_name: str,
) -> tuple[Any, Any, Dict[str, Any]]:
    """Load trainer and session payloads from a saved checkpoint directory."""
    root = resolve_training_checkpoint_root(checkpoint_dir)
    trainer_payload_raw = torch.load(root / "training_checkpoint.pt", map_location=device, weights_only=False)
    trainer_payload = checkpoint_payload_factory().from_dict(trainer_payload_raw)
    if trainer is None:
        normalized = normalize_trainer_config(
            trainer_payload.trainer_config,
            model_name=model_name,
            device=device,
        )
        trainer = trainer_factory(normalized)
    trainer_payload = trainer.restore_training_state(trainer_payload)

    session_payload: Dict[str, Any] = {}
    session_path = root / "session_state.json"
    if session_path.exists():
        session_payload = read_json_dict(session_path)
    return trainer, trainer_payload, session_payload


def build_checkpoint_load_result(*, trainer_payload: Any, session_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trainer_state": trainer_payload.to_dict(),
        "session_state": session_payload,
        "run_id": str(dict(session_payload).get("run_id", "")),
        "progress_state": dict(session_payload.get("progress_state", {})) if isinstance(session_payload, dict) else {},
        "history": dict(session_payload.get("history", {})) if isinstance(session_payload, dict) else {},
        "best_metric_state": (
            dict(session_payload.get("best_metric_state", {})) if isinstance(session_payload, dict) else {}
        ),
    }


def build_runtime_adapter_metadata(
    *,
    trainer: Any,
    class_to_idx: Dict[str, int],
    schema_version: str,
    engine: str,
    crop_name: str,
    part_name: str = "",
    model_name: str,
    ood_calibration_version: int,
    target_modules_resolved: list[str],
) -> Dict[str, Any]:
    trainer_config = (
        trainer.config.as_contract_dict()
        if hasattr(trainer.config, "as_contract_dict")
        else {"backbone": {"model_name": getattr(trainer.config, "backbone_model_name", model_name)}}
    )
    ood_detector = getattr(trainer, "ood_detector", None)
    ood_state = (
        serialize_ood_state(ood_detector, strict=False)
        if hasattr(ood_detector, "class_stats")
        else {
            "threshold_factor": 2.0,
            "primary_score_method": "ensemble",
            "calibration_version": ood_calibration_version,
            "class_stats": {},
            "knn_k": 10,
            "knn_bank_cap": 256,
        }
    )
    return AdapterMetadata(
        schema_version=schema_version,
        engine=engine,
        crop_name=str(crop_name),
        part_name=str(part_name),
        trainer_config=trainer_config,
        config_hash=str(getattr(trainer, "_config_hash", "")),
        backbone={
            "model_name": getattr(trainer.config, "backbone_model_name", model_name),
            "frozen": True,
        },
        fusion={
            "layers": list(getattr(trainer.config, "fusion_layers", [2, 5, 8, 11])),
            "output_dim": int(getattr(trainer.config, "fusion_output_dim", 768)),
            "dropout": float(getattr(trainer.config, "fusion_dropout", 0.1)),
            "gating": str(getattr(trainer.config, "fusion_gating", "softmax")),
        },
        class_to_idx=dict(class_to_idx),
        ood_calibration={"version": int(ood_calibration_version)},
        ood_state=ood_state,
        target_modules_resolved=list(target_modules_resolved),
        adapter_runtime={
            "peft_available": bool(getattr(trainer, "_peft_available", False)),
            "adapter_wrapped": bool(getattr(trainer, "_adapter_wrapped", False)),
            "degraded_without_peft": bool(
                getattr(trainer, "_peft_available", False) and not getattr(trainer, "_adapter_wrapped", False)
            ),
        },
    ).to_dict()
