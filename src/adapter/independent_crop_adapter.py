#!/usr/bin/env python3
"""v6 independent crop adapter with continual SD-LoRA lifecycle."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional

import torch

from src.adapter.checkpointing import (
    build_checkpoint_load_result,
    build_runtime_adapter_metadata,
    normalize_trainer_config,
)
from src.adapter.checkpointing import (
    load_training_checkpoint as load_adapter_training_checkpoint,
)
from src.adapter.checkpointing import (
    save_training_checkpoint as save_adapter_training_checkpoint,
)
from src.shared.adapter_paths import normalize_adapter_name, resolve_adapter_bundle_dir
from src.shared.contracts import AdapterMetadata
from src.shared.json_utils import deep_merge, read_json_dict, write_json
from src.training.services.config_surface import extract_continual_training_config
from src.training.services.runtime import resolve_runtime_device, resolve_session_num_epochs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer
    from src.training.session import ContinualTrainingSession
    from src.training.types import TrainingCheckpointPayload


def _trainer_types() -> tuple[type["ContinualSDLoRAConfig"], type["ContinualSDLoRATrainer"]]:
    from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer

    return ContinualSDLoRAConfig, ContinualSDLoRATrainer


def _training_session_type() -> type["ContinualTrainingSession"]:
    from src.training.session import ContinualTrainingSession

    return ContinualTrainingSession


def _checkpoint_payload_type() -> type["TrainingCheckpointPayload"]:
    from src.training.types import TrainingCheckpointPayload

    return TrainingCheckpointPayload


class IndependentCropAdapter:
    """Per-crop continual adapter runtime surface for v6."""

    def __init__(
        self,
        crop_name: str,
        part_name: str = "unspecified",
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        device: str = "cuda",
    ) -> None:
        self.crop_name = str(crop_name)
        self.part_name = normalize_adapter_name(part_name)
        self.model_name = str(model_name)
        self.device = resolve_runtime_device(device)

        self.engine = "continual_sd_lora"
        self.schema_version = "v6"
        self.class_to_idx: Dict[str, int] = {}
        self.is_trained = False
        self._trainer: Optional["ContinualSDLoRATrainer"] = None
        self._calibration_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None
        self._metadata_overrides: Dict[str, Any] = {}

        logger.info(
            "IndependentCropAdapter initialized for %s/%s on %s",
            self.crop_name,
            self.part_name,
            self.device,
        )

    @property
    def target_modules_resolved(self) -> list[str]:
        if self._trainer is None:
            return []
        return list(self._trainer.target_modules_resolved)

    def _require_trainer(self) -> "ContinualSDLoRATrainer":
        if self._trainer is None:
            raise RuntimeError("Adapter is not initialized.")
        return self._trainer

    def _store_trainer(self, trainer: "ContinualSDLoRATrainer") -> "ContinualSDLoRATrainer":
        self._trainer = trainer
        self.class_to_idx = dict(getattr(trainer, "class_to_idx", {}))
        self.is_trained = True
        return trainer

    def _build_trainer_from_config(self, config: Optional[Dict[str, Any]]) -> "ContinualSDLoRATrainer":
        continual_dict = self._normalize_continual_config(config)
        config_cls, trainer_cls = _trainer_types()
        trainer_config = config_cls.from_training_config(continual_dict)
        return trainer_cls(trainer_config)

    @staticmethod
    def _resolve_asset_dir(checkpoint_dir: str | Path) -> Path:
        return resolve_adapter_bundle_dir(checkpoint_dir)

    @property
    def trainer(self) -> "ContinualSDLoRATrainer":
        return self._require_trainer()

    @property
    def ood_calibration_version(self) -> int:
        if self._trainer is None:
            return 0
        return int(self._trainer.ood_detector.calibration_version)

    def parameters(self):
        if self._trainer is None:
            return iter(())
        modules = [self._trainer.adapter_model, self._trainer.classifier, self._trainer.fusion]
        for module in modules:
            if module is None:
                continue
            yield from module.parameters()

    def _normalize_continual_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return extract_continual_training_config(
            config,
            model_name=self.model_name,
            device=self.device,
        )

    def initialize_engine(
        self,
        *,
        class_names: Optional[Iterable[str]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Initialize frozen backbone + continual adapter engine."""
        if class_to_idx is not None:
            self.class_to_idx = {str(k): int(v) for k, v in class_to_idx.items()}
        elif class_names is not None:
            self.class_to_idx = {str(name): idx for idx, name in enumerate(class_names)}

        trainer = self._build_trainer_from_config(config)
        trainer.initialize_engine(class_to_idx=self.class_to_idx)
        self._store_trainer(trainer)

        return {
            "status": "initialized",
            "engine": self.engine,
            "schema_version": self.schema_version,
            "num_classes": len(self.class_to_idx),
        }

    def add_classes(self, new_classes: Iterable[str]) -> Dict[str, Any]:
        """Add new class labels and expand classifier."""
        trainer = self._require_trainer()
        self.class_to_idx = trainer.add_classes(new_classes)
        return {
            "status": "classes_added",
            "num_classes": len(self.class_to_idx),
            "class_to_idx": dict(self.class_to_idx),
        }

    def build_training_session(
        self,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        *,
        num_epochs: Optional[int] = None,
        val_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
        observers: Optional[list[Callable[[Dict[str, Any]], None]]] = None,
        stop_policy: Optional[Callable[[], bool]] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        run_id: str = "",
        checkpoint_every_n_steps: int = 0,
        checkpoint_on_exception: bool = False,
        validation_every_n_epochs: int = 1,
    ) -> "ContinualTrainingSession":
        """Build a training session around the initialized trainer."""
        trainer = self._require_trainer()
        self._calibration_loader = val_loader if val_loader is not None else train_loader
        resolved_epochs = resolve_session_num_epochs(trainer.config, num_epochs)
        training_session_cls = _training_session_type()
        return training_session_cls(
            trainer,
            train_loader,
            resolved_epochs,
            val_loader=val_loader,
            observers=observers,
            stop_policy=stop_policy,
            resume_state=resume_state,
            run_id=run_id,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            checkpoint_on_exception=checkpoint_on_exception,
            validation_every_n_epochs=validation_every_n_epochs,
        )

    def save_training_checkpoint(
        self,
        checkpoint_dir: str,
        *,
        session_state: Optional[Dict[str, Any]] = None,
        run_id: str = "",
    ) -> Path:
        """Persist trainer and session state for fault-tolerant notebook runs."""
        trainer = self._require_trainer()
        return save_adapter_training_checkpoint(
            trainer=trainer,
            checkpoint_dir=checkpoint_dir,
            session_state=session_state,
            run_id=run_id,
        )

    def load_training_checkpoint(self, checkpoint_dir: str) -> Dict[str, Any]:
        """Load resumable trainer and session state from checkpoint directory."""
        config_cls, trainer_cls = _trainer_types()

        def _trainer_factory(normalized: Dict[str, Any]) -> "ContinualSDLoRATrainer":
            cfg = config_cls.from_training_config(normalized)
            return trainer_cls(cfg)

        trainer, trainer_payload, session_payload = load_adapter_training_checkpoint(
            checkpoint_dir=checkpoint_dir,
            device=self.device,
            trainer=self._trainer,
            trainer_factory=_trainer_factory,
            checkpoint_payload_factory=_checkpoint_payload_type,
            model_name=self.model_name,
        )
        self._store_trainer(trainer)
        class_to_idx = trainer_payload.class_to_idx
        if isinstance(class_to_idx, dict):
            self.class_to_idx = {str(k): int(v) for k, v in class_to_idx.items()}
        return build_checkpoint_load_result(trainer_payload=trainer_payload, session_payload=session_payload)

    def calibrate_ood(self, loader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Calibrate OOD statistics for current classes."""
        trainer = self._require_trainer()
        self._calibration_loader = loader
        result = trainer.calibrate_ood(loader)
        return {
            "status": "calibrated",
            "ood_calibration": {
                "version": self.ood_calibration_version,
                "num_classes": int(result.get("num_classes", 0)),
            },
        }

    def set_export_metadata(
        self,
        *,
        ood_calibration: Optional[Dict[str, Any]] = None,
        adapter_runtime: Optional[Dict[str, Any]] = None,
    ) -> None:
        overrides: Dict[str, Any] = {}
        if isinstance(ood_calibration, dict) and ood_calibration:
            overrides["ood_calibration"] = dict(ood_calibration)
        if isinstance(adapter_runtime, dict) and adapter_runtime:
            overrides["adapter_runtime"] = dict(adapter_runtime)
        if overrides:
            self._metadata_overrides = deep_merge(self._metadata_overrides, overrides)

    def _ensure_ood_calibrated_for_export(self) -> None:
        trainer = self._require_trainer()
        issue = trainer.ood_detector.calibration_issue()
        if issue is None:
            return
        if self._calibration_loader is None:
            raise RuntimeError(f"{issue} No calibration loader is available for automatic export calibration.")
        logger.info("Auto-calibrating OOD state before adapter export for %s", self.crop_name)
        self.calibrate_ood(self._calibration_loader)

    def predict_with_ood(self, image: torch.Tensor) -> Dict[str, Any]:
        """Return diagnosis + v6 OOD payload for a single image tensor."""
        trainer = self._require_trainer()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return trainer.predict_with_ood(image.to(self.device))

    def predict_batch_with_ood(self, images: torch.Tensor) -> list[Dict[str, Any]]:
        """Return diagnosis + v6 OOD payloads for a tensor batch."""
        trainer = self._require_trainer()
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if hasattr(trainer, "predict_with_ood_batch"):
            return trainer.predict_with_ood_batch(images.to(self.device))
        return [trainer.predict_with_ood(image.unsqueeze(0).to(self.device)) for image in images]

    def detect_ood_dynamic(self, image: torch.Tensor) -> Dict[str, Any]:
        """Compatibility helper returning OOD analysis fields only."""
        result = self.predict_with_ood(image)
        ood = result.get("ood_analysis", {})
        return {
            "score_method": str(ood.get("score_method", "ensemble")),
            "primary_score": float(ood.get("primary_score", 0.0)),
            "decision_threshold": float(ood.get("decision_threshold", 0.0)),
            "is_ood": bool(ood.get("is_ood", False)),
            "calibration_version": int(ood.get("calibration_version", 0)),
        }

    def _metadata_payload(self) -> Dict[str, Any]:
        trainer = self._require_trainer()
        return build_runtime_adapter_metadata(
            trainer=trainer,
            class_to_idx=self.class_to_idx,
            schema_version=self.schema_version,
            engine=self.engine,
            crop_name=self.crop_name,
            part_name=self.part_name,
            model_name=self.model_name,
            ood_calibration_version=self.ood_calibration_version,
            target_modules_resolved=list(self.target_modules_resolved),
        )

    def save_adapter(self, checkpoint_dir: str) -> Path:
        """Persist adapter assets with v6 metadata schema."""
        trainer = self._require_trainer()
        self._ensure_ood_calibrated_for_export()

        root = Path(checkpoint_dir)
        asset_dir = trainer.save_adapter(str(root))
        meta_path = asset_dir / "adapter_meta.json"
        metadata = self._metadata_payload()
        if meta_path.exists():
            metadata = deep_merge(read_json_dict(meta_path), metadata)
        if self._metadata_overrides:
            metadata = deep_merge(metadata, self._metadata_overrides)
        write_json(meta_path, metadata)
        return asset_dir

    def load_adapter(self, checkpoint_dir: str) -> None:
        """Load adapter assets and metadata from disk."""
        asset_dir = self._resolve_asset_dir(checkpoint_dir)

        meta_path = asset_dir / "adapter_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"adapter_meta.json not found in {asset_dir}")

        meta = AdapterMetadata.from_dict(read_json_dict(meta_path))
        exported_crop = str(meta.crop_name or "").strip().lower()
        exported_part = str(meta.part_name or "").strip().lower()
        requested_crop = str(self.crop_name or "").strip().lower()
        requested_part = str(self.part_name or "").strip().lower()
        if exported_crop and requested_crop and exported_crop != requested_crop:
            raise ValueError(
                f"Adapter crop mismatch: requested '{self.crop_name}' but bundle declares '{meta.crop_name}'."
            )
        if (
            exported_part
            and exported_part != "unspecified"
            and requested_part
            and requested_part != "unspecified"
            and exported_part != requested_part
        ):
            raise ValueError(
                f"Adapter part mismatch: requested '{self.part_name}' but bundle declares '{meta.part_name}'."
            )
        if exported_crop:
            self.crop_name = str(meta.crop_name)
        if exported_part and exported_part != "unspecified":
            self.part_name = normalize_adapter_name(meta.part_name)
        self.class_to_idx = dict(meta.class_to_idx)

        normalized = normalize_trainer_config(
            meta.trainer_config,
            model_name=self.model_name,
            device=self.device,
            backbone=dict(meta.backbone or {"model_name": self.model_name}),
            fusion=dict(meta.fusion or {"layers": [2, 5, 8, 11]}),
        )

        config_cls, trainer_cls = _trainer_types()
        cfg = config_cls.from_training_config(normalized)
        trainer = trainer_cls(cfg)
        trainer.load_adapter(str(asset_dir))
        self._store_trainer(trainer)

    def get_summary(self) -> Dict[str, Any]:
        """Return concise runtime adapter summary."""
        return {
            "crop_name": self.crop_name,
            "part_name": self.part_name,
            "model_name": self.model_name,
            "engine": self.engine,
            "schema_version": self.schema_version,
            "is_trained": self.is_trained,
            "num_classes": len(self.class_to_idx),
            "class_to_idx": dict(self.class_to_idx),
            "ood_calibration_version": self.ood_calibration_version,
        }
