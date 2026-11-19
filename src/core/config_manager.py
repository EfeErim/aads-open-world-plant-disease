from __future__ import annotations

import copy
import logging
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Mapping, Optional

from src.shared.json_utils import deep_merge, read_json_dict
from src.training.services.config_surface import (
    DEFAULT_BACKBONE_MODEL_NAME,
    normalize_continual_training_config,
)

CONFIG_SCHEMA_VERSION_KEY = "config_schema_version"
CURRENT_CONFIG_SCHEMA_VERSION = 2

_CONFIG_SURFACE_KEYS = frozenset({"training", "router", "ood", "inference", "colab"})
_UNSUPPORTED_TOP_LEVEL_KEYS = frozenset({"ood"})
_UNSUPPORTED_COLAB_TRAINING_KEYS = frozenset({"checkpoint_interval"})


def is_versioned_config_surface_payload(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return bool(_CONFIG_SURFACE_KEYS & set(payload.keys()))


def _read_config_schema_version(payload: Dict[str, Any]) -> int:
    raw = payload.get(CONFIG_SCHEMA_VERSION_KEY, 0)
    try:
        version = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{CONFIG_SCHEMA_VERSION_KEY} must be an integer-compatible value.") from None
    if version < 0:
        raise ValueError(f"{CONFIG_SCHEMA_VERSION_KEY} must be non-negative.")
    return version


def _raise_if_unsupported_config_keys_present(payload: Dict[str, Any]) -> None:
    unsupported_root_keys = sorted(_UNSUPPORTED_TOP_LEVEL_KEYS & set(payload.keys()))
    if unsupported_root_keys:
        raise ValueError(
            "Unsupported top-level config sections: "
            + ", ".join(unsupported_root_keys)
            + ". Move these values under training.continual."
        )

    colab_training = payload.get("colab", {}).get("training")
    if isinstance(colab_training, dict):
        unsupported_colab_training_keys = sorted(_UNSUPPORTED_COLAB_TRAINING_KEYS & set(colab_training.keys()))
        if unsupported_colab_training_keys:
            raise ValueError(
                "Unsupported colab.training keys: "
                + ", ".join(unsupported_colab_training_keys)
                + ". Use checkpoint_every_n_steps."
            )


def validate_config_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    validated = dict(payload or {})
    version = _read_config_schema_version(validated)
    if version > CURRENT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {CONFIG_SCHEMA_VERSION_KEY}={version}; "
            f"current loader supports up to {CURRENT_CONFIG_SCHEMA_VERSION}."
        )
    if version < CURRENT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {CONFIG_SCHEMA_VERSION_KEY}={version}. "
            f"Config files must declare {CONFIG_SCHEMA_VERSION_KEY}={CURRENT_CONFIG_SCHEMA_VERSION}."
        )

    _raise_if_unsupported_config_keys_present(validated)
    validated[CONFIG_SCHEMA_VERSION_KEY] = CURRENT_CONFIG_SCHEMA_VERSION
    return validated

#!/usr/bin/env python3




logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Dict[str, Any]:
    return read_json_dict(path)


# The repository's own config/ directory. Used when no config_dir is given, so the CLI and library calls work
# from any working directory; an explicit config_dir keeps its usual meaning (relative paths resolve from cwd).
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class ConfigurationManager:
    """Load `config/base.json` and optionally merge `config/<environment>.json`."""

    def __init__(self, config_dir: str | Path | None = None, environment: Optional[str] = None) -> None:
        self.config_dir = DEFAULT_CONFIG_DIR if config_dir is None else Path(config_dir)
        self._environment = environment
        self._base_config: Optional[Dict[str, Any]] = None
        self._merged_config: Optional[Dict[str, Any]] = None

    def load_base_config(self) -> Dict[str, Any]:
        if self._base_config is not None:
            return copy.deepcopy(self._base_config)
        base_path = self.config_dir / "base.json"
        self._base_config = validate_config_payload(_read_json(base_path))
        return copy.deepcopy(self._base_config)

    def _available_environment_names(self) -> list[str]:
        names: list[str] = []
        for candidate in sorted(self.config_dir.glob("*.json")):
            if candidate.name == "base.json":
                continue
            try:
                payload = _read_json(candidate)
            except (OSError, TypeError, ValueError) as exc:
                logger.debug("Skipping unreadable config candidate %s: %s", candidate, exc)
                continue
            if is_versioned_config_surface_payload(payload):
                names.append(candidate.stem)
        return names

    def get_environment_config(self, env: str) -> Dict[str, Any]:
        if not env:
            return {}
        env_path = self.config_dir / f"{env}.json"
        if not env_path.exists():
            available = self._available_environment_names()
            available_text = ", ".join(available) if available else "(none)"
            raise FileNotFoundError(
                f"Config environment '{env}' was not found under {self.config_dir}. "
                f"Available environments: {available_text}"
            )
        return validate_config_payload(_read_json(env_path))

    def load_all_configs(self) -> Dict[str, Any]:
        if self._merged_config is not None:
            return copy.deepcopy(self._merged_config)
        merged = self.load_base_config()
        if self._environment:
            merged = deep_merge(merged, self.get_environment_config(self._environment))
        self._normalize_training_surface(merged)
        merged[CONFIG_SCHEMA_VERSION_KEY] = CURRENT_CONFIG_SCHEMA_VERSION
        self._merged_config = merged
        return copy.deepcopy(merged)

    def _normalize_training_surface(self, merged_config: Dict[str, Any]) -> None:
        training = merged_config.setdefault("training", {})
        continual = normalize_continual_training_config(
            training.get("continual", {}),
            model_name=str(
                dict(training.get("continual", {})).get("backbone", {}).get(
                    "model_name",
                    DEFAULT_BACKBONE_MODEL_NAME,
                )
            ),
            device=dict(training.get("continual", {})).get("device", "cuda"),
        )
        training["continual"] = continual

        colab = merged_config.setdefault("colab", {})
        colab_training = colab.setdefault("training", {})
        checkpoint_every_n_steps = colab_training.get("checkpoint_every_n_steps")

        colab_training["num_workers"] = int(colab_training.get("num_workers", 2))
        colab_training["pin_memory"] = bool(colab_training.get("pin_memory", True))
        colab_training["validation_every_n_epochs"] = max(
            1,
            int(colab_training.get("validation_every_n_epochs", 1)),
        )
        colab_training["stdout_progress_batch_interval"] = int(colab_training.get("stdout_progress_batch_interval", 50))
        colab_training["stdout_progress_min_interval_sec"] = float(
            colab_training.get("stdout_progress_min_interval_sec", 15.0)
        )
        colab_training["checkpoint_every_n_steps"] = int(
            checkpoint_every_n_steps if checkpoint_every_n_steps is not None else 200
        )
        colab_training["checkpoint_on_exception"] = bool(colab_training.get("checkpoint_on_exception", True))


_CONFIG_LOCK = RLock()
_CONFIG_CACHE: Dict[tuple[str, Optional[str]], ConfigurationManager] = {}


def _get_manager(config_dir: str | Path | None = None, environment: Optional[str] = None) -> ConfigurationManager:
    key = (str(DEFAULT_CONFIG_DIR if config_dir is None else Path(config_dir)), environment)
    with _CONFIG_LOCK:
        manager = _CONFIG_CACHE.get(key)
        if manager is None:
            manager = ConfigurationManager(config_dir=config_dir, environment=environment)
            _CONFIG_CACHE[key] = manager
        return manager


def get_config(environment: Optional[str] = None, config_dir: str | Path | None = None) -> Dict[str, Any]:
    return _get_manager(config_dir=config_dir, environment=environment).load_all_configs()


