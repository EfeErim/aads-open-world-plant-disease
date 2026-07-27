#!/usr/bin/env python3
"""Minimal JSON config loader for the slimmed training + inference repo."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

from src.core.config_migrations import (
    CONFIG_SCHEMA_VERSION_KEY,
    CURRENT_CONFIG_SCHEMA_VERSION,
    is_versioned_config_surface_payload,
    migrate_config_payload,
)
from src.shared.json_utils import deep_merge, read_json_dict
from src.training.quantization import assert_no_prohibited_4bit_flags
from src.training.services.config_surface import (
    DEFAULT_BACKBONE_MODEL_NAME,
    normalize_continual_training_config,
)

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Dict[str, Any]:
    return read_json_dict(path)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    return deep_merge(base, override)


class ConfigurationManager:
    """Load `config/base.json` and optionally merge `config/<environment>.json`."""

    def __init__(self, config_dir: str = "config", environment: Optional[str] = None) -> None:
        self.config_dir = Path(config_dir)
        self._environment = environment
        self._base_config: Optional[Dict[str, Any]] = None
        self._merged_config: Optional[Dict[str, Any]] = None

    def load_base_config(self) -> Dict[str, Any]:
        if self._base_config is not None:
            return copy.deepcopy(self._base_config)
        base_path = self.config_dir / "base.json"
        self._base_config = migrate_config_payload(_read_json(base_path))
        return copy.deepcopy(self._base_config)

    def load_config_file(self, filename: str, schema_name: Optional[str] = None) -> Dict[str, Any]:
        config_path = self.config_dir / filename
        if not config_path.exists():
            return {}
        payload = _read_json(config_path)
        if is_versioned_config_surface_payload(payload):
            payload = migrate_config_payload(payload)
        if schema_name and schema_name not in payload:
            return {schema_name: payload}
        return payload

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
        return migrate_config_payload(_read_json(env_path))

    def load_all_configs(self) -> Dict[str, Any]:
        if self._merged_config is not None:
            return copy.deepcopy(self._merged_config)
        merged = self.load_base_config()
        if self._environment:
            merged = _deep_merge(merged, self.get_environment_config(self._environment))
        self._normalize_training_surface(merged)
        merged[CONFIG_SCHEMA_VERSION_KEY] = CURRENT_CONFIG_SCHEMA_VERSION
        assert_no_prohibited_4bit_flags(merged)
        self._merged_config = merged
        return copy.deepcopy(merged)

    def reload_config(self) -> Dict[str, Any]:
        self._base_config = None
        self._merged_config = None
        return self.load_all_configs()

    def get_config(self, key: str, default: Any = None) -> Any:
        config = self._merged_config if self._merged_config is not None else self.load_all_configs()
        current: Any = config
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def validate_merged_config(self) -> bool:
        config = self._merged_config if self._merged_config is not None else self.load_all_configs()
        if int(config.get(CONFIG_SCHEMA_VERSION_KEY, 0)) != CURRENT_CONFIG_SCHEMA_VERSION:
            return False
        required = ("training", "router")
        if any(section not in config for section in required):
            return False
        continual = config.get("training", {}).get("continual")
        return isinstance(continual, dict)

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


def _get_manager(config_dir: str = "config", environment: Optional[str] = None) -> ConfigurationManager:
    key = (str(Path(config_dir)), environment)
    with _CONFIG_LOCK:
        manager = _CONFIG_CACHE.get(key)
        if manager is None:
            manager = ConfigurationManager(config_dir=config_dir, environment=environment)
            _CONFIG_CACHE[key] = manager
        return manager


def get_config(environment: Optional[str] = None, config_dir: str = "config") -> Dict[str, Any]:
    return _get_manager(config_dir=config_dir, environment=environment).load_all_configs()


def reload_configuration(environment: Optional[str] = None, config_dir: str = "config") -> Dict[str, Any]:
    return _get_manager(config_dir=config_dir, environment=environment).reload_config()
