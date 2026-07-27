"""Schema validation helpers for the public JSON config surface."""

from __future__ import annotations

from typing import Any, Dict, Mapping

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


def migrate_config_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    migrated = dict(payload or {})
    version = _read_config_schema_version(migrated)
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

    _raise_if_unsupported_config_keys_present(migrated)
    migrated[CONFIG_SCHEMA_VERSION_KEY] = CURRENT_CONFIG_SCHEMA_VERSION
    return migrated
