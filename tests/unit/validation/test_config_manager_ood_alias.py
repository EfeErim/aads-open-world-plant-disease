import json

import pytest

from src.core.config_manager import ConfigurationManager
from src.core.config_migrations import CURRENT_CONFIG_SCHEMA_VERSION


def _write_base_config(path, *, training_ood=None, top_level_ood=None):
    payload = {
        "config_schema_version": CURRENT_CONFIG_SCHEMA_VERSION,
        "router": {"enabled": True, "type": "enhanced"},
        "training": {
            "continual": {
                "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
                "adapter": {"target_modules_strategy": "all_linear_transformer"},
                "fusion": {"layers": [2, 5, 8, 11]},
            }
        },
    }
    if training_ood is not None:
        payload["training"]["continual"]["ood"] = training_ood
    if top_level_ood is not None:
        payload["ood"] = dict(top_level_ood)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_config_manager_backfills_training_ood_threshold_from_top_level(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_base_config(config_dir / "base.json", top_level_ood={"threshold_factor": 2.5})

    manager = ConfigurationManager(config_dir=str(config_dir))
    with pytest.raises(ValueError, match="Unsupported top-level config sections: ood"):
        manager.load_all_configs()


def test_config_manager_keeps_canonical_training_ood_threshold(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_base_config(
        config_dir / "base.json",
        training_ood={"threshold_factor": 1.8},
    )

    manager = ConfigurationManager(config_dir=str(config_dir))
    merged = manager.load_all_configs()

    assert merged["training"]["continual"]["ood"]["threshold_factor"] == 1.8
    assert "ood" not in merged
