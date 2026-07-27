import json

import pytest

from src.core.config_manager import ConfigurationManager
from src.core.config_migrations import CURRENT_CONFIG_SCHEMA_VERSION, is_versioned_config_surface_payload


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _canonical_base_payload():
    return {
        "config_schema_version": CURRENT_CONFIG_SCHEMA_VERSION,
        "router": {"enabled": True},
        "training": {
            "continual": {
                "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
                "adapter": {"target_modules_strategy": "all_linear_transformer"},
                "fusion": {"layers": [2, 5, 8, 11]},
                "ood": {"threshold_factor": 2.0, "primary_score_method": "auto"},
            }
        },
    }


def test_manager_rejects_unversioned_surface_payload(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = _canonical_base_payload()
    payload.pop("config_schema_version", None)
    _write_json(config_dir / "base.json", payload)

    with pytest.raises(ValueError, match=f"must declare config_schema_version={CURRENT_CONFIG_SCHEMA_VERSION}"):
        ConfigurationManager(config_dir=str(config_dir)).load_all_configs()


def test_manager_rejects_top_level_ood_section(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = _canonical_base_payload()
    payload["ood"] = {"threshold_factor": 2.5, "primary_score_method": "knn"}
    _write_json(config_dir / "base.json", payload)

    with pytest.raises(ValueError, match="Unsupported top-level config sections: ood"):
        ConfigurationManager(config_dir=str(config_dir)).load_all_configs()


def test_manager_rejects_checkpoint_interval_alias(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "base.json", _canonical_base_payload())
    _write_json(
        config_dir / "env_alias.json",
        {
            "config_schema_version": CURRENT_CONFIG_SCHEMA_VERSION,
            "colab": {"training": {"checkpoint_interval": 321}},
        },
    )

    with pytest.raises(ValueError, match="Use checkpoint_every_n_steps"):
        ConfigurationManager(config_dir=str(config_dir), environment="env_alias").load_all_configs()


def test_validate_merged_config_requires_current_schema_version(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "base.json", _canonical_base_payload())

    manager = ConfigurationManager(config_dir=str(config_dir))
    assert manager.validate_merged_config() is True


def test_manager_rejects_future_config_schema_version(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = _canonical_base_payload()
    payload["config_schema_version"] = CURRENT_CONFIG_SCHEMA_VERSION + 1
    _write_json(config_dir / "base.json", payload)

    with pytest.raises(ValueError, match="Unsupported config_schema_version"):
        ConfigurationManager(config_dir=str(config_dir)).load_all_configs()


def test_manager_rejects_missing_environment_config(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "base.json", _canonical_base_payload())
    _write_json(
        config_dir / "env_alias.json",
        {
            "config_schema_version": CURRENT_CONFIG_SCHEMA_VERSION,
            "colab": {"training": {"checkpoint_every_n_steps": 123}},
        },
    )

    with pytest.raises(FileNotFoundError, match="Available environments: env_alias"):
        ConfigurationManager(config_dir=str(config_dir), environment="missing_env").load_all_configs()


def test_load_config_file_requires_versioned_surface(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = _canonical_base_payload()
    payload.pop("config_schema_version", None)
    _write_json(config_dir / "env_alias.json", payload)

    with pytest.raises(ValueError, match=f"must declare config_schema_version={CURRENT_CONFIG_SCHEMA_VERSION}"):
        ConfigurationManager(config_dir=str(config_dir)).load_config_file("env_alias.json")


def test_load_config_file_leaves_non_surface_payload_untouched(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "plant_taxonomy.json", {"crops": {"tomato": ["leaf"]}})

    payload = ConfigurationManager(config_dir=str(config_dir)).load_config_file("plant_taxonomy.json")

    assert payload == {"crops": {"tomato": ["leaf"]}}


def test_load_all_configs_reuses_cached_merged_payload(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "base.json", _canonical_base_payload())
    _write_json(
        config_dir / "env_alias.json",
        {
            "config_schema_version": CURRENT_CONFIG_SCHEMA_VERSION,
            "colab": {"training": {"checkpoint_every_n_steps": 123}},
        },
    )

    read_calls = {"count": 0}
    original_read_json = __import__("src.core.config_manager", fromlist=["_read_json"])._read_json

    def _counting_read_json(path):
        read_calls["count"] += 1
        return original_read_json(path)

    monkeypatch.setattr("src.core.config_manager._read_json", _counting_read_json)

    manager = ConfigurationManager(config_dir=str(config_dir), environment="env_alias")
    first = manager.load_all_configs()
    second = manager.load_all_configs()

    assert first == second
    assert read_calls["count"] == 2


def test_is_versioned_config_surface_payload_matches_supported_sections():
    assert is_versioned_config_surface_payload({"training": {}}) is True
    assert is_versioned_config_surface_payload({"colab": {}}) is True
    assert is_versioned_config_surface_payload({"crops": {"tomato": ["leaf"]}}) is False
