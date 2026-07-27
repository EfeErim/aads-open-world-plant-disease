import json
from pathlib import Path

from scripts import validate_config_schema as schema
from src.core.config_migrations import CURRENT_CONFIG_SCHEMA_VERSION


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_config_schema_versions_accepts_current_surface_files(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config_dir / "base.json",
        {"config_schema_version": CURRENT_CONFIG_SCHEMA_VERSION, "training": {}, "router": {}, "ood": {}},
    )
    _write_json(
        config_dir / "colab.json",
        {"config_schema_version": CURRENT_CONFIG_SCHEMA_VERSION, "colab": {}, "training": {}},
    )
    _write_json(config_dir / "plant_taxonomy.json", {"crops": {"tomato": ["leaf"]}})

    checked_paths, errors = schema.validate_config_schema_versions(config_dir)

    assert errors == []
    assert {path.name for path in checked_paths} == {"base.json", "colab.json"}


def test_validate_config_schema_versions_reports_missing_and_mismatched_versions(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "base.json", {"training": {}, "router": {}, "ood": {}})
    _write_json(
        config_dir / "colab.json",
        {"config_schema_version": CURRENT_CONFIG_SCHEMA_VERSION - 1, "colab": {}, "training": {}},
    )

    checked_paths, errors = schema.validate_config_schema_versions(config_dir)

    assert {path.name for path in checked_paths} == {"base.json", "colab.json"}
    assert f"base.json: missing config_schema_version; expected {CURRENT_CONFIG_SCHEMA_VERSION}." in errors
    assert (
        f"colab.json: config_schema_version={CURRENT_CONFIG_SCHEMA_VERSION - 1}; "
        f"expected {CURRENT_CONFIG_SCHEMA_VERSION}."
    ) in errors


def test_validate_config_schema_versions_reports_non_integer_version(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "base.json", {"config_schema_version": "v1", "training": {}, "router": {}, "ood": {}})

    _, errors = schema.validate_config_schema_versions(config_dir)

    assert errors == ["base.json: config_schema_version must be integer-compatible, got 'v1'."]
