from pathlib import Path

import pytest

from src.core import config_manager
from src.core.config_manager import DEFAULT_CONFIG_DIR, ConfigurationManager, get_config

REPO_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def test_default_config_dir_is_the_repository_config_directory():
    assert DEFAULT_CONFIG_DIR == REPO_CONFIG_DIR
    assert (DEFAULT_CONFIG_DIR / "base.json").is_file()


def test_default_config_loads_from_any_working_directory(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_manager, "_CONFIG_CACHE", {})

    from_elsewhere = get_config(environment="colab")

    expected = ConfigurationManager(config_dir=REPO_CONFIG_DIR, environment="colab").load_all_configs()
    assert from_elsewhere == expected


def test_explicit_relative_config_dir_still_resolves_from_the_working_directory(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    manager = ConfigurationManager(config_dir="config")

    assert manager.config_dir == Path("config")
    # Nothing is read from the repository: a missing base.json loads as {} and fails the schema-version check.
    with pytest.raises(ValueError, match="config_schema_version"):
        manager.load_base_config()
