import json
from pathlib import Path

import pytest

from scripts import validate_notebook_imports as validator


def _write_notebook(path: Path, cells: list[dict]) -> None:
    path.write_text(json.dumps({"cells": cells}), encoding="utf-8")


def test_load_notebook_sources_from_path_collects_code_cells(tmp_path: Path):
    notebook_path = tmp_path / "sample.ipynb"
    _write_notebook(
        notebook_path,
        [
            {"cell_type": "markdown", "source": ["# title"]},
            {"cell_type": "code", "source": ["print('first')\n"]},
            {"cell_type": "code", "source": ["print('second')\n"]},
        ],
    )

    sources = validator._load_notebook_sources_from_path(notebook_path)

    assert sources.notebook_path == notebook_path
    assert sources.first_code_source == "print('first')\n"
    assert sources.code_cells == ("print('first')\n", "print('second')\n")
    assert "print('second')" in sources.full_source


def test_find_code_cell_source_returns_first_match(tmp_path: Path):
    notebook_path = tmp_path / "sample.ipynb"
    _write_notebook(
        notebook_path,
        [
            {"cell_type": "code", "source": ["alpha\n"]},
            {"cell_type": "code", "source": ["beta marker\n"]},
            {"cell_type": "code", "source": ["beta marker again\n"]},
        ],
    )
    sources = validator._load_notebook_sources_from_path(notebook_path)

    matched = validator._find_code_cell_source(sources, "beta marker", "missing")

    assert matched == "beta marker\n"


def test_assert_repo_bootstrap_contract_rejects_drive_bootstrap():
    with pytest.raises(AssertionError):
        validator._assert_repo_bootstrap_contract(
            "\n".join(
                [
                    "CLONE_TARGET = Path('/content/bitirmeprojesi')",
                    "REPO_URL = os.environ.get('AADS_REPO_URL', '')",
                    "['git', 'clone', '--depth', '1', clone_url, str(CLONE_TARGET)]",
                    "mount_drive_if_available()",
                ]
            ),
            "Notebook X",
        )


def test_training_notebook_bootstrap_contract_matches_repo_notebook():
    validator.test_training_notebook_bootstrap_contract()


def test_training_notebook_dataset_contract_matches_repo_notebook():
    validator.test_training_notebook_dataset_contract_detection()


def test_ood_oe_quality_notebook_contract_matches_repo_notebook():
    validator.test_ood_oe_quality_notebook_contract()


def test_adapter_ood_oe_recovery_notebook_contract_matches_repo_notebook():
    validator.test_adapter_ood_oe_recovery_notebook_contract()


def test_private_repo_bootstrap_security_contract_matches_all_notebooks():
    validator.test_private_repo_bootstrap_security_contract()
