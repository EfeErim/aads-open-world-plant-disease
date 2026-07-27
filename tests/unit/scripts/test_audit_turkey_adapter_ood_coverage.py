from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_turkey_adapter_ood_coverage.py"
SPEC = importlib.util.spec_from_file_location("audit_turkey_adapter_ood_coverage", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_path_matching_is_disease_specific() -> None:
    assert MODULE.path_matches("grape_specific_unknowns/grape_black_rot_01.jpg", "grape_black_rot")
    assert not MODULE.path_matches("grape_specific_unknowns/grape_black_rot_01.jpg", "grape_phomopsis")


def test_path_matching_rejects_explicit_opposite_part() -> None:
    leaf_image = "data/prepared_runtime_datasets/tomato__fruit/oe/unsupported_leaf/early_blight_leaf.jpg"
    fruit_image = "data/prepared_runtime_datasets/strawberry__leaf/oe/anthracnose_fruit_rot.jpg"

    assert not MODULE.path_matches(leaf_image, "tomato_alternaria_fruit_rot", "tomato__fruit")
    assert not MODULE.path_matches(fruit_image, "strawberry_anthracnose", "strawberry__leaf")


def test_audit_reports_every_confirmed_ood_disease() -> None:
    catalog_module = sys.modules["build_turkey_adapter_ood_catalog"]
    payload = catalog_module.load_catalog(REPO_ROOT / "docs/research/turkey_adapter_disease_catalog.json")

    rows = MODULE.audit(payload)

    assert len(rows) == 27
    assert all(row["target"] and row["disease_id"] for row in rows)
    assert all(row["coverage_tier"] in {"below_minimum", "minimum", "good", "strong"} for row in rows)
    assert all(row["coverage_complete"] == row["minimum_coverage_complete"] for row in rows)


def test_coverage_tiers_use_minimum_good_and_strong_thresholds(monkeypatch) -> None:
    payload = {
        "targets": {
            "sample__leaf": [
                {
                    "rank": 1,
                    "status": "out_of_distribution",
                    "disease_id": "sample_disease",
                    "name_tr": "Örnek",
                }
            ]
        }
    }
    monkeypatch.setattr(MODULE, "materialized_paths", lambda _root: [])
    monkeypatch.setattr(
        MODULE,
        "staged_counts",
        lambda _root: {
            ("sample__leaf", "sample_disease", "oe_train"): 20,
            ("sample__leaf", "sample_disease", "ood_dev"): 10,
            ("sample__leaf", "sample_disease", "ood_test"): 10,
        },
    )

    row = MODULE.audit(payload, staged_manifest_root=Path("unused"))[0]

    assert row["coverage_tier"] == "good"
    assert row["minimum_coverage_complete"] is True
    assert row["good_coverage_complete"] is True
    assert row["strong_coverage_complete"] is False
    assert row["coverage_complete"] is True
