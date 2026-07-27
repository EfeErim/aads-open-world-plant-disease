from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_turkey_adapter_ood_catalog.py"
SPEC = importlib.util.spec_from_file_location("build_turkey_adapter_ood_catalog", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_catalog_has_eight_targets_and_ten_ranked_diseases_each() -> None:
    payload = MODULE.load_catalog(REPO_ROOT / "docs/research/turkey_adapter_disease_catalog.json")

    assert len(payload["targets"]) == 8
    assert all(len(rows) == 10 for rows in payload["targets"].values())
    assert all([row["rank"] for row in rows] == list(range(1, 11)) for rows in payload["targets"].values())


def test_collection_plan_only_contains_confirmed_ood_candidates(tmp_path: Path) -> None:
    payload = MODULE.load_catalog(REPO_ROOT / "docs/research/turkey_adapter_disease_catalog.json")
    output = tmp_path / "plan.csv"

    MODULE.write_collection_plan(payload, output)

    text = output.read_text(encoding="utf-8-sig")
    assert "out_of_distribution" not in text
    assert "tomato_bacterial_canker" in text
    assert "grape_bacterial_crown_gall" not in text
    assert "tomato_buckeye_rot" not in text
    assert "oe_train" in text and "ood_dev" in text and "ood_test" in text
    assert "minimum_target_count,good_target_count,strong_target_count" in text
    assert MODULE.COVERAGE_TIERS["minimum"] == {"oe_train": 10, "ood_dev": 5, "ood_test": 5}
    assert MODULE.COVERAGE_TIERS["good"] == {"oe_train": 20, "ood_dev": 10, "ood_test": 10}
    assert MODULE.COVERAGE_TIERS["strong"] == {"oe_train": 30, "ood_dev": 15, "ood_test": 15}
    assert MODULE.ACTIVE_RANK_LIMIT == 8
