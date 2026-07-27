from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

from PIL import Image


def load_module():
    path = Path(__file__).parents[3] / "scripts" / "derive_good_tier_crops.py"
    spec = importlib.util.spec_from_file_location("derive_good_tier_crops", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_derive_crops_preserves_parent_source_urls(tmp_path: Path) -> None:
    module = load_module()
    evidence = tmp_path / "evidence"
    source_dir = evidence / "crop__leaf" / "ood_dev" / "disease"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (100, 80), "red").save(source_dir / "one.jpg")
    Image.new("RGB", (120, 90), "blue").save(source_dir / "two.jpg")
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    with (manifests / "sources_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "source_url"])
        writer.writeheader()
        writer.writerows(
            [
                {"relative_path": "crop__leaf/ood_dev/disease/one.jpg", "source_url": "https://one.test"},
                {"relative_path": "crop__leaf/ood_dev/disease/two.jpg", "source_url": "https://two.test"},
            ]
        )
    candidate_manifest = tmp_path / "derived_candidates.csv"
    rows = module.derive_crops(
        evidence_root=evidence,
        manifest_root=manifests,
        output_dir=tmp_path / "derived",
        candidate_manifest=candidate_manifest,
        target="crop__leaf",
        disease_id="disease",
        role="ood_dev",
        count=3,
    )
    assert len(rows) == 3
    assert [row["source_page_url"] for row in rows] == ["https://one.test", "https://two.test", "https://one.test"]
    assert len(list((tmp_path / "derived").glob("*.jpg"))) == 3
