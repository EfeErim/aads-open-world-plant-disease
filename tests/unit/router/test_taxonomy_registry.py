import json
from pathlib import Path

from src.router.taxonomy_registry import build_taxonomy_registry, discover_dataset_targets


def _write_text(path: Path, value: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_build_taxonomy_registry_discovers_dataset_targets_and_metadata(tmp_path: Path):
    dataset_root = tmp_path / "data" / "prepared_runtime_datasets"
    _write_text(dataset_root / "tomato__leaf" / "train" / "domates_healthy_leaf" / "a.txt")
    _write_text(dataset_root / "tomato__leaf" / "test" / "domates_late_blight_leaf" / "b.txt")
    taxonomy_path = tmp_path / "config" / "plant_taxonomy.json"
    taxonomy_path.parent.mkdir(parents=True)
    taxonomy_path.write_text(
        json.dumps({"crop_specific_diseases": {"tomato": ["late blight"]}}),
        encoding="utf-8",
    )

    payload = build_taxonomy_registry(
        dataset_root=dataset_root,
        adapter_root=None,
        taxonomy_path=taxonomy_path,
        created_at="20260617T000000Z",
    )

    assert payload["schema_version"] == "taxonomy_registry.v1"
    assert payload["summary"]["targets"] == ["tomato__leaf"]
    entry = payload["targets"][0]
    assert entry["target_id"] == "tomato__leaf"
    assert entry["scientific_name"] == "Solanum lycopersicum"
    assert entry["family"] == "Solanaceae"
    assert entry["class_labels"] == ["domates_healthy_leaf", "domates_late_blight_leaf"]
    assert entry["supported_disease_labels"] == ["domates_late_blight_leaf"]
    assert entry["split_counts"] == {"test": 1, "train": 1}
    assert entry["unresolved"] is False


def test_build_taxonomy_registry_marks_unknown_crop_unresolved(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    _write_text(dataset_root / "dragonfruit__fruit" / "train" / "healthy" / "a.txt")

    payload = build_taxonomy_registry(
        dataset_root=dataset_root,
        adapter_root=None,
        taxonomy_path=None,
        created_at="20260617T000000Z",
    )

    assert payload["summary"]["unresolved_count"] == 1
    assert payload["targets"][0]["target_id"] == "dragonfruit__fruit"
    assert payload["targets"][0]["unresolved"] is True


def test_build_taxonomy_registry_uses_external_cache_for_unknown_crop(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    _write_text(dataset_root / "dragonfruit__fruit" / "train" / "healthy" / "a.txt")
    cache_path = tmp_path / "external_taxonomy_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": "external_taxonomy_cache.v1",
                "crops": {
                    "dragonfruit": {
                        "scientific_name": "Selenicereus undatus",
                        "genus": "Selenicereus",
                        "family": "Cactaceae",
                        "source": "gbif_species_match",
                        "usage_key": 123,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    payload = build_taxonomy_registry(
        dataset_root=dataset_root,
        adapter_root=None,
        taxonomy_path=None,
        external_cache_path=cache_path,
        created_at="20260617T000000Z",
    )

    entry = payload["targets"][0]
    assert entry["scientific_name"] == "Selenicereus undatus"
    assert entry["family"] == "Cactaceae"
    assert entry["source_metadata"]["external_taxonomy_source"] == "gbif_species_match"
    assert entry["unresolved"] is False


def test_external_cache_overrides_default_but_explicit_override_wins(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    _write_text(dataset_root / "tomato__leaf" / "train" / "healthy" / "a.txt")
    cache_path = tmp_path / "external_taxonomy_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "crops": {
                    "tomato": {
                        "scientific_name": "GBIF tomato",
                        "genus": "GbifGenus",
                        "family": "GbifFamily",
                        "source": "gbif_species_match",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text(json.dumps({"crops": {"tomato": {"family": "OverrideFamily"}}}), encoding="utf-8")

    payload = build_taxonomy_registry(
        dataset_root=dataset_root,
        adapter_root=None,
        taxonomy_path=None,
        external_cache_path=cache_path,
        overrides_path=overrides_path,
        created_at="20260617T000000Z",
    )

    entry = payload["targets"][0]
    assert entry["scientific_name"] == "GBIF tomato"
    assert entry["genus"] == "GbifGenus"
    assert entry["family"] == "OverrideFamily"


def test_discover_dataset_targets_ignores_non_target_dirs(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    _write_text(dataset_root / "tomato__fruit" / "test" / "healthy" / "a.txt")
    _write_text(dataset_root / "README" / "test" / "healthy" / "a.txt")

    targets = discover_dataset_targets(dataset_root)

    assert [target.target_id for target in targets] == ["tomato__fruit"]
