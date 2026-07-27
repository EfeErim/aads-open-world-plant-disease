"""Taxonomy registry builder for supported router adapter targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

from src.data.dataset_release_runtime import DEFAULT_MATERIALIZED_DATASET_ROOT
from src.router.label_normalization import normalize_part_label
from src.shared.hash_utils import sha256_file
from src.shared.json_utils import read_json, write_json

SUPPORTED_SPLITS = ("train", "val", "test", "continual")
TARGET_SEPARATOR = "__"

DEFAULT_CROP_METADATA: dict[str, dict[str, Any]] = {
    "apricot": {
        "scientific_name": "Prunus armeniaca",
        "common_names": ["apricot"],
        "synonyms": [],
        "genus": "Prunus",
        "family": "Rosaceae",
    },
    "grape": {
        "scientific_name": "Vitis vinifera",
        "common_names": ["grape", "grapevine"],
        "synonyms": [],
        "genus": "Vitis",
        "family": "Vitaceae",
    },
    "strawberry": {
        "scientific_name": "Fragaria x ananassa",
        "common_names": ["strawberry"],
        "synonyms": ["garden strawberry"],
        "genus": "Fragaria",
        "family": "Rosaceae",
    },
    "tomato": {
        "scientific_name": "Solanum lycopersicum",
        "common_names": ["tomato"],
        "synonyms": ["Lycopersicon esculentum"],
        "genus": "Solanum",
        "family": "Solanaceae",
    },
}

HEALTHY_TOKENS = ("healthy", "sa\u011fl\u0131kl\u0131", "saglikli")
GBIF_SPECIES_MATCH_URL = "https://api.gbif.org/v1/species/match"


@dataclass(frozen=True)
class AdapterTarget:
    """Supported crop/part target discovered from local datasets or adapters."""

    target_id: str
    crop: str
    part: str
    dataset_path: str | None = None
    adapter_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaxonomyRegistryEntry:
    """Serializable taxonomy metadata for one supported target."""

    target_id: str
    crop_canonical_name: str
    part: str
    scientific_name: str | None = None
    common_names: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    genus: str | None = None
    family: str | None = None
    class_labels: list[str] = field(default_factory=list)
    supported_disease_labels: list[str] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=dict)
    adapter_paths: list[str] = field(default_factory=list)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    unresolved: bool = False


def now_utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def split_target_id(target_id: str) -> tuple[str, str]:
    if TARGET_SEPARATOR not in target_id:
        raise ValueError(f"Expected target id shaped like crop__part, got {target_id!r}")
    crop, part = target_id.split(TARGET_SEPARATOR, 1)
    crop = normalize_crop_name(crop)
    part = normalize_part_label(part)
    if not crop or not part:
        raise ValueError(f"Invalid target id {target_id!r}")
    return crop, part


def make_target_id(crop: str, part: str) -> str:
    crop_key = normalize_crop_name(crop)
    part_key = normalize_part_label(part).replace(" ", "_")
    if not crop_key or not part_key:
        raise ValueError(f"Cannot build target id from crop={crop!r}, part={part!r}")
    return f"{crop_key}{TARGET_SEPARATOR}{part_key}"


def normalize_crop_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def _is_supported_dataset_target(path: Path) -> bool:
    if not path.is_dir() or TARGET_SEPARATOR not in path.name:
        return False
    return any((path / split_name).is_dir() for split_name in SUPPORTED_SPLITS)


def discover_dataset_targets(dataset_root: Path) -> list[AdapterTarget]:
    if not dataset_root.exists():
        return []

    targets: list[AdapterTarget] = []
    for target_dir in sorted(path for path in dataset_root.iterdir() if _is_supported_dataset_target(path)):
        crop, part = split_target_id(target_dir.name)
        targets.append(
            AdapterTarget(
                target_id=make_target_id(crop, part),
                crop=crop,
                part=part,
                dataset_path=str(target_dir),
            )
        )
    return targets


def discover_adapter_targets(adapter_root: Path) -> list[AdapterTarget]:
    if not adapter_root.exists():
        return []

    targets: dict[str, AdapterTarget] = {}
    for meta_path in sorted(adapter_root.rglob("adapter_meta.json")):
        if meta_path.parent.name != "continual_sd_lora_adapter":
            continue
        crop = part = ""
        payload = read_json(meta_path, default={}, expect_type=dict)
        crop = normalize_crop_name(payload.get("crop_name"))
        part = normalize_part_label(payload.get("part_name"))
        if not crop or not part:
            parts = meta_path.parts
            if len(parts) >= 4:
                crop = crop or normalize_crop_name(parts[-4])
                part = part or normalize_part_label(parts[-3])
        if not crop or not part:
            continue
        target_id = make_target_id(crop, part)
        previous = targets.get(target_id)
        adapter_paths = tuple(sorted((*previous.adapter_paths, str(meta_path.parent)))) if previous else (str(meta_path.parent),)
        targets[target_id] = AdapterTarget(target_id=target_id, crop=crop, part=part, adapter_paths=adapter_paths)
    return [targets[key] for key in sorted(targets)]


def merge_targets(*groups: Iterable[AdapterTarget]) -> list[AdapterTarget]:
    merged: dict[str, AdapterTarget] = {}
    for group in groups:
        for target in group:
            previous = merged.get(target.target_id)
            dataset_path = target.dataset_path or (previous.dataset_path if previous else None)
            adapter_paths = tuple(
                sorted(set((previous.adapter_paths if previous else ()) + target.adapter_paths))
            )
            merged[target.target_id] = AdapterTarget(
                target_id=target.target_id,
                crop=target.crop,
                part=target.part,
                dataset_path=dataset_path,
                adapter_paths=adapter_paths,
            )
    return [merged[key] for key in sorted(merged)]


def load_registry_overrides(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path, default={}, expect_type=dict)
    return dict(payload)


def load_crop_diseases(taxonomy_path: Path | None) -> dict[str, list[str]]:
    if taxonomy_path is None or not taxonomy_path.exists():
        return {}
    payload = read_json(taxonomy_path, default={}, expect_type=dict)
    crop_diseases = payload.get("crop_specific_diseases", {})
    if not isinstance(crop_diseases, dict):
        return {}
    return {
        normalize_crop_name(crop): sorted({str(label).strip() for label in labels if str(label).strip()})
        for crop, labels in crop_diseases.items()
        if isinstance(labels, list)
    }


def load_external_taxonomy_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path, default={}, expect_type=dict)
    crops = payload.get("crops", {})
    return dict(crops) if isinstance(crops, dict) else {}


def _gbif_match_crop(crop: str, *, timeout_sec: float = 10.0) -> dict[str, Any]:
    query = urlencode({"name": crop, "rank": "SPECIES", "kingdom": "Plantae"})
    with urlopen(f"{GBIF_SPECIES_MATCH_URL}?{query}", timeout=timeout_sec) as response:  # noqa: S310
        payload = read_json_from_bytes(response.read())
    if not isinstance(payload, dict) or not payload.get("usageKey"):
        return {}
    canonical_name = str(payload.get("canonicalName") or payload.get("scientificName") or "").strip()
    return {
        "scientific_name": canonical_name or None,
        "genus": str(payload.get("genus") or "").strip() or None,
        "family": str(payload.get("family") or "").strip() or None,
        "source": "gbif_species_match",
        "usage_key": payload.get("usageKey"),
        "confidence": payload.get("confidence"),
        "match_type": payload.get("matchType"),
        "status": payload.get("status"),
    }


def read_json_from_bytes(payload: bytes) -> Any:
    import json

    return json.loads(payload.decode("utf-8"))


def build_external_taxonomy_cache(
    crops: Iterable[str],
    *,
    existing_cache: dict[str, Any] | None = None,
    refresh_gbif: bool = False,
) -> dict[str, Any]:
    cache = dict(existing_cache or {})
    for crop in sorted({normalize_crop_name(crop) for crop in crops if normalize_crop_name(crop)}):
        if crop in cache and not refresh_gbif:
            continue
        try:
            match = _gbif_match_crop(crop)
        except Exception as exc:
            cache[crop] = {**dict(cache.get(crop, {})), "source": "gbif_species_match", "error": str(exc)}
            continue
        if match:
            cache[crop] = {**dict(cache.get(crop, {})), **match, "fetched_at": now_utc_timestamp()}
    return cache


def write_external_taxonomy_cache(path: Path, crops: dict[str, Any]) -> Path:
    payload = {
        "schema_version": "external_taxonomy_cache.v1",
        "updated_at": now_utc_timestamp(),
        "crops": dict(sorted(crops.items())),
    }
    return write_json(path, payload, ensure_ascii=False, sort_keys=False)


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def summarize_dataset_classes(dataset_path: Path | None) -> tuple[list[str], list[str], dict[str, int]]:
    if dataset_path is None or not dataset_path.exists():
        return [], [], {}

    classes: set[str] = set()
    split_counts: dict[str, int] = {}
    for split_name in SUPPORTED_SPLITS:
        split_dir = dataset_path / split_name
        if not split_dir.exists():
            continue
        count = 0
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            classes.add(class_dir.name)
            count += sum(1 for item in class_dir.iterdir() if item.is_file())
        split_counts[split_name] = count

    class_labels = sorted(classes)
    disease_labels = [label for label in class_labels if not _looks_healthy(label)]
    return class_labels, disease_labels, split_counts


def _looks_healthy(label: str) -> bool:
    lower = label.lower()
    return any(token in lower for token in HEALTHY_TOKENS)


def build_taxonomy_registry(
    *,
    dataset_root: Path = DEFAULT_MATERIALIZED_DATASET_ROOT,
    adapter_root: Path | None = Path("runs"),
    taxonomy_path: Path | None = Path("config/plant_taxonomy.json"),
    overrides_path: Path | None = None,
    external_cache_path: Path | None = Path("runs/_index/router_prototypes/external_taxonomy_cache.json"),
    refresh_gbif: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    dataset_targets = discover_dataset_targets(dataset_root)
    adapter_targets = discover_adapter_targets(adapter_root) if adapter_root is not None else []
    targets = merge_targets(dataset_targets, adapter_targets)
    overrides = load_registry_overrides(overrides_path)
    crop_diseases = load_crop_diseases(taxonomy_path)
    external_cache = load_external_taxonomy_cache(external_cache_path)
    if refresh_gbif:
        external_cache = build_external_taxonomy_cache(
            (target.crop for target in targets),
            existing_cache=external_cache,
            refresh_gbif=True,
        )
        if external_cache_path is not None:
            write_external_taxonomy_cache(external_cache_path, external_cache)

    entries: list[TaxonomyRegistryEntry] = []
    for target in targets:
        crop_override = dict(overrides.get("crops", {}).get(target.crop, {})) if isinstance(overrides.get("crops"), dict) else {}
        target_override = (
            dict(overrides.get("targets", {}).get(target.target_id, {})) if isinstance(overrides.get("targets"), dict) else {}
        )
        external_metadata = dict(external_cache.get(target.crop, {})) if isinstance(external_cache, dict) else {}
        metadata = {**DEFAULT_CROP_METADATA.get(target.crop, {}), **external_metadata, **crop_override, **target_override}
        class_labels, disease_labels, split_counts = summarize_dataset_classes(
            Path(target.dataset_path) if target.dataset_path else None
        )
        taxonomy_diseases = crop_diseases.get(target.crop, [])
        if not disease_labels and taxonomy_diseases:
            disease_labels = taxonomy_diseases
        entry = TaxonomyRegistryEntry(
            target_id=target.target_id,
            crop_canonical_name=str(metadata.get("crop_canonical_name") or target.crop),
            part=target.part,
            scientific_name=metadata.get("scientific_name"),
            common_names=sorted({*_coerce_string_list(metadata.get("common_names")), target.crop}),
            synonyms=sorted({*_coerce_string_list(metadata.get("synonyms"))}),
            genus=metadata.get("genus"),
            family=metadata.get("family"),
            class_labels=class_labels,
            supported_disease_labels=sorted(disease_labels),
            split_counts=split_counts,
            adapter_paths=list(target.adapter_paths),
            source_metadata={
                "dataset_path": target.dataset_path,
                "taxonomy_path": str(taxonomy_path) if taxonomy_path else None,
                "overrides_path": str(overrides_path) if overrides_path else None,
                "external_cache_path": str(external_cache_path) if external_cache_path else None,
                "external_taxonomy_source": external_metadata.get("source"),
                "external_usage_key": external_metadata.get("usage_key"),
            },
            unresolved=not bool(metadata.get("scientific_name")),
        )
        entries.append(entry)

    payload = {
        "schema_version": "taxonomy_registry.v1",
        "created_at": created_at or now_utc_timestamp(),
        "source_roots": {
            "dataset_root": str(dataset_root),
            "adapter_root": str(adapter_root) if adapter_root else None,
            "taxonomy_path": str(taxonomy_path) if taxonomy_path else None,
            "overrides_path": str(overrides_path) if overrides_path else None,
            "external_cache_path": str(external_cache_path) if external_cache_path else None,
            "refresh_gbif": bool(refresh_gbif),
        },
        "summary": {
            "target_count": len(entries),
            "unresolved_count": sum(1 for entry in entries if entry.unresolved),
            "targets": [entry.target_id for entry in entries],
        },
        "targets": [asdict(entry) for entry in entries],
    }
    return payload


def write_taxonomy_registry(payload: dict[str, Any], output_path: Path) -> Path:
    return write_json(output_path, payload, ensure_ascii=False, sort_keys=False)


def artifact_sha256(path: Path) -> str:
    return sha256_file(path)
