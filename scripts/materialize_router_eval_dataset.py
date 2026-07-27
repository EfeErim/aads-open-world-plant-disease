#!/usr/bin/env python3
"""Materialize deterministic router eval and holdout datasets from local pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOCAL_ID_DATASETS = {
    "apricot__fruit",
    "apricot__leaf",
    "grape__fruit",
    "grape__leaf",
    "strawberry__leaf",
    "tomato__fruit",
    "tomato__leaf",
}
SUPPORTED_LOCAL_CROPS = {"apricot", "grape", "strawberry", "tomato"}
ID_QUOTA_PER_CROP_PART = 40
NEGATIVE_QUOTAS = {
    "off_crop": 120,
    "non_plant": 80,
    "ambiguous": 60,
}
WRONG_PART_QUOTA_PER_CROP = 30


@dataclass(frozen=True)
class Candidate:
    source_path: Path
    source_split: str
    group: str
    crop: str
    expected_part: str
    label: str
    selection_reason: str
    source_dataset: str = ""
    source_pool: str = ""
    source_slice: str = ""
    unsupported_part: str = ""


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _slug(value: str) -> str:
    normalized = []
    for char in str(value).strip().lower():
        if char.isalnum():
            normalized.append(char)
        elif normalized and normalized[-1] != "_":
            normalized.append("_")
    return "".join(normalized).strip("_") or "unknown"


def _stable_shuffle(items: list[Candidate], *, seed: int, key: str) -> list[Candidate]:
    rng = random.Random(f"{seed}:{key}")
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled


def _balanced_select(candidates: Iterable[Candidate], *, limit: int, seed: int) -> list[Candidate]:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        source_group = candidate.source_path.parent.as_posix()
        if candidate.source_pool:
            source_group = f"{candidate.source_pool}/{candidate.source_slice}"
        groups[source_group].append(candidate)

    queues: dict[str, list[Candidate]] = {
        group: _stable_shuffle(sorted(items, key=lambda item: item.source_path.as_posix()), seed=seed, key=group)
        for group, items in sorted(groups.items())
    }
    selected: list[Candidate] = []
    while len(selected) < limit and any(queues.values()):
        for group in sorted(queues):
            if len(selected) >= limit:
                break
            if queues[group]:
                selected.append(queues[group].pop(0))
    return selected


def _iter_id_candidates(runtime_root: Path, *, split: str) -> list[Candidate]:
    rows: list[Candidate] = []
    for dataset_root in sorted(path for path in runtime_root.iterdir() if path.is_dir()):
        dataset_name = dataset_root.name
        if dataset_name not in LOCAL_ID_DATASETS or "__" not in dataset_name:
            continue
        crop, part = dataset_name.split("__", 1)
        split_root = dataset_root / split
        if not split_root.is_dir():
            continue
        for image_path in sorted(path for path in split_root.rglob("*") if _is_image(path)):
            rows.append(
                Candidate(
                    source_path=image_path,
                    source_split=split,
                    group="id",
                    crop=crop,
                    expected_part=part,
                    label=f"{crop}__{part}",
                    source_dataset=dataset_name,
                    selection_reason=f"ID {split} sample from prepared runtime dataset {dataset_name}",
                )
            )
    return rows


def _parse_ood_pool(pool_name: str) -> tuple[str, str]:
    dataset_key = pool_name.removesuffix("_ood_final")
    if "__" not in dataset_key:
        return ("unknown", "unknown")
    crop, part = dataset_key.split("__", 1)
    return (_slug(crop), _slug(part))


def _classify_ood_slice(*, source_crop: str, source_part: str, slice_name: str) -> tuple[str, str, str, str]:
    normalized = _slug(slice_name)
    if "non_plant" in normalized:
        return ("non_plant", "unknown", "unknown", "clearly non-plant OOD slice")
    if any(token in normalized for token in ("scene_context", "blur_or_occlusion", "failure_cases", "off_coverage")):
        return ("ambiguous", "unknown", "unknown", "ambiguous/context-heavy OOD slice")
    if any(token in normalized for token in ("unsupported", "root", "specific_unknowns")):
        unsupported_part = "root_or_unknown" if "root" in normalized else "unsupported_unknown"
        return ("wrong_part", source_crop, unsupported_part, "same-crop unsupported or wrong-part pressure slice")
    if any(token in normalized for token in ("off_crop", "other_crop")):
        return ("off_crop", "unknown", "unknown", "off-crop false-accept pressure slice")
    return ("ambiguous", "unknown", "unknown", "unmapped OOD slice kept as ambiguous pressure")


def _iter_ood_candidates(ood_root: Path) -> list[Candidate]:
    rows: list[Candidate] = []
    if not ood_root.is_dir():
        return rows
    for pool_root in sorted(path for path in ood_root.iterdir() if path.is_dir()):
        source_crop, source_part = _parse_ood_pool(pool_root.name)
        if source_crop not in SUPPORTED_LOCAL_CROPS:
            continue
        for slice_root in sorted(path for path in pool_root.iterdir() if path.is_dir()):
            group, expected_crop, unsupported_part, reason = _classify_ood_slice(
                source_crop=source_crop,
                source_part=source_part,
                slice_name=slice_root.name,
            )
            label = f"{pool_root.name}__{_slug(slice_root.name)}"
            for image_path in sorted(path for path in slice_root.rglob("*") if _is_image(path)):
                rows.append(
                    Candidate(
                        source_path=image_path,
                        source_split="ood",
                        group=group,
                        crop=expected_crop,
                        expected_part="unknown",
                        label=label,
                        source_pool=pool_root.name,
                        source_slice=slice_root.name,
                        unsupported_part=unsupported_part,
                        selection_reason=reason,
                    )
                )
    return rows


def _destination_for(candidate: Candidate, *, root: Path, digest: str) -> Path:
    suffix = candidate.source_path.suffix.lower()
    stem = _slug(candidate.source_path.stem)[:80]
    filename = f"{stem}_{digest[:12]}{suffix}"
    if candidate.group == "id":
        return root / "id" / candidate.crop / candidate.expected_part / filename
    if candidate.group in {"off_crop", "non_plant"}:
        return root / "negatives" / candidate.group / _slug(candidate.label) / filename
    if candidate.group == "ambiguous":
        return root / "ambiguous" / _slug(candidate.label) / filename
    if candidate.group == "wrong_part":
        return root / "wrong_part" / candidate.crop / _slug(candidate.unsupported_part) / filename
    raise ValueError(f"Unsupported candidate group: {candidate.group}")


def _copy_candidates(
    candidates: Iterable[Candidate],
    *,
    root: Path,
    repo_root: Path,
    used_hashes: set[str],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for candidate in candidates:
        digest = _sha256(candidate.source_path)
        if digest in used_hashes:
            continue
        destination = _destination_for(candidate, root=root, digest=digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate.source_path, destination)
        used_hashes.add(digest)
        entries.append(
            {
                "source_path": _repo_relative(candidate.source_path, repo_root),
                "destination_path": _repo_relative(destination, repo_root),
                "source_split": candidate.source_split,
                "source_dataset": candidate.source_dataset,
                "source_pool": candidate.source_pool,
                "source_slice": candidate.source_slice,
                "group": candidate.group,
                "crop": candidate.crop,
                "expected_crop": candidate.crop if candidate.group in {"id", "wrong_part"} else "unknown",
                "expected_part": candidate.expected_part,
                "unsupported_part": candidate.unsupported_part,
                "label": candidate.label,
                "sha256": digest,
                "selection_reason": candidate.selection_reason,
            }
        )
    return entries


def _select_id(candidates: list[Candidate], *, seed: int) -> list[Candidate]:
    by_crop_part: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_crop_part[(candidate.crop, candidate.expected_part)].append(candidate)
    selected: list[Candidate] = []
    for (crop, part), rows in sorted(by_crop_part.items()):
        selected.extend(
            _balanced_select(
                rows,
                limit=ID_QUOTA_PER_CROP_PART,
                seed=seed,
            )
        )
    return selected


def _select_negatives(candidates: list[Candidate], *, seed: int) -> list[Candidate]:
    selected: list[Candidate] = []
    for group, quota in NEGATIVE_QUOTAS.items():
        selected.extend(
            _balanced_select(
                [candidate for candidate in candidates if candidate.group == group],
                limit=quota,
                seed=seed,
            )
        )
    wrong_by_crop: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.group == "wrong_part":
            wrong_by_crop[candidate.crop].append(candidate)
    for crop, rows in sorted(wrong_by_crop.items()):
        selected.extend(_balanced_select(rows, limit=WRONG_PART_QUOTA_PER_CROP, seed=seed))
    return selected


def _split_selected(rows: list[Candidate], *, quota: int) -> tuple[list[Candidate], list[Candidate]]:
    dev_limit = min(int(quota), (len(rows) + 1) // 2)
    holdout_limit = min(int(quota), max(0, len(rows) - dev_limit))
    return rows[:dev_limit], rows[dev_limit : dev_limit + holdout_limit]


def _select_negative_splits(candidates: list[Candidate], *, seed: int) -> tuple[list[Candidate], list[Candidate]]:
    dev_selected: list[Candidate] = []
    holdout_selected: list[Candidate] = []
    for group, quota in NEGATIVE_QUOTAS.items():
        rows = _balanced_select(
            [candidate for candidate in candidates if candidate.group == group],
            limit=quota * 2,
            seed=seed,
        )
        dev_rows, holdout_rows = _split_selected(rows, quota=quota)
        dev_selected.extend(dev_rows)
        holdout_selected.extend(holdout_rows)

    wrong_by_crop: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.group == "wrong_part":
            wrong_by_crop[candidate.crop].append(candidate)
    for crop, crop_rows in sorted(wrong_by_crop.items()):
        rows = _balanced_select(crop_rows, limit=WRONG_PART_QUOTA_PER_CROP * 2, seed=seed)
        dev_rows, holdout_rows = _split_selected(rows, quota=WRONG_PART_QUOTA_PER_CROP)
        dev_selected.extend(dev_rows)
        holdout_selected.extend(holdout_rows)
    return dev_selected, holdout_selected


def _write_manifest(
    root: Path,
    *,
    repo_root: Path,
    entries: list[dict[str, Any]],
    seed: int,
    source_split: str,
) -> None:
    counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        counts[str(entry["group"])] += 1
    hash_counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        hash_counts[str(entry["sha256"])] += 1
    duplicate_hashes = sorted(digest for digest, count in hash_counts.items() if count > 1)
    payload = {
        "schema_version": "v1_router_eval_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": _repo_relative(root, repo_root),
        "source_split": source_split,
        "seed": seed,
        "quotas": {
            "id_per_crop_part": ID_QUOTA_PER_CROP_PART,
            **NEGATIVE_QUOTAS,
            "wrong_part_per_crop": WRONG_PART_QUOTA_PER_CROP,
        },
        "summary": {
            "image_count": len(entries),
            "counts_by_group": dict(sorted(counts.items())),
            "duplicate_sha256_count": len(duplicate_hashes),
            "duplicate_sha256": duplicate_hashes,
        },
        "entries": entries,
    }
    (root / "router_eval_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prepare_root(root: Path, *, force: bool) -> None:
    if root.exists() and any(root.iterdir()):
        if not force:
            raise RuntimeError(f"{root} already exists and is not empty. Pass --force to rebuild it.")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def materialize_router_eval_datasets(
    *,
    repo_root: Path,
    runtime_root: Path,
    ood_root: Path,
    dev_root: Path,
    holdout_root: Path,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    _prepare_root(dev_root, force=force)
    _prepare_root(holdout_root, force=force)

    ood_candidates = _iter_ood_candidates(ood_root)
    dev_negative_candidates, holdout_negative_candidates = _select_negative_splits(ood_candidates, seed=seed)
    dev_used_hashes: set[str] = set()

    dev_entries = _copy_candidates(
        [
            *_select_id(_iter_id_candidates(runtime_root, split="val"), seed=seed),
            *dev_negative_candidates,
        ],
        root=dev_root,
        repo_root=repo_root,
        used_hashes=dev_used_hashes,
    )
    holdout_used_hashes: set[str] = set(dev_used_hashes)
    holdout_entries = _copy_candidates(
        [
            *_select_id(_iter_id_candidates(runtime_root, split="test"), seed=seed + 1),
            *holdout_negative_candidates,
        ],
        root=holdout_root,
        repo_root=repo_root,
        used_hashes=holdout_used_hashes,
    )

    _write_manifest(dev_root, repo_root=repo_root, entries=dev_entries, seed=seed, source_split="val")
    _write_manifest(
        holdout_root,
        repo_root=repo_root,
        entries=holdout_entries,
        seed=seed + 1,
        source_split="test",
    )
    return {
        "dev_root": str(dev_root),
        "holdout_root": str(holdout_root),
        "dev_count": len(dev_entries),
        "holdout_count": len(holdout_entries),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=Path("data/prepared_runtime_datasets"))
    parser.add_argument("--ood-root", type=Path, default=Path("data/ood_dataset/final"))
    parser.add_argument("--dev-root", type=Path, default=Path("data/router_eval"))
    parser.add_argument("--holdout-root", type=Path, default=Path("data/router_eval_holdout"))
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--force", action="store_true", help="Delete and rebuild existing eval roots.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    result = materialize_router_eval_datasets(
        repo_root=repo_root,
        runtime_root=args.runtime_root,
        ood_root=args.ood_root,
        dev_root=args.dev_root,
        holdout_root=args.holdout_root,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
