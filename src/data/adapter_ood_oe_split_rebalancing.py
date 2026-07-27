from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROLE_SPLITS = ("continual", "val", "test")


@dataclass(frozen=True)
class FamilyMove:
    disease_id: str
    family_key: str
    source_split: str
    destination_split: str
    row_count: int
    reason: str


def _family_key(row: dict[str, Any]) -> str:
    return str(row.get("family_bundle_key") or row.get("family_id") or row.get("relative_path") or "").casefold()


def _active_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not bool(row.get("runtime_skipped")) and str(row.get("split") or "") in ROLE_SPLITS
    ]


def _families(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _active_rows(rows):
        grouped[(str(row.get("normalized_class_name") or ""), _family_key(row))].append(row)
    return dict(grouped)


def _assignments(families: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for key, family_rows in families.items():
        splits = Counter(str(row.get("split") or "") for row in family_rows)
        result[key] = sorted(splits, key=lambda split: (-splits[split], ROLE_SPLITS.index(split)))[0]
    return result


def _family_counts(
    assignments: dict[tuple[str, str], str],
    disease_id: str,
    supplemental_counts: dict[str, dict[str, int]],
) -> Counter[str]:
    counts = Counter(split for (candidate, _), split in assignments.items() if candidate == disease_id)
    counts.update(supplemental_counts.get(disease_id, {}))
    return counts


def plan_runtime_id_rebalance(
    rows: list[dict[str, Any]],
    *,
    min_train_families: int = 100,
    min_val_families: int = 15,
    min_test_families: int = 15,
    supplemental_counts: dict[str, dict[str, int]] | None = None,
) -> tuple[list[FamilyMove], dict[str, dict[str, int]]]:
    families = _families(rows)
    assignments = _assignments(families)
    supplements = supplemental_counts or {}
    moves: list[FamilyMove] = []

    for key, family_rows in sorted(families.items()):
        observed = {str(row.get("split") or "") for row in family_rows}
        if len(observed) <= 1:
            continue
        disease_id, family_key = key
        destination = assignments[key]
        for source in sorted(observed - {destination}):
            moves.append(
                FamilyMove(disease_id, family_key, source, destination, len(family_rows), "family_role_overlap")
            )

    floors = {"continual": min_train_families, "val": min_val_families, "test": min_test_families}
    disease_ids = sorted({disease_id for disease_id, _ in families})
    for disease_id in disease_ids:
        for destination in ("val", "test", "continual"):
            while True:
                counts = _family_counts(assignments, disease_id, supplements)
                if counts[destination] >= floors[destination]:
                    break
                donors = sorted(
                    (split for split in ROLE_SPLITS if counts[split] > floors[split]),
                    key=lambda split: (-(counts[split] - floors[split]), ROLE_SPLITS.index(split)),
                )
                candidates: list[tuple[int, str, tuple[str, str], str]] = []
                for donor in donors:
                    for key, assigned in assignments.items():
                        if key[0] != disease_id or assigned != donor:
                            continue
                        candidates.append((len(families[key]), key[1], key, donor))
                if not candidates:
                    break
                _, family_key, key, donor = sorted(candidates)[0]
                assignments[key] = destination
                moves.append(
                    FamilyMove(
                        disease_id,
                        family_key,
                        donor,
                        destination,
                        len(families[key]),
                        "minimum_family_floor",
                    )
                )

    final_counts = {
        disease_id: {split: _family_counts(assignments, disease_id, supplements)[split] for split in ROLE_SPLITS}
        for disease_id in disease_ids
    }
    return moves, final_counts


def _destination_path(dataset_root: Path, row: dict[str, Any], destination_split: str) -> Path:
    runtime_path = Path(str(row.get("runtime_relative_path") or ""))
    if len(runtime_path.parts) < 2:
        raise ValueError(f"Invalid runtime_relative_path: {runtime_path}")
    return dataset_root / destination_split / Path(*runtime_path.parts[1:])


def rebalance_runtime_id_splits(
    dataset_root: str | Path,
    *,
    dry_run: bool = True,
    min_train_families: int = 100,
    min_val_families: int = 15,
    min_test_families: int = 15,
    supplemental_counts: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root)
    manifest_path = root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = list(manifest.get("rows") or [])
    moves, final_counts = plan_runtime_id_rebalance(
        rows,
        min_train_families=min_train_families,
        min_val_families=min_val_families,
        min_test_families=min_test_families,
        supplemental_counts=supplemental_counts,
    )

    move_by_family: dict[tuple[str, str], FamilyMove] = {}
    for move in moves:
        move_by_family[(move.disease_id, move.family_key)] = move

    file_moves: list[tuple[Path, Path, dict[str, Any], str]] = []
    reserved: set[Path] = set()
    for row in _active_rows(rows):
        key = (str(row.get("normalized_class_name") or ""), _family_key(row))
        move = move_by_family.get(key)
        if move is None or str(row.get("split") or "") == move.destination_split:
            continue
        source = root / Path(str(row.get("runtime_relative_path") or ""))
        destination = _destination_path(root, row, move.destination_split)
        if destination in reserved or (destination.exists() and destination != source):
            family_suffix = hashlib.sha256(move.family_key.encode("utf-8")).hexdigest()[:8]
            destination = destination.with_name(f"{destination.stem}__family_{family_suffix}{destination.suffix}")
        if not source.is_file():
            raise FileNotFoundError(f"Runtime split source is missing: {source}")
        reserved.add(destination)
        file_moves.append((source, destination, row, move.destination_split))

    if not dry_run:
        for source, destination, row, destination_split in file_moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            row["split"] = destination_split
            row["family_assignment"] = destination_split
            row["runtime_relative_path"] = destination.relative_to(root).as_posix()
        manifest["rows"] = rows
        manifest["adapter_ood_oe_rebalance"] = {
            "policy": "family_coherent_minimum_floors",
            "min_train_families": min_train_families,
            "min_val_families": min_val_families,
            "min_test_families": min_test_families,
            "family_move_count": len(move_by_family),
            "file_move_count": len(file_moves),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    unmet = {
        disease_id: {
            split: {"actual": counts[split], "required": required}
            for split, required in (
                ("continual", min_train_families),
                ("val", min_val_families),
                ("test", min_test_families),
            )
            if counts[split] < required
        }
        for disease_id, counts in final_counts.items()
    }
    unmet = {disease_id: deficits for disease_id, deficits in unmet.items() if deficits}
    return {
        "ok": not unmet,
        "dry_run": dry_run,
        "dataset_root": str(root),
        "family_move_count": len(move_by_family),
        "file_move_count": len(file_moves),
        "final_family_counts": final_counts,
        "unmet": unmet,
        "moves": [move.__dict__ for move in moves],
    }
