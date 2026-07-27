#!/usr/bin/env python3
"""Apply staged internet image placement manifests into runtime datasets.

The repo already stores reviewed internet-image candidate manifests under
``data/internet_image_candidates/<run>/``. This utility turns those manifests
into real files under ``data/prepared_runtime_datasets/<dataset_key>/ood`` or
``oe`` while enforcing the repo policy that OOD evidence and OE training
inputs stay disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_MANIFEST_SUFFIX = "_placement_manifest.json"
SUPPORTED_TARGET_SPLITS = {"ood", "oe"}


@dataclass(frozen=True)
class CandidatePlacement:
    manifest_path: Path
    source_path: Path
    target_path: Path
    sha256: str
    dataset_key: str
    target_split: str


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(path_value: str, repo_root: Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _extract_dataset_key(target_path: Path) -> str:
    parts = list(target_path.parts)
    if "prepared_runtime_datasets" in parts:
        index = parts.index("prepared_runtime_datasets")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "unknown"


def _extract_target_split(target_path: Path) -> str:
    for split in SUPPORTED_TARGET_SPLITS:
        if split in target_path.parts:
            return split
    return "unknown"


def discover_placement_manifests(manifest_root: Path) -> List[Path]:
    manifests: List[Path] = []
    for path in sorted(manifest_root.rglob(f"*{IMAGE_MANIFEST_SUFFIX}")):
        if path.is_file():
            manifests.append(path)
    return manifests


def _records_from_manifest(manifest_path: Path, repo_root: Path) -> List[CandidatePlacement]:
    payload = _load_json(manifest_path)
    placements = payload.get("placements") if isinstance(payload, dict) else None
    if not isinstance(placements, list):
        return []

    records: List[CandidatePlacement] = []
    for entry in placements:
        if not isinstance(entry, dict):
            continue
        source_raw = entry.get("candidate_path") or entry.get("source_path")
        target_raw = entry.get("target_path")
        if not source_raw or not target_raw:
            continue
        source_path = _resolve_path(str(source_raw), repo_root)
        target_path = _resolve_path(str(target_raw), repo_root)
        sha256 = str(entry.get("sha256") or "").strip()
        if not sha256 and source_path.is_file():
            sha256 = _sha256(source_path)
        records.append(
            CandidatePlacement(
                manifest_path=manifest_path,
                source_path=source_path,
                target_path=target_path,
                sha256=sha256,
                dataset_key=_extract_dataset_key(target_path),
                target_split=_extract_target_split(target_path),
            )
        )
    return records


def _candidate_resolution_roots(manifest_path: Path, repo_root: Path) -> List[Path]:
    manifest_dir = manifest_path.parent.resolve()
    roots = [repo_root.resolve(), manifest_dir]
    manifest_run_root = manifest_dir.parent
    if manifest_run_root not in roots:
        roots.append(manifest_run_root)
    return roots


def _resolve_existing_candidate_path(path_value: str, manifest_path: Path, repo_root: Path) -> Optional[Path]:
    candidate = Path(str(path_value).replace("\\", "/"))
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    for base_root in _candidate_resolution_roots(manifest_path, repo_root):
        resolved = (base_root / candidate).resolve()
        if resolved.exists():
            return resolved
    return None


def collect_candidate_placements(manifest_root: Path, repo_root: Path) -> List[CandidatePlacement]:
    placements: List[CandidatePlacement] = []
    for manifest_path in discover_placement_manifests(manifest_root):
        placements.extend(_records_from_manifest(manifest_path, repo_root))
    return placements


def _validate_disjoint_ood_oe(placements: Iterable[CandidatePlacement]) -> None:
    by_dataset_and_hash: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    for placement in placements:
        if placement.target_split not in SUPPORTED_TARGET_SPLITS:
            continue
        if not placement.sha256:
            continue
        by_dataset_and_hash[(placement.dataset_key, placement.sha256)].add(placement.target_split)

    conflicts = [
        (dataset_key, sha256, sorted(splits))
        for (dataset_key, sha256), splits in by_dataset_and_hash.items()
        if len(splits) > 1
    ]
    if conflicts:
        lines = ["The same file hash was routed to both OOD and OE for at least one dataset:"]
        for dataset_key, sha256, splits in conflicts:
            lines.append(f"- {dataset_key}: {sha256} -> {', '.join(splits)}")
        raise RuntimeError("\n".join(lines))


def apply_candidate_placements(
    manifest_root: Path,
    *,
    repo_root: Path,
    dry_run: bool = False,
    force: bool = False,
    skip_missing_sources: bool = True,
) -> Dict[str, Any]:
    manifest_root = Path(manifest_root).resolve()
    repo_root = Path(repo_root).resolve()
    if not manifest_root.exists():
        raise FileNotFoundError(f"Manifest root not found: {manifest_root}")
    if not manifest_root.is_dir():
        raise NotADirectoryError(f"Manifest root is not a directory: {manifest_root}")

    placements = collect_candidate_placements(manifest_root, repo_root)
    if not placements:
        return {
            "manifest_root": str(manifest_root),
            "repo_root": str(repo_root),
            "placement_count": 0,
            "copied_count": 0,
            "skipped_existing_count": 0,
            "dry_run": bool(dry_run),
            "manifests": [],
        }

    _validate_disjoint_ood_oe(placements)

    copied_count = 0
    skipped_existing_count = 0
    skipped_missing_source_count = 0
    by_manifest: Dict[str, int] = defaultdict(int)
    missing_sources: List[str] = []
    actions: List[Dict[str, Any]] = []

    for placement in placements:
        by_manifest[str(placement.manifest_path)] += 1
        resolved_source_path = placement.source_path
        if not resolved_source_path.is_file():
            fallback = _resolve_existing_candidate_path(str(placement.source_path), placement.manifest_path, repo_root)
            if fallback is not None:
                resolved_source_path = fallback
            elif skip_missing_sources:
                skipped_missing_source_count += 1
                missing_sources.append(str(placement.source_path))
                actions.append(
                    {
                        "manifest_path": str(placement.manifest_path),
                        "source_path": str(placement.source_path),
                        "target_path": str(placement.target_path),
                        "dataset_key": placement.dataset_key,
                        "target_split": placement.target_split,
                        "sha256": placement.sha256,
                        "action": "skipped_missing_source",
                    }
                )
                continue
            raise FileNotFoundError(f"Candidate file not found: {placement.source_path}")

        source_hash = _sha256(resolved_source_path)
        if placement.sha256 and source_hash != placement.sha256:
            raise RuntimeError(
                "SHA-256 mismatch for candidate file: "
                f"{resolved_source_path} (manifest={placement.sha256}, actual={source_hash})"
            )

        target_path = placement.target_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        action = "copied"
        if target_path.exists():
            if _sha256(target_path) == source_hash:
                action = "skipped_existing"
                skipped_existing_count += 1
            elif not force:
                raise RuntimeError(f"Target already exists with different content: {target_path}")

        if action == "copied":
            if not dry_run:
                shutil.copy2(resolved_source_path, target_path)
            copied_count += 1

        actions.append(
            {
                "manifest_path": str(placement.manifest_path),
                "source_path": str(resolved_source_path),
                "target_path": str(target_path),
                "dataset_key": placement.dataset_key,
                "target_split": placement.target_split,
                "sha256": source_hash,
                "action": action,
            }
        )

    summary: Dict[str, Any] = {
        "manifest_root": str(manifest_root),
        "repo_root": str(repo_root),
        "placement_count": len(placements),
        "copied_count": int(copied_count),
        "skipped_existing_count": int(skipped_existing_count),
        "skipped_missing_source_count": int(skipped_missing_source_count),
        "dry_run": bool(dry_run),
        "missing_sources": missing_sources,
        "manifests": [
            {"path": manifest_path, "count": int(count)} for manifest_path, count in sorted(by_manifest.items())
        ],
        "actions": actions,
    }
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data") / "internet_image_candidates",
        help="Root folder containing staged placement manifests.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root used to resolve relative source and target paths.",
    )
    parser.add_argument("--summary-out", type=Path, default=Path("outputs") / "internet_candidate_apply_summary.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without copying files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing targets if content differs.")
    parser.add_argument(
        "--fail-on-missing-source",
        action="store_true",
        help="Abort instead of skipping manifest entries whose source files are unavailable locally.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    summary = apply_candidate_placements(
        args.manifest_root,
        repo_root=args.repo_root,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        skip_missing_sources=not bool(args.fail_on_missing_source),
    )
    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"[APPLY] placement_count={summary['placement_count']}")
    print(f"[APPLY] copied_count={summary['copied_count']}")
    print(f"[APPLY] skipped_existing_count={summary['skipped_existing_count']}")
    print(f"[APPLY] skipped_missing_source_count={summary['skipped_missing_source_count']}")
    print(f"[APPLY] summary_out={summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())