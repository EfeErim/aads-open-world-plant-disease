#!/usr/bin/env python3
"""Prepare a cleaned class-root working copy from Notebook 0 audit reports."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from src.data.dataset_layout import IMAGE_EXTENSIONS
from scripts.prepare_grouped_runtime_dataset import (
    DEFAULT_BIOCLIP_MODEL_ID,
    DEFAULT_DINOV3_MODEL_ID,
    DEFAULT_NEIGHBORS,
    build_prepared_dataset_key,
    build_grouped_dataset_plan,
)
from scripts.prune_exact_duplicates import (
    DEFAULT_SEED,
    DuplicateCleanupAction,
    apply_cleanup_plan,
    build_combined_cleanup_plan,
    write_cleanup_report,
)
from src.shared.csv_utils import read_csv_rows
from src.shared.json_utils import write_json


def _progress(progress_fn: Optional[Callable[[str], None]], message: str) -> None:
    if callable(progress_fn):
        progress_fn(str(message))


def _resolve_report_path(artifact_root: Path, filename: str) -> Path:
    candidate = Path(artifact_root) / filename
    if not candidate.is_file():
        raise FileNotFoundError(f"Audit report not found: {candidate}")
    return candidate


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    return read_csv_rows(Path(path))


def _mirror_class_root_dataset(
    *,
    source_root: Path,
    destination_root: Path,
    materialization_strategy: str,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, int]:
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    mirrored_files = 0
    skipped_files = 0
    for source_path in sorted(source_root.rglob("*"), key=lambda item: str(item).lower()):
        if not source_path.is_file():
            continue
        if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
            skipped_files += 1
            continue
        destination_path = destination_root / source_path.relative_to(source_root)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source_path), str(destination_path))
        mirrored_files += 1

    _progress(
        progress_fn,
        (
            f"Mirrored {mirrored_files} image(s) into working class-root copy"
            f" at {destination_root}."
        ),
    )
    return {
        "mirrored_images": int(mirrored_files),
        "skipped_non_images": int(skipped_files),
    }


def _build_cross_class_quarantine_actions(
    *,
    cross_class_conflicts_path: Path,
    starting_group_index: int,
) -> List[DuplicateCleanupAction]:
    actions: List[DuplicateCleanupAction] = []
    seen_paths: set[str] = set()
    rows = _read_csv_rows(cross_class_conflicts_path)
    group_index = int(starting_group_index)

    for row in rows:
        pair_paths = [
            str(row.get("path_a", "")).strip(),
            str(row.get("path_b", "")).strip(),
        ]
        pair_paths = [item for item in pair_paths if item]
        if not pair_paths:
            continue
        group_index += 1
        for relative_path in pair_paths:
            if relative_path in seen_paths:
                continue
            seen_paths.add(relative_path)
            actions.append(
                DuplicateCleanupAction(
                    duplicate_group_index=group_index,
                    duplicate_count=len(pair_paths),
                    kept_relative_paths="",
                    selected_relative_path=relative_path,
                    deleted_relative_path=relative_path,
                    delete_reason="cross_class_conflict_quarantine",
                )
            )
    return actions


def prepare_class_root_for_materialization(
    *,
    class_root: Path,
    crop_name: str,
    part_name: str = "unspecified",
    audit_artifact_root: Path,
    prepared_class_root: Path,
    prepared_artifact_root: Path,
    taxonomy_path: Optional[Path] = None,
    dino_model_id: str = DEFAULT_DINOV3_MODEL_ID,
    bioclip_model_id: str = DEFAULT_BIOCLIP_MODEL_ID,
    device: str = "cpu",
    batch_size: int = 16,
    neighbors: int = DEFAULT_NEIGHBORS,
    cleanup_seed: int = DEFAULT_SEED,
    quarantine_cross_class_conflicts: bool = True,
    under_min_eval_policy: str = "block",
    materialization_strategy: str = "auto",
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    class_root = Path(class_root).resolve()
    audit_artifact_root = Path(audit_artifact_root).resolve()
    prepared_class_root = Path(prepared_class_root).resolve()
    prepared_artifact_root = Path(prepared_artifact_root).resolve()

    if not class_root.is_dir():
        raise FileNotFoundError(f"Class-root dataset not found: {class_root}")
    if prepared_artifact_root == audit_artifact_root:
        raise ValueError("prepared_artifact_root must differ from audit_artifact_root.")
    source_is_prepared_target = prepared_class_root == class_root
    if prepared_artifact_root.exists():
        shutil.rmtree(prepared_artifact_root)
    prepared_artifact_root.mkdir(parents=True, exist_ok=True)

    exact_duplicates_source = _resolve_report_path(audit_artifact_root, "exact_duplicates.csv")
    review_source = _resolve_report_path(audit_artifact_root, "same_class_review_candidates.csv")
    dataset_manifest_source = _resolve_report_path(audit_artifact_root, "dataset_manifest.csv")
    cross_class_conflicts_source = _resolve_report_path(audit_artifact_root, "cross_class_conflicts.csv")

    cleanup_plan_path = prepared_artifact_root / "cleanup_plan.csv"
    conflict_plan_path = prepared_artifact_root / "cross_class_conflict_quarantine_plan.csv"
    if source_is_prepared_target:
        _progress(
            progress_fn,
            (
                "Source class-root already matches the prepared class-root target; "
                "skipping mirror and cleanup copy, then re-running audit in the prepared artifact root."
            ),
        )
        mirror_summary = {
            "mirrored_images": 0,
            "skipped_non_images": 0,
            "source_already_prepared": 1,
        }
        cleanup_actions: List[DuplicateCleanupAction] = []
        cleanup_deleted_files = 0
        conflict_actions: List[DuplicateCleanupAction] = []
        conflict_deleted_files = 0
        write_cleanup_report(cleanup_plan_path, cleanup_actions)
        write_cleanup_report(conflict_plan_path, conflict_actions)
    else:
        _progress(progress_fn, "Creating a working class-root copy from the audited dataset.")
        mirror_summary = _mirror_class_root_dataset(
            source_root=class_root,
            destination_root=prepared_class_root,
            materialization_strategy=materialization_strategy,
            progress_fn=progress_fn,
        )

        _progress(progress_fn, "Applying exact-duplicate and auto-resolved same-class cleanup actions.")
        cleanup_actions = build_combined_cleanup_plan(
            dataset_root=prepared_class_root,
            exact_duplicates_source=exact_duplicates_source,
            review_source=review_source,
            dataset_manifest_source=dataset_manifest_source,
            seed=cleanup_seed,
        )
        write_cleanup_report(cleanup_plan_path, cleanup_actions)
        cleanup_deleted_files = apply_cleanup_plan(dataset_root=prepared_class_root, actions=cleanup_actions)

        conflict_actions = []
        conflict_deleted_files = 0
        if quarantine_cross_class_conflicts:
            _progress(progress_fn, "Quarantining both sides of reported cross-class conflicts.")
            conflict_actions = _build_cross_class_quarantine_actions(
                cross_class_conflicts_path=cross_class_conflicts_source,
                starting_group_index=max((action.duplicate_group_index for action in cleanup_actions), default=0),
            )
            write_cleanup_report(conflict_plan_path, conflict_actions)
            conflict_deleted_files = apply_cleanup_plan(dataset_root=prepared_class_root, actions=conflict_actions)
        else:
            write_cleanup_report(conflict_plan_path, [])

    _progress(progress_fn, "Re-running the grouped audit on the prepared working copy.")
    rerun_summary = build_grouped_dataset_plan(
        class_root=prepared_class_root,
        crop_name=crop_name,
        part_name=part_name,
        artifact_root=prepared_artifact_root,
        taxonomy_path=taxonomy_path,
        dino_model_id=dino_model_id,
        bioclip_model_id=bioclip_model_id,
        device=device,
        batch_size=batch_size,
        neighbors=neighbors,
        under_min_eval_policy=under_min_eval_policy,
        progress_fn=progress_fn,
    )

    result: Dict[str, Any] = {
        "source_class_root": str(class_root),
        "prepared_class_root": str(prepared_class_root),
        "audit_artifact_root": str(audit_artifact_root),
        "prepared_artifact_root": str(prepared_artifact_root),
        "crop_name": str(crop_name),
        "part_name": str(part_name),
        "dataset_key": build_prepared_dataset_key(crop_name, part_name),
        "cleanup_seed": int(cleanup_seed),
        "under_min_eval_policy": str(under_min_eval_policy),
        "materialization_strategy": str(materialization_strategy),
        "quarantine_cross_class_conflicts": bool(quarantine_cross_class_conflicts),
        "source_already_prepared": bool(source_is_prepared_target),
        "mirror_summary": dict(mirror_summary),
        "cleanup_plan_path": str(cleanup_plan_path),
        "cleanup_candidate_files": len(cleanup_actions),
        "cleanup_deleted_files": int(cleanup_deleted_files),
        "cross_class_conflict_quarantine_plan_path": str(conflict_plan_path),
        "cross_class_conflict_candidate_files": len(conflict_actions),
        "cross_class_conflict_deleted_files": int(conflict_deleted_files),
        "prepared_runtime_ready": bool(rerun_summary.get("runtime_ready")),
        "rerun_summary": dict(rerun_summary),
    }
    write_json(
        prepared_artifact_root / "materialization_prep_summary.json",
        result,
        ensure_ascii=False,
    )
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-root", type=Path, required=True, help="Original flat class-root dataset.")
    parser.add_argument("--crop", type=str, required=True, help="Crop name for taxonomy alignment.")
    parser.add_argument("--part", type=str, default="unspecified", help="Part name used for prepared dataset naming.")
    parser.add_argument(
        "--audit-artifact-root",
        type=Path,
        required=True,
        help="Artifact root that already contains Notebook 0 audit CSV reports.",
    )
    parser.add_argument(
        "--prepared-class-root",
        type=Path,
        required=True,
        help="Destination class-root working copy that will be cleaned for materialization.",
    )
    parser.add_argument(
        "--prepared-artifact-root",
        type=Path,
        required=True,
        help="Artifact root for the post-cleanup rerun summary.",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=Path,
        default=Path("config") / "plant_taxonomy.json",
        help="Taxonomy path used during the post-cleanup rerun audit.",
    )
    parser.add_argument("--dino-model-id", type=str, default=DEFAULT_DINOV3_MODEL_ID)
    parser.add_argument("--bioclip-model-id", type=str, default=DEFAULT_BIOCLIP_MODEL_ID)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS)
    parser.add_argument("--cleanup-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--materialization-strategy",
        type=str,
        default="auto",
        help="Class-root copy strategy: auto, copy, symlink, or hardlink.",
    )
    parser.add_argument(
        "--skip-cross-class-quarantine",
        action="store_true",
        help="Do not quarantine images listed in cross_class_conflicts.csv.",
    )
    parser.add_argument(
        "--under-min-eval-policy",
        type=str,
        default="block",
        choices=("block", "skip"),
        help="Whether classes with <3 eval families/source bundles should block or be skipped.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    result = prepare_class_root_for_materialization(
        class_root=args.class_root,
        crop_name=args.crop,
        part_name=args.part,
        audit_artifact_root=args.audit_artifact_root,
        prepared_class_root=args.prepared_class_root,
        prepared_artifact_root=args.prepared_artifact_root,
        taxonomy_path=args.taxonomy_path,
        dino_model_id=args.dino_model_id,
        bioclip_model_id=args.bioclip_model_id,
        device=args.device,
        batch_size=args.batch_size,
        neighbors=args.neighbors,
        cleanup_seed=args.cleanup_seed,
        quarantine_cross_class_conflicts=not bool(args.skip_cross_class_quarantine),
        under_min_eval_policy=args.under_min_eval_policy,
        materialization_strategy=args.materialization_strategy,
        progress_fn=lambda message: print(f"[PREP] {message}"),
    )
    print(f"[PREP] prepared_runtime_ready={result.get('prepared_runtime_ready')}")
    print(f"[PREP] prepared_class_root={result.get('prepared_class_root')}")
    print(f"[PREP] prepared_artifact_root={result.get('prepared_artifact_root')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
