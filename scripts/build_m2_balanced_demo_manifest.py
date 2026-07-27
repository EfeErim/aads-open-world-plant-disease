#!/usr/bin/env python3
"""Build a target-balanced M2 demo manifest with external iNaturalist images."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.enrich_m2_demo_image_set import (
    IMAGE_EXT,
    PLAN,
    SUPPORTED_BEHAVIOR,
    USER_LIKE_BUCKETS,
    ImagePlan,
    _collect_external_images,
    _existing_hashes,
    _load_reject_ids,
    _next_demo_number,
    _read_csv,
    _sha256,
    _write_csv,
)

SUPPORTED_TARGETS = tuple(plan.expected_target for plan in PLAN)
BALANCED_REPLACEMENT_ACTION = "balanced_external_inaturalist_append"
PHOTO_ID_RE = re.compile(r"(?:^|[; ])photo_id=([^; ]+)")
BALANCED_QUERY_OVERRIDES = {
    "strawberry__fruit": (
        "ripe strawberry",
        "strawberry plant fruit",
        "wild strawberry fruit",
        "Fragaria fruit",
    ),
    "grape__fruit": (
        "grape bunch",
        "grapes on vine",
        "grape cluster",
        "Vitis vinifera fruit",
    ),
    "apricot__fruit": (
        "ripe apricot",
        "apricot tree fruit",
        "Prunus armeniaca fruit",
        "apricot fruit",
    ),
}


def _photo_ids_from_rows(rows: list[dict[str, str]]) -> set[str]:
    photo_ids: set[str] = set()
    for row in rows:
        haystack = f"{row.get('notes', '')}; {row.get('manual_review_note', '')}"
        match = PHOTO_ID_RE.search(haystack)
        if match:
            photo_ids.add(match.group(1).strip())
    return photo_ids


def _remove_prior_balanced_images(
    *,
    image_dir: Path,
    manifest_rows: list[dict[str, str]],
    qc_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], int]:
    balanced_ids = {
        row["image_id"]
        for row in qc_rows
        if str(row.get("replacement_action") or "") == BALANCED_REPLACEMENT_ACTION
    }
    image_dir_resolved = image_dir.resolve(strict=False)
    for row in qc_rows:
        if row.get("image_id") not in balanced_ids:
            continue
        final_path = Path(str(row.get("final_path") or ""))
        if final_path.is_file() and final_path.resolve(strict=False).parent == image_dir_resolved:
            final_path.unlink()
    return (
        [row for row in manifest_rows if row.get("image_id") not in balanced_ids],
        [row for row in qc_rows if row.get("image_id") not in balanced_ids],
        len(balanced_ids),
    )


def _bucket_rank(row: dict[str, str]) -> tuple[int, str, str]:
    bucket = str(row.get("user_like_bucket") or "")
    rank = {"difficult": 0, "borderline": 1, "clear": 2, "internet_negative": 3}.get(bucket, 4)
    return rank, str(row.get("disease_class") or row.get("expected_class") or ""), str(row.get("image_id") or "")


def _select_existing_rows(rows: list[dict[str, str]], *, target_count: int) -> list[dict[str, str]]:
    rows_by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("expected_target") in SUPPORTED_TARGETS:
            rows_by_target[str(row["expected_target"])].append(row)

    selected: list[dict[str, str]] = []
    for target in SUPPORTED_TARGETS:
        target_rows = sorted(rows_by_target[target], key=_bucket_rank)
        selected.extend(target_rows[:target_count])

    selected_ids = {row["image_id"] for row in selected}
    selected.extend(row for row in rows if row.get("expected_target") not in SUPPORTED_TARGETS)
    return [
        row
        for row in rows
        if row.get("image_id") in selected_ids or row.get("expected_target") not in SUPPORTED_TARGETS
    ]


def _required_external_counts(rows: list[dict[str, str]], *, target_count: int) -> dict[str, int]:
    counts = Counter(str(row.get("expected_target") or "") for row in rows)
    return {target: max(0, target_count - counts[target]) for target in SUPPORTED_TARGETS}


def _fieldnames_union(existing: list[str], rows: list[dict[str, str]]) -> list[str]:
    fieldnames = list(existing)
    seen = set(fieldnames)
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames


def build_balanced_manifest(
    *,
    package_dir: Path,
    target_count: int,
    output_manifest: Path,
    staging_dir: Path,
    reject_ids_path: Path,
    per_page: int,
    max_pages: int,
    timeout: int,
    sleep_seconds: float,
    min_side: int,
    min_bytes: int,
    min_contrast: float,
    max_aspect_ratio: float,
) -> dict[str, object]:
    image_dir = package_dir / "images"
    manifest_dir = package_dir / "manifests"
    canonical_manifest = manifest_dir / "m2_full_image_set_run_manifest.csv"
    qc_path = manifest_dir / "m2_full_image_set_qc.csv"

    manifest_rows, manifest_fields = _read_csv(canonical_manifest)
    qc_rows, qc_fields = _read_csv(qc_path)
    manifest_rows, qc_rows, removed_prior = _remove_prior_balanced_images(
        image_dir=image_dir,
        manifest_rows=manifest_rows,
        qc_rows=qc_rows,
    )

    requirements = _required_external_counts(manifest_rows, target_count=target_count)
    used_hashes = _existing_hashes(image_dir)
    reject_ids = _load_reject_ids(reject_ids_path) | _photo_ids_from_rows(manifest_rows) | _photo_ids_from_rows(qc_rows)
    next_number = _next_demo_number(manifest_rows)

    added_rows: list[dict[str, str]] = []
    added_qc_rows: list[dict[str, str]] = []
    added_counts: Counter[str] = Counter()
    plan_by_target = {plan.expected_target: plan for plan in PLAN}

    for target, needed in requirements.items():
        if needed <= 0:
            continue
        queries = BALANCED_QUERY_OVERRIDES.get(target, plan_by_target[target].queries)
        plan = replace(plan_by_target[target], queries=queries, count=needed)
        accepted_images = _collect_external_images(
            plan,
            staging_dir=staging_dir / target,
            used_hashes=used_hashes,
            reject_ids=reject_ids,
            per_page=per_page,
            max_pages=max_pages,
            timeout=timeout,
            sleep_seconds=sleep_seconds,
            min_side=min_side,
            min_bytes=min_bytes,
            min_contrast=min_contrast,
            max_aspect_ratio=max_aspect_ratio,
        )
        if len(accepted_images) != needed:
            raise RuntimeError(f"{target} selected {len(accepted_images)}/{needed} external images")
        for accepted in accepted_images:
            image_id = f"demo_{next_number:03d}"
            final_path = image_dir / f"{image_id}_{target.replace('__', '_')}{IMAGE_EXT}"
            shutil.copy2(accepted.path, final_path)
            bucket = USER_LIKE_BUCKETS[(next_number - 1) % len(USER_LIKE_BUCKETS)]
            notes = (
                f"balanced external iNaturalist enrichment; bucket={bucket}; "
                f"target_count={target_count}; query={accepted.candidate.query}; "
                f"taxon={accepted.candidate.taxon_name}; observation={accepted.candidate.observation_url}; "
                f"photo_id={accepted.candidate.photo_id}"
            )
            added_rows.append(
                {
                    "image_id": image_id,
                    "source": f"staged_external:{final_path.as_posix()}",
                    "expected_target": target,
                    "expected_crop": plan.expected_crop,
                    "expected_part": plan.expected_part,
                    "expected_class": "",
                    "expected_behavior": SUPPORTED_BEHAVIOR,
                    "notes": notes,
                    "original_source": accepted.candidate.observation_url,
                    "resolved_source_path": final_path.as_posix(),
                    "disease_class": "",
                    "user_like_bucket": bucket,
                }
            )
            added_qc_rows.append(
                {
                    "image_id": image_id,
                    "expected_target": target,
                    "qc_status": "keep",
                    "qc_reason": "",
                    "replacement_action": BALANCED_REPLACEMENT_ACTION,
                    "source_path": accepted.candidate.source_url,
                    "final_path": final_path.as_posix(),
                    "sha256": accepted.sha256,
                    "width": str(accepted.width),
                    "height": str(accepted.height),
                    "contrast": f"{accepted.contrast:.2f}",
                    "expected_class": "",
                    "user_like_bucket": bucket,
                    "manual_review_note": (
                        f"photo_id={accepted.candidate.photo_id}; observation={accepted.candidate.observation_url}; "
                        f"license={accepted.candidate.license_code}; attribution={accepted.candidate.attribution}"
                    ),
                }
            )
            added_counts[target] += 1
            reject_ids.add(accepted.candidate.photo_id)
            next_number += 1

    package_rows = manifest_rows + added_rows
    package_qc_rows = qc_rows + added_qc_rows
    balanced_rows = _select_existing_rows(package_rows, target_count=target_count)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    balanced_manifest_fields = _fieldnames_union(manifest_fields, balanced_rows)
    _write_csv(output_manifest, balanced_rows, balanced_manifest_fields)
    _write_csv(qc_path, package_qc_rows, _fieldnames_union(qc_fields, package_qc_rows))

    target_counts = Counter(str(row.get("expected_target") or "") for row in balanced_rows)
    qc_hashes = [str(row.get("sha256") or "") for row in package_qc_rows if row.get("sha256")]
    summary = {
        "schema_version": "v1_m2_balanced_demo_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": canonical_manifest.as_posix(),
        "output_manifest": output_manifest.as_posix(),
        "target_count": target_count,
        "row_count": len(balanced_rows),
        "supported_row_count": sum(target_counts[target] for target in SUPPORTED_TARGETS),
        "non_supported_row_count": len(balanced_rows) - sum(target_counts[target] for target in SUPPORTED_TARGETS),
        "target_counts": dict(sorted(target_counts.items())),
        "added_count": len(added_rows),
        "added_target_counts": dict(sorted(added_counts.items())),
        "removed_prior_balanced_count": removed_prior,
        "package_qc_row_count": len(package_qc_rows),
        "package_duplicate_hash_count": len(qc_hashes) - len(set(qc_hashes)),
        "quality_filters": {
            "min_side": int(min_side),
            "min_bytes": int(min_bytes),
            "min_contrast": float(min_contrast),
            "max_aspect_ratio": float(max_aspect_ratio),
        },
    }
    summary_path = output_manifest.with_name(output_manifest.stem.replace("_run_manifest", "_summary") + ".json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path("docs/demo_assets/m2_full_image_set"))
    parser.add_argument("--target-count", type=int, default=80)
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("docs/demo_assets/m2_full_image_set/manifests/m2_balanced_80_run_manifest.csv"),
    )
    parser.add_argument("--staging-dir", type=Path, default=Path(".runtime_tmp/m2_balanced_external_downloads"))
    parser.add_argument(
        "--reject-ids",
        type=Path,
        default=Path("docs/demo_assets/m2_full_image_set/manifests/m2_external_inaturalist_reject_photo_ids.txt"),
    )
    parser.add_argument("--per-page", type=int, default=80)
    parser.add_argument("--max-pages", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--min-side", type=int, default=360)
    parser.add_argument("--min-bytes", type=int, default=25_000)
    parser.add_argument("--min-contrast", type=float, default=18.0)
    parser.add_argument("--max-aspect-ratio", type=float, default=2.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_balanced_manifest(
        package_dir=args.package_dir,
        target_count=max(1, int(args.target_count)),
        output_manifest=args.output_manifest,
        staging_dir=args.staging_dir,
        reject_ids_path=args.reject_ids,
        per_page=max(1, int(args.per_page)),
        max_pages=max(1, int(args.max_pages)),
        timeout=max(1, int(args.timeout)),
        sleep_seconds=max(0.0, float(args.sleep_seconds)),
        min_side=max(1, int(args.min_side)),
        min_bytes=max(1, int(args.min_bytes)),
        min_contrast=float(args.min_contrast),
        max_aspect_ratio=float(args.max_aspect_ratio),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
