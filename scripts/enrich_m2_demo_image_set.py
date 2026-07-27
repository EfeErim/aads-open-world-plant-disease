#!/usr/bin/env python3
"""Append externally sourced iNaturalist images to the saved M2 demo set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat, UnidentifiedImageError

IMAGE_EXT = ".jpg"
MANIFEST_NAME = "m2_full_image_set_run_manifest.csv"
QC_NAME = "m2_full_image_set_qc.csv"
SUMMARY_NAME = "m2_full_image_set_summary.json"
SUPPORTED_BEHAVIOR = "external supported crop/part image; disease answer or review expected"
USER_LIKE_BUCKETS = ("clear",) * 7 + ("difficult",) * 2 + ("borderline",)


@dataclass(frozen=True)
class ImagePlan:
    expected_target: str
    expected_crop: str
    expected_part: str
    taxon_name: str
    queries: tuple[str, ...]
    count: int


@dataclass(frozen=True)
class Candidate:
    source_url: str
    observation_url: str
    photo_id: str
    query: str
    taxon_name: str
    license_code: str
    attribution: str


@dataclass(frozen=True)
class AcceptedImage:
    path: Path
    sha256: str
    width: int
    height: int
    contrast: float
    candidate: Candidate


PLAN: tuple[ImagePlan, ...] = (
    ImagePlan(
        "tomato__fruit",
        "tomato",
        "fruit",
        "Solanum lycopersicum",
        ("tomato fruit", "ripe tomato", "unripe tomato", "tomato plant fruit", "Solanum lycopersicum fruit"),
        10,
    ),
    ImagePlan(
        "tomato__leaf",
        "tomato",
        "leaf",
        "Solanum lycopersicum",
        ("tomato leaf", "tomato leaves", "tomato plant leaf", "Solanum lycopersicum leaf"),
        10,
    ),
    ImagePlan(
        "strawberry__fruit",
        "strawberry",
        "fruit",
        "Fragaria",
        ("strawberry fruit", "ripe strawberry", "strawberries plant", "Fragaria fruit"),
        10,
    ),
    ImagePlan(
        "strawberry__leaf",
        "strawberry",
        "leaf",
        "Fragaria",
        ("strawberry leaf", "strawberry leaves", "Fragaria leaf", "Fragaria leaves"),
        10,
    ),
    ImagePlan(
        "grape__fruit",
        "grape",
        "fruit",
        "Vitis vinifera",
        ("grape fruit", "grapes vine", "grape bunch", "Vitis vinifera fruit"),
        10,
    ),
    ImagePlan(
        "grape__leaf",
        "grape",
        "leaf",
        "Vitis vinifera",
        ("grape leaf", "grape leaves", "grapevine leaf", "Vitis vinifera leaf"),
        10,
    ),
    ImagePlan(
        "apricot__fruit",
        "apricot",
        "fruit",
        "Prunus armeniaca",
        ("apricot fruit", "ripe apricot", "apricot tree fruit", "Prunus armeniaca fruit"),
        10,
    ),
    ImagePlan(
        "apricot__leaf",
        "apricot",
        "leaf",
        "Prunus armeniaca",
        (
            "apricot leaf",
            "apricot leaves",
            "apricot tree leaf",
            "apricot tree leaves",
            "apricot tree",
            "apricot plant",
            "Prunus armeniaca leaf",
            "Prunus armeniaca",
        ),
        10,
    ),
)


def _json_get(url: str, *, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "aads-m2-demo-external-enrichment/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, output_path: Path, *, timeout: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "aads-m2-demo-external-enrichment/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _photo_url(photo: dict[str, Any]) -> str:
    url = str(photo.get("url") or "")
    if not url:
        return ""
    return url.replace("/square.", "/large.").replace("square.jpg", "large.jpg")


def _fetch_observations(
    plan: ImagePlan,
    *,
    query: str,
    page: int,
    per_page: int,
    timeout: int,
) -> list[dict[str, Any]]:
    payload = {
        "photos": "true",
        "taxon_name": plan.taxon_name,
        "q": query,
        "per_page": per_page,
        "page": page,
        "order_by": "created_at",
        "order": "desc",
    }
    url = f"https://api.inaturalist.org/v1/observations?{urllib.parse.urlencode(payload)}"
    return list(_json_get(url, timeout=timeout).get("results") or [])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_image(path: Path) -> tuple[int, int, float] | None:
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            stat = ImageStat.Stat(rgb.convert("L"))
            return int(rgb.width), int(rgb.height), float(stat.stddev[0])
    except (OSError, UnidentifiedImageError):
        return None


def _passes_quality(
    path: Path,
    *,
    min_side: int,
    min_bytes: int,
    min_contrast: float,
    max_aspect_ratio: float,
) -> tuple[int, int, float] | None:
    if path.stat().st_size < min_bytes:
        return None
    profile = _inspect_image(path)
    if profile is None:
        return None
    width, height, contrast = profile
    if min(width, height) < min_side:
        return None
    if contrast < min_contrast:
        return None
    aspect_ratio = max(width / max(height, 1), height / max(width, 1))
    if aspect_ratio > max_aspect_ratio:
        return None
    return width, height, contrast


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _next_demo_number(rows: list[dict[str, str]]) -> int:
    numbers: list[int] = []
    for row in rows:
        image_id = str(row.get("image_id") or "")
        if image_id.startswith("demo_"):
            try:
                numbers.append(int(image_id.split("_", 1)[1]))
            except ValueError:
                continue
    return max(numbers, default=0) + 1


def _existing_hashes(image_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for path in image_dir.glob("demo_*"):
        if not path.is_file():
            continue
        try:
            hashes.add(_sha256(path))
        except OSError:
            continue
    return hashes


def _load_reject_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _remove_prior_enrichment(
    *,
    image_dir: Path,
    rows: list[dict[str, str]],
    qc_rows: list[dict[str, str]],
    replace_existing_enrichment: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]], int]:
    if not replace_existing_enrichment:
        return rows, qc_rows, 0
    enrichment_ids = {
        row["image_id"]
        for row in qc_rows
        if str(row.get("replacement_action") or "")
        in {"internet_enrichment_append", "prepared_dataset_enrichment_append", "external_inaturalist_enrichment_append"}
    }
    for row in rows:
        if row.get("image_id") not in enrichment_ids:
            continue
        resolved = Path(str(row.get("resolved_source_path") or ""))
        if resolved.is_file() and resolved.resolve(strict=False).parent == image_dir.resolve(strict=False):
            resolved.unlink()
    return (
        [row for row in rows if row.get("image_id") not in enrichment_ids],
        [row for row in qc_rows if row.get("image_id") not in enrichment_ids],
        len(enrichment_ids),
    )


def _collect_external_images(
    plan: ImagePlan,
    *,
    staging_dir: Path,
    used_hashes: set[str],
    reject_ids: set[str],
    per_page: int,
    max_pages: int,
    timeout: int,
    sleep_seconds: float,
    min_side: int,
    min_bytes: int,
    min_contrast: float,
    max_aspect_ratio: float,
) -> list[AcceptedImage]:
    accepted: list[AcceptedImage] = []
    seen_photo_ids: set[str] = set()
    staging_dir.mkdir(parents=True, exist_ok=True)
    for query in plan.queries:
        for page in range(1, max_pages + 1):
            if len(accepted) >= plan.count:
                return accepted
            observations = _fetch_observations(plan, query=query, page=page, per_page=per_page, timeout=timeout)
            if not observations:
                break
            for observation in observations:
                if len(accepted) >= plan.count:
                    return accepted
                for photo_record in observation.get("observation_photos") or []:
                    photo = photo_record.get("photo") or {}
                    photo_id = str(photo.get("id") or photo.get("uuid") or photo.get("url") or "")
                    if not photo_id or photo_id in seen_photo_ids or photo_id in reject_ids:
                        continue
                    seen_photo_ids.add(photo_id)
                    source_url = _photo_url(photo)
                    if not source_url:
                        continue
                    temp_path = staging_dir / f"{plan.expected_target}_{len(accepted) + 1:03d}_{photo_id}{IMAGE_EXT}"
                    try:
                        _download(source_url, temp_path, timeout=timeout)
                    except Exception as exc:
                        print(f"[WARN] download failed {source_url}: {exc}")
                        continue
                    profile = _passes_quality(
                        temp_path,
                        min_side=min_side,
                        min_bytes=min_bytes,
                        min_contrast=min_contrast,
                        max_aspect_ratio=max_aspect_ratio,
                    )
                    if profile is None:
                        temp_path.unlink(missing_ok=True)
                        continue
                    sha256 = _sha256(temp_path)
                    if sha256 in used_hashes:
                        temp_path.unlink(missing_ok=True)
                        continue
                    used_hashes.add(sha256)
                    width, height, contrast = profile
                    candidate = Candidate(
                        source_url=source_url,
                        observation_url=str(observation.get("uri") or ""),
                        photo_id=photo_id,
                        query=query,
                        taxon_name=plan.taxon_name,
                        license_code=str(photo.get("license_code") or ""),
                        attribution=str(photo.get("attribution") or ""),
                    )
                    accepted.append(
                        AcceptedImage(
                            path=temp_path,
                            sha256=sha256,
                            width=width,
                            height=height,
                            contrast=contrast,
                            candidate=candidate,
                        )
                    )
                    print(f"[OK] {plan.expected_target} {len(accepted)}/{plan.count}: photo_id={photo_id}")
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    break
    return accepted


def _recompute_summary(
    rows: list[dict[str, str]],
    qc_rows: list[dict[str, str]],
    *,
    added_count: int,
    removed_prior_enrichment_count: int,
    added_target_counts: dict[str, int],
    quality_filters: dict[str, object],
) -> dict[str, object]:
    per_disease_class_counts: dict[str, Counter[str]] = {}
    for row in rows:
        target = str(row.get("expected_target") or "")
        disease_class = str(row.get("disease_class") or row.get("expected_class") or "").strip()
        if disease_class:
            per_disease_class_counts.setdefault(target, Counter())[disease_class] += 1
    return {
        "schema_version": "v2_m2_user_like_curated_image_set_with_external_enrichment",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "target_counts": dict(sorted(Counter(str(row.get("expected_target") or "") for row in rows).items())),
        "duplicate_hash_count": 0,
        "qc_status_counts": dict(sorted(Counter(str(row.get("qc_status") or "unknown") for row in qc_rows).items())),
        "user_like_bucket_counts": dict(sorted(Counter(str(row.get("user_like_bucket") or "") for row in rows).items())),
        "per_disease_class_counts": {
            target: dict(sorted(counts.items())) for target, counts in sorted(per_disease_class_counts.items())
        },
        "last_enrichment": {
            "source": "inaturalist",
            "added_count": int(added_count),
            "removed_prior_enrichment_count": int(removed_prior_enrichment_count),
            "added_target_counts": dict(sorted(added_target_counts.items())),
            "quality_filters": quality_filters,
        },
    }


def enrich_package(
    *,
    package_dir: Path,
    staging_dir: Path,
    reject_ids_path: Path,
    replace_existing_enrichment: bool,
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
    manifest_path = manifest_dir / MANIFEST_NAME
    qc_path = manifest_dir / QC_NAME
    rows, manifest_fields = _read_csv(manifest_path)
    qc_rows, qc_fields = _read_csv(qc_path)
    rows, qc_rows, removed_count = _remove_prior_enrichment(
        image_dir=image_dir,
        rows=rows,
        qc_rows=qc_rows,
        replace_existing_enrichment=replace_existing_enrichment,
    )
    used_hashes = _existing_hashes(image_dir)
    reject_ids = _load_reject_ids(reject_ids_path)
    next_number = _next_demo_number(rows)
    added_rows: list[dict[str, str]] = []
    added_qc_rows: list[dict[str, str]] = []
    added_target_counts: Counter[str] = Counter()
    for plan in PLAN:
        accepted_images = _collect_external_images(
            plan,
            staging_dir=staging_dir / plan.expected_target,
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
        if len(accepted_images) != plan.count:
            raise RuntimeError(f"{plan.expected_target} selected {len(accepted_images)}/{plan.count} external images")
        for accepted in accepted_images:
            image_id = f"demo_{next_number:03d}"
            final_path = image_dir / f"{image_id}_{plan.expected_target.replace('__', '_')}{IMAGE_EXT}"
            shutil.copy2(accepted.path, final_path)
            bucket = USER_LIKE_BUCKETS[(next_number - 1) % len(USER_LIKE_BUCKETS)]
            notes = (
                f"external iNaturalist enrichment; bucket={bucket}; query={accepted.candidate.query}; "
                f"taxon={accepted.candidate.taxon_name}; observation={accepted.candidate.observation_url}; "
                f"photo_id={accepted.candidate.photo_id}"
            )
            added_rows.append(
                {
                    "image_id": image_id,
                    "source": f"staged_external:{final_path.as_posix()}",
                    "expected_target": plan.expected_target,
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
                    "expected_target": plan.expected_target,
                    "qc_status": "keep",
                    "qc_reason": "",
                    "replacement_action": "external_inaturalist_enrichment_append",
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
            added_target_counts[plan.expected_target] += 1
            next_number += 1
    rows.extend(added_rows)
    qc_rows.extend(added_qc_rows)
    _write_csv(manifest_path, rows, manifest_fields)
    _write_csv(manifest_dir / "m2_full_image_set_manifest.csv", rows, manifest_fields)
    _write_csv(qc_path, qc_rows, qc_fields)
    summary = _recompute_summary(
        rows,
        qc_rows,
        added_count=len(added_rows),
        removed_prior_enrichment_count=removed_count,
        added_target_counts=dict(added_target_counts),
        quality_filters={
            "min_side": int(min_side),
            "min_bytes": int(min_bytes),
            "min_contrast": float(min_contrast),
            "max_aspect_ratio": float(max_aspect_ratio),
        },
    )
    (manifest_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path("docs/demo_assets/m2_full_image_set"))
    parser.add_argument("--staging-dir", type=Path, default=Path(".runtime_tmp/m2_external_enrichment_downloads"))
    parser.add_argument(
        "--reject-ids",
        type=Path,
        default=Path("docs/demo_assets/m2_full_image_set/manifests/m2_external_inaturalist_reject_photo_ids.txt"),
    )
    parser.add_argument("--replace-existing-enrichment", action="store_true")
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
    summary = enrich_package(
        package_dir=args.package_dir,
        staging_dir=args.staging_dir,
        reject_ids_path=args.reject_ids,
        replace_existing_enrichment=bool(args.replace_existing_enrichment),
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
