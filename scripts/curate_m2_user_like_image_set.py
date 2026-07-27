#!/usr/bin/env python3
"""Rebuild the saved M2 image set from cleaner user-like local pools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageStat, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_release_runtime import DEFAULT_MATERIALIZED_DATASET_ROOT  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_PRIORITY = ("test", "val", "continual", "train")
SUPPORTED_TARGET_COUNTS = {
    "apricot__fruit": 54,
    "apricot__leaf": 37,
    "grape__fruit": 55,
    "grape__leaf": 77,
    "strawberry__fruit": 47,
    "strawberry__leaf": 45,
    "tomato__fruit": 75,
    "tomato__leaf": 108,
}
SPECIAL_TARGET_COUNTS = {
    "unknown_crop": 6,
    "non_plant": 6,
    "tomato__unknown_part": 1,
    "grape__unknown_part": 1,
}
BAD_NAME_TOKENS = (
    "ekran",
    "screenshot",
    "screen_shot",
    "petri",
    "culture",
    "laboratory",
    "lab_",
    "microscope",
    "insect_only",
    "flower_only",
    "sadece_sap",
    "stem_only",
    "text",
    "watermark",
)
UNKNOWN_CROP_TOKENS = (
    "apple",
    "elma",
    "pepper",
    "biber",
    "cucumber",
    "cucurbit",
    "salatal",
    "potato",
    "patates",
)
NON_PLANT_TOKENS = (
    "tractor",
    "traktor",
    "sulama",
    "irrigation",
    "tool",
    "equipment",
    "maintenance",
    "bakim",
    "duvar",
    "fabric",
    "kumas",
    "soil",
    "toprak",
    "table",
    "desk",
    "non_plant_misc",
    "alakasiz",
    "alakasız",
    "kopek",
    "köpek",
)
SUPPORTED_CROP_TOKENS = (
    "apricot",
    "kayisi",
    "kayısı",
    "tomato",
    "domates",
    "strawberry",
    "grape",
    "uzum",
    "üzüm",
)
USER_MIX = ("clear",) * 7 + ("difficult",) * 2 + ("borderline",)
COMMONS_NON_PLANT_FILES = (
    "Garden tools.jpg",
    "Potting soil.jpg",
    "Garden tools rack (i).jpg",
    "Soft plastic pot for planting.jpg",
    "Swedish mobile irrigation equipment 001.JPG",
    "HandsInSoil.jpg",
    "Wood table in front of white wall (Unsplash).jpg",
    "Garden watering hose lying on grassy lawn.jpg",
)


@dataclass(frozen=True)
class Candidate:
    path: Path
    target: str
    expected_crop: str
    expected_part: str
    expected_class: str
    split: str
    sha256: str
    width: int
    height: int
    contrast: float
    semantic_warnings: tuple[str, ...]


def _label_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower().replace("Ä±", "i")).replace("ı", "i")
    return "".join(ch for ch in normalized if ch.isalnum() and not unicodedata.combining(ch))


def is_healthy_class(class_name: str) -> bool:
    key = _label_key(class_name)
    return "healthy" in key or "saglikli" in key


def image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> tuple[int, int, float]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        stat = ImageStat.Stat(rgb.convert("L"))
        return int(rgb.width), int(rgb.height), float(stat.stddev[0])


def technical_profile(path: Path) -> tuple[list[str], int, int, float] | None:
    try:
        width, height, contrast = inspect_image(path)
    except (OSError, UnidentifiedImageError):
        return None
    reasons: list[str] = []
    if min(width, height) < 128:
        reasons.append("too_blurry_or_tiny")
    if contrast < 18:
        reasons.append("too_blurry_or_tiny")
    if max(width / max(height, 1), height / max(width, 1)) > 3.0:
        reasons.append("wrong_crop")
    name_key = _label_key(str(path))
    if any(_label_key(token) in name_key for token in BAD_NAME_TOKENS):
        reasons.append("not_user_like")
    return reasons, width, height, contrast


def candidate_from_path(path: Path, *, target: str, expected_class: str, split: str) -> Candidate | None:
    profile = technical_profile(path)
    if profile is None:
        return None
    reasons, width, height, contrast = profile
    if "too_blurry_or_tiny" in reasons or "wrong_crop" in reasons:
        return None
    try:
        sha256 = image_hash(path)
    except (OSError, UnidentifiedImageError):
        return None
    crop, part = target_parts(target)
    warnings = tuple(reason for reason in reasons if reason != "too_blurry_or_tiny")
    return Candidate(
        path=path,
        target=target,
        expected_crop=crop,
        expected_part=part,
        expected_class=expected_class,
        split=split,
        sha256=sha256,
        width=width,
        height=height,
        contrast=contrast,
        semantic_warnings=warnings,
    )


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: str(path).lower(),
    )


def target_parts(target: str) -> tuple[str, str]:
    if "__" not in target:
        return target, "unknown"
    crop, part = target.split("__", 1)
    return crop, part


def collect_supported_candidates(
    dataset_root: Path,
    *,
    max_candidates_per_class: int = 48,
) -> dict[str, dict[str, list[Candidate]]]:
    candidates: dict[str, dict[str, list[Candidate]]] = defaultdict(lambda: defaultdict(list))
    for target_root in sorted(dataset_root.iterdir(), key=lambda path: path.name.lower()) if dataset_root.exists() else []:
        if not target_root.is_dir() or target_root.name not in SUPPORTED_TARGET_COUNTS:
            continue
        for split in SPLIT_PRIORITY:
            split_root = target_root / split
            if not split_root.is_dir():
                continue
            for class_dir in sorted(split_root.iterdir(), key=lambda path: _label_key(path.name)):
                if not class_dir.is_dir():
                    continue
                for image_path in iter_images(class_dir):
                    if len(candidates[target_root.name][class_dir.name]) >= max_candidates_per_class:
                        break
                    candidate = candidate_from_path(
                        image_path,
                        target=target_root.name,
                        expected_class=class_dir.name,
                        split=split,
                    )
                    if candidate is not None:
                        candidates[target_root.name][class_dir.name].append(candidate)
    return candidates


def collect_negative_candidates(root: Path, *, mode: str, max_candidates: int = 500) -> list[Candidate]:
    if mode == "non_plant":
        roots = [
            root / "data" / "router_eval" / "negatives" / "non_plant",
            root / "data" / "router_eval_holdout" / "negatives" / "non_plant",
            root / "data" / "router_eval" / "ambiguous",
            root / "data" / "router_eval_holdout" / "ambiguous",
        ]
        target = "non_plant"
        expected_class = "non_plant"
        preferred_tokens = NON_PLANT_TOKENS
    elif mode == "unknown_crop":
        roots = [
            root / "data" / "router_eval" / "negatives" / "off_crop",
            root / "data" / "router_eval_holdout" / "negatives" / "off_crop",
            root / "data" / "ood_dataset" / "final",
            root / DEFAULT_MATERIALIZED_DATASET_ROOT,
        ]
        target = "unknown_crop"
        expected_class = "unsupported_crop"
        preferred_tokens = UNKNOWN_CROP_TOKENS
    else:
        roots = [root / "data" / "router_eval" / "wrong_part", root / "data" / "router_eval_holdout" / "wrong_part"]
        target = mode
        expected_class = "unsupported_part"
        preferred_tokens = ()

    candidates: list[Candidate] = []
    for search_root in roots:
        for image_path in iter_images(search_root):
            image_name_key = _label_key(image_path.name)
            preferred = any(_label_key(token) in image_name_key for token in preferred_tokens)
            if mode == "non_plant" and not preferred:
                continue
            if mode == "unknown_crop":
                if not preferred:
                    continue
                if any(_label_key(token) in image_name_key for token in SUPPORTED_CROP_TOKENS):
                    continue
            if mode.endswith("__unknown_part"):
                crop = mode.split("__", 1)[0]
                if crop not in _label_key(str(image_path)):
                    continue
            split = "test" if preferred else "router_eval"
            candidate = candidate_from_path(image_path, target=target, expected_class=expected_class, split=split)
            if candidate is not None and (not candidate.semantic_warnings or mode == "non_plant"):
                candidates.append(candidate)
                if len(candidates) >= max_candidates:
                    return sorted(candidates, key=lambda candidate: (candidate.path.name.lower(), str(candidate.path).lower()))
    return sorted(candidates, key=lambda candidate: (candidate.path.name.lower(), str(candidate.path).lower()))


def download_commons_nonplant_images(output_dir: Path) -> list[Candidate]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[Candidate] = []
    for file_name in COMMONS_NON_PLANT_FILES:
        suffix = Path(file_name).suffix or ".jpg"
        output_path = output_dir / f"commons_nonplant_{_label_key(file_name)[:80]}{suffix.lower()}"
        if not output_path.exists():
            quoted = urllib.parse.quote(file_name, safe="")
            url = f"https://commons.wikimedia.org/wiki/Special:Redirect/file/{quoted}"
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "AADS-M2-demo-curation/1.0"})
                with urllib.request.urlopen(request, timeout=30) as response, output_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            except OSError:
                continue
        candidate = candidate_from_path(
            output_path,
            target="non_plant",
            expected_class="non_plant",
            split="test",
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def choose_unique(
    candidates: list[Candidate],
    *,
    count: int,
    used_hashes: set[str],
    prefer_test: bool = True,
) -> list[Candidate]:
    split_rank = {split: index for index, split in enumerate(SPLIT_PRIORITY)}
    if not prefer_test:
        split_rank = {split: 0 for split in SPLIT_PRIORITY}
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            split_rank.get(candidate.split, 99),
            len(candidate.semantic_warnings),
            -min(candidate.width, candidate.height),
            str(candidate.path).lower(),
        ),
    )
    selected: list[Candidate] = []
    for candidate in ranked:
        if candidate.sha256 in used_hashes:
            continue
        selected.append(candidate)
        used_hashes.add(candidate.sha256)
        if len(selected) >= count:
            return selected
    return selected


def build_supported_plan(
    candidates_by_target: dict[str, dict[str, list[Candidate]]],
    *,
    used_hashes: set[str],
    min_per_disease_class: int,
) -> tuple[dict[str, list[Candidate]], dict[str, dict[str, int]]]:
    selected: dict[str, list[Candidate]] = defaultdict(list)
    per_class_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for target, target_count in SUPPORTED_TARGET_COUNTS.items():
        classes = candidates_by_target.get(target, {})
        disease_classes = [class_name for class_name in sorted(classes, key=_label_key) if not is_healthy_class(class_name)]
        for class_name in disease_classes:
            picks = choose_unique(classes[class_name], count=min_per_disease_class, used_hashes=used_hashes)
            selected[target].extend(picks)
            per_class_counts[target][class_name] = len(picks)

        remaining = target_count - len(selected[target])
        if remaining < 0:
            raise RuntimeError(f"{target} needs {target_count} rows but disease minimums selected {len(selected[target])}")
        filler_pool: list[Candidate] = []
        for class_name in sorted(classes, key=lambda name: (not is_healthy_class(name), _label_key(name))):
            filler_pool.extend(classes[class_name])
        selected[target].extend(choose_unique(filler_pool, count=remaining, used_hashes=used_hashes, prefer_test=False))
        if len(selected[target]) != target_count:
            raise RuntimeError(f"{target} selected {len(selected[target])}/{target_count} clean unique images")
    return selected, per_class_counts


def canonical_name(image_id: str, target: str, source_path: Path) -> str:
    clean_target = target.replace("__", "_")
    return f"{image_id}_{clean_target}{source_path.suffix.lower()}"


def existing_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def materialize_package(
    *,
    package_dir: Path,
    original_rows: list[dict[str, str]],
    selected_by_target: dict[str, list[Candidate]],
    per_class_counts: dict[str, dict[str, int]],
    min_per_disease_class: int,
) -> dict[str, object]:
    image_dir = package_dir / "images"
    manifest_dir = package_dir / "manifests"
    image_dir.mkdir(parents=True, exist_ok=True)
    for old_image in image_dir.glob("demo_*"):
        if old_image.is_file() and old_image.suffix.lower() in IMAGE_SUFFIXES:
            old_image.unlink()

    queues = {target: list(candidates) for target, candidates in selected_by_target.items()}
    manifest_rows: list[dict[str, str]] = []
    qc_rows: list[dict[str, str]] = []
    final_hashes: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()

    for index, old_row in enumerate(original_rows):
        image_id = old_row["image_id"]
        target = old_row["expected_target"]
        if target not in queues or not queues[target]:
            raise RuntimeError(f"No selected candidate left for {image_id} target={target}")
        candidate = queues[target].pop(0)
        final_name = canonical_name(image_id, target, candidate.path)
        final_path = image_dir / final_name
        shutil.copy2(candidate.path, final_path)
        final_hashes[candidate.sha256] += 1
        target_counts[target] += 1
        bucket = USER_MIX[index % len(USER_MIX)]
        bucket_counts[bucket] += 1
        expected_behavior = (
            "known supported target; disease answer or review expected"
            if target in SUPPORTED_TARGET_COUNTS
            else "abstain/review expected for unsupported or non-plant input"
        )
        notes = (
            f"user-like M2 curated set; bucket={bucket}; source_split={candidate.split}; "
            f"source_class={candidate.expected_class}"
        )
        staged_source = f"staged_external:{final_path.as_posix()}"
        manifest_rows.append(
            {
                "image_id": image_id,
                "source": staged_source,
                "expected_target": target,
                "expected_crop": candidate.expected_crop,
                "expected_part": candidate.expected_part,
                "expected_class": candidate.expected_class,
                "expected_behavior": expected_behavior,
                "notes": notes,
                "original_source": str(candidate.path),
                "resolved_source_path": final_path.as_posix(),
                "disease_class": "" if target not in SUPPORTED_TARGET_COUNTS else candidate.expected_class,
                "user_like_bucket": bucket,
            }
        )
        qc_rows.append(
            {
                "image_id": image_id,
                "expected_target": target,
                "qc_status": "keep",
                "qc_reason": "",
                "replacement_action": "rebuilt_from_clean_pool",
                "source_path": str(candidate.path),
                "final_path": final_path.as_posix(),
                "sha256": candidate.sha256,
                "width": str(candidate.width),
                "height": str(candidate.height),
                "contrast": f"{candidate.contrast:.2f}",
                "expected_class": candidate.expected_class,
                "user_like_bucket": bucket,
                "manual_review_note": ";".join(candidate.semantic_warnings),
            }
        )

    duplicate_hashes = {sha for sha, count in final_hashes.items() if count > 1}
    if duplicate_hashes:
        raise RuntimeError(f"Final package still has duplicate hashes: {len(duplicate_hashes)}")

    manifest_fields = [
        "image_id",
        "source",
        "expected_target",
        "expected_crop",
        "expected_part",
        "expected_class",
        "expected_behavior",
        "notes",
        "original_source",
        "resolved_source_path",
        "disease_class",
        "user_like_bucket",
    ]
    qc_fields = [
        "image_id",
        "expected_target",
        "qc_status",
        "qc_reason",
        "replacement_action",
        "source_path",
        "final_path",
        "sha256",
        "width",
        "height",
        "contrast",
        "expected_class",
        "user_like_bucket",
        "manual_review_note",
    ]
    write_rows(manifest_dir / "m2_full_image_set_run_manifest.csv", manifest_rows, manifest_fields)
    write_rows(manifest_dir / "m2_full_image_set_manifest.csv", manifest_rows, manifest_fields)
    write_rows(manifest_dir / "m2_full_image_set_qc.csv", qc_rows, qc_fields)

    summary = {
        "schema_version": "v1_m2_user_like_curated_image_set",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(manifest_rows),
        "target_counts": dict(sorted(target_counts.items())),
        "duplicate_hash_count": 0,
        "qc_status_counts": {"keep": len(qc_rows)},
        "user_like_bucket_counts": dict(sorted(bucket_counts.items())),
        "min_per_disease_class_required": min_per_disease_class,
        "per_disease_class_counts": {
            target: dict(sorted(counts.items())) for target, counts in sorted(per_class_counts.items())
        },
    }
    (manifest_dir / "m2_full_image_set_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def build_contact_sheets(manifest_path: Path, output_dir: Path, *, thumb_size: int = 160, columns: int = 8) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = existing_rows(manifest_path)
    by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_target[row["expected_target"]].append(row)
    for target, target_rows in sorted(by_target.items()):
        thumbs: list[Image.Image] = []
        for row in target_rows[:80]:
            path = Path(row["resolved_source_path"])
            with Image.open(path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((thumb_size, thumb_size))
                canvas = Image.new("RGB", (thumb_size, thumb_size + 28), "white")
                x = (thumb_size - thumb.width) // 2
                y = (thumb_size - thumb.height) // 2
                canvas.paste(thumb, (x, y))
                thumbs.append(canvas)
        if not thumbs:
            continue
        rows_needed = (len(thumbs) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * thumb_size, rows_needed * (thumb_size + 28)), "white")
        for idx, thumb in enumerate(thumbs):
            x = (idx % columns) * thumb_size
            y = (idx // columns) * (thumb_size + 28)
            sheet.paste(thumb, (x, y))
        sheet.save(output_dir / f"{target}.jpg", quality=90)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_MATERIALIZED_DATASET_ROOT)
    parser.add_argument("--package-dir", type=Path, default=Path("docs/demo_assets/m2_full_image_set"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--min-per-disease-class", type=int, default=10)
    parser.add_argument("--contact-sheet-dir", type=Path, default=Path(".runtime_tmp/m2_user_like_contact_sheets"))
    parser.add_argument("--skip-contact-sheets", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    package_dir = args.package_dir
    manifest_path = package_dir / "manifests" / "m2_full_image_set_run_manifest.csv"
    original_rows = existing_rows(manifest_path)
    if len(original_rows) != 512:
        raise RuntimeError(f"Expected canonical 512-row manifest, found {len(original_rows)}")

    candidates = collect_supported_candidates(args.dataset_root)
    used_hashes: set[str] = set()
    selected_by_target, per_class_counts = build_supported_plan(
        candidates,
        used_hashes=used_hashes,
        min_per_disease_class=max(1, int(args.min_per_disease_class)),
    )
    for target, count in SPECIAL_TARGET_COUNTS.items():
        mode = target if target.endswith("__unknown_part") else target
        special = collect_negative_candidates(args.repo_root, mode=mode)
        if target == "non_plant":
            special = download_commons_nonplant_images(args.repo_root / ".runtime_tmp" / "m2_nonplant_internet") + special
        selected = choose_unique(special, count=count, used_hashes=used_hashes, prefer_test=True)
        if len(selected) != count:
            raise RuntimeError(f"{target} selected {len(selected)}/{count} clean unique negative images")
        selected_by_target[target] = selected

    summary = materialize_package(
        package_dir=package_dir,
        original_rows=original_rows,
        selected_by_target=selected_by_target,
        per_class_counts=per_class_counts,
        min_per_disease_class=max(1, int(args.min_per_disease_class)),
    )
    if not args.skip_contact_sheets:
        build_contact_sheets(manifest_path, args.contact_sheet_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
