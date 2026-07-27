#!/usr/bin/env python3
"""Stage a reviewed, labeled image source into hash-disjoint OE/OOD roles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class StagedRow:
    target: str
    disease_id: str
    role: str
    ood_type: str
    relative_path: str
    sha256: str
    source: str
    original_name: str
    source_url: str
    license: str
    review_status: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_tree_hashes(prefix: str) -> set[str]:
    output = subprocess.check_output(
        ["git", "-c", "core.quotePath=false", "ls-tree", "-r", "--name-only", "HEAD", prefix],
        text=True,
        encoding="utf-8",
        stderr=subprocess.DEVNULL,
    )
    image_paths = [
        relative_path
        for relative_path in output.splitlines()
        if Path(relative_path).suffix.casefold() in IMAGE_SUFFIXES
    ]
    hashes: set[str] = set()
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        for relative_path in image_paths:
            process.stdin.write(f"HEAD:{relative_path}\n".encode())
            process.stdin.flush()
            header = process.stdout.readline().decode("utf-8", errors="replace").strip().split()
            if len(header) < 3 or header[1] != "blob":
                raise RuntimeError(f"Could not read tracked image blob: {relative_path}")
            payload = process.stdout.read(int(header[2]))
            process.stdout.read(1)  # trailing newline emitted by cat-file --batch
            hashes.add(hashlib.sha256(payload).hexdigest())
    finally:
        process.stdin.close()
        process.wait()
    return hashes


def readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        # ``verify`` checks container structure but can miss a truncated pixel
        # stream. Re-open and decode the full image before admitting evidence.
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, ValueError):
        return False


def roboflow_source_group(path: Path) -> str:
    stem = path.name.casefold().split(".rf.", 1)[0]
    stem = re.sub(r"(?:[-_]copy(?:[-_]\d+)?[-_]*)+", "", stem)
    stem = re.sub(r"[-_]zoom[-_]\d+", "", stem)
    stem = re.sub(r"[-_]?(?:jpg|jpeg|png|webp)$", "", stem)
    stem = re.sub(r"[-_]?copy(?:[-_]?\d+)?$", "", stem)
    stem = re.sub(r"(?<=[a-z])_(?=\d)", "", stem)
    return re.sub(r"[-_]+", "_", stem).strip("_")


def role_sequence(count: int, *, oe_target: int, dev_target: int, test_target: int) -> list[str]:
    desired = {"oe_train": oe_target, "ood_dev": dev_target, "ood_test": test_target}
    total_desired = sum(desired.values())
    if total_desired <= 0:
        raise ValueError("at least one role target must be positive")
    if count >= total_desired:
        return [role for role, amount in desired.items() for _ in range(amount)]
    active_roles = {role: amount for role, amount in desired.items() if amount > 0}
    if count < len(active_roles):
        raise ValueError("not enough reviewed images to keep every requested role non-empty")
    raw = {role: count * amount / total_desired for role, amount in desired.items()}
    allocated = {role: (max(1, int(raw[role])) if amount > 0 else 0) for role, amount in desired.items()}
    while sum(allocated.values()) > count:
        role = max(active_roles, key=lambda item: (allocated[item] - raw[item], allocated[item]))
        if allocated[role] == 1:
            break
        allocated[role] -= 1
    while sum(allocated.values()) < count:
        role = max(active_roles, key=lambda item: (raw[item] - allocated[item], desired[item]))
        allocated[role] += 1
    return [role for role in ("oe_train", "ood_dev", "ood_test") for _ in range(allocated[role])]


def load_source_urls(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    urls: dict[str, str] = {}
    for row in rows:
        filename = str(row.get("filename") or "").strip()
        source_url = str(row.get("source_page_url") or row.get("image_url") or "").strip()
        if filename and source_url:
            urls[filename] = source_url
    return urls


def stage_source(
    source_dir: Path,
    output_root: Path,
    *,
    target: str,
    disease_id: str,
    source_name: str,
    source_url: str,
    license_name: str,
    excluded_names: set[str],
    source_urls: dict[str, str] | None = None,
    excluded_hashes: set[str] | None = None,
    included_names: set[str] | None = None,
    name_prefix: str = "",
    dedupe_roboflow: bool = False,
    recursive: bool = False,
    oe_target: int = 30,
    dev_target: int = 15,
    test_target: int = 15,
) -> tuple[list[StagedRow], dict[str, object]]:
    source_paths = source_dir.rglob("*") if recursive else source_dir.iterdir()
    candidates = [
        path
        for path in sorted(source_paths, key=lambda item: item.relative_to(source_dir).as_posix().casefold())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.name not in excluded_names
    ]
    if included_names is not None:
        candidates = [path for path in candidates if path.name in included_names]
    if name_prefix:
        candidates = [path for path in candidates if path.name.casefold().startswith(name_prefix.casefold())]
    source_group_duplicates: list[str] = []
    if dedupe_roboflow:
        by_source_group: dict[str, Path] = {}
        for path in candidates:
            group = roboflow_source_group(path)
            if group in by_source_group:
                source_group_duplicates.append(path.name)
            else:
                by_source_group[group] = path
        candidates = list(by_source_group.values())
    unreadable = [path.name for path in candidates if not readable_image(path)]
    candidates = [path for path in candidates if path.name not in unreadable]
    excluded_hashes = excluded_hashes or set()
    by_hash: dict[str, Path] = {}
    duplicates: list[str] = []
    existing_hash_matches: list[str] = []
    for path in candidates:
        digest = sha256_file(path)
        if digest in excluded_hashes:
            existing_hash_matches.append(path.name)
        elif digest in by_hash:
            duplicates.append(path.name)
        else:
            by_hash[digest] = path
    ordered = sorted(by_hash.items())
    roles = role_sequence(len(ordered), oe_target=oe_target, dev_target=dev_target, test_target=test_target)
    ordered = ordered[: len(roles)]
    rows: list[StagedRow] = []
    for (digest, source_path), role in zip(ordered, roles, strict=True):
        extension = source_path.suffix.lower()
        destination = output_root / target / role / disease_id / f"{digest[:16]}{extension}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        rows.append(
            StagedRow(
                target=target,
                disease_id=disease_id,
                role=role,
                ood_type="same_crop_unsupported_disease" if role.startswith("ood_") else "",
                relative_path=destination.relative_to(output_root).as_posix(),
                sha256=digest,
                source=source_name,
                original_name=source_path.name,
                source_url=(source_urls or {}).get(source_path.name, source_url),
                license=license_name,
                review_status="source_label_and_visual_review_accepted",
            )
        )
    counts = {role: sum(row.role == role for row in rows) for role in ("oe_train", "ood_dev", "ood_test")}
    summary: dict[str, object] = {
        "target": target,
        "disease_id": disease_id,
        "source": source_name,
        "source_url": source_url,
        "license": license_name,
        "accepted_count": len(rows),
        "role_counts": counts,
        "desired_role_counts": {"oe_train": oe_target, "ood_dev": dev_target, "ood_test": test_target},
        "target_complete": counts == {"oe_train": oe_target, "ood_dev": dev_target, "ood_test": test_target},
        "excluded_names": sorted(excluded_names),
        "unreadable_names": unreadable,
        "duplicate_names": duplicates,
        "source_group_duplicate_names": source_group_duplicates,
        "existing_hash_match_names": existing_hash_matches,
    }
    return rows, summary


def write_manifest(rows: list[StagedRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/adapter_ood_oe_evidence"))
    parser.add_argument("--target", required=True)
    parser.add_argument("--disease-id", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Candidate CSV mapping filename to source_page_url (falling back to image_url).",
    )
    parser.add_argument("--license", required=True)
    parser.add_argument("--oe-target", type=int, default=30)
    parser.add_argument("--dev-target", type=int, default=15)
    parser.add_argument("--test-target", type=int, default=15)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--name-prefix", default="", help="Accept only source filenames beginning with this prefix.")
    parser.add_argument(
        "--dedupe-roboflow",
        action="store_true",
        help="Keep one file per Roboflow source stem before the .rf. augmentation hash.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search image files below source_dir recursively.")
    parser.add_argument(
        "--include-list",
        type=Path,
        help="UTF-8 text file containing one explicitly accepted source filename per line.",
    )
    parser.add_argument(
        "--exclude-git-json",
        action="append",
        default=[],
        help="Git object in REV:path form; all nested sha256 values are excluded.",
    )
    parser.add_argument(
        "--exclude-git-tree",
        action="append",
        default=[],
        help="Tracked HEAD tree prefix whose image contents must be hash-excluded.",
    )
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        type=Path,
        help="CSV evidence manifest whose sha256 values must be excluded.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    excluded_hashes: set[str] = set()
    for object_name in args.exclude_git_json:
        payload = json.loads(subprocess.check_output(["git", "show", object_name], text=True, encoding="utf-8"))
        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                digest = value.get("sha256")
                if isinstance(digest, str) and len(digest) == 64:
                    excluded_hashes.add(digest.lower())
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    for prefix in args.exclude_git_tree:
        excluded_hashes.update(git_tree_hashes(prefix))
    for manifest_path in args.exclude_manifest:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                digest = str(row.get("sha256") or "").strip().lower()
                if len(digest) == 64:
                    excluded_hashes.add(digest)
    rows, summary = stage_source(
        args.source_dir,
        args.output_root,
        target=args.target,
        disease_id=args.disease_id,
        source_name=args.source_name,
        source_url=args.source_url,
        license_name=args.license,
        excluded_names=set(args.exclude),
        source_urls=load_source_urls(args.source_manifest) if args.source_manifest else None,
        excluded_hashes=excluded_hashes,
        included_names=(
            {
                line.strip()
                for line in args.include_list.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
            if args.include_list
            else None
        ),
        name_prefix=args.name_prefix,
        dedupe_roboflow=bool(args.dedupe_roboflow),
        recursive=bool(args.recursive),
        oe_target=args.oe_target,
        dev_target=args.dev_target,
        test_target=args.test_target,
    )
    write_manifest(rows, args.manifest)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: accepted={summary['accepted_count']} roles={summary['role_counts']}")
    print(f"target_complete={summary['target_complete']} manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
