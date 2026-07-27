#!/usr/bin/env python3
"""Human-review audit for OOD/OE leakage and slice quality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_release_runtime import DEFAULT_MATERIALIZED_DATASET_ROOT  # noqa: E402
from src.shared.csv_utils import write_csv_rows_with_order  # noqa: E402
from src.shared.hash_utils import sha256_file  # noqa: E402
from src.shared.json_utils import write_json  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ID_ROLES = {"continual", "val", "test"}
AUDIT_ROLES = ("continual", "val", "test", "ood", "oe")
DEFAULT_NEAR_DUPLICATE_HAMMING = 6
DEFAULT_NEAR_DUPLICATE_AVERAGE_HAMMING = 4
DEFAULT_NEAR_DUPLICATE_RMSE = 24.0
WINDOWS_SAFE_QUARANTINE_PATH_LENGTH = 220


@dataclass(frozen=True)
class ImageRecord:
    role: str
    relative_path: str
    absolute_path: str
    top_slice: str
    sha256: str
    dhash: str
    width: int
    height: int
    readable: bool
    error: str = ""


@dataclass(frozen=True)
class Issue:
    issue_id: str
    issue_type: str
    severity: str
    role_a: str
    path_a: str
    role_b: str = ""
    path_b: str = ""
    score: str = ""
    reason: str = ""
    suggested_action: str = ""
    target_path: str = ""
    decision: str = ""
    notes: str = ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_images(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _dhash(image: Image.Image, *, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.tobytes())
    bits: List[str] = []
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            bits.append("1" if pixels[offset + col] > pixels[offset + col + 1] else "0")
    return f"{int(''.join(bits), 2):0{hash_size * hash_size // 4}x}"


def _ahash(image: Image.Image, *, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.tobytes())
    mean = sum(pixels) / max(len(pixels), 1)
    bits = ["1" if pixel >= mean else "0" for pixel in pixels]
    return f"{int(''.join(bits), 2):0{hash_size * hash_size // 4}x}"


def _thumbnail_rmse(left_path: str, right_path: str, *, size: int = 32) -> float:
    with Image.open(left_path) as left_image, Image.open(right_path) as right_image:
        left = left_image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        right = right_image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        squared_error = sum((int(a) - int(b)) ** 2 for a, b in zip(left.tobytes(), right.tobytes()))
    return (squared_error / float(size * size)) ** 0.5


def _secondary_near_duplicate_scores(left: ImageRecord, right: ImageRecord) -> Tuple[int, float]:
    with Image.open(left.absolute_path) as left_image, Image.open(right.absolute_path) as right_image:
        average_hamming = _hamming_hex(_ahash(left_image), _ahash(right_image))
    return average_hamming, _thumbnail_rmse(left.absolute_path, right.absolute_path)


def _hamming_hex(a: str, b: str) -> int:
    if not a or not b:
        return 10**9
    return int(int(a, 16) ^ int(b, 16)).bit_count()


def _top_slice(role_root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(role_root)
    except ValueError:
        return "unlabeled"
    parts = rel.parts
    if len(parts) <= 1:
        return "unlabeled"
    return str(parts[0] or "unlabeled")


def collect_records(dataset_root: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for role in AUDIT_ROLES:
        role_root = dataset_root / role
        for path in _iter_images(role_root):
            relative_path = path.relative_to(dataset_root).as_posix()
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    digest = _dhash(image)
                readable = True
                error = ""
            except Exception as exc:
                width = 0
                height = 0
                digest = ""
                readable = False
                error = str(exc)
            records.append(
                ImageRecord(
                    role=role,
                    relative_path=relative_path,
                    absolute_path=str(path.resolve()),
                    top_slice=_top_slice(role_root, path),
                    sha256=sha256_file(path),
                    dhash=digest,
                    width=int(width),
                    height=int(height),
                    readable=readable,
                    error=error,
                )
            )
    return records


def _issue_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:05d}"


def find_exact_overlap_issues(records: Sequence[ImageRecord]) -> List[Issue]:
    by_hash: Dict[str, List[ImageRecord]] = {}
    for record in records:
        by_hash.setdefault(record.sha256, []).append(record)

    issues: List[Issue] = []
    for digest, group in sorted(by_hash.items(), key=lambda item: item[0]):
        if len(group) <= 1:
            continue
        ordered = sorted(group, key=lambda item: item.relative_path)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                roles = {left.role, right.role}
                if len(roles) == 1:
                    severity = "warning"
                elif "ood" in roles and "oe" in roles:
                    severity = "blocker"
                elif roles & ID_ROLES and roles & {"ood", "oe"}:
                    severity = "blocker"
                else:
                    severity = "warning"
                target = right.relative_path
                if right.role == "ood" and left.role == "oe":
                    target = left.relative_path
                issues.append(
                    Issue(
                        issue_id=_issue_id("exact_overlap", len(issues) + 1),
                        issue_type="exact_hash_overlap",
                        severity=severity,
                        role_a=left.role,
                        path_a=left.relative_path,
                        role_b=right.role,
                        path_b=right.relative_path,
                        score=digest,
                        reason="same SHA-256 image appears in more than one audit role",
                        suggested_action="quarantine one copy after human review",
                        target_path=target,
                    )
                )
    return issues


def find_near_duplicate_issues(
    records: Sequence[ImageRecord],
    *,
    max_hamming: int,
) -> List[Issue]:
    readable = [record for record in records if record.readable and record.dhash]
    issues: List[Issue] = []
    for left_index, left in enumerate(readable):
        for right in readable[left_index + 1 :]:
            if left.sha256 == right.sha256:
                continue
            roles = {left.role, right.role}
            if not roles & {"ood", "oe"}:
                continue
            if len(roles) == 1:
                continue
            distance = _hamming_hex(left.dhash, right.dhash)
            if distance > int(max_hamming):
                continue
            average_hamming, thumbnail_rmse = _secondary_near_duplicate_scores(left, right)
            if average_hamming > DEFAULT_NEAR_DUPLICATE_AVERAGE_HAMMING:
                continue
            if thumbnail_rmse > DEFAULT_NEAR_DUPLICATE_RMSE:
                continue
            if "ood" in roles and "oe" in roles:
                severity = "review"
            elif roles & ID_ROLES and roles & {"ood", "oe"}:
                severity = "review"
            else:
                severity = "info"
            issues.append(
                Issue(
                    issue_id=_issue_id("near_duplicate", len(issues) + 1),
                    issue_type="near_duplicate_perceptual_hash",
                    severity=severity,
                    role_a=left.role,
                    path_a=left.relative_path,
                    role_b=right.role,
                    path_b=right.relative_path,
                    score=(
                        f"dhash={distance};ahash={average_hamming};"
                        f"thumbnail_rmse={thumbnail_rmse:.3f}"
                    ),
                    reason=(
                        f"dHash <= {int(max_hamming)}, aHash <= "
                        f"{DEFAULT_NEAR_DUPLICATE_AVERAGE_HAMMING}, and thumbnail RMSE <= "
                        f"{DEFAULT_NEAR_DUPLICATE_RMSE:.1f}"
                    ),
                    suggested_action="review visually; quarantine one copy if it is the same source image",
                    target_path=right.relative_path,
                )
            )
    return issues


def _tokenize(value: str) -> set[str]:
    lowered = value.lower().replace("\\", "/")
    for char in "-_.()/":
        lowered = lowered.replace(char, " ")
    return {token for token in lowered.split() if token}


def infer_dataset_crop_part(dataset_key: str) -> Tuple[str, str]:
    parts = str(dataset_key).split("__", 1)
    crop = parts[0].strip().lower()
    part = parts[1].strip().lower() if len(parts) > 1 else ""
    return crop, part


def find_semantic_slice_issues(records: Sequence[ImageRecord], *, dataset_key: str) -> List[Issue]:
    crop, part = infer_dataset_crop_part(dataset_key)
    issues: List[Issue] = []
    plant_tokens = {"plant", "leaf", "leaves", "fruit", "fruits", "stem", "crop", "disease", crop}
    for record in records:
        if record.role not in {"ood", "oe"}:
            continue
        slice_tokens = _tokenize(record.top_slice)
        relative_parts = Path(record.relative_path).parts
        payload = "/".join(relative_parts[2:]) if len(relative_parts) > 2 else Path(record.relative_path).name
        payload_tokens = _tokenize(payload) - slice_tokens
        expected_opposite_part_slice = (
            {"same", "crop", "leaf", "negatives"} <= slice_tokens
            or {"same", "crop", "fruit", "negatives"} <= slice_tokens
            or {"same", "crop", "disease", "negatives"} <= slice_tokens
            or {"mixed", "part", "negatives"} <= slice_tokens
            or {"scene", "context", "leak", "check"} <= slice_tokens
        )
        reason = ""
        severity = "review"
        if (
            part == "fruit"
            and crop in payload_tokens
            and ({"leaf", "leaves", "foliage"} & payload_tokens)
            and not expected_opposite_part_slice
        ):
            reason = "fruit adapter pool item has leaf-like path or slice tokens"
        elif (
            part == "leaf"
            and crop in payload_tokens
            and ({"fruit", "fruits"} & payload_tokens)
            and not expected_opposite_part_slice
        ):
            reason = "leaf adapter pool item has fruit-like path or slice tokens"
        elif "non" in slice_tokens and "plant" in slice_tokens and payload_tokens & plant_tokens:
            reason = "non-plant slice has plant/crop-like path tokens"
        elif "off" in slice_tokens and "crop" in slice_tokens and crop in payload_tokens:
            reason = "off-crop slice has same-crop path tokens"
        if not reason:
            continue
        issues.append(
            Issue(
                issue_id=_issue_id("semantic_slice", len(issues) + 1),
                issue_type="semantic_slice_suspicion",
                severity=severity,
                role_a=record.role,
                path_a=record.relative_path,
                score=record.top_slice,
                reason=reason,
                suggested_action="review visually; fix slice folder only when the reviewer confirms mismatch",
                target_path=record.relative_path,
            )
        )
    return issues


def summarize(records: Sequence[ImageRecord], issues: Sequence[Issue], *, dataset_root: Path, dataset_key: str) -> Dict[str, Any]:
    counts_by_role: Dict[str, int] = {}
    for record in records:
        counts_by_role[record.role] = counts_by_role.get(record.role, 0) + 1
    counts_by_type: Dict[str, int] = {}
    counts_by_severity: Dict[str, int] = {}
    for issue in issues:
        counts_by_type[issue.issue_type] = counts_by_type.get(issue.issue_type, 0) + 1
        counts_by_severity[issue.severity] = counts_by_severity.get(issue.severity, 0) + 1
    return {
        "schema_version": "v1_ood_oe_quality_audit",
        "created_at": _utc_now_iso(),
        "dataset_key": dataset_key,
        "dataset_root": str(dataset_root.resolve()),
        "record_count": len(records),
        "counts_by_role": counts_by_role,
        "issue_count": len(issues),
        "counts_by_issue_type": counts_by_type,
        "counts_by_severity": counts_by_severity,
        "human_loop": {
            "review_csv": "review_decisions.csv",
            "allowed_decisions": ["accept", "ignore", "quarantine"],
            "apply_mode": "only rows with decision=quarantine are moved; no files are deleted",
        },
    }


def write_markdown_report(output_path: Path, summary: Mapping[str, Any], issues: Sequence[Issue]) -> None:
    lines = [
        "# OOD/OE Quality Audit",
        "",
        f"- dataset: `{summary.get('dataset_key')}`",
        f"- records: {summary.get('record_count')}",
        f"- issues: {summary.get('issue_count')}",
        f"- roles: `{json.dumps(summary.get('counts_by_role', {}), sort_keys=True)}`",
        f"- issue types: `{json.dumps(summary.get('counts_by_issue_type', {}), sort_keys=True)}`",
        "",
        "## Top Review Items",
        "",
    ]
    for issue in list(issues)[:50]:
        lines.append(
            f"- `{issue.severity}` `{issue.issue_type}` `{issue.path_a}`"
            + (f" vs `{issue.path_b}`" if issue.path_b else "")
            + f" - {issue.reason}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    *,
    dataset_root: Path,
    dataset_key: str,
    output_dir: Path,
    near_duplicate_hamming: int,
) -> Dict[str, Any]:
    records = collect_records(dataset_root)
    issues: List[Issue] = []
    issues.extend(find_exact_overlap_issues(records))
    issues.extend(find_near_duplicate_issues(records, max_hamming=near_duplicate_hamming))
    issues.extend(find_semantic_slice_issues(records, dataset_key=dataset_key))
    issues = sorted(issues, key=lambda item: (item.severity != "blocker", item.issue_type, item.path_a, item.path_b))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records, issues, dataset_root=dataset_root, dataset_key=dataset_key)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "records.json", [asdict(record) for record in records])
    write_json(output_dir / "issues.json", [asdict(issue) for issue in issues])
    fieldnames = list(Issue.__dataclass_fields__.keys())
    write_csv_rows_with_order(
        output_dir / "review_decisions.csv",
        [asdict(issue) for issue in issues],
        preferred_headers=fieldnames,
    )
    write_markdown_report(output_dir / "review_report.md", summary, issues)
    return {"summary": summary, "issues": [asdict(issue) for issue in issues]}


def discover_prepared_datasets(prepared_root: Path) -> List[Path]:
    root = Path(prepared_root).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"Prepared runtime dataset root is not a directory: {root}")
    datasets: List[Path] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        if (child / "ood").is_dir() or (child / "oe").is_dir():
            datasets.append(child)
    return datasets


def run_batch_audit(
    *,
    prepared_root: Path,
    output_dir: Path,
    near_duplicate_hamming: int,
) -> Dict[str, Any]:
    dataset_roots = discover_prepared_datasets(prepared_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for dataset_root in dataset_roots:
        dataset_key = dataset_root.name
        dataset_output = output_dir / dataset_key
        result = run_audit(
            dataset_root=dataset_root,
            dataset_key=dataset_key,
            output_dir=dataset_output,
            near_duplicate_hamming=near_duplicate_hamming,
        )
        summary = dict(result["summary"])
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_root": str(dataset_root.resolve()),
                "output_dir": str(dataset_output.resolve()),
                "record_count": int(summary.get("record_count", 0) or 0),
                "issue_count": int(summary.get("issue_count", 0) or 0),
                "blocker_count": int(dict(summary.get("counts_by_severity", {})).get("blocker", 0) or 0),
                "review_count": int(dict(summary.get("counts_by_severity", {})).get("review", 0) or 0),
                "info_count": int(dict(summary.get("counts_by_severity", {})).get("info", 0) or 0),
                "exact_overlap_count": int(dict(summary.get("counts_by_issue_type", {})).get("exact_hash_overlap", 0) or 0),
                "near_duplicate_count": int(
                    dict(summary.get("counts_by_issue_type", {})).get("near_duplicate_perceptual_hash", 0) or 0
                ),
                "semantic_suspicion_count": int(
                    dict(summary.get("counts_by_issue_type", {})).get("semantic_slice_suspicion", 0) or 0
                ),
            }
        )
    batch_summary = {
        "schema_version": "v1_ood_oe_quality_batch_audit",
        "created_at": _utc_now_iso(),
        "prepared_root": str(Path(prepared_root).expanduser().resolve()),
        "output_dir": str(output_dir.resolve()),
        "dataset_count": len(rows),
        "total_issue_count": sum(int(row["issue_count"]) for row in rows),
        "total_blocker_count": sum(int(row["blocker_count"]) for row in rows),
        "datasets": rows,
    }
    write_json(output_dir / "batch_summary.json", batch_summary)
    write_csv_rows_with_order(
        output_dir / "batch_summary.csv",
        rows,
        preferred_headers=[
            "dataset_key",
            "record_count",
            "issue_count",
            "blocker_count",
            "review_count",
            "info_count",
            "exact_overlap_count",
            "near_duplicate_count",
            "semantic_suspicion_count",
            "output_dir",
            "dataset_root",
        ],
    )
    return batch_summary


def _safe_quarantine_path(dataset_root: Path, quarantine_root: Path, target_relative: str) -> Path:
    source = (dataset_root / target_relative).resolve()
    dataset_resolved = dataset_root.resolve()
    if dataset_resolved not in [source, *source.parents]:
        raise ValueError(f"Decision target escapes dataset root: {target_relative}")
    destination = (quarantine_root / target_relative).resolve()
    quarantine_resolved = quarantine_root.resolve()
    if quarantine_resolved not in [destination, *destination.parents]:
        raise ValueError(f"Decision target escapes quarantine root: {target_relative}")
    if len(str(destination)) > WINDOWS_SAFE_QUARANTINE_PATH_LENGTH:
        relative = Path(target_relative)
        role = relative.parts[0] if relative.parts else "file"
        digest = hashlib.sha256(target_relative.encode("utf-8")).hexdigest()[:16]
        suffix = relative.suffix.lower()
        destination = (quarantine_resolved / role / f"{digest}{suffix}").resolve()
    return destination


def _remove_quarantined_manifest_rows(dataset_root: Path, target_relatives: Sequence[str]) -> Dict[str, int]:
    normalized_targets = {Path(value).as_posix() for value in target_relatives if str(value).strip()}
    removed_split_rows = 0
    split_manifest_path = dataset_root / "split_manifest.json"
    if normalized_targets and split_manifest_path.is_file():
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        rows = list(split_manifest.get("rows") or [])
        kept_rows = [
            row
            for row in rows
            if Path(str(row.get("runtime_relative_path") or "")).as_posix() not in normalized_targets
        ]
        removed_split_rows = len(rows) - len(kept_rows)
        if removed_split_rows:
            split_manifest["rows"] = kept_rows
            write_json(split_manifest_path, split_manifest)

    removed_evidence_rows = 0
    evidence_manifest_path = dataset_root.parent / "adapter_ood_oe_evidence_manifest.csv"
    if normalized_targets and evidence_manifest_path.is_file():
        with evidence_manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            evidence_rows = list(reader)
        expected_paths = {
            f"data/prepared_runtime_datasets/{dataset_root.name}/{target}" for target in normalized_targets
        }
        kept_evidence_rows = [
            row
            for row in evidence_rows
            if str(row.get("relative_path") or "").replace("\\", "/") not in expected_paths
        ]
        removed_evidence_rows = len(evidence_rows) - len(kept_evidence_rows)
        if removed_evidence_rows:
            write_csv_rows_with_order(
                evidence_manifest_path,
                kept_evidence_rows,
                preferred_headers=fieldnames,
            )

    return {
        "removed_split_manifest_rows": removed_split_rows,
        "removed_evidence_manifest_rows": removed_evidence_rows,
    }


def apply_review_decisions(*, dataset_root: Path, decisions_csv: Path, quarantine_root: Path) -> Dict[str, Any]:
    applied: List[Dict[str, str]] = []
    already_quarantined: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []
    with decisions_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            decision = str(row.get("decision", "") or "").strip().lower()
            target_relative = str(row.get("target_path", "") or "").strip()
            if decision != "quarantine":
                skipped.append({"issue_id": str(row.get("issue_id", "")), "reason": "decision_not_quarantine"})
                continue
            if not target_relative:
                skipped.append({"issue_id": str(row.get("issue_id", "")), "reason": "missing_target_path"})
                continue
            source = (dataset_root / target_relative).resolve()
            if not source.exists():
                destination = _safe_quarantine_path(dataset_root, quarantine_root, target_relative)
                if destination.is_file():
                    already_quarantined.append(
                        {
                            "issue_id": str(row.get("issue_id", "")),
                            "source": str(source),
                            "destination": str(destination),
                            "target_relative": target_relative,
                        }
                    )
                    continue
                skipped.append({"issue_id": str(row.get("issue_id", "")), "reason": "target_missing"})
                continue
            destination = _safe_quarantine_path(dataset_root, quarantine_root, target_relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            applied.append(
                {
                    "issue_id": str(row.get("issue_id", "")),
                    "source": str(source),
                    "destination": str(destination),
                    "target_relative": target_relative,
                }
            )
    manifest_sync = _remove_quarantined_manifest_rows(
        dataset_root,
        [item["target_relative"] for item in [*applied, *already_quarantined]],
    )
    summary = {
        "schema_version": "v1_ood_oe_quality_decision_apply",
        "created_at": _utc_now_iso(),
        "dataset_root": str(dataset_root.resolve()),
        "decisions_csv": str(decisions_csv.resolve()),
        "quarantine_root": str(quarantine_root.resolve()),
        "applied_count": len(applied),
        "already_quarantined_count": len(already_quarantined),
        "skipped_count": len(skipped),
        "applied": applied,
        "already_quarantined": already_quarantined,
        "skipped": skipped,
        "manifest_sync": manifest_sync,
    }
    write_json(quarantine_root / "decision_apply_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit OOD/OE exact, near-duplicate, and slice-quality risks.")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--prepared-root", default=DEFAULT_MATERIALIZED_DATASET_ROOT, type=Path)
    parser.add_argument("--all", action="store_true", help="Audit every prepared dataset with ood/ or oe/.")
    parser.add_argument("--dataset-key", default="")
    parser.add_argument("--output-dir", default=Path(".runtime_tmp/ood_oe_quality_audit"), type=Path)
    parser.add_argument("--near-duplicate-hamming", default=DEFAULT_NEAR_DUPLICATE_HAMMING, type=int)
    parser.add_argument("--fail-on-exact-overlap", action="store_true")
    parser.add_argument("--fail-on-near-duplicate", action="store_true")
    parser.add_argument("--fail-on-suspicious-slice", action="store_true")
    parser.add_argument("--apply-decisions", type=Path)
    parser.add_argument("--quarantine-root", default=Path(".runtime_tmp/ood_oe_quality_quarantine"), type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.all:
        summary = run_batch_audit(
            prepared_root=Path(args.prepared_root).expanduser(),
            output_dir=Path(args.output_dir).expanduser(),
            near_duplicate_hamming=int(args.near_duplicate_hamming),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.fail_on_exact_overlap and int(summary.get("total_blocker_count", 0) or 0) > 0:
            return 2
        return 0

    if args.dataset_root is None:
        raise SystemExit("--dataset-root is required unless --all is used.")
    dataset_root = Path(args.dataset_root).expanduser()
    dataset_key = str(args.dataset_key or dataset_root.name)
    if args.apply_decisions:
        summary = apply_review_decisions(
            dataset_root=dataset_root,
            decisions_csv=Path(args.apply_decisions).expanduser(),
            quarantine_root=Path(args.quarantine_root).expanduser(),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    result = run_audit(
        dataset_root=dataset_root,
        dataset_key=dataset_key,
        output_dir=Path(args.output_dir).expanduser(),
        near_duplicate_hamming=int(args.near_duplicate_hamming),
    )
    summary = result["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    issues = result["issues"]
    if args.fail_on_exact_overlap and any(item["issue_type"] == "exact_hash_overlap" for item in issues):
        return 2
    if args.fail_on_near_duplicate and any(item["issue_type"] == "near_duplicate_perceptual_hash" for item in issues):
        return 3
    if args.fail_on_suspicious_slice and any(item["issue_type"] == "semantic_slice_suspicion" for item in issues):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
