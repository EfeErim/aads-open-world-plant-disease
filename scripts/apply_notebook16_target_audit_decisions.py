"""Apply reviewed Notebook 16 target-audit decisions with a dry-run default."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DECISIONS_CSV = Path("docs/ablation_results/dual_view_inference/tomato_leaf_missed_wrong_audit.csv")
DEFAULT_DATASET_ROOT = Path(".runtime_tmp/dataset_working/prepared_runtime_datasets/tomato__leaf")
DEFAULT_QUARANTINE_ROOT = Path(".runtime_tmp/notebook16_target_audit_quarantine/tomato__leaf")
SUPPORTED_NOOP_DECISIONS = {"", "keep", "accept", "ignore"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_inside(root: Path, path: Path, *, label: str) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_root not in [resolved_path, *resolved_path.parents]:
        raise ValueError(f"{label} escapes root: {path}")
    return resolved_path


def _source_path(dataset_root: Path, local_path: str) -> Path:
    path = Path(local_path)
    parts = path.parts
    if "tomato__leaf" in parts:
        index = parts.index("tomato__leaf")
        relative = Path(*parts[index + 1 :])
    else:
        relative = path
    return _resolve_inside(dataset_root, dataset_root / relative, label="source path")


def _quarantine_path(dataset_root: Path, quarantine_root: Path, source: Path) -> Path:
    relative = source.resolve().relative_to(dataset_root.resolve())
    return _resolve_inside(quarantine_root, quarantine_root / relative, label="quarantine path")


def _relabel_path(dataset_root: Path, source: Path, target_class: str) -> Path:
    if not target_class or "/" in target_class or "\\" in target_class:
        raise ValueError(f"Invalid relabel class: {target_class}")
    relative = source.resolve().relative_to(dataset_root.resolve())
    if len(relative.parts) < 3 or relative.parts[0] != "test":
        raise ValueError(f"Relabel only supports test/<class>/<file> paths: {relative}")
    destination = dataset_root / "test" / target_class / relative.name
    return _resolve_inside(dataset_root, destination, label="relabel destination")


def apply_audit_decisions(
    *,
    decisions_csv: Path,
    dataset_root: Path,
    quarantine_root: Path,
    packet_dir: Path | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    pending_review: list[dict[str, str]] = []
    rows, packet_decision_count = load_decision_rows(decisions_csv=decisions_csv, packet_dir=packet_dir)
    for row in rows:
        rank = str(row.get("rank") or "")
        decision = str(row.get("review_decision") or "").strip()
        normalized = decision.lower()
        if not normalized:
            pending_review.append({"rank": rank, "local_path": str(row.get("local_path") or "")})
            skipped.append({"rank": rank, "reason": "pending_review_decision"})
            continue
        if normalized in SUPPORTED_NOOP_DECISIONS:
            skipped.append({"rank": rank, "reason": "noop_decision"})
            continue
        local_path = str(row.get("local_path") or "").strip()
        if not local_path:
            skipped.append({"rank": rank, "reason": "missing_local_path"})
            continue
        source = _source_path(dataset_root, local_path)
        if not source.exists():
            skipped.append({"rank": rank, "reason": "source_missing", "source": str(source)})
            continue
        if normalized == "remove_from_test":
            destination = _quarantine_path(dataset_root, quarantine_root, source)
            action = "quarantine"
        elif normalized.startswith("relabel:"):
            target_class = decision.split(":", 1)[1].strip()
            destination = _relabel_path(dataset_root, source, target_class)
            action = "relabel"
        else:
            skipped.append({"rank": rank, "reason": f"unsupported_decision:{decision}"})
            continue

        if apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        actions.append(
            {
                "rank": rank,
                "action": action,
                "decision": decision,
                "source": str(source),
                "destination": str(destination),
                "applied": bool(apply),
            }
        )

    summary = {
        "schema_version": "v1_notebook16_target_audit_decision_apply",
        "created_at": _utc_now_iso(),
        "mode": "apply" if apply else "dry_run",
        "decisions_csv": str(decisions_csv),
        "dataset_root": str(dataset_root),
        "quarantine_root": str(quarantine_root),
        "packet_dir": str(packet_dir or ""),
        "packet_decision_count": packet_decision_count,
        "action_count": len(actions),
        "skipped_count": len(skipped),
        "pending_review_count": len(pending_review),
        "actions": actions,
        "skipped": skipped,
        "pending_review": pending_review,
    }
    return summary


def load_decision_rows(*, decisions_csv: Path, packet_dir: Path | None = None) -> tuple[list[dict[str, str]], int]:
    """Load full audit rows and overlay packet-level review decisions when provided."""
    with decisions_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not packet_dir:
        return rows, 0

    packet_decisions = _load_packet_decisions(packet_dir)
    merged_count = 0
    for row in rows:
        key = _decision_key(row)
        packet_row = packet_decisions.get(key)
        if not packet_row:
            continue
        decision = str(packet_row.get("review_decision") or "").strip()
        notes = str(packet_row.get("review_notes") or "").strip()
        if decision:
            row["review_decision"] = decision
            merged_count += 1
        if notes:
            row["review_notes"] = notes
    return rows, merged_count


def _load_packet_decisions(packet_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    packet_root = Path(packet_dir)
    if not packet_root.exists():
        return {}
    decisions: dict[tuple[str, str], dict[str, str]] = {}
    for csv_path in sorted(packet_root.glob("*/review_rows.csv")):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = _decision_key(row)
                if key == ("", ""):
                    continue
                decisions[key] = dict(row)
    return decisions


def _decision_key(row: dict[str, str]) -> tuple[str, str]:
    return (str(row.get("rank") or "").strip(), str(row.get("local_path") or "").strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions-csv", type=Path, default=DEFAULT_DECISIONS_CSV)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--quarantine-root", type=Path, default=DEFAULT_QUARANTINE_ROOT)
    parser.add_argument("--packet-dir", type=Path, help="Optional packet directory containing */review_rows.csv files.")
    parser.add_argument("--output", type=Path, default=Path(".runtime_tmp/notebook16_target_audit_decision_apply.json"))
    parser.add_argument("--apply", action="store_true", help="Move files. Without this flag, only writes a dry-run plan.")
    parser.add_argument("--require-reviewed", action="store_true", help="Return 2 when any review_decision is empty.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = apply_audit_decisions(
        decisions_csv=args.decisions_csv,
        dataset_root=args.dataset_root,
        quarantine_root=args.quarantine_root,
        packet_dir=args.packet_dir,
        apply=bool(args.apply),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": summary["mode"],
                "action_count": summary["action_count"],
                "skipped_count": summary["skipped_count"],
                "pending_review_count": summary["pending_review_count"],
                "packet_decision_count": summary["packet_decision_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    if args.require_reviewed and int(summary["pending_review_count"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
