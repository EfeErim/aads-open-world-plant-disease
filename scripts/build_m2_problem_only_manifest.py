#!/usr/bin/env python3
"""Build a runnable M2 manifest from a reviewed hard-example audit CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.csv_utils import write_csv_rows_with_order

MANIFEST_COLUMNS = (
    "image_id",
    "source",
    "expected_target",
    "expected_crop",
    "expected_part",
    "expected_class",
    "expected_behavior",
    "notes",
    "original_source",
)


def build_problem_only_manifest(*, audit_csv: Path, output: Path) -> dict[str, object]:
    with audit_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]

    manifest_rows = []
    seen_ids: set[str] = set()
    for row in rows:
        image_id = str(row.get("image_id") or "").strip()
        source = str(row.get("source") or "").strip()
        expected_target = str(row.get("expected_target") or "").strip()
        if not image_id or not source or not expected_target or image_id in seen_ids:
            continue
        seen_ids.add(image_id)
        reason = str(row.get("reconcile_reason") or "").strip()
        review_decision = str(row.get("review_decision") or "").strip()
        notes = str(row.get("review_notes") or "").strip()
        manifest_rows.append(
            {
                "image_id": image_id,
                "source": source,
                "expected_target": expected_target,
                "expected_crop": str(row.get("expected_crop") or "").strip(),
                "expected_part": str(row.get("expected_part") or "").strip(),
                "expected_class": str(row.get("expected_class") or "").strip(),
                "expected_behavior": "router/prototype hard example; disease answer or review expected",
                "notes": "; ".join(part for part in (reason, review_decision, notes) if part),
                "original_source": str(row.get("resolved_image") or "").strip(),
            }
        )

    if not manifest_rows:
        raise ValueError(f"No runnable rows found in {audit_csv}")

    write_csv_rows_with_order(output, manifest_rows, preferred_headers=MANIFEST_COLUMNS, encoding="utf-8-sig")
    return {
        "audit_csv": str(audit_csv),
        "output": str(output),
        "row_count": len(manifest_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(build_problem_only_manifest(audit_csv=args.audit_csv, output=args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
