"""Fill conservative draft decisions for an answered-wrong M2 disease-class audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def review_answered_wrong_audit(csv_path: Path, *, packet_dir: Path | None = None) -> dict[str, object]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []

    for row in rows:
        decision, prototype_quality, adapter_quality, notes = _decision_for_row(row)
        row["review_decision"] = decision
        row["prototype_quality"] = prototype_quality
        row["adapter_training_quality"] = adapter_quality
        row["review_notes"] = notes

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    packet_row_count = _update_packet_rows(rows, packet_dir=packet_dir) if packet_dir else 0
    return {
        "csv": str(csv_path),
        "row_count": len(rows),
        "packet_row_count": packet_row_count,
        "decisions": dict(sorted(Counter(row["review_decision"] for row in rows).items())),
        "adapter_training_quality": dict(sorted(Counter(row["adapter_training_quality"] for row in rows).items())),
        "prototype_quality": dict(sorted(Counter(row["prototype_quality"] for row in rows).items())),
    }


def _decision_for_row(row: dict[str, str]) -> tuple[str, str, str, str]:
    expected_target = str(row.get("expected_target") or "").strip()
    expected_crop = str(row.get("expected_crop") or "").strip()
    expected_part = str(row.get("expected_part") or "").strip()
    expected_class = str(row.get("expected_class") or "").strip()
    predicted_crop = str(row.get("predicted_crop") or "").strip()
    predicted_part = str(row.get("predicted_part") or "").strip()
    predicted_disease = str(row.get("predicted_disease") or "").strip()

    if not expected_target or not expected_class or not predicted_disease:
        return (
            "exclude_ambiguous",
            "not_for_prototype",
            "not_for_adapter_training",
            "missing target, expected class, or predicted disease; keep out of training",
        )
    if expected_crop == predicted_crop and expected_part == predicted_part:
        return (
            "add_adapter_train",
            "not_for_prototype",
            "crop_part_correct_hard_example",
            "draft decision: crop/part are correct and disease class is wrong; use as adapter hard example",
        )
    return (
        "exclude_ambiguous",
        "not_for_prototype",
        "not_for_adapter_training",
        "draft decision: crop/part mismatch or incomplete route; do not train adapter from this row",
    )


def _update_packet_rows(rows: list[dict[str, str]], *, packet_dir: Path) -> int:
    if not packet_dir.exists():
        return 0
    row_by_key = {
        (str(row.get("rank") or "").strip(), str(row.get("image_id") or "").strip()): row
        for row in rows
    }
    updated = 0
    for csv_path in sorted(packet_dir.glob("*/review_rows.csv")):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            packet_rows = list(csv.DictReader(handle))
            fieldnames = list(packet_rows[0].keys()) if packet_rows else []
        for packet_row in packet_rows:
            key = (str(packet_row.get("rank") or "").strip(), str(packet_row.get("image_id") or "").strip())
            source_row = row_by_key.get(key)
            if not source_row:
                continue
            for field in (
                "review_decision",
                "prototype_quality",
                "adapter_training_quality",
                "review_notes",
            ):
                packet_row[field] = str(source_row.get(field) or "")
            updated += 1
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(packet_rows)
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="answered_wrong_audit.csv to update in place.")
    parser.add_argument(
        "--packet-dir",
        type=Path,
        help="Optional packet directory containing */review_rows.csv files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        json.dumps(
            review_answered_wrong_audit(args.csv, packet_dir=args.packet_dir),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
