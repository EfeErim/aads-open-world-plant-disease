"""Turn reviewed M2 hard-example audit rows into curation manifests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.csv_utils import write_csv_rows_with_order  # noqa: E402

NOOP_DECISIONS = {"keep", "accept", "ignore"}
MANIFEST_COLUMNS = (
    "image_id",
    "source",
    "resolved_image",
    "expected_target",
    "expected_crop",
    "expected_part",
    "expected_class",
    "corrected_crop",
    "corrected_part",
    "corrected_class",
    "prototype_target",
    "prototype_class_label",
    "predicted_crop",
    "predicted_part",
    "predicted_disease",
    "prototype_similarity",
    "prototype_margin",
    "reconcile_reason",
    "failure_bucket",
    "pass_fail",
    "review_decision",
    "prototype_quality",
    "adapter_training_quality",
    "review_notes",
)


def load_decision_rows(*, decisions_csv: Path, packet_dir: Path | None = None) -> tuple[list[dict[str, str]], int]:
    with decisions_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not packet_dir:
        return rows, 0
    packet_decisions = _load_packet_decisions(packet_dir)
    merged_count = 0
    for row in rows:
        packet_row = packet_decisions.get(_decision_key(row))
        if not packet_row:
            continue
        for field in (
            "review_decision",
            "corrected_crop",
            "corrected_part",
            "corrected_class",
            "prototype_quality",
            "adapter_training_quality",
            "review_notes",
        ):
            value = str(packet_row.get(field) or "").strip()
            if value:
                row[field] = value
        if str(packet_row.get("review_decision") or "").strip():
            merged_count += 1
    return rows, merged_count


def apply_hard_example_decisions(
    *,
    decisions_csv: Path,
    output_dir: Path,
    packet_dir: Path | None = None,
) -> dict[str, Any]:
    rows, packet_decision_count = load_decision_rows(decisions_csv=decisions_csv, packet_dir=packet_dir)
    buckets: dict[str, list[dict[str, str]]] = {
        "prototype_positive": [],
        "prototype_hard_negative": [],
        "adapter_hard_example": [],
        "excluded_ambiguous": [],
        "relabeled": [],
    }
    skipped: list[dict[str, str]] = []
    pending_review: list[dict[str, str]] = []

    for row in rows:
        decision = str(row.get("review_decision") or "").strip()
        normalized = decision.lower()
        if not normalized:
            pending_review.append(_row_ref(row))
            skipped.append({**_row_ref(row), "reason": "pending_review_decision"})
            continue
        if normalized in NOOP_DECISIONS:
            skipped.append({**_row_ref(row), "reason": "noop_decision"})
            continue
        manifest_row = _manifest_row(row)
        if normalized == "add_prototype_positive":
            buckets["prototype_positive"].append(manifest_row)
        elif normalized == "add_prototype_hard_negative":
            buckets["prototype_hard_negative"].append(manifest_row)
        elif normalized == "add_adapter_train":
            buckets["adapter_hard_example"].append(manifest_row)
        elif normalized == "exclude_ambiguous":
            buckets["excluded_ambiguous"].append(manifest_row)
        elif normalized.startswith("relabel:"):
            corrected_class = decision.split(":", 1)[1].strip()
            if not corrected_class:
                skipped.append({**_row_ref(row), "reason": "empty_relabel_class"})
                continue
            manifest_row["corrected_class"] = str(row.get("corrected_class") or corrected_class)
            buckets["relabeled"].append(manifest_row)
        else:
            skipped.append({**_row_ref(row), "reason": f"unsupported_decision:{decision}"})

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "prototype_positive_manifest": output_dir / "prototype_positive_manifest.csv",
        "prototype_hard_negative_manifest": output_dir / "prototype_hard_negative_manifest.csv",
        "adapter_hard_example_manifest": output_dir / "adapter_hard_example_manifest.csv",
        "excluded_ambiguous_rows": output_dir / "excluded_ambiguous_rows.csv",
        "relabel_manifest": output_dir / "relabel_manifest.csv",
        "relabeled_rows": output_dir / "relabeled_rows.csv",
    }
    write_csv_rows_with_order(
        outputs["prototype_positive_manifest"],
        buckets["prototype_positive"],
        preferred_headers=MANIFEST_COLUMNS,
        encoding="utf-8-sig",
    )
    write_csv_rows_with_order(
        outputs["prototype_hard_negative_manifest"],
        buckets["prototype_hard_negative"],
        preferred_headers=MANIFEST_COLUMNS,
        encoding="utf-8-sig",
    )
    write_csv_rows_with_order(
        outputs["adapter_hard_example_manifest"],
        buckets["adapter_hard_example"],
        preferred_headers=MANIFEST_COLUMNS,
        encoding="utf-8-sig",
    )
    write_csv_rows_with_order(
        outputs["excluded_ambiguous_rows"],
        buckets["excluded_ambiguous"],
        preferred_headers=MANIFEST_COLUMNS,
        encoding="utf-8-sig",
    )
    write_csv_rows_with_order(
        outputs["relabel_manifest"],
        buckets["relabeled"],
        preferred_headers=MANIFEST_COLUMNS,
        encoding="utf-8-sig",
    )
    write_csv_rows_with_order(
        outputs["relabeled_rows"],
        buckets["relabeled"],
        preferred_headers=MANIFEST_COLUMNS,
        encoding="utf-8-sig",
    )

    summary = {
        "schema_version": "v1_m2_hard_example_curation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dry_run_manifest_write",
        "decisions_csv": str(decisions_csv),
        "packet_dir": str(packet_dir or ""),
        "packet_decision_count": packet_decision_count,
        "output_dir": str(output_dir),
        "prototype_positive_count": len(buckets["prototype_positive"]),
        "prototype_hard_negative_count": len(buckets["prototype_hard_negative"]),
        "adapter_hard_example_count": len(buckets["adapter_hard_example"]),
        "excluded_ambiguous_count": len(buckets["excluded_ambiguous"]),
        "relabeled_count": len(buckets["relabeled"]),
        "pending_review_count": len(pending_review),
        "skipped_count": len(skipped),
        "outputs": {key: str(value) for key, value in outputs.items()},
        "skipped": skipped,
        "pending_review": pending_review,
    }
    summary_path = output_dir / "curation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["summary_output"] = str(summary_path)
    return summary


def _load_packet_decisions(packet_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not packet_dir.exists():
        return {}
    decisions: dict[tuple[str, str], dict[str, str]] = {}
    for csv_path in sorted(packet_dir.glob("*/review_rows.csv")):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = _decision_key(row)
                if key == ("", ""):
                    continue
                decisions[key] = dict(row)
    return decisions


def _decision_key(row: dict[str, str]) -> tuple[str, str]:
    return (str(row.get("rank") or "").strip(), str(row.get("image_id") or "").strip())


def _row_ref(row: dict[str, str]) -> dict[str, str]:
    return {"rank": str(row.get("rank") or ""), "image_id": str(row.get("image_id") or "")}


def _manifest_row(row: dict[str, str]) -> dict[str, str]:
    manifest = {column: str(row.get(column) or "") for column in MANIFEST_COLUMNS}
    if not manifest["corrected_crop"]:
        manifest["corrected_crop"] = manifest["expected_crop"]
    if not manifest["corrected_part"]:
        manifest["corrected_part"] = manifest["expected_part"]
    if not manifest["corrected_class"]:
        manifest["corrected_class"] = manifest["expected_class"]
    return manifest


def _default_output_dir(decisions_csv: Path) -> Path:
    run_id = decisions_csv.parent.name if decisions_csv.parent.name else "unknown_run"
    return Path("docs/demo_assets/prototype_curation") / run_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--packet-dir", type=Path, help="Optional packet directory containing */review_rows.csv files.")
    parser.add_argument("--output-dir", type=Path, help="Curation manifest output directory.")
    parser.add_argument(
        "--require-reviewed",
        action="store_true",
        help="Return 2 when any selected row has empty review_decision.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or _default_output_dir(args.decisions_csv)
    summary = apply_hard_example_decisions(
        decisions_csv=args.decisions_csv,
        packet_dir=args.packet_dir,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "mode": summary["mode"],
                "output_dir": summary["output_dir"],
                "prototype_positive_count": summary["prototype_positive_count"],
                "prototype_hard_negative_count": summary["prototype_hard_negative_count"],
                "adapter_hard_example_count": summary["adapter_hard_example_count"],
                "excluded_ambiguous_count": summary["excluded_ambiguous_count"],
                "relabeled_count": summary["relabeled_count"],
                "pending_review_count": summary["pending_review_count"],
                "summary_output": summary["summary_output"],
            },
            sort_keys=True,
        )
    )
    if args.require_reviewed and int(summary["pending_review_count"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
