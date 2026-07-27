"""Fill conservative review decisions for a router-only M2 hard-example audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

PRIORITY_TARGETS = {
    "apricot__fruit",
    "apricot__leaf",
    "grape__fruit",
    "grape__leaf",
    "tomato__leaf",
    "tomato__fruit",
    "strawberry__fruit",
    "strawberry__leaf",
}
POSITIVE_REASONS = {"prototype_evidence_weak", "prototype_policy_not_calibrated"}
FRUIT_PART_CONFLICT_TARGETS = {"strawberry__fruit", "grape__fruit"}


def review_router_only_audit(csv_path: Path) -> dict[str, object]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []

    for row in rows:
        decision, quality, notes = _decision_for_row(row)
        row["review_decision"] = decision
        row["prototype_quality"] = quality
        row["adapter_training_quality"] = "deferred_router_only"
        row["review_notes"] = notes

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "csv": str(csv_path),
        "row_count": len(rows),
        "decisions": dict(sorted(Counter(row["review_decision"] for row in rows).items())),
        "prototype_quality": dict(sorted(Counter(row["prototype_quality"] for row in rows).items())),
    }


def _decision_for_row(row: dict[str, str]) -> tuple[str, str, str]:
    target = str(row.get("expected_target") or "")
    prototype_target = str(row.get("prototype_target") or "")
    reason = str(row.get("reconcile_reason") or "")

    if reason == "negative_prototype_too_close" and prototype_target and prototype_target != target:
        return (
            "add_prototype_hard_negative",
            "cross_target_negative",
            "cross-target near prototype; use as hard negative for prototype_target without weakening gates",
        )
    if reason == "part_conflict" and target in FRUIT_PART_CONFLICT_TARGETS and prototype_target == target:
        return (
            "add_prototype_positive",
            "clean_fruit_positive",
            "fruit part-conflict row with same-target prototype evidence; add as clean fruit positive",
        )
    if reason in POSITIVE_REASONS and target in PRIORITY_TARGETS and prototype_target == target:
        return (
            "add_prototype_positive",
            "same_target_positive",
            "same-target router/prototype failure in priority target; "
            "add as curated positive for target/class coverage",
        )
    if (
        reason == "negative_prototype_too_close"
        and prototype_target == target
        and str(row.get("prototype_class_label") or "") == str(row.get("expected_class") or "")
    ):
        return (
            "add_prototype_positive",
            "same_target_exact_class_positive",
            "same-target exact-class row blocked by hard-negative proximity; "
            "add as curated positive, not hard negative",
        )
    return (
        "exclude_ambiguous",
        "reviewed_router_only",
        "router-only conservative review: deferred as ambiguous or outside current priority slice",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Router-only hard_example_audit.csv to update in place.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(review_router_only_audit(args.csv), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
