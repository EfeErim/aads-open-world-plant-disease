import csv
from pathlib import Path

from scripts.review_m2_router_only_audit import review_router_only_audit

HEADERS = [
    "image_id",
    "expected_target",
    "expected_class",
    "prototype_target",
    "prototype_class_label",
    "reconcile_reason",
    "review_decision",
    "prototype_quality",
    "adapter_training_quality",
    "review_notes",
]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in HEADERS})


def test_review_router_only_audit_fills_conservative_decisions(tmp_path: Path):
    audit = tmp_path / "hard_example_audit.csv"
    _write_rows(
        audit,
        [
            {
                "image_id": "demo_001",
                "expected_target": "tomato__leaf",
                "prototype_target": "tomato__leaf",
                "reconcile_reason": "prototype_evidence_weak",
            },
            {
                "image_id": "demo_002",
                "expected_target": "tomato__leaf",
                "prototype_target": "tomato__fruit",
                "reconcile_reason": "negative_prototype_too_close",
            },
            {
                "image_id": "demo_003",
                "expected_target": "apricot__leaf",
                "prototype_target": "apricot__leaf",
                "reconcile_reason": "prototype_evidence_weak",
            },
        ],
    )

    summary = review_router_only_audit(audit)

    assert summary["decisions"] == {
        "add_prototype_hard_negative": 1,
        "add_prototype_positive": 2,
    }
    with audit.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["adapter_training_quality"] == "deferred_router_only"
    assert rows[1]["prototype_quality"] == "cross_target_negative"
    assert rows[2]["review_decision"] == "add_prototype_positive"
