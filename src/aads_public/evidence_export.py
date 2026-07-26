"""Sanitize an archived controlled-demo run into public row-level evidence."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .evidence import EXPECTED_MANIFEST_SHA256, EXPECTED_RUN_ID

_REVIEW_TARGETS = frozenset({"non_plant", "unknown_crop"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _expected_outcome(target: str) -> str:
    if target in _REVIEW_TARGETS or target.endswith("__unknown_part"):
        return "review"
    return "answer"


def _actual_outcome(row: dict[str, Any]) -> str:
    predicted_class = str(row.get("predicted_disease") or "").strip()
    if row.get("actual_status") == "success" and predicted_class:
        return "answer"
    return "review"


def export_public_evidence(
    *,
    source_summary_path: Path,
    source_rows_path: Path,
    destination_summary_path: Path,
    destination_rows_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_summary = _read_object(source_summary_path)
    source_run = _read_object(source_rows_path)
    if source_summary.get("created_at") != EXPECTED_RUN_ID:
        raise ValueError("source summary does not match the accepted run")
    if source_summary.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("source summary does not match the accepted manifest")
    if source_summary.get("runner_exit_code") != 0 or source_run.get("device") != "cuda":
        raise ValueError("source run is not the accepted successful CUDA run")

    raw_rows = source_run.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("source run has no rows")

    public_rows: list[dict[str, Any]] = []
    per_target: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise ValueError("source row must be an object")
        row_id = str(raw_row.get("image_id") or "").strip()
        expected_target = str(raw_row.get("expected_target") or "").strip()
        expected_outcome = _expected_outcome(expected_target)
        actual_outcome = _actual_outcome(raw_row)
        expected_class = str(raw_row.get("expected_class") or "").strip() if expected_outcome == "answer" else None
        predicted_class = str(raw_row.get("predicted_disease") or "").strip() if actual_outcome == "answer" else None
        actual_target = (
            f"{raw_row.get('predicted_crop')}__{raw_row.get('predicted_part')}"
            if actual_outcome == "answer"
            else None
        )
        source_passed = raw_row.get("pass_fail") == "pass"
        class_correct = source_passed and expected_outcome == "answer" and actual_outcome == "answer"
        negative_false_accept = expected_outcome == "review" and actual_outcome == "answer"
        expected_crop, expected_part = (
            expected_target.split("__", maxsplit=1) if "__" in expected_target else (expected_target, "")
        )
        actual_crop, actual_part = (
            actual_target.split("__", maxsplit=1) if actual_target is not None else ("", "")
        )
        wrong_part = (
            expected_outcome == "answer"
            and actual_outcome == "answer"
            and expected_crop == actual_crop
            and expected_part != actual_part
        )
        derived_pass = (
            actual_outcome == "review"
            if expected_outcome == "review"
            else actual_outcome == "answer" and actual_target == expected_target and class_correct
        )
        if source_passed != derived_pass:
            raise ValueError(f"cannot reproduce source pass decision for {row_id}")

        public_rows.append(
            {
                "row_id": row_id,
                "expected_target": expected_target,
                "expected_outcome": expected_outcome,
                "expected_class": expected_class,
                "actual_status": str(raw_row.get("actual_status") or ""),
                "actual_outcome": actual_outcome,
                "actual_target": actual_target,
                "predicted_class": predicted_class,
                "class_correct": class_correct,
                "negative_false_accept": negative_false_accept,
                "wrong_part_disease_label": wrong_part,
                "passed": source_passed,
            }
        )
        per_target[expected_target]["total"] += 1
        per_target[expected_target]["passed"] += int(source_passed)

    rows_payload = {
        "schema_version": "aads.public_controlled_demo_rows.v1",
        "run_id": EXPECTED_RUN_ID,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "privacy_note": "Sanitized decisions only; no image paths, source URLs, or image contents are included.",
        "rows": public_rows,
    }
    destination_rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_content = (json.dumps(rows_payload, indent=2, ensure_ascii=False) + "\n").encode()
    destination_rows_path.write_bytes(rows_content)

    summary_payload = {
        "schema_version": "aads.public_controlled_demo.v2",
        "run_id": EXPECTED_RUN_ID,
        "source_surface": "controlled_customer_demo",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_summary_sha256": _sha256(source_summary_path),
        "source_rows_sha256": _sha256(source_rows_path),
        "rows_file_sha256": hashlib.sha256(rows_content).hexdigest(),
        "runtime": "cuda",
        "runner_exit_code": 0,
        "total": len(public_rows),
        "passed": sum(row["passed"] for row in public_rows),
        "failed": sum(not row["passed"] for row in public_rows),
        "answered": sum(row["actual_outcome"] == "answer" for row in public_rows),
        "reviewed_or_abstained": sum(row["actual_outcome"] == "review" for row in public_rows),
        "negative_false_accepts": sum(row["negative_false_accept"] for row in public_rows),
        "wrong_part_disease_labels": sum(row["wrong_part_disease_label"] for row in public_rows),
        "production_ready": False,
        "scope_note": (
            "Frozen row-level controlled-demo snapshot; this validates recorded decisions, not fresh inference, "
            "field performance, or production readiness."
        ),
        "per_target": dict(sorted(per_target.items())),
    }
    destination_summary_path.parent.mkdir(parents=True, exist_ok=True)
    destination_summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary_payload, rows_payload
