"""Strict validation for the frozen, row-level controlled-demo snapshot."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_RUN_ID = "20260706T153334Z"
EXPECTED_MANIFEST_SHA256 = "be31975d197e98ff4376b03fac45d4cdde4c6040f29458de783a7ec579bb5d06"
EXPECTED_SOURCE_SUMMARY_SHA256 = "c26515bbbef07a1fe73b09d0baf431b0cf9ef64fc8cd9503bd9d9e428b7d715c"
EXPECTED_SOURCE_ROWS_SHA256 = "01954ab73437e37e36c8769166e374682b041e222e129e009444fa740eee192d"
EXPECTED_PUBLIC_ROWS_SHA256 = "ddf3cf57cf1e3632c474682d4b676c2af7a147e2b46ba0da152ee852ddfe15fd"
EXPECTED_TOTAL = 48
EXPECTED_ANSWERED = 36
EXPECTED_REVIEWED = 12
_TARGET_PATTERN = re.compile(r"^[a-z0-9]+__[a-z0-9_]+$")
_ROW_ID_PATTERN = re.compile(r"^demo_[0-9]+$")
_SPECIAL_REVIEW_TARGETS = frozenset({"non_plant", "unknown_crop"})
_ROWS_PAYLOAD_KEYS = frozenset({"schema_version", "run_id", "manifest_sha256", "privacy_note", "rows"})
_ROW_KEYS = frozenset(
    {
        "row_id",
        "expected_target",
        "expected_outcome",
        "expected_class",
        "actual_status",
        "actual_outcome",
        "actual_target",
        "predicted_class",
        "class_correct",
        "negative_false_accept",
        "wrong_part_disease_label",
        "passed",
    }
)
_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "source_surface",
        "manifest_sha256",
        "source_summary_sha256",
        "source_rows_sha256",
        "rows_file_sha256",
        "runtime",
        "runner_exit_code",
        "total",
        "passed",
        "failed",
        "answered",
        "reviewed_or_abstained",
        "negative_false_accepts",
        "wrong_part_disease_labels",
        "production_ready",
        "scope_note",
        "per_target",
    }
)


@dataclass(frozen=True)
class AcceptanceRow:
    row_id: str
    expected_target: str
    expected_outcome: str
    actual_outcome: str
    passed: bool
    negative_false_accept: bool
    wrong_part_disease_label: bool


@dataclass(frozen=True)
class AcceptanceReport:
    run_id: str
    manifest_sha256: str
    total: int
    passed: int
    failed: int
    answered: int
    reviewed: int
    negative_false_accepts: int
    wrong_part_disease_labels: int
    production_ready: bool
    identity_verified: bool
    rows_verified: bool

    @property
    def controlled_demo_passed(self) -> bool:
        return (
            self.identity_verified
            and self.rows_verified
            and self.total == EXPECTED_TOTAL
            and self.passed == self.total
            and self.failed == 0
            and self.answered == EXPECTED_ANSWERED
            and self.reviewed == EXPECTED_REVIEWED
            and self.negative_false_accepts == 0
            and self.wrong_part_disease_labels == 0
            and not self.production_ready
        )


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_exact_keys(payload: dict[str, Any], expected: frozenset[str], name: str) -> None:
    if payload.keys() != expected:
        missing = sorted(expected - payload.keys())
        unexpected = sorted(payload.keys() - expected)
        raise ValueError(f"{name} contains missing fields {missing} or unexpected fields {unexpected}")


def _text(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(payload: dict[str, Any], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _boolean(payload: dict[str, Any], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _optional_text(payload: dict[str, Any], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be null or a non-empty string")
    return value


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _target_parts(target: str) -> tuple[str, str]:
    if target in _SPECIAL_REVIEW_TARGETS:
        return target, ""
    if not _TARGET_PATTERN.fullmatch(target):
        raise ValueError(f"invalid target identifier: {target}")
    crop, part = target.split("__", maxsplit=1)
    return crop, part


def _normalized_class_label(value: str | None) -> str | None:
    if value is None:
        return None
    decomposed = unicodedata.normalize("NFD", value.lower())
    return re.sub(r"[^a-z0-9_]", "_", decomposed)


def _load_rows(path: Path, *, summary: dict[str, Any]) -> tuple[AcceptanceRow, ...]:
    content = path.read_bytes()
    expected_digest = _text(summary, "rows_file_sha256")
    _validate_digest(expected_digest, "rows_file_sha256")
    if expected_digest != EXPECTED_PUBLIC_ROWS_SHA256:
        raise ValueError("row evidence identity does not match the frozen public snapshot")
    if _sha256_bytes(content) != expected_digest:
        raise ValueError("row evidence digest does not match the frozen summary")

    payload = _object(json.loads(content), "row evidence")
    _require_exact_keys(payload, _ROWS_PAYLOAD_KEYS, "row evidence")
    if payload.get("schema_version") != "aads.public_controlled_demo_rows.v1":
        raise ValueError("unsupported row evidence schema")
    if payload.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("row evidence run_id does not match the frozen run")
    if payload.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("row evidence manifest identity does not match the frozen run")

    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("row evidence must contain a non-empty rows list")

    rows: list[AcceptanceRow] = []
    seen_ids: set[str] = set()
    for index, raw_row in enumerate(raw_rows):
        row = _object(raw_row, f"rows[{index}]")
        _require_exact_keys(row, _ROW_KEYS, f"rows[{index}]")
        row_id = _text(row, "row_id")
        if not _ROW_ID_PATTERN.fullmatch(row_id) or row_id in seen_ids:
            raise ValueError(f"row_id must be unique and normalized: {row_id}")
        seen_ids.add(row_id)

        expected_target = _text(row, "expected_target")
        expected_crop, expected_part = _target_parts(expected_target)
        expected_outcome = _text(row, "expected_outcome")
        actual_outcome = _text(row, "actual_outcome")
        if expected_outcome not in {"answer", "review"} or actual_outcome not in {"answer", "review"}:
            raise ValueError(f"invalid outcome for {row_id}")

        expected_class = _optional_text(row, "expected_class")
        actual_status = _text(row, "actual_status")
        actual_target = _optional_text(row, "actual_target")
        predicted_class = _optional_text(row, "predicted_class")
        class_correct = _boolean(row, "class_correct")
        negative_false_accept = _boolean(row, "negative_false_accept")
        wrong_part_disease_label = _boolean(row, "wrong_part_disease_label")
        passed = _boolean(row, "passed")

        if expected_outcome == "answer" and expected_class is None:
            raise ValueError(f"expected_class is required for answer row {row_id}")
        if expected_outcome == "review" and expected_class is not None:
            raise ValueError(f"expected_class must be null for review row {row_id}")
        if actual_outcome == "answer":
            if actual_status != "success":
                raise ValueError(f"answer row {row_id} must record success status")
            if actual_target is None or predicted_class is None:
                raise ValueError(f"answer row {row_id} requires actual_target and predicted_class")
            actual_crop, actual_part = _target_parts(actual_target)
        else:
            if actual_status != "router_uncertain":
                raise ValueError(f"review row {row_id} must record router_uncertain status")
            if actual_target is not None or predicted_class is not None or class_correct:
                raise ValueError(f"review row {row_id} cannot carry an accepted target or class")
            actual_crop, actual_part = "", ""

        derived_class_correct = (
            expected_outcome == "answer"
            and actual_outcome == "answer"
            and _normalized_class_label(predicted_class) == _normalized_class_label(expected_class)
        )
        derived_false_accept = expected_outcome == "review" and actual_outcome == "answer"
        derived_wrong_part = (
            expected_outcome == "answer"
            and actual_outcome == "answer"
            and actual_crop == expected_crop
            and actual_part != expected_part
        )
        derived_pass = (
            actual_outcome == "review"
            if expected_outcome == "review"
            else actual_outcome == "answer" and actual_target == expected_target and class_correct
        )
        if class_correct != derived_class_correct:
            raise ValueError(f"class_correct is inconsistent for {row_id}")
        if negative_false_accept != derived_false_accept:
            raise ValueError(f"negative_false_accept is inconsistent for {row_id}")
        if wrong_part_disease_label != derived_wrong_part:
            raise ValueError(f"wrong_part_disease_label is inconsistent for {row_id}")
        if passed != derived_pass:
            raise ValueError(f"passed is inconsistent with the row decision for {row_id}")

        rows.append(
            AcceptanceRow(
                row_id=row_id,
                expected_target=expected_target,
                expected_outcome=expected_outcome,
                actual_outcome=actual_outcome,
                passed=passed,
                negative_false_accept=negative_false_accept,
                wrong_part_disease_label=wrong_part_disease_label,
            )
        )
    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(f"row evidence must contain exactly {EXPECTED_TOTAL} rows")
    return tuple(rows)


def _per_target(rows: tuple[AcceptanceRow, ...]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        counts = result.setdefault(row.expected_target, {"total": 0, "passed": 0})
        counts["total"] += 1
        counts["passed"] += int(row.passed)
    return dict(sorted(result.items()))


def load_acceptance_report(
    path: str | Path,
    rows_path: str | Path | None = None,
) -> AcceptanceReport:
    summary_path = Path(path)
    payload = _object(json.loads(summary_path.read_text(encoding="utf-8")), "evidence summary")
    _require_exact_keys(payload, _SUMMARY_KEYS, "evidence summary")
    if payload.get("schema_version") != "aads.public_controlled_demo.v2":
        raise ValueError("unsupported evidence schema")
    if payload.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("run_id does not match the frozen controlled-demo run")
    if payload.get("source_surface") != "controlled_customer_demo":
        raise ValueError("source_surface must identify the controlled customer demo")
    if payload.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("manifest identity does not match the frozen controlled-demo run")
    if payload.get("runtime") != "cuda":
        raise ValueError("the frozen controlled-demo run must record the CUDA runtime")
    if payload.get("runner_exit_code") != 0:
        raise ValueError("the frozen controlled-demo runner did not exit successfully")
    if payload.get("source_summary_sha256") != EXPECTED_SOURCE_SUMMARY_SHA256:
        raise ValueError("source summary identity does not match the archived run")
    if payload.get("source_rows_sha256") != EXPECTED_SOURCE_ROWS_SHA256:
        raise ValueError("source row identity does not match the archived run")
    if payload.get("production_ready") is not False:
        raise ValueError("controlled-demo evidence must explicitly set production_ready to false")

    resolved_rows_path = (
        Path(rows_path) if rows_path is not None else summary_path.with_name("controlled_demo_rows.json")
    )
    rows = _load_rows(resolved_rows_path, summary=payload)
    computed = {
        "total": len(rows),
        "passed": sum(row.passed for row in rows),
        "failed": sum(not row.passed for row in rows),
        "answered": sum(row.actual_outcome == "answer" for row in rows),
        "reviewed_or_abstained": sum(row.actual_outcome == "review" for row in rows),
        "negative_false_accepts": sum(row.negative_false_accept for row in rows),
        "wrong_part_disease_labels": sum(row.wrong_part_disease_label for row in rows),
    }
    for name, value in computed.items():
        if _integer(payload, name) != value:
            raise ValueError(f"{name} does not match the row-level evidence")

    reported_per_target = _object(payload.get("per_target"), "per_target")
    if reported_per_target != _per_target(rows):
        raise ValueError("per_target does not match the row-level evidence")

    report = AcceptanceReport(
        run_id=EXPECTED_RUN_ID,
        manifest_sha256=EXPECTED_MANIFEST_SHA256,
        total=computed["total"],
        passed=computed["passed"],
        failed=computed["failed"],
        answered=computed["answered"],
        reviewed=computed["reviewed_or_abstained"],
        negative_false_accepts=computed["negative_false_accepts"],
        wrong_part_disease_labels=computed["wrong_part_disease_labels"],
        production_ready=False,
        identity_verified=True,
        rows_verified=True,
    )
    return report
