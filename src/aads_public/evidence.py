"""Validation for the frozen controlled-demo evidence summary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AcceptanceReport:
    run_id: str
    total: int
    passed: int
    failed: int
    answered: int
    reviewed: int
    negative_false_accepts: int
    wrong_part_disease_labels: int
    production_ready: bool

    @property
    def controlled_demo_passed(self) -> bool:
        return (
            self.total > 0
            and self.passed == self.total
            and self.failed == 0
            and self.answered + self.reviewed == self.total
            and self.negative_false_accepts == 0
            and self.wrong_part_disease_labels == 0
        )


def _integer(payload: dict[str, Any], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def load_acceptance_report(path: str | Path) -> AcceptanceReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "aads.public_controlled_demo.v1":
        raise ValueError("unsupported evidence schema")
    report = AcceptanceReport(
        run_id=str(payload.get("run_id") or ""),
        total=_integer(payload, "total"),
        passed=_integer(payload, "passed"),
        failed=_integer(payload, "failed"),
        answered=_integer(payload, "answered"),
        reviewed=_integer(payload, "reviewed_or_abstained"),
        negative_false_accepts=_integer(payload, "negative_false_accepts"),
        wrong_part_disease_labels=_integer(payload, "wrong_part_disease_labels"),
        production_ready=payload.get("production_ready") is True,
    )
    if not report.run_id:
        raise ValueError("run_id is required")
    if report.passed + report.failed != report.total:
        raise ValueError("passed + failed must equal total")
    if report.answered + report.reviewed != report.total:
        raise ValueError("answered + reviewed must equal total")
    if report.production_ready:
        raise ValueError("controlled-demo evidence cannot claim production readiness")
    return report
