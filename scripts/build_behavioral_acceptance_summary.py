#!/usr/bin/env python3
"""Rebuild the public eight-target behavioral acceptance summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "aads.public_behavioral_acceptance_summary.v2"
SOURCE_COMMIT = "539397bb72bde59e4b092ac1286b5415fe78dbac"
SNAPSHOT_DATE = "2026-07-27"
TARGET_NAMES = (
    "apricot__fruit",
    "apricot__leaf",
    "grape__fruit",
    "grape__leaf",
    "strawberry__fruit",
    "strawberry__leaf",
    "tomato__fruit",
    "tomato__leaf",
)
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "id_test_samples",
    "id_false_rejection_rate",
    "same_crop_ood_test_samples",
    "same_crop_ood_rejection_rate",
    "forced_supported_answer_count",
)
THRESHOLD_FIELDS = {
    "accuracy": "accuracy_min",
    "balanced_accuracy": "balanced_accuracy_min",
    "macro_f1": "macro_f1_min",
    "id_test_samples": "id_test_samples_min",
    "id_false_rejection_rate": "id_false_rejection_rate_max",
    "same_crop_ood_test_samples": "same_crop_ood_test_samples_min",
    "forced_supported_answer_count": "forced_supported_answer_count_max",
    "same_crop_ood_rejection_rate": "same_crop_ood_rejection_rate_min",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_summary(evidence_root: Path) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    thresholds: dict[str, Any] | None = None
    for target_name in TARGET_NAMES:
        source_path = evidence_root / "behavioral_acceptance" / "targets" / f"{target_name}.json"
        source = _read_json(source_path)
        context = dict(source.get("context") or {})
        internal_target = f"{context.get('crop_name')}__{context.get('part_name')}"
        if internal_target != target_name:
            raise ValueError(
                f"Behavioral acceptance target mismatch: expected={target_name}, actual={internal_target}"
            )
        checks = dict(source.get("checks") or {})
        current_thresholds = {
            output_name: checks[check_name]["threshold"]
            for check_name, output_name in THRESHOLD_FIELDS.items()
        }
        if thresholds is None:
            thresholds = current_thresholds
        elif current_thresholds != thresholds:
            raise ValueError(f"Behavioral acceptance thresholds differ for {target_name}")
        metrics = dict(source.get("metrics") or {})
        target_payload = {
            "run_id": str(context.get("run_id") or ""),
            "pass": bool(source.get("pass")),
            **{name: metrics.get(name) for name in METRIC_NAMES},
            "failure_reasons": list(source.get("failure_reasons") or []),
            "source_path": source_path.relative_to(evidence_root.parent).as_posix(),
            "source_sha256": _sha256(source_path),
        }
        for count_field in (
            "id_test_samples",
            "same_crop_ood_test_samples",
            "forced_supported_answer_count",
        ):
            value = target_payload[count_field]
            target_payload[count_field] = int(value) if value is not None else None
        targets[target_name] = target_payload

    passed_targets = sum(1 for payload in targets.values() if payload["pass"])
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": SNAPSHOT_DATE,
        "source_commit": SOURCE_COMMIT,
        "selection_rule": (
            "Newest tracked checkpoint_state/artifacts/adapter_behavioral_acceptance.json per target "
            "at source_commit, copied into evidence/behavioral_acceptance/targets."
        ),
        "decision_authority": "adapter_behavioral_acceptance",
        "passed_targets": passed_targets,
        "total_targets": len(targets),
        "thresholds": thresholds or {},
        "scope_note": (
            "These are tracked adapter-level acceptance records, not a new rerun and not field-performance "
            "estimates. A missing same-crop OOD rejection rate means the selected artifact contained zero "
            "eligible same-crop OOD test samples."
        ),
        "rebuild_command": "python scripts/build_behavioral_acceptance_summary.py --check",
        "targets": targets,
    }


def _serialized(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=Path("evidence"))
    parser.add_argument("--output", type=Path, default=Path("evidence/latest_behavioral_acceptance_summary.json"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()
    serialized = _serialized(build_summary(args.evidence_root))
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != serialized:
            raise SystemExit("Behavioral acceptance summary is stale; rebuild it from the public target records.")
        print(f"PASS: {args.output} matches all {len(TARGET_NAMES)} public target records.")
        return 0
    if args.stdout:
        print(serialized, end="")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(f"WROTE: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
