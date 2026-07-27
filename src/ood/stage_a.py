"""Stage-A recovery analysis from committed OOD method-comparison artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.ood.behavioral_acceptance import behavioral_acceptance_pass
from src.ood.recovery import TARGET_ADAPTERS, discover_latest_readiness
from src.training.services.ood_score_selection import select_best_ood_score_method

STAGE_A_SCHEMA = "v1_adapter_ood_stage_a_report"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _comparison_path(readiness_path: Path, split: str) -> Path:
    return readiness_path.parent / split / "ood_method_comparison.json"


def _method_metrics(payload: Mapping[str, Any], method: str) -> dict[str, Any]:
    methods = payload.get("methods")
    if not isinstance(methods, Mapping):
        return {}
    details = methods.get(method)
    return dict(details) if isinstance(details, Mapping) else {}


def build_stage_a_report(
    runs_root: Path,
    *,
    readiness_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Select score methods on validation and inspect the frozen test artifact once."""
    selected_paths = dict(readiness_paths or discover_latest_readiness(runs_root))
    adapters: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    for target in TARGET_ADAPTERS:
        readiness_path = selected_paths.get(target)
        if readiness_path is None:
            adapters.append({"target": target, "status": "missing", "selected_method": ""})
            issues.append({"target": target, "code": "readiness_missing"})
            continue
        validation_path = _comparison_path(readiness_path, "validation")
        test_path = _comparison_path(readiness_path, "test")
        if not validation_path.is_file() or not test_path.is_file():
            adapters.append({"target": target, "status": "missing_comparison", "selected_method": ""})
            issues.append({"target": target, "code": "comparison_missing"})
            continue
        validation = _read_json(validation_path)
        test = _read_json(test_path)
        readiness = _read_json(readiness_path)
        if str(validation.get("split_name") or "").lower() not in {"validation", "val"}:
            issues.append({"target": target, "code": "selection_split_not_validation"})
            continue
        if str(test.get("split_name") or "").lower() != "test":
            issues.append({"target": target, "code": "evaluation_split_not_test"})
            continue
        selected_method = select_best_ood_score_method(validation)
        validation_details = _method_metrics(validation, selected_method)
        test_details = _method_metrics(test, selected_method)
        validation_metrics = dict(validation_details.get("pooled_metrics") or {})
        test_metrics = dict(test_details.get("pooled_metrics") or {})
        test_worst = dict(test_details.get("worst_slice") or {})
        baseline_method = str(test.get("selected_primary_score_method") or "ensemble")
        baseline_test_details = _method_metrics(test, baseline_method)
        baseline_worst = dict(baseline_test_details.get("worst_slice") or {})
        behavioral_path = readiness_path.parent / "adapter_behavioral_acceptance.json"
        behavioral = _read_json(behavioral_path) if behavioral_path.is_file() else {}
        selected_worst_fpr = test_worst.get("ood_false_positive_rate")
        baseline_worst_fpr = baseline_worst.get("ood_false_positive_rate")
        worst_slice_regressed = (
            selected_worst_fpr is not None
            and baseline_worst_fpr is not None
            and float(selected_worst_fpr) > float(baseline_worst_fpr)
        )
        sample_checks = readiness.get("ood_type_sample_checks")
        sample_floors_passed = bool(sample_checks) and all(
            bool(check.get("passed"))
            for check in sample_checks.values()
            if isinstance(check, Mapping) and bool(check.get("asserted", True))
        )
        pooled_gate_passed = bool(test_details.get("pooled_gate_eligible", False))
        stage_a_promotable = behavioral_acceptance_pass(behavioral)
        adapters.append(
            {
                "target": target,
                "status": "selected",
                "selection_source": "validation_ood_dev_artifact",
                "selected_method": selected_method,
                "validation_comparison_path": str(validation_path),
                "test_comparison_path": str(test_path),
                "validation_metrics": validation_metrics,
                "validation_worst_slice": dict(validation_details.get("worst_slice") or {}),
                "test_metrics": test_metrics,
                "test_worst_slice": test_worst,
                "baseline_method": baseline_method,
                "baseline_test_worst_slice": baseline_worst,
                "behavioral_acceptance_path": str(behavioral_path) if behavioral_path.is_file() else "",
                "behavioral_acceptance_pass": stage_a_promotable,
                "pooled_ood_gate_passed": pooled_gate_passed,
                "sample_floors_passed": sample_floors_passed,
                "worst_slice_regressed": worst_slice_regressed,
                "stage_a_promotable": stage_a_promotable,
            }
        )
    return {
        "schema_version": STAGE_A_SCHEMA,
        "decision_authority": "adapter_behavioral_acceptance",
        "ok": not issues,
        "selection_split": "validation",
        "final_test_used_for_selection": False,
        "selected_target_count": sum(item.get("status") == "selected" for item in adapters),
        "pooled_test_gate_pass_target_count": sum(bool(item.get("pooled_ood_gate_passed")) for item in adapters),
        "stage_a_promotable_target_count": sum(bool(item.get("stage_a_promotable")) for item in adapters),
        "adapters": adapters,
        "issues": issues,
    }


def render_stage_a_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Adapter OOD/OE Stage A Score Selection",
        "",
        "- Selection source: validation OOD-dev comparison artifacts",
        "- Final test used for selection: no",
        f"- Selected targets: `{report['selected_target_count']}/8`",
        f"- Selected methods passing the pooled test OOD gate: `{report['pooled_test_gate_pass_target_count']}/8`",
        f"- Stage A behaviorally accepted: `{report['stage_a_promotable_target_count']}/8`",
        "",
        "| Target | Selected | Val AUROC | Val FPR | Test AUROC | Test FPR | Worst slice FPR | Floors | Regression | Promote |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]

    def metric(item: Mapping[str, Any], bucket: str, name: str) -> str:
        payload = item.get(bucket)
        value = payload.get(name) if isinstance(payload, Mapping) else None
        return "-" if value is None else f"{float(value):.4f}"

    for item in report["adapters"]:
        worst = item.get("test_worst_slice")
        worst_fpr = worst.get("ood_false_positive_rate") if isinstance(worst, Mapping) else None
        worst_fpr_text = "-" if worst_fpr is None else f"{float(worst_fpr):.4f}"
        lines.append(
            f"| `{item['target']}` | `{item.get('selected_method') or '-'}` | "
            f"{metric(item, 'validation_metrics', 'ood_auroc')} | "
            f"{metric(item, 'validation_metrics', 'ood_false_positive_rate')} | "
            f"{metric(item, 'test_metrics', 'ood_auroc')} | "
            f"{metric(item, 'test_metrics', 'ood_false_positive_rate')} | {worst_fpr_text} | "
            f"`{'pass' if item.get('sample_floors_passed') else 'fail'}` | "
            f"`{'yes' if item.get('worst_slice_regressed') else 'no'}` | "
            f"`{'yes' if item.get('stage_a_promotable') else 'no'}` |"
        )
    lines.extend(
        [
            "",
            "AUROC/FPR and worst-slice fields are diagnostics. Promotion requires adapter_behavioral_acceptance.json.",
            "",
        ]
    )
    return "\n".join(lines)
