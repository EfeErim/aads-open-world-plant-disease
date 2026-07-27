"""Helpers for evaluating BER rollout candidates against a baseline run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _optional_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _load_artifact_bundle(artifact_root: Path) -> Dict[str, Any]:
    production_readiness = _read_json(artifact_root / "production_readiness.json")
    classification = dict(production_readiness.get("classification_evidence", {}))
    ood_evidence = dict(production_readiness.get("ood_evidence", {}))

    return {
        "artifact_root": str(artifact_root),
        "production_readiness": production_readiness,
        "classification_split": str(classification.get("split_name", "")),
        "accuracy": classification.get("metrics", {}).get("accuracy"),
        "ood_evidence_source": str(production_readiness.get("ood_evidence_source", "unavailable")),
        "ood_auroc": ood_evidence.get("metrics", {}).get("ood_auroc"),
        "ood_false_positive_rate": ood_evidence.get("metrics", {}).get("ood_false_positive_rate"),
        "validation_metric_gate": _read_json(artifact_root / "validation" / "metric_gate.json"),
        "test_metric_gate": _read_json(artifact_root / "test" / "metric_gate.json"),
        "ood_benchmark_summary": _optional_json(artifact_root / "ood_benchmark" / "summary.json"),
    }


def _check(passed: bool, detail: str) -> Dict[str, Any]:
    return {"passed": bool(passed), "detail": str(detail)}


def evaluate_ber_candidate(
    *,
    baseline_artifact_root: str | Path,
    candidate_artifact_root: str | Path,
    max_accuracy_drop: float = 0.002,
    min_ood_auroc_gain: float = 0.01,
    min_fpr_improvement: float = 0.01,
) -> Dict[str, Any]:
    """Compare one BER candidate artifact root against a BER-off baseline."""

    baseline_root = Path(baseline_artifact_root)
    candidate_root = Path(candidate_artifact_root)
    baseline = _load_artifact_bundle(baseline_root)
    candidate = _load_artifact_bundle(candidate_root)

    baseline_ready = bool(baseline["production_readiness"].get("passed", False))
    candidate_ready = bool(candidate["production_readiness"].get("passed", False))

    baseline_accuracy = baseline.get("accuracy")
    candidate_accuracy = candidate.get("accuracy")
    accuracy_drop = (
        float(baseline_accuracy) - float(candidate_accuracy)
        if baseline_accuracy is not None and candidate_accuracy is not None
        else None
    )

    baseline_auroc = baseline.get("ood_auroc")
    candidate_auroc = candidate.get("ood_auroc")
    ood_auroc_gain = (
        float(candidate_auroc) - float(baseline_auroc)
        if baseline_auroc is not None and candidate_auroc is not None
        else None
    )

    baseline_fpr = baseline.get("ood_false_positive_rate")
    candidate_fpr = candidate.get("ood_false_positive_rate")
    fpr_improvement = (
        float(baseline_fpr) - float(candidate_fpr)
        if baseline_fpr is not None and candidate_fpr is not None
        else None
    )

    same_evidence_source = baseline["ood_evidence_source"] == candidate["ood_evidence_source"]
    readiness_not_worse = (not baseline_ready) or candidate_ready
    accuracy_within_limit = accuracy_drop is not None and accuracy_drop <= float(max_accuracy_drop)
    ood_improved = (
        (ood_auroc_gain is not None and ood_auroc_gain >= float(min_ood_auroc_gain))
        or (fpr_improvement is not None and fpr_improvement >= float(min_fpr_improvement))
    )

    checks = {
        "same_ood_evidence_source": _check(
            same_evidence_source,
            f"baseline={baseline['ood_evidence_source']} candidate={candidate['ood_evidence_source']}",
        ),
        "readiness_not_worse": _check(
            readiness_not_worse,
            f"baseline_ready={baseline_ready} candidate_ready={candidate_ready}",
        ),
        "accuracy_drop_within_limit": _check(
            accuracy_within_limit,
            f"accuracy_drop={accuracy_drop} limit={max_accuracy_drop}",
        ),
        "ood_improvement": _check(
            ood_improved,
            (
                f"ood_auroc_gain={ood_auroc_gain} min_gain={min_ood_auroc_gain}; "
                f"fpr_improvement={fpr_improvement} min_improvement={min_fpr_improvement}"
            ),
        ),
    }

    passed = all(check["passed"] for check in checks.values())
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "baseline": baseline,
        "candidate": candidate,
        "comparisons": {
            "accuracy_drop": accuracy_drop,
            "ood_auroc_gain": ood_auroc_gain,
            "fpr_improvement": fpr_improvement,
        },
        "thresholds": {
            "max_accuracy_drop": float(max_accuracy_drop),
            "min_ood_auroc_gain": float(min_ood_auroc_gain),
            "min_fpr_improvement": float(min_fpr_improvement),
        },
        "checks": checks,
    }
