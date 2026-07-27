import json

from src.training.services.ber_rollout import evaluate_ber_candidate


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_artifacts(root, *, passed, accuracy, ood_auroc, ood_fpr, ood_evidence_source="held_out_benchmark"):
    _write_json(
        root / "production_readiness.json",
        {
            "status": "ready" if passed else "failed",
            "passed": passed,
            "ood_evidence_source": ood_evidence_source,
            "classification_evidence": {
                "split_name": "test",
                "metrics": {"accuracy": accuracy},
            },
            "ood_evidence": {
                "source": ood_evidence_source,
                "metrics": {
                    "ood_auroc": ood_auroc,
                    "ood_false_positive_rate": ood_fpr,
                },
            },
        },
    )
    _write_json(root / "validation" / "metric_gate.json", {"metrics": {"accuracy": accuracy}})
    _write_json(root / "test" / "metric_gate.json", {"metrics": {"accuracy": accuracy}})


def test_evaluate_ber_candidate_accepts_candidate_that_meets_rollout_thresholds(tmp_path):
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    _write_artifacts(baseline_root, passed=True, accuracy=0.950, ood_auroc=0.920, ood_fpr=0.050)
    _write_artifacts(candidate_root, passed=True, accuracy=0.949, ood_auroc=0.932, ood_fpr=0.045)

    result = evaluate_ber_candidate(
        baseline_artifact_root=baseline_root,
        candidate_artifact_root=candidate_root,
    )

    assert result["passed"] is True
    assert result["comparisons"]["accuracy_drop"] == 0.0010000000000000009
    assert result["checks"]["ood_improvement"]["passed"] is True


def test_evaluate_ber_candidate_rejects_candidate_without_required_improvement(tmp_path):
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    _write_artifacts(baseline_root, passed=True, accuracy=0.950, ood_auroc=0.920, ood_fpr=0.050)
    _write_artifacts(candidate_root, passed=False, accuracy=0.947, ood_auroc=0.925, ood_fpr=0.049)

    result = evaluate_ber_candidate(
        baseline_artifact_root=baseline_root,
        candidate_artifact_root=candidate_root,
    )

    assert result["passed"] is False
    assert result["checks"]["readiness_not_worse"]["passed"] is False
    assert result["checks"]["accuracy_drop_within_limit"]["passed"] is False
