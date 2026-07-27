#!/usr/bin/env python3
"""Recompute production_readiness for a run using `knn` as primary OOD method.

Usage: python scripts/recompute_readiness_knn.py <run_artifact_root>
Example: python scripts/recompute_readiness_knn.py runs/grape/fruit/grape_fruit_2026-07-13_18-07-48/telemetry/artifacts
"""
import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path so `src` imports work when script is executed directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shared.json_utils import read_json, write_json
from src.training.services.metrics import build_production_readiness


def main():
    if len(sys.argv) < 2:
        print("Usage: recompute_readiness_knn.py <artifact_root_path>")
        sys.exit(2)
    artifact_root = Path(sys.argv[1])
    if not artifact_root.exists():
        print(f"Path not found: {artifact_root}")
        sys.exit(2)

    metric_gate = read_json(artifact_root / "test" / "metric_gate.json", default={}, expect_type=dict)
    ood_methods = read_json(artifact_root / "test" / "ood_method_comparison.json", default={}, expect_type=dict)

    # Extract classification metric gate and knn pooled metrics
    classification_metric_gate = dict(metric_gate) if isinstance(metric_gate, dict) else {}
    methods = dict(ood_methods.get("methods", {})) if isinstance(ood_methods, dict) else {}
    knn_metrics = None
    if "knn" in methods and isinstance(methods["knn"], dict):
        knn_metrics = dict(methods["knn"].get("pooled_metrics", {}))

    if not knn_metrics:
        print("knn pooled metrics not found in ood_method_comparison.json")
        sys.exit(1)

    readiness = build_production_readiness(
        classification_metric_gate=classification_metric_gate,
        classification_split=str(classification_metric_gate.get("context", {}).get("split_name", "test") or "test"),
        ood_evidence_source="real_ood_split",
        ood_metrics=knn_metrics,
        targets=None,
        context={"ood_primary_score_method": "knn", "ood_method_comparison": ood_methods},
        require_ood=True,
    )

    out_path = artifact_root / "production_readiness.knn.json"
    write_json(out_path, readiness)
    print(f"Wrote: {out_path}")
    print(json.dumps(readiness, indent=2))


if __name__ == "__main__":
    main()
