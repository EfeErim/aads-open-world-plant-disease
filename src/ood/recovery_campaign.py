"""Build the bounded, dev-selected adapter OOD/OE recovery campaign."""

from __future__ import annotations

from typing import Any, Mapping

from src.ood.recovery import TARGET_ADAPTERS

CAMPAIGN_SCHEMA = "v4_bounded_adapter_recovery_campaign"
MAX_ATTEMPTS_PER_TARGET = 4


def _adapter_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("target") or ""): item
        for item in report.get("adapters", [])
        if isinstance(item, Mapping)
    }


def _base_config(defaults: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    return {
        **dict(defaults),
        "OE_ENABLED": True,
        "LOSS_NAME": "logitnorm",
        "LOGITNORM_TAU": 1.0,
        "SEED": int(seed),
        "DETERMINISTIC": True,
        "ENABLE_BAYESIAN_OPTIMIZATION": False,
    }


def _experiment(
    *,
    target: str,
    candidate: str,
    parent_experiment_id: str | None,
    run_when: str,
    resolved_config: Mapping[str, Any],
    config_adjustments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": f"{target}__candidate_{candidate.lower()}",
        "candidate": candidate,
        "stage": candidate,
        "parent_experiment_id": parent_experiment_id,
        "run_when": run_when,
        "attempt_cap": 1,
        "resolved_config": dict(resolved_config),
        "config_adjustments": dict(config_adjustments or {}),
    }


def build_recovery_campaign(
    baseline: Mapping[str, Any],
    stage_a: Mapping[str, Any],
    *,
    seed: int = 42,
    target_defaults: Mapping[str, Mapping[str, Any]] | None = None,
    evidence_manifest_digest: str = "",
    evidence_ready_targets: set[str] | None = None,
) -> dict[str, Any]:
    baseline_map = _adapter_map(baseline)
    stage_a_map = _adapter_map(stage_a)
    targets: list[dict[str, Any]] = []
    for target in TARGET_ADAPTERS:
        baseline_item = baseline_map.get(target, {})
        stage_a_item = stage_a_map.get(target, {})
        sample_floors_passed = bool(stage_a_item.get("sample_floors_passed", False))
        evidence_ready = (
            target in evidence_ready_targets if evidence_ready_targets is not None else sample_floors_passed
        )
        base = _base_config(dict((target_defaults or {}).get(target, {})), seed=seed)
        candidate_b = {
            **base,
            "LABEL_SMOOTHING": 0.05,
            "CLASSIFIER_REBALANCE_ENABLED": True,
            "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU": 1.0,
            "OE_LOSS_WEIGHT": round(max(0.10, float(base.get("OE_LOSS_WEIGHT", 0.20)) - 0.10), 6),
        }
        candidate_c = {
            **base,
            "OE_LOSS_WEIGHT": round(min(0.40, float(base.get("OE_LOSS_WEIGHT", 0.20)) + 0.10), 6),
        }
        candidate_d = {
            **candidate_c,
            "LORA_R": min(32, int(base.get("LORA_R", 16)) + 8),
        }
        experiments = [
            _experiment(
                target=target,
                candidate="A",
                parent_experiment_id=None,
                run_when="evidence_repaired",
                resolved_config=base,
            ),
            _experiment(
                target=target,
                candidate="B",
                parent_experiment_id=f"{target}__candidate_a",
                run_when="id_dev_or_false_rejection_fails",
                resolved_config=candidate_b,
            ),
            _experiment(
                target=target,
                candidate="C",
                parent_experiment_id="last_id_passing_candidate",
                run_when="id_dev_passes_and_ood_dev_fails",
                resolved_config=candidate_c,
                config_adjustments={"OE_LOSS_WEIGHT": {"add": 0.10, "cap": 0.40}},
            ),
            _experiment(
                target=target,
                candidate="D",
                parent_experiment_id=f"{target}__candidate_c",
                run_when="candidate_c_still_fails_dev",
                resolved_config=candidate_d,
                config_adjustments={"LORA_R": {"add": 8, "cap": 32}},
            ),
        ]
        targets.append(
            {
                "target": target,
                "baseline_status": baseline_item.get("status", "missing"),
                "evidence_repair_required": not evidence_ready,
                "evidence_validation_passed": evidence_ready,
                "stage_a_promotable": bool(stage_a_item.get("stage_a_promotable", False)),
                "selected_stage_a_method": stage_a_item.get("selected_method", ""),
                "attempt_cap": MAX_ATTEMPTS_PER_TARGET,
                "experiments": experiments,
                "promotion_requires": [
                    "authoritative_behavioral_acceptance",
                    "evidence_integrity",
                    "serialization_reload_parity",
                    "controlled_demo_non_regression",
                    "atomic_eight_target_bundle",
                ],
            }
        )
    return {
        "schema_version": CAMPAIGN_SCHEMA,
        "seed": int(seed),
        "evidence_manifest_digest": str(evidence_manifest_digest),
        "evidence_ready_target_count": sum(not item["evidence_repair_required"] for item in targets),
        "selection_evidence": ["id_validation", "ood_dev"],
        "locked_final_evidence": ["id_test", "ood_test"],
        "selection_test_reuse_forbidden": True,
        "stop_on_dev_pass": True,
        "final_evaluation_once_per_target": True,
        "target_count": len(targets),
        "training_run_count": len(TARGET_ADAPTERS) * MAX_ATTEMPTS_PER_TARGET,
        "experiment_count": sum(len(item["experiments"]) for item in targets),
        "max_attempts_per_target": MAX_ATTEMPTS_PER_TARGET,
        "targets": targets,
    }


def render_campaign_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Adapter OOD/OE Recovery Campaign",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Targets: `{report['target_count']}`",
        f"- Candidate cap: `{report['max_attempts_per_target']}` per target",
        "- Candidate selection evidence: ID validation plus OOD dev only",
        "- Locked ID/OOD test evaluation: once for the frozen dev winner",
        "",
        "| Target | Evidence repair | Score method | Bounded candidates |",
        "|---|---|---|---|",
    ]
    for item in report["targets"]:
        candidates = ", ".join(experiment["candidate"] for experiment in item["experiments"])
        lines.append(
            f"| `{item['target']}` | `{'yes' if item['evidence_repair_required'] else 'no'}` | "
            f"`{item['selected_stage_a_method'] or '-'}` | `{candidates}` |"
        )
    lines.extend(
        [
            "",
            "Each candidate writes a non-authoritative `adapter_behavioral_dev_report.json`. The first dev-passing "
            "candidate is frozen and evaluated once on locked ID test plus OOD test. A final-test failure ends the "
            "target cycle; it does not trigger test-driven tuning.",
            "",
        ]
    )
    return "\n".join(lines)
