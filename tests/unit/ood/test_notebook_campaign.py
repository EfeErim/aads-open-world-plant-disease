import json

from src.ood.behavioral_acceptance import BEHAVIORAL_ACCEPTANCE_SCHEMA, build_behavioral_dev_report
from src.ood.notebook_campaign import (
    build_campaign_lineage,
    build_notebook_completion_report,
    evidence_preflight_blocked_targets,
    experiment_gate,
    json_digest,
    load_campaign_ledger,
    readiness_ready,
    resumable_experiment,
    write_campaign_ledger,
)
from src.ood.recovery import TARGET_ADAPTERS


def _acceptance(*, classification: bool, id_behavior: bool, ood: bool, floor: bool, dev: bool = True) -> dict:
    checks = {
        "accuracy": {"passed": classification},
        "balanced_accuracy": {"passed": classification},
        "macro_f1": {"passed": classification},
        "id_test_samples": {"passed": classification},
        "id_false_rejection_rate": {"passed": id_behavior},
        "id_decisions_complete": {"passed": id_behavior},
        "same_crop_ood_test_samples": {"passed": floor},
        "forced_supported_answer_count": {"passed": ood},
        "same_crop_ood_rejection_rate": {"passed": ood},
        "same_crop_ood_decisions_complete": {"passed": ood},
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "schema_version": "v1_adapter_behavioral_dev_report" if dev else BEHAVIORAL_ACCEPTANCE_SCHEMA,
        "decision_authority": "adapter_behavioral_dev_selection" if dev else "adapter_behavioral_acceptance",
        "authoritative": False if dev else True,
        "evaluation_scope": "id_validation_and_ood_dev" if dev else "locked_test",
        "status": "pass" if passed else "fail",
        "pass": passed,
        "checks": checks,
    }


def test_stage_gates_enforce_id_behavior_and_same_crop_floor() -> None:
    failing_classification = _acceptance(classification=False, id_behavior=False, ood=False, floor=True)
    assert experiment_gate("B", failing_classification) == (True, "id_dev_or_false_rejection_recovery_required")
    assert experiment_gate("C", failing_classification) == (False, "id_behavior_gate_failed")

    missing_floor = _acceptance(classification=True, id_behavior=True, ood=False, floor=False)
    assert experiment_gate("C", missing_floor) == (False, "same_crop_ood_sample_floor_failed")

    stage_c_candidate = _acceptance(classification=True, id_behavior=True, ood=False, floor=True)
    assert experiment_gate("C", stage_c_candidate) == (True, "ood_dev_recovery_required")
    assert experiment_gate("D", stage_c_candidate) == (True, "bounded_capacity_recovery_required")


def test_direct_training_runs_without_legacy_stage_prerequisites() -> None:
    missing_current_artifact = {}

    assert experiment_gate("A", missing_current_artifact) == (True, "candidate_a_required")


def test_passed_candidate_stops_all_later_stages() -> None:
    passed = _acceptance(classification=True, id_behavior=True, ood=True, floor=True, dev=False)

    assert readiness_ready(passed) is True
    dev_passed = _acceptance(classification=True, id_behavior=True, ood=True, floor=True)
    assert experiment_gate("D", dev_passed) == (False, "already_passed_dev_selection")


def test_completion_requires_all_eight_targets_and_preflight() -> None:
    results = {target: {"pass": True, "status": "passed"} for target in TARGET_ADAPTERS}
    report = build_notebook_completion_report(target_results=results, preflight_ok=True)
    assert report["pass"] is True
    assert report["passed_target_count"] == 8
    results["tomato__leaf"] = {"pass": False, "status": "blocked"}
    report = build_notebook_completion_report(target_results=results, preflight_ok=True)
    assert report["pass"] is False
    assert report["blocked_targets"] == ["tomato__leaf"]


def test_manifest_preflight_blocks_only_affected_targets_unless_error_is_global() -> None:
    targeted = evidence_preflight_blocked_targets(
        {
            "issues": [
                {"severity": "error", "code": "id_train_class_floor", "target": "grape__fruit"},
                {"severity": "warning", "code": "note", "target": "tomato__leaf"},
            ]
        },
        required_targets=TARGET_ADAPTERS,
    )
    global_failure = evidence_preflight_blocked_targets(
        {"issues": [{"severity": "error", "code": "manifest_unreadable", "target": ""}]},
        required_targets=TARGET_ADAPTERS,
    )

    assert targeted == {"grape__fruit"}
    assert global_failure == set(TARGET_ADAPTERS)


def test_nb17_report_builder_survives_shared_notebook_completion_name_collision(tmp_path) -> None:
    script_path = "scripts/notebook_cells/nb17_cell03_run_recovery.py"
    source = open(script_path, encoding="utf-8").read().split("\nROOT = ", maxsplit=1)[0]
    namespace = {"RECOVERY_PREFLIGHT_ERROR": "", "CAMPAIGN_PATH": tmp_path / "campaign.json"}
    exec(compile(source, script_path, "exec"), namespace)

    def build_notebook_completion_report(**kwargs):
        raise TypeError(f"wrong shared helper called with {kwargs}")

    namespace["build_notebook_completion_report"] = build_notebook_completion_report
    report = namespace["_build_recovery_report"](
        target_results={target: {"pass": True, "status": "passed"} for target in TARGET_ADAPTERS},
        preflight_ok=True,
        report_path=tmp_path / "summary.json",
        completed_experiment_count=1,
        max_completed_experiments=0,
        max_targets=0,
        phase="checkpoint",
    )

    assert report["pass"] is True
    assert report["passed_target_count"] == len(TARGET_ADAPTERS)


def test_nb17_publish_report_uses_authenticated_path_publisher(tmp_path) -> None:
    script_path = "scripts/notebook_cells/nb17_cell03_run_recovery.py"
    source = open(script_path, encoding="utf-8").read().split("\nROOT = ", maxsplit=1)[0]
    namespace = {}
    exec(compile(source, script_path, "exec"), namespace)
    calls = []

    def push_paths(root, paths, *, commit_message):
        calls.append((root, list(paths), commit_message))
        return {"pushed": True, "staged_files": list(paths)}

    namespace["push_repo_paths_to_github"] = push_paths
    namespace["RECOVERY_AUTO_PUSH_TO_GITHUB"] = True
    report_path = tmp_path / "docs" / "ablation_results" / "adapter_ood_oe_recovery_notebook" / "ts" / "summary.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("{}", encoding="utf-8")

    result = namespace["_publish_report"](tmp_path, report_path)

    assert calls == [
        (
            tmp_path,
            ["docs/ablation_results/adapter_ood_oe_recovery_notebook/ts/summary.json"],
            "Add adapter OOD/OE recovery report ts",
        )
    ]
    assert result["publish_ok"] is True


def test_campaign_ledger_resumes_only_matching_readable_dev_report(tmp_path) -> None:
    lineage = build_campaign_lineage(
        input_revision="abc",
        campaign_digest="campaign",
        evidence_manifest_digest="manifest",
    )
    ledger_path = tmp_path / "campaign_ledger.json"
    ledger = load_campaign_ledger(ledger_path, lineage=lineage)
    report = build_behavioral_dev_report(
        classification_metrics={"accuracy": 0.94, "balanced_accuracy": 0.91, "macro_f1": 0.91},
        prediction_rows=[],
    )
    report_path = tmp_path / "adapter_behavioral_dev_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    adapter_artifact_path = tmp_path / "adapter"
    adapter_artifact_path.mkdir()
    config_digest = json_digest({"seed": 42})
    ledger["experiments"]["candidate-a"] = {
        "status": "completed",
        "resolved_config_digest": config_digest,
        "dev_report_path": str(report_path),
        "adapter_artifact_path": str(adapter_artifact_path),
    }
    write_campaign_ledger(ledger_path, ledger)

    loaded = load_campaign_ledger(ledger_path, lineage=lineage)
    resumed_path, resumed = resumable_experiment(
        loaded,
        experiment_id="candidate-a",
        resolved_config_digest=config_digest,
    )

    assert resumed_path == report_path
    assert resumed["authoritative"] is False
    assert resumable_experiment(
        loaded,
        experiment_id="candidate-a",
        resolved_config_digest="different",
    ) == (None, {})
