import json
from pathlib import Path

from src.training.services.run_registry import build_run_registry, collect_trial_records


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _canonical_record(*, run_id: str, cohort_key: str, artifact_root: Path) -> tuple[dict, dict]:
    manifest = {
        "schema_version": "v1_training_experiment_manifest",
        "record_quality": "canonical",
        "run_id": run_id,
        "run_label": run_id,
        "created_at": "2026-04-14T12:00:00+00:00",
        "surface": "notebook_2",
        "crop_name": "tomato",
        "part_name": "leaf",
        "dataset_key": "tomato__leaf",
        "dataset_lineage_key": "tomato__leaf::sha_a",
        "model_family": {"engine": "continual_sd_lora", "backbone_model_name": "fake/backbone"},
        "artifacts": {"artifact_root": str(artifact_root)},
    }
    optimization = {
        "schema_version": "v1_training_optimization_record",
        "record_quality": "canonical",
        "run_id": run_id,
        "run_label": run_id,
        "created_at": "2026-04-14T12:00:00+00:00",
        "surface": "notebook_2",
        "crop_name": "tomato",
        "part_name": "leaf",
        "dataset_key": "tomato__leaf",
        "dataset_lineage_key": "tomato__leaf::sha_a",
        "comparability": {
            "dataset_lineage_key": "tomato__leaf::sha_a",
            "crop_name": "tomato",
            "part_name": "leaf",
            "engine": "continual_sd_lora",
            "backbone_model_name": "fake/backbone",
            "cohort_key": cohort_key,
        },
        "status": {
            "readiness_status": "ready",
            "readiness_passed": True,
            "authoritative_split": "test",
            "ood_evidence_source": "real_ood_split",
        },
        "parameters": {"training.learning_rate": 0.00015},
        "objectives": {"classification.macro_f1": 0.93},
        "objective_directions": {"classification.macro_f1": "maximize"},
        "artifacts": {"artifact_root": str(artifact_root)},
    }
    return manifest, optimization


def test_collect_trial_records_dedupes_mirrored_notebook_artifacts(tmp_path: Path):
    runs_root = tmp_path / "runs"
    outputs_root = runs_root / "run_1" / "outputs" / "colab_notebook_training" / "artifacts"
    telemetry_root = runs_root / "run_1" / "telemetry" / "artifacts"
    for root in (outputs_root, telemetry_root):
        manifest, optimization = _canonical_record(
            run_id="tomato_leaf_2026-04-14_12-00-00",
            cohort_key="cohort_a",
            artifact_root=root,
        )
        _write_json(root / "training" / "experiment_manifest.json", manifest)
        _write_json(root / "training" / "optimization_record.json", optimization)

    trials = collect_trial_records(runs_root)

    assert len(trials) == 1
    assert trials[0]["record_quality"] == "canonical"
    assert str(trials[0]["registry_source"]["artifact_root"]).replace("\\", "/").endswith(
        "outputs/colab_notebook_training/artifacts"
    )


def test_collect_trial_records_resolves_nested_crop_part_run_dir(tmp_path: Path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "tomato" / "leaf" / "tomato_leaf_2026-04-14_12-00-00"
    artifact_root = run_dir / "outputs" / "colab_notebook_training" / "artifacts"
    manifest, optimization = _canonical_record(
        run_id="tomato_leaf_2026-04-14_12-00-00",
        cohort_key="cohort_a",
        artifact_root=artifact_root,
    )
    _write_json(artifact_root / "training" / "experiment_manifest.json", manifest)
    _write_json(artifact_root / "training" / "optimization_record.json", optimization)

    trials = collect_trial_records(runs_root)

    assert len(trials) == 1
    assert Path(trials[0]["registry_source"]["run_dir"]) == run_dir


def test_build_run_registry_can_write_bayesian_recommendations_when_enabled(tmp_path: Path):
    runs_root = tmp_path / "runs"
    cohort_key = "tomato__leaf::sha_a::tomato::leaf::continual_sd_lora::fake/backbone"
    for index, learning_rate in enumerate((0.00010, 0.00014, 0.00018), start=1):
        artifact_root = runs_root / f"run_{index}" / "training_metrics"
        manifest, optimization = _canonical_record(
            run_id=f"run_{index}",
            cohort_key=cohort_key,
            artifact_root=artifact_root,
        )
        optimization["parameters"] = {"training.learning_rate": learning_rate}
        optimization["objectives"] = {
            "classification.macro_f1": 0.80 + index * 0.02,
            "ood.ood_auroc": 0.75 + index * 0.02,
            "ood.ood_false_positive_rate": 0.20 - index * 0.02,
        }
        optimization["objective_directions"] = {
            "classification.macro_f1": "maximize",
            "ood.ood_auroc": "maximize",
            "ood.ood_false_positive_rate": "minimize",
        }
        _write_json(artifact_root / "training" / "experiment_manifest.json", manifest)
        _write_json(artifact_root / "training" / "optimization_record.json", optimization)

    result = build_run_registry(
        runs_root=runs_root,
        enable_bayesian_proposals=True,
        proposal_count=2,
        candidate_pool_size=16,
        search_space_payload={
            "parameters": [
                {"name": "training.learning_rate", "type": "float", "low": 5e-5, "high": 3e-4, "scale": "log"}
            ]
        },
    )

    recommendations_path = runs_root / "_index" / "bayesian_recommendations.json"
    recommendations = json.loads(recommendations_path.read_text(encoding="utf-8"))
    assert result["latest_registry"]["bayesian_optimization_enabled"] is True
    assert result["latest_registry"]["paths"]["bayesian_recommendations_json"].endswith(
        "bayesian_recommendations.json"
    )
    assert recommendations["cohort_count"] == 1
    assert len(recommendations["cohorts"][0]["proposals"]) == 2
