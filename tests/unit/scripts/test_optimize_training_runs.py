import argparse
import json
from pathlib import Path

import pytest

from scripts.optimize_training_runs import run_optimizer


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _canonical_record(*, run_id: str, artifact_root: Path, created_at: str, macro_f1: float, auroc: float, fpr: float, lr: float) -> tuple[dict, dict]:
    cohort_key = "tomato__leaf::sha_a::tomato::leaf::continual_sd_lora::fake/backbone"
    manifest = {
        "schema_version": "v1_training_experiment_manifest",
        "record_quality": "canonical",
        "run_id": run_id,
        "run_label": run_id,
        "created_at": created_at,
        "surface": "workflow",
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
        "created_at": created_at,
        "surface": "workflow",
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
        "parameters": {
            "training.learning_rate": lr,
            "training.weight_decay": 0.01,
            "training.num_epochs": 12,
            "training.batch_size": 8,
            "training.adapter.lora_r": 24,
            "training.adapter.lora_alpha": 24,
            "training.adapter.lora_dropout": 0.1,
            "training.fusion.dropout": 0.1,
            "training.ood.threshold_factor": 3.0,
            "training.optimization.logitnorm_tau": 1.0,
            "training.data.randaugment_num_ops": 2,
            "training.data.randaugment_magnitude": 7,
        },
        "objectives": {
            "classification.macro_f1": macro_f1,
            "ood.ood_auroc": auroc,
            "ood.ood_false_positive_rate": fpr,
        },
        "objective_directions": {
            "classification.macro_f1": "maximize",
            "ood.ood_auroc": "maximize",
            "ood.ood_false_positive_rate": "minimize",
        },
        "artifacts": {"artifact_root": str(artifact_root)},
    }
    return manifest, optimization


def _materialize_runs(runs_root: Path) -> None:
    for index, values in enumerate(
        [
            ("run_a", "2026-04-14T12:00:00+00:00", 0.81, 0.74, 0.20, 0.00010),
            ("run_b", "2026-04-14T12:10:00+00:00", 0.84, 0.79, 0.17, 0.00014),
            ("run_c", "2026-04-14T12:20:00+00:00", 0.83, 0.82, 0.11, 0.00018),
        ]
    ):
        run_id, created_at, macro_f1, auroc, fpr, lr = values
        artifact_root = runs_root / f"trial_{index}" / "training_metrics"
        manifest, optimization = _canonical_record(
            run_id=run_id,
            artifact_root=artifact_root,
            created_at=created_at,
            macro_f1=macro_f1,
            auroc=auroc,
            fpr=fpr,
            lr=lr,
        )
        _write_json(artifact_root / "training" / "experiment_manifest.json", manifest)
        _write_json(artifact_root / "training" / "optimization_record.json", optimization)


def test_run_optimizer_analyzes_single_cohort(tmp_path: Path):
    runs_root = tmp_path / "runs"
    _materialize_runs(runs_root)
    args = argparse.Namespace(
        runs_root=runs_root,
        index_root=runs_root / "_index",
        cohort_key=None,
        dataset_lineage_key="tomato__leaf::sha_a",
        dataset_key=None,
        crop_name="tomato",
        part_name="leaf",
        backbone_model_name="fake/backbone",
        engine="continual_sd_lora",
        objectives=[],
        proposal_count=2,
        candidate_pool_size=64,
        random_seed=11,
        search_space=None,
        execute=False,
        config_env="colab",
        device="cpu",
        data_dir=None,
        run_output_root=runs_root,
        num_workers=None,
        validation_every_n_epochs=None,
        enable_bayesian_optimization=False,
    )

    result = run_optimizer(args)

    assert result["selected_cohort"]["comparability"]["cohort_key"].endswith("fake/backbone")
    assert result["selected_cohort"]["pareto_frontier"]["frontier_count"] >= 1
    assert result["bayesian_optimization_enabled"] is False
    assert result["selected_cohort"]["bayesian_recommendations"]["proposal_count"] == 0
    assert result["selected_cohort"]["bayesian_recommendations"]["search_strategy"] == "disabled"
    assert Path(result["registry_paths"]["pareto_frontiers_json"]).exists()
    assert "bayesian_recommendations_json" not in result["registry_paths"]


def test_run_optimizer_writes_adapter_cohort_bayesian_proposals_when_enabled(tmp_path: Path):
    runs_root = tmp_path / "runs"
    _materialize_runs(runs_root)
    search_space_json = tmp_path / "search_space.json"
    _write_json(
        search_space_json,
        {
            "parameters": [
                {"name": "training.learning_rate", "type": "float", "low": 5e-5, "high": 3e-4, "scale": "log"}
            ]
        },
    )
    args = argparse.Namespace(
        runs_root=runs_root,
        index_root=runs_root / "_index",
        cohort_key=None,
        dataset_lineage_key="tomato__leaf::sha_a",
        dataset_key=None,
        crop_name="tomato",
        part_name="leaf",
        backbone_model_name="fake/backbone",
        engine="continual_sd_lora",
        objectives=[],
        proposal_count=2,
        candidate_pool_size=32,
        random_seed=11,
        search_space_json=search_space_json,
        enable_bayesian_optimization=True,
        execute=False,
        config_env="colab",
        device="cpu",
        data_dir=None,
        run_output_root=runs_root,
        num_workers=None,
        validation_every_n_epochs=None,
    )

    result = run_optimizer(args)

    recommendations = result["selected_cohort"]["bayesian_recommendations"]
    assert result["bayesian_optimization_enabled"] is True
    assert recommendations["enabled"] is True
    assert recommendations["eligible_run_count"] == 3
    assert recommendations["proposal_count"] == 2
    assert Path(result["registry_paths"]["bayesian_recommendations_json"]).exists()


def test_run_optimizer_executes_proposals(monkeypatch, tmp_path: Path):
    import src.workflows.training as training_module

    runs_root = tmp_path / "runs"
    _materialize_runs(runs_root)
    calls: list[dict] = []

    class FakeResult:
        def __init__(self, run_id: str) -> None:
            self._run_id = run_id

        def to_dict(self) -> dict:
            return {"run_id": self._run_id, "status": "completed"}

    class FakeWorkflow:
        def __init__(self, *, config, environment, device):
            calls.append({"init": {"config": config, "environment": environment, "device": device}})

        def run(self, *, crop_name, data_dir, output_dir, num_workers=None, validation_every_n_epochs=None, run_id=""):
            calls.append(
                {
                    "run": {
                        "crop_name": crop_name,
                        "data_dir": data_dir,
                        "output_dir": output_dir,
                        "num_workers": num_workers,
                        "validation_every_n_epochs": validation_every_n_epochs,
                        "run_id": run_id,
                    }
                }
            )
            return FakeResult(run_id)

    monkeypatch.setattr(training_module, "TrainingWorkflow", FakeWorkflow)
    args = argparse.Namespace(
        runs_root=runs_root,
        index_root=runs_root / "_index",
        cohort_key=None,
        dataset_lineage_key="tomato__leaf::sha_a",
        dataset_key=None,
        crop_name="tomato",
        part_name="leaf",
        backbone_model_name="fake/backbone",
        engine="continual_sd_lora",
        objectives=[],
        proposal_count=1,
        candidate_pool_size=32,
        random_seed=5,
        search_space=None,
        execute=True,
        config_env="colab",
        device="cpu",
        data_dir=tmp_path / "runtime_dataset",
        run_output_root=runs_root,
        num_workers=2,
        validation_every_n_epochs=3,
    )

    with pytest.raises(ValueError, match="Bayesian optimization execution is disabled"):
        run_optimizer(args)
    assert calls == []


def test_run_optimizer_blocks_execute_without_eligible_bayesian_evidence(monkeypatch, tmp_path: Path):
    import src.workflows.training as training_module

    runs_root = tmp_path / "runs"
    _materialize_runs(runs_root)

    class FailIfCalledWorkflow:
        def __init__(self, *, config, environment, device):
            raise AssertionError("TrainingWorkflow should not be constructed when execution is blocked")

    monkeypatch.setattr(training_module, "TrainingWorkflow", FailIfCalledWorkflow)

    args = argparse.Namespace(
        runs_root=runs_root,
        index_root=runs_root / "_index",
        cohort_key=None,
        dataset_lineage_key="tomato__leaf::sha_a",
        dataset_key=None,
        crop_name="tomato",
        part_name="leaf",
        backbone_model_name="fake/backbone",
        engine="continual_sd_lora",
        objectives=[],
        proposal_count=1,
        candidate_pool_size=32,
        random_seed=5,
        search_space=None,
        execute=True,
        allow_bootstrap_execute=False,
        config_env="colab",
        device="cpu",
        data_dir=tmp_path / "runtime_dataset",
        run_output_root=runs_root,
        num_workers=2,
        validation_every_n_epochs=3,
    )

    from scripts.optimize_training_runs import run_optimizer as run_optimizer_local

    with pytest.raises(ValueError, match="Bayesian optimization execution is disabled"):
        run_optimizer_local(args)


def test_run_optimizer_allows_execute_with_bootstrap_override(monkeypatch, tmp_path: Path):
    import src.workflows.training as training_module

    runs_root = tmp_path / "runs"
    _materialize_runs(runs_root)
    calls: list[dict] = []

    class FakeResult:
        def __init__(self, run_id: str) -> None:
            self._run_id = run_id

        def to_dict(self) -> dict:
            return {"run_id": self._run_id, "status": "completed"}

    class FakeWorkflow:
        def __init__(self, *, config, environment, device):
            calls.append({"init": {"config": config, "environment": environment, "device": device}})

        def run(self, *, crop_name, data_dir, output_dir, num_workers=None, validation_every_n_epochs=None, run_id=""):
            calls.append(
                {
                    "run": {
                        "crop_name": crop_name,
                        "data_dir": data_dir,
                        "output_dir": output_dir,
                        "num_workers": num_workers,
                        "validation_every_n_epochs": validation_every_n_epochs,
                        "run_id": run_id,
                    }
                }
            )
            return FakeResult(run_id)

    monkeypatch.setattr(training_module, "TrainingWorkflow", FakeWorkflow)

    args = argparse.Namespace(
        runs_root=runs_root,
        index_root=runs_root / "_index",
        cohort_key=None,
        dataset_lineage_key="tomato__leaf::sha_a",
        dataset_key=None,
        crop_name="tomato",
        part_name="leaf",
        backbone_model_name="fake/backbone",
        engine="continual_sd_lora",
        objectives=[],
        proposal_count=1,
        candidate_pool_size=32,
        random_seed=5,
        search_space=None,
        execute=True,
        allow_bootstrap_execute=True,
        config_env="colab",
        device="cpu",
        data_dir=tmp_path / "runtime_dataset",
        run_output_root=runs_root,
        num_workers=2,
        validation_every_n_epochs=3,
    )

    with pytest.raises(ValueError, match="Bayesian optimization execution is disabled"):
        run_optimizer(args)
    assert calls == []
