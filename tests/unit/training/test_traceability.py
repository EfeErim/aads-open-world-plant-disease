import json
from pathlib import Path

from src.training.services.traceability import (
    build_experiment_manifest,
    build_optimization_record,
    persist_traceability_artifacts,
)


def _summary_payload() -> dict:
    return {
        "run_id": "tomato_leaf_run_1",
        "run_label": "tomato_leaf_run_1",
        "created_at": "2026-04-14T10:00:00+00:00",
        "surface": "workflow",
        "crop_name": "tomato",
        "part_name": "leaf",
        "dataset_key": "tomato__leaf",
        "class_count": 2,
        "loader_sizes": {"train": 120, "val": 30, "test": 30, "ood": 20},
        "optimizer_steps": 64,
        "global_step": 128,
        "checkpoint_count": 3,
    }


def _run_context_payload() -> dict:
    return {
        "run_id": "tomato_leaf_run_1",
        "created_at": "2026-04-14T10:00:00+00:00",
        "crop_name": "tomato",
        "part_name": "leaf",
        "device": "cpu",
        "python_version": "3.11.9",
        "git": {"head": "abc123", "branch": "main", "is_dirty": False},
        "package_versions": {"torch": "2.10.0"},
        "resolved_config": {
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake/backbone"},
                    "adapter": {"lora_r": 24, "lora_alpha": 24, "lora_dropout": 0.1},
                    "fusion": {"layers": [2, 5, 8, 11], "output_dim": 768, "dropout": 0.1, "gating": "softmax"},
                    "ood": {
                        "threshold_factor": 3.0,
                        "primary_score_method": "ensemble",
                        "energy_temperature_mode": "auto",
                        "react_enabled": True,
                        "react_percentile": 0.98,
                        "oe_enabled": True,
                        "oe_loss_weight": 0.5,
                        "oe_target": "uniform",
                        "radial_l2_enabled": True,
                        "radial_beta_range": [0.5, 2.0],
                        "radial_beta_steps": 16,
                        "sure_enabled": True,
                        "sure_semantic_percentile": 90.0,
                        "sure_confidence_percentile": 97.0,
                        "conformal_enabled": True,
                        "conformal_alpha": 0.05,
                        "conformal_method": "raps",
                        "conformal_raps_lambda": 0.2,
                        "conformal_raps_k_reg": 1,
                    },
                    "learning_rate": 0.00015,
                    "weight_decay": 0.01,
                    "num_epochs": 16,
                    "batch_size": 8,
                    "seed": 42,
                    "optimization": {
                        "loss_name": "logitnorm",
                        "logitnorm_tau": 1.0,
                        "label_smoothing": 0.1,
                        "grad_accumulation_steps": 4,
                        "mixed_precision": "auto",
                        "max_grad_norm": 1.0,
                        "scheduler": {
                            "name": "cosine",
                            "warmup_ratio": 0.1,
                            "min_lr": 1e-6,
                            "step_on": "batch",
                        },
                    },
                    "data": {
                        "sampler": "auto",
                        "augmentation_policy": "augmix",
                        "randaugment_num_ops": 2,
                        "randaugment_magnitude": 7,
                        "augmix_severity": 3,
                    },
                    "classifier_rebalance": {
                        "enabled": True,
                        "epochs": 3,
                        "learning_rate": 5e-5,
                        "sampler": "weighted",
                        "objective": "logit_adjusted_cross_entropy",
                    },
                }
            }
        },
        "dataset": {
            "crop_root": "/tmp/runtime/tomato__leaf",
            "dataset_key": "tomato__leaf",
            "resolution_source": "manifest:split_manifest.json",
            "manifests": {
                "split_manifest.json": {
                    "path": "/tmp/runtime/tomato__leaf/split_manifest.json",
                    "exists": True,
                    "sha256": "sha256_dataset",
                    "schema_version": "v1_grouped_runtime_layout",
                    "source_root": "/tmp/source/tomato_leaf",
                    "crop_name": "tomato",
                    "part_name": "leaf",
                    "dataset_key": "tomato__leaf",
                    "split_policy": "grouped_family_canonical_eval_60_20_20",
                    "ood": {
                        "source_root": "/tmp/ood/tomato_leaf",
                        "image_count": 20,
                        "image_fingerprint": "ood_fp_1",
                    },
                }
            },
        },
        "training_runtime": {
            "train_sampler": {"resolved_sampler": "weighted"},
        },
    }


def _production_readiness_payload() -> dict:
    return {
        "status": "ready",
        "passed": True,
        "ood_evidence_source": "real_ood_split",
        "classification_evidence": {
            "split_name": "test",
            "metrics": {
                "accuracy": 0.95,
                "balanced_accuracy": 0.94,
                "macro_f1": 0.93,
            },
        },
        "ood_evidence": {
            "metrics": {
                "ood_auroc": 0.91,
                "ood_false_positive_rate": 0.04,
                "sure_ds_f1": 0.90,
                "conformal_empirical_coverage": 0.96,
                "conformal_avg_set_size": 1.2,
                "ood_samples": 20,
                "in_distribution_samples": 30,
            }
        },
    }


def _authoritative_artifacts() -> dict:
    return {
        "report_dict": {
            "accuracy": 0.95,
            "macro avg": {"f1-score": 0.93},
            "weighted avg": {"f1-score": 0.92},
        },
        "metric_gate": {
            "metrics": {
                "accuracy": 0.95,
                "balanced_accuracy": 0.94,
                "macro_f1": 0.93,
                "classification_samples": 30,
            }
        },
    }


def test_build_experiment_manifest_includes_dataset_lineage_and_ood_provenance(tmp_path: Path):
    manifest = build_experiment_manifest(
        summary_payload=_summary_payload(),
        run_context_payload=_run_context_payload(),
        artifact_root=tmp_path / "training_metrics",
        explicit_surface="workflow",
        created_at="2026-04-14T10:00:00+00:00",
        record_quality="canonical",
    )

    assert manifest["dataset_lineage_key"] == "tomato__leaf::sha256_dataset"
    assert manifest["dataset"]["manifest"]["split_policy"] == "grouped_family_canonical_eval_60_20_20"
    assert manifest["dataset"]["ood"]["image_count"] == 20
    assert manifest["model_family"]["engine"] == "continual_sd_lora"


def test_build_optimization_record_flattens_parameters_and_sets_objective_directions(tmp_path: Path):
    record = build_optimization_record(
        summary_payload=_summary_payload(),
        run_context_payload=_run_context_payload(),
        production_readiness_payload=_production_readiness_payload(),
        authoritative_artifacts=_authoritative_artifacts(),
        artifact_root=tmp_path / "training_metrics",
        explicit_surface="workflow",
        created_at="2026-04-14T10:00:00+00:00",
        record_quality="canonical",
    )

    assert record["comparability"]["cohort_key"] == (
        "tomato__leaf::sha256_dataset::tomato::leaf::continual_sd_lora::fake/backbone"
    )
    assert record["parameters"]["training.learning_rate"] == 0.00015
    assert record["parameters"]["training.adapter.lora_r"] == 24
    assert record["parameters"]["training.optimization.label_smoothing"] == 0.1
    assert record["parameters"]["training.ood.energy_temperature_mode"] == "auto"
    assert record["parameters"]["training.ood.react_enabled"] is True
    assert record["parameters"]["training.ood.oe_enabled"] is True
    assert record["parameters"]["training.data.augmentation_policy"] == "augmix"
    assert record["parameters"]["training.classifier_rebalance.enabled"] is True
    assert record["objectives"]["classification.weighted_f1"] == 0.92
    assert record["objectives"]["ood.ood_false_positive_rate"] == 0.04
    assert record["objective_directions"]["classification.weighted_f1"] == "maximize"
    assert record["objective_directions"]["ood.ood_false_positive_rate"] == "minimize"


def test_persist_traceability_artifacts_writes_files(tmp_path: Path):
    artifact_root = tmp_path / "training_metrics"
    (artifact_root / "training").mkdir(parents=True, exist_ok=True)
    (artifact_root / "training" / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "run_1",
                "run_label": "run_1",
                "surface": "workflow",
                "crop_name": "tomato",
                "part_name": "leaf",
                "dataset_key": "tomato__leaf",
            }
        ),
        encoding="utf-8",
    )
    (artifact_root / "production_readiness.json").write_text("{}", encoding="utf-8")
    (artifact_root / "test").mkdir(parents=True, exist_ok=True)
    (artifact_root / "test" / "metric_gate.json").write_text("{}", encoding="utf-8")

    persist_traceability_artifacts(
        artifact_root=artifact_root,
        experiment_manifest=build_experiment_manifest(
            summary_payload=_summary_payload(),
            run_context_payload=_run_context_payload(),
            artifact_root=artifact_root,
            explicit_surface="workflow",
            created_at="2026-04-14T10:00:00+00:00",
            record_quality="canonical",
        ),
        optimization_record=build_optimization_record(
            summary_payload=_summary_payload(),
            run_context_payload=_run_context_payload(),
            production_readiness_payload=_production_readiness_payload(),
            authoritative_artifacts=_authoritative_artifacts(),
            artifact_root=artifact_root,
            explicit_surface="workflow",
            created_at="2026-04-14T10:00:00+00:00",
            record_quality="canonical",
        ),
    )

    assert (artifact_root / "training" / "experiment_manifest.json").exists()
    assert (artifact_root / "training" / "optimization_record.json").exists()
