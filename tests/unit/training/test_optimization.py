from src.training.services.optimization import (
    build_bayesian_recommendations,
    build_pareto_frontiers,
    build_training_config_override,
)


def _trial(
    *,
    run_id: str,
    macro_f1: float,
    ood_auroc: float,
    ood_fpr: float,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    batch_size: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    fusion_dropout: float,
    threshold_factor: float,
    logitnorm_tau: float,
    randaugment_magnitude: int,
    randaugment_num_ops: int = 2,
    readiness_passed: bool = True,
) -> dict:
    parameters = {
        "training.learning_rate": learning_rate,
        "training.weight_decay": weight_decay,
        "training.num_epochs": num_epochs,
        "training.batch_size": batch_size,
        "training.adapter.lora_r": lora_r,
        "training.adapter.lora_alpha": lora_alpha,
        "training.adapter.lora_dropout": lora_dropout,
        "training.fusion.dropout": fusion_dropout,
        "training.ood.threshold_factor": threshold_factor,
        "training.ood.react_enabled": False,
        "training.ood.react_percentile": 0.99,
        "training.optimization.logitnorm_tau": logitnorm_tau,
        "training.optimization.label_smoothing": 0.0,
        "training.data.augmentation_policy": "randaugment",
        "training.data.randaugment_num_ops": randaugment_num_ops,
        "training.data.randaugment_magnitude": randaugment_magnitude,
        "training.data.augmix_severity": 3,
        "training.classifier_rebalance.enabled": False,
        "training.classifier_rebalance.logit_adjustment_tau": 1.0,
    }
    return {
        "run_id": run_id,
        "run_label": run_id,
        "created_at": f"2026-04-14T12:00:0{run_id[-1]}+00:00",
        "record_quality": "canonical",
        "status": {
            "readiness_status": "ready" if readiness_passed else "failed",
            "readiness_passed": readiness_passed,
            "authoritative_split": "test",
            "ood_evidence_source": "real_ood_split",
        },
        "comparability": {
            "dataset_lineage_key": "tomato__leaf::sha_a",
            "crop_name": "tomato",
            "part_name": "leaf",
            "engine": "continual_sd_lora",
            "backbone_model_name": "fake/backbone",
            "cohort_key": "tomato__leaf::sha_a::tomato::leaf::continual_sd_lora::fake/backbone",
        },
        "parameters": parameters,
        "objectives": {
            "classification.macro_f1": macro_f1,
            "ood.ood_auroc": ood_auroc,
            "ood.ood_false_positive_rate": ood_fpr,
        },
        "objective_directions": {
            "classification.macro_f1": "maximize",
            "ood.ood_auroc": "maximize",
            "ood.ood_false_positive_rate": "minimize",
        },
        "registry_source": {"artifact_root": f"/tmp/{run_id}"},
    }


def test_build_training_config_override_maps_flat_parameters():
    override = build_training_config_override(
        {
            "training.learning_rate": 0.0002,
            "training.adapter.lora_r": 32,
            "training.data.randaugment_num_ops": 3,
            "training.data.randaugment_magnitude": 9,
        }
    )

    assert override == {
        "training": {
            "continual": {
                "learning_rate": 0.0002,
                "adapter": {"lora_r": 32},
                "data": {"randaugment_num_ops": 3, "randaugment_magnitude": 9},
            }
        }
    }


def test_build_pareto_frontiers_keeps_only_non_dominated_runs():
    trials = [
        _trial(
            run_id="run1",
            macro_f1=0.80,
            ood_auroc=0.70,
            ood_fpr=0.20,
            learning_rate=0.00010,
            weight_decay=0.01,
            num_epochs=10,
            batch_size=8,
            lora_r=16,
            lora_alpha=16,
            lora_dropout=0.10,
            fusion_dropout=0.10,
            threshold_factor=3.0,
            logitnorm_tau=1.0,
            randaugment_magnitude=7,
        ),
        _trial(
            run_id="run2",
            macro_f1=0.85,
            ood_auroc=0.78,
            ood_fpr=0.18,
            learning_rate=0.00012,
            weight_decay=0.012,
            num_epochs=12,
            batch_size=8,
            lora_r=24,
            lora_alpha=24,
            lora_dropout=0.09,
            fusion_dropout=0.08,
            threshold_factor=2.8,
            logitnorm_tau=1.1,
            randaugment_magnitude=8,
        ),
        _trial(
            run_id="run3",
            macro_f1=0.84,
            ood_auroc=0.83,
            ood_fpr=0.10,
            learning_rate=0.00016,
            weight_decay=0.008,
            num_epochs=14,
            batch_size=12,
            lora_r=24,
            lora_alpha=24,
            lora_dropout=0.12,
            fusion_dropout=0.09,
            threshold_factor=2.6,
            logitnorm_tau=0.9,
            randaugment_magnitude=6,
        ),
        _trial(
            run_id="run4",
            macro_f1=0.82,
            ood_auroc=0.75,
            ood_fpr=0.25,
            learning_rate=0.00022,
            weight_decay=0.02,
            num_epochs=16,
            batch_size=16,
            lora_r=32,
            lora_alpha=32,
            lora_dropout=0.15,
            fusion_dropout=0.14,
            threshold_factor=3.5,
            logitnorm_tau=1.3,
            randaugment_magnitude=9,
        ),
        _trial(
            run_id="run5",
            macro_f1=0.90,
            ood_auroc=0.90,
            ood_fpr=0.05,
            learning_rate=0.00018,
            weight_decay=0.01,
            num_epochs=18,
            batch_size=8,
            lora_r=24,
            lora_alpha=24,
            lora_dropout=0.10,
            fusion_dropout=0.10,
            threshold_factor=2.5,
            logitnorm_tau=1.0,
            randaugment_magnitude=8,
            readiness_passed=False,
        ),
    ]

    payload = build_pareto_frontiers(trials)
    cohort = payload["cohorts"][0]

    assert cohort["eligible_run_count"] == 5
    assert cohort["excluded_run_count"] == 0
    assert set(cohort["frontier_run_ids"]) == {"run5"}

    strict_payload = build_pareto_frontiers(trials, require_readiness_passed=True)
    strict_cohort = strict_payload["cohorts"][0]
    assert strict_cohort["eligible_run_count"] == 4
    assert strict_cohort["excluded_run_count"] == 1
    assert set(strict_cohort["frontier_run_ids"]) == {"run2", "run3"}


def test_build_bayesian_recommendations_proposes_new_parameters():
    trials = [
        _trial(
            run_id="run1",
            macro_f1=0.80,
            ood_auroc=0.70,
            ood_fpr=0.20,
            learning_rate=0.00010,
            weight_decay=0.01,
            num_epochs=10,
            batch_size=8,
            lora_r=16,
            lora_alpha=16,
            lora_dropout=0.10,
            fusion_dropout=0.10,
            threshold_factor=3.0,
            logitnorm_tau=1.0,
            randaugment_magnitude=7,
        ),
        _trial(
            run_id="run2",
            macro_f1=0.85,
            ood_auroc=0.78,
            ood_fpr=0.18,
            learning_rate=0.00012,
            weight_decay=0.012,
            num_epochs=12,
            batch_size=8,
            lora_r=24,
            lora_alpha=24,
            lora_dropout=0.09,
            fusion_dropout=0.08,
            threshold_factor=2.8,
            logitnorm_tau=1.1,
            randaugment_magnitude=8,
        ),
        _trial(
            run_id="run3",
            macro_f1=0.84,
            ood_auroc=0.83,
            ood_fpr=0.10,
            learning_rate=0.00016,
            weight_decay=0.008,
            num_epochs=14,
            batch_size=12,
            lora_r=24,
            lora_alpha=24,
            lora_dropout=0.12,
            fusion_dropout=0.09,
            threshold_factor=2.6,
            logitnorm_tau=0.9,
            randaugment_magnitude=6,
        ),
        _trial(
            run_id="run4",
            macro_f1=0.82,
            ood_auroc=0.75,
            ood_fpr=0.25,
            learning_rate=0.00022,
            weight_decay=0.02,
            num_epochs=16,
            batch_size=16,
            lora_r=32,
            lora_alpha=32,
            lora_dropout=0.15,
            fusion_dropout=0.14,
            threshold_factor=3.5,
            logitnorm_tau=1.3,
            randaugment_magnitude=9,
        ),
    ]

    payload = build_bayesian_recommendations(trials, proposal_count=2, random_seed=7)
    cohort = payload["cohorts"][0]
    observed_signatures = {
        tuple(sorted(trial["parameters"].items()))
        for trial in trials
    }

    assert cohort["eligible_run_count"] == 4
    assert len(cohort["proposals"]) == 2
    for proposal in cohort["proposals"]:
        assert proposal["parameters"]
        assert proposal["config_override"]["training"]["continual"]
        signature = tuple(sorted(proposal["parameters"].items()))
        assert signature not in observed_signatures
