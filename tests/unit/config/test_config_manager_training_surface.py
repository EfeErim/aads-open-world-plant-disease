import pytest

from src.core.config_manager import ConfigurationManager
from src.core.config_migrations import CURRENT_CONFIG_SCHEMA_VERSION
from src.training.services.config_surface import extract_continual_training_config


def test_colab_training_surface_normalizes_runtime_aliases():
    cfg = ConfigurationManager(config_dir="config", environment="colab").load_all_configs()

    assert cfg["config_schema_version"] == CURRENT_CONFIG_SCHEMA_VERSION
    colab_training = cfg["colab"]["training"]
    assert colab_training["num_workers"] == 12
    assert colab_training["checkpoint_every_n_steps"] == 250
    assert colab_training["checkpoint_on_exception"] is True
    assert colab_training["validation_every_n_epochs"] == 2
    assert colab_training["stdout_progress_batch_interval"] == 12
    assert colab_training["stdout_progress_min_interval_sec"] == 15.0
    assert "lazy_load" not in cfg["inference"]
    assert "cache_enabled" not in cfg["inference"]


def test_training_continual_surface_exposes_reliability_defaults():
    cfg = ConfigurationManager(config_dir="config", environment="colab").load_all_configs()

    assert cfg["config_schema_version"] == CURRENT_CONFIG_SCHEMA_VERSION
    continual = cfg["training"]["continual"]
    assert continual["ood"]["ber_enabled"] is False
    assert continual["ood"]["ber_lambda_old"] == 0.1
    assert continual["ood"]["ber_lambda_new"] == 0.1
    assert continual["ood"]["ber_warmup_steps"] == 50
    assert continual["ood"]["primary_score_method"] == "auto"
    assert continual["ood"]["real_dev_selection_enabled"] is True
    assert continual["ood"]["real_dev_target_fpr"] == 0.05
    assert continual["ood"]["energy_temperature_mode"] == "auto"
    assert continual["ood"]["react_enabled"] is False
    assert continual["ood"]["oe_enabled"] is True
    assert continual["seed"] == 42
    assert continual["batch_size"] == 96
    assert continual["learning_rate"] == 0.0002
    assert continual["deterministic"] is False
    assert continual["optimization"]["grad_accumulation_steps"] == 1
    assert continual["optimization"]["loss_name"] == "logitnorm"
    assert continual["optimization"]["logitnorm_tau"] == 1.0
    assert continual["optimization"]["scheduler"]["name"] == "cosine"
    assert continual["evaluation"]["best_metric"] == "val_loss"
    assert continual["evaluation"]["require_ood_for_gate"] is True
    assert continual["evaluation"]["ood_benchmark_min_classes"] == 3
    assert continual["evaluation"]["min_in_distribution_samples"] == 30
    assert continual["evaluation"]["min_ood_samples"] == 30
    assert continual["evaluation"]["min_ood_samples_per_type"] == 5
    assert continual["evaluation"]["gate_auxiliary_ood_diagnostics"] is False
    assert continual["data"]["sampler"] == "auto"
    assert continual["data"]["loader_error_policy"] == "tolerant"
    assert continual["data"]["augmentation_policy"] == "randaugment"
    assert continual["data"]["randaugment_num_ops"] == 2
    assert continual["data"]["randaugment_magnitude"] == 7
    assert continual["data"]["augmix_severity"] == 3
    assert continual["data"]["augmix_width"] == 3
    assert continual["data"]["augmix_depth"] == -1
    assert continual["data"]["augmix_alpha"] == 1.0
    assert continual["data"]["allow_under_min_training"] is False
    assert continual["data"]["cache_size"] == 20000
    assert continual["data"]["cache_train_split"] is True
    assert continual["data"]["validate_images_on_init"] is False
    assert continual["classifier_rebalance"]["enabled"] is False
    assert continual["class_balance"]["allow_sampler_and_loss_rebalance"] is False


def test_extract_continual_training_config_normalizes_root_shape():
    root_payload = {
        "training": {
            "continual": {
                "backbone": {"model_name": "demo-model"},
                "evaluation": {"best_metric": "macro_f1"},
            }
        }
    }

    root_normalized = extract_continual_training_config(root_payload, model_name="ignored", device="cuda")

    assert root_normalized["backbone"]["model_name"] == "demo-model"
    assert root_normalized["evaluation"]["best_metric"] == "macro_f1"
    assert root_normalized["optimization"]["scheduler"]["name"] == "cosine"
    assert root_normalized["ood"]["primary_score_method"] == "auto"
    assert root_normalized["ood"]["threshold_factor"] == 3.0
    assert root_normalized["ood"]["ber_enabled"] is False
    assert root_normalized["ood"]["energy_temperature_mode"] == "auto"
    assert root_normalized["ood"]["react_enabled"] is False
    assert root_normalized["ood"]["oe_enabled"] is True
    assert root_normalized["ood"]["sure_semantic_percentile"] == 90.0
    assert root_normalized["ood"]["sure_confidence_percentile"] == 97.0
    assert root_normalized["ood"]["real_split_enabled"] is True
    assert root_normalized["ood"]["real_split_dev_fraction"] == 0.4
    assert root_normalized["ood"]["real_split_min_per_slice"] == 2
    assert root_normalized["ood"]["real_split_min_total"] == 30
    assert root_normalized["ood"]["real_split_manifest_name"] == "ood_split_manifest.json"
    assert root_normalized["ood"]["enforce_oe_disjoint"] is True
    assert root_normalized["ood"]["real_dev_selection_enabled"] is True
    assert root_normalized["ood"]["real_dev_target_fpr"] == 0.05
    assert root_normalized["ood"]["conformal_method"] == "raps"
    assert root_normalized["ood"]["conformal_raps_lambda"] == 0.2
    assert root_normalized["data"]["augmentation_policy"] == "randaugment"
    assert root_normalized["data"]["validate_images_on_init"] is False
    assert root_normalized["data"]["randaugment_num_ops"] == 2
    assert root_normalized["data"]["randaugment_magnitude"] == 7
    assert root_normalized["data"]["augmix_severity"] == 3
    assert root_normalized["data"]["allow_under_min_training"] is False
    assert root_normalized["class_balance"]["allow_sampler_and_loss_rebalance"] is False


def test_extract_continual_training_config_rejects_flat_noncanonical_shape():
    noncanonical_payload = {
        "model_name": "noncanonical-model",
        "lora_r": 4,
        "fusion_output_dim": 256,
        "device": "cpu",
    }

    with pytest.raises(ValueError, match="must be provided under training.continual"):
        extract_continual_training_config(noncanonical_payload, model_name="ignored", device="cuda")


def test_extract_continual_training_config_normalizes_ber_fields():
    payload = {
        "training": {
            "continual": {
                "ood": {
                    "ber_enabled": True,
                    "ber_lambda_old": "0.05",
                    "ber_lambda_new": 0.2,
                }
            }
        }
    }

    normalized = extract_continual_training_config(payload)

    assert normalized["ood"]["ber_enabled"] is True
    assert normalized["ood"]["ber_lambda_old"] == 0.05
    assert normalized["ood"]["ber_lambda_new"] == 0.2


def test_extract_continual_training_config_normalizes_primary_score_method():
    payload = {
        "training": {
            "continual": {
                "ood": {
                    "primary_score_method": "KNN",
                }
            }
        }
    }

    normalized = extract_continual_training_config(payload)

    assert normalized["ood"]["primary_score_method"] == "knn"


def test_extract_continual_training_config_normalizes_logitnorm_fields():
    payload = {
        "training": {
            "continual": {
                "optimization": {
                    "loss_name": "logitnorm",
                    "logitnorm_tau": "0.7",
                }
            }
        }
    }

    normalized = extract_continual_training_config(payload)

    assert normalized["optimization"]["loss_name"] == "logitnorm"
    assert normalized["optimization"]["logitnorm_tau"] == 0.7


def test_extract_continual_training_config_rejects_invalid_augmentation_policy():
    payload = {
        "training": {
            "continual": {
                "data": {
                    "augmentation_policy": "too_much",
                }
            }
        }
    }

    with pytest.raises(ValueError, match="augmentation_policy"):
        extract_continual_training_config(payload)


def test_extract_continual_training_config_normalizes_augmix_oe_and_rebalance_fields():
    payload = {
        "training": {
            "continual": {
                "ood": {
                    "react_enabled": True,
                    "react_percentile": "0.975",
                    "oe_enabled": True,
                    "oe_loss_weight": "0.7",
                    "oe_target": "uniform",
                },
                "data": {
                    "augmentation_policy": "augmix",
                    "augmix_severity": "4",
                },
                "classifier_rebalance": {
                    "enabled": True,
                    "epochs": "2",
                    "learning_rate": "0.00005",
                    "sampler": "weighted",
                    "objective": "logit_adjusted_cross_entropy",
                },
            }
        }
    }

    normalized = extract_continual_training_config(payload)

    assert normalized["ood"]["react_enabled"] is True
    assert normalized["ood"]["react_percentile"] == pytest.approx(0.975)
    assert normalized["ood"]["oe_enabled"] is True
    assert normalized["ood"]["oe_loss_weight"] == pytest.approx(0.7)
    assert normalized["data"]["augmentation_policy"] == "augmix"
    assert normalized["data"]["augmix_severity"] == 4
    assert normalized["classifier_rebalance"]["enabled"] is True
    assert normalized["classifier_rebalance"]["epochs"] == 2


def test_extract_continual_training_config_normalizes_allow_under_min_training():
    payload = {
        "training": {
            "continual": {
                "data": {
                    "allow_under_min_training": True,
                }
            }
        }
    }

    normalized = extract_continual_training_config(payload)

    assert normalized["data"]["allow_under_min_training"] is True
