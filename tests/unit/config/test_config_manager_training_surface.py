import pytest

from src.core.config_manager import CURRENT_CONFIG_SCHEMA_VERSION, ConfigurationManager
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
