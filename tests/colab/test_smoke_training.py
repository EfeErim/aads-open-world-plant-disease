"""Smoke tests for continual SD-LoRA training surfaces."""

import torch
import torch.nn as nn

from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer


class TestContinualSmoke:
    def test_config_creation(self):
        cfg = ContinualSDLoRAConfig.from_training_config(
            {
                'backbone': {'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m'},
                'adapter': {'target_modules_strategy': 'all_linear_transformer', 'lora_r': 4, 'lora_alpha': 8},
                'fusion': {'layers': [2, 5, 8, 11]},
                'device': 'cpu',
            }
        )
        assert cfg.target_modules_strategy == 'all_linear_transformer'

    def test_add_classes_without_backbone_download(self):
        cfg = ContinualSDLoRAConfig(
            backbone_model_name='facebook/dinov3-vitl16-pretrain-lvd1689m',
            target_modules_strategy='all_linear_transformer',
            fusion_layers=[2],
            fusion_output_dim=8,
            device='cpu',
        )
        trainer = ContinualSDLoRATrainer(cfg)
        trainer.classifier = nn.Linear(8, 1)
        trainer.class_to_idx = {'healthy': 0}

        trainer.add_classes(['disease_a'])

        assert trainer.classifier.out_features == 2
        assert set(trainer.class_to_idx.keys()) == {'healthy', 'disease_a'}

    def test_predict_payload_shape_with_mocks(self):
        cfg = ContinualSDLoRAConfig(
            backbone_model_name='facebook/dinov3-vitl16-pretrain-lvd1689m',
            target_modules_strategy='all_linear_transformer',
            fusion_layers=[2],
            fusion_output_dim=4,
            device='cpu',
        )
        trainer = ContinualSDLoRATrainer(cfg)
        trainer.classifier = nn.Linear(4, 1)
        trainer.class_to_idx = {'healthy': 0}

        class Dummy(nn.Module):
            def forward(self, *args, **kwargs):
                return args[0]

        trainer.adapter_model = Dummy()
        trainer.fusion = Dummy()
        trainer.encode = lambda images: torch.zeros(images.shape[0], 4)  # type: ignore[assignment]
        trainer._ensure_ood_calibrated = lambda *, operation: None  # type: ignore[assignment]
        trainer.ood_detector.score = lambda features, logits, predicted_labels=None: {
            'mahalanobis_z': torch.tensor([0.1]),
            'energy_z': torch.tensor([0.2]),
            "primary_score": torch.tensor([0.15]),
        "decision_threshold": torch.tensor([0.5]),
        "candidate_scores": {"ensemble": torch.tensor([0.15])},
        "candidate_thresholds": {"ensemble": torch.tensor([0.5])},
            'is_ood': torch.tensor([False]),
            'calibration_version': torch.tensor([1]),
        }

        result = trainer.predict_with_ood(torch.zeros(1, 3, 224, 224))
        assert result['status'] == 'success'
        assert {
            'score_method',
            'primary_score',
            'decision_threshold',
            'is_ood',
            'calibration_version',
        } <= set(result['ood_analysis'].keys())
