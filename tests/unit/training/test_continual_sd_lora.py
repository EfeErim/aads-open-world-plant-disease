import pytest
import torch
import torch.nn as nn

from src.training.ber_loss import BERLoss
from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer
from src.training.session import ContinualTrainingSession


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.block = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
        ])
        self.router_head = nn.Linear(8, 2)
        self.classifier = nn.Linear(8, 2)
        self.config = type("Cfg", (), {"hidden_size": 8})()

    def forward(self, images, output_hidden_states=False):
        batch = images.shape[0]
        hidden = torch.randn(batch, 4, 8, device=images.device)
        if output_hidden_states:
            return type("Output", (), {"hidden_states": [hidden] * 12})()
        return type("Output", (), {"last_hidden_state": hidden})()


def test_config_from_training_config_accepts_v6_contract():
    cfg = ContinualSDLoRAConfig.from_training_config(
        {
            "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
            "adapter": {"target_modules_strategy": "all_linear_transformer", "lora_r": 4, "lora_alpha": 8},
            "fusion": {"layers": [2, 5, 8, 11]},
            "ood": {
                "ber_enabled": True,
                "ber_lambda_old": 0.05,
                "ber_lambda_new": 0.2,
                "energy_temperature_mode": "auto",
                "energy_temperature": 1.2,
                "react_enabled": True,
                "react_percentile": 0.975,
                "oe_enabled": True,
                "oe_loss_weight": 0.7,
                "knn_backend": "chunked",
                "conformal_method": "aps",
            },
            "classifier_rebalance": {
                "enabled": True,
                "epochs": 2,
                "learning_rate": 5e-5,
            },
            "optimization": {
                "grad_accumulation_steps": 2,
                "loss_name": "cross_entropy",
                "scheduler": {"name": "linear"},
            },
            "evaluation": {"best_metric": "macro_f1"},
            "device": "cpu",
        }
    )
    assert cfg.backbone_model_name == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert cfg.target_modules_strategy == "all_linear_transformer"
    assert cfg.ber_enabled is True
    assert cfg.ber_lambda_old == pytest.approx(0.05)
    assert cfg.ber_lambda_new == pytest.approx(0.2)
    assert cfg.energy_temperature_mode == "auto"
    assert cfg.energy_temperature == pytest.approx(1.2)
    assert cfg.react_enabled is True
    assert cfg.react_percentile == pytest.approx(0.975)
    assert cfg.oe_enabled is True
    assert cfg.oe_loss_weight == pytest.approx(0.7)
    assert cfg.knn_backend == "chunked"
    assert cfg.conformal_method == "aps"
    assert cfg.classifier_rebalance_enabled is True
    assert cfg.classifier_rebalance_epochs == 2
    assert cfg.grad_accumulation_steps == 2
    assert cfg.loss_name == "cross_entropy"
    assert cfg.logitnorm_tau == pytest.approx(1.0)
    assert cfg.scheduler_name == "linear"
    assert cfg.evaluation_best_metric == "macro_f1"
    assert cfg.ood_primary_score_method == "auto"


def test_as_contract_dict_emits_normalized_training_surface():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        device="cpu",
        seed=7,
        grad_accumulation_steps=2,
        scheduler_name="linear",
        evaluation_best_metric="macro_f1",
        ber_enabled=True,
        ber_lambda_old=0.05,
        ber_lambda_new=0.2,
        loss_name="cross_entropy",
        energy_temperature_mode="auto",
        energy_temperature=1.1,
        react_enabled=True,
        react_percentile=0.98,
        oe_enabled=True,
        oe_loss_weight=0.6,
        knn_backend="chunked",
        conformal_method="raps",
        conformal_raps_lambda=0.02,
        conformal_raps_k_reg=2,
        classifier_rebalance_enabled=True,
        classifier_rebalance_epochs=2,
    )

    payload = cfg.as_contract_dict()

    assert payload["seed"] == 7
    assert payload["ood"]["ber_enabled"] is True
    assert payload["ood"]["ber_lambda_old"] == pytest.approx(0.05)
    assert payload["ood"]["ber_lambda_new"] == pytest.approx(0.2)
    assert payload["ood"]["energy_temperature_mode"] == "auto"
    assert payload["ood"]["energy_temperature"] == pytest.approx(1.1)
    assert payload["ood"]["react_enabled"] is True
    assert payload["ood"]["react_percentile"] == pytest.approx(0.98)
    assert payload["ood"]["oe_enabled"] is True
    assert payload["ood"]["oe_loss_weight"] == pytest.approx(0.6)
    assert payload["ood"]["knn_backend"] == "chunked"
    assert payload["ood"]["conformal_method"] == "raps"
    assert payload["ood"]["conformal_raps_lambda"] == pytest.approx(0.02)
    assert payload["ood"]["conformal_raps_k_reg"] == 2
    assert payload["ood"]["primary_score_method"] == "auto"
    assert payload["optimization"]["grad_accumulation_steps"] == 2
    assert payload["optimization"]["loss_name"] == "cross_entropy"
    assert payload["optimization"]["logitnorm_tau"] == pytest.approx(1.0)
    assert payload["optimization"]["scheduler"]["name"] == "linear"
    assert payload["evaluation"]["best_metric"] == "macro_f1"
    assert payload["classifier_rebalance"]["enabled"] is True
    assert payload["classifier_rebalance"]["epochs"] == 2


def test_set_ood_primary_score_method_updates_runtime_and_contract():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)

    resolved = trainer.set_ood_primary_score_method("knn")

    assert resolved == "knn"
    assert trainer.config.ood_primary_score_method == "knn"
    assert trainer.ood_detector.primary_score_method == "knn"
    assert trainer._contract["ood"]["primary_score_method"] == "knn"


def test_trainer_seed_configuration_is_repeatable():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        device="cpu",
        seed=123,
    )

    _ = ContinualSDLoRATrainer(cfg)
    first = torch.rand(2)
    _ = ContinualSDLoRATrainer(cfg)
    second = torch.rand(2)

    assert torch.equal(first, second)


def test_trainer_raises_when_cuda_is_requested_without_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        device="cuda",
    )

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        ContinualSDLoRATrainer(cfg)


def test_target_resolver_excludes_classifier_and_router_heads():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2, 5, 8, 11],
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    names = trainer.resolve_target_modules(DummyBackbone())
    assert names
    assert all("classifier" not in n.lower() for n in names)
    assert all("router" not in n.lower() for n in names)


def test_add_classes_expands_classifier_shape():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
        react_enabled=True,
        react_percentile=0.98,
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.classifier = nn.Linear(8, 1)
    trainer.class_to_idx = {"healthy": 0}

    updated = trainer.add_classes(["disease_a", "disease_b"])

    assert set(updated.keys()) == {"healthy", "disease_a", "disease_b"}
    assert trainer.classifier.out_features == 3


def test_add_classes_updates_ber_partition_and_keeps_ber_metrics_available():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
        ber_enabled=True,
        loss_name="cross_entropy",
        ber_lambda_old=0.1,
        ber_lambda_new=0.1,
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.classifier = nn.Linear(8, 1)
    trainer.class_to_idx = {"healthy": 0}
    trainer.adapter_model = DummyBackbone()
    trainer.fusion = nn.Identity()
    trainer.ber_loss = BERLoss(lambda_old=0.1, lambda_new=0.1, num_old_classes=0)

    updated = trainer.add_classes(["disease_a"])

    assert updated == {"healthy": 0, "disease_a": 1}
    assert trainer.classifier.out_features == 2
    assert trainer.ber_loss is not None
    assert trainer.ber_loss.num_old_classes == 1

    logits = nn.Parameter(torch.tensor([[1.2, -0.4], [-0.3, 1.1]], dtype=torch.float32))
    trainer.optimizer = torch.optim.SGD([logits], lr=0.1)
    trainer.forward_logits = lambda images: logits  # type: ignore[assignment]

    stats = trainer.train_batch(
        {"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)}
    )

    assert stats.ber_ce_loss is not None
    assert stats.ber_old_loss is not None
    assert stats.ber_new_loss is not None


def test_add_classes_refreshes_existing_optimizer_params():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
        scheduler_name="linear",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer._is_initialized = True
    trainer.adapter_model = nn.Linear(8, 8)
    trainer.fusion = nn.Linear(8, 8)
    trainer.classifier = nn.Linear(8, 1)
    trainer.class_to_idx = {"healthy": 0}
    trainer.configure_training_plan(total_batches=2, num_epochs=1)
    trainer.setup_optimizer()
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None

    old_classifier_weight_id = id(trainer.classifier.weight)

    trainer.add_classes(["disease_a"])

    assert trainer.optimizer is not None
    assert trainer.scheduler is not None
    optimizer_param_ids = {
        id(param)
        for group in trainer.optimizer.param_groups
        for param in group["params"]
    }
    assert id(trainer.classifier.weight) in optimizer_param_ids
    assert old_classifier_weight_id not in optimizer_param_ids


def test_predict_payload_contains_v6_ood_keys():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.class_to_idx = {"healthy": 0}

    class DummyModule(nn.Module):
        def forward(self, *args, **kwargs):
            return args[0]

    trainer.adapter_model = DummyModule()
    trainer.fusion = DummyModule()
    trainer.classifier = nn.Linear(4, 1)
    trainer.encode = lambda images: torch.zeros(images.shape[0], 4)  # type: ignore[assignment]
    trainer.ood_detector.calibration_version = 1
    trainer.ood_detector.class_stats = {0: object()}
    trainer.ood_detector.score = lambda features, logits, predicted_labels=None: {
        "mahalanobis_z": torch.tensor([0.1]),
        "energy_z": torch.tensor([0.2]),
        "primary_score": torch.tensor([0.15]),
        "decision_threshold": torch.tensor([0.8]),
        "candidate_scores": {"ensemble": torch.tensor([0.15])},
        "candidate_thresholds": {"ensemble": torch.tensor([0.8])},
        "is_ood": torch.tensor([False]),
        "calibration_version": torch.tensor([3]),
    }

    result = trainer.predict_with_ood(torch.zeros(1, 3, 224, 224))

    assert "ood_analysis" in result
    assert {"score_method", "primary_score", "decision_threshold", "is_ood", "calibration_version"} <= set(
        result["ood_analysis"].keys()
    )


def test_predict_payload_refreshes_cached_class_index_after_class_update():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.class_to_idx = {"healthy": 0}

    class DummyModule(nn.Module):
        def forward(self, *args, **kwargs):
            return args[0]

    trainer.adapter_model = DummyModule()
    trainer.fusion = DummyModule()
    trainer.classifier = nn.Linear(4, 1)
    trainer.encode = lambda images: torch.zeros(images.shape[0], 4)  # type: ignore[assignment]
    trainer.ood_detector.calibration_version = 1
    trainer.ood_detector.class_stats = {0: object()}
    trainer.ood_detector.score = lambda features, logits, predicted_labels=None: {
        "mahalanobis_z": torch.tensor([0.1]),
        "energy_z": torch.tensor([0.2]),
        "primary_score": torch.tensor([0.15]),
        "decision_threshold": torch.tensor([0.8]),
        "candidate_scores": {"ensemble": torch.tensor([0.15])},
        "candidate_thresholds": {"ensemble": torch.tensor([0.8])},
        "is_ood": torch.tensor([False]),
        "calibration_version": torch.tensor([3]),
    }

    first = trainer.predict_with_ood(torch.zeros(1, 3, 8, 8))
    trainer.class_to_idx = {"disease_a": 0}
    second = trainer.predict_with_ood(torch.zeros(1, 3, 8, 8))

    assert first["disease"]["name"] == "healthy"
    assert second["disease"]["name"] == "disease_a"


def test_raises_when_peft_is_missing(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
        react_enabled=True,
        react_percentile=0.98,
    )
    trainer = ContinualSDLoRATrainer(cfg)

    monkeypatch.setattr(continual_module, "LoraConfig", None)

    with pytest.raises(RuntimeError, match="peft is required for SD-LoRA adapter wrapping"):
        trainer._apply_lora(DummyBackbone(), ["transformer.block.0.0"])


def test_apply_lora_avoids_low_cpu_mem_usage_meta_tensors(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    call_kwargs = {}

    def fake_get_peft_model(model, _cfg, **kwargs):
        call_kwargs.update(kwargs)
        return model

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
        react_enabled=True,
        react_percentile=0.98,
    )
    trainer = ContinualSDLoRATrainer(cfg)

    monkeypatch.setattr(continual_module, "LoraConfig", FakeLoraConfig)
    monkeypatch.setattr(continual_module, "get_peft_model", fake_get_peft_model)

    wrapped = trainer._apply_lora(DummyBackbone(), ["transformer.block.0.0"])

    assert isinstance(wrapped, DummyBackbone)
    assert call_kwargs.get("low_cpu_mem_usage") is False


def test_apply_lora_without_low_cpu_mem_usage_kwarg_support(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    called = {"count": 0}

    def fake_get_peft_model(model, _cfg):
        called["count"] += 1
        return model

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
        react_enabled=True,
        react_percentile=0.98,
    )
    trainer = ContinualSDLoRATrainer(cfg)

    monkeypatch.setattr(continual_module, "LoraConfig", FakeLoraConfig)
    monkeypatch.setattr(continual_module, "get_peft_model", fake_get_peft_model)

    wrapped = trainer._apply_lora(DummyBackbone(), ["transformer.block.0.0"])

    assert isinstance(wrapped, DummyBackbone)
    assert called["count"] == 1


def test_train_batch_emits_step_metrics():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
        grad_accumulation_steps=1,
    )
    trainer = ContinualSDLoRATrainer(cfg)

    class DummyModule(nn.Module):
        def forward(self, x, *args, **kwargs):
            return x

    trainer.adapter_model = DummyModule()
    trainer.classifier = DummyModule()
    trainer.fusion = DummyModule()

    trainable = nn.Parameter(torch.tensor([1.0], requires_grad=True))
    trainer.optimizer = torch.optim.SGD([trainable], lr=0.1)
    trainer.training_step = lambda _batch: (trainable ** 2).sum()  # type: ignore[assignment]

    stats = trainer.train_batch(
        {"images": torch.zeros(2, 3, 8, 8), "labels": torch.zeros(2, dtype=torch.long)}
    )

    assert stats.loss > 0.0
    assert stats.ber_ce_loss is None
    assert stats.ber_old_loss is None
    assert stats.ber_new_loss is None
    assert stats.lr == pytest.approx(0.1)
    assert stats.grad_norm > 0.0
    assert stats.batch_size == 2
    assert stats.step_time_sec >= 0.0


def test_training_step_uses_class_balanced_ce_weights_when_active(monkeypatch):
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
        loss_name="cross_entropy",
        label_smoothing=0.1,
        extra={"class_balance": {"weights_by_class": {"healthy": 0.8, "disease_a": 1.2}}},
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.optimizer = object()
    trainer.class_to_idx = {"healthy": 0, "disease_a": 1}
    trainer._refresh_class_index_cache()
    expected_loss = torch.tensor(1.23, requires_grad=True)
    recorded = {}

    def fake_cross_entropy(logits, labels, weight=None, label_smoothing=0.0):
        recorded["labels"] = labels.detach().clone()
        recorded["weight"] = None if weight is None else weight.detach().cpu().clone()
        recorded["label_smoothing"] = label_smoothing
        return expected_loss

    monkeypatch.setattr("src.training.continual_sd_lora.nn.functional.cross_entropy", fake_cross_entropy)
    trainer.forward_logits = lambda images: torch.full((images.shape[0], 2), 0.5, requires_grad=True)  # type: ignore[assignment]

    loss = trainer.training_step(
        {"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)}
    )

    assert loss is expected_loss
    assert recorded["labels"].tolist() == [0, 1]
    assert recorded["label_smoothing"] == pytest.approx(0.1)
    assert torch.allclose(recorded["weight"], torch.tensor([0.8, 1.2]))


def test_training_step_uses_ber_loss_when_enabled():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
        ber_enabled=True,
        loss_name="cross_entropy",
        label_smoothing=0.1,
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.optimizer = object()
    trainer.class_to_idx = {"healthy": 0, "disease_a": 1}
    trainer.class_balance_runtime = {"weights_by_class": {"healthy": 0.8, "disease_a": 1.2}}
    trainer._refresh_class_index_cache()
    recorded = {}
    expected_loss = torch.tensor(1.23, requires_grad=True)

    class FakeBER:
        def __call__(self, logits, labels, label_smoothing=0.0, class_weight=None):
            recorded["logits"] = logits.detach().clone()
            recorded["labels"] = labels.detach().clone()
            recorded["label_smoothing"] = label_smoothing
            recorded["class_weight"] = None if class_weight is None else class_weight.detach().cpu().clone()
            return expected_loss, {"ce": 0.7, "ber_old": 0.1, "ber_new": 0.2}

    trainer.ber_loss = FakeBER()
    trainer.forward_logits = lambda images: torch.full((images.shape[0], 2), 0.5, requires_grad=True)  # type: ignore[assignment]

    loss = trainer.training_step(
        {"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)}
    )

    assert loss is expected_loss
    assert recorded["labels"].tolist() == [0, 1]
    assert recorded["label_smoothing"] == pytest.approx(0.1)
    assert torch.allclose(recorded["class_weight"], torch.tensor([0.8, 1.2]))
    assert trainer._last_ber_components == {"ce": 0.7, "ber_old": 0.1, "ber_new": 0.2}


def test_training_step_uses_logitnorm_loss_when_configured(monkeypatch):
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
        loss_name="logitnorm",
        logitnorm_tau=0.7,
        label_smoothing=0.05,
        extra={"class_balance": {"weights_by_class": {"healthy": 0.8, "disease_a": 1.2}}},
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.optimizer = object()
    trainer.class_to_idx = {"healthy": 0, "disease_a": 1}
    trainer._refresh_class_index_cache()
    expected_loss = torch.tensor(1.23, requires_grad=True)
    recorded = {}

    def fake_logitnorm_loss(logits, labels, *, tau, weight=None, label_smoothing=0.0):
        recorded["labels"] = labels.detach().clone()
        recorded["tau"] = tau
        recorded["weight"] = None if weight is None else weight.detach().cpu().clone()
        recorded["label_smoothing"] = label_smoothing
        return expected_loss

    monkeypatch.setattr("src.training.continual_sd_lora.compute_logitnorm_loss", fake_logitnorm_loss)
    trainer.forward_logits = lambda images: torch.full((images.shape[0], 2), 0.5, requires_grad=True)  # type: ignore[assignment]

    loss = trainer.training_step(
        {"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)}
    )

    assert loss is expected_loss
    assert recorded["labels"].tolist() == [0, 1]
    assert recorded["tau"] == pytest.approx(0.7)
    assert recorded["label_smoothing"] == pytest.approx(0.05)
    assert torch.allclose(recorded["weight"], torch.tensor([0.8, 1.2]))


def test_training_step_adds_oe_uniform_loss_when_enabled(monkeypatch):
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
        loss_name="cross_entropy",
        oe_enabled=True,
        oe_loss_weight=0.5,
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.optimizer = object()
    trainer.classifier = nn.Identity()
    trainer.encode = lambda images: images.float()  # type: ignore[assignment]
    trainer.set_oe_loader(
        [{"images": torch.tensor([[1.0, 0.0], [0.0, 1.0]]), "labels": torch.tensor([-1, -1], dtype=torch.long)}]
    )

    def fake_cross_entropy(logits, labels, weight=None, label_smoothing=0.0):
        del logits, labels, weight, label_smoothing
        return torch.tensor(2.0, requires_grad=True)

    monkeypatch.setattr("src.training.continual_sd_lora.nn.functional.cross_entropy", fake_cross_entropy)

    loss = trainer.training_step(
        {"images": torch.tensor([[2.0, 0.0], [0.0, 2.0]]), "labels": torch.tensor([0, 1], dtype=torch.long)}
    )

    expected_oe = -torch.log_softmax(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1).mean(dim=1).mean()
    assert loss.item() == pytest.approx(2.0 + (0.5 * float(expected_oe.item())))


def test_configure_classifier_rebalance_stage_freezes_non_classifier_params():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
        classifier_rebalance_enabled=True,
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.adapter_model = nn.Linear(4, 4)
    trainer.fusion = nn.Linear(4, 4)
    trainer.classifier = nn.Linear(4, 2)

    trainer.configure_classifier_rebalance_stage(
        log_priors=torch.log(torch.tensor([0.6, 0.4], dtype=torch.float32))
    )

    assert all(not parameter.requires_grad for parameter in trainer.adapter_model.parameters())
    assert all(not parameter.requires_grad for parameter in trainer.fusion.parameters())
    assert all(parameter.requires_grad for parameter in trainer.classifier.parameters())
    assert trainer._active_training_stage == "classifier_rebalance"


def test_config_rejects_logitnorm_when_ber_is_enabled():
    with pytest.raises(ValueError, match="ber_enabled.*loss_name='logitnorm'"):
        ContinualSDLoRAConfig(
            backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
            target_modules_strategy="all_linear_transformer",
            fusion_layers=[2],
            fusion_output_dim=4,
            device="cpu",
            ber_enabled=True,
            loss_name="logitnorm",
        ).validate()


def test_train_batch_emits_ber_metrics_when_enabled():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
        ber_enabled=True,
        loss_name="cross_entropy",
        ber_lambda_old=0.1,
        ber_lambda_new=0.1,
    )
    trainer = ContinualSDLoRATrainer(cfg)

    class DummyModule(nn.Module):
        def forward(self, x, *args, **kwargs):
            return x

    trainer.adapter_model = DummyModule()
    trainer.classifier = DummyModule()
    trainer.fusion = DummyModule()

    logits = nn.Parameter(torch.tensor([[1.2, -0.4], [-0.3, 1.1]], dtype=torch.float32))
    trainer.optimizer = torch.optim.SGD([logits], lr=0.1)
    trainer.ber_loss = BERLoss(
        lambda_old=trainer.config.ber_lambda_old,
        lambda_new=trainer.config.ber_lambda_new,
        num_old_classes=1,
    )
    trainer.forward_logits = lambda images: logits  # type: ignore[assignment]

    stats = trainer.train_batch(
        {"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)}
    )

    assert stats.loss > 0.0
    assert stats.ber_ce_loss is not None
    assert stats.ber_old_loss is not None
    assert stats.ber_new_loss is not None


def test_initialize_engine_with_non_quantized_backbone(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name):
            return DummyBackbone()

    monkeypatch.setattr(continual_module, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(ContinualSDLoRATrainer, "_apply_lora", lambda self, model, _targets: model)

    cfg = ContinualSDLoRAConfig.from_training_config(
        {
            "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
            "adapter": {"target_modules_strategy": "all_linear_transformer", "lora_r": 4, "lora_alpha": 8},
            "fusion": {"layers": [2, 5, 8, 11], "output_dim": 8},
            "device": "cpu",
        }
    )

    trainer = ContinualSDLoRATrainer(cfg)
    trainer.initialize_engine(class_to_idx={"healthy": 0})

    assert trainer.backbone is not None
    assert trainer.classifier is not None
    assert trainer.classifier.out_features == 1


def test_initialize_engine_raises_when_backbone_contains_meta_tensors(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    class MetaBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(8, 8, device="meta")
            self.config = type("Cfg", (), {"hidden_size": 8})()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name):
            return MetaBackbone()

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)

    monkeypatch.setattr(continual_module, "AutoModel", FakeAutoModel)

    with pytest.raises(RuntimeError, match="backbone contains meta tensors before device move"):
        trainer.initialize_engine(class_to_idx={"healthy": 0})


def test_initialize_engine_raises_when_adapter_contains_meta_tensors(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name):
            return DummyBackbone()

    class MetaAdapterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = nn.Linear(8, 8, device="meta")

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)

    monkeypatch.setattr(continual_module, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(ContinualSDLoRATrainer, "_apply_lora", lambda self, _model, _targets: MetaAdapterModel())

    with pytest.raises(RuntimeError, match="adapter_model contains meta tensors before device move"):
        trainer.initialize_engine(class_to_idx={"healthy": 0})


def test_initialize_engine_allows_dispatch_managed_meta_adapter(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name):
            return DummyBackbone()

    class DispatchManagedMetaAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = nn.Linear(8, 8, device="meta")
            self.hf_device_map = {"": "cpu"}

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)

    monkeypatch.setattr(continual_module, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(
        ContinualSDLoRATrainer,
        "_apply_lora",
        lambda self, _model, _targets: DispatchManagedMetaAdapter(),
    )

    trainer.initialize_engine(class_to_idx={"healthy": 0})

    assert trainer.adapter_model is not None
    assert trainer.classifier is not None
    assert trainer.classifier.out_features == 1


def test_prepare_module_for_device_skips_dispatch_managed_not_implemented():
    class NotImplementedOnTo(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)
            self.hf_device_map = {"": "cpu"}

        def to(self, *args, **kwargs):
            raise NotImplementedError("Cannot copy out of meta tensor; no data!")

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    module = NotImplementedOnTo()

    assert trainer._prepare_module_for_device(module, module_name="adapter_model") is module


def test_prepare_module_for_device_re_raises_not_implemented_without_dispatch():
    class NotImplementedOnTo(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)

        def to(self, *args, **kwargs):
            raise NotImplementedError("Cannot copy out of meta tensor; no data!")

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)

    with pytest.raises(NotImplementedError, match="Cannot copy out of meta tensor; no data!"):
        trainer._prepare_module_for_device(NotImplementedOnTo(), module_name="adapter_model")


def test_trainer_end_to_end_surface_with_dummy_backbone(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name):
            return DummyBackbone()

    monkeypatch.setattr(continual_module, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(ContinualSDLoRATrainer, "_apply_lora", lambda self, model, _targets: model)

    cfg = ContinualSDLoRAConfig.from_training_config(
        {
            "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
            "adapter": {"target_modules_strategy": "all_linear_transformer", "lora_r": 4, "lora_alpha": 8},
            "fusion": {"layers": [2, 5, 8, 11], "output_dim": 8},
            "ood": {"threshold_factor": 2.0},
            "device": "cpu",
            "num_epochs": 1,
        }
    )

    trainer = ContinualSDLoRATrainer(cfg)
    trainer.initialize_engine(class_to_idx={"healthy": 0})
    trainer.add_classes(["disease_a"])

    for batch in [
        {"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)},
        {"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([1, 0], dtype=torch.long)},
    ]:
        stats = trainer.train_batch(batch)
        assert stats.loss >= 0.0

    cal = trainer.calibrate_ood(
        [{"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)}]
    )
    assert int(cal["num_classes"]) >= 1

    pred = trainer.predict_with_ood(torch.zeros(1, 3, 8, 8))
    assert pred["status"] == "success"
    assert {"score_method", "primary_score", "decision_threshold", "is_ood", "calibration_version"} <= set(
        pred["ood_analysis"].keys()
    )


def test_save_and_load_adapter_roundtrip_restores_raw_adapter_weights(monkeypatch, tmp_path):
    from src.training import continual_sd_lora as continual_module

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name):
            return DummyBackbone()

    monkeypatch.setattr(continual_module, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(ContinualSDLoRATrainer, "_apply_lora", lambda self, model, _targets: model)

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.initialize_engine(class_to_idx={"healthy": 0})
    assert trainer.adapter_model is not None
    assert trainer.classifier is not None
    assert trainer.fusion is not None

    first_weight_name, first_weight = next(iter(trainer.adapter_model.state_dict().items()))
    trainer.classifier.weight.data.fill_(0.25)
    trainer.fusion.projections[0].weight.data.fill_(0.5)
    calibration_loader = [
        {"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([0, 0], dtype=torch.long)},
        {"images": torch.ones(2, 3, 8, 8), "labels": torch.tensor([0, 0], dtype=torch.long)},
    ]
    trainer.calibrate_ood(calibration_loader)

    save_dir = tmp_path / "adapter"
    trainer.save_adapter(str(save_dir))

    reloaded = ContinualSDLoRATrainer(cfg)
    reloaded.load_adapter(str(save_dir / "continual_sd_lora_adapter"))

    assert reloaded.adapter_model is not None
    assert reloaded.classifier is not None
    assert reloaded.fusion is not None
    assert torch.equal(reloaded.adapter_model.state_dict()[first_weight_name], first_weight)
    assert torch.allclose(reloaded.classifier.weight, torch.full_like(reloaded.classifier.weight, 0.25))
    assert torch.allclose(
        reloaded.fusion.projections[0].weight,
        torch.full_like(reloaded.fusion.projections[0].weight, 0.5),
    )


def test_save_and_load_adapter_roundtrip_restores_ood_state(monkeypatch, tmp_path):
    from src.training import continual_sd_lora as continual_module

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name):
            return DummyBackbone()

    monkeypatch.setattr(continual_module, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(ContinualSDLoRATrainer, "_apply_lora", lambda self, model, _targets: model)

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
        react_enabled=True,
        react_percentile=0.98,
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.initialize_engine(class_to_idx={"healthy": 0, "disease_a": 1})

    calibration_loader = [
        {"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)},
        {"images": torch.ones(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)},
    ]
    calibration = trainer.calibrate_ood(calibration_loader)
    assert int(calibration["num_classes"]) == 2

    save_dir = tmp_path / "adapter_with_ood"
    trainer.save_adapter(str(save_dir))

    reloaded = ContinualSDLoRATrainer(cfg)
    reloaded.load_adapter(str(save_dir / "continual_sd_lora_adapter"))

    assert int(reloaded.ood_detector.calibration_version) == int(trainer.ood_detector.calibration_version)
    assert set(reloaded.ood_detector.class_stats.keys()) == {0, 1}
    assert reloaded.ood_detector.react_enabled is True
    assert reloaded.ood_detector.react_threshold is not None


def test_save_adapter_auto_calibrates_from_session_loader(monkeypatch, tmp_path):
    from src.training import continual_sd_lora as continual_module

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name):
            return DummyBackbone()

    monkeypatch.setattr(continual_module, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(ContinualSDLoRATrainer, "_apply_lora", lambda self, model, _targets: model)

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.initialize_engine(class_to_idx={"healthy": 0, "disease_a": 1})

    calibration_loader = [
        {"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)},
        {"images": torch.ones(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)},
    ]
    _ = ContinualTrainingSession(trainer, calibration_loader, 1)

    save_dir = tmp_path / "auto_calibrated_adapter"
    trainer.save_adapter(str(save_dir))

    assert trainer.ood_detector.calibration_version > 0
    assert trainer.ood_detector.class_stats
    assert (save_dir / "continual_sd_lora_adapter" / "adapter_meta.json").exists()


def test_predict_with_ood_raises_when_ood_not_calibrated():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.class_to_idx = {"healthy": 0}

    class DummyModule(nn.Module):
        def forward(self, *args, **kwargs):
            return args[0]

    trainer.adapter_model = DummyModule()
    trainer.fusion = DummyModule()
    trainer.classifier = nn.Linear(4, 1)
    trainer.encode = lambda images: torch.zeros(images.shape[0], 4)  # type: ignore[assignment]

    with pytest.raises(
        RuntimeError,
        match="No calibration loader is available for automatic OOD calibration before predict_with_ood\\(\\)",
    ):
        trainer.predict_with_ood(torch.zeros(1, 3, 8, 8))


def test_save_adapter_rejects_uncalibrated_ood_state(monkeypatch, tmp_path):
    from src.training import continual_sd_lora as continual_module

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name):
            return DummyBackbone()

    monkeypatch.setattr(continual_module, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(ContinualSDLoRATrainer, "_apply_lora", lambda self, model, _targets: model)

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.initialize_engine(class_to_idx={"healthy": 0})

    with pytest.raises(
        RuntimeError,
        match="No calibration loader is available for automatic OOD calibration before save_adapter\\(\\)",
    ):
        trainer.save_adapter(str(tmp_path / "adapter"))
