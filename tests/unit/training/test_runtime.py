from types import SimpleNamespace

import torch

from src.training.services.persistence import restore_rng_state
from src.training.services.runtime import build_adamw_optimizer, configure_runtime_reproducibility


def test_configure_runtime_reproducibility_enables_fast_cudnn_when_not_deterministic(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda enabled: calls.append(bool(enabled)))

    original_deterministic = torch.backends.cudnn.deterministic
    original_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        configure_runtime_reproducibility(SimpleNamespace(seed=7, deterministic=False))

        assert calls == [False]
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.backends.cudnn.deterministic = original_deterministic
        torch.backends.cudnn.benchmark = original_benchmark


def test_configure_runtime_reproducibility_enforces_deterministic_cudnn(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda enabled: calls.append(bool(enabled)))

    original_deterministic = torch.backends.cudnn.deterministic
    original_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        configure_runtime_reproducibility(SimpleNamespace(seed=7, deterministic=True))

        assert calls == [True]
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.deterministic = original_deterministic
        torch.backends.cudnn.benchmark = original_benchmark


def test_build_adamw_optimizer_prefers_fused_on_cuda_when_supported(monkeypatch):
    calls = []

    class DummyOptimizer:
        pass

    def fake_adamw(params, lr, weight_decay, fused=None, foreach=None):
        calls.append(
            {
                "param_count": len(list(params)),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "fused": fused,
                "foreach": foreach,
            }
        )
        return DummyOptimizer()

    monkeypatch.setattr(torch.optim, "AdamW", fake_adamw)

    optimizer = build_adamw_optimizer(
        [torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))],
        lr=3e-4,
        weight_decay=1e-2,
        device=torch.device("cuda"),
    )

    assert isinstance(optimizer, DummyOptimizer)
    assert calls == [
        {
            "param_count": 1,
            "lr": 3e-4,
            "weight_decay": 1e-2,
            "fused": True,
            "foreach": None,
        }
    ]


def test_build_adamw_optimizer_falls_back_when_fused_is_rejected(monkeypatch):
    calls = []

    class DummyOptimizer:
        pass

    def fake_adamw(params, lr, weight_decay, fused=None, foreach=None):
        calls.append({"fused": fused, "foreach": foreach})
        if fused:
            raise RuntimeError("fused unsupported")
        return DummyOptimizer()

    monkeypatch.setattr(torch.optim, "AdamW", fake_adamw)

    optimizer = build_adamw_optimizer(
        [torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))],
        lr=3e-4,
        weight_decay=1e-2,
        device=torch.device("cuda"),
    )

    assert isinstance(optimizer, DummyOptimizer)
    assert calls == [
        {"fused": True, "foreach": None},
        {"fused": None, "foreach": True},
    ]


def test_configure_runtime_reproducibility_raises_when_deterministic_mode_cannot_be_enabled(monkeypatch):
    def _fail(_enabled):
        raise RuntimeError("unsupported deterministic mode")

    monkeypatch.setattr(torch, "use_deterministic_algorithms", _fail)

    try:
        configure_runtime_reproducibility(SimpleNamespace(seed=7, deterministic=True))
        raise AssertionError("configure_runtime_reproducibility was expected to fail for deterministic runs.")
    except RuntimeError as exc:
        assert "Deterministic training was requested" in str(exc)


def test_restore_rng_state_raises_when_checkpoint_rng_payload_is_invalid(monkeypatch):
    monkeypatch.setattr(torch, "set_rng_state", lambda _state: (_ for _ in ()).throw(RuntimeError("bad state")))

    try:
        restore_rng_state({"torch": object()})
        raise AssertionError("restore_rng_state was expected to fail for invalid torch RNG state.")
    except RuntimeError as exc:
        assert "checkpoint torch RNG state" in str(exc)
