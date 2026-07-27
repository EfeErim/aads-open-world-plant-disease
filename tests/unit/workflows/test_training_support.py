import json
from datetime import datetime
from pathlib import Path

import pytest

from src.workflows.training_support import prepare_training_run


class FakeDataset:
    def __init__(self, classes, labels):
        self.classes = list(classes)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)


class FakeLoader:
    def __init__(self, classes, labels, batch_count):
        self.dataset = FakeDataset(classes, labels)
        self._batch_count = int(batch_count)

    def __len__(self):
        return self._batch_count


class FakeAdapter:
    def initialize_engine(self, *, class_names=None, config=None):
        self.initialized = {"class_names": list(class_names or []), "config": dict(config or {})}
        return self.initialized


def _base_config():
    return {
        "training": {
            "continual": {
                "backbone": {"model_name": "fake"},
                "batch_size": 2,
                "seed": 7,
                "data": {"target_size": 224, "cache_size": 10, "loader_error_policy": "tolerant"},
            }
        },
        "colab": {"training": {"num_workers": 0, "pin_memory": False}},
    }


def test_prepare_training_run_rejects_zero_train_batches():
    loaders = {
        "train": FakeLoader(["healthy"], [0], 0),
        "val": FakeLoader(["healthy"], [0], 1),
        "test": FakeLoader(["healthy"], [0], 1),
    }

    with pytest.raises(ValueError, match="zero batches"):
        prepare_training_run(
            config=_base_config(),
            device="cpu",
            crop_name="tomato",
            part_name=None,
            data_dir=Path("runtime"),
            class_names=None,
            num_workers=0,
            pin_memory=False,
            use_cache=False,
            sampler=None,
            error_policy=None,
            run_id="run_1",
            loader_factory=lambda **kwargs: loaders,
            adapter_factory=lambda **kwargs: FakeAdapter(),
        )


def test_prepare_training_run_rejects_eval_only_classes():
    loaders = {
        "train": FakeLoader(["healthy", "disease_a"], [0], 1),
        "val": FakeLoader(["healthy", "disease_a"], [1], 1),
        "test": FakeLoader(["healthy", "disease_a"], [0], 1),
    }

    with pytest.raises(ValueError, match="Validation/test splits contain classes"):
        prepare_training_run(
            config=_base_config(),
            device="cpu",
            crop_name="tomato",
            part_name=None,
            data_dir=Path("runtime"),
            class_names=None,
            num_workers=0,
            pin_memory=False,
            use_cache=False,
            sampler=None,
            error_policy=None,
            run_id="run_1",
            loader_factory=lambda **kwargs: loaders,
            adapter_factory=lambda **kwargs: FakeAdapter(),
        )


def test_prepare_training_run_generates_microsecond_resolution_run_ids(monkeypatch):
    loaders = {
        "train": FakeLoader(["healthy"], [0] * 100, 1),
        "val": FakeLoader(["healthy"], [0] * 100, 1),
        "test": FakeLoader(["healthy"], [0] * 100, 1),
    }

    class _FakeDateTime:
        values = [
            datetime(2026, 3, 11, 12, 0, 0, 1),
            datetime(2026, 3, 11, 12, 0, 0, 2),
        ]

        @classmethod
        def utcnow(cls):
            return cls.values.pop(0)

    monkeypatch.setattr("src.workflows.training_support.datetime", _FakeDateTime)

    first = prepare_training_run(
        config=_base_config(),
        device="cpu",
        crop_name="tomato",
        part_name=None,
        data_dir=Path("runtime"),
        class_names=None,
        num_workers=0,
        pin_memory=False,
        use_cache=False,
        sampler=None,
        error_policy=None,
        run_id="",
        loader_factory=lambda **kwargs: loaders,
        adapter_factory=lambda **kwargs: FakeAdapter(),
    )
    second = prepare_training_run(
        config=_base_config(),
        device="cpu",
        crop_name="tomato",
        part_name=None,
        data_dir=Path("runtime"),
        class_names=None,
        num_workers=0,
        pin_memory=False,
        use_cache=False,
        sampler=None,
        error_policy=None,
        run_id="",
        loader_factory=lambda **kwargs: loaders,
        adapter_factory=lambda **kwargs: FakeAdapter(),
    )

    assert first.run_id == "tomato_20260311_120000_000001"
    assert second.run_id == "tomato_20260311_120000_000002"


def test_prepare_training_run_rejects_supported_classes_below_min_reference_count(tmp_path: Path):
    crop_root = tmp_path / "runtime" / "tomato"
    crop_root.mkdir(parents=True, exist_ok=True)
    (crop_root / "split_manifest.json").write_text(
        json.dumps(
            {
                "classes": [
                    {"class_name": "healthy", "image_count": 240},
                    {"class_name": "disease_a", "image_count": 54},
                ]
            }
        ),
        encoding="utf-8",
    )
    loaders = {
        "train": FakeLoader(["healthy", "disease_a"], [0, 1], 1),
        "val": FakeLoader(["healthy", "disease_a"], [0, 1], 1),
        "test": FakeLoader(["healthy", "disease_a"], [0, 1], 1),
    }

    with pytest.raises(ValueError, match="minimum reference count of 100"):
        prepare_training_run(
            config=_base_config(),
            device="cpu",
            crop_name="tomato",
            part_name=None,
            data_dir=tmp_path / "runtime",
            class_names=None,
            num_workers=0,
            pin_memory=False,
            use_cache=False,
            sampler=None,
            error_policy=None,
            run_id="run_1",
            loader_factory=lambda **kwargs: loaders,
            adapter_factory=lambda **kwargs: FakeAdapter(),
        )


def test_prepare_training_run_allows_under_min_training_when_enabled(tmp_path: Path):
    crop_root = tmp_path / "runtime" / "tomato"
    crop_root.mkdir(parents=True, exist_ok=True)
    (crop_root / "split_manifest.json").write_text(
        json.dumps(
            {
                "classes": [
                    {"class_name": "healthy", "image_count": 12},
                    {"class_name": "disease_a", "image_count": 8},
                ]
            }
        ),
        encoding="utf-8",
    )
    config = _base_config()
    config["training"]["continual"]["data"]["allow_under_min_training"] = True
    adapter = FakeAdapter()
    loaders = {
        "train": FakeLoader(["healthy", "disease_a"], [0, 1], 1),
        "val": FakeLoader(["healthy", "disease_a"], [0, 1], 1),
        "test": FakeLoader(["healthy", "disease_a"], [0, 1], 1),
    }

    setup = prepare_training_run(
        config=config,
        device="cpu",
        crop_name="tomato",
        part_name=None,
        data_dir=tmp_path / "runtime",
        class_names=None,
        num_workers=0,
        pin_memory=False,
        use_cache=False,
        sampler=None,
        error_policy=None,
        run_id="run_1",
        loader_factory=lambda **kwargs: loaders,
        adapter_factory=lambda **kwargs: adapter,
    )

    assert setup.class_balance_runtime["allow_under_min_training"] is True
    assert setup.class_balance_runtime["production_guardrail_bypassed"] is True
    assert setup.class_balance_runtime["production_under_min_classes"] == ["healthy", "disease_a"]
    injected = adapter.initialized["config"]["training"]["continual"]["class_balance"]
    assert injected["allow_under_min_training"] is True


def test_prepare_training_run_injects_active_class_balance_runtime(tmp_path: Path):
    crop_root = tmp_path / "runtime" / "tomato"
    crop_root.mkdir(parents=True, exist_ok=True)
    (crop_root / "split_manifest.json").write_text(
        json.dumps(
            {
                "classes": [
                    {"class_name": "healthy", "image_count": 260},
                    {"class_name": "disease_a", "image_count": 120},
                ]
            }
        ),
        encoding="utf-8",
    )
    adapter = FakeAdapter()
    loaders = {
        "train": FakeLoader(["healthy", "disease_a"], [0, 1], 1),
        "val": FakeLoader(["healthy", "disease_a"], [0, 1], 1),
        "test": FakeLoader(["healthy", "disease_a"], [0, 1], 1),
    }

    setup = prepare_training_run(
        config=_base_config(),
        device="cpu",
        crop_name="tomato",
        part_name=None,
        data_dir=tmp_path / "runtime",
        class_names=None,
        num_workers=0,
        pin_memory=False,
        use_cache=False,
        sampler=None,
        error_policy=None,
        run_id="run_1",
        loader_factory=lambda **kwargs: loaders,
        adapter_factory=lambda **kwargs: adapter,
    )

    assert setup.class_balance_runtime["active"] is True
    assert setup.class_balance_runtime["eligible_classes"] == ["disease_a"]
    injected = adapter.initialized["config"]["training"]["continual"]["class_balance"]
    assert injected["active"] is True
    assert set(injected["weights_by_class"].keys()) == {"healthy", "disease_a"}


def test_prepare_training_run_keeps_class_balance_inactive_when_all_classes_are_large(tmp_path: Path):
    crop_root = tmp_path / "runtime" / "tomato"
    crop_root.mkdir(parents=True, exist_ok=True)
    (crop_root / "split_manifest.json").write_text(
        json.dumps(
            {
                "classes": [
                    {"class_name": "healthy", "image_count": 260},
                    {"class_name": "disease_a", "image_count": 220},
                ]
            }
        ),
        encoding="utf-8",
    )
    loaders = {
        "train": FakeLoader(["healthy", "disease_a"], [0, 1], 1),
        "val": FakeLoader(["healthy", "disease_a"], [0, 1], 1),
        "test": FakeLoader(["healthy", "disease_a"], [0, 1], 1),
    }

    setup = prepare_training_run(
        config=_base_config(),
        device="cpu",
        crop_name="tomato",
        part_name=None,
        data_dir=tmp_path / "runtime",
        class_names=None,
        num_workers=0,
        pin_memory=False,
        use_cache=False,
        sampler=None,
        error_policy=None,
        run_id="run_1",
        loader_factory=lambda **kwargs: loaders,
        adapter_factory=lambda **kwargs: FakeAdapter(),
    )

    assert setup.class_balance_runtime["active"] is False
    assert setup.class_balance_runtime["eligible_classes"] == []
