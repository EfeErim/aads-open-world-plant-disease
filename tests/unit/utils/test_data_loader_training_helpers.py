from pathlib import Path

import pytest
import torch

import src.data.datasets as datasets
from src.data.loaders import create_training_loaders, dict_collate_fn
from tests.utils.test_helpers import make_image


class DummyCropDataset:
    def __init__(self, *args, **kwargs):
        self.samples = [
            (torch.zeros(3, 8, 8), 0),
            (torch.ones(3, 8, 8), 1),
            (torch.full((3, 8, 8), 2.0), 0),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



def test_dict_collate_fn_transforms_tuple_batch():
    batch = [
        (torch.zeros(3, 8, 8), 0),
        (torch.ones(3, 8, 8), 1),
    ]

    collated = dict_collate_fn(batch)

    assert set(collated.keys()) == {'images', 'labels'}
    assert collated['images'].shape == (2, 3, 8, 8)
    assert torch.equal(collated['labels'], torch.tensor([0, 1], dtype=torch.long))



def test_create_training_loaders_uses_dict_collation(monkeypatch):
    monkeypatch.setattr("src.data.loaders.CropDataset", DummyCropDataset)

    loaders = create_training_loaders(
        data_dir='unused',
        crop='tomato',
        batch_size=2,
        num_workers=0,
        use_cache=False,
        dataset_cls=DummyCropDataset,
        infer_classes_fn=lambda *_args: ["healthy", "disease_a"],
    )

    assert {'train', 'val', 'test'} <= set(loaders.keys())

    batch = next(iter(loaders['train']))
    assert set(batch.keys()) == {'images', 'labels'}
    assert batch['images'].ndim == 4
    assert batch['labels'].dtype == torch.long


def _write_image(path: Path, *, color: tuple[int, int, int] = (255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    make_image(path, size=(8, 8), color=color)



def test_create_training_loaders_supports_weighted_sampler(tmp_path: Path):
    for idx in range(3):
        _write_image(tmp_path / "tomato" / "continual" / "healthy" / f"healthy_{idx}.jpg")
    _write_image(tmp_path / "tomato" / "continual" / "disease_a" / "disease.jpg", color=(0, 255, 0))
    _write_image(tmp_path / "tomato" / "val" / "healthy" / "val.jpg")
    _write_image(tmp_path / "tomato" / "test" / "healthy" / "test.jpg")

    loaders = create_training_loaders(
        data_dir=str(tmp_path),
        crop="tomato",
        batch_size=2,
        num_workers=0,
        sampler="weighted",
        seed=7,
    )

    assert loaders["train"].sampler is not None
    assert loaders["train"].sampler.__class__.__name__ == "WeightedRandomSampler"
    assert loaders["train"]._resolved_sampler == "weighted"



def test_create_training_loaders_auto_sampler_promotes_imbalanced_train_split(tmp_path: Path):
    for idx in range(6):
        _write_image(tmp_path / "tomato" / "continual" / "healthy" / f"healthy_{idx}.jpg")
    for idx in range(2):
        _write_image(tmp_path / "tomato" / "continual" / "disease_a" / f"disease_{idx}.jpg", color=(0, 255, 0))
    _write_image(tmp_path / "tomato" / "val" / "healthy" / "val.jpg")
    _write_image(tmp_path / "tomato" / "test" / "healthy" / "test.jpg")

    loaders = create_training_loaders(
        data_dir=str(tmp_path),
        crop="tomato",
        batch_size=2,
        num_workers=0,
        sampler="auto",
        seed=7,
    )

    assert loaders["train"].sampler is not None
    assert loaders["train"].sampler.__class__.__name__ == "WeightedRandomSampler"
    assert loaders["train"]._requested_sampler == "auto"
    assert loaders["train"]._resolved_sampler == "weighted"
    assert loaders["train"]._sampler_decision_reason == "imbalance_detected"
    assert loaders["train"]._sampler_class_counts == {"healthy": 6, "disease_a": 2}



def test_create_training_loaders_auto_sampler_keeps_balanced_train_split_on_shuffle(tmp_path: Path):
    for idx in range(4):
        _write_image(tmp_path / "tomato" / "continual" / "healthy" / f"healthy_{idx}.jpg")
        _write_image(tmp_path / "tomato" / "continual" / "disease_a" / f"disease_{idx}.jpg", color=(0, 255, 0))
    _write_image(tmp_path / "tomato" / "val" / "healthy" / "val.jpg")
    _write_image(tmp_path / "tomato" / "test" / "healthy" / "test.jpg")

    loaders = create_training_loaders(
        data_dir=str(tmp_path),
        crop="tomato",
        batch_size=2,
        num_workers=0,
        sampler="auto",
        seed=7,
    )

    assert loaders["train"].sampler.__class__.__name__ != "WeightedRandomSampler"
    assert loaders["train"]._resolved_sampler == "shuffle"
    assert loaders["train"]._sampler_decision_reason == "balanced_enough"



def test_create_training_loaders_adds_optional_ood_loader(tmp_path: Path):
    _write_image(tmp_path / "tomato" / "continual" / "healthy" / "train.jpg")
    _write_image(tmp_path / "tomato" / "val" / "healthy" / "val.jpg")
    _write_image(tmp_path / "tomato" / "test" / "healthy" / "test.jpg")
    _write_image(tmp_path / "tomato" / "ood" / "unknown" / "ood.jpg", color=(0, 0, 255))

    loaders = create_training_loaders(
        data_dir=str(tmp_path),
        crop="tomato",
        batch_size=2,
        num_workers=0,
        seed=7,
    )

    assert "ood" in loaders
    batch = next(iter(loaders["ood"]))
    assert batch["labels"].tolist() == [-1]


def test_create_training_loaders_accepts_explicit_ood_root(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    external_ood_root = tmp_path / "external_ood"
    _write_image(runtime_root / "tomato" / "continual" / "healthy" / "train.jpg")
    _write_image(runtime_root / "tomato" / "val" / "healthy" / "val.jpg")
    _write_image(runtime_root / "tomato" / "test" / "healthy" / "test.jpg")
    _write_image(external_ood_root / "unknown" / "ood.jpg", color=(0, 0, 255))

    loaders = create_training_loaders(
        data_dir=str(runtime_root),
        crop="tomato",
        batch_size=2,
        num_workers=0,
        seed=7,
        ood_root=external_ood_root,
    )

    assert "ood" in loaders
    assert len(loaders["ood"].dataset) == 1
    assert loaders["ood"].dataset.image_paths[0] == external_ood_root / "unknown" / "ood.jpg"


def test_create_training_loaders_keeps_oe_separate_from_real_ood(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    external_oe_root = tmp_path / "external_oe"
    _write_image(runtime_root / "tomato" / "continual" / "healthy" / "train.jpg")
    _write_image(runtime_root / "tomato" / "val" / "healthy" / "val.jpg")
    _write_image(runtime_root / "tomato" / "test" / "healthy" / "test.jpg")
    _write_image(runtime_root / "tomato" / "ood" / "unknown" / "ood.jpg", color=(0, 0, 255))
    _write_image(external_oe_root / "unknown_pool" / "oe.jpg", color=(0, 255, 255))

    loaders = create_training_loaders(
        data_dir=str(runtime_root),
        crop="tomato",
        batch_size=2,
        num_workers=0,
        seed=7,
        oe_root=external_oe_root,
    )

    assert "ood" in loaders
    assert "oe" in loaders
    assert len(loaders["ood"].dataset) == 1
    assert len(loaders["oe"].dataset) == 1
    assert loaders["ood"].dataset.image_paths[0] == runtime_root / "tomato" / "ood" / "unknown" / "ood.jpg"
    assert loaders["oe"].dataset.image_paths[0] == external_oe_root / "unknown_pool" / "oe.jpg"


def test_create_training_loaders_auto_splits_real_ood_by_slice(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    _write_image(runtime_root / "tomato" / "continual" / "healthy" / "train.jpg")
    _write_image(runtime_root / "tomato" / "val" / "healthy" / "val.jpg")
    _write_image(runtime_root / "tomato" / "test" / "healthy" / "test.jpg")
    for idx in range(4):
        _write_image(runtime_root / "tomato" / "ood" / "unsupported_disease" / f"u_{idx}.jpg", color=(0, 0, 255))
        _write_image(runtime_root / "tomato" / "ood" / "blur" / f"b_{idx}.jpg", color=(0, 255, 0))

    loaders = create_training_loaders(
        data_dir=str(runtime_root),
        crop="tomato",
        batch_size=2,
        num_workers=0,
        seed=7,
        real_ood_split_dev_fraction=0.5,
        real_ood_split_min_total=8,
    )

    assert "ood" in loaders
    assert "ood_dev" in loaders
    assert len(loaders["ood"].dataset) == 4
    assert len(loaders["ood_dev"].dataset) == 4
    assert sorted(set(loaders["ood"].dataset.ood_sample_types)) == ["blur", "unsupported_disease"]
    assert sorted(set(loaders["ood_dev"].dataset.ood_sample_types)) == ["blur", "unsupported_disease"]
    manifest_path = runtime_root / "tomato" / "ood" / "ood_split_manifest.json"
    assert manifest_path.exists()
    assert getattr(loaders["ood"], "_ood_split_name") == "ood_test"


def test_create_training_loaders_reuses_pooled_ood_when_split_has_no_dev(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    _write_image(runtime_root / "tomato" / "continual" / "healthy" / "train.jpg")
    _write_image(runtime_root / "tomato" / "val" / "healthy" / "val.jpg")
    _write_image(runtime_root / "tomato" / "test" / "healthy" / "test.jpg")
    _write_image(runtime_root / "tomato" / "ood" / "unsupported_disease" / "u_0.jpg", color=(0, 0, 255))

    loaders = create_training_loaders(
        data_dir=str(runtime_root),
        crop="tomato",
        batch_size=2,
        num_workers=0,
        seed=7,
    )

    assert "ood" in loaders
    assert "ood_dev" not in loaders
    assert len(loaders["ood"].dataset) == 1
    assert getattr(loaders["ood"], "_ood_split_name") == "ood"


def test_create_training_loaders_keeps_small_ood_pool_test_only(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    _write_image(runtime_root / "tomato" / "continual" / "healthy" / "train.jpg")
    _write_image(runtime_root / "tomato" / "val" / "healthy" / "val.jpg")
    _write_image(runtime_root / "tomato" / "test" / "healthy" / "test.jpg")
    for idx in range(4):
        _write_image(runtime_root / "tomato" / "ood" / "unsupported_disease" / f"u_{idx}.jpg", color=(0, 0, 255))

    loaders = create_training_loaders(
        data_dir=str(runtime_root),
        crop="tomato",
        batch_size=2,
        num_workers=0,
        seed=7,
        real_ood_split_dev_fraction=0.5,
        real_ood_split_min_total=30,
    )

    assert "ood" in loaders
    assert "ood_dev" not in loaders
    assert len(loaders["ood"].dataset) == 4
    assert getattr(loaders["ood"], "_ood_split_name") == "ood"


def test_create_training_loaders_rejects_oe_overlap_with_real_ood(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    _write_image(runtime_root / "tomato" / "continual" / "healthy" / "train.jpg")
    _write_image(runtime_root / "tomato" / "val" / "healthy" / "val.jpg")
    _write_image(runtime_root / "tomato" / "test" / "healthy" / "test.jpg")
    _write_image(runtime_root / "tomato" / "ood" / "unsupported_disease" / "ood.jpg", color=(0, 0, 255))
    _write_image(runtime_root / "tomato" / "oe" / "aux_unknown" / "oe.jpg", color=(0, 0, 255))

    with pytest.raises(ValueError, match="Exact image overlap between real OOD evidence and OE"):
        create_training_loaders(
            data_dir=str(runtime_root),
            crop="tomato",
            batch_size=2,
            num_workers=0,
            seed=7,
        )



def test_crop_dataset_strict_error_policy_rejects_invalid_images(tmp_path: Path):
    bad_image = tmp_path / "tomato" / "continual" / "healthy" / "bad.jpg"
    bad_image.parent.mkdir(parents=True, exist_ok=True)
    bad_image.write_bytes(b"not-an-image")

    try:
        datasets.CropDataset(
            data_dir=str(tmp_path),
            crop="tomato",
            split="train",
            use_cache=False,
            error_policy="strict",
            validate_images_on_init=True,
        )
        assert False, "strict mode should reject invalid images"
    except ValueError as exc:
        assert "Failed to validate dataset image" in str(exc)



def test_crop_dataset_tolerant_error_policy_skips_invalid_images(tmp_path: Path):
    _write_image(tmp_path / "tomato" / "continual" / "healthy" / "good.jpg")
    bad_image = tmp_path / "tomato" / "continual" / "healthy" / "bad.jpg"
    bad_image.write_bytes(b"not-an-image")

    dataset = datasets.CropDataset(
        data_dir=str(tmp_path),
        crop="tomato",
        split="train",
        use_cache=False,
        error_policy="tolerant",
        validate_images_on_init=True,
    )

    assert len(dataset) == 1
    stats = dataset.get_cache_stats()
    assert stats["load_error_count"] == 1
    assert any("bad.jpg" in item for item in stats["skipped_files"])



def test_crop_dataset_supports_optional_ood_split(tmp_path: Path):
    _write_image(tmp_path / "tomato" / "ood" / "unknown_a" / "a.jpg")
    _write_image(tmp_path / "tomato" / "ood" / "unknown_b" / "b.jpg", color=(0, 0, 255))

    dataset = datasets.CropDataset(
        data_dir=str(tmp_path),
        crop="tomato",
        split="ood",
        transform=False,
        use_cache=False,
    )

    assert len(dataset) == 2
    assert dataset.labels == [-1, -1]


def test_crop_dataset_supports_explicit_ood_split_root(tmp_path: Path):
    external_ood_root = tmp_path / "custom_ood"
    _write_image(external_ood_root / "unknown_a" / "a.jpg")
    _write_image(external_ood_root / "unknown_b" / "b.jpg", color=(0, 0, 255))

    dataset = datasets.CropDataset(
        data_dir=str(tmp_path / "runtime"),
        crop="tomato",
        split="ood",
        split_root=external_ood_root,
        transform=False,
        use_cache=False,
    )

    assert len(dataset) == 2
    assert dataset.labels == [-1, -1]



def test_crop_dataset_can_cache_train_split_when_enabled(tmp_path: Path, monkeypatch):
    _write_image(tmp_path / "tomato" / "continual" / "healthy" / "train.jpg")

    dataset = datasets.CropDataset(
        data_dir=str(tmp_path),
        crop="tomato",
        split="train",
        transform=False,
        use_cache=True,
        cache_size=4,
        cache_train_split=True,
    )
    decode_calls = {"count": 0}
    original_decode = datasets.CropDataset._decode_image

    def _counting_decode(path):
        decode_calls["count"] += 1
        return original_decode(path)

    monkeypatch.setattr(datasets.CropDataset, "_decode_image", staticmethod(_counting_decode))

    dataset[0]
    dataset[0]

    assert decode_calls["count"] == 1
    stats = dataset.get_cache_stats()
    assert stats["cache_train_split"] is True
    assert stats["cache_size"] == 1



def test_create_training_loaders_rejects_class_name_mismatch(tmp_path: Path):
    _write_image(tmp_path / "tomato" / "continual" / "healthy" / "train.jpg")
    _write_image(tmp_path / "tomato" / "val" / "healthy" / "val.jpg")
    _write_image(tmp_path / "tomato" / "test" / "healthy" / "test.jpg")

    with pytest.raises(ValueError, match="Provided class_names do not match"):
        create_training_loaders(
            data_dir=str(tmp_path),
            crop="tomato",
            class_names=["healthy", "disease_a"],
            batch_size=2,
            num_workers=0,
            seed=7,
        )


def test_crop_dataset_reads_nested_class_images(tmp_path: Path):
    _write_image(tmp_path / "tomato" / "continual" / "healthy" / "camera_a" / "nested.jpg")

    dataset = datasets.CropDataset(
        data_dir=str(tmp_path),
        crop="tomato",
        split="train",
        transform=False,
        use_cache=False,
    )

    assert len(dataset) == 1
    assert dataset.image_paths[0].name == "nested.jpg"


def test_crop_dataset_default_skips_init_validation(tmp_path: Path, monkeypatch):
    _write_image(tmp_path / "tomato" / "continual" / "healthy" / "train.jpg")
    verify_calls = {"count": 0}
    original_verify = datasets.CropDataset._verify_image_file

    def _counting_verify(path):
        verify_calls["count"] += 1
        return original_verify(path)

    monkeypatch.setattr(datasets.CropDataset, "_verify_image_file", staticmethod(_counting_verify))

    dataset = datasets.CropDataset(
        data_dir=str(tmp_path),
        crop="tomato",
        split="train",
        transform=False,
        use_cache=False,
    )

    assert len(dataset) == 1
    assert verify_calls["count"] == 0



def test_crop_dataset_init_validation_uses_lightweight_verify_before_first_decode(tmp_path: Path, monkeypatch):
    _write_image(tmp_path / "tomato" / "continual" / "healthy" / "train.jpg")
    decode_calls = {"count": 0}
    verify_calls = {"count": 0}
    original_decode = datasets.CropDataset._decode_image
    original_verify = datasets.CropDataset._verify_image_file

    def _counting_decode(path):
        decode_calls["count"] += 1
        return original_decode(path)

    def _counting_verify(path):
        verify_calls["count"] += 1
        return original_verify(path)

    monkeypatch.setattr(datasets.CropDataset, "_decode_image", staticmethod(_counting_decode))
    monkeypatch.setattr(datasets.CropDataset, "_verify_image_file", staticmethod(_counting_verify))

    dataset = datasets.CropDataset(
        data_dir=str(tmp_path),
        crop="tomato",
        split="train",
        transform=False,
        use_cache=False,
        validate_images_on_init=True,
    )

    assert len(dataset) == 1
    assert verify_calls["count"] == 1
    assert decode_calls["count"] == 0

    dataset[0]

    assert decode_calls["count"] == 1
