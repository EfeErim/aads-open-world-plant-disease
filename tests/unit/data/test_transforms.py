import numpy as np
import pytest
from PIL import Image

from src.data.transforms import build_image_transform, get_inference_image_transform, preprocess_image


def test_build_image_transform_training_policy_contains_expected_transforms():
    transform = build_image_transform(target_size=224, training=True)
    names = [step.__class__.__name__ for step in transform.transforms]

    assert names == [
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "RandomRotation",
        "RandAugment",
        "RandomApply",
        "ToTensor",
        "Normalize",
    ]


def test_build_image_transform_basic_training_policy_preserves_legacy_augments():
    transform = build_image_transform(target_size=224, training=True, augmentation_policy="basic")
    names = [step.__class__.__name__ for step in transform.transforms]

    assert names == [
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "RandomRotation",
        "ColorJitter",
        "RandomApply",
        "ToTensor",
        "Normalize",
    ]
    color_jitter = transform.transforms[4]
    assert color_jitter.brightness == pytest.approx((0.86, 1.14))
    assert color_jitter.contrast == pytest.approx((0.82, 1.18))
    assert color_jitter.saturation == pytest.approx((0.84, 1.16))
    assert color_jitter.hue == pytest.approx((-0.05, 0.05))


def test_build_image_transform_augmix_training_policy_uses_augmix():
    transform = build_image_transform(target_size=224, training=True, augmentation_policy="augmix")
    names = [step.__class__.__name__ for step in transform.transforms]

    assert names == [
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "RandomRotation",
        "AugMix",
        "RandomApply",
        "ToTensor",
        "Normalize",
    ]


def test_build_image_transform_none_training_policy_is_deterministic_resize():
    transform = build_image_transform(target_size=224, training=True, augmentation_policy="none")
    names = [step.__class__.__name__ for step in transform.transforms]

    assert names == ["Resize", "ToTensor", "Normalize"]


def test_build_image_transform_inference_policy_is_deterministic():
    transform = build_image_transform(target_size=224, training=False)
    names = [step.__class__.__name__ for step in transform.transforms]

    assert names == ["Resize", "ToTensor", "Normalize"]


def test_preprocess_image_returns_expected_tensor_shape():
    image = Image.new("RGB", (320, 240), color=(120, 200, 80))
    tensor = preprocess_image(image, target_size=224)

    assert tensor.shape == (3, 224, 224)


def test_preprocess_image_supports_numpy_rgb_input():
    array = np.zeros((64, 96, 3), dtype=np.uint8)
    array[..., 0] = 255

    tensor = preprocess_image(array, target_size=224)

    assert tensor.shape == (3, 224, 224)


def test_get_inference_image_transform_caches_by_target_size():
    get_inference_image_transform.cache_clear()

    first = get_inference_image_transform(224)
    second = get_inference_image_transform(224)
    third = get_inference_image_transform(256)

    assert first is second
    assert first is not third
