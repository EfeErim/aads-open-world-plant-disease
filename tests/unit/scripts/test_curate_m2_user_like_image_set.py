from scripts.curate_m2_user_like_image_set import is_healthy_class


def test_is_healthy_class_handles_turkish_normalization() -> None:
    assert is_healthy_class("kayısı_sağlıklı_meyve_800")
    assert is_healthy_class("domates_sağlıklı_yaprak")
    assert not is_healthy_class("domates_late_blight_meyve")
