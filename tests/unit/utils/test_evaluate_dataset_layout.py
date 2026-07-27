from pathlib import Path

from scripts.evaluate_dataset_layout import evaluate_layout


def _touch_images(root: Path, class_name: str, count: int) -> None:
    class_dir = root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        (class_dir / f"image_{idx}.jpg").write_bytes(b"fake")


def test_evaluate_layout_missing_root_returns_error_payload(tmp_path):
    missing_root = tmp_path / "does_not_exist"
    result = evaluate_layout(missing_root)

    assert result["ok"] is False
    assert result["classes"] == []
    assert result["errors"]
    assert "does not exist" in result["errors"][0]


def test_evaluate_layout_rejects_pre_split_layout(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    _touch_images(tmp_path, "tomato_healthy", 3)

    result = evaluate_layout(tmp_path)

    assert result["ok"] is False
    assert any("flat class-root layout" in item for item in result["errors"])


def test_evaluate_layout_matches_notebook_contract_and_counts(tmp_path):
    _touch_images(tmp_path, "tomato_leaf_blight", 10)
    _touch_images(tmp_path, "pepper_healthy", 2)

    result = evaluate_layout(
        tmp_path,
        min_classes=2,
        min_images_per_class=3,
    )

    assert result["ok"] is True
    assert result["split_policy"] == "Grouped family-aware split after audit (raw image counts are an upper bound only)"
    assert result["expected_layout"] == "<root>/<class_name>/<images>"
    assert result["summary"]["num_classes"] == 2
    assert result["summary"]["total_images"] == 12
    assert result["summary"]["estimated_train_images"] == 10
    assert result["summary"]["estimated_val_images"] == 1
    assert result["summary"]["estimated_test_images"] == 1
    assert any("pepper_healthy" in item for item in result["warnings"])
