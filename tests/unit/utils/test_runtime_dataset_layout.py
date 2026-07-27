import shutil
import zipfile
from pathlib import Path

from PIL import Image

from scripts.colab_dataset_layout import (
    build_runtime_split_manifest,
    list_dataset_directories_from_parent,
    list_repo_dataset_directories,
    looks_like_class_root_dataset,
    prepare_runtime_dataset_layout,
    resolve_dataset_directory_from_parent,
    resolve_direct_repo_dataset_root,
    resolve_notebook_training_classes,
    resolve_repo_dataset_directory,
)

# test helpers constants
IMAGE_SIZE = (8, 8)
IMAGE_COLOR = (255, 0, 0)
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _write_images(root: Path, class_name: str, count: int) -> None:
    class_dir = root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        Image.new("RGB", IMAGE_SIZE, color=IMAGE_COLOR).save(class_dir / f"image_{idx}.jpg")


def _write_zip_dataset(archive_path: Path, dataset_name: str, class_name: str, count: int) -> None:
    source_root = archive_path.parent / f"{archive_path.stem}_source"
    _write_images(source_root / dataset_name, class_name, count)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        for file_path in source_root.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(source_root).as_posix())
    # remove transient source tree to avoid leaving test artifacts on disk
    shutil.rmtree(source_root, ignore_errors=True)


def test_build_runtime_split_manifest_contains_counts(tmp_path: Path):
    _write_images(tmp_path, "Tomato Healthy", 4)
    _write_images(tmp_path, "Tomato Blight", 2)

    manifest = build_runtime_split_manifest(class_root=tmp_path, crop_name="tomato", seed=123)

    assert manifest["crop"] == "tomato"
    assert manifest["seed"] == 123
    assert manifest["split_policy"] == "80/10/10"
    assert manifest["summary"]["num_classes"] == 2
    assert any(item["class_name"] == "tomato_healthy" for item in manifest["classes"])


def test_prepare_runtime_dataset_layout_writes_split_manifest(tmp_path: Path):
    source_root = tmp_path / "source"
    runtime_root = tmp_path / "runtime"
    _write_images(source_root, "Healthy", 5)
    _write_images(source_root, "Disease A", 4)

    result_root = prepare_runtime_dataset_layout(
        source_root,
        "tomato",
        seed=42,
        runtime_root=runtime_root,
    )

    crop_root = result_root / "tomato"
    assert (crop_root / "split_manifest.json").is_file()
    assert (crop_root / "continual").is_dir()
    assert (crop_root / "val").is_dir()
    assert (crop_root / "test").is_dir()


def test_prepare_runtime_dataset_layout_preserves_nested_relative_paths(tmp_path: Path):
    source_root = tmp_path / "source"
    runtime_root = tmp_path / "runtime"
    nested_dir = source_root / "Healthy" / "camera_a"
    nested_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", IMAGE_SIZE, color=IMAGE_COLOR).save(nested_dir / "image_nested.jpg")

    result_root = prepare_runtime_dataset_layout(
        source_root,
        "tomato",
        seed=42,
        runtime_root=runtime_root,
    )

    crop_root = result_root / "tomato"
    nested_targets = list(crop_root.rglob("image_nested.jpg"))
    assert len(nested_targets) == 1
    assert "camera_a" in nested_targets[0].parts


def test_prepare_runtime_dataset_layout_rebuilds_when_manifest_matches_but_split_tree_is_incomplete(tmp_path: Path):
    source_root = tmp_path / "source"
    runtime_root = tmp_path / "runtime"
    _write_images(source_root, "Healthy", 5)

    result_root = prepare_runtime_dataset_layout(
        source_root,
        "tomato",
        seed=42,
        runtime_root=runtime_root,
    )

    crop_root = result_root / "tomato"
    shutil.rmtree(crop_root / "continual")

    prepare_runtime_dataset_layout(
        source_root,
        "tomato",
        seed=42,
        runtime_root=runtime_root,
    )

    assert (crop_root / "continual").is_dir()
    files = [p for p in (crop_root / "continual").rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    assert len(files) > 0


def test_resolve_notebook_training_classes_uses_taxonomy_when_aliases_cover_dataset():
    resolution = resolve_notebook_training_classes(
        available_classes=[
            "Tomato Early Blight",
            "Tomato Healthy Leaf",
            "Tomato Late Blight",
        ],
        crop_name="tomato",
        taxonomy={
            "crop_specific_diseases": {
                "tomato": [
                    "early blight",
                    "late blight",
                ]
            }
        },
    )

    assert resolution["used_taxonomy_filter"] is True
    assert resolution["reason"] == "full_taxonomy_alignment"
    assert resolution["unmatched_classes"] == []
    assert set(resolution["selected_classes"]) == {
        "tomato_early_blight",
        "tomato_healthy_leaf",
        "tomato_late_blight",
    }


def test_resolve_notebook_training_classes_falls_back_to_all_available_when_taxonomy_is_incomplete():
    resolution = resolve_notebook_training_classes(
        available_classes=[
            "Tomato Early Blight",
            "Tomato Healthy Leaf",
            "Tomato Spider Mites",
        ],
        crop_name="tomato",
        taxonomy={
            "crop_specific_diseases": {
                "tomato": [
                    "early blight",
                ]
            }
        },
    )

    assert resolution["used_taxonomy_filter"] is False
    assert resolution["reason"] == "partial_taxonomy_alignment_fallback"
    assert "tomato_spider_mites" in resolution["unmatched_classes"]
    assert set(resolution["selected_classes"]) == {
        "tomato_early_blight",
        "tomato_healthy_leaf",
        "tomato_spider_mites",
    }


def test_list_repo_dataset_directories_returns_sorted_child_dirs(tmp_path: Path):
    repo_root = tmp_path / "repo"
    dataset_parent = repo_root / "data" / "class_root_dataset"
    (dataset_parent / "grape_leaf").mkdir(parents=True)
    (dataset_parent / "grape_fruit").mkdir(parents=True)

    result = list_repo_dataset_directories(
        repo_root=repo_root,
        repo_relative_root="data/class_root_dataset",
    )

    assert [path.name for path in result] == ["grape_fruit", "grape_leaf"]


def test_list_repo_dataset_directories_includes_dataset_roots_from_zip_archive(tmp_path: Path):
    repo_root = tmp_path / "repo"
    dataset_parent = repo_root / "data" / "ood_dataset"
    archive_path = dataset_parent / "ood_archive_sources.zip"
    _write_zip_dataset(archive_path, "tomato_leaf_ood_best", "unsupported_tomato_unknowns", 1)

    result = list_repo_dataset_directories(
        repo_root=repo_root,
        repo_relative_root="data/ood_dataset",
    )

    selected_paths = [path for path in result if path.name == "tomato_leaf_ood_best"]
    assert len(selected_paths) == 1
    assert selected_paths[0].is_dir()
    assert ".runtime_tmp" in selected_paths[0].parts


def test_resolve_repo_dataset_directory_materializes_zip_archive_on_demand(tmp_path: Path):
    repo_root = tmp_path / "repo"
    dataset_parent = repo_root / "data" / "ood_dataset"
    archive_path = dataset_parent / "ood_archive_sources.zip"
    _write_zip_dataset(archive_path, "tomato_leaf_ood_best", "unsupported_tomato_unknowns", 1)

    selected_name, selected_path, dataset_names = resolve_repo_dataset_directory(
        repo_root=repo_root,
        repo_relative_root="data/ood_dataset",
        requested_name="tomato_leaf_ood_best",
        prompt_label="OOD dataset",
    )

    assert selected_name == "tomato_leaf_ood_best"
    assert selected_path.is_dir()
    assert selected_path.name == "tomato_leaf_ood_best"
    assert dataset_names == ["tomato_leaf_ood_best"]
    assert (selected_path / "unsupported_tomato_unknowns" / "image_0.jpg").exists()


def test_resolve_dataset_directory_from_parent_prompts_for_drive_style_parent(tmp_path: Path):
    dataset_parent = tmp_path / "drive" / "datasets"
    (dataset_parent / "grape_leaf" / "healthy").mkdir(parents=True)
    (dataset_parent / "tomato_leaf" / "healthy").mkdir(parents=True)
    prompts: list[str] = []
    printed: list[str] = []

    def _input(prompt: str) -> str:
        prompts.append(prompt)
        return "1"

    def _print(message: str) -> None:
        printed.append(message)

    selected_name, selected_path, dataset_names = resolve_dataset_directory_from_parent(
        dataset_parent=dataset_parent,
        requested_name="",
        prompt_label="Drive dataset",
        input_fn=_input,
        print_fn=_print,
    )

    assert selected_name == "grape_leaf"
    assert selected_path == dataset_parent / "grape_leaf"
    assert dataset_names == ["grape_leaf", "tomato_leaf"]
    assert prompts
    assert any("Drive dataset" in line for line in printed)


def test_list_dataset_directories_from_parent_accepts_direct_class_root(tmp_path: Path):
    dataset_root = tmp_path / "drive" / "Uzum Yaprak"
    _write_images(dataset_root, "healthy", 1)

    result = list_dataset_directories_from_parent(dataset_parent=dataset_root)

    assert looks_like_class_root_dataset(dataset_root)
    assert result == [dataset_root]


def test_resolve_direct_repo_dataset_root_accepts_dataset_path_under_repo_staging_parent(tmp_path: Path):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "data" / "class_root_dataset" / "grape_fruit"
    (dataset_root / "healthy").mkdir(parents=True)

    selected = resolve_direct_repo_dataset_root(
        repo_root=repo_root,
        repo_relative_root="data/class_root_dataset/grape_fruit",
    )

    assert selected == ("grape_fruit", dataset_root)


def test_resolve_repo_dataset_directory_accepts_direct_dataset_root(tmp_path: Path):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "data" / "class_root_dataset" / "grape_fruit"
    (dataset_root / "healthy").mkdir(parents=True)

    selected_name, selected_path, dataset_names = resolve_repo_dataset_directory(
        repo_root=repo_root,
        repo_relative_root="data/class_root_dataset/grape_fruit",
        prompt_label="class-root dataset",
    )

    assert selected_name == "grape_fruit"
    assert selected_path == dataset_root
    assert dataset_names == ["grape_fruit"]


def test_resolve_repo_dataset_directory_accepts_explicit_name(tmp_path: Path):
    repo_root = tmp_path / "repo"
    dataset_parent = repo_root / "data" / "class_root_dataset"
    (dataset_parent / "grape_leaf").mkdir(parents=True)
    (dataset_parent / "grape_fruit").mkdir(parents=True)

    selected_name, selected_path, dataset_names = resolve_repo_dataset_directory(
        repo_root=repo_root,
        repo_relative_root="data/class_root_dataset",
        requested_name="grape_leaf",
        prompt_label="class-root dataset",
    )

    assert selected_name == "grape_leaf"
    assert selected_path == dataset_parent / "grape_leaf"
    assert dataset_names == ["grape_fruit", "grape_leaf"]


def test_resolve_repo_dataset_directory_prompts_for_index_when_name_missing(tmp_path: Path):
    repo_root = tmp_path / "repo"
    dataset_parent = repo_root / "data" / "class_root_dataset"
    (dataset_parent / "grape_leaf").mkdir(parents=True)
    (dataset_parent / "grape_fruit").mkdir(parents=True)
    prompts: list[str] = []
    printed: list[str] = []

    def _input(prompt: str) -> str:
        prompts.append(prompt)
        return "2"

    def _print(message: str) -> None:
        printed.append(message)

    selected_name, selected_path, dataset_names = resolve_repo_dataset_directory(
        repo_root=repo_root,
        repo_relative_root="data/class_root_dataset",
        requested_name="",
        prompt_label="class-root dataset",
        input_fn=_input,
        print_fn=_print,
    )

    assert selected_name == "grape_leaf"
    assert selected_path == dataset_parent / "grape_leaf"
    assert dataset_names == ["grape_fruit", "grape_leaf"]
    assert prompts
    assert any("grape_fruit" in line for line in printed)


def test_prepare_runtime_dataset_layout_refuses_to_delete_existing_runtime_tree_when_manifest_validation_fails(
    tmp_path: Path,
    monkeypatch,
):
    source_root = tmp_path / "source"
    runtime_root = tmp_path / "runtime"
    _write_images(source_root, "Healthy", 5)

    result_root = prepare_runtime_dataset_layout(
        source_root,
        "tomato",
        seed=42,
        runtime_root=runtime_root,
    )

    crop_root = result_root / "tomato"
    marker_path = crop_root / "keep_me.txt"
    marker_path.write_text("preserve", encoding="utf-8")

    def _fail_read_json(*args, **kwargs):
        raise ValueError("bad manifest")

    monkeypatch.setattr("scripts.colab_dataset_layout.read_json", _fail_read_json)

    try:
        prepare_runtime_dataset_layout(
            source_root,
            "tomato",
            seed=42,
            runtime_root=runtime_root,
        )
        raise AssertionError("prepare_runtime_dataset_layout was expected to fail when manifest validation breaks.")
    except RuntimeError as exc:
        assert "refusing to delete" in str(exc)

    assert marker_path.exists()
