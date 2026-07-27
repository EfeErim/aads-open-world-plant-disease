from pathlib import Path

import pytest
from PIL import Image

from src.router.prototype_bank import build_prototype_bank, centroid, euclidean_distance


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), color=color).save(path)


def _write_curation_csv(path: Path, rows: list[dict[str, str]]) -> None:
    import csv

    headers = [
        "image_id",
        "source",
        "resolved_image",
        "expected_target",
        "expected_class",
        "corrected_class",
        "prototype_target",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def test_build_prototype_bank_from_synthetic_fixture(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    _write_image(dataset_root / "tomato__leaf" / "train" / "healthy" / "a.png", (20, 80, 40))
    _write_image(dataset_root / "tomato__leaf" / "test" / "late_blight" / "b.png", (180, 30, 20))
    _write_image(dataset_root / "tomato__leaf" / "ood" / "nonplant" / "c.png", (0, 0, 0))

    payload = build_prototype_bank(
        dataset_root=dataset_root,
        max_images_per_class=2,
        created_at="20260617T000000Z",
    )

    assert payload["schema_version"] == "router_prototype_bank.v1"
    assert payload["embedding_backend"] == "image_stats_v1"
    assert payload["summary"]["targets"] == ["tomato__leaf"]
    assert payload["summary"]["sample_count"] == 2
    assert payload["summary"]["class_prototype_count"] == 2
    assert payload["target_prototypes"]["tomato__leaf"]["split_counts"] == {"test": 1, "train": 1}
    assert "tomato__leaf::late_blight" in payload["class_prototypes"]
    assert payload["skipped"] == []


def test_build_prototype_bank_honors_max_images_per_class(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    _write_image(dataset_root / "grape__fruit" / "train" / "healthy" / "a.png", (10, 10, 10))
    _write_image(dataset_root / "grape__fruit" / "train" / "healthy" / "b.png", (20, 20, 20))

    payload = build_prototype_bank(
        dataset_root=dataset_root,
        max_images_per_class=1,
        created_at="20260617T000000Z",
    )

    assert payload["summary"]["sample_count"] == 1
    assert payload["target_prototypes"]["grape__fruit"]["sample_count"] == 1


def test_build_prototype_bank_consumes_reviewed_curation_manifests(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    curation_root = tmp_path / "curation"
    curated_positive = tmp_path / "reviewed" / "positive.png"
    curated_cross_target_negative = tmp_path / "reviewed" / "cross_target_negative.png"
    curated_same_target_negative = tmp_path / "reviewed" / "same_target_negative.png"
    _write_image(dataset_root / "tomato__leaf" / "train" / "healthy" / "a.png", (20, 80, 40))
    _write_image(curated_positive, (25, 85, 45))
    _write_image(curated_cross_target_negative, (190, 40, 30))
    _write_image(curated_same_target_negative, (30, 90, 50))
    _write_curation_csv(
        curation_root / "prototype_positive_manifest.csv",
        [
            {
                "image_id": "demo_001",
                "resolved_image": str(curated_positive),
                "expected_target": "tomato__leaf",
                "expected_class": "late_blight",
                "corrected_class": "late_blight",
            }
        ],
    )
    _write_curation_csv(
        curation_root / "prototype_hard_negative_manifest.csv",
        [
            {
                "image_id": "demo_002",
                "resolved_image": str(curated_cross_target_negative),
                "expected_target": "tomato__leaf",
                "expected_class": "healthy",
                "prototype_target": "tomato__fruit",
            },
            {
                "image_id": "demo_003",
                "resolved_image": str(curated_same_target_negative),
                "expected_target": "tomato__leaf",
                "expected_class": "healthy",
                "prototype_target": "tomato__leaf",
            }
        ],
    )

    payload = build_prototype_bank(
        dataset_root=dataset_root,
        curation_root=curation_root,
        repo_root=tmp_path,
        created_at="20260617T000000Z",
    )

    assert payload["summary"]["sample_count"] == 2
    assert payload["summary"]["dataset_sample_count"] == 1
    assert payload["summary"]["curation_positive_count"] == 1
    assert payload["summary"]["hard_negative_count"] == 1
    assert payload["summary"]["skipped_count"] == 1
    assert payload["target_prototypes"]["tomato__leaf"]["split_counts"] == {"curated": 1, "train": 1}
    assert "tomato__leaf::late_blight" in payload["class_prototypes"]
    assert payload["hard_negative_prototypes"]["tomato__fruit"]["sample_count"] == 1
    assert "tomato__leaf" not in payload["hard_negative_prototypes"]
    assert payload["skipped"] == [
        {"image_id": "demo_003", "role": "prototype_hard_negative", "reason": "same_target_hard_negative_not_used"}
    ]


def test_build_prototype_bank_fails_when_curation_manifest_is_missing(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    curation_root = tmp_path / "curation"
    _write_image(dataset_root / "tomato__leaf" / "train" / "healthy" / "a.png", (20, 80, 40))
    _write_curation_csv(
        curation_root / "prototype_positive_manifest.csv",
        [
            {
                "image_id": "demo_001",
                "resolved_image": str(tmp_path / "missing.png"),
                "expected_target": "tomato__leaf",
                "expected_class": "late_blight",
            }
        ],
    )

    with pytest.raises(FileNotFoundError, match="required prototype curation manifest"):
        build_prototype_bank(
            dataset_root=dataset_root,
            curation_root=curation_root,
            repo_root=tmp_path,
            created_at="20260617T000000Z",
        )


def test_build_prototype_bank_fails_when_curation_rows_load_zero_records(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    curation_root = tmp_path / "curation"
    _write_image(dataset_root / "tomato__leaf" / "train" / "healthy" / "a.png", (20, 80, 40))
    _write_curation_csv(
        curation_root / "prototype_positive_manifest.csv",
        [
            {
                "image_id": "demo_001",
                "resolved_image": str(tmp_path / "missing_positive.png"),
                "expected_target": "tomato__leaf",
                "expected_class": "late_blight",
            }
        ],
    )
    _write_curation_csv(
        curation_root / "prototype_hard_negative_manifest.csv",
        [
            {
                "image_id": "demo_002",
                "resolved_image": str(tmp_path / "missing_negative.png"),
                "expected_target": "tomato__leaf",
                "expected_class": "healthy",
                "prototype_target": "tomato__fruit",
            }
        ],
    )

    with pytest.raises(ValueError, match="loaded zero usable prototype curation rows"):
        build_prototype_bank(
            dataset_root=dataset_root,
            curation_root=curation_root,
            repo_root=tmp_path,
            created_at="20260617T000000Z",
        )


def test_vector_math_helpers_are_deterministic():
    center = centroid([(0.0, 2.0), (2.0, 4.0)])

    assert center == (1.0, 3.0)
    assert round(euclidean_distance((0.0, 0.0), (3.0, 4.0)), 4) == 5.0
