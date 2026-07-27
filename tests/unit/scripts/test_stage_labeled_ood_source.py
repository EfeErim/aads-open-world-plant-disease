from __future__ import annotations

import csv
import importlib.util
import io
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "stage_labeled_ood_source.py"
SPEC = importlib.util.spec_from_file_location("stage_labeled_ood_source", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_roboflow_source_group_collapses_export_and_augmentation_suffixes() -> None:
    names = (
        "Fruit_Cracking_25_jpg.rf.abc123.jpg",
        "Fruit_Cracking25-Copy_jpg.rf.def456.jpg",
        "Fruit_Cracking25_zoom_3_png.rf.ghi789.png",
    )

    assert {MODULE.roboflow_source_group(Path(name)) for name in names} == {"fruit_cracking25"}


def test_readable_image_rejects_truncated_pixel_stream(tmp_path: Path) -> None:
    path = tmp_path / "truncated.jpg"
    Image.new("RGB", (64, 64), "red").save(path, quality=95)
    payload = path.read_bytes()
    path.write_bytes(payload[:-50])

    assert not MODULE.readable_image(path)


def test_git_tree_hashes_reads_only_tracked_images(monkeypatch) -> None:
    image_payload = b"image-bytes"

    def fake_check_output(command, **_kwargs):
        assert "ls-tree" in command
        return "data/a.jpg\ndata/readme.txt\n"

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(b"deadbeef blob 11\nimage-bytes\n")

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(MODULE.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(MODULE.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())

    assert MODULE.git_tree_hashes("data") == {MODULE.hashlib.sha256(image_payload).hexdigest()}


def test_stage_source_keeps_roles_hash_disjoint(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(12):
        Image.new("RGB", (8, 8), (index, index, index)).save(source / f"{index}.png")

    rows, summary = MODULE.stage_source(
        source,
        tmp_path / "out",
        target="grape__leaf",
        disease_id="grape_black_rot",
        source_name="test",
        source_url="https://example.test",
        license_name="CC BY 4.0",
        excluded_names=set(),
    )

    assert len(rows) == 12
    assert len({row.sha256 for row in rows}) == 12
    assert set(summary["role_counts"]) == {"oe_train", "ood_dev", "ood_test"}
    assert all((tmp_path / "out" / row.relative_path).is_file() for row in rows)


def test_stage_source_excludes_rejected_names(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(4):
        Image.new("RGB", (8, 8), (index, 0, 0)).save(source / f"{index}.png")

    rows, _ = MODULE.stage_source(
        source,
        tmp_path / "out",
        target="grape__leaf",
        disease_id="grape_black_rot",
        source_name="test",
        source_url="https://example.test",
        license_name="CC BY 4.0",
        excluded_names={"3.png"},
    )

    assert len(rows) == 3


def test_stage_source_excludes_existing_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    paths = []
    for index in range(4):
        path = source / f"{index}.png"
        Image.new("RGB", (8, 8), (0, index, 0)).save(path)
        paths.append(path)

    rows, summary = MODULE.stage_source(
        source,
        tmp_path / "out",
        target="grape__leaf",
        disease_id="grape_black_rot",
        source_name="test",
        source_url="https://example.test",
        license_name="CC BY 4.0",
        excluded_names=set(),
        excluded_hashes={MODULE.sha256_file(paths[0])},
    )

    assert len(rows) == 3
    assert summary["existing_hash_match_names"] == ["0.png"]


def test_stage_source_can_fill_only_oe(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(8):
        Image.new("RGB", (8, 8), (0, 0, index)).save(source / f"{index}.png")

    rows, summary = MODULE.stage_source(
        source,
        tmp_path / "out",
        target="grape__leaf",
        disease_id="grape_black_rot",
        source_name="test",
        source_url="https://example.test",
        license_name="CC BY 4.0",
        excluded_names=set(),
        oe_target=5,
        dev_target=0,
        test_target=0,
    )

    assert len(rows) == 5
    assert {row.role for row in rows} == {"oe_train"}
    assert summary["role_counts"] == {"oe_train": 5, "ood_dev": 0, "ood_test": 0}


def test_source_manifest_urls_override_batch_fallback(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (8, 8), "red").save(source / "candidate.jpg")
    manifest = tmp_path / "candidates.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "image_url", "source_page_url"])
        writer.writeheader()
        writer.writerow(
            {
                "filename": "candidate.jpg",
                "image_url": "https://images.example/candidate.jpg",
                "source_page_url": "https://source.example/article",
            }
        )

    rows, _ = MODULE.stage_source(
        source,
        tmp_path / "out",
        target="tomato__fruit",
        disease_id="tomato_buckeye_rot",
        source_name="reviewed search",
        source_url="https://fallback.example",
        source_urls=MODULE.load_source_urls(manifest),
        license_name="not_gating",
        excluded_names=set(),
        oe_target=1,
        dev_target=0,
        test_target=0,
    )

    assert rows[0].source_url == "https://source.example/article"
