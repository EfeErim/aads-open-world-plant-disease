from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.data.dataset_release import (
    DatasetContractError,
    DatasetResourceLimits,
    canonical_sha256,
    read_json_dict,
    validate_archive,
)


def test_archive_rejects_path_escape(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.jpg", b"not-an-image")
    with pytest.raises(DatasetContractError, match="Unsafe relative path"):
        validate_archive(archive)


def test_archive_rejects_disallowed_extension_and_duplicates(tmp_path: Path) -> None:
    script = tmp_path / "script.zip"
    with zipfile.ZipFile(script, "w") as handle:
        handle.writestr("tomato__leaf/run.sh", b"echo")
    with pytest.raises(DatasetContractError, match="extension is not allowed"):
        validate_archive(script)

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as handle:
        handle.writestr("a.jpg", b"1")
        with pytest.warns(UserWarning):
            handle.writestr("a.jpg", b"2")
    with pytest.raises(DatasetContractError, match="Duplicate archive member"):
        validate_archive(duplicate)


def test_archive_enforces_size_and_expansion_limits(tmp_path: Path) -> None:
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("a.jpg", b"\x00" * 100_000)
    with pytest.raises(DatasetContractError, match="file-size limit"):
        validate_archive(archive, DatasetResourceLimits(max_file_bytes=10))
    with pytest.raises(DatasetContractError, match="expansion ratio"):
        validate_archive(archive, DatasetResourceLimits(max_archive_expansion_ratio=2.0))


def test_archive_accepts_runtime_layout(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("adapter_ood_oe_evidence_manifest.csv", b"path,role\n")
        handle.writestr("tomato__leaf/split_manifest.json", b"{}")
    assert validate_archive(archive)["member_count"] == 2


def test_canonical_hash_ignores_key_order_and_json_must_be_object(tmp_path: Path) -> None:
    assert canonical_sha256({"a": 1, "b": [1, 2]}) == canonical_sha256({"b": [1, 2], "a": 1})
    path = tmp_path / "list.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(DatasetContractError, match="Expected a JSON object"):
        read_json_dict(path)
