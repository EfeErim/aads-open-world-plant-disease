import json
from pathlib import Path

import pytest

from scripts.notebook_helpers import nb0_grouped_dataset_prep_helpers as nb0
from src.data import dataset_release_runtime


class _FakeTelemetry:
    def __init__(self) -> None:
        self.latest_payloads = []
        self.summary_payloads = []
        self.closed_payloads = []

    def update_latest(self, payload):
        self.latest_payloads.append(dict(payload))

    def merge_summary_metadata(self, payload):
        self.summary_payloads.append(dict(payload))

    def close(self, payload):
        self.closed_payloads.append(dict(payload))


def test_runtime_materialization_uses_copy_for_portable_prepared_dataset(tmp_path: Path, monkeypatch):
    calls = []

    def _fake_materialize_grouped_runtime_dataset(**kwargs):
        calls.append(dict(kwargs))
        return Path(kwargs["runtime_root"])

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset.materialize_grouped_runtime_dataset",
        _fake_materialize_grouped_runtime_dataset,
    )
    monkeypatch.setattr(
        nb0,
        "export_current_colab_notebook",
        lambda target: str(target),
    )

    state = {
        "validated": True,
        "audit_summary": {"runtime_ready": True},
        "dataset_root": tmp_path / "prepared_class_root" / "tomato__leaf",
        "artifact_root": tmp_path / "artifacts",
    }
    telemetry = _FakeTelemetry()

    nb0.run_materialize_runtime_dataset(
        ROOT=tmp_path,
        STATE=state,
        TELEMETRY=telemetry,
        CROP_NAME="tomato",
        PART_NAME="leaf",
        OOD_ROOT="",
        OOD_DATASET_NAME="",
        OOD_DATASET_ROOT="data/ood_dataset",
        ASK_FOR_OOD_ROOT=False,
        PREPARED_RUNTIME_ROOT="data/prepared_runtime_datasets",
        MATERIALIZE_AFTER_REVIEW=True,
        REPO_NOTEBOOK_OUTPUT_PATH=tmp_path / "runs" / "run_1" / "notebooks" / "executed.ipynb",
        REPO_RUN_DIR=tmp_path / "runs" / "run_1",
        REPO_RUN_EXPORTS={},
    )

    assert calls
    assert calls[0]["materialization_strategy"] == "copy"
    assert state["runtime_dataset_root"] == tmp_path / "data" / "prepared_runtime_datasets"
    assert telemetry.closed_payloads[-1]["materialized"] is True


def _write_release_manifest(root: Path) -> None:
    path = root / "docs" / "evidence" / "current" / "dataset_release" / "test_github_release_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "repository": "owner/repo",
                "release_tag": "aads-dataset-v1.0.0",
                "release_id": 42,
                "release_manifest_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )


def test_release_fetch_requires_read_only_token(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AADS_GITHUB_RELEASE_READ_TOKEN", raising=False)
    monkeypatch.delenv("AADS_GITHUB_RELEASE_CI_READ_TOKEN", raising=False)

    with pytest.raises(nb0.DatasetReleaseAccessBlocker, match="AADS_GITHUB_RELEASE_READ_TOKEN"):
        nb0.fetch_materialize_dataset_release(
            root=tmp_path,
            repository="owner/repo",
            release_tag="aads-dataset-v1.0.0",
            target="tomato__leaf",
            cache_root="cache",
        )


def test_release_fetch_materializes_verified_target(tmp_path: Path, monkeypatch):
    _write_release_manifest(tmp_path)
    calls = []

    def _fetch(manifest_path, cache_root, **kwargs):
        calls.append((manifest_path, cache_root, kwargs))
        return {"verified": True, "downloaded_count": 1}

    def _materialize(_manifest_path, _cache_root, destination):
        for split in ("continual", "val", "test"):
            (destination / "tomato__leaf" / split).mkdir(parents=True, exist_ok=True)
        return {"verified": True, "file_count": 3}

    monkeypatch.setattr(dataset_release_runtime, "fetch_dataset_release", _fetch)
    monkeypatch.setattr(dataset_release_runtime, "materialize_dataset_release", _materialize)

    result = nb0.fetch_materialize_dataset_release(
        root=tmp_path,
        repository="owner/repo",
        release_tag="aads-dataset-v1.0.0",
        target="tomato__leaf",
        cache_root="cache",
        token="read-only-token",
    )

    assert result["verified"] is True
    assert result["read_only"] is True
    assert Path(result["selected_dataset_root"]).is_dir()
    assert calls[0][2]["repository"] == "owner/repo"


def test_release_network_failure_is_access_blocker(tmp_path: Path, monkeypatch):
    _write_release_manifest(tmp_path)

    def _fail(*args, **kwargs):
        raise RuntimeError("GitHub API GET failed (503)")

    monkeypatch.setattr(dataset_release_runtime, "fetch_dataset_release", _fail)

    with pytest.raises(nb0.DatasetReleaseAccessBlocker, match="DATASET_RELEASE_ACCESS_BLOCKER"):
        nb0.fetch_materialize_dataset_release(
            root=tmp_path,
            repository="owner/repo",
            release_tag="aads-dataset-v1.0.0",
            target="tomato__leaf",
            cache_root="cache",
            token="read-only-token",
        )
