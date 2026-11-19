from datetime import datetime
from pathlib import Path

from src.colab.training_helpers import (
    maybe_auto_disconnect_colab_runtime,
)
from src.colab.training_session import (
    build_notebook_completion_report,
    build_notebook_run_dir,
    build_notebook_run_id,
    cleanup_notebook_training_state,
    persist_production_readiness_artifact,
    persist_validation_artifacts,
)


def test_cleanup_notebook_training_state_releases_loaders_and_globals(monkeypatch):
    class Iterator:
        def __init__(self):
            self.shutdown = False

        def _shutdown_workers(self):
            self.shutdown = True

    class Loader:
        def __init__(self):
            self._iterator = Iterator()

    loader = Loader()
    iterator = loader._iterator
    state = {"adapter": object(), "loaders": {"train": loader}, "history": {"loss": [1.0]}}
    shared_globals = {"STATE": state, "adapter": object(), "trainer": object(), "session": object()}
    messages = []
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    cleanup_notebook_training_state(shared_globals, label="tomato__leaf", print_fn=messages.append)

    assert iterator.shutdown is True
    assert loader._iterator is None
    assert "adapter" not in state
    assert "loaders" not in state
    assert "history" not in state
    assert "adapter" not in shared_globals
    assert "trainer" not in shared_globals
    assert "session" not in shared_globals
    assert messages[-1] == "[CLEANUP] Released runtime state after tomato__leaf."


def test_persist_validation_artifacts_writes_metric_gate(tmp_path: Path):
    result = persist_validation_artifacts(
        root=tmp_path,
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 1, 1],
        classes=["healthy", "disease_a"],
        telemetry=None,
        require_ood=False,
        context={"crop": "tomato"},
    )

    metric_gate_path = result["paths"]["metric_gate_json"]
    assert metric_gate_path.exists()
    assert result["metric_gate"]["evaluation"]["gating"]["status"] in {"soft", "ready"}


def test_persist_production_readiness_artifact_forwards_require_ood_flag(tmp_path: Path):
    readiness = persist_production_readiness_artifact(
        root=tmp_path,
        classification_metric_gate={
            "metrics": {
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "weighted_f1": 1.0,
                "balanced_accuracy": 1.0,
            }
        },
        classification_split="test",
        ood_evidence_source="unavailable",
        ood_metrics={},
        require_ood=False,
    )

    assert readiness["payload"]["ood_evidence"]["evaluation"]["require_ood"] is False


class _FakeTelemetry:
    def __init__(self, summary_path: Path):
        self.local_summary_path = summary_path
        self.latest_payloads = []
        self.sync_calls = 0

    def update_latest(self, payload, **_kwargs):
        self.latest_payloads.append(dict(payload))

    def sync_pending(self):
        self.sync_calls += 1


def test_build_notebook_completion_report_marks_ready_when_outputs_exist(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    notebook_export_path = tmp_path / "notebooks" / "executed.ipynb"
    notebook_export_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_export_path.write_text("{}", encoding="utf-8")

    repo_run_exports = {}
    for name in ("outputs", "telemetry", "checkpoint_state"):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        repo_run_exports[name] = str(path)

    report = build_notebook_completion_report(
        state={
            "evaluation_artifacts": {"test": {"metric_gate": {}}},
            "behavioral_acceptance": {"status": "fail", "pass": False},
        },
        telemetry=_FakeTelemetry(summary_path),
        repo_run_exports=repo_run_exports,
        notebook_export_path=notebook_export_path,
    )

    assert report["ready"] is False
    assert report["missing"] == ["adapter_behavioral_acceptance_failed"]
    assert report["soft_missing"] == []
    assert report["evaluation_splits"] == ["test"]
    assert report["adapter_behavioral_acceptance_status"] == "fail"


def test_build_notebook_completion_report_treats_missing_notebook_export_as_soft_missing(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    repo_run_exports = {}
    for name in ("outputs", "telemetry", "checkpoint_state"):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        repo_run_exports[name] = str(path)

    report = build_notebook_completion_report(
        state={
            "evaluation_artifacts": {"test": {"metric_gate": {}}},
            "behavioral_acceptance": {"status": "pass", "pass": True},
        },
        telemetry=_FakeTelemetry(summary_path),
        repo_run_exports=repo_run_exports,
        notebook_export_path=tmp_path / "missing.ipynb",
    )

    assert report["ready"] is True
    assert report["missing"] == []
    assert report["soft_missing"] == ["executed_notebook_export"]
    assert report["checks"]["executed_notebook_export"] is False


def test_maybe_auto_disconnect_colab_runtime_calls_unassign_when_ready(tmp_path: Path, monkeypatch):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    notebook_export_path = tmp_path / "executed.ipynb"
    notebook_export_path.write_text("{}", encoding="utf-8")

    repo_run_exports = {}
    for name in ("outputs", "telemetry", "checkpoint_state"):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        repo_run_exports[name] = str(path)

    telemetry = _FakeTelemetry(summary_path)
    completion_report = build_notebook_completion_report(
        state={
            "evaluation_artifacts": {"test": {"metric_gate": {}}},
            "behavioral_acceptance": {"status": "pass", "pass": True},
        },
        telemetry=telemetry,
        repo_run_exports=repo_run_exports,
        notebook_export_path=notebook_export_path,
    )

    calls = []

    class _FakeRuntime:
        def unassign(self):
            calls.append("unassign")

    monkeypatch.setattr(
        "src.colab.training_helpers._resolve_colab_runtime_api",
        lambda: _FakeRuntime(),
    )

    lines = []
    result = maybe_auto_disconnect_colab_runtime(
        enabled=True,
        grace_period_sec=0.0,
        telemetry=telemetry,
        completion_report=completion_report,
        print_fn=lines.append,
    )

    assert result["disconnect_requested"] is True
    assert calls == ["unassign"]
    assert telemetry.sync_calls == 1
    assert telemetry.latest_payloads[-1]["phase"] == "auto_disconnect_pending"
    assert lines == ["[COLAB] Work complete. Disconnecting runtime now to avoid idle credit use."]


def test_build_notebook_run_id_includes_crop_part_and_timestamp():
    run_id = build_notebook_run_id(
        "Tomato Leaf",
        "Upper Part",
        now=datetime(2026, 3, 23, 10, 11, 12),
    )

    assert run_id == "tomato_leaf_upper_part_2026-03-23_10-11-12"


def test_build_notebook_run_dir_groups_by_crop_and_part(tmp_path: Path):
    run_dir = build_notebook_run_dir(
        tmp_path,
        "Tomato Leaf",
        "Upper Part",
        "tomato_leaf_upper_part_2026-03-23_10-11-12",
    )

    assert run_dir == tmp_path / "runs" / "tomato_leaf" / "upper_part" / "tomato_leaf_upper_part_2026-03-23_10-11-12"


def test_resolve_colab_runtime_api_returns_none_outside_colab(monkeypatch):
    import sys

    from src.colab.training_session import _resolve_colab_runtime_api

    monkeypatch.setitem(sys.modules, "google.colab", None)

    assert _resolve_colab_runtime_api() is None


def test_auto_disconnect_survives_telemetry_write_failure(tmp_path: Path):
    class _BrokenTelemetry:
        local_summary_path = None

        def update_latest(self, payload):
            raise OSError("disk full")

        def sync_pending(self):
            raise OSError("disk full")

    lines = []
    result = maybe_auto_disconnect_colab_runtime(
        enabled=False,
        telemetry=_BrokenTelemetry(),
        completion_report={"ready": False},
        print_fn=lines.append,
    )

    assert result["disconnect_requested"] is False
    assert lines == ["[COLAB] Auto-disconnect disabled."]
