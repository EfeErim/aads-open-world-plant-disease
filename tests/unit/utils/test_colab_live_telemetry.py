import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

from scripts.colab_live_telemetry import ColabLiveTelemetry


def test_live_telemetry_writes_events_logs_and_artifacts(tmp_path):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_001",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=0.1,
    )
    telemetry.emit_event("train_batch", {"global_step": 1}, phase="train")
    telemetry.emit_log("batch completed", phase="train", level="info")
    telemetry.update_latest({"epoch": 1, "batch": 1})
    telemetry.write_json_artifact("training/history.json", {"train_loss": [0.1]})
    telemetry.close({"status": "ok"})

    run_dir = drive_root / "telemetry" / "run_001"
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "runtime.log").exists()
    assert (run_dir / "latest_status.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "artifact_index.json").exists()
    assert (run_dir / "artifacts" / "training" / "history.json").exists()


def test_live_telemetry_recovers_after_drive_write_failure(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_002",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=0.1,
    )

    def _fail_sync_events(*args, **kwargs):
        raise OSError("drive unavailable")

    monkeypatch.setattr(telemetry, "_sync_events", _fail_sync_events)
    telemetry.emit_event("train_batch", {"global_step": 1}, phase="train")
    local_events = telemetry.local_events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(local_events) >= 2  # includes run_started + train_batch

    monkeypatch.undo()
    telemetry.sync_pending()
    drive_events = telemetry.drive_events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(drive_events) >= 2
    telemetry.close({"status": "ok"})

    sync_state = json.loads(telemetry.local_sync_state_path.read_text(encoding="utf-8"))
    assert sync_state["events_synced"] >= 2


def test_live_telemetry_disables_secondary_writes_after_storage_exhaustion(tmp_path, monkeypatch):
    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_enospc",
        drive_root=tmp_path / "drive",
        local_root=tmp_path / "local",
        sync_interval_sec=0.1,
    )
    telemetry._record_drive_unavailable(operation="sync_events", error="OSError: [Errno 28] No space left on device")

    def _unexpected_write(*args, **kwargs):
        raise AssertionError("telemetry attempted a secondary write after ENOSPC")

    monkeypatch.setattr(telemetry, "_append_event_local", _unexpected_write)
    telemetry.emit_event("cleanup", {"status": "failed"})
    telemetry.update_latest({"event_type": "auto_disconnect_disabled"})
    telemetry.sync_pending()

    assert telemetry._storage_exhausted is True


def test_live_telemetry_throttles_batch_latest_status_writes(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_002b",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=60.0,
        latest_status_min_interval_sec=15.0,
    )

    clock = {"value": 1.0}
    monkeypatch.setattr("scripts.colab_live_telemetry.time.monotonic", lambda: clock["value"])

    telemetry.update_latest({"event_type": "batch_end", "epoch": 1, "batch": 1})
    latest_path = drive_root / "telemetry" / "run_002b" / "latest_status.json"
    first = json.loads(latest_path.read_text(encoding="utf-8"))
    assert first["status"]["batch"] == 1

    clock["value"] = 5.0
    telemetry.update_latest({"event_type": "batch_end", "epoch": 1, "batch": 2})
    second = json.loads(latest_path.read_text(encoding="utf-8"))
    assert second["status"]["batch"] == 1

    clock["value"] = 20.0
    telemetry.update_latest({"event_type": "batch_end", "epoch": 1, "batch": 3})
    third = json.loads(latest_path.read_text(encoding="utf-8"))
    assert third["status"]["batch"] == 3


def test_live_telemetry_retries_latest_status_after_transient_drive_write_failure(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_latest_retry",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=60.0,
        latest_status_min_interval_sec=15.0,
    )

    import scripts.colab_live_telemetry as telemetry_module

    original_write_json = telemetry_module.write_json
    state = {"drive_latest_failures": 0}

    def _flaky_write_json(path, payload, **kwargs):
        target = Path(path)
        if target == telemetry.drive_latest_path and state["drive_latest_failures"] == 0:
            state["drive_latest_failures"] += 1
            raise OSError("drive latest write blocked")
        return original_write_json(path, payload, **kwargs)

    monkeypatch.setattr(telemetry_module, "write_json", _flaky_write_json)

    telemetry.update_latest({"event_type": "batch_end", "epoch": 1, "batch": 1})

    assert telemetry.local_latest_path.exists()
    assert not telemetry.drive_latest_path.exists()
    assert telemetry._pending_latest_payload is not None
    assert telemetry._pending_latest_drive_sync is True

    telemetry.sync_pending()

    drive_latest = json.loads(telemetry.drive_latest_path.read_text(encoding="utf-8"))
    assert drive_latest["status"]["batch"] == 1
    assert telemetry._pending_latest_payload is None
    assert telemetry._pending_latest_drive_sync is False


def test_capture_cell_output_writes_timestamped_drive_artifact(tmp_path):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_003",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=0.1,
    )

    with telemetry.capture_cell_output("Cell 3: Parameters") as artifact_path:
        print("hello from stdout")
        print("warning from stderr", file=sys.stderr)

    assert artifact_path.exists()
    assert re.search(r"cell_outputs[\\/]\d{8}_\d{6}_\d{6}_cell_3_parameters\.log$", str(artifact_path))

    body = artifact_path.read_text(encoding="utf-8")
    assert "[CELL_CAPTURE] cell=Cell 3: Parameters" in body
    assert "run_id=run_003" in body
    assert "started_at=" in body
    assert "finished_at=" in body
    assert "duration_sec=" in body
    assert "hello from stdout" in body
    assert "[STDERR]" in body
    assert "warning from stderr" in body


def test_capture_cell_output_persists_output_on_exception(tmp_path):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_004",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=0.1,
    )

    with pytest.raises(RuntimeError, match="boom"):
        with telemetry.capture_cell_output("Cell 4: Failure"):
            print("about to fail")
            raise RuntimeError("boom")

    artifacts = list((drive_root / "telemetry" / "run_004" / "artifacts" / "cell_outputs").glob("*.log"))
    assert len(artifacts) == 1
    body = artifacts[0].read_text(encoding="utf-8")
    assert "about to fail" in body
    assert "[EXCEPTION] RuntimeError: boom" in body
    assert "[TRACEBACK]" in body
    assert "RuntimeError: boom" in body


def test_live_telemetry_spools_locally_when_drive_mount_is_missing(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    monkeypatch.setattr("scripts.colab_live_telemetry._requires_google_drive_mount", lambda path: True)
    monkeypatch.setattr("scripts.colab_live_telemetry._google_drive_is_mounted", lambda: False)

    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_005",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=0.1,
    )
    telemetry.emit_event("train_batch", {"global_step": 1}, phase="train")
    telemetry.write_json_artifact("training/history.json", {"train_loss": [0.1]})
    telemetry.close({"status": "ok"})

    drive_run_dir = drive_root / "telemetry" / "run_005"
    local_run_dir = local_root / "run_005"
    assert not drive_run_dir.exists()
    assert (local_run_dir / "events.jsonl").exists()
    assert (local_run_dir / "summary.json").exists()
    assert (local_run_dir / "artifacts" / "training" / "history.json").exists()
    events = (local_run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert any("drive_sync_unavailable" in line for line in events)


def test_live_telemetry_auto_run_ids_do_not_collide_within_same_second(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"

    class _FakeDateTime:
        values = [
            datetime(2026, 3, 11, 12, 0, 0, 1),
            datetime(2026, 3, 11, 12, 0, 0, 2),
        ]

        @classmethod
        def now(cls, tz=None):
            value = cls.values.pop(0) if cls.values else datetime(2026, 3, 11, 12, 0, 0, 3)
            if tz is not None:
                return value.replace(tzinfo=tz)
            return value

    monkeypatch.setattr("scripts.colab_live_telemetry.datetime", _FakeDateTime)
    monkeypatch.setattr("scripts.colab_live_telemetry._utc_now_iso", lambda: "2026-03-11T12:00:00Z")

    first = ColabLiveTelemetry(notebook_name="nb2", drive_root=drive_root, local_root=local_root)
    second = ColabLiveTelemetry(notebook_name="nb2", drive_root=drive_root, local_root=local_root)

    assert first.run_id == "20260311_120000_000001"
    assert second.run_id == "20260311_120000_000002"


def test_configure_repo_output_export_honors_explicit_notebook_path(tmp_path):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_export_path",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=0.1,
    )

    target = tmp_path / "repo_runs" / "run_export_path" / "notebooks" / "executed.ipynb"

    def _export(destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}", encoding="utf-8")
        return destination

    telemetry.configure_repo_output_export(
        notebook_path=target,
        export_notebook_fn=_export,
    )
    telemetry.close({"status": "ok"})

    assert target.exists()
    lines = (drive_root / "telemetry" / "run_export_path" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    exported_events = [json.loads(line) for line in lines if "repo_notebook_exported" in line]
    assert exported_events
    assert exported_events[-1]["payload"]["saved_path"] == str(target)


def test_close_emits_repo_notebook_export_unavailable_when_exporter_returns_none(tmp_path):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_export_missing",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=0.1,
    )

    target = tmp_path / "repo_runs" / "run_export_missing" / "notebooks" / "executed.ipynb"

    telemetry.configure_repo_output_export(
        notebook_path=target,
        export_notebook_fn=lambda _destination: None,
    )
    telemetry.close({"status": "ok"})

    assert not target.exists()
    lines = (drive_root / "telemetry" / "run_export_missing" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    unavailable_events = [json.loads(line) for line in lines if "repo_notebook_export_unavailable" in line]
    assert unavailable_events
    assert unavailable_events[-1]["payload"]["warning"] == "Notebook exporter returned no payload."


def test_live_telemetry_merges_guided_catalog_and_summary_metadata(tmp_path):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_guided",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=0.1,
    )

    telemetry.merge_artifact_catalog(
        [
            {
                "relative_path": "guided/00_start_here.md",
                "category": "training",
                "priority": "critical",
                "title_tr": "Baslangic rehberi",
                "description_tr": "Kullanici icin ilk okuma noktasi.",
                "reader_goal": "Nereden baslayacagini gormek",
                "format": "md",
                "generated_by": "test",
                "decision_importance": "run_overview",
                "read_order": 1,
            }
        ]
    )
    telemetry.merge_summary_metadata({"guided_artifacts": {"start_here": "guided/00_start_here.md"}})
    telemetry.close({"status": "ok"})

    artifact_index = json.loads(
        (drive_root / "telemetry" / "run_guided" / "artifact_index.json").read_text(encoding="utf-8")
    )
    summary = json.loads((drive_root / "telemetry" / "run_guided" / "summary.json").read_text(encoding="utf-8"))

    assert any(entry["relative_path"] == "guided/00_start_here.md" for entry in artifact_index)
    assert summary["metadata"]["guided_artifacts"]["start_here"] == "guided/00_start_here.md"


def test_write_text_artifact_emits_drive_warning_event_when_drive_write_fails(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_drive_warning",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=60.0,
    )

    def _fail_write_text(_relative_path, _text):
        raise OSError("drive write blocked")

    assert telemetry._drive_artifact_store is not None
    monkeypatch.setattr(telemetry._drive_artifact_store, "write_text", _fail_write_text)

    telemetry.write_text_artifact("training/history.json", '{"loss": [0.1]}')

    lines = telemetry.local_events_path.read_text(encoding="utf-8").splitlines()
    warning_events = [json.loads(line) for line in lines if "drive_sync_unavailable" in line]
    assert warning_events
    assert warning_events[-1]["payload"]["operation"] == "write_text_artifact"
    assert "drive write blocked" in warning_events[-1]["payload"]["error"]
