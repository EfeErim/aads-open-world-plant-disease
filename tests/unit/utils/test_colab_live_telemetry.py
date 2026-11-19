import json
import re
import sys

import pytest

from src.colab.live_telemetry import ColabLiveTelemetry


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
    telemetry.write_text_artifact("training/history.json", '{"train_loss": [0.1]}')
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
    monkeypatch.setattr("src.colab.live_telemetry._requires_google_drive_mount", lambda path: True)
    monkeypatch.setattr("src.colab.live_telemetry._google_drive_is_mounted", lambda: False)

    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_005",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=0.1,
    )
    telemetry.emit_event("train_batch", {"global_step": 1}, phase="train")
    telemetry.write_text_artifact("training/history.json", '{"train_loss": [0.1]}')
    telemetry.close({"status": "ok"})

    drive_run_dir = drive_root / "telemetry" / "run_005"
    local_run_dir = local_root / "run_005"
    assert not drive_run_dir.exists()
    assert (local_run_dir / "events.jsonl").exists()
    assert (local_run_dir / "summary.json").exists()
    assert (local_run_dir / "artifacts" / "training" / "history.json").exists()
    events = (local_run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert any("drive_sync_unavailable" in line for line in events)


def test_live_telemetry_starts_fresh_when_spool_state_files_are_corrupt(tmp_path, caplog):
    local_run_dir = tmp_path / "local" / "run_corrupt"
    local_run_dir.mkdir(parents=True)
    (local_run_dir / "sync_state.json").write_text("{not json", encoding="utf-8")
    (local_run_dir / "artifact_index.json").write_text('{"not": "a list"}', encoding="utf-8")

    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_corrupt",
        drive_root=tmp_path / "drive",
        local_root=tmp_path / "local",
    )

    warnings = [record.getMessage() for record in caplog.records]
    assert any("telemetry sync state" in message for message in warnings)
    assert any("artifact index" in message for message in warnings)
    assert isinstance(telemetry._artifact_index, list)
    telemetry.close({"status": "ok"})


def test_requires_google_drive_mount_falls_back_when_path_cannot_be_resolved(monkeypatch):
    from pathlib import Path

    from src.colab import live_telemetry

    def _raise(self, strict=False):
        raise OSError("unresolvable")

    monkeypatch.setattr(Path, "resolve", _raise)

    assert live_telemetry._requires_google_drive_mount(Path("/content/drive/MyDrive/x")) is True
    assert live_telemetry._requires_google_drive_mount(Path("relative/dir")) is False
