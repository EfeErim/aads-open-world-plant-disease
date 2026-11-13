#!/usr/bin/env python3
"""Live Colab telemetry utilities with local spool + Drive synchronization."""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from src.shared.artifacts import ArtifactStore
from src.shared.json_utils import deep_merge, ensure_parent, read_json, write_json

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _slugify_capture_name(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    return slug or "cell"


def _append_line(path: Path, line: str) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        if not line.endswith("\n"):
            handle.write("\n")


def _requires_google_drive_mount(path: Path) -> bool:
    drive_mount = Path("/content/drive")
    try:
        resolved = Path(path).expanduser().resolve()
    except Exception:
        import logging
        logging.exception('Unhandled exception')
        raise
        resolved = Path(path).expanduser()
    try:
        resolved.relative_to(drive_mount)
        return True
    except ValueError:
        return False


def _google_drive_is_mounted() -> bool:
    return os.path.ismount("/content/drive")


class _TeeTextStream:
    def __init__(self, *streams: Any) -> None:
        self._streams = [stream for stream in streams if stream is not None]

    def write(self, data: str) -> int:
        text = str(data)
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            flush = getattr(stream, "flush", None)
            if callable(flush):
                flush()

    def isatty(self) -> bool:
        return any(bool(getattr(stream, "isatty", lambda: False)()) for stream in self._streams)

    def writable(self) -> bool:
        return True

    @property
    def encoding(self) -> str:
        for stream in self._streams:
            encoding = getattr(stream, "encoding", None)
            if encoding:
                return str(encoding)
        return "utf-8"

    def __getattr__(self, name: str) -> Any:
        if not self._streams:
            raise AttributeError(name)
        return getattr(self._streams[0], name)


@dataclass
class _SyncState:
    events_synced: int = 0
    logs_synced_bytes: int = 0
    last_sync_ts: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events_synced": int(self.events_synced),
            "logs_synced_bytes": int(self.logs_synced_bytes),
            "last_sync_ts": self.last_sync_ts,
        }

    @classmethod
    def from_path(cls, path: Path) -> "_SyncState":
        if not path.exists():
            return cls()
        try:
            payload = read_json(path, default={}, expect_type=dict)
            return cls(
                events_synced=int(payload.get("events_synced", 0)),
                logs_synced_bytes=int(payload.get("logs_synced_bytes", 0)),
                last_sync_ts=str(payload.get("last_sync_ts", "")),
            )
        except Exception:
            import logging
            logging.exception('Unhandled exception')
            raise
            return cls()


class ColabLiveTelemetry:
    """Write telemetry locally first, then synchronize to Drive best-effort."""

    def __init__(
        self,
        *,
        notebook_name: str,
        run_id: Optional[str] = None,
        drive_root: Optional[str | Path] = None,
        local_root: Optional[str | Path] = None,
        sync_interval_sec: float = 5.0,
        latest_status_min_interval_sec: float = 15.0,
    ) -> None:
        self.notebook_name = str(notebook_name)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.sync_interval_sec = float(max(0.5, sync_interval_sec))
        self.latest_status_min_interval_sec = float(max(0.0, latest_status_min_interval_sec))

        self.drive_root = Path(
            drive_root
            or os.environ.get("AADS_DRIVE_LOG_ROOT", "/content/drive/MyDrive/aads_ulora")
        )
        local_base = Path(local_root or "/content/aads_ulora/telemetry_spool")

        self.drive_run_dir = self.drive_root / "telemetry" / self.run_id
        self.local_run_dir = local_base / self.run_id
        self.artifacts_dir = self.drive_run_dir / "artifacts"
        self.local_artifacts_dir = self.local_run_dir / "artifacts"

        self.local_events_path = self.local_run_dir / "events.jsonl"
        self.drive_events_path = self.drive_run_dir / "events.jsonl"
        self.local_log_path = self.local_run_dir / "runtime.log"
        self.drive_log_path = self.drive_run_dir / "runtime.log"
        self.local_latest_path = self.local_run_dir / "latest_status.json"
        self.drive_latest_path = self.drive_run_dir / "latest_status.json"
        self.local_summary_path = self.local_run_dir / "summary.json"
        self.drive_summary_path = self.drive_run_dir / "summary.json"
        self.local_artifact_index_path = self.local_run_dir / "artifact_index.json"
        self.drive_artifact_index_path = self.drive_run_dir / "artifact_index.json"
        self.local_sync_state_path = self.local_run_dir / "sync_state.json"
        self._drive_mount_required = _requires_google_drive_mount(self.drive_root)

        self._sync_state = _SyncState.from_path(self.local_sync_state_path)
        self._artifact_index: List[Dict[str, Any]] = self._load_artifact_index()
        self._summary_metadata: Dict[str, Any] = {}
        self._last_sync_attempt = 0.0
        self._drive_available = False
        self._sequence = 0
        self._repo_output_dir: Optional[Path] = None
        self._repo_notebook_path: Optional[Path] = None
        self._repo_notebook_exporter: Optional[Callable[[Path], Optional[Path]]] = None
        self._last_latest_write_at = 0.0
        self._pending_latest_payload: Optional[Dict[str, Any]] = None
        self._pending_latest_drive_sync = False
        self._last_drive_issue: Optional[tuple[str, str]] = None
        self._storage_exhausted = False

        self.local_run_dir.mkdir(parents=True, exist_ok=True)
        self._local_artifact_store = ArtifactStore(self.local_artifacts_dir)
        self._drive_artifact_store: Optional[ArtifactStore] = None
        self._ensure_drive_surface()
        self.emit_event(
            "run_started",
            {
                "notebook": self.notebook_name,
                "run_id": self.run_id,
                "drive_run_dir": str(self.drive_run_dir),
                "local_run_dir": str(self.local_run_dir),
            },
            phase="bootstrap",
        )


    def _record_drive_unavailable(self, *, operation: str, error: str) -> None:
        normalized_error = str(error).strip() or "unknown_drive_error"
        issue = (str(operation), normalized_error)
        self._drive_available = False
        if self._last_drive_issue == issue:
            return
        self._last_drive_issue = issue
        logger.warning("Drive telemetry unavailable during %s: %s", operation, normalized_error)
        if "no space left on device" in normalized_error.lower():
            self._storage_exhausted = True
            logger.error("Telemetry disabled after storage exhaustion; training failure remains authoritative.")
            return
        self.emit_event(
            "drive_sync_unavailable",
            {"operation": str(operation), "error": normalized_error},
            phase="telemetry",
            level="warning",
            force_sync=False,
        )

    def _record_drive_available(self, *, operation: str) -> None:
        was_unavailable = self._last_drive_issue is not None
        self._drive_available = True
        if was_unavailable:
            self.emit_event(
                "drive_sync_restored",
                {"operation": str(operation)},
                phase="telemetry",
                level="info",
                force_sync=False,
            )
        self._last_drive_issue = None

    def _ensure_drive_surface(self) -> bool:
        if self._drive_mount_required and not _google_drive_is_mounted():
            self._drive_artifact_store = None
            self._record_drive_unavailable(
                operation="ensure_drive_surface",
                error="Google Drive mount is unavailable.",
            )
            return False
        try:
            self.drive_run_dir.mkdir(parents=True, exist_ok=True)
            if self._drive_artifact_store is None:
                self._drive_artifact_store = ArtifactStore(self.artifacts_dir)
            self._record_drive_available(operation="ensure_drive_surface")
            return True
        except Exception as exc:
            self._drive_artifact_store = None
            self._record_drive_unavailable(operation="ensure_drive_surface", error=f"{exc.__class__.__name__}: {exc}")
            return False

    def _current_artifacts_dir(self) -> Path:
        return self.artifacts_dir if self._ensure_drive_surface() else self.local_artifacts_dir

    def configure_repo_output_export(
        self,
        *,
        output_dir: Optional[str | Path] = None,
        notebook_filename: Optional[str] = None,
        notebook_path: Optional[str | Path] = None,
        export_notebook_fn: Callable[[Path], Optional[Path]],
    ) -> None:
        if notebook_path is not None:
            self._repo_notebook_path = Path(notebook_path)
            self._repo_output_dir = self._repo_notebook_path.parent
        else:
            if output_dir is None or notebook_filename is None:
                raise ValueError(
                    "configure_repo_output_export requires notebook_path or both output_dir and notebook_filename"
                )
            self._repo_output_dir = Path(output_dir)
            self._repo_notebook_path = self._repo_output_dir / str(notebook_filename)
        self._repo_notebook_exporter = export_notebook_fn
        self.emit_event(
            "repo_output_export_enabled",
            {
                "output_dir": str(self._repo_output_dir),
                "notebook_path": str(self._repo_notebook_path),
            },
            phase="repo_sync",
            force_sync=False,
        )

    def _export_repo_notebook(
        self,
        *,
        reason: str,
        cell_name: Optional[str] = None,
    ) -> Optional[Path]:
        if self._repo_notebook_exporter is None or self._repo_notebook_path is None:
            return None

        payload: Dict[str, Any] = {
            "reason": str(reason),
            "output_dir": str(self._repo_output_dir) if self._repo_output_dir is not None else "",
            "notebook_path": str(self._repo_notebook_path),
        }
        if cell_name:
            payload["cell_name"] = str(cell_name)

        try:
            saved_path = self._repo_notebook_exporter(self._repo_notebook_path)
            if saved_path is not None:
                payload["saved_path"] = str(saved_path)
                self.emit_event(
                    "repo_notebook_exported",
                    payload,
                    phase="repo_sync",
                    force_sync=False,
                )
                return saved_path

            payload["warning"] = "Notebook exporter returned no payload."
            self.emit_event(
                "repo_notebook_export_unavailable",
                payload,
                phase="repo_sync",
                level="warning",
                force_sync=False,
            )
            return None
        except Exception as exc:
            payload["error"] = f"{exc.__class__.__name__}: {exc}"
            self.emit_event(
                "repo_notebook_export_failed",
                payload,
                phase="repo_sync",
                level="error",
                force_sync=False,
            )
            return None

    def _load_artifact_index(self) -> List[Dict[str, Any]]:
        if not self.local_artifact_index_path.exists():
            return []
        try:
            payload = read_json(self.local_artifact_index_path, default=[], expect_type=list)
            if isinstance(payload, list):
                return [dict(item) for item in payload if isinstance(item, dict)]
        except Exception:
            import logging
            logging.exception('Unhandled exception')
            raise
            pass
        return []

    def _upsert_artifact_index_entry(self, payload: Dict[str, Any]) -> None:
        relative_path = str(payload.get("relative_path", "")).strip().replace("\\", "/")
        if not relative_path:
            return
        normalized = {**dict(payload), "relative_path": relative_path}
        for index in range(len(self._artifact_index) - 1, -1, -1):
            current = self._artifact_index[index]
            current_relative_path = str(current.get("relative_path", "")).strip().replace("\\", "/")
            if current_relative_path != relative_path:
                continue
            self._artifact_index[index] = {**current, **normalized}
            self._store_artifact_index()
            return
        self._artifact_index.append(normalized)
        self._store_artifact_index()

    def merge_artifact_catalog(self, entries: List[Dict[str, Any]]) -> None:
        for entry in list(entries or []):
            if not isinstance(entry, dict):
                continue
            self._upsert_artifact_index_entry({"ts": _utc_now_iso(), **dict(entry)})

    def merge_summary_metadata(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict) or not payload:
            return
        self._summary_metadata = deep_merge(self._summary_metadata, payload)

    def _store_sync_state(self) -> None:
        self._sync_state.last_sync_ts = _utc_now_iso()
        write_json(self.local_sync_state_path, self._sync_state.to_dict(), sort_keys=True)

    def _store_artifact_index(self) -> None:
        write_json(self.local_artifact_index_path, self._artifact_index, sort_keys=True)
        if self._ensure_drive_surface():
            try:
                write_json(self.drive_artifact_index_path, self._artifact_index, sort_keys=True)
                self._record_drive_available(operation="store_artifact_index")
            except Exception as exc:
                self._record_drive_unavailable(
                    operation="store_artifact_index",
                    error=f"{exc.__class__.__name__}: {exc}",
                )

    def _periodic_sync(self, force: bool = False) -> None:
        now = time.time()
        should_sync = force or (now - self._last_sync_attempt) >= self.sync_interval_sec
        if should_sync:
            self.sync_pending()
            self._last_sync_attempt = now

    def _latest_status_event_type(self, payload: Dict[str, Any]) -> str:
        status = payload.get("status", {})
        if not isinstance(status, dict):
            return ""
        return str(status.get("event_type", "")).strip().lower()

    def _should_write_latest_now(self, payload: Dict[str, Any], *, force: bool = False) -> bool:
        if force:
            return True
        event_type = self._latest_status_event_type(payload)
        if event_type != "batch_end":
            return True
        if self.latest_status_min_interval_sec <= 0.0:
            return True
        if self._last_latest_write_at <= 0.0:
            return True
        return (time.monotonic() - self._last_latest_write_at) >= self.latest_status_min_interval_sec

    def _write_latest_payload(self, payload: Dict[str, Any]) -> None:
        write_json(self.local_latest_path, payload, sort_keys=True)
        drive_synced = False
        if self._ensure_drive_surface():
            try:
                write_json(self.drive_latest_path, payload, sort_keys=True)
                self._record_drive_available(operation="write_latest_payload")
                drive_synced = True
            except Exception as exc:
                self._record_drive_unavailable(
                    operation="write_latest_payload",
                    error=f"{exc.__class__.__name__}: {exc}",
                )
        self._last_latest_write_at = time.monotonic()
        if drive_synced:
            self._pending_latest_payload = None
            self._pending_latest_drive_sync = False
        else:
            self._pending_latest_payload = dict(payload)
            self._pending_latest_drive_sync = True

    def _append_event_local(self, payload: Dict[str, Any]) -> None:
        _append_line(self.local_events_path, _safe_json(payload))

    def _append_event_drive(self, payload: Dict[str, Any]) -> None:
        _append_line(self.drive_events_path, _safe_json(payload))

    def _append_log_local(self, line: str) -> None:
        _append_line(self.local_log_path, line)

    def _append_log_drive(self, line: str) -> None:
        _append_line(self.drive_log_path, line)

    def emit_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        *,
        phase: str = "runtime",
        level: str = "info",
        force_sync: bool = True,
    ) -> Dict[str, Any]:
        self._sequence += 1
        record = {
            "ts": _utc_now_iso(),
            "seq": int(self._sequence),
            "run_id": self.run_id,
            "notebook": self.notebook_name,
            "phase": str(phase),
            "event_type": str(event_type),
            "level": str(level).lower(),
            "payload": dict(payload or {}),
        }
        if self._storage_exhausted:
            return record
        self._append_event_local(record)
        if force_sync:
            self._periodic_sync()
        return record

    def emit_log(self, message: str, *, phase: str = "runtime", level: str = "info") -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} [{level.upper()}] [{phase}] {message}"
        self._append_log_local(line)
        self.emit_event("log_line", {"message": message}, phase=phase, level=level, force_sync=False)
        self._periodic_sync()

    @contextmanager
    def capture_cell_output(self, cell_name: str, *, phase: str = "notebook_cell") -> Iterator[Path]:
        label = str(cell_name or "cell")
        capture_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        started_at = _utc_now_iso()
        start_clock = time.time()
        relative_path = Path("cell_outputs") / f"{capture_ts}_{_slugify_capture_name(label)}.log"
        relative_path_text = str(relative_path).replace("\\", "/")
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        failure_message = ""
        failure_traceback = ""

        self.emit_event(
            "cell_capture_started",
            {"cell_name": label, "relative_path": relative_path_text},
            phase=phase,
            force_sync=False,
        )
        sys.stdout = _TeeTextStream(original_stdout, stdout_buffer)
        sys.stderr = _TeeTextStream(original_stderr, stderr_buffer)
        try:
            yield self._current_artifacts_dir() / relative_path
        except Exception as exc:
            failure_message = f"{exc.__class__.__name__}: {exc}"
            failure_traceback = traceback.format_exc()
            self.emit_event(
                "cell_capture_failed",
                {
                    "cell_name": label,
                    "relative_path": relative_path_text,
                    "error": failure_message,
                },
                phase=phase,
                level="error",
                force_sync=False,
            )
            raise
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            duration_sec = round(time.time() - start_clock, 3)
            finished_at = _utc_now_iso()
            sections = [
                f"[CELL_CAPTURE] cell={label}",
                f"run_id={self.run_id}",
                f"started_at={started_at}",
                f"finished_at={finished_at}",
                f"duration_sec={duration_sec:.3f}",
                "",
                stdout_buffer.getvalue(),
            ]
            body = "\n".join(section for section in sections if section is not None)
            stderr_text = stderr_buffer.getvalue()
            if stderr_text:
                if not body.endswith("\n"):
                    body += "\n"
                body += "\n[STDERR]\n"
                body += stderr_text
            if failure_message:
                if not body.endswith("\n"):
                    body += "\n"
                body += f"\n[EXCEPTION] {failure_message}\n"
            if failure_traceback:
                if not body.endswith("\n"):
                    body += "\n"
                body += "\n[TRACEBACK]\n"
                body += failure_traceback
            artifact_path = self.write_text_artifact(relative_path_text, body)
            self.emit_event(
                "cell_capture_finished",
                {
                    "cell_name": label,
                    "relative_path": relative_path_text,
                    "artifact_path": str(artifact_path),
                    "duration_sec": duration_sec,
                },
                phase=phase,
                force_sync=False,
            )
            self._periodic_sync(force=True)

    def update_latest(self, status_payload: Dict[str, Any], *, force: bool = False) -> None:
        if self._storage_exhausted:
            return
        payload = {
            "ts": _utc_now_iso(),
            "run_id": self.run_id,
            "notebook": self.notebook_name,
            "status": dict(status_payload or {}),
        }
        self._pending_latest_payload = payload
        self._pending_latest_drive_sync = False
        if self._should_write_latest_now(payload, force=force):
            self._write_latest_payload(payload)
        self._periodic_sync()

    def write_json_artifact(self, relative_path: str, payload: Dict[str, Any]) -> Path:
        body = _safe_json(payload)
        return self.write_text_artifact(relative_path, body)

    def write_text_artifact(self, relative_path: str, text: str) -> Path:
        rel = Path(relative_path)
        local_path = self._local_artifact_store.write_text(rel, text)
        drive_path = self.artifacts_dir / rel
        if self._ensure_drive_surface() and self._drive_artifact_store is not None:
            try:
                drive_path = self._drive_artifact_store.write_text(rel, text)
                self._record_drive_available(operation="write_text_artifact")
            except Exception as exc:
                self._record_drive_unavailable(
                    operation="write_text_artifact",
                    error=f"{exc.__class__.__name__}: {exc}",
                )
        self._upsert_artifact_index_entry(
            {
                "ts": _utc_now_iso(),
                "relative_path": str(rel).replace("\\", "/"),
                "type": "text",
            }
        )
        self.emit_event("artifact_written", {"relative_path": str(rel)}, phase="artifact", force_sync=False)
        return drive_path if drive_path.exists() else local_path

    def write_binary_artifact(self, relative_path: str, content: bytes) -> Path:
        rel = Path(relative_path)
        local_path = self._local_artifact_store.write_bytes(rel, content)
        drive_path = self.artifacts_dir / rel
        if self._ensure_drive_surface() and self._drive_artifact_store is not None:
            try:
                drive_path = self._drive_artifact_store.write_bytes(rel, content)
                self._record_drive_available(operation="write_binary_artifact")
            except Exception as exc:
                self._record_drive_unavailable(
                    operation="write_binary_artifact",
                    error=f"{exc.__class__.__name__}: {exc}",
                )
        self._upsert_artifact_index_entry(
            {
                "ts": _utc_now_iso(),
                "relative_path": str(rel).replace("\\", "/"),
                "type": "binary",
            }
        )
        self.emit_event("artifact_written", {"relative_path": str(rel)}, phase="artifact", force_sync=False)
        return drive_path if drive_path.exists() else local_path

    def copy_artifact_file(self, source_path: str | Path, relative_path: str) -> Path:
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Artifact source does not exist: {src}")
        rel = Path(relative_path)
        local_path = self._local_artifact_store.copy_file(src, rel)
        drive_path = self.artifacts_dir / rel
        if self._ensure_drive_surface() and self._drive_artifact_store is not None:
            try:
                drive_path = self._drive_artifact_store.copy_file(src, rel)
                self._record_drive_available(operation="copy_artifact_file")
            except Exception as exc:
                self._record_drive_unavailable(operation="copy_artifact_file", error=f"{exc.__class__.__name__}: {exc}")
        self._upsert_artifact_index_entry(
            {
                "ts": _utc_now_iso(),
                "relative_path": str(rel).replace("\\", "/"),
                "type": "file_copy",
            }
        )
        self.emit_event("artifact_copied", {"relative_path": str(rel)}, phase="artifact", force_sync=False)
        return drive_path if drive_path.exists() else local_path

    def sync_pending(self) -> None:
        if self._storage_exhausted:
            return
        pending = self._pending_latest_payload
        if pending is not None and (
            self._pending_latest_drive_sync
            or self._should_write_latest_now(pending, force=False)
        ):
            self._write_latest_payload(pending)
        self._sync_events()
        self._sync_logs()
        self._store_artifact_index()
        self._store_sync_state()

    def _sync_events(self) -> None:
        if not self.local_events_path.exists():
            return
        lines = self.local_events_path.read_text(encoding="utf-8").splitlines()
        start = max(0, int(self._sync_state.events_synced))
        if start >= len(lines):
            return
        unsynced = lines[start:]
        if self._ensure_drive_surface():
            try:
                ensure_parent(self.drive_events_path)
                with self.drive_events_path.open("a", encoding="utf-8") as handle:
                    for line in unsynced:
                        handle.write(line + "\n")
                self._sync_state.events_synced = len(lines)
                self._record_drive_available(operation="sync_events")
            except Exception as exc:
                self._record_drive_unavailable(operation="sync_events", error=f"{exc.__class__.__name__}: {exc}")

    def _sync_logs(self) -> None:
        if not self.local_log_path.exists():
            return
        local_bytes = self.local_log_path.read_bytes()
        offset = max(0, int(self._sync_state.logs_synced_bytes))
        if offset >= len(local_bytes):
            return
        chunk = local_bytes[offset:]
        if self._ensure_drive_surface():
            try:
                ensure_parent(self.drive_log_path)
                with self.drive_log_path.open("ab") as handle:
                    handle.write(chunk)
                self._sync_state.logs_synced_bytes = len(local_bytes)
                self._record_drive_available(operation="sync_logs")
            except Exception as exc:
                self._record_drive_unavailable(operation="sync_logs", error=f"{exc.__class__.__name__}: {exc}")

    def close(self, final_payload: Optional[Dict[str, Any]] = None) -> None:
        pending = self._pending_latest_payload
        if pending is not None:
            self._write_latest_payload(pending)
        self._export_repo_notebook(reason="run_close")
        summary = {
            "ts": _utc_now_iso(),
            "run_id": self.run_id,
            "notebook": self.notebook_name,
            "final_payload": dict(final_payload or {}),
            "artifact_count": len(self._artifact_index),
            "drive_available": bool(self._drive_available),
            "metadata": dict(self._summary_metadata),
        }
        write_json(self.local_summary_path, summary, sort_keys=True)
        if self._ensure_drive_surface():
            try:
                write_json(self.drive_summary_path, summary, sort_keys=True)
                self._record_drive_available(operation="write_summary")
            except Exception as exc:
                self._record_drive_unavailable(operation="write_summary", error=f"{exc.__class__.__name__}: {exc}")
        self.emit_event("run_finished", summary, phase="final", force_sync=False)
        self.sync_pending()
