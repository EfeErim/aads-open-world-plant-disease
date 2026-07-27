#!/usr/bin/env python3
"""Checkpoint lifecycle helpers for Colab notebook training."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.shared.contracts import CheckpointRecord
from src.shared.json_utils import read_json, write_json

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class TrainingCheckpointManager:
    """Manage rolling training checkpoints with latest/best manifests."""

    def __init__(self, root_dir: str | Path, *, retention: int = 3) -> None:
        self.root_dir = Path(root_dir)
        self.retention = int(max(1, retention))
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.latest_manifest_path = self.root_dir / "latest_checkpoint.json"
        self.best_manifest_path = self.root_dir / "best_checkpoint.json"
        self.index_path = self.root_dir / "checkpoint_index.json"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return read_json(path, default=default)
        except Exception as exc:
            logger.warning("Failed to read checkpoint metadata %s: %s", path, exc)
            return default

    def _write_json(self, path: Path, payload: Any) -> None:
        write_json(path, payload, ensure_ascii=False)

    def _load_index(self) -> List[Dict[str, Any]]:
        payload = self._read_json(self.index_path, [])
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
        return []

    def _store_index(self, rows: List[Dict[str, Any]]) -> None:
        ordered = sorted(rows, key=lambda row: str(row.get("created_at", "")))
        self._write_json(self.index_path, ordered)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        rows = self._load_index()
        rows.sort(key=lambda row: str(row.get("created_at", "")), reverse=True)
        return rows

    def get_latest(self) -> Optional[Dict[str, Any]]:
        payload = self._read_json(self.latest_manifest_path, None)
        return dict(payload) if isinstance(payload, dict) else None

    def get_best(self) -> Optional[Dict[str, Any]]:
        payload = self._read_json(self.best_manifest_path, None)
        return dict(payload) if isinstance(payload, dict) else None

    def save_checkpoint(
        self,
        *,
        adapter: Any,
        session: Any,
        reason: str,
        run_id: str,
        mark_best: bool = False,
        val_loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        session_state = session.snapshot_state()
        progress_state = dict(session_state.get("progress_state", {}))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        epoch = int(progress_state.get("epoch", 0))
        global_step = int(progress_state.get("global_step", 0))
        name = f"ckpt_e{epoch}_s{global_step}_{timestamp}"
        checkpoint_root = self.checkpoints_dir / name
        checkpoint_root.mkdir(parents=True, exist_ok=True)

        checkpoint_dir = adapter.save_training_checkpoint(
            str(checkpoint_root),
            session_state=session_state,
            run_id=run_id,
        )
        checkpoint_dir = Path(checkpoint_dir)

        record = CheckpointRecord(
            name=name,
            path=checkpoint_dir,
            created_at=_utc_now_iso(),
            global_step=global_step,
            epoch=epoch,
            reason=str(reason),
            is_best=bool(mark_best),
            val_loss=(float(val_loss) if val_loss is not None else None),
        )
        payload = record.to_dict()

        index_rows = self._load_index()
        index_rows.append(payload)
        self._store_index(index_rows)

        self._write_json(self.latest_manifest_path, payload)
        if mark_best:
            self._write_json(self.best_manifest_path, payload)
        self._prune_old()
        return payload

    def _prune_old(self) -> None:
        rows = self._load_index()
        if not rows:
            return
        rows.sort(key=lambda row: str(row.get("created_at", "")), reverse=True)
        best_path = None
        if self.best_manifest_path.exists():
            try:
                best = read_json(self.best_manifest_path, default=None)
            except Exception as exc:
                logger.warning(
                    "Failed to read best checkpoint manifest %s; skipping checkpoint pruning to avoid "
                    "losing the best checkpoint: %s",
                    self.best_manifest_path,
                    exc,
                )
                return
            if isinstance(best, dict) and isinstance(best.get("path"), str):
                best_path = Path(best["path"]).resolve()

        kept: List[Dict[str, Any]] = []
        removed: List[Dict[str, Any]] = []
        rolling_kept = 0
        for row in rows:
            row_path = Path(str(row.get("path", "")))
            resolved = row_path.resolve() if row_path.exists() else row_path
            is_best = best_path is not None and resolved == best_path
            if is_best or rolling_kept < self.retention:
                kept.append(row)
                if not is_best:
                    rolling_kept += 1
            else:
                removed.append(row)

        for row in removed:
            row_path = Path(str(row.get("path", "")))
            try:
                if row_path.exists():
                    shutil.rmtree(row_path, ignore_errors=True)
            except Exception:
                import logging
                logging.exception('Unhandled exception')
                raise
                logger.warning("Failed to prune checkpoint path: %s", row_path)
        self._store_index(kept)

    def load_checkpoint(self, *, adapter: Any, checkpoint_path: str | Path) -> Dict[str, Any]:
        loaded = adapter.load_training_checkpoint(str(checkpoint_path))
        return dict(loaded)

    def load_latest_checkpoint(self, *, adapter: Any) -> Optional[Dict[str, Any]]:
        latest = self.get_latest()
        if not latest:
            return None
        checkpoint_path = latest.get("path")
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            return None
        return self.load_checkpoint(adapter=adapter, checkpoint_path=checkpoint_path)
