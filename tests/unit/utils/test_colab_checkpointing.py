from pathlib import Path

from scripts.colab_checkpointing import TrainingCheckpointManager


class _FakeAdapter:
    def save_training_checkpoint(self, checkpoint_dir: str, *, session_state, run_id: str) -> Path:
        root = Path(checkpoint_dir) / "continual_sd_lora_adapter"
        root.mkdir(parents=True, exist_ok=True)
        (root / "checkpoint.pt").write_text("weights", encoding="utf-8")
        return root


class _FakeSession:
    def snapshot_state(self):
        return {
            "progress_state": {
                "epoch": 2,
                "global_step": 5,
            }
        }


def test_checkpoint_manager_marks_best_without_duplicate_best_tree(tmp_path: Path):
    manager = TrainingCheckpointManager(tmp_path / "telemetry" / "run_1", retention=2)

    record = manager.save_checkpoint(
        adapter=_FakeAdapter(),
        session=_FakeSession(),
        reason="epoch_end",
        run_id="run_1",
        mark_best=True,
        val_loss=0.2,
    )

    best = manager.get_best()

    assert isinstance(best, dict)
    assert best["path"] == record["path"]
    assert Path(record["path"]).exists()
    assert not (manager.checkpoints_dir / "best").exists()


def test_checkpoint_manager_uses_unique_names_for_same_step_in_same_second(tmp_path: Path, monkeypatch):
    class _FakeStamp:
        def __init__(self, index: int):
            self.index = index

        def strftime(self, fmt: str) -> str:
            if fmt == "%Y%m%d_%H%M%S_%f":
                return f"20260101_010203_{self.index:06d}"
            raise AssertionError(f"Unexpected format: {fmt}")

        def isoformat(self) -> str:
            return f"2026-01-01T01:02:03.{self.index:06d}+00:00"

    class _FakeDateTime:
        calls = 0

        @classmethod
        def now(cls, tz=None):
            cls.calls += 1
            return _FakeStamp(cls.calls)

    monkeypatch.setattr("scripts.colab_checkpointing.datetime", _FakeDateTime)

    manager = TrainingCheckpointManager(tmp_path / "telemetry" / "run_2", retention=2)
    first = manager.save_checkpoint(
        adapter=_FakeAdapter(),
        session=_FakeSession(),
        reason="batch_interval",
        run_id="run_2",
    )
    second = manager.save_checkpoint(
        adapter=_FakeAdapter(),
        session=_FakeSession(),
        reason="epoch_end",
        run_id="run_2",
    )

    assert first["name"] != second["name"]
    assert first["path"] != second["path"]
    assert Path(first["path"]).exists()
    assert Path(second["path"]).exists()


def test_checkpoint_manager_skips_prune_when_best_manifest_is_corrupt(tmp_path: Path):
    manager = TrainingCheckpointManager(tmp_path / "telemetry" / "run_3", retention=1)

    first = manager.save_checkpoint(
        adapter=_FakeAdapter(),
        session=_FakeSession(),
        reason="best",
        run_id="run_3",
        mark_best=True,
    )
    manager.save_checkpoint(
        adapter=_FakeAdapter(),
        session=_FakeSession(),
        reason="latest",
        run_id="run_3",
    )

    manager.best_manifest_path.write_text("{not-json", encoding="utf-8")

    third = manager.save_checkpoint(
        adapter=_FakeAdapter(),
        session=_FakeSession(),
        reason="after_corruption",
        run_id="run_3",
    )

    rows = manager.list_checkpoints()

    assert Path(first["path"]).exists()
    assert Path(third["path"]).exists()
    assert len(rows) == 3
