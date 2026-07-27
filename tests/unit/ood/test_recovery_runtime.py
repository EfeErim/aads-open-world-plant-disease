from __future__ import annotations

from pathlib import Path

import pytest

from src.ood.recovery_runtime import RecoveryDiskBudgetError, reclaim_failed_run_payloads, require_recovery_disk_budget


def test_disk_budget_fails_before_training_when_free_space_is_below_floor(tmp_path: Path) -> None:
    with pytest.raises(RecoveryDiskBudgetError, match="below the safe floor"):
        require_recovery_disk_budget(tmp_path, min_free_gib=10**9)


def test_failed_pushed_run_reclaims_only_explicitly_skipped_payloads(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "crop" / "part" / "run-1"
    tracked = run_dir / "summary.json"
    skipped = run_dir / "checkpoints" / "model.safetensors"
    tracked.parent.mkdir(parents=True)
    skipped.parent.mkdir(parents=True)
    tracked.write_text("{}", encoding="utf-8")
    skipped.write_bytes(b"large-payload")

    report = reclaim_failed_run_payloads(
        repo_root=tmp_path,
        run_dir=run_dir,
        git_push_report={
            "pushed": True,
            "run_dir": str(run_dir),
            "skipped_files": [skipped.relative_to(tmp_path).as_posix()],
        },
        behavioral_passed=False,
        enabled=True,
    )

    assert report["reclaimed"] is True
    assert report["removed_file_count"] == 1
    assert report["reclaimed_bytes"] == len(b"large-payload")
    assert tracked.is_file()
    assert not skipped.exists()


def test_reclaimer_preserves_passing_or_unpublished_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "crop" / "part" / "run-1"
    payload = run_dir / "adapter.safetensors"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"adapter")
    push_report = {
        "pushed": True,
        "run_dir": str(run_dir),
        "skipped_files": [payload.relative_to(tmp_path).as_posix()],
    }

    passing = reclaim_failed_run_payloads(
        repo_root=tmp_path,
        run_dir=run_dir,
        git_push_report=push_report,
        behavioral_passed=True,
        enabled=True,
    )
    unpublished = reclaim_failed_run_payloads(
        repo_root=tmp_path,
        run_dir=run_dir,
        git_push_report={**push_report, "pushed": False},
        behavioral_passed=False,
        enabled=True,
    )

    assert passing["reason"] == "passing_run_preserved"
    assert unpublished["reason"] == "eligible_run_evidence_not_pushed"
    assert payload.is_file()


def test_reclaimer_rejects_a_run_outside_repo_runs(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    (tmp_path / "runs").mkdir()
    outside.mkdir()

    with pytest.raises(ValueError, match="must stay under"):
        reclaim_failed_run_payloads(
            repo_root=tmp_path,
            run_dir=outside,
            git_push_report={},
            behavioral_passed=False,
            enabled=True,
        )
