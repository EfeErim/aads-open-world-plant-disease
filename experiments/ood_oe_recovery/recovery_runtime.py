"""Resource guards for long-running adapter recovery campaigns."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

GIB = 1024**3


class RecoveryDiskBudgetError(RuntimeError):
    """Raised before a recovery experiment can exhaust its runtime disk."""


def recovery_disk_status(path: Path, *, min_free_gib: float) -> dict[str, Any]:
    """Return a deterministic free-space verdict for a recovery workspace."""
    resolved = path.expanduser().resolve(strict=True)
    usage = shutil.disk_usage(resolved)
    required_bytes = max(0, int(float(min_free_gib) * GIB))
    return {
        "path": str(resolved),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "min_free_bytes": required_bytes,
        "min_free_gib": float(min_free_gib),
        "ok": int(usage.free) >= required_bytes,
    }


def require_recovery_disk_budget(path: Path, *, min_free_gib: float) -> dict[str, Any]:
    """Fail before training when the configured disk headroom is unavailable."""
    status = recovery_disk_status(path, min_free_gib=min_free_gib)
    if status["ok"]:
        return status
    free_gib = int(status["free_bytes"]) / GIB
    raise RecoveryDiskBudgetError(
        f"Recovery disk budget is below the safe floor: free={free_gib:.2f}GiB "
        f"required={float(min_free_gib):.2f}GiB path={status['path']}"
    )


def reclaim_failed_run_payloads(
    *,
    repo_root: Path,
    run_dir: Path,
    git_push_report: Mapping[str, Any],
    behavioral_passed: bool,
    enabled: bool,
) -> dict[str, Any]:
    """Delete only excluded payload files after a failed run's tracked evidence was pushed.

    The helper deliberately preserves every tracked report and every passing run. It only
    removes exact files listed by the canonical Git push report as skipped, and only after
    that report proves the eligible run evidence reached the remote.
    """
    repo = repo_root.expanduser().resolve(strict=True)
    resolved_run = run_dir.expanduser().resolve(strict=True)
    runs_root = (repo / "runs").resolve(strict=True)
    if runs_root not in resolved_run.parents:
        raise ValueError(f"Recovery run directory must stay under {runs_root}: {resolved_run}")
    result: dict[str, Any] = {
        "enabled": bool(enabled),
        "reclaimed": False,
        "reason": "",
        "run_dir": str(resolved_run),
        "removed_file_count": 0,
        "reclaimed_bytes": 0,
    }
    if not enabled:
        result["reason"] = "disabled"
        return result
    if behavioral_passed:
        result["reason"] = "passing_run_preserved"
        return result
    if not bool(git_push_report.get("pushed")):
        result["reason"] = "eligible_run_evidence_not_pushed"
        return result
    reported_run = Path(str(git_push_report.get("run_dir") or "")).expanduser().resolve()
    if reported_run != resolved_run:
        raise ValueError(f"Git push report run mismatch: expected={resolved_run} actual={reported_run}")

    removed = 0
    reclaimed_bytes = 0
    for raw_path in sorted({str(value) for value in git_push_report.get("skipped_files", []) if str(value)}):
        candidate = (repo / raw_path).resolve()
        if resolved_run not in candidate.parents or not candidate.is_file():
            continue
        reclaimed_bytes += candidate.stat().st_size
        candidate.unlink()
        removed += 1
    result.update(
        {
            "reclaimed": removed > 0,
            "reason": "excluded_failed_run_payloads_removed" if removed else "no_excluded_payloads_present",
            "removed_file_count": removed,
            "reclaimed_bytes": reclaimed_bytes,
        }
    )
    return result
