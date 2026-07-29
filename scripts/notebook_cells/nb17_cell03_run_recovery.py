# Auto-extracted from colab_notebooks/17_adapter_ood_oe_recovery.ipynb cell 3.
"""Run the complete gate-aware adapter OOD/OE recovery campaign."""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.colab_notebook_helpers import maybe_auto_disconnect_colab_runtime
from scripts.notebook_helpers.adapter_recommendations import get_adapter_recs
from scripts.notebook_helpers.cell_script_runner import run_cell_script
from src.notebook.repo_bootstrap import push_repo_paths_to_github
from src.ood import notebook_campaign as _ood_notebook_campaign
from src.ood.behavioral_acceptance import behavioral_acceptance_pass, behavioral_dev_report_pass
from src.ood.notebook_campaign import experiment_gate, read_behavioral_acceptance, read_behavioral_dev_report
from src.ood.recovery_runtime import reclaim_failed_run_payloads, require_recovery_disk_budget


def _run_checked(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    print("[COMMAND]", " ".join(command))
    return subprocess.run(command, cwd=str(cwd), check=True, text=True)


def _low_resource_overrides() -> dict:
    if not bool(globals().get("RECOVERY_LOW_RESOURCE_MODE", True)):
        return {}
    return {
        "BATCH_SIZE": int(globals().get("RECOVERY_BATCH_SIZE", 16)),
        "GRAD_ACCUM_STEPS": int(globals().get("RECOVERY_GRAD_ACCUM_STEPS", 2)),
        "NUM_WORKERS": int(globals().get("RECOVERY_NUM_WORKERS", 2)),
        "PREFETCH": int(globals().get("RECOVERY_PREFETCH", 2)),
        "PIN_MEMORY": False,
        "USE_CACHE": False,
        "CACHE_TRAIN_SPLIT": False,
        "CHECKPOINT_EVERY_N_STEPS": 0,
    }


def _cleanup_runtime_state(shared_globals: dict, *, label: str) -> None:
    state = shared_globals.get("STATE")
    if isinstance(state, dict):
        for key in (
            "adapter",
            "loaders",
            "history",
            "calibration",
            "evaluation_artifacts",
            "behavioral_dev_report",
            "ood_benchmark",
            "optimization_campaign",
            "recommendation_report",
        ):
            if key in state:
                state[key] = None
    for name in (
        "adapter",
        "loaders",
        "trainer",
        "test_loader",
        "val_loader",
        "ood_loader",
        "results",
        "benchmark_summary",
        "selected_artifacts",
        "evaluation",
    ):
        shared_globals.pop(name, None)
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print(
                "[RECOVERY] cuda_cleanup "
                f"allocated={torch.cuda.memory_allocated() / (1024 ** 3):.2f}GiB "
                f"reserved={torch.cuda.memory_reserved() / (1024 ** 3):.2f}GiB"
            )
    except Exception as exc:
        print(f"[RECOVERY] cuda cleanup skipped: {exc}")
    gc.collect()
    print(f"[RECOVERY] cleanup complete after {label}")


def _require_disk_headroom(root: Path) -> dict:
    status = require_recovery_disk_budget(
        root,
        min_free_gib=float(globals().get("RECOVERY_MIN_FREE_DISK_GIB", 12.0)),
    )
    print(
        "[RECOVERY] disk_budget "
        f"free={int(status['free_bytes']) / (1024 ** 3):.2f}GiB "
        f"required={float(status['min_free_gib']):.2f}GiB"
    )
    return status


def _reclaim_failed_experiment_payloads(shared_globals: dict, *, behavioral_passed: bool) -> dict:
    state = shared_globals.get("STATE")
    git_push_report = dict(state.get("git_push_report") or {}) if isinstance(state, dict) else {}
    run_dir_text = str(shared_globals.get("REPO_RUN_DIR") or "").strip()
    if not run_dir_text:
        return {"reclaimed": False, "reason": "run_dir_missing"}
    report = reclaim_failed_run_payloads(
        repo_root=Path(shared_globals["ROOT"]),
        run_dir=Path(run_dir_text),
        git_push_report=git_push_report,
        behavioral_passed=behavioral_passed,
        enabled=bool(shared_globals.get("RECOVERY_RECLAIM_FAILED_RUN_PAYLOADS", True)),
    )
    print(
        "[RECOVERY] disk_reclaim "
        f"reason={report['reason']} files={report['removed_file_count']} "
        f"bytes={report['reclaimed_bytes']}"
    )
    return report


def _run_training_experiment(
    *,
    target: str,
    experiment: dict,
    shared_globals: dict,
) -> tuple[Path, dict]:
    overrides = dict(experiment.get("resolved_config") or experiment.get("manual_param_overrides") or {})
    overrides.update(_low_resource_overrides())
    overrides.update(
        {
            "AUTO_DISCONNECT_RUNTIME": False,
            "AUTO_PUSH_TO_GITHUB": bool(shared_globals["RECOVERY_AUTO_PUSH_TO_GITHUB"]),
            "ENABLE_BAYESIAN_OPTIMIZATION": False,
            "RESUME_MODE": "fresh",
        }
    )
    shared_globals["ADAPTER_KEY"] = target
    shared_globals["RUNTIME_DATASET_ROOT"] = RUNTIME_DATASET_ROOT
    shared_globals["DATASET_SOURCE_KIND"] = DATASET_SOURCE_KIND
    shared_globals["DATASET_RELEASE_MANIFEST_PATH"] = DATASET_RELEASE_MANIFEST_PATH
    shared_globals["ALLOW_LOCAL_LEGACY_DATASET"] = ALLOW_LOCAL_LEGACY_DATASET
    shared_globals["DATASET_LEGACY_COMPATIBILITY_REASON"] = DATASET_LEGACY_COMPATIBILITY_REASON
    shared_globals["RECOVERY_SELECTION_ONLY"] = True
    shared_globals["MANUAL_PARAM_OVERRIDES"] = overrides
    shared_globals["DEFAULT_RUNTIME_PARAMS"] = {
        "AUTO_DISCONNECT_RUNTIME": False,
        "AUTO_PUSH_TO_GITHUB": bool(shared_globals["RECOVERY_AUTO_PUSH_TO_GITHUB"]),
        "ENABLE_BAYESIAN_OPTIMIZATION": False,
    }
    print(
        f"\n[RECOVERY] target={target} stage={experiment['stage']} "
        f"experiment={experiment['experiment_id']} overrides={json.dumps(overrides, sort_keys=True)}"
    )
    for cell_script in (
        "nb2_cell03_runtime_setup.py",
        "nb2_cell04_parameter_resolution.py",
        "nb2_cell05_access_check.py",
        "nb2_cell06_dataset_validation.py",
        "nb2_cell07_engine_init.py",
        "nb2_cell08_ood_config_verify.py",
        "nb2_cell09_training.py",
        "nb2_cell10_ood_calibration.py",
        "nb2_cell11_adapter_save.py",
        "nb2_cell12_final_evaluation.py",
    ):
        run_cell_script(cell_script, shared_globals)
    run_dir = Path(shared_globals["REPO_RUN_DIR"])
    dev_report_path = (
        run_dir
        / "outputs"
        / "colab_notebook_training"
        / "artifacts"
        / "adapter_behavioral_dev_report.json"
    )
    dev_report = read_behavioral_dev_report(dev_report_path)
    if not dev_report:
        raise FileNotFoundError(f"Training completed without behavioral dev report: {dev_report_path}")
    return dev_report_path, dev_report


def _run_locked_final_evaluation(shared_globals: dict) -> tuple[Path, dict]:
    shared_globals["RECOVERY_SELECTION_ONLY"] = False
    run_cell_script("nb2_cell12_final_evaluation.py", shared_globals)
    run_dir = Path(shared_globals["REPO_RUN_DIR"])
    acceptance_path = (
        run_dir
        / "outputs"
        / "colab_notebook_training"
        / "artifacts"
        / "adapter_behavioral_acceptance.json"
    )
    acceptance = read_behavioral_acceptance(acceptance_path)
    if not acceptance:
        raise FileNotFoundError(f"Frozen winner produced no behavioral acceptance artifact: {acceptance_path}")
    return acceptance_path, acceptance


def _publish_report(root: Path, report_path: Path, *extra_paths: Path) -> dict:
    if not bool(globals().get("RECOVERY_AUTO_PUSH_TO_GITHUB")):
        return {"publish_attempted": False, "publish_ok": False, "publish_status": "disabled"}
    relative_paths = [report_path.relative_to(root).as_posix()]
    relative_paths.extend(path.relative_to(root).as_posix() for path in extra_paths)
    push_report = push_repo_paths_to_github(
        root,
        relative_paths,
        commit_message=f"Add adapter OOD/OE recovery report {report_path.parent.name}",
    )
    pushed = bool(push_report.get("pushed"))
    unchanged = not pushed and not list(push_report.get("staged_files") or [])
    return {
        "publish_attempted": True,
        "publish_ok": pushed or unchanged,
        "publish_status": "pushed" if pushed else "unchanged",
    }


def _build_recovery_report(
    *,
    target_results: dict,
    preflight_ok: bool,
    report_path: Path,
    completed_experiment_count: int,
    max_completed_experiments: int,
    max_targets: int,
    phase: str,
) -> dict:
    report = _ood_notebook_campaign.build_notebook_completion_report(
        target_results=target_results,
        preflight_ok=preflight_ok,
    )
    report["preflight_error"] = RECOVERY_PREFLIGHT_ERROR
    report["preflight_blocked_targets"] = sorted(globals().get("RECOVERY_PREFLIGHT_BLOCKED_TARGETS", set()))
    report["campaign_path"] = str(CAMPAIGN_PATH)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["report_path"] = str(report_path)
    report["phase"] = phase
    report["resource_policy"] = {
        "low_resource_mode": bool(globals().get("RECOVERY_LOW_RESOURCE_MODE", True)),
        "max_completed_experiments": max_completed_experiments,
        "max_targets": max_targets,
        "completed_experiment_count": completed_experiment_count,
    }
    return report


def _write_and_publish_recovery_report(
    *,
    report_path: Path,
    phase: str,
    completed_experiment_count: int,
    max_completed_experiments: int,
    max_targets: int,
    ledger_path: Path | None = None,
) -> dict:
    report = _build_recovery_report(
        target_results=RECOVERY_RESULTS,
        preflight_ok=RECOVERY_PREFLIGHT_OK,
        report_path=report_path,
        completed_experiment_count=completed_experiment_count,
        max_completed_experiments=max_completed_experiments,
        max_targets=max_targets,
        phase=phase,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[RECOVERY] wrote {phase} summary: {report_path.relative_to(ROOT)} "
        f"passed={report['passed_target_count']}/{report['required_target_count']}"
    )
    try:
        extra_paths = (ledger_path,) if ledger_path is not None and ledger_path.is_file() else ()
        report.update(_publish_report(ROOT, report_path, *extra_paths))
    except Exception as exc:
        print(f"[GIT] Recovery {phase} report auto-push failed: {exc}")
        report["publish_attempted"] = bool(globals().get("RECOVERY_AUTO_PUSH_TO_GITHUB"))
        report["publish_ok"] = False
        report["publish_error"] = str(exc)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return report


ROOT = Path(globals().get("ROOT") or Path.cwd()).resolve()
CAMPAIGN_PATH = ROOT / str(
    globals().get(
        "RECOVERY_CAMPAIGN_PATH",
        "docs/architecture/adapter_ood_oe_recovery_campaign.json",
    )
)
campaign = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
if (
    not isinstance(campaign, dict)
    or campaign.get("schema_version") != "v4_bounded_adapter_recovery_campaign"
    or int(campaign.get("target_count", 0)) != 8
    or int(campaign.get("max_attempts_per_target", 0)) != 4
):
    raise RuntimeError(f"Invalid eight-target recovery campaign: {CAMPAIGN_PATH}")

release_targets = [str(entry["target"]) for entry in campaign["targets"]]
target_allowlist = {
    str(target)
    for target in list(globals().get("RECOVERY_TARGETS") or [])
    if str(target).strip()
}
unknown_selected_targets = sorted(target_allowlist.difference(release_targets))
if unknown_selected_targets:
    raise RuntimeError(f"Unknown RECOVERY_TARGETS: {', '.join(unknown_selected_targets)}")
selected_targets = [target for target in release_targets if not target_allowlist or target in target_allowlist]
if bool(globals().get("RECOVERY_USE_GITHUB_DATASET_RELEASE", False)):
    from src.data.dataset_release_runtime import fetch_materialize_dataset_release

    DATASET_RELEASE_REPORT = fetch_materialize_dataset_release(
        root=ROOT,
        repository=str(globals().get("DATASET_RELEASE_REPOSITORY", "EfeErim/aads-open-world-plant-disease")),
        release_tag=str(globals().get("DATASET_RELEASE_TAG", "aads-dataset-v1.0.0")),
        targets=selected_targets,
        cache_root=str(globals().get("DATASET_RELEASE_CACHE_ROOT", ".runtime_tmp/dataset_release_cache")),
    )
    RUNTIME_DATASET_ROOT = Path(DATASET_RELEASE_REPORT["runtime_dataset_root"])
    DATASET_RELEASE_MANIFEST_PATH = str(DATASET_RELEASE_REPORT["manifest_path"])
    DATASET_SOURCE_KIND = "github_release"
    EVIDENCE_MANIFEST_BASE_DIR = RUNTIME_DATASET_ROOT
    ALLOW_LOCAL_LEGACY_DATASET = False
    DATASET_LEGACY_COMPATIBILITY_REASON = ""
else:
    DATASET_RELEASE_REPORT = {}
    RUNTIME_DATASET_ROOT = Path(
        str(globals().get("RECOVERY_LOCAL_DATASET_ROOT", "data/prepared_runtime_datasets"))
    )
    DATASET_RELEASE_MANIFEST_PATH = ""
    DATASET_SOURCE_KIND = "local_legacy"
    EVIDENCE_MANIFEST_BASE_DIR = ROOT
    ALLOW_LOCAL_LEGACY_DATASET = True
    DATASET_LEGACY_COMPATIBILITY_REASON = "notebook17_active_recovery_campaign_until_phase9_parity"
EVIDENCE_MANIFEST_PATH = RUNTIME_DATASET_ROOT / str(
    globals().get("RECOVERY_EVIDENCE_MANIFEST_RELATIVE_PATH", "adapter_ood_oe_evidence_manifest.csv")
)
input_revision = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=str(ROOT),
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
campaign_digest = hashlib.sha256(CAMPAIGN_PATH.read_bytes()).hexdigest()
evidence_manifest_digest = (
    hashlib.sha256(EVIDENCE_MANIFEST_PATH.read_bytes()).hexdigest() if EVIDENCE_MANIFEST_PATH.is_file() else "missing"
)
lineage = _ood_notebook_campaign.build_campaign_lineage(
    input_revision=input_revision,
    campaign_digest=campaign_digest,
    evidence_manifest_digest=evidence_manifest_digest,
)
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
campaign_output_dir = (
    ROOT
    / "docs"
    / "ablation_results"
    / "adapter_ood_oe_recovery_notebook"
    / lineage["campaign_id"]
)
ledger_path = campaign_output_dir / "campaign_ledger.json"
report_path = campaign_output_dir / f"{timestamp}_summary.json"
ledger = _ood_notebook_campaign.load_campaign_ledger(ledger_path, lineage=lineage)

audit_output = ROOT / ".runtime_tmp" / "adapter_ood_oe_recovery_preflight"
RECOVERY_PREFLIGHT_BLOCKED_TARGETS = set()
try:
    if not EVIDENCE_MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Frozen v2 evidence manifest is missing: {EVIDENCE_MANIFEST_PATH}")
    manifest_validation_path = audit_output / "manifest_validation.json"
    manifest_command = [
        sys.executable,
        "scripts/validate_adapter_ood_oe_manifest.py",
        str(EVIDENCE_MANIFEST_PATH),
        "--base-dir",
        str(EVIDENCE_MANIFEST_BASE_DIR),
        "--output",
        str(manifest_validation_path),
    ]
    if DATASET_SOURCE_KIND == "github_release":
        manifest_command.extend(["--file-path-prefix", "data/prepared_runtime_datasets"])
    for target in selected_targets:
        manifest_command.extend(["--target", target])
    print("[COMMAND]", " ".join(manifest_command))
    manifest_completed = subprocess.run(manifest_command, cwd=str(ROOT), check=False, text=True)
    manifest_report = json.loads(manifest_validation_path.read_text(encoding="utf-8"))
    manifest_errors = [
        dict(issue)
        for issue in list(manifest_report.get("issues") or [])
        if str(issue.get("severity") or "") == "error"
    ]
    RECOVERY_PREFLIGHT_BLOCKED_TARGETS.update(
        _ood_notebook_campaign.evidence_preflight_blocked_targets(
            manifest_report,
            required_targets=selected_targets,
        )
    )
    audit_commands = []
    if selected_targets == release_targets:
        audit_commands.append(
            [
                sys.executable,
                "scripts/audit_ood_oe_quality.py",
                "--all",
                "--prepared-root",
                str(RUNTIME_DATASET_ROOT),
                "--output-dir",
                str(audit_output),
                "--fail-on-exact-overlap",
                "--fail-on-near-duplicate",
            ]
        )
    else:
        for target in selected_targets:
            audit_commands.append(
                [
                    sys.executable,
                    "scripts/audit_ood_oe_quality.py",
                    "--dataset-root",
                    str(RUNTIME_DATASET_ROOT / target),
                    "--dataset-key",
                    target,
                    "--output-dir",
                    str(audit_output),
                    "--fail-on-exact-overlap",
                    "--fail-on-near-duplicate",
                ]
            )
    for audit_command in audit_commands:
        _run_checked(
            audit_command,
            cwd=ROOT,
        )
    RECOVERY_PREFLIGHT_OK = manifest_completed.returncode == 0
    RECOVERY_PREFLIGHT_ERROR = (
        ""
        if RECOVERY_PREFLIGHT_OK
        else f"strict_manifest_errors={len(manifest_errors)} "
        f"blocked_targets={','.join(sorted(RECOVERY_PREFLIGHT_BLOCKED_TARGETS))}"
    )
except Exception as exc:
    RECOVERY_PREFLIGHT_OK = False
    RECOVERY_PREFLIGHT_ERROR = str(exc)
    RECOVERY_PREFLIGHT_BLOCKED_TARGETS.update(selected_targets)
    if not bool(globals().get("RECOVERY_CONTINUE_ON_ERROR", True)):
        raise

RECOVERY_RESULTS = {}
adapter_recs = get_adapter_recs()
completed_experiment_count = 0
max_completed_experiments = int(globals().get("RECOVERY_MAX_COMPLETED_EXPERIMENTS", 0))
max_targets = int(globals().get("RECOVERY_MAX_TARGETS", 0))
resume_from_ledger = bool(globals().get("RECOVERY_RESUME_FROM_LEDGER", True))
selected_target_count = 0

for target_entry in campaign["targets"]:
    target = str(target_entry["target"])
    if target_allowlist and target not in target_allowlist:
        RECOVERY_RESULTS[target] = {
            "status": "blocked",
            "pass": False,
            "ready": False,
            "reason": "target_not_selected",
            "experiments": [],
        }
        continue
    if max_targets > 0 and selected_target_count >= max_targets:
        RECOVERY_RESULTS[target] = {
            "status": "blocked",
            "pass": False,
            "ready": False,
            "reason": "target_budget_reached",
            "experiments": [],
        }
        continue
    selected_target_count += 1
    if max_completed_experiments > 0 and completed_experiment_count >= max_completed_experiments:
        RECOVERY_RESULTS[target] = {
            "status": "blocked",
            "pass": False,
            "ready": False,
            "reason": "campaign_budget_reached",
            "experiments": [],
        }
        continue
    prior_target = dict((ledger.get("targets") or {}).get(target) or {})
    prior_acceptance_path = Path(str(prior_target.get("behavioral_acceptance_path") or ""))
    prior_acceptance = read_behavioral_acceptance(prior_acceptance_path)
    if resume_from_ledger and behavioral_acceptance_pass(prior_acceptance):
        RECOVERY_RESULTS[target] = {
            "status": "passed",
            "pass": True,
            "ready": True,
            "reason": "resumed_verified_final_acceptance",
            "final_behavioral_acceptance_path": str(prior_acceptance_path),
            "final_behavioral_acceptance_status": prior_acceptance.get("status"),
            "experiments": [],
        }
        continue
    if target not in adapter_recs:
        RECOVERY_RESULTS[target] = {
            "status": "blocked",
            "pass": False,
            "ready": False,
            "reason": "adapter_recommendation_missing",
            "experiments": [],
        }
        continue
    current_path = None
    current = {}
    final_path = None
    final_acceptance = {}
    experiment_results = []
    target_status = "pending"
    target_reason = ""
    for attempt_index, experiment in enumerate(target_entry.get("experiments", []), start=1):
        if attempt_index > int(target_entry.get("attempt_cap", 4)):
            break
        should_run, gate_reason = experiment_gate(str(experiment.get("stage") or ""), current)
        if target in RECOVERY_PREFLIGHT_BLOCKED_TARGETS:
            should_run = False
            gate_reason = "target_evidence_preflight_failed"
        if not should_run:
            experiment_results.append(
                {
                    "experiment_id": experiment.get("experiment_id"),
                    "stage": experiment.get("stage"),
                    "status": "skipped",
                    "reason": gate_reason,
                }
            )
            if gate_reason in {"same_crop_ood_sample_floor_failed", "target_evidence_preflight_failed"}:
                target_status = "blocked"
                target_reason = gate_reason
            continue
        resolved_config = dict(experiment.get("resolved_config") or experiment.get("manual_param_overrides") or {})
        resolved_config.update(_low_resource_overrides())
        resolved_config_digest = _ood_notebook_campaign.json_digest(resolved_config)
        experiment_id = str(experiment.get("experiment_id") or "")
        resumed_path = None
        resumed_report = {}
        if resume_from_ledger:
            resumed_path, resumed_report = _ood_notebook_campaign.resumable_experiment(
                ledger,
                experiment_id=experiment_id,
                resolved_config_digest=resolved_config_digest,
            )
        if resumed_path is not None and not behavioral_dev_report_pass(resumed_report):
            current_path, current = resumed_path, resumed_report
            experiment_results.append(
                {
                    "experiment_id": experiment_id,
                    "stage": experiment.get("stage"),
                    "status": "resumed",
                    "dev_report_path": str(current_path),
                    "dev_report_status": current.get("status"),
                }
            )
            continue
        try:
            try:
                _require_disk_headroom(ROOT)
                current_path, current = _run_training_experiment(
                    target=target,
                    experiment=dict(experiment),
                    shared_globals=globals(),
                )
                adapter_artifact_path = Path(str(globals().get("REPO_OUTPUT_DIR") or ""))
                completed_experiment_count += 1
                ledger.setdefault("experiments", {})[experiment_id] = {
                    "target": target,
                    "candidate": experiment.get("candidate"),
                    "status": "completed",
                    "dev_report_path": str(current_path),
                    "dev_report_pass": behavioral_dev_report_pass(current),
                    "adapter_artifact_path": str(adapter_artifact_path),
                    "resolved_config": resolved_config,
                    "resolved_config_digest": resolved_config_digest,
                    "seed": resolved_config.get("SEED", campaign.get("seed")),
                    "manifest_digest": evidence_manifest_digest,
                    "input_revision": input_revision,
                }
                _ood_notebook_campaign.write_campaign_ledger(ledger_path, ledger)
                if behavioral_dev_report_pass(current):
                    final_path, final_acceptance = _run_locked_final_evaluation(globals())
            finally:
                _cleanup_runtime_state(globals(), label=str(experiment.get("experiment_id")))
            reclaim_report = _reclaim_failed_experiment_payloads(
                globals(),
                behavioral_passed=behavioral_acceptance_pass(final_acceptance),
            )
            experiment_results.append(
                {
                    "experiment_id": experiment_id,
                    "stage": experiment.get("stage"),
                    "status": "completed",
                    "dev_report_path": str(current_path),
                    "dev_report_status": current.get("status"),
                    "dev_report_pass": behavioral_dev_report_pass(current),
                    "resource_reclaim": reclaim_report,
                }
            )
            if behavioral_dev_report_pass(current):
                final_passed = behavioral_acceptance_pass(final_acceptance)
                target_status = "passed" if final_passed else "failed"
                target_reason = (
                    "frozen_winner_behavioral_acceptance_passed"
                    if final_passed
                    else "frozen_winner_failed_locked_test_no_test_driven_tuning"
                )
                ledger.setdefault("targets", {})[target] = {
                    "status": target_status,
                    "dev_winner_experiment_id": experiment_id,
                    "dev_report_path": str(current_path),
                    "behavioral_acceptance_path": "" if final_path is None else str(final_path),
                    "behavioral_acceptance_pass": final_passed,
                }
                _ood_notebook_campaign.write_campaign_ledger(ledger_path, ledger)
                RECOVERY_RESULTS[target] = {
                    "status": target_status,
                    "pass": final_passed,
                    "ready": final_passed,
                    "reason": target_reason,
                    "frozen_dev_winner": experiment_id,
                    "final_behavioral_acceptance_path": "" if final_path is None else str(final_path),
                    "final_behavioral_acceptance_status": final_acceptance.get("status", "missing"),
                    "experiments": list(experiment_results),
                }
                _write_and_publish_recovery_report(
                    report_path=report_path,
                    phase="checkpoint",
                    completed_experiment_count=completed_experiment_count,
                    max_completed_experiments=max_completed_experiments,
                    max_targets=max_targets,
                    ledger_path=ledger_path,
                )
                break
            RECOVERY_RESULTS[target] = {
                "status": "running",
                "pass": False,
                "ready": False,
                "reason": "candidate_dev_evaluation_completed",
                "latest_dev_report_path": "" if current_path is None else str(current_path),
                "latest_dev_report_status": current.get("status", "missing"),
                "experiments": list(experiment_results),
            }
            _write_and_publish_recovery_report(
                report_path=report_path,
                phase="checkpoint",
                completed_experiment_count=completed_experiment_count,
                max_completed_experiments=max_completed_experiments,
                max_targets=max_targets,
                ledger_path=ledger_path,
            )
            if max_completed_experiments > 0 and completed_experiment_count >= max_completed_experiments:
                target_status = "blocked"
                target_reason = "campaign_budget_reached"
                break
        except Exception as exc:
            experiment_results.append(
                {
                    "experiment_id": experiment.get("experiment_id"),
                    "stage": experiment.get("stage"),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            target_status = "failed"
            normalized_error = str(exc).lower()
            target_reason = (
                "resource_failure"
                if any(marker in normalized_error for marker in ("out of memory", "no space left", "disk budget"))
                else str(exc)
            )
            print(f"[RECOVERY] FAILED target={target} experiment={experiment.get('experiment_id')}: {exc}")
            if not bool(globals().get("RECOVERY_CONTINUE_ON_ERROR", True)):
                raise
            break

    passed = behavioral_acceptance_pass(final_acceptance)
    if passed:
        target_status = "passed"
        target_reason = "frozen_winner_behavioral_acceptance_passed"
    elif target_status == "pending":
        target_status = "failed"
        target_reason = "bounded_candidates_exhausted_return_to_data_or_label_audit"
    RECOVERY_RESULTS[target] = {
        "status": target_status,
        "pass": passed,
        "ready": passed,
        "reason": target_reason,
        "latest_dev_report_path": "" if current_path is None else str(current_path),
        "latest_dev_report_status": current.get("status", "missing"),
        "final_behavioral_acceptance_path": "" if final_path is None else str(final_path),
        "final_behavioral_acceptance_status": final_acceptance.get("status", "missing"),
        "experiments": experiment_results,
    }

RECOVERY_COMPLETION_REPORT = _write_and_publish_recovery_report(
    report_path=report_path,
    phase="final",
    completed_experiment_count=completed_experiment_count,
    max_completed_experiments=max_completed_experiments,
    max_targets=max_targets,
    ledger_path=ledger_path,
)
print("\n[RECOVERY] FINAL SUMMARY")
print(json.dumps(RECOVERY_COMPLETION_REPORT, indent=2, ensure_ascii=False))
print(
    f"[RECOVERY] passed_target_count={RECOVERY_COMPLETION_REPORT['passed_target_count']}/"
    f"{RECOVERY_COMPLETION_REPORT['required_target_count']}"
)

RECOVERY_DISCONNECT_REPORT = dict(RECOVERY_COMPLETION_REPORT)
RECOVERY_DISCONNECT_REPORT["campaign_passed"] = bool(RECOVERY_COMPLETION_REPORT["pass"])
RECOVERY_DISCONNECT_REPORT["ready"] = True
RECOVERY_DISCONNECT_REPORT["checks"] = {
    "campaign_loop_completed": True,
    "final_report_written": report_path.is_file(),
    "final_report_publish_attempt_completed": True,
}
RECOVERY_DISCONNECT_REPORT["missing"] = []
RECOVERY_DISCONNECT_REPORT["soft_missing"] = (
    []
    if RECOVERY_COMPLETION_REPORT["pass"]
    else ["campaign_not_8_of_8_behavioral_pass; inspect the pushed recovery summary"]
)
maybe_auto_disconnect_colab_runtime(
    enabled=bool(globals().get("RECOVERY_AUTO_DISCONNECT_RUNTIME", False))
    and bool(RECOVERY_COMPLETION_REPORT.get("publish_ok", False)),
    grace_period_sec=float(globals().get("RECOVERY_AUTO_DISCONNECT_GRACE_SECONDS", 30)),
    telemetry=globals().get("TELEMETRY"),
    completion_report=RECOVERY_DISCONNECT_REPORT,
)
