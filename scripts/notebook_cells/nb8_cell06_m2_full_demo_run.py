# Auto-extracted from colab_notebooks/8_auto_router_adapter_prediction.ipynb cell 6.
# Keep notebook execute-only cells thin; edit behavior here.
# ruff: noqa: F821

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.colab_notebook_helpers import maybe_auto_disconnect_colab_runtime
from scripts.colab_repo_bootstrap import push_repo_paths_to_github
from scripts.compare_m2_demo_results import enrich_summary_manifest_sha256
from scripts.notebook_helpers.nb8_m2_commands import build_m2_demo_checklist_command
from scripts.notebook_helpers.nb8_m2_reporting import (
    format_elapsed_seconds,
    write_m2_result_comparison,
)
from scripts.notebook_helpers.nb8_m2_runtime_helpers import (
    copy_reusable_calibration_if_available,
    discard_stale_calibration,
    m2_subprocess_env,
    prototype_bank_matches_curation_root,
    restore_latest_handoff_cache,
    run_open_world_router_validation,
    validate_curated_prototype_bank,
    write_and_push_m2_failure_summary,
)
from scripts.notebook_helpers.nb8_m2_settings import load_m2_cell_settings

globals().update(load_m2_cell_settings(globals()))
_m2_settings = globals()


cell_script_root = Path(str(globals().get("__notebook_cell_script_root__", ""))).resolve()
repo_root = cell_script_root.parents[1] if cell_script_root.name == "notebook_cells" else Path.cwd().resolve()
adapter_root_path = Path(str(globals().get("ADAPTER_ROOT") or "runs"))
prototype_calibration_output_path = (repo_root / M2_PROTOTYPE_CALIBRATION_OUTPUT).resolve()

if M2_OPEN_WORLD_ONLY:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    open_world_validation_report = run_open_world_router_validation(
        repo_root,
        stamp,
        adapter_root_path,
        _m2_settings,
        prototype_calibration_output_path=prototype_calibration_output_path,
    )
    if M2_AUTO_PUSH_RESULTS:
        try:
            m2_demo_publish_report = push_repo_paths_to_github(
                repo_root=repo_root,
                relative_paths=[str(open_world_validation_report["run_dir"])],
                remote_name=M2_AUTO_PUSH_REMOTE_NAME,
                branch=M2_AUTO_PUSH_BRANCH,
                commit_message=f"Add open-world router demo {stamp}",
                print_fn=print,
            )
        except Exception as exc:
            m2_demo_publish_report = {
                "enabled": True,
                "pushed": False,
                "paths": [str(open_world_validation_report["run_dir"])],
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            print(f"[GIT] Open-world result auto-push failed: {m2_demo_publish_report['error']}")
    else:
        m2_demo_publish_report = {
            "enabled": False,
            "pushed": False,
            "paths": [str(open_world_validation_report["run_dir"])],
        }
    push_done = bool(
        m2_demo_publish_report.get("pushed")
        or (
            m2_demo_publish_report.get("enabled")
            and not m2_demo_publish_report.get("error")
            and m2_demo_publish_report.get("staged_files") == []
        )
    )
    completion_checks = {
        "git_push": bool(push_done),
        "open_world_router_validation_passed": bool(open_world_validation_report.get("ready")),
    }
    m2_completion_report = {
        "ready": bool(push_done and open_world_validation_report.get("ready")),
        "checks": completion_checks,
        "missing": [name for name, passed in completion_checks.items() if not passed],
        "soft_missing": [],
        "open_world_router_validation": open_world_validation_report,
    }
    print(f"[COLAB] Open-world-only completion checks -> {m2_completion_report['checks']}")
    m2_demo_disconnect_report = maybe_auto_disconnect_colab_runtime(
        enabled=bool(M2_AUTO_DISCONNECT_RUNTIME),
        grace_period_sec=M2_AUTO_DISCONNECT_GRACE_SECONDS,
        completion_report=m2_completion_report,
        print_fn=print,
    )
    m2_demo_result = {"open_world_router_validation": open_world_validation_report}
elif not M2_RUN_FULL_DEMO:
    m2_demo_result = None
    m2_demo_publish_report = {"enabled": False, "pushed": False, "reason": "M2_RUN_FULL_DEMO=False"}
    m2_demo_disconnect_report = {"ready": False, "missing": ["m2_full_demo_skipped"]}
    print("[M2] Full demo manifest run skipped because M2_RUN_FULL_DEMO=False.")
else:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    m2_run_started_at = datetime.now(timezone.utc)
    m2_run_start_perf = time.perf_counter()

    active_manifest = M2_PROBLEM_ONLY_MANIFEST if M2_RUN_PROBLEM_ONLY_DEMO else M2_DEMO_MANIFEST
    active_comparison_baseline = (
        M2_PROBLEM_ONLY_COMPARISON_BASELINE if M2_RUN_PROBLEM_ONLY_DEMO else M2_COMPARISON_BASELINE
    )
    active_calibration_manifest = (
        M2_PROBLEM_ONLY_CALIBRATION_MANIFEST if M2_RUN_PROBLEM_ONLY_DEMO else active_manifest
    )
    active_output = ".runtime_tmp/m2_problem_only_demo_checklist_run.json" if M2_RUN_PROBLEM_ONLY_DEMO else M2_DEMO_OUTPUT
    active_markdown_output = (
        ".runtime_tmp/m2_problem_only_demo_checklist_run.md" if M2_RUN_PROBLEM_ONLY_DEMO else M2_DEMO_MARKDOWN_OUTPUT
    )
    active_analysis_output = ".runtime_tmp/m2_problem_only_analysis_summary.json" if M2_RUN_PROBLEM_ONLY_DEMO else M2_ANALYSIS_OUTPUT
    active_analysis_markdown_output = (
        ".runtime_tmp/m2_problem_only_analysis_summary.md" if M2_RUN_PROBLEM_ONLY_DEMO else M2_ANALYSIS_MARKDOWN_OUTPUT
    )

    manifest_path = (repo_root / active_manifest).resolve()
    calibration_manifest_path = (repo_root / active_calibration_manifest).resolve()
    output_path = (repo_root / active_output).resolve()
    markdown_output_path = (repo_root / active_markdown_output).resolve()
    analysis_output_path = (repo_root / active_analysis_output).resolve()
    analysis_markdown_output_path = (repo_root / active_analysis_markdown_output).resolve()
    handoff_cache_path = (repo_root / M2_HANDOFF_CACHE).resolve()
    if (
        M2_ENABLE_PROTOTYPE_RECONCILER
        and M2_REUSE_EXISTING_PROTOTYPES
        and (not M2_PROTOTYPE_BANK or not M2_TAXONOMY_REGISTRY)
    ):
        for candidate_dir in sorted((repo_root / M2_REPO_RESULTS_ROOT).glob("*"), reverse=True):
            prototype_bank_candidate = candidate_dir / "prototype_bank.json"
            taxonomy_registry_candidate = candidate_dir / "taxonomy_registry.json"
            if not prototype_bank_candidate.is_file() or not taxonomy_registry_candidate.is_file():
                continue
            if not prototype_bank_matches_curation_root(repo_root, prototype_bank_candidate, M2_PROTOTYPE_CURATION_ROOT):
                continue
            if not M2_PROTOTYPE_BANK:
                M2_PROTOTYPE_BANK = str(prototype_bank_candidate.relative_to(repo_root))
            if not M2_TAXONOMY_REGISTRY:
                M2_TAXONOMY_REGISTRY = str(taxonomy_registry_candidate.relative_to(repo_root))
            print(f"[M2] Reusing existing prototype artifacts from {candidate_dir.relative_to(repo_root)}.")
            break

    if M2_ENABLE_PROTOTYPE_RECONCILER and M2_AUTO_BUILD_PROTOTYPES and (
        not M2_PROTOTYPE_BANK or not M2_TAXONOMY_REGISTRY
    ):
        prototype_run_id = M2_PROTOTYPE_RUN_ID or f"m2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        prototype_command = [
            sys.executable,
            str(repo_root / "scripts" / "build_router_prototype_bank.py"),
            "--run-id",
            prototype_run_id,
            "--embedding-backend",
            M2_PROTOTYPE_EMBEDDING_BACKEND,
            "--embedding-model-id",
            M2_PROTOTYPE_EMBEDDING_MODEL_ID,
            "--embedding-device",
            M2_PROTOTYPE_EMBEDDING_DEVICE,
        ]
        if M2_PROTOTYPE_MAX_IMAGES_PER_CLASS is not None:
            prototype_command.extend(["--max-images-per-class", str(int(M2_PROTOTYPE_MAX_IMAGES_PER_CLASS))])
        if M2_PROTOTYPE_CURATION_ROOT:
            prototype_command.extend(["--curation-root", M2_PROTOTYPE_CURATION_ROOT])
        print("[M2] Building router prototype artifacts before manifest run.")
        prototype_completed = subprocess.run(prototype_command, cwd=repo_root, check=False, env=m2_subprocess_env(_m2_settings))
        print(f"[M2] prototype_builder_exit_code={prototype_completed.returncode}")
        if prototype_completed.returncode != 0:
            if M2_PROTOTYPE_CURATION_ROOT:
                discard_stale_calibration(prototype_calibration_output_path)
                write_and_push_m2_failure_summary(
                    repo_root=repo_root,
                    settings=_m2_settings,
                    stamp=stamp,
                    prototype_calibration_output_path=prototype_calibration_output_path,
                    phase="prototype_builder",
                    error=(
                        "Prototype builder failed while M2_PROTOTYPE_CURATION_ROOT is set; "
                        "stopping instead of running an uncurated M2 demo."
                    ),
                    started_at=m2_run_started_at,
                    manifest_path=manifest_path,
                    calibration_manifest_path=calibration_manifest_path,
                    output_path=output_path,
                    markdown_output_path=markdown_output_path,
                    analysis_output_path=analysis_output_path,
                    analysis_markdown_output_path=analysis_markdown_output_path,
                    runner_exit_code=prototype_completed.returncode,
                )
                raise RuntimeError(
                    "Prototype builder failed while M2_PROTOTYPE_CURATION_ROOT is set; "
                    "stopping instead of running an uncurated M2 demo."
                )
            if M2_REQUIRE_CALIBRATED_PROTOTYPE_POLICY:
                M2_ENABLE_PROTOTYPE_RECONCILER = False
            discard_stale_calibration(prototype_calibration_output_path)
            print("[M2] Prototype builder failed; prototype reconciler disabled for this run.")
        prototype_dir = repo_root / "runs" / "_index" / "router_prototypes" / prototype_run_id
        if not M2_PROTOTYPE_BANK:
            M2_PROTOTYPE_BANK = str((prototype_dir / "prototype_bank.json").relative_to(repo_root))
        if not M2_TAXONOMY_REGISTRY:
            M2_TAXONOMY_REGISTRY = str((prototype_dir / "taxonomy_registry.json").relative_to(repo_root))

    if M2_ENABLE_PROTOTYPE_RECONCILER:
        validate_curated_prototype_bank(repo_root, M2_PROTOTYPE_BANK, M2_PROTOTYPE_CURATION_ROOT)

    prototype_calibration_selected = False
    prototype_target_policy_selected = False
    if M2_ENABLE_PROTOTYPE_RECONCILER and M2_AUTO_CALIBRATE_PROTOTYPE_RECONCILER and M2_PROTOTYPE_BANK:
        prototype_bank_path = (repo_root / M2_PROTOTYPE_BANK).resolve()
        reused_calibration_path = copy_reusable_calibration_if_available(
            repo_root,
            calibration_manifest_path,
            prototype_bank_path,
            prototype_calibration_output_path,
            _m2_settings,
        )
        if reused_calibration_path is not None:
            print(
                "[M2] Reusing existing prototype calibration from "
                f"{reused_calibration_path.relative_to(repo_root)}."
            )
        else:
            calibration_command = [
                sys.executable,
                str(repo_root / "scripts" / "calibrate_router_prototype_reconciler.py"),
                "--manifest",
                str(calibration_manifest_path),
                "--prototype-bank",
                str(prototype_bank_path),
                "--output",
                str(prototype_calibration_output_path),
                "--min-precision",
                str(M2_PROTOTYPE_CALIBRATION_MIN_PRECISION),
                "--min-coverage",
                str(M2_PROTOTYPE_CALIBRATION_MIN_COVERAGE),
                "--max-negative-false-accepts",
                str(M2_PROTOTYPE_CALIBRATION_MAX_NEGATIVE_FALSE_ACCEPTS),
                "--max-negative-false-accept-rate",
                str(M2_PROTOTYPE_CALIBRATION_MAX_NEGATIVE_FALSE_ACCEPT_RATE),
                "--target-min-precision",
                str(M2_PROTOTYPE_TARGET_MIN_PRECISION),
                "--target-max-supported-wrong",
                str(int(M2_PROTOTYPE_TARGET_MAX_SUPPORTED_WRONG)),
                "--target-max-cross-part-supported-wrong",
                str(M2_PROTOTYPE_TARGET_MAX_CROSS_PART_SUPPORTED_WRONG),
                "--target-class-min-accepted",
                str(int(M2_PROTOTYPE_TARGET_CLASS_MIN_ACCEPTED)),
                "--similarity-grid",
                M2_PROTOTYPE_SIMILARITY_GRID,
                "--margin-grid",
                M2_PROTOTYPE_MARGIN_GRID,
                "--negative-gap-grid",
                M2_PROTOTYPE_NEGATIVE_GAP_GRID,
                "--target-policy-negative-mode",
                M2_PROTOTYPE_TARGET_POLICY_NEGATIVE_MODE,
            ]
            if M2_ALLOW_NON_PLANT_FALSE_ACCEPTS:
                calibration_command.append("--allow-non-plant-false-accepts")
            if M2_PROTOTYPE_CALIBRATION_LIMIT is not None:
                calibration_command.extend(["--limit", str(int(M2_PROTOTYPE_CALIBRATION_LIMIT))])
            print("[M2] Calibrating prototype reconciler thresholds.")
            calibration_completed = subprocess.run(
                calibration_command,
                cwd=repo_root,
                check=False,
                env=m2_subprocess_env(_m2_settings),
            )
            print(f"[M2] prototype_calibration_exit_code={calibration_completed.returncode}")
            if calibration_completed.returncode != 0:
                if M2_REQUIRE_CALIBRATED_PROTOTYPE_POLICY:
                    M2_ENABLE_PROTOTYPE_RECONCILER = False
                print("[M2] Prototype calibration failed; ignoring any stale calibration output for this run.")
                discard_stale_calibration(prototype_calibration_output_path)
        if prototype_calibration_output_path.is_file():
            calibration_payload = json.loads(prototype_calibration_output_path.read_text(encoding="utf-8"))
            selected_policy = calibration_payload.get("selected_policy")
            target_policies = calibration_payload.get("target_policies")
            prototype_calibration_selected = isinstance(selected_policy, dict)
            prototype_target_policy_selected = (
                any(
                    isinstance(entry, dict) and isinstance(entry.get("selected_policy"), dict)
                    or (
                        isinstance(entry, dict)
                        and any(
                            isinstance(class_entry, dict)
                            and (
                                isinstance(class_entry.get("selected_policy"), dict)
                                or isinstance(class_entry.get("exact_class_rescue_policy"), dict)
                            )
                            for class_entry in (
                                entry.get("class_policies")
                                if isinstance(entry.get("class_policies"), dict)
                                else {}
                            ).values()
                        )
                    )
                    for entry in target_policies.values()
                )
                if isinstance(target_policies, dict)
                else False
            )
            if prototype_calibration_selected:
                if M2_PROTOTYPE_MIN_SIMILARITY is None:
                    M2_PROTOTYPE_MIN_SIMILARITY = selected_policy.get("min_similarity")
                if M2_PROTOTYPE_MIN_MARGIN is None:
                    M2_PROTOTYPE_MIN_MARGIN = selected_policy.get("min_margin")
                if M2_PROTOTYPE_MIN_NEGATIVE_GAP is None:
                    M2_PROTOTYPE_MIN_NEGATIVE_GAP = selected_policy.get("min_negative_gap")
                print("[M2] Prototype calibration selected policy:")
                print(json.dumps(selected_policy, indent=2, ensure_ascii=False))
            else:
                print("[M2] Prototype calibration did not select a runtime policy.")
            if prototype_target_policy_selected:
                print("[M2] Prototype calibration selected at least one target-specific policy.")
        if M2_REQUIRE_CALIBRATED_PROTOTYPE_POLICY and not (
            prototype_calibration_selected or prototype_target_policy_selected
        ):
            M2_ENABLE_PROTOTYPE_RECONCILER = False
            print("[M2] Prototype reconciler disabled because no calibrated policy was selected.")

    restored_handoff_cache = restore_latest_handoff_cache(repo_root, handoff_cache_path, _m2_settings)
    if restored_handoff_cache:
        print(f"[M2] Restored handoff cache from {restored_handoff_cache.relative_to(repo_root)}")

    command = build_m2_demo_checklist_command(
        python_executable=sys.executable,
        repo_root=repo_root,
        manifest_path=manifest_path,
        device=str(DEVICE),
        config_env=str(CONFIG_ENV),
        adapter_root_path=adapter_root_path,
        output_path=output_path,
        markdown_output_path=markdown_output_path,
        analysis_output_path=analysis_output_path,
        analysis_markdown_output_path=analysis_markdown_output_path,
        batch_size=int(M2_BATCH_SIZE),
        adapter_batch_size=int(M2_ADAPTER_BATCH_SIZE),
        handoff_cache_path=handoff_cache_path,
        demo_limit=M2_DEMO_LIMIT,
        stop_on_dependency_blocker=bool(M2_STOP_ON_DEPENDENCY_BLOCKER),
        refresh_handoff_cache=bool(M2_REFRESH_HANDOFF_CACHE),
        enable_prototype_reconciler=bool(M2_ENABLE_PROTOTYPE_RECONCILER),
        prototype_bank=M2_PROTOTYPE_BANK,
        taxonomy_registry=M2_TAXONOMY_REGISTRY,
        prototype_min_similarity=M2_PROTOTYPE_MIN_SIMILARITY,
        prototype_min_margin=M2_PROTOTYPE_MIN_MARGIN,
        prototype_min_negative_gap=M2_PROTOTYPE_MIN_NEGATIVE_GAP,
        prototype_calibration_output_path=prototype_calibration_output_path,
    )

    print(f"[M2] repo_root={repo_root}")
    print(f"[M2] manifest={manifest_path}")
    if calibration_manifest_path != manifest_path:
        print(f"[M2] calibration_manifest={calibration_manifest_path}")
    print(f"[M2] output={output_path}")
    print(f"[M2] markdown_output={markdown_output_path}")
    print(f"[M2] analysis_output={analysis_output_path}")
    print(f"[M2] analysis_markdown_output={analysis_markdown_output_path}")
    print(f"[M2] handoff_cache={handoff_cache_path}")
    print(f"[M2] prototype_reconciler={M2_ENABLE_PROTOTYPE_RECONCILER}")
    if M2_ENABLE_PROTOTYPE_RECONCILER:
        print(f"[M2] prototype_bank={M2_PROTOTYPE_BANK or 'missing'}")
        print(f"[M2] taxonomy_registry={M2_TAXONOMY_REGISTRY or 'missing'}")
        print(f"[M2] prototype_calibration={prototype_calibration_output_path}")
    print("[M2] Starting full manifest run. This can take a while on 522 images.")

    runner_started_at = datetime.now(timezone.utc)
    runner_start_perf = time.perf_counter()
    completed = subprocess.run(command, cwd=repo_root, check=False, env=m2_subprocess_env(_m2_settings))
    runner_finished_at = datetime.now(timezone.utc)
    runner_elapsed_seconds = time.perf_counter() - runner_start_perf
    m2_run_finished_at = runner_finished_at
    m2_run_elapsed_seconds = time.perf_counter() - m2_run_start_perf
    print(f"[M2] runner_exit_code={completed.returncode}")
    print(f"[M2] runner_elapsed={format_elapsed_seconds(runner_elapsed_seconds)}")

    m2_demo_result = None
    report_ready = False
    if output_path.is_file():
        m2_demo_result = json.loads(output_path.read_text(encoding="utf-8"))
        report_ready = True
        print("[M2] Summary:")
        print(json.dumps(m2_demo_result.get("summary", {}), indent=2, ensure_ascii=False))
    else:
        print("[M2] Output report was not written. Check the cell log above.")
        m2_demo_failure_report = write_and_push_m2_failure_summary(
            repo_root=repo_root,
            settings=_m2_settings,
            stamp=stamp,
            prototype_calibration_output_path=prototype_calibration_output_path,
            phase="runner_output_missing",
            error="M2 runner finished without writing the output report.",
            started_at=m2_run_started_at,
            manifest_path=manifest_path,
            calibration_manifest_path=calibration_manifest_path,
            output_path=output_path,
            markdown_output_path=markdown_output_path,
            analysis_output_path=analysis_output_path,
            analysis_markdown_output_path=analysis_markdown_output_path,
            runner_exit_code=int(completed.returncode),
        )

    repo_results_rel = Path(M2_REPO_RESULTS_ROOT) / stamp
    repo_results_dir = repo_root / repo_results_rel
    m2_demo_publish_report = {"enabled": bool(M2_AUTO_PUSH_RESULTS), "pushed": False}
    m2_comparison_report = {
        "baseline": active_comparison_baseline,
        "enabled": bool(active_comparison_baseline),
        "written": False,
        "status": "not_run",
        "checks": {},
    }
    open_world_validation_report = {
        "enabled": bool(M2_RUN_OPEN_WORLD_ROUTER_VALIDATION),
        "run": False,
        "status": "not_run",
        "ready": False,
        "checks": {},
    }

    if report_ready:
        repo_results_dir.mkdir(parents=True, exist_ok=True)
        copied_paths = []
        provenance_paths = []
        if prototype_calibration_output_path.is_file():
            provenance_paths.append(prototype_calibration_output_path)
        if M2_PROTOTYPE_BANK:
            prototype_bank_path = (repo_root / M2_PROTOTYPE_BANK).resolve()
            if prototype_bank_path.is_file():
                provenance_paths.append(prototype_bank_path)
                prototype_summary_path = prototype_bank_path.parent / "summary.md"
                if prototype_summary_path.is_file():
                    provenance_paths.append(prototype_summary_path)
        if M2_TAXONOMY_REGISTRY:
            taxonomy_registry_path = (repo_root / M2_TAXONOMY_REGISTRY).resolve()
            if taxonomy_registry_path.is_file():
                provenance_paths.append(taxonomy_registry_path)
        if handoff_cache_path.is_file():
            provenance_paths.append(handoff_cache_path)
        for source_path in (
            output_path,
            markdown_output_path,
            analysis_output_path,
            analysis_markdown_output_path,
            *provenance_paths,
        ):
            if source_path.is_file():
                destination = repo_results_dir / source_path.name
                shutil.copy2(source_path, destination)
                copied_paths.append(destination.relative_to(repo_root).as_posix())

        summary_path = repo_results_dir / "summary.json"
        summary_payload = {
            "created_at": stamp,
            "started_at": m2_run_started_at.isoformat(),
            "finished_at": m2_run_finished_at.isoformat(),
            "elapsed_seconds": m2_run_elapsed_seconds,
            "elapsed_human": format_elapsed_seconds(m2_run_elapsed_seconds),
            "runner_started_at": runner_started_at.isoformat(),
            "runner_finished_at": runner_finished_at.isoformat(),
            "runner_elapsed_seconds": runner_elapsed_seconds,
            "runner_elapsed_human": format_elapsed_seconds(runner_elapsed_seconds),
            "runner_exit_code": int(completed.returncode),
            "manifest": str(manifest_path.relative_to(repo_root)),
            "problem_only": bool(M2_RUN_PROBLEM_ONLY_DEMO),
            "prototype_calibration_manifest": str(calibration_manifest_path.relative_to(repo_root)),
            "batch_size": int(max(1, M2_BATCH_SIZE)),
            "adapter_batch_size": int(max(1, M2_ADAPTER_BATCH_SIZE)),
            "pytorch_cuda_alloc_conf": M2_PYTORCH_CUDA_ALLOC_CONF,
            "handoff_cache": {
                "path": str(M2_HANDOFF_CACHE),
                "refresh": bool(M2_REFRESH_HANDOFF_CACHE),
                "restored_from": str(restored_handoff_cache.relative_to(repo_root)) if restored_handoff_cache else "",
            },
            "output": str(output_path.relative_to(repo_root)),
            "markdown_output": str(markdown_output_path.relative_to(repo_root)),
            "analysis_output": str(analysis_output_path.relative_to(repo_root)),
            "analysis_markdown_output": str(analysis_markdown_output_path.relative_to(repo_root)),
            "prototype_reconciler": {
                "enabled": bool(M2_ENABLE_PROTOTYPE_RECONCILER),
                "reuse_existing_prototypes": bool(M2_REUSE_EXISTING_PROTOTYPES),
                "reuse_existing_prototype_calibration": bool(M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION),
                "auto_build_prototypes": bool(M2_AUTO_BUILD_PROTOTYPES),
                "prototype_max_images_per_class": M2_PROTOTYPE_MAX_IMAGES_PER_CLASS,
                "prototype_curation_root": M2_PROTOTYPE_CURATION_ROOT,
                "auto_calibrate": bool(M2_AUTO_CALIBRATE_PROTOTYPE_RECONCILER),
                "require_calibrated_policy": bool(M2_REQUIRE_CALIBRATED_PROTOTYPE_POLICY),
                "prototype_bank": M2_PROTOTYPE_BANK,
                "taxonomy_registry": M2_TAXONOMY_REGISTRY,
                "prototype_calibration_output": str(prototype_calibration_output_path.relative_to(repo_root))
                if prototype_calibration_output_path.is_file()
                else "",
                "prototype_min_similarity": M2_PROTOTYPE_MIN_SIMILARITY,
                "prototype_min_margin": M2_PROTOTYPE_MIN_MARGIN,
                "prototype_min_negative_gap": M2_PROTOTYPE_MIN_NEGATIVE_GAP,
                "calibration_selected_policy": bool(prototype_calibration_selected),
                "calibration_selected_target_policy": bool(prototype_target_policy_selected),
                "target_min_precision": M2_PROTOTYPE_TARGET_MIN_PRECISION,
                "target_max_supported_wrong": M2_PROTOTYPE_TARGET_MAX_SUPPORTED_WRONG,
                "target_max_cross_part_supported_wrong": M2_PROTOTYPE_TARGET_MAX_CROSS_PART_SUPPORTED_WRONG,
                "target_class_min_accepted": M2_PROTOTYPE_TARGET_CLASS_MIN_ACCEPTED,
                "target_policy_negative_mode": M2_PROTOTYPE_TARGET_POLICY_NEGATIVE_MODE,
            },
            "copied_artifacts": copied_paths,
            "summary": m2_demo_result.get("summary", {}) if isinstance(m2_demo_result, dict) else {},
            "analysis_summary": m2_demo_result.get("analysis_summary", {}) if isinstance(m2_demo_result, dict) else {},
        }
        enrich_summary_manifest_sha256(summary_payload, repo_root=repo_root)
        summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        copied_paths.append(summary_path.relative_to(repo_root).as_posix())
        summary_payload["copied_artifacts"] = copied_paths
        m2_comparison_report = write_m2_result_comparison(
            repo_root=repo_root,
            repo_results_dir=repo_results_dir,
            candidate_summary_payload=summary_payload,
            comparison_baseline=active_comparison_baseline,
            copied_paths=copied_paths,
        )

        summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        m2_demo_checkpoint_publish_report = {
            "enabled": bool(M2_AUTO_PUSH_RESULTS),
            "pushed": False,
            "path": repo_results_rel.as_posix(),
            "phase": "before_open_world_router_validation",
        }
        if M2_AUTO_PUSH_RESULTS:
            try:
                print("[GIT] M2 checkpoint auto-push before open-world router validation...")
                m2_demo_checkpoint_publish_report = push_repo_paths_to_github(
                    repo_root=repo_root,
                    relative_paths=[repo_results_rel.as_posix()],
                    remote_name=M2_AUTO_PUSH_REMOTE_NAME,
                    branch=M2_AUTO_PUSH_BRANCH,
                    commit_message=f"Add M2 demo checkpoint {stamp}",
                    print_fn=print,
                )
                m2_demo_checkpoint_publish_report["phase"] = "before_open_world_router_validation"
            except Exception as exc:
                m2_demo_checkpoint_publish_report = {
                    "enabled": True,
                    "pushed": False,
                    "path": repo_results_rel.as_posix(),
                    "phase": "before_open_world_router_validation",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
                print(f"[GIT] M2 checkpoint auto-push failed: {m2_demo_checkpoint_publish_report['error']}")

        if M2_RUN_OPEN_WORLD_ROUTER_VALIDATION:
            open_world_validation_report = run_open_world_router_validation(
                repo_root,
                stamp,
                adapter_root_path,
                _m2_settings,
                prototype_calibration_output_path=prototype_calibration_output_path,
            )
            copied_paths.append(str(open_world_validation_report["run_dir"]))
            summary_payload["open_world_router_validation"] = open_world_validation_report
            summary_payload["copied_artifacts"] = copied_paths
        summary_payload["checkpoint_publish"] = m2_demo_checkpoint_publish_report
        summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[M2] Repo result copy:")
        print(json.dumps(copied_paths, indent=2, ensure_ascii=False))

        if M2_AUTO_PUSH_RESULTS:
            try:
                push_paths = [repo_results_rel.as_posix()]
                if open_world_validation_report.get("run_dir"):
                    push_paths.append(str(open_world_validation_report["run_dir"]))
                m2_demo_publish_report = push_repo_paths_to_github(
                    repo_root=repo_root,
                    relative_paths=push_paths,
                    remote_name=M2_AUTO_PUSH_REMOTE_NAME,
                    branch=M2_AUTO_PUSH_BRANCH,
                    commit_message=f"Add M2 demo results {stamp}",
                    print_fn=print,
                )
            except Exception as exc:
                m2_demo_publish_report = {
                    "enabled": True,
                    "pushed": False,
                    "paths": push_paths,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
                print(f"[GIT] M2 result auto-push failed: {m2_demo_publish_report['error']}")
        else:
            push_paths = [repo_results_rel.as_posix()]
            if open_world_validation_report.get("run_dir"):
                push_paths.append(str(open_world_validation_report["run_dir"]))
            m2_demo_publish_report = {
                "enabled": False,
                "pushed": False,
                "paths": push_paths,
            }
    else:
        m2_demo_publish_report = {
            "enabled": bool(M2_AUTO_PUSH_RESULTS),
            "pushed": False,
            "error": "M2 output report was not written.",
            "failure_report": m2_demo_failure_report,
        }

    push_done = bool(
        m2_demo_publish_report.get("pushed")
        or (
            m2_demo_publish_report.get("enabled")
            and not m2_demo_publish_report.get("error")
            and m2_demo_publish_report.get("staged_files") == []
        )
    )
    comparison_required = bool(m2_comparison_report.get("enabled"))
    comparison_written = bool(m2_comparison_report.get("written")) if comparison_required else True
    comparison_passed = m2_comparison_report.get("status") == "pass" if comparison_required else True
    open_world_required = bool(open_world_validation_report.get("enabled"))
    open_world_passed = bool(open_world_validation_report.get("ready")) if open_world_required else True
    runner_succeeded = bool(report_ready and int(completed.returncode) == 0)
    completion_checks = {
        "m2_report_written": bool(report_ready),
        "m2_runner_succeeded": bool(runner_succeeded),
        "git_push": bool(push_done),
        "m2_comparison_written": bool(comparison_written),
        "m2_comparison_passed": bool(comparison_passed),
        "open_world_router_validation_passed": bool(open_world_passed),
    }
    m2_completion_report = {
        "ready": bool(
            report_ready
            and runner_succeeded
            and push_done
            and comparison_written
            and comparison_passed
            and open_world_passed
        ),
        "checks": completion_checks,
        "missing": [
            name
            for name in (
                "m2_report_written",
                "m2_runner_succeeded",
                "git_push",
                "m2_comparison_written",
                "m2_comparison_passed",
                "open_world_router_validation_passed",
            )
            if not completion_checks[name]
        ],
        "soft_missing": [],
        "comparison": m2_comparison_report,
        "open_world_router_validation": open_world_validation_report,
    }
    print(f"[COLAB] M2 completion checks -> {m2_completion_report['checks']}")
    m2_demo_disconnect_report = maybe_auto_disconnect_colab_runtime(
        enabled=bool(M2_AUTO_DISCONNECT_RUNTIME),
        grace_period_sec=M2_AUTO_DISCONNECT_GRACE_SECONDS,
        completion_report=m2_completion_report,
        print_fn=print,
    )

m2_demo_result
