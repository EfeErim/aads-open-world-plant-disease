"""Notebook 8 M2 full-demo settings and run-state setup."""

from __future__ import annotations

from scripts.notebook_helpers.nb8_m2_run_state import (
    apply_m2_run_state,
    format_m2_run_state_message,
    has_m2_run_state_operator_override,
    load_m2_run_state_config,
    operator_override_names,
    validate_balanced_manifest_request,
)


def load_m2_cell_settings(notebook_globals: dict[str, object]) -> dict[str, object]:
    initial_global_names = set(notebook_globals)
    settings: dict[str, object] = {}

    m2_auto_apply_run_state = bool(notebook_globals.get("M2_AUTO_APPLY_RUN_STATE", True))
    m2_force_run_state = bool(notebook_globals.get("M2_FORCE_RUN_STATE", True))
    m2_run_state_config = str(notebook_globals.get("M2_RUN_STATE_CONFIG", "docs/notebook8_m2_run_state.json"))
    m2_run_full_demo = bool(notebook_globals.get("M2_RUN_FULL_DEMO", True))
    m2_open_world_only = bool(notebook_globals.get("M2_OPEN_WORLD_ONLY", False))
    m2_demo_manifest = str(
        notebook_globals.get(
            "M2_DEMO_MANIFEST",
            "docs/demo_assets/m2_full_image_set/manifests/m2_balanced_80_run_manifest.csv",
        )
    )
    m2_run_problem_only_demo = bool(notebook_globals.get("M2_RUN_PROBLEM_ONLY_DEMO", False))
    m2_problem_only_manifest = str(
        notebook_globals.get(
            "M2_PROBLEM_ONLY_MANIFEST",
            "docs/demo_assets/m2_problem_only_manifests/20260628T113313Z_router_failures.csv",
        )
    )
    m2_problem_only_calibration_manifest = str(
        notebook_globals.get(
            "M2_PROBLEM_ONLY_CALIBRATION_MANIFEST",
            "docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv",
        )
    )
    m2_problem_only_comparison_baseline = str(
        notebook_globals.get("M2_PROBLEM_ONLY_COMPARISON_BASELINE", "") or ""
    )
    m2_demo_output = str(notebook_globals.get("M2_DEMO_OUTPUT", ".runtime_tmp/m2_demo_checklist_run.json"))
    m2_demo_markdown_output = str(
        notebook_globals.get("M2_DEMO_MARKDOWN_OUTPUT", ".runtime_tmp/m2_demo_checklist_run.md")
    )
    m2_analysis_output = str(notebook_globals.get("M2_ANALYSIS_OUTPUT", ".runtime_tmp/analysis_summary.json"))
    m2_analysis_markdown_output = str(
        notebook_globals.get("M2_ANALYSIS_MARKDOWN_OUTPUT", ".runtime_tmp/analysis_summary.md")
    )
    m2_demo_limit = notebook_globals.get("M2_DEMO_LIMIT", None)
    m2_batch_size = int(notebook_globals.get("M2_BATCH_SIZE", 4))
    m2_adapter_batch_size = int(notebook_globals.get("M2_ADAPTER_BATCH_SIZE", 2))
    m2_handoff_cache = str(
        notebook_globals.get("M2_HANDOFF_CACHE", ".runtime_tmp/m2_balanced_80_handoff_cache.json")
    )
    m2_refresh_handoff_cache = bool(notebook_globals.get("M2_REFRESH_HANDOFF_CACHE", True))
    m2_reuse_existing_prototype_calibration = bool(
        notebook_globals.get("M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION", True)
    )
    m2_stop_on_dependency_blocker = bool(notebook_globals.get("M2_STOP_ON_DEPENDENCY_BLOCKER", True))
    m2_auto_push_results = bool(notebook_globals.get("M2_AUTO_PUSH_RESULTS", True))
    m2_auto_push_remote_name = str(notebook_globals.get("M2_AUTO_PUSH_REMOTE_NAME", "origin"))
    m2_auto_push_branch = str(notebook_globals.get("M2_AUTO_PUSH_BRANCH", "master") or "").strip() or None
    m2_repo_results_root = str(notebook_globals.get("M2_REPO_RESULTS_ROOT", "docs/demo_results/m2"))
    m2_comparison_baseline = str(notebook_globals.get("M2_COMPARISON_BASELINE", "") or "")
    m2_run_open_world_router_validation = bool(
        notebook_globals.get("M2_RUN_OPEN_WORLD_ROUTER_VALIDATION", False)
    )
    m2_open_world_supported_manifest = str(
        notebook_globals.get(
            "M2_OPEN_WORLD_SUPPORTED_MANIFEST",
            "docs/demo_assets/m2_full_image_set/manifests/m2_balanced_80_run_manifest.csv",
        )
    )
    m2_open_world_manifest = str(
        notebook_globals.get(
            "M2_OPEN_WORLD_MANIFEST",
            "docs/demo_assets/open_world_router/manifests/m2_open_world_router_manifest.csv",
        )
    )
    m2_open_world_output_root = str(
        notebook_globals.get("M2_OPEN_WORLD_OUTPUT_ROOT", "docs/demo_results/router_open_world")
    )
    m2_open_world_handoff_cache = str(
        notebook_globals.get("M2_OPEN_WORLD_HANDOFF_CACHE", ".runtime_tmp/router_open_world_handoff_cache.json")
    )
    m2_open_world_baseline_summary = str(
        notebook_globals.get("M2_OPEN_WORLD_BASELINE_SUMMARY", m2_comparison_baseline) or ""
    )
    m2_open_world_prototype_artifact_dir = str(
        notebook_globals.get("M2_OPEN_WORLD_PROTOTYPE_ARTIFACT_DIR", "docs/demo_results/m2/20260630T192242Z")
        or ""
    )
    m2_open_world_min_rows = int(notebook_globals.get("M2_OPEN_WORLD_MIN_ROWS", 300))
    m2_open_world_min_supported_route_coverage = float(
        notebook_globals.get("M2_OPEN_WORLD_MIN_SUPPORTED_ROUTE_COVERAGE", 0.80)
    )
    m2_open_world_require_latency_baseline = bool(
        notebook_globals.get("M2_OPEN_WORLD_REQUIRE_LATENCY_BASELINE", True)
    )
    m2_open_world_fail_on_not_ready = bool(notebook_globals.get("M2_OPEN_WORLD_FAIL_ON_NOT_READY", True))
    m2_auto_disconnect_runtime = bool(notebook_globals.get("M2_AUTO_DISCONNECT_RUNTIME", True))
    m2_auto_disconnect_grace_seconds = float(notebook_globals.get("M2_AUTO_DISCONNECT_GRACE_SECONDS", 20))
    m2_pytorch_cuda_alloc_conf = str(
        notebook_globals.get("M2_PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True") or ""
    ).strip()

    run_state_operator_override_names = operator_override_names(initial_global_names)
    m2_run_state: dict[str, object] | None = None
    if m2_auto_apply_run_state:
        m2_run_state = load_m2_run_state_config(m2_run_state_config)
        if m2_run_state:
            run_settings = apply_m2_run_state(
                {
                    "M2_RUN_PROBLEM_ONLY_DEMO": m2_run_problem_only_demo,
                    "M2_RUN_FULL_DEMO": m2_run_full_demo,
                    "M2_OPEN_WORLD_ONLY": m2_open_world_only,
                    "M2_REFRESH_HANDOFF_CACHE": m2_refresh_handoff_cache,
                    "M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION": m2_reuse_existing_prototype_calibration,
                    "M2_BATCH_SIZE": m2_batch_size,
                    "M2_ADAPTER_BATCH_SIZE": m2_adapter_batch_size,
                    "M2_DEMO_MANIFEST": m2_demo_manifest,
                    "M2_HANDOFF_CACHE": m2_handoff_cache,
                    "M2_PROBLEM_ONLY_MANIFEST": m2_problem_only_manifest,
                    "M2_PROBLEM_ONLY_CALIBRATION_MANIFEST": m2_problem_only_calibration_manifest,
                    "M2_PROBLEM_ONLY_COMPARISON_BASELINE": m2_problem_only_comparison_baseline,
                    "M2_COMPARISON_BASELINE": m2_comparison_baseline,
                    "M2_RUN_OPEN_WORLD_ROUTER_VALIDATION": m2_run_open_world_router_validation,
                    "M2_OPEN_WORLD_BASELINE_SUMMARY": m2_open_world_baseline_summary,
                    "M2_OPEN_WORLD_PROTOTYPE_ARTIFACT_DIR": m2_open_world_prototype_artifact_dir,
                },
                m2_run_state,
                force_run_state=m2_force_run_state,
                operator_overrides=run_state_operator_override_names,
            )
            m2_run_problem_only_demo = run_settings["M2_RUN_PROBLEM_ONLY_DEMO"]
            m2_run_full_demo = run_settings["M2_RUN_FULL_DEMO"]
            m2_open_world_only = run_settings["M2_OPEN_WORLD_ONLY"]
            m2_refresh_handoff_cache = run_settings["M2_REFRESH_HANDOFF_CACHE"]
            m2_reuse_existing_prototype_calibration = run_settings["M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION"]
            m2_batch_size = run_settings["M2_BATCH_SIZE"]
            m2_adapter_batch_size = run_settings["M2_ADAPTER_BATCH_SIZE"]
            m2_demo_manifest = run_settings["M2_DEMO_MANIFEST"]
            m2_handoff_cache = run_settings["M2_HANDOFF_CACHE"]
            m2_problem_only_manifest = run_settings["M2_PROBLEM_ONLY_MANIFEST"]
            m2_problem_only_calibration_manifest = run_settings["M2_PROBLEM_ONLY_CALIBRATION_MANIFEST"]
            m2_problem_only_comparison_baseline = run_settings["M2_PROBLEM_ONLY_COMPARISON_BASELINE"]
            m2_comparison_baseline = run_settings["M2_COMPARISON_BASELINE"]
            m2_run_open_world_router_validation = run_settings["M2_RUN_OPEN_WORLD_ROUTER_VALIDATION"]
            m2_open_world_baseline_summary = run_settings["M2_OPEN_WORLD_BASELINE_SUMMARY"]
            m2_open_world_prototype_artifact_dir = run_settings["M2_OPEN_WORLD_PROTOTYPE_ARTIFACT_DIR"]
            print(
                format_m2_run_state_message(
                    run_settings,
                    force_run_state=m2_force_run_state,
                    operator_overrides=run_state_operator_override_names,
                )
            )
            validate_balanced_manifest_request(run_settings, m2_run_state)

    device = str(notebook_globals.get("DEVICE", "cuda"))
    config_env = str(notebook_globals.get("CONFIG_ENV", "colab"))
    m2_enable_prototype_reconciler = bool(notebook_globals.get("M2_ENABLE_PROTOTYPE_RECONCILER", True))
    m2_auto_build_prototypes = bool(notebook_globals.get("M2_AUTO_BUILD_PROTOTYPES", True))
    m2_prototype_run_id = str(notebook_globals.get("M2_PROTOTYPE_RUN_ID", "") or "")
    m2_prototype_embedding_backend = str(
        notebook_globals.get("M2_PROTOTYPE_EMBEDDING_BACKEND", "bioclip_open_clip")
    )
    m2_prototype_embedding_model_id = str(
        notebook_globals.get("M2_PROTOTYPE_EMBEDDING_MODEL_ID", "imageomics/bioclip-2.5-vith14")
    )
    m2_prototype_embedding_device = str(notebook_globals.get("M2_PROTOTYPE_EMBEDDING_DEVICE", device))
    m2_reuse_existing_prototypes = bool(notebook_globals.get("M2_REUSE_EXISTING_PROTOTYPES", True))
    m2_prototype_max_images_per_class = notebook_globals.get("M2_PROTOTYPE_MAX_IMAGES_PER_CLASS", 50)
    m2_prototype_curation_root = str(notebook_globals.get("M2_PROTOTYPE_CURATION_ROOT", "") or "")
    if (
        m2_auto_apply_run_state
        and m2_run_state
        and not has_m2_run_state_operator_override(
            "M2_PROTOTYPE_CURATION_ROOT",
            force_run_state=m2_force_run_state,
            operator_overrides=run_state_operator_override_names,
        )
    ):
        m2_prototype_curation_root = str(
            m2_run_state.get("m2_prototype_curation_root") or m2_prototype_curation_root
        )
    m2_prototype_bank = str(notebook_globals.get("M2_PROTOTYPE_BANK", "") or "")
    m2_taxonomy_registry = str(notebook_globals.get("M2_TAXONOMY_REGISTRY", "") or "")
    m2_prototype_min_similarity = notebook_globals.get("M2_PROTOTYPE_MIN_SIMILARITY", None)
    m2_prototype_min_margin = notebook_globals.get("M2_PROTOTYPE_MIN_MARGIN", None)
    m2_prototype_min_negative_gap = notebook_globals.get("M2_PROTOTYPE_MIN_NEGATIVE_GAP", None)
    m2_auto_calibrate_prototype_reconciler = bool(
        notebook_globals.get(
            "M2_AUTO_CALIBRATE_PROTOTYPE_RECONCILER",
            notebook_globals.get("M2_AUTO_CALIBRATE_PROTOTYPES", True),
        )
    )
    m2_require_calibrated_prototype_policy = bool(
        notebook_globals.get("M2_REQUIRE_CALIBRATED_PROTOTYPE_POLICY", True)
    )
    m2_prototype_calibration_output = str(
        notebook_globals.get("M2_PROTOTYPE_CALIBRATION_OUTPUT", ".runtime_tmp/router_prototype_calibration.json")
    )
    m2_prototype_calibration_limit = notebook_globals.get("M2_PROTOTYPE_CALIBRATION_LIMIT", None)
    m2_prototype_calibration_min_precision = float(
        notebook_globals.get("M2_PROTOTYPE_CALIBRATION_MIN_PRECISION", 0.985)
    )
    m2_prototype_calibration_min_coverage = float(
        notebook_globals.get("M2_PROTOTYPE_CALIBRATION_MIN_COVERAGE", 0.80)
    )
    m2_prototype_calibration_max_negative_false_accepts = int(
        notebook_globals.get("M2_PROTOTYPE_CALIBRATION_MAX_NEGATIVE_FALSE_ACCEPTS", 0)
    )
    m2_prototype_calibration_max_negative_false_accept_rate = float(
        notebook_globals.get("M2_PROTOTYPE_CALIBRATION_MAX_NEGATIVE_FALSE_ACCEPT_RATE", 0.05)
    )
    m2_prototype_target_min_precision = float(notebook_globals.get("M2_PROTOTYPE_TARGET_MIN_PRECISION", 0.98))
    m2_prototype_target_max_supported_wrong = notebook_globals.get("M2_PROTOTYPE_TARGET_MAX_SUPPORTED_WRONG", 1)
    m2_prototype_target_class_min_accepted = int(
        notebook_globals.get("M2_PROTOTYPE_TARGET_CLASS_MIN_ACCEPTED", 5)
    )
    m2_prototype_target_max_cross_part_supported_wrong = int(
        notebook_globals.get("M2_PROTOTYPE_TARGET_MAX_CROSS_PART_SUPPORTED_WRONG", 0)
    )
    m2_prototype_similarity_grid = str(
        notebook_globals.get("M2_PROTOTYPE_SIMILARITY_GRID", "0.20,0.30,0.40,0.50,0.60,0.70")
    )
    m2_prototype_margin_grid = str(
        notebook_globals.get("M2_PROTOTYPE_MARGIN_GRID", "0.00,0.02,0.04,0.06,0.08,0.10")
    )
    m2_prototype_negative_gap_grid = str(
        notebook_globals.get("M2_PROTOTYPE_NEGATIVE_GAP_GRID", "0.00,0.02,0.04,0.06,0.08,0.10")
    )
    m2_allow_non_plant_false_accepts = bool(notebook_globals.get("M2_ALLOW_NON_PLANT_FALSE_ACCEPTS", False))
    m2_prototype_target_policy_negative_mode = str(
        notebook_globals.get("M2_PROTOTYPE_TARGET_POLICY_NEGATIVE_MODE", "none")
    ).strip().lower()
    if m2_prototype_target_policy_negative_mode not in {"all", "none"}:
        m2_prototype_target_policy_negative_mode = "none"

    settings.update(
        {
            "_M2_INITIAL_GLOBAL_NAMES": initial_global_names,
            "_M2_RUN_STATE_OPERATOR_OVERRIDE_NAMES": run_state_operator_override_names,
            "_m2_run_state": m2_run_state,
            "M2_AUTO_APPLY_RUN_STATE": m2_auto_apply_run_state,
            "M2_FORCE_RUN_STATE": m2_force_run_state,
            "M2_RUN_STATE_CONFIG": m2_run_state_config,
            "M2_RUN_FULL_DEMO": m2_run_full_demo,
            "M2_OPEN_WORLD_ONLY": m2_open_world_only,
            "M2_DEMO_MANIFEST": m2_demo_manifest,
            "M2_RUN_PROBLEM_ONLY_DEMO": m2_run_problem_only_demo,
            "M2_PROBLEM_ONLY_MANIFEST": m2_problem_only_manifest,
            "M2_PROBLEM_ONLY_CALIBRATION_MANIFEST": m2_problem_only_calibration_manifest,
            "M2_PROBLEM_ONLY_COMPARISON_BASELINE": m2_problem_only_comparison_baseline,
            "M2_DEMO_OUTPUT": m2_demo_output,
            "M2_DEMO_MARKDOWN_OUTPUT": m2_demo_markdown_output,
            "M2_ANALYSIS_OUTPUT": m2_analysis_output,
            "M2_ANALYSIS_MARKDOWN_OUTPUT": m2_analysis_markdown_output,
            "M2_DEMO_LIMIT": m2_demo_limit,
            "M2_BATCH_SIZE": m2_batch_size,
            "M2_ADAPTER_BATCH_SIZE": m2_adapter_batch_size,
            "M2_HANDOFF_CACHE": m2_handoff_cache,
            "M2_REFRESH_HANDOFF_CACHE": m2_refresh_handoff_cache,
            "M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION": m2_reuse_existing_prototype_calibration,
            "M2_STOP_ON_DEPENDENCY_BLOCKER": m2_stop_on_dependency_blocker,
            "M2_AUTO_PUSH_RESULTS": m2_auto_push_results,
            "M2_AUTO_PUSH_REMOTE_NAME": m2_auto_push_remote_name,
            "M2_AUTO_PUSH_BRANCH": m2_auto_push_branch,
            "M2_REPO_RESULTS_ROOT": m2_repo_results_root,
            "M2_COMPARISON_BASELINE": m2_comparison_baseline,
            "M2_RUN_OPEN_WORLD_ROUTER_VALIDATION": m2_run_open_world_router_validation,
            "M2_OPEN_WORLD_SUPPORTED_MANIFEST": m2_open_world_supported_manifest,
            "M2_OPEN_WORLD_MANIFEST": m2_open_world_manifest,
            "M2_OPEN_WORLD_OUTPUT_ROOT": m2_open_world_output_root,
            "M2_OPEN_WORLD_HANDOFF_CACHE": m2_open_world_handoff_cache,
            "M2_OPEN_WORLD_BASELINE_SUMMARY": m2_open_world_baseline_summary,
            "M2_OPEN_WORLD_PROTOTYPE_ARTIFACT_DIR": m2_open_world_prototype_artifact_dir,
            "M2_OPEN_WORLD_MIN_ROWS": m2_open_world_min_rows,
            "M2_OPEN_WORLD_MIN_SUPPORTED_ROUTE_COVERAGE": m2_open_world_min_supported_route_coverage,
            "M2_OPEN_WORLD_REQUIRE_LATENCY_BASELINE": m2_open_world_require_latency_baseline,
            "M2_OPEN_WORLD_FAIL_ON_NOT_READY": m2_open_world_fail_on_not_ready,
            "M2_AUTO_DISCONNECT_RUNTIME": m2_auto_disconnect_runtime,
            "M2_AUTO_DISCONNECT_GRACE_SECONDS": m2_auto_disconnect_grace_seconds,
            "M2_PYTORCH_CUDA_ALLOC_CONF": m2_pytorch_cuda_alloc_conf,
            "DEVICE": device,
            "CONFIG_ENV": config_env,
            "M2_ENABLE_PROTOTYPE_RECONCILER": m2_enable_prototype_reconciler,
            "M2_AUTO_BUILD_PROTOTYPES": m2_auto_build_prototypes,
            "M2_PROTOTYPE_RUN_ID": m2_prototype_run_id,
            "M2_PROTOTYPE_EMBEDDING_BACKEND": m2_prototype_embedding_backend,
            "M2_PROTOTYPE_EMBEDDING_MODEL_ID": m2_prototype_embedding_model_id,
            "M2_PROTOTYPE_EMBEDDING_DEVICE": m2_prototype_embedding_device,
            "M2_REUSE_EXISTING_PROTOTYPES": m2_reuse_existing_prototypes,
            "M2_PROTOTYPE_MAX_IMAGES_PER_CLASS": m2_prototype_max_images_per_class,
            "M2_PROTOTYPE_CURATION_ROOT": m2_prototype_curation_root,
            "M2_PROTOTYPE_BANK": m2_prototype_bank,
            "M2_TAXONOMY_REGISTRY": m2_taxonomy_registry,
            "M2_PROTOTYPE_MIN_SIMILARITY": m2_prototype_min_similarity,
            "M2_PROTOTYPE_MIN_MARGIN": m2_prototype_min_margin,
            "M2_PROTOTYPE_MIN_NEGATIVE_GAP": m2_prototype_min_negative_gap,
            "M2_AUTO_CALIBRATE_PROTOTYPE_RECONCILER": m2_auto_calibrate_prototype_reconciler,
            "M2_REQUIRE_CALIBRATED_PROTOTYPE_POLICY": m2_require_calibrated_prototype_policy,
            "M2_PROTOTYPE_CALIBRATION_OUTPUT": m2_prototype_calibration_output,
            "M2_PROTOTYPE_CALIBRATION_LIMIT": m2_prototype_calibration_limit,
            "M2_PROTOTYPE_CALIBRATION_MIN_PRECISION": m2_prototype_calibration_min_precision,
            "M2_PROTOTYPE_CALIBRATION_MIN_COVERAGE": m2_prototype_calibration_min_coverage,
            "M2_PROTOTYPE_CALIBRATION_MAX_NEGATIVE_FALSE_ACCEPTS": (
                m2_prototype_calibration_max_negative_false_accepts
            ),
            "M2_PROTOTYPE_CALIBRATION_MAX_NEGATIVE_FALSE_ACCEPT_RATE": (
                m2_prototype_calibration_max_negative_false_accept_rate
            ),
            "M2_PROTOTYPE_TARGET_MIN_PRECISION": m2_prototype_target_min_precision,
            "M2_PROTOTYPE_TARGET_MAX_SUPPORTED_WRONG": m2_prototype_target_max_supported_wrong,
            "M2_PROTOTYPE_TARGET_CLASS_MIN_ACCEPTED": m2_prototype_target_class_min_accepted,
            "M2_PROTOTYPE_TARGET_MAX_CROSS_PART_SUPPORTED_WRONG": (
                m2_prototype_target_max_cross_part_supported_wrong
            ),
            "M2_PROTOTYPE_SIMILARITY_GRID": m2_prototype_similarity_grid,
            "M2_PROTOTYPE_MARGIN_GRID": m2_prototype_margin_grid,
            "M2_PROTOTYPE_NEGATIVE_GAP_GRID": m2_prototype_negative_gap_grid,
            "M2_ALLOW_NON_PLANT_FALSE_ACCEPTS": m2_allow_non_plant_false_accepts,
            "M2_PROTOTYPE_TARGET_POLICY_NEGATIVE_MODE": m2_prototype_target_policy_negative_mode,
        }
    )
    return settings
