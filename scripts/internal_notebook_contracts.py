"""Internal-maintenance and report-only notebook contract checks."""

from __future__ import annotations

from scripts.notebook_validation_support import (
    ROOT,
    _assert_clone_bootstrap_contract,
    _assert_code_cells_compile,
    _assert_contains,
    _assert_not_contains,
    _assert_repo_bootstrap_contract,
    _assert_update_check_contract,
    _load_notebook_sources,
)


def test_roi_ablation_notebook_contract() -> None:
    from scripts.colab_roi_ablation import ABLATION_CONFIGS

    expected = {
        "16_ablation_dual_view_inference.ipynb": "dual_view_inference",
    }
    assert set(expected.values()).issubset(ABLATION_CONFIGS), "ROI ablation configs are missing notebook keys"

    for notebook_name, ablation_name in expected.items():
        sources = _load_notebook_sources(notebook_name)
        _assert_code_cells_compile(sources, notebook_name)
        _assert_contains(
            sources.first_code_source,
            "def _ensure_aads_repo_on_path():",
            f"{notebook_name} should bootstrap the repo before importing helpers: {{snippet}}",
        )
        _assert_contains(
            sources.first_code_source,
            "os.environ['AADS_GITHUB_RELEASE_READ_TOKEN'] = token",
            f"{notebook_name} must export the Colab dataset Release secret under the canonical runtime token name.",
        )
        _assert_contains(
            sources.full_source,
            f"ABLATION_NAME = '{ablation_name}'",
            f"{notebook_name} should pin exactly one ablation condition: {{snippet}}",
        )
        _assert_contains(
            sources.full_source,
            "commit_and_push_ablation_results",
            f"{notebook_name} should commit and push its repo-visible output folder: {{snippet}}",
        )
        _assert_contains(
            sources.full_source,
            "run_dual_view_inference_targets(",
            f"{notebook_name} should run the shared multi-target dual-view inference helper: {{snippet}}",
        )
        _assert_contains(
            sources.full_source,
            "discover_dual_view_targets(",
            f"{notebook_name} should discover matching datasets/adapters instead of pinning tomato__fruit: {{snippet}}",
        )
        _assert_contains(
            sources.full_source,
            "TARGETS = []",
            f"{notebook_name} should allow explicit multi-adapter target overrides: {{snippet}}",
        )
        _assert_contains(
            sources.full_source,
            "REQUIRE_SEMANTIC_ROI_MATCH = True",
            f"{notebook_name} should gate ROI by adapter crop/part semantics: {{snippet}}",
        )
        _assert_contains(
            sources.full_source,
            "FULL_CONFIDENCE_REVIEW_THRESHOLD = 0.70",
            f"{notebook_name} should keep full image primary and use ROI as review evidence: {{snippet}}",
        )
        _assert_contains(
            sources.full_source,
            "TARGET_ROI_BACKEND = 'router_then_grounding_dino'",
            f"{notebook_name} should enable target-aware Grounding DINO ROI fallback: {{snippet}}",
        )
        _assert_not_contains(
            sources.full_source,
            "GROUNDING_DINO_PROMPTS = ['tomato fruit.",
            f"{notebook_name} should not hard-code tomato-only Grounding DINO prompts: {{snippet}}",
        )
        _assert_contains(
            sources.full_source,
            "GROUNDING_DINO_BOX_THRESHOLD = 0.15",
            f"{notebook_name} should use the low-threshold detector sweep default: {{snippet}}",
        )
        _assert_contains(
            sources.full_source,
            "AUTO_DISCONNECT_RUNTIME = True",
            f"{notebook_name} should auto-disconnect after successful report export and push: {{snippet}}",
        )
        _assert_contains(
            sources.full_source,
            "from scripts.colab_notebook_helpers import maybe_auto_disconnect_colab_runtime",
            f"{notebook_name} should use the maintained Colab auto-disconnect helper: {{snippet}}",
        )
        _assert_contains(
            sources.full_source,
            "'git_push': push_ok",
            f"{notebook_name} should require a successful push before auto-disconnect: {{snippet}}",
        )
        _assert_contains(
            sources.full_source,
            "maybe_auto_disconnect_colab_runtime(",
            f"{notebook_name} should request Colab runtime disconnect at the end: {{snippet}}",
        )


def test_adapter_smoke_notebook_surface() -> None:
    from scripts.colab_adapter_smoke_test import (
        discover_adapter_candidates,
        load_adapter_summary,
        predict_image_folder,
        predict_single_image,
    )

    assert callable(discover_adapter_candidates)
    assert callable(load_adapter_summary)
    assert callable(predict_single_image)
    assert callable(predict_image_folder)


def test_adapter_smoke_notebook_bootstrap_contract() -> None:
    sources = _load_notebook_sources("3_validate_exported_adapter_directly.ipynb")

    _assert_repo_bootstrap_contract(sources.first_code_source, "Notebook 3")

    assert "Path('/content/drive/MyDrive/aads_ulora')" not in sources.full_source
    assert "SEARCH_ROOTS = [" in sources.full_source
    assert "INCLUDE_RUN_ADAPTERS = False" in sources.full_source
    assert "ROOT / 'outputs' / 'colab_notebook_training'" in sources.full_source
    assert "SEARCH_ROOTS.append(ROOT / 'runs')" in sources.full_source


def test_simple_adapter_smoke_notebook_bootstrap_contract() -> None:
    sources = _load_notebook_sources("4_simple_direct_adapter_test_ui.ipynb")

    _assert_code_cells_compile(sources, "Notebook 4")
    _assert_clone_bootstrap_contract(sources.first_code_source, "Notebook 4")
    assert "collect_notebook_access_report" in sources.full_source
    assert "install_colab_requirements(ROOT / 'colab_notebooks' / 'requirements_colab.txt', running_in_colab())" in sources.full_source
    assert "print_notebook_access_report" in sources.full_source
    assert "from scripts import colab_simple_adapter_smoke_ui" in sources.full_source
    assert "importlib.reload(colab_simple_adapter_smoke_ui)" in sources.full_source
    assert "['reset', '--hard', f'origin/{REPO_REF}']" in sources.full_source
    assert "['pull', '--ff-only', 'origin', REPO_REF]" not in sources.full_source
    assert "ROOT / 'outputs' / 'colab_notebook_training'" in sources.full_source
    assert "ROOT / 'models' / 'adapters'" in sources.full_source
    assert "ROOT / 'runs'" not in sources.full_source
    assert "show_all_adapters=True" not in sources.full_source
    assert "show_mirror_adapters=True" not in sources.full_source
    assert "launch_simple_adapter_smoke_ui(ROOT, search_roots=SEARCH_ROOTS)" in sources.full_source

def test_batch_training_notebook_contract() -> None:
    sources = _load_notebook_sources("6_train_all_continual_sd_lora_adapters.ipynb")

    _assert_update_check_contract(
        sources.first_code_source,
        "Notebook 6",
        forbid_drive_bootstrap=True,
    )
    for snippet in (
        'NOTEBOOK_NAME = "6_train_all_continual_sd_lora_adapters.ipynb"',
        'NOTEBOOK_FILENAME = "6_train_all_continual_sd_lora_adapters.executed.ipynb"',
        "NB6_AUTO_DISCONNECT_RUNTIME = True",
        "NB6_AUTO_DISCONNECT_GRACE_SECONDS = 20",
        "DATASET_RELEASE_TAG = 'aads-dataset-v1.0.0'",
        "os.environ['AADS_GITHUB_RELEASE_READ_TOKEN'] = token",
        '"AUTO_DISCONNECT_RUNTIME": False',
        '"AUTO_PUSH_TO_GITHUB": True',
        "NB6_MANUAL_PARAM_OVERRIDES = {}",
        "NB6_LOW_RESOURCE_MODE = True",
        '"ENABLE_BAYESIAN_OPTIMIZATION": False',
        '"NUM_WORKERS": 2',
        '"PREFETCH": 2',
        '"USE_CACHE": False',
        "from scripts.colab_notebook_helpers import cleanup_notebook_training_state",
        "from src.data.dataset_release_runtime import DatasetReleaseAccessBlocker",
        "from src.pipeline.adapter_release import resolve_token",
        'if not str(resolve_token(write=False) or "").strip():',
        "Add AADS_GITHUB_RELEASE_READ_TOKEN under Colab > Secrets",
        "[NB6][PREFLIGHT] Private dataset Release read token is ready.",
        "NB6_RESEARCH_ALLOW_UNDER_MIN_ADAPTERS = []",
        "from scripts.notebook_helpers.adapter_recommendations import get_adapter_recs",
        "ADAPTER_RECS = get_adapter_recs()",
        "NB6_ADAPTER_SEQUENCE = [",
        "for index, adapter_key in enumerate(NB6_ADAPTER_SEQUENCE, start=1):",
        "MANUAL_PARAM_OVERRIDES = dict(NB6_LOW_RESOURCE_OVERRIDES if NB6_LOW_RESOURCE_MODE else {})",
        "MANUAL_PARAM_OVERRIDES.update(NB6_MANUAL_PARAM_OVERRIDES.get(adapter_key, {}))",
        "cleanup_notebook_training_state(globals(), label=adapter_key)",
        'adapter_rec["allow_under_min"] = True',
        '"research_under_min_bypass": bool(research_under_min_bypass)',
        "from scripts.colab_notebook_helpers import maybe_auto_disconnect_colab_runtime",
        '"batch_loop_completed": True',
        '"all_adapters_attempted": NB6_ALL_ADAPTERS_ATTEMPTED',
        '"all_adapters_succeeded": NB6_ALL_ADAPTERS_SUCCEEDED',
        '"ready": NB6_ALL_ADAPTERS_SUCCEEDED',
        '"missing": NB6_FAILED_ADAPTERS',
        "enabled=bool(NB6_AUTO_DISCONNECT_RUNTIME)",
    ):
        _assert_contains(
            sources.full_source,
            snippet,
            "Notebook 6 batch surface is missing required batch-training contract: {snippet}",
        )
    if sources.full_source.index("[NB6][PREFLIGHT]") > sources.full_source.index(
        "for index, adapter_key in enumerate(NB6_ADAPTER_SEQUENCE, start=1):"
    ):
        raise AssertionError("Notebook 6 dataset Release access preflight must run before the eight-target loop.")
    for stale_snippet in (
        '"grape__fruit": {"crop": "grape"',
        '"strawberry__leaf": {"crop": "strawberry"',
        '"tomato__leaf": {"crop": "tomato"',
    ):
        _assert_not_contains(
            sources.full_source,
            stale_snippet,
            "Notebook 6 should not embed adapter recommendation copies; use get_adapter_recs(): {snippet}",
        )
    for script_name in (
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
        _assert_contains(
            sources.full_source,
            f"run_cell_script('{script_name}', globals())",
            "Notebook 6 should execute the maintained Notebook 2 cell script sequence: {snippet}",
        )
    calibration_source = (ROOT / "scripts" / "notebook_cells" / "nb2_cell10_ood_calibration.py").read_text(
        encoding="utf-8"
    )
    final_evaluation_source = (
        ROOT / "scripts" / "notebook_cells" / "nb2_cell12_final_evaluation.py"
    ).read_text(encoding="utf-8")
    for snippet in ("calibrate_notebook_ood_policy(", 'STATE["ood_policy_selection"]'):
        _assert_contains(
            calibration_source,
            snippet,
            "Notebook 6 calibration must preserve automatic OOD-dev policy selection: {snippet}",
        )
    for snippet in (
        'ood_dev_loader = loaders.get("ood_dev")',
        'ood_test_loader = loaders.get("ood")',
        "val_loader,\n            ood_dev_loader,",
        "test_loader,\n            ood_test_loader,",
        '"ood_primary_score_selection_source": selection_source',
    ):
        _assert_contains(
            final_evaluation_source,
            snippet,
            "Notebook 6 final evaluation must keep OOD dev and test evidence separate: {snippet}",
        )


def test_adapter_ood_oe_recovery_notebook_contract() -> None:
    sources = _load_notebook_sources("17_adapter_ood_oe_recovery.ipynb")
    _assert_contains(
        sources.first_code_source,
        "os.environ['AADS_GITHUB_RELEASE_READ_TOKEN'] = token",
        "Notebook 17 must export the Colab dataset Release secret under the canonical runtime token name.",
    )
    _assert_update_check_contract(
        sources.first_code_source,
        "Notebook 17",
        forbid_drive_bootstrap=True,
    )
    for snippet in (
        "NOTEBOOK_NAME = '17_adapter_ood_oe_recovery.ipynb'",
        "RECOVERY_CAMPAIGN_PATH = 'docs/architecture/adapter_ood_oe_recovery_campaign.json'",
        "RECOVERY_CONTINUE_ON_ERROR = True",
        "RECOVERY_AUTO_PUSH_TO_GITHUB = True",
        "RECOVERY_AUTO_DISCONNECT_RUNTIME = True",
        "RECOVERY_LOW_RESOURCE_MODE = True",
        "RECOVERY_TARGETS = []",
        "RECOVERY_RESUME_FROM_LEDGER = True",
        "RECOVERY_USE_GITHUB_DATASET_RELEASE = True",
        "RECOVERY_LOCAL_DATASET_ROOT = 'data/prepared_runtime_datasets'",
        "DATASET_RELEASE_REPOSITORY = 'EfeErim/aads-open-world-plant-disease'",
        "DATASET_RELEASE_TAG = 'aads-dataset-v1.0.0'",
        "RECOVERY_EVIDENCE_MANIFEST_RELATIVE_PATH = 'adapter_ood_oe_evidence_manifest.csv'",
        "RECOVERY_MAX_COMPLETED_EXPERIMENTS = 0",
        "RECOVERY_MIN_FREE_DISK_GIB = 12.0",
        "RECOVERY_RECLAIM_FAILED_RUN_PAYLOADS = True",
        "run_cell_script('nb17_cell03_run_recovery.py', globals())",
        "'.runtime_tmp/dataset_release_cache'",
    ):
        _assert_contains(
            sources.full_source,
            snippet,
            "Notebook 17 recovery surface is missing required contract: {snippet}",
        )
    script_source = (
        ROOT / "scripts" / "notebook_cells" / "nb17_cell03_run_recovery.py"
    ).read_text(encoding="utf-8")
    for snippet in (
        "--fail-on-exact-overlap",
        "--fail-on-near-duplicate",
        "--file-path-prefix",
        "fetch_materialize_dataset_release(",
        'DATASET_SOURCE_KIND = "local_legacy"',
        'DATASET_SOURCE_KIND = "github_release"',
        '"v4_bounded_adapter_recovery_campaign"',
        "RECOVERY_SELECTION_ONLY",
        "adapter_behavioral_dev_report.json",
        "_run_locked_final_evaluation(",
        "build_campaign_lineage(",
        "load_campaign_ledger(",
        "resumable_experiment(",
        "experiment_gate(",
        "behavioral_acceptance_pass(",
        '"nb2_cell09_training.py"',
        '"nb2_cell10_ood_calibration.py"',
        '"nb2_cell12_final_evaluation.py"',
        "build_notebook_completion_report(",
        "_cleanup_runtime_state(",
        "require_recovery_disk_budget(",
        "reclaim_failed_run_payloads(",
        "_write_and_publish_recovery_report(",
        "RECOVERY_MAX_COMPLETED_EXPERIMENTS",
        "passed_target_count",
        'RECOVERY_DISCONNECT_REPORT["campaign_passed"]',
        'RECOVERY_DISCONNECT_REPORT["ready"] = True',
        "push_repo_paths_to_github(",
        'RECOVERY_COMPLETION_REPORT.get("publish_ok", False)',
    ):
        _assert_contains(
            script_source,
            snippet,
            "Notebook 17 recovery orchestrator is missing required gate: {snippet}",
        )


def test_router_calibration_notebook_contract() -> None:
    sources = _load_notebook_sources("5_calibrate_router_handoff_thresholds.ipynb")

    _assert_repo_bootstrap_contract(sources.first_code_source, "Notebook 5")

    assert "from scripts.evaluate_router_surface import discover_eval_samples, evaluate_router_surface" in sources.full_source
    assert "from scripts.calibrate_router_surface import calibrate_router_surface" in sources.full_source
    assert "ROUTER_EVAL_ROOT = 'data/router_eval'" in sources.full_source
    assert "HOLDOUT_EVAL_ROOT = 'data/router_eval_holdout'" in sources.full_source
    assert "RUN_BASELINE_EVAL = False" in sources.full_source
    assert "RUN_CALIBRATION = True" in sources.full_source
    assert "RUN_HOLDOUT_VALIDATION = True" in sources.full_source
    assert "CALIBRATION_STRATEGY = 'replay-thresholds'" in sources.full_source
    assert "CALIBRATION_PRESET = 'handoff'" in sources.full_source
    assert "COLLECT_INPUT_GUARD_SCORES = True" in sources.full_source
    assert "'input_guard_enabled=false,true'" in sources.full_source
    assert "Notebook 5 first cell started." in sources.full_source
    assert "['git', 'clone', '--depth', '1', '--progress'" in sources.full_source
    assert "ensure_router_dependencies_nb5" in sources.full_source
    assert "validate_router_candidate_overrides" in sources.full_source
    assert "run_cell_script('nb5_cell06_holdout_validation.py', globals())" in sources.full_source
    assert "run_cell_script('nb5_cell07_publish_results.py', globals())" in sources.full_source
    assert "PUBLISH_RESULTS_ROOT = 'runs/_index/router_calibration'" in sources.full_source
    assert "target_negative_false_accept_rate=TARGET_NEGATIVE_FALSE_ACCEPT_RATE" in sources.full_source
    assert "max_crop_accuracy_drop=MAX_CROP_ACCURACY_DROP" in sources.full_source
    assert "max_part_precision_drop=MAX_PART_PRECISION_DROP" in sources.full_source
    assert "max_wrong_part_rejection_drop=MAX_WRONG_PART_REJECTION_DROP" in sources.full_source
    assert "strategy=CALIBRATION_STRATEGY" in sources.full_source


def test_ood_oe_quality_notebook_contract() -> None:
    sources = _load_notebook_sources("7_ood_oe_quality.ipynb")

    _assert_code_cells_compile(sources, "Notebook 7")
    _assert_contains(
        sources.full_source,
        "RUN_ALL_DATASETS = True",
        "Notebook 7 should default to the batch prepared-runtime audit flow: {snippet}",
    )
    _assert_contains(
        sources.full_source,
        "review_decisions.csv",
        "Notebook 7 should expose the maintained review CSV contract: {snippet}",
    )
    _assert_contains(
        sources.full_source,
        "APPLY_REVIEW_DECISIONS = False",
        "Notebook 7 should keep quarantine application opt-in: {snippet}",
    )
    _assert_contains(
        sources.full_source,
        "--apply-decisions",
        "Notebook 7 should apply decisions through the maintained audit script: {snippet}",
    )
    for stale_snippet in (
        "Fully Automated",
        "AUTO_QUARANTINE",
        "APPLY_NOW",
        "Auto-generate",
        "auto-quarantine",
    ):
        _assert_not_contains(
            sources.full_source,
            stale_snippet,
            "Notebook 7 should stay human-in-loop and avoid automatic quarantine wording: {snippet}",
        )
