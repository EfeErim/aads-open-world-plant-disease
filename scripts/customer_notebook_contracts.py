"""Customer-facing notebook contract checks."""

from __future__ import annotations

from scripts.notebook_validation_support import (
    ACCESS_CHECK_CAPTURE,
    PARAMETER_CAPTURE,
    ROOT,
    _assert_code_cells_compile,
    _assert_contains,
    _assert_not_contains,
    _assert_update_check_contract,
    _find_code_cell_source,
    _load_notebook_sources,
)


def test_auto_router_adapter_notebook_contract() -> None:
    import inspect

    from scripts.colab_auto_router_adapter_prediction import run_auto_router_adapter_prediction

    sources = _load_notebook_sources("8_auto_router_adapter_prediction.ipynb")
    helper_source = inspect.getsource(run_auto_router_adapter_prediction)
    m2_cell_source = (ROOT / "scripts" / "notebook_cells" / "nb8_cell06_m2_full_demo_run.py").read_text(
        encoding="utf-8"
    )
    m2_run_state_helper_source = (ROOT / "scripts" / "notebook_helpers" / "nb8_m2_run_state.py").read_text(
        encoding="utf-8"
    )
    m2_reporting_helper_source = (ROOT / "scripts" / "notebook_helpers" / "nb8_m2_reporting.py").read_text(
        encoding="utf-8"
    )
    m2_command_helper_source = (ROOT / "scripts" / "notebook_helpers" / "nb8_m2_commands.py").read_text(
        encoding="utf-8"
    )
    m2_settings_helper_source = (ROOT / "scripts" / "notebook_helpers" / "nb8_m2_settings.py").read_text(
        encoding="utf-8"
    )
    m2_runtime_helper_source = (ROOT / "scripts" / "notebook_helpers" / "nb8_m2_runtime_helpers.py").read_text(
        encoding="utf-8"
    )
    m2_full_demo_contract_source = "\n".join(
        (
            m2_cell_source,
            m2_run_state_helper_source,
            m2_reporting_helper_source,
            m2_command_helper_source,
            m2_settings_helper_source,
            m2_runtime_helper_source,
        )
    )

    _assert_code_cells_compile(sources, "Notebook 8")
    _assert_contains(
        sources.first_code_source,
        "def _ensure_aads_repo_on_path():",
        "Notebook 8 first code cell should bootstrap the repo before cell runner import: {snippet}",
    )
    for snippet in (
        "AADS_GITHUB_RELEASE_READ_TOKEN",
        "userdata.get('AADS_GITHUB_RELEASE_READ_TOKEN')",
        "GIT_ASKPASS",
        "GIT_ASKPASS_REQUIRE",
        "AADS_GIT_READ_TOKEN",
        "GIT_TERMINAL_PROMPT",
    ):
        _assert_contains(
            sources.first_code_source,
            snippet,
            "Notebook 8 private-repository bootstrap must use and redact its read-only token: {snippet}",
        )
    _assert_not_contains(
        sources.first_code_source,
        "AADS_GITHUB_RELEASE_WRITE_TOKEN",
        "Notebook 8 clone bootstrap must never fall back to the publisher credential: {snippet}",
    )
    _assert_contains(
        sources.first_code_source,
        "run_cell_script('nb1_cell01_bootstrap.py', globals())",
        "Notebook 8 should reuse Notebook 1 bootstrap cell script: {snippet}",
    )
    for script_name in (
        "nb1_cell02_access_check.py",
        "nb1_cell03_runtime_setup.py",
        "nb8_cell04_adapter_release_fetch.py",
        "nb1_cell04_analysis.py",
        "nb8_cell05_adapter_prediction.py",
    ):
        _assert_contains(
            sources.full_source,
            f"run_cell_script('{script_name}', globals())",
            "Notebook 8 should stay a thin wrapper over maintained cell scripts: {snippet}",
        )
    _assert_contains(
        sources.full_source,
        "from scripts.colab_auto_router_adapter_prediction import run_auto_router_adapter_prediction",
        "Notebook 8 should use the maintained auto router-adapter helper: {snippet}",
    )
    full_prediction_cell = _find_code_cell_source(
        sources,
        "router_result = result",
        "Notebook 8 should have a full prediction cell that keeps the router result.",
    )
    _assert_contains(
        full_prediction_cell,
        "run_cell_script('nb1_cell04_analysis.py', globals())",
        "Notebook 8 full prediction cell should run Notebook 1 router analysis first: {snippet}",
    )
    _assert_contains(
        full_prediction_cell,
        "run_cell_script('nb8_cell05_adapter_prediction.py', globals())",
        "Notebook 8 full prediction cell should immediately load the adapter and predict: {snippet}",
    )
    assert full_prediction_cell.index("run_cell_script('nb1_cell04_analysis.py', globals())") < full_prediction_cell.index(
        "run_cell_script('nb8_cell05_adapter_prediction.py', globals())"
    ), "Notebook 8 should run router analysis before adapter prediction in the same cell"
    assert full_prediction_cell.index("run_cell_script('nb8_cell05_adapter_prediction.py', globals())") < full_prediction_cell.index(
        "auto_result"
    ), "Notebook 8 should return the adapter prediction result, not only the router target"
    _assert_contains(
        helper_source,
        "workflow_factory: WorkflowFactory = InferenceWorkflow",
        "Notebook 8 helper should call the canonical InferenceWorkflow: {snippet}",
    )
    _assert_contains(
        helper_source,
        "trust_crop_hint=True",
        "Notebook 8 helper should avoid duplicating Notebook 1 routing by using a trusted router handoff: {snippet}",
    )
    for snippet in (
        "M2_AUTO_APPLY_RUN_STATE = True",
        "M2_FORCE_RUN_STATE = False",
        "M2_RUN_STATE_CONFIG = 'docs/notebook8_m2_run_state.json'",
        "M2_RUN_FULL_DEMO = False",
        "M2_OPEN_WORLD_ONLY = False",
        "M2_RUN_PROBLEM_ONLY_DEMO = False",
        "M2_DEMO_LIMIT = None",
        "M2_BATCH_SIZE = 4",
        "M2_ADAPTER_BATCH_SIZE = 2",
        "M2_DEMO_MANIFEST = 'docs/demo_assets/m2_full_image_set/manifests/m2_balanced_80_run_manifest.csv'",
        "M2_HANDOFF_CACHE = '.runtime_tmp/m2_balanced_80_handoff_cache.json'",
        "M2_REFRESH_HANDOFF_CACHE = True",
        "M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION = True",
        "M2_PROBLEM_ONLY_MANIFEST = 'docs/demo_assets/m2_problem_only_manifests/20260628T113313Z_router_failures.csv'",
        "M2_PROBLEM_ONLY_CALIBRATION_MANIFEST = 'docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv'",
        "M2_PROBLEM_ONLY_COMPARISON_BASELINE = ''",
        "M2_PROTOTYPE_CURATION_ROOT = 'docs/demo_assets/prototype_curation/20260704T095107Z_router_refinement'",
        "M2_COMPARISON_BASELINE = ''",
        "M2_RUN_OPEN_WORLD_ROUTER_VALIDATION = False",
        "M2_OPEN_WORLD_SUPPORTED_MANIFEST = 'docs/demo_assets/m2_full_image_set/manifests/m2_balanced_80_run_manifest.csv'",
        "M2_OPEN_WORLD_MANIFEST = 'docs/demo_assets/open_world_router/manifests/m2_open_world_router_manifest.csv'",
        "M2_OPEN_WORLD_OUTPUT_ROOT = 'docs/demo_results/router_open_world'",
        "M2_OPEN_WORLD_BASELINE_SUMMARY = M2_COMPARISON_BASELINE",
        "M2_OPEN_WORLD_PROTOTYPE_ARTIFACT_DIR = 'docs/demo_results/m2/20260630T192242Z'",
        "M2_PROTOTYPE_TARGET_MIN_PRECISION = 0.98",
        "M2_PROTOTYPE_TARGET_MAX_SUPPORTED_WRONG = 1",
        "M2_PROTOTYPE_TARGET_CLASS_MIN_ACCEPTED = 5",
    ):
        _assert_contains(
            sources.full_source,
            snippet,
            "Notebook 8 public defaults should keep the single-image contract: {snippet}",
        )
    for snippet in (
        'm2_auto_apply_run_state = bool(notebook_globals.get("M2_AUTO_APPLY_RUN_STATE", True))',
        'm2_force_run_state = bool(notebook_globals.get("M2_FORCE_RUN_STATE", True))',
        'm2_run_state_config = str(notebook_globals.get("M2_RUN_STATE_CONFIG", "docs/notebook8_m2_run_state.json"))',
        'm2_run_full_demo = bool(notebook_globals.get("M2_RUN_FULL_DEMO", True))',
        'm2_open_world_only = bool(notebook_globals.get("M2_OPEN_WORLD_ONLY", False))',
        'm2_run_problem_only_demo = bool(notebook_globals.get("M2_RUN_PROBLEM_ONLY_DEMO", False))',
        "initial_global_names = set(notebook_globals)",
        "load_m2_run_state_config(m2_run_state_config)",
        "def load_m2_run_state_config(",
        "def apply_m2_run_state(",
        "def validate_balanced_manifest_request(",
        "Applied run-state config",
        "force_run_state",
        "Refusing to run stale Notebook 8 customer-demo manifest",
        'm2_problem_only_manifest = str(',
        'm2_problem_only_calibration_manifest = str(',
        'notebook_globals.get("M2_PROBLEM_ONLY_COMPARISON_BASELINE", "") or ""',
        "run_state_operator_override_names = operator_override_names(initial_global_names)",
        "RUN_STATE_OPERATOR_OVERRIDE_NAMES = (",
        '"M2_RUN_PROBLEM_ONLY_DEMO"',
        '"M2_RUN_FULL_DEMO"',
        '"M2_OPEN_WORLD_ONLY"',
        '"M2_REFRESH_HANDOFF_CACHE"',
        '"M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION"',
        '"M2_DEMO_MANIFEST"',
        '"M2_HANDOFF_CACHE"',
        '"M2_PROTOTYPE_CURATION_ROOT"',
        'm2_batch_size = int(notebook_globals.get("M2_BATCH_SIZE", 4))',
        'm2_adapter_batch_size = int(notebook_globals.get("M2_ADAPTER_BATCH_SIZE", 2))',
        'm2_pytorch_cuda_alloc_conf = str(',
        "PYTORCH_CUDA_ALLOC_CONF",
        'm2_demo_manifest = str(',
        'notebook_globals.get("M2_HANDOFF_CACHE", ".runtime_tmp/m2_balanced_80_handoff_cache.json")',
        'm2_refresh_handoff_cache = bool(notebook_globals.get("M2_REFRESH_HANDOFF_CACHE", True))',
        'notebook_globals.get("M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION", True)',
        'm2_run_open_world_router_validation = bool(',
        'notebook_globals.get("M2_OPEN_WORLD_BASELINE_SUMMARY", m2_comparison_baseline) or ""',
        'm2_open_world_prototype_artifact_dir = str(',
        'm2_prototype_curation_root = str(notebook_globals.get("M2_PROTOTYPE_CURATION_ROOT", "") or "")',
        'm2_prototype_curation_root',
        "def run_open_world_router_validation(",
        "write_m2_failure_summary(",
        "failed_before_complete_report",
        "def write_m2_failure_summary(",
        "write_m2_result_comparison(",
        "def write_m2_result_comparison(",
        "def copy_existing_artifacts(",
        "build_m2_demo_checklist_command(",
        "def build_m2_demo_checklist_command(",
        "if M2_OPEN_WORLD_ONLY:",
        "def validate_curated_prototype_bank(",
        "Selected prototype bank contains zero usable curated rows",
        "Prototype builder failed while M2_PROTOTYPE_CURATION_ROOT is set",
        "operator_overrides",
        '"--curation-root"',
        "enrich_summary_manifest_sha256",
        "manifest_sha256",
        "m2_result_comparison.json",
        "m2_result_comparison.md",
        "m2_comparison_written",
        "m2_comparison_passed",
        "m2_runner_succeeded",
        "run_router_open_world_validation.py",
        "open_world_router_validation_passed",
        "M2_OPEN_WORLD_SUPPORTED_MANIFEST",
        "M2_OPEN_WORLD_MANIFEST",
        "--min-open-world-rows",
        "--require-latency-baseline",
        "if M2_RUN_OPEN_WORLD_ROUTER_VALIDATION:",
        "and open_world_passed",
        "--target-min-precision",
        "--target-max-supported-wrong",
        "--target-class-min-accepted",
    ):
        _assert_contains(
            m2_full_demo_contract_source,
            snippet,
            "Notebook 8 M2 full-demo calibration should pass bounded target-policy controls: {snippet}",
        )
    _assert_not_contains(
        m2_cell_source,
        "M2_RUN_OPEN_WORLD_ROUTER_VALIDATION and int(completed.returncode) == 0",
        "Open-world validation should run independently of the 602-row M2 runner exit code: {snippet}",
    )
    _assert_not_contains(
        m2_cell_source,
        "skipped_m2_runner_failed",
        "Open-world validation should not be skipped just because the separate M2 checklist failed: {snippet}",
    )
    assert sources.full_source.count("run_inference(") == 1, (
        "Notebook 8 should inherit the single Notebook 1 router call, not add a second router-only implementation"
    )

def test_colab_helpers() -> None:
    from scripts.colab_checkpointing import TrainingCheckpointManager
    from scripts.colab_live_telemetry import ColabLiveTelemetry
    from scripts.colab_repo_bootstrap import (
        export_current_colab_notebook,
        mirror_checkpoint_state_to_repo,
        mirror_path_to_repo,
        push_repo_paths_to_github,
        push_repo_run_to_github,
    )
    from scripts.colab_simple_adapter_smoke_ui import launch_simple_adapter_smoke_ui
    from scripts.evaluate_dataset_layout import evaluate_layout
    from scripts.prepare_grouped_runtime_dataset import (
        build_grouped_dataset_plan,
        materialize_grouped_runtime_dataset,
        scan_class_root_dataset,
    )
    from scripts.prepare_materialization_dataset import prepare_class_root_for_materialization

    assert hasattr(ColabLiveTelemetry, "configure_repo_output_export")
    assert callable(export_current_colab_notebook)
    assert callable(mirror_checkpoint_state_to_repo)
    assert callable(mirror_path_to_repo)
    assert callable(push_repo_paths_to_github)
    assert callable(push_repo_run_to_github)
    assert callable(launch_simple_adapter_smoke_ui)
    assert callable(prepare_class_root_for_materialization)
    _ = (
        TrainingCheckpointManager,
        ColabLiveTelemetry,
        evaluate_layout,
        build_grouped_dataset_plan,
        materialize_grouped_runtime_dataset,
        scan_class_root_dataset,
    )


def test_data_prep_notebook_contract() -> None:
    sources = _load_notebook_sources("0_prepare_grouped_dataset_for_training.ipynb")
    grouped_prep_helper_source = (
        ROOT / "scripts" / "notebook_helpers" / "nb0_grouped_dataset_prep_helpers.py"
    ).read_text(encoding="utf-8")
    data_prep_contract_source = sources.full_source + "\n" + grouped_prep_helper_source
    bootstrap_source = _find_code_cell_source(
        sources,
        "from scripts.colab_live_telemetry import ColabLiveTelemetry",
        "Notebook 0 bootstrap cell was not found",
    )
    parameter_source = _find_code_cell_source(
        sources,
        PARAMETER_CAPTURE,
        "Notebook 0 parameter cell was not found",
    )
    access_check_source = _find_code_cell_source(
        sources,
        ACCESS_CHECK_CAPTURE,
        "Notebook 0 access-check cell was not found",
    )

    _assert_update_check_contract(
        sources.first_code_source,
        "Notebook 0",
        forbid_drive_bootstrap=True,
    )
    assert "os.environ['AADS_GITHUB_RELEASE_READ_TOKEN'] = token" in sources.first_code_source
    for snippet in (
        "RUN_ID =",
        "TELEMETRY = ColabLiveTelemetry(",
        "REPO_RUN_DIR =",
        "REPO_NOTEBOOK_OUTPUT_PATH =",
    ):
        assert snippet in bootstrap_source, f"Notebook 0 bootstrap is missing: {snippet}"
    for snippet in (
        "mount_drive_if_available",
        "def _mount_drive_inline()",
        "Path('/content/drive/MyDrive/bitirme projesi')",
        "Path('/content/drive/MyDrive/bitirmeprojesi')",
        "def _copy_path_to_drive_exports",
    ):
        _assert_not_contains(
            bootstrap_source,
            snippet,
            "Notebook 0 bootstrap should not mirror repo prep outputs through Drive: {snippet}",
        )
    for snippet in (
        "REPO_DATASET_ROOT =",
        'REPO_DATASET_NAME = ""',
        "DATASET_ROOT =",
        "IMPORT_FROM_DRIVE = False",
        "DRIVE_DATASET_PATH =",
        "DRIVE_DATASET_NAME =",
        "CROP_NAME =",
        "PART_NAME =",
    ):
        assert snippet in parameter_source, f"Notebook 0 parameter cell is missing: {snippet}"
    assert "IMPORT_FROM_DRIVE = FALSE" not in parameter_source
    for snippet in (
        "PREP_ARTIFACT_ROOT =",
        "PREPARED_RUNTIME_ROOT =",
        "OOD_DATASET_ROOT =",
        "OOD_DATASET_NAME =",
        "OOD_ROOT =",
        "ASK_FOR_OOD_ROOT =",
        "PREPARED_CLASS_ROOT =",
        "PREPARE_DATASET_FROM_REPORTS =",
        "MATERIALIZE_AFTER_REVIEW =",
        "INTERACTIVE_AUDIT_REVIEW =",
        "MAX_INTERACTIVE_REVIEW_ITEMS =",
        "DATASET_RELEASE_REPOSITORY =",
        "DATASET_RELEASE_TAG =",
        "DATASET_RELEASE_TARGET =",
        "DATASET_RELEASE_CACHE_ROOT =",
        "CLEANUP_SEED =",
        "PREP_DINOV3_MODEL_ID =",
        "PREP_BIOCLIP_MODEL_ID =",
    ):
        assert snippet in bootstrap_source, f"Notebook 0 bootstrap is missing: {snippet}"
    assert "collect_notebook_access_report" in access_check_source
    assert "print_notebook_access_report" in access_check_source
    assert "resolve_token(write=False)" in access_check_source
    assert '"write_token_requested": False' in access_check_source
    assert "run_dataset_audit(" in sources.full_source
    assert "run_materialize_runtime_dataset(" in sources.full_source
    assert "build_grouped_dataset_plan" in data_prep_contract_source
    assert "build_human_review_packet" in data_prep_contract_source
    assert "format_human_review_packet" in data_prep_contract_source
    assert "evaluate_layout(root=dataset_root)" in data_prep_contract_source
    assert "ASK_FOR_OOD_ROOT" in data_prep_contract_source
    assert "resolve_dataset_directory_from_parent" in data_prep_contract_source
    assert "build_prepared_dataset_key" in data_prep_contract_source
    assert "prepare_class_root_for_materialization" in data_prep_contract_source
    assert "def _resolve_repo_dataset_root" in data_prep_contract_source
    assert "resolve_repo_dataset_directory" in data_prep_contract_source
    assert 'dataset_source = "drive" if IMPORT_FROM_DRIVE' in data_prep_contract_source
    assert 'STATE["dataset_name"] = dataset_name' in data_prep_contract_source
    assert 'STATE["dataset_source"] = dataset_source' in data_prep_contract_source
    assert "MATERIALIZE_AFTER_REVIEW = True" in sources.full_source
    assert "materialize_grouped_runtime_dataset" in data_prep_contract_source
    assert "fetch_materialize_dataset_release" in data_prep_contract_source
    assert "DatasetReleaseAccessBlocker" in data_prep_contract_source
    assert "SAVE_RUNTIME_DATASET_TO_GITHUB" not in data_prep_contract_source
    assert "runtime_dataset_push_report" not in data_prep_contract_source

def test_training_notebook_dataset_contract_detection() -> None:
    sources = _load_notebook_sources("2_train_continual_sd_lora_adapter.ipynb")
    _assert_contains(
        sources.first_code_source,
        "os.environ['AADS_GITHUB_RELEASE_READ_TOKEN'] = token",
        "Notebook 2 must export the Colab dataset Release secret under the canonical runtime token name.",
    )
    assert 'DATASET_RELEASE_REPOSITORY = str(' in sources.full_source
    assert 'DATASET_RELEASE_TAG = str(' in sources.full_source
    assert "fetch_materialize_dataset_release(" in sources.full_source
    assert "from scripts.colab_training_recommendations import inspect_runtime_dataset" in sources.full_source
    assert "inspect_runtime_dataset" in sources.full_source
    assert "resolve_notebook_params" in sources.full_source
    assert 'selected_release_root = Path(release_report["selected_dataset_root"])' in sources.full_source
    assert 'STATE["runtime_dataset_key"] = selected_dataset_name' in sources.full_source
    assert "from src.data.loaders import create_training_loaders" in sources.full_source
    assert "src.utils.data_loader" not in sources.full_source
    assert "No verified runtime datasets were materialized" in sources.full_source
    assert "Prepared runtime dataset is missing split folder(s)" in sources.full_source
    assert 'STATE["resolved_ood_root"] = resolved_ood_root_value' in sources.full_source
    assert 'STATE["resolved_oe_root"] = resolved_oe_root_value' in sources.full_source
    assert 'STATE["dataset_inspection"] = dataset_inspection' in sources.full_source
    assert 'STATE["hardware_inspection"] = {}' in sources.full_source
    assert 'STATE["recommendation_report"] = {}' in sources.full_source
    assert 'STATE["recommendation_decision"] = "disabled"' in sources.full_source
    assert 'STATE["effective_params"] = effective_params' in sources.full_source
    assert "ASK_FOR_OOD_ROOT = True" in sources.full_source
    assert "ASK_FOR_OE_ROOT = True" in sources.full_source
    assert "OOD klasoru yolunu girin" in sources.full_source
    assert "OE klasoru yolunu girin" in sources.full_source
    assert "ood_root=resolved_ood_root or None" in sources.full_source
    assert "oe_root=resolved_oe_root or None" in sources.full_source
    assert "build_grouped_dataset_plan" not in sources.full_source
    assert "materialize_grouped_runtime_dataset" not in sources.full_source


def test_training_notebook_bootstrap_contract() -> None:
    sources = _load_notebook_sources("2_train_continual_sd_lora_adapter.ipynb")
    bootstrap_source = _find_code_cell_source(
        sources,
        "from scripts.colab_live_telemetry import ColabLiveTelemetry",
        "Notebook 2 bootstrap cell was not found",
    )
    parameter_source = _find_code_cell_source(
        sources,
        PARAMETER_CAPTURE,
        "Notebook 2 parameter cell was not found",
    )
    run_identity_source = _find_code_cell_source(
        sources,
        "# Notebook 2 calisma kimligi",
        "Notebook 2 run identity cell was not found",
    )
    access_check_source = _find_code_cell_source(
        sources,
        ACCESS_CHECK_CAPTURE,
        "Notebook 2 access-check cell was not found",
    )

    _assert_update_check_contract(
        sources.first_code_source,
        "Notebook 2",
        forbid_drive_bootstrap=True,
    )

    required_bootstrap_snippets = (
        "RUN_ID =",
        "TELEMETRY = ColabLiveTelemetry(",
        "LOCAL_TELEMETRY_ROOT = ROOT / 'outputs' / 'colab_notebook_training' / 'telemetry_runtime'",
        'exclude_dir_names=("checkpoints", "telemetry_runtime")',
        "CHECKPOINT_MANAGER =",
        "DEVICE =",
        "def rt(",
        "REPO_RUN_DIR =",
        "REPO_NOTEBOOK_OUTPUT_PATH =",
        "def save_run_outputs_to_repo()",
        "build_notebook_run_dir",
        "build_notebook_run_id",
    )
    missing = [snippet for snippet in required_bootstrap_snippets if snippet not in bootstrap_source]
    if missing:
        raise AssertionError(f"Notebook 2 bootstrap cell is missing required setup: {', '.join(missing)}")

    assert PARAMETER_CAPTURE in parameter_source
    assert sources.full_source.index("TELEMETRY = ColabLiveTelemetry(") < sources.full_source.index(
        PARAMETER_CAPTURE
    )
    assert sources.full_source.index("RUN_ID =") < sources.full_source.index("run_id = RUN_ID")
    assert sources.full_source.index("CHECKPOINT_MANAGER =") < sources.full_source.index('"checkpoint_manager": CHECKPOINT_MANAGER')
    assert 'PART_NAME = "unspecified"' in run_identity_source
    assert "collect_notebook_access_report" in access_check_source
    assert "print_notebook_access_report" in access_check_source
    assert "REPO_RUN_DIR = build_notebook_run_dir(ROOT, CROP_NAME, PART_NAME, RUN_ID)" in bootstrap_source

    required_parameter_snippets = (
        'PART_NAME = globals().get("PART_NAME", "unspecified")',
        'DATASET_RELEASE_REPOSITORY = str(',
        'DATASET_RELEASE_TAG = str(',
        'DATASET_RELEASE_CACHE_ROOT = str(',
        'DATASET_SOURCE_KIND = "github_release"',
        'DATASET_RELEASE_MANIFEST_PATH = str(',
        'ALLOW_LOCAL_LEGACY_DATASET = False',
        'DATASET_NAME = ""',
        'OOD_ROOT = ""',
        'ASK_FOR_OOD_ROOT = True',
        'OE_ROOT = ""',
        'ASK_FOR_OE_ROOT = True',
        'OE_ENABLED = False',
        '"OE_ENABLED": bool(OE_ENABLED)',
        'OE_ENABLED = bool(INITIAL_EFFECTIVE_PARAMS["OE_ENABLED"])',
        'OE_LOSS_WEIGHT = 0.5',
        'from scripts.notebook_helpers.adapter_recommendations import get_adapter_recs',
        'ADAPTER_RECS = get_adapter_recs()',
        'MANUAL_PARAM_OVERRIDES = {}',
        'EPOCHS = ',
        'BATCH_SIZE = ',
        'LEARNING_RATE = ',
        'LORA_R = ',
        'AUGMENTATION_POLICY = str(CONTINUAL_DATA_CFG.get("augmentation_policy", "randaugment")).strip().lower()',
        'RANDAUGMENT_NUM_OPS = int(CONTINUAL_DATA_CFG.get("randaugment_num_ops", 2))',
        'RANDAUGMENT_MAGNITUDE = int(CONTINUAL_DATA_CFG.get("randaugment_magnitude", 7))',
        'ALLOW_UNDER_MIN_TRAINING = False',
        'ALLOW_UNDER_MIN_TRAINING = bool(ALLOW_UNDER_MIN_TRAINING)',
        'BER_ENABLED = False',
        'LOSS_NAME = "logitnorm"',
        'LOGITNORM_TAU = 1.0',
        'OOD_FACTOR = ',
        'CHECKPOINT_EVERY_N_STEPS = ',
        'source=notebook_cell',
        'defaults=notebook_cell',
        'parameter_source": "notebook_cell"',
    )
    for snippet in required_parameter_snippets:
        _assert_contains(
            parameter_source,
            snippet,
            "Notebook 2 parameter cell is missing required direct parameter surface: {snippet}",
        )

    required_training_surface_snippets = (
        'optimization_cfg["loss_name"] = str(effective_params["LOSS_NAME"]).strip().lower()',
        'optimization_cfg["logitnorm_tau"] = float(effective_params["LOGITNORM_TAU"])',
        'data_cfg["augmentation_policy"] = str(effective_params.get("AUGMENTATION_POLICY", AUGMENTATION_POLICY))',
        'data_cfg["allow_under_min_training"] = bool(effective_params["ALLOW_UNDER_MIN_TRAINING"])',
        'augmentation_policy=str(effective_params.get("AUGMENTATION_POLICY", AUGMENTATION_POLICY))',
        'STATE["resolved_ood_root"] = resolved_ood_root_value',
        'STATE["resolved_oe_root"] = resolved_oe_root_value',
        'continual_cfg["ood"]["oe_enabled"] = bool(effective_params["OE_ENABLED"])',
        'continual_cfg["ood"]["oe_root"] = resolved_oe_root',
        'fetch_materialize_dataset_release(',
        'selected_release_root = Path(release_report["selected_dataset_root"])',
        'STATE["dataset_release_report"] = release_report',
        'inspect_runtime_dataset',
        'resolve_notebook_params',
        'STATE["recommendation_decision"] = "disabled"',
        'STATE["effective_params"] = effective_params',
        'effective_params = dict(STATE.get("effective_params") or {})',
        'STATE["runtime_dataset_key"] = selected_dataset_name',
        'STATE["selected_dataset_name"] = selected_dataset_name',
    )
    for snippet in required_training_surface_snippets:
        _assert_contains(
            sources.full_source,
            snippet,
            "Notebook 2 training surface is missing required explicit config wiring: {snippet}",
        )

    forbidden_parameter_snippets = (
        'MAX_STABLE_PROFILE = {',
        'profile_payload = dict(MAX_STABLE_PROFILE)',
        'profile=max_stable',
        'NOTEBOOK_OVERRIDE_CASTERS = {',
        'NOTEBOOK_SETTINGS = {',
        'NOTEBOOK_OVERRIDES =',
        'EPOCHS = int(CONTINUAL_CFG.get("num_epochs"',
        'BATCH_SIZE = int(CONTINUAL_CFG.get("batch_size"',
        'LEARNING_RATE = float(CONTINUAL_CFG.get("learning_rate"',
        'source=merged_config(colab)',
        'defaults=config(colab)',
        'DATASET_ROOT = "data/class_root_dataset"',
        'OOD_DATASET_ROOT = "data/ood_dataset"',
        'OOD_DATASET_NAME = ""',
        'inspect_runtime_hardware',
        'recommend_notebook_training_params',
        'resolve_effective_notebook_params',
        'Apply recommended parameters? [y/N]:',
        'accepted_recommendations',
        'recommendation_report = recommend_notebook_training_params',
    )
    for snippet in forbidden_parameter_snippets:
        _assert_not_contains(
            sources.full_source,
            snippet,
            (
                "Notebook 2 parameter cell should not contain hidden overrides "
                "or config-derived parameter remapping: {snippet}"
            ),
        )
