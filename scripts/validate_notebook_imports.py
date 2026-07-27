#!/usr/bin/env python3
"""Validate the maintained notebook support surfaces."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.notebook_validation_support import (  # noqa: E402
    ROOT,
    VALIDATION_GROUP_ORDER,
    ValidationCheck,
    _assert_repo_bootstrap_contract,  # noqa: F401 - compatibility export
    _find_code_cell_source,  # noqa: F401 - compatibility export
    _load_notebook_sources,  # noqa: F401 - compatibility export
    _load_notebook_sources_from_path,  # noqa: F401 - compatibility export
    _run_check,
    _safe_print,
)

print = _safe_print


def _check_runtime_dependencies() -> None:
    required = (
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "accelerate",
        "huggingface_hub",
        "PIL",
    )
    missing = []
    for module_name in required:
        try:
            __import__(module_name)
        except Exception:
            import logging
            logging.exception('Unhandled exception')
            raise
            missing.append(module_name)

    if missing:
        missing_display = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Missing dependencies: {missing_display}. Install requirements.txt before running this validation."
        )


def test_config_surface() -> None:
    from src.core.config_manager import ConfigurationManager

    cfg = ConfigurationManager(config_dir=str(ROOT / "config"), environment="colab").load_all_configs()
    assert {"training", "router", "colab", "inference"} <= set(cfg.keys())


def test_private_repo_bootstrap_security_contract() -> None:
    notebooks = (
        "0_prepare_grouped_dataset_for_training.ipynb",
        "1_identify_crop_part_with_router.ipynb",
        "2_train_continual_sd_lora_adapter.ipynb",
        "3_validate_exported_adapter_directly.ipynb",
        "4_simple_direct_adapter_test_ui.ipynb",
        "5_calibrate_router_handoff_thresholds.ipynb",
        "6_train_all_continual_sd_lora_adapters.ipynb",
        "7_ood_oe_quality.ipynb",
        "8_auto_router_adapter_prediction.ipynb",
        "16_ablation_dual_view_inference.ipynb",
        "17_adapter_ood_oe_recovery.ipynb",
    )
    required = (
        "AADS_GITHUB_RELEASE_READ_TOKEN",
        "userdata.get(",
        "GIT_ASKPASS",
        "GIT_ASKPASS_REQUIRE",
        "AADS_GIT_READ_TOKEN",
        "GIT_TERMINAL_PROMPT",
    )
    forbidden = (
        "AADS_GITHUB_RELEASE_WRITE_TOKEN",
        "_repo_url_with_token",
        "_build_repo_access_url(repo_url",
        "f'{token}@",
        "x-access-token:' + token",
    )
    for notebook in notebooks:
        sources = _load_notebook_sources(notebook)
        first_cell = sources.first_code_source
        compile(first_cell, notebook, "exec")
        for snippet in required:
            assert snippet in first_cell, f"{notebook} missing secure private-repo bootstrap: {snippet}"
        for snippet in forbidden:
            assert snippet not in first_cell, f"{notebook} persists or exposes a GitHub credential: {snippet}"

    for relative_path in (
        "scripts/notebook_cells/nb3_cell01_bootstrap_access.py",
        "scripts/notebook_cells/nb4_cell01_bootstrap_access.py",
        "scripts/notebook_cells/nb5_cell01_bootstrap_access.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        for snippet in required:
            assert snippet in source, f"{relative_path} missing secure private-repo bootstrap: {snippet}"
        for snippet in forbidden:
            assert snippet not in source, f"{relative_path} exposes a GitHub credential: {snippet}"


def test_continual_trainer_imports() -> None:
    from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer
    from src.training.session import ContinualTrainingSession
    from src.training.validation import evaluate_model
    from src.workflows.training import TrainingWorkflow

    config = ContinualSDLoRAConfig.from_training_config(
        {
            "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
            "adapter": {
                "target_modules_strategy": "all_linear_transformer",
                "lora_r": 4,
                "lora_alpha": 8,
            },
            "fusion": {"layers": [2, 5, 8, 11]},
            "ood": {"threshold_factor": 2.0},
            "device": "cpu",
        }
    )
    trainer = ContinualSDLoRATrainer(config)
    assert hasattr(trainer, "initialize_engine")
    assert hasattr(trainer, "add_classes")
    assert hasattr(trainer, "train_batch")
    assert hasattr(trainer, "snapshot_training_state")
    assert hasattr(trainer, "restore_training_state")
    assert hasattr(trainer, "save_adapter")
    assert hasattr(trainer, "load_adapter")
    assert ContinualTrainingSession is not None
    assert TrainingWorkflow is not None
    assert callable(evaluate_model)


def test_quantization_guard() -> None:
    from src.training.quantization import assert_no_prohibited_4bit_flags

    valid_payload = {
        "training": {
            "continual": {
                "adapter": {"target_modules_strategy": "all_linear_transformer"}
            }
        }
    }
    assert_no_prohibited_4bit_flags(valid_payload)

    rejected = False
    try:
        forbidden_key = "load_in_" + "4bit"
        assert_no_prohibited_4bit_flags({forbidden_key: True})
    except ValueError:
        rejected = True

    assert rejected, "4-bit payload was expected to be rejected"


def test_adapter_surface() -> None:
    from src.adapter.independent_crop_adapter import IndependentCropAdapter

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    assert hasattr(adapter, "initialize_engine")
    assert hasattr(adapter, "add_classes")
    assert hasattr(adapter, "build_training_session")
    assert hasattr(adapter, "save_adapter")
    assert hasattr(adapter, "load_adapter")


def test_runtime_surface() -> None:
    from scripts.colab_auto_router_adapter_prediction import run_auto_router_adapter_prediction
    from src.pipeline.router_adapter_runtime import RouterAdapterRuntime
    from src.workflows.inference import InferenceWorkflow

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {
                "continual": {
                    "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
                    "adapter": {"target_modules_strategy": "all_linear_transformer"},
                    "fusion": {"layers": [2, 5, 8, 11]},
                    "ood": {"threshold_factor": 2.0},
                }
            },
            "inference": {"adapter_root": "models/adapters", "target_size": 224},
        },
        device="cpu",
    )
    assert hasattr(runtime, "load_router")
    assert hasattr(runtime, "load_adapter")
    assert hasattr(runtime, "predict")
    assert InferenceWorkflow is not None
    assert callable(run_auto_router_adapter_prediction)



def test_repo_dataset_scaffold() -> None:
    required_paths = (
        ROOT / "data" / "README.md",
        ROOT / "data" / "class_root_dataset" / ".gitkeep",
        ROOT / "data" / "ood_dataset" / ".gitkeep",
        ROOT / "data" / "prepared_class_root_datasets" / ".gitkeep",
        ROOT / "data" / "prepared_runtime_datasets" / ".gitkeep",
    )
    missing = [str(path.relative_to(ROOT)) for path in required_paths if not path.exists()]
    assert not missing, f"Missing dataset scaffold path(s): {', '.join(missing)}"

from scripts.customer_notebook_contracts import (  # noqa: E402
    test_auto_router_adapter_notebook_contract,
    test_colab_helpers,
    test_data_prep_notebook_contract,
    test_training_notebook_bootstrap_contract,
    test_training_notebook_dataset_contract_detection,
)
from scripts.internal_notebook_contracts import (  # noqa: E402
    test_adapter_ood_oe_recovery_notebook_contract,
    test_adapter_smoke_notebook_bootstrap_contract,
    test_adapter_smoke_notebook_surface,
    test_batch_training_notebook_contract,
    test_ood_oe_quality_notebook_contract,
    test_roi_ablation_notebook_contract,
    test_router_calibration_notebook_contract,
    test_simple_adapter_smoke_notebook_bootstrap_contract,
)

CHECKS = (
    ValidationCheck(
        result_name="Runtime Dependencies",
        step_id="ENV",
        description="runtime dependencies",
        success_message="Runtime dependencies available",
        failure_prefix="Missing dependencies",
        callback=_check_runtime_dependencies,
        validation_group="shared-prerequisite",
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Minimal Config",
        step_id="CONFIG",
        description="minimal config load",
        success_message="Configuration loaded successfully",
        failure_prefix="Configuration load failed",
        callback=test_config_surface,
        validation_group="shared-prerequisite",
    ),
    ValidationCheck(
        result_name="Private Repo Bootstrap",
        step_id="PRIVATE_REPO_BOOTSTRAP",
        description="private GitHub repository bootstrap security contract",
        success_message="All maintained Colab notebooks use read-only non-persistent Git authentication",
        failure_prefix="Private repository bootstrap contract failed",
        callback=test_private_repo_bootstrap_security_contract,
        validation_group="shared-prerequisite",
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Continual Trainer",
        step_id="TRAINING",
        description="continual trainer imports",
        success_message="Continual trainer surface imported and validated",
        failure_prefix="Continual trainer test failed",
        callback=test_continual_trainer_imports,
        validation_group="customer-notebook-support",
    ),
    ValidationCheck(
        result_name="Quantization Guard",
        step_id="LOW_BIT_GUARD",
        description="4-bit rejection guard",
        success_message="Quantization guard behaves correctly",
        failure_prefix="Quantization guard failed",
        callback=test_quantization_guard,
        validation_group="customer-notebook-support",
    ),
    ValidationCheck(
        result_name="Adapter Lifecycle",
        step_id="ADAPTER_API",
        description="adapter lifecycle surface",
        success_message="Adapter lifecycle surface available",
        failure_prefix="Adapter API test failed",
        callback=test_adapter_surface,
        validation_group="customer-notebook-support",
    ),
    ValidationCheck(
        result_name="Router Runtime",
        step_id="INFERENCE",
        description="router runtime surface",
        success_message="Router runtime surface available",
        failure_prefix="Router runtime test failed",
        callback=test_runtime_surface,
        validation_group="customer-notebook-support",
    ),
    ValidationCheck(
        result_name="Adapter Smoke Notebook",
        step_id="ADAPTER_SMOKE",
        description="adapter smoke-test helper surface",
        success_message="Adapter smoke-test helper surface available",
        failure_prefix="Adapter smoke-test surface failed",
        callback=test_adapter_smoke_notebook_surface,
        validation_group="customer-notebook-support",
    ),
    ValidationCheck(
        result_name="Notebook 3 Bootstrap",
        step_id="NB3_BOOTSTRAP",
        description="Notebook 3 bootstrap contract",
        success_message="Notebook 3 bootstrap uses GitHub/local repo discovery without Drive repo mounts",
        failure_prefix="Notebook 3 bootstrap contract failed",
        callback=test_adapter_smoke_notebook_bootstrap_contract,
        validation_group="customer-facing-notebooks",
    ),
    ValidationCheck(
        result_name="Notebook 4 Bootstrap (internal-maintenance)",
        step_id="NB4_BOOTSTRAP",
        description="Notebook 4 internal-maintenance bootstrap contract",
        success_message="Notebook 4 bootstrap clones the repo and launches the minimal smoke UI",
        failure_prefix="Notebook 4 bootstrap contract failed",
        callback=test_simple_adapter_smoke_notebook_bootstrap_contract,
        validation_group="internal-maintenance-notebooks",
    ),
    ValidationCheck(
        result_name="Notebook 5 Router Calibration (internal-maintenance)",
        step_id="NB5_ROUTER_CAL",
        description="Notebook 5 internal-maintenance router calibration contract",
        success_message="Notebook 5 wraps maintained router evaluation and calibration scripts",
        failure_prefix="Notebook 5 router calibration contract failed",
        callback=test_router_calibration_notebook_contract,
        validation_group="internal-maintenance-notebooks",
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Notebook 6 Batch Training (internal-maintenance)",
        step_id="NB6_BATCH",
        description="Notebook 6 internal-maintenance batch training contract",
        success_message="Notebook 6 bootstraps Colab and wraps maintained Notebook 2 training cells",
        failure_prefix="Notebook 6 batch training contract failed",
        callback=test_batch_training_notebook_contract,
        validation_group="internal-maintenance-notebooks",
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Notebook 7 OOD/OE Review (internal-maintenance)",
        step_id="NB7_OOD_OE_REVIEW",
        description="Notebook 7 internal-maintenance OOD/OE human-review contract",
        success_message="Notebook 7 stays batchable and human-in-loop for quarantine decisions",
        failure_prefix="Notebook 7 OOD/OE review contract failed",
        callback=test_ood_oe_quality_notebook_contract,
        validation_group="internal-maintenance-notebooks",
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Notebook 17 Adapter OOD/OE Recovery (internal-maintenance)",
        step_id="NB17_ADAPTER_RECOVERY",
        description="Notebook 17 gate-aware adapter OOD/OE recovery contract",
        success_message="Notebook 17 runs the bounded v4 dev-selection and locked-test-once recovery workflow",
        failure_prefix="Notebook 17 adapter recovery contract failed",
        callback=test_adapter_ood_oe_recovery_notebook_contract,
        validation_group="internal-maintenance-notebooks",
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Notebook 8 Auto Inference",
        step_id="NB8_AUTO_INFER",
        description="Notebook 8 auto router-adapter contract",
        success_message="Notebook 8 wraps Notebook 1 routing and canonical adapter inference",
        failure_prefix="Notebook 8 auto inference contract failed",
        callback=test_auto_router_adapter_notebook_contract,
        validation_group="customer-facing-notebooks",
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Notebook 16 ROI Evidence (historical/report-only)",
        step_id="NB16_ROI_EVIDENCE",
        description="Notebook 16 historical/report-only ROI/bbox evidence-gate contract",
        success_message="Notebook 16 wraps the shared ROI evidence helper surface",
        failure_prefix="ROI ablation notebook contract failed",
        callback=test_roi_ablation_notebook_contract,
        validation_group="historical-report-only-notebooks",
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Colab Helpers",
        step_id="COLAB",
        description="colab support helpers",
        success_message="Colab helper surfaces imported successfully",
        failure_prefix="Colab helper import failed",
        callback=test_colab_helpers,
        validation_group="shared-prerequisite",
    ),
    ValidationCheck(
        result_name="Notebook 2 Bootstrap",
        step_id="NB2_BOOTSTRAP",
        description="Notebook 2 bootstrap contract",
        success_message="Notebook 2 bootstrap globals are defined before use",
        failure_prefix="Notebook 2 bootstrap contract failed",
        callback=test_training_notebook_bootstrap_contract,
        validation_group="customer-facing-notebooks",
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Notebook 0 Bootstrap",
        step_id="NB0_BOOTSTRAP",
        description="Notebook 0 bootstrap contract",
        success_message="Notebook 0 bootstrap globals are defined before use",
        failure_prefix="Notebook 0 bootstrap contract failed",
        callback=test_data_prep_notebook_contract,
        validation_group="customer-facing-notebooks",
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Data Scaffold",
        step_id="DATA_LAYOUT",
        description="repo dataset scaffold",
        success_message="Repo-local dataset scaffold is present",
        failure_prefix="Repo dataset scaffold check failed",
        callback=test_repo_dataset_scaffold,
        validation_group="shared-prerequisite",
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Notebook 2 Dataset Contract",
        step_id="NB2_RUNTIME",
        description="Notebook 2 runtime dataset contract",
        success_message="Notebook 2 supports the public sample and verified prepared datasets",
        failure_prefix="Notebook 2 dataset contract check failed",
        callback=test_training_notebook_dataset_contract_detection,
        validation_group="customer-facing-notebooks",
        requires_runtime_dependencies=False,
    ),
)


def main() -> int:
    print("=" * 60)
    print("AADS v6 Minimal Surface Validation")
    print("=" * 60)

    results: list[tuple[str, str, bool]] = []
    runtime_dependencies_ready = True
    for index, check in enumerate(CHECKS):
        if check.requires_runtime_dependencies and not runtime_dependencies_ready:
            continue
        ok = _run_check(check, leading_newline=index > 1)
        results.append((check.validation_group, check.result_name, ok))
        if check.step_id == "ENV":
            runtime_dependencies_ready = ok

    passed = sum(1 for _group, _name, ok in results if ok)
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for group in (*VALIDATION_GROUP_ORDER, *sorted({group for group, _name, _ok in results} - set(VALIDATION_GROUP_ORDER))):
        group_results = [(name, ok) for result_group, name, ok in results if result_group == group]
        if not group_results:
            continue
        print(f"\n[{group}]")
        for name, ok in group_results:
            print(f"{'PASS' if ok else 'FAIL'}: {name}")
    print("=" * 60)
    print(f"Results: {passed}/{len(results)} tests passed")
    print("=" * 60)

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
