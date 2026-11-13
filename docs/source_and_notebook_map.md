# Source and notebook map

This page answers a simple question: what is actually included in the public repository?

The public repository started as a path-for-path copy of the maintained private project tree (source commit
`539397bb72bde59e4b092ac1286b5415fe78dbac`). It has since been restructured on its own: one-off research scripts
were removed, the Colab layer was gathered into one package and report-only research moved to `experiments/`. It
therefore no longer mirrors the private tree file for file.

The application implementation is not embedded only in notebooks. Colab notebooks call the same maintained modules
used by the developer CLI and the test suite.

## Application source

The 132 files under [`src/`](../src/) are grouped as follows.

| Package | Files | Responsibility |
|---|---:|---|
| [`src/training/`](../src/training/) | 26 | continual SD-LoRA training, validation, persistence and reporting |
| [`src/router/`](../src/router/) | 26 | crop/part routing, taxonomy, prototypes, calibration and abstention |
| [`src/data/`](../src/data/) | 18 | dataset contracts, lineage, releases, public smoke data, OOD splits and integrity |
| [`src/colab/`](../src/colab/) | 17 | Colab bootstrap, widgets, telemetry, checkpoint mirroring and notebook run helpers |
| [`src/shared/`](../src/shared/) | 11 | typed contracts, serialization/path/hash utilities and the GitHub Releases client |
| [`src/ood/`](../src/ood/) | 7 | OOD scoring, conformal prediction and behavioral acceptance |
| [`src/workflows/`](../src/workflows/) | 7 | training and inference entry points and their run stages |
| [`src/pipeline/`](../src/pipeline/) | 6 | adapter discovery, release loading, input guard and inference payloads |
| [`src/adapter/`](../src/adapter/) | 4 | adapter configuration, checkpointing and lifecycle helpers |
| [`src/core/`](../src/core/) | 3 | configuration loading and migrations |
| [`src/app/`](../src/app/) | 2 | developer CLI (`aads inference`, `aads training`) |
| [`src/utils/`](../src/utils/) | 2 | small maintained utilities |
| root modules | 3 | package marker and guided artifact contracts |

Report-only research lives outside `src/`, in [`experiments/`](../experiments/): the Notebook 16 dual-view ROI
ablation with its evidence-gate analysis, and the Notebook 17 adapter OOD/OE recovery campaign. Their tests are under
[`tests/experiments/`](../tests/experiments/).

The main code paths are:

- [`src/workflows/training.py`](../src/workflows/training.py) for training;
- [`src/training/continual_sd_lora.py`](../src/training/continual_sd_lora.py) for the trainer;
- [`src/workflows/inference.py`](../src/workflows/inference.py) for inference;
- [`src/pipeline/router_adapter_runtime.py`](../src/pipeline/router_adapter_runtime.py) for router-to-adapter handoff;
- [`src/router/router_pipeline.py`](../src/router/router_pipeline.py) for crop/part routing;
- [`src/ood/continual_ood.py`](../src/ood/continual_ood.py) for OOD scoring and calibration;
- [`src/shared/contracts.py`](../src/shared/contracts.py) for typed payloads shared across the system.

## Every authored notebook

| Notebook | Audience | Role |
|---|---|---|
| [Notebook 0](https://colab.research.google.com/github/EfeErim/aads-open-world-plant-disease/blob/main/colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb) | user-facing | audits and materializes the grouped runtime dataset |
| [`1_identify_crop_part_with_router.ipynb`](../colab_notebooks/1_identify_crop_part_with_router.ipynb) | support | router building block used by Notebook 8 |
| [Notebook 2](https://colab.research.google.com/github/EfeErim/aads-open-world-plant-disease/blob/main/colab_notebooks/2_train_continual_sd_lora_adapter.ipynb) | user-facing | primary training; calibrates, evaluates, and exports one adapter |
| [Notebook 3](https://colab.research.google.com/github/EfeErim/aads-open-world-plant-disease/blob/main/colab_notebooks/3_validate_exported_adapter_directly.ipynb) | user-facing | reloads and validates an exported adapter |
| [`4_simple_direct_adapter_test_ui.ipynb`](../colab_notebooks/4_simple_direct_adapter_test_ui.ipynb) | internal maintenance | lightweight direct-adapter inspection |
| [`5_calibrate_router_handoff_thresholds.ipynb`](../colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb) | internal maintenance | evaluates and calibrates handoff thresholds |
| [`6_train_all_continual_sd_lora_adapters.ipynb`](../colab_notebooks/6_train_all_continual_sd_lora_adapters.ipynb) | internal maintenance | runs the Notebook 2 path for all eight targets |
| [`7_ood_oe_quality.ipynb`](../colab_notebooks/7_ood_oe_quality.ipynb) | internal maintenance | audits OOD/OE evidence and review decisions |
| [Notebook 8](https://colab.research.google.com/github/EfeErim/aads-open-world-plant-disease/blob/main/colab_notebooks/8_auto_router_adapter_prediction.ipynb) | user-facing | primary inference/demo; routes an image and returns a decision |
| [`16_ablation_dual_view_inference.ipynb`](../colab_notebooks/16_ablation_dual_view_inference.ipynb) | report-only research | compares full-image and ROI evidence; not the default inference policy |
| [`17_adapter_ood_oe_recovery.ipynb`](../colab_notebooks/17_adapter_ood_oe_recovery.ipynb) | internal recovery | runs candidate selection on dev evidence and a locked-test-once protocol |

Notebook 8 differs intentionally from the maintained private-project version: its public defaults fetch the immutable
public adapter Release and start in single-image mode. The model and routing implementation it calls remains in
`src/`.

Notebook 2 can generate a deterministic synthetic dataset inside the Colab runtime, so a public user can exercise the
training workflow without the private dataset Release. The sample is smoke-test data, is explicitly ineligible for
production evidence and does not support accuracy claims. Notebook 0 accepts the user's own class-root data.
Notebooks 6, 16 and 17 are internal batch/research workflows whose original private experiment inputs are not
redistributed; their source remains public for inspection.

## Notebook support code

The public repository also includes:

- 36 extracted notebook cell scripts under [`scripts/notebook_cells/`](../scripts/notebook_cells/);
- 15 reusable notebook helpers under [`scripts/notebook_helpers/`](../scripts/notebook_helpers/);
- the operational, validation, calibration, release and evidence scripts under [`scripts/`](../scripts/), listed below;
- unit, integration, experiment and Colab smoke tests under [`tests/`](../tests/).

These files matter because the notebooks orchestrate maintained Python code instead of hiding large independent
implementations inside notebook cells.

| Script | Role |
|---|---|
| `audit_code_organization.py` | CI: import-direction and layout audit |
| `validate_config_schema.py` | CI: config schema check |
| `validate_notebook_imports.py` + `notebook_contracts/` | CI: notebook bootstrap and import contracts |
| `build_behavioral_acceptance_summary.py` | CI: rebuilds and checks the behavioral-acceptance evidence summary |
| `benchmark_surfaces.py` | CI (mypy): deterministic orchestration benchmarks |
| `replay_public_demo.py` | replays the controlled demo from checked-in evidence without models |
| `fetch_public_adapters.py` | downloads and verifies the public adapter Release |
| `manage_adapter_release.py` | builds, verifies and publishes the immutable adapter Release |
| `manage_dataset_release.py` | audits, packages and verifies dataset Releases |
| `build_controlled_demo_manifest.py` | how the 48-row controlled-demo manifest was selected (needs private run outputs) |
| `run_demo_checklist.py` + `demo_checklist/` | Notebook 8 batch demo runner and grading |
| `compare_m2_demo_results.py` | Notebook 8: compares two demo runs |
| `run_router_open_world_validation.py` + `router_open_world/` | Notebook 8 open-world router gate |
| `build_router_prototype_bank.py`, `build_taxonomy_registry.py` | Notebook 8: router prototype bank and taxonomy |
| `calibrate_router_prototype_reconciler.py` | Notebook 8: prototype reconciler calibration |
| `evaluate_router_surface.py`, `calibrate_router_surface.py` | Notebook 5: router threshold evaluation and calibration |
| `evaluate_router_part_surface.py` | Notebook 1: part-routing evaluation |
| `evaluate_dataset_layout.py` | Notebook 0: dataset layout audit |
| `prepare_materialization_dataset.py`, `prune_exact_duplicates.py` | Notebook 0: cleaned working copy before materialization |
| `audit_ood_oe_quality.py` | Notebooks 7 and 17: OOD/OE evidence audit |
| `optimize_training_runs.py` | analyses a comparable training cohort and writes Bayesian proposals |

## Deliberate exclusions

The following private-tree surfaces are not application source and are not copied:

- `.ai/`, `.agents/` and the legacy `skills/` tree: local AI-agent instructions, not runtime code;
- `scripts/validate_ai_suite.py` and its test: validation for those private AI instructions;
- `tmp/dataset_repair/`: one-off local repair utilities;
- `runs/**/notebooks/*.executed.ipynb`: generated notebook outputs, not authored notebooks;
- `runs/`, `outputs/` and `.runtime_tmp/`: generated experiments, caches and telemetry;
- training images: redistribution permission is not documented for every source image;
- model binaries in Git history: released separately with checksums in
  [`aads-public-demo-v1.1.2`](https://github.com/EfeErim/aads-open-world-plant-disease/releases/tag/aads-public-demo-v1.1.2);
- private credentials, internal project state and unpublished dataset manifests.

This boundary keeps the repository cloneable without pretending that generated outputs or internal automation are
part of the product implementation.
