# Source and notebook map

This page answers a simple question: what is actually included in the public repository?

The public tree was compared against the maintained project tree at source commit
`539397bb72bde59e4b092ac1286b5415fe78dbac`. The comparison is path-based because the public edition has a few
necessary changes for anonymous Release access, safer defaults and cross-platform CI.

## Coverage snapshot

| Surface | Maintained project | Public repository | Public coverage |
|---|---:|---:|---:|
| `src/` files | 130 | 130 | 100% |
| authored Colab notebooks | 11 | 11 | 100% |
| `colab_notebooks/requirements_colab.txt` | 1 | 1 | 100% |
| `config/` files | 3 | 3 | 100% |
| operational scripts shared by both trees | 177 | 177 | 100% |
| public-only adapter downloader | 0 | 1 | added for public use |
| tests shared by both trees | 166 | 166 | 100% |

The application implementation is not embedded only in notebooks. Notebooks call the same maintained modules used by
the CLI and tests.

## Application source

The 130 files under [`src/`](../src/) are grouped as follows:

| Package | Files | Responsibility |
|---|---:|---|
| [`src/training/`](../src/training/) | 27 | continual SD-LoRA training, validation, persistence and reporting |
| [`src/router/`](../src/router/) | 26 | crop/part routing, taxonomy, prototypes, calibration and abstention |
| [`src/data/`](../src/data/) | 15 | dataset contracts, lineage, releases, OOD splits and integrity |
| [`src/ood/`](../src/ood/) | 14 | OOD scoring, conformal prediction and behavioral acceptance |
| [`src/pipeline/`](../src/pipeline/) | 13 | adapter discovery, release loading, inference payloads and evidence analysis |
| [`src/shared/`](../src/shared/) | 10 | typed contracts and shared serialization/path utilities |
| [`src/workflows/`](../src/workflows/) | 8 | stable training and inference entry points |
| [`src/adapter/`](../src/adapter/) | 4 | adapter configuration, checkpointing and lifecycle helpers |
| [`src/core/`](../src/core/) | 3 | configuration loading and core runtime support |
| [`src/notebook/`](../src/notebook/) | 3 | notebook bootstrap and Git integration |
| [`src/app/`](../src/app/) | 2 | CLI surface |
| [`src/utils/`](../src/utils/) | 2 | small maintained utilities |
| root modules | 2 | guided artifact contracts |
| package marker | 1 | `src/__init__.py` |

The main code paths are:

- [`src/workflows/training.py`](../src/workflows/training.py) for training;
- [`src/training/continual_sd_lora.py`](../src/training/continual_sd_lora.py) for the trainer;
- [`src/workflows/inference.py`](../src/workflows/inference.py) for inference;
- [`src/pipeline/router_adapter_runtime.py`](../src/pipeline/router_adapter_runtime.py) for router-to-adapter handoff;
- [`src/router/router_pipeline.py`](../src/router/router_pipeline.py) for crop/part routing;
- [`src/ood/continual_ood.py`](../src/ood/continual_ood.py) for OOD scoring and calibration;
- [`src/shared/contracts.py`](../src/shared/contracts.py) for typed payloads shared across the system.

## Every authored notebook

| Notebook | Role | What it runs |
|---|---|---|
| [`0_prepare_grouped_dataset_for_training.ipynb`](../colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb) | data preparation | validates and materializes the grouped runtime dataset |
| [`1_identify_crop_part_with_router.ipynb`](../colab_notebooks/1_identify_crop_part_with_router.ipynb) | router building block | crop/part routing used by Notebook 8 |
| [`2_train_continual_sd_lora_adapter.ipynb`](../colab_notebooks/2_train_continual_sd_lora_adapter.ipynb) | primary training | trains, calibrates, evaluates and exports one adapter |
| [`3_validate_exported_adapter_directly.ipynb`](../colab_notebooks/3_validate_exported_adapter_directly.ipynb) | adapter validation | reloads an exported adapter and checks direct inference |
| [`4_simple_direct_adapter_test_ui.ipynb`](../colab_notebooks/4_simple_direct_adapter_test_ui.ipynb) | maintenance UI | lightweight direct-adapter inspection |
| [`5_calibrate_router_handoff_thresholds.ipynb`](../colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb) | router calibration | evaluates and calibrates handoff thresholds |
| [`6_train_all_continual_sd_lora_adapters.ipynb`](../colab_notebooks/6_train_all_continual_sd_lora_adapters.ipynb) | batch training | runs the maintained Notebook 2 path for all eight targets |
| [`7_ood_oe_quality.ipynb`](../colab_notebooks/7_ood_oe_quality.ipynb) | evidence QC | audits OOD/OE evidence and review decisions |
| [`8_auto_router_adapter_prediction.ipynb`](../colab_notebooks/8_auto_router_adapter_prediction.ipynb) | primary demo | routes an image, loads the matching adapter and returns a decision |
| [`16_ablation_dual_view_inference.ipynb`](../colab_notebooks/16_ablation_dual_view_inference.ipynb) | report-only research | compares full-image and ROI evidence; it is not the default inference policy |
| [`17_adapter_ood_oe_recovery.ipynb`](../colab_notebooks/17_adapter_ood_oe_recovery.ipynb) | bounded recovery | runs candidate selection on dev evidence and a locked-test-once recovery protocol |

Notebook 8 differs intentionally from the maintained private-project version: its public defaults fetch the immutable
public adapter Release and start in single-image mode. The model and routing implementation it calls remains in
`src/`.

## Notebook support code

The public repository also includes:

- 36 extracted notebook cell scripts under [`scripts/notebook_cells/`](../scripts/notebook_cells/);
- 15 reusable notebook helpers under [`scripts/notebook_helpers/`](../scripts/notebook_helpers/);
- the remaining operational, validation, calibration and evidence scripts under [`scripts/`](../scripts/);
- 166 tracked unit, integration, Colab smoke and test-fixture files under [`tests/`](../tests/).

These files matter because the notebooks orchestrate maintained Python code instead of hiding large independent
implementations inside notebook cells.

## Deliberate exclusions

The following private-tree surfaces are not application source and are not copied:

- `.ai/`, `.agents/` and the legacy `skills/` tree: local AI-agent instructions, not runtime code;
- `scripts/validate_ai_suite.py` and its test: validation for those private AI instructions;
- `tmp/dataset_repair/`: one-off local repair utilities;
- `runs/**/notebooks/*.executed.ipynb`: generated notebook outputs, not authored notebooks;
- `runs/`, `outputs/` and `.runtime_tmp/`: generated experiments, caches and telemetry;
- training images: redistribution permission is not documented for every source image;
- model binaries in Git history: released separately with checksums in
  [`aads-public-demo-v1.1.1`](https://github.com/EfeErim/bitirmeprojesi/releases/tag/aads-public-demo-v1.1.1);
- private credentials, internal project state and unpublished dataset manifests.

This boundary keeps the repository cloneable without pretending that generated outputs or internal automation are
part of the product implementation.
