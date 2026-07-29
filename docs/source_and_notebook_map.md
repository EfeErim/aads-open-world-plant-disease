# Source and notebook map

This page answers a simple question: what is actually included in the public repository?

The public tree was compared against the maintained project tree at source commit
`539397bb72bde59e4b092ac1286b5415fe78dbac`. The comparison is path-based because the public edition has a few
necessary changes for anonymous Release access, safer defaults and cross-platform CI.

## Coverage snapshot

| Surface | Maintained project | Public repository | Public coverage |
|---|---:|---:|---:|
| maintained `src/` files | 130 | 130 | 100% |
| public-only sample-data module | 0 | 1 | added for public use |
| authored Colab notebooks | 11 | 11 | 100% |
| `colab_notebooks/requirements_colab.txt` | 1 | 1 | 100% |
| `config/` files | 3 | 3 | 100% |
| operational scripts shared by both trees | 177 | 177 | 100% |
| public-only operational or extracted scripts | 0 | 5 | added for public use |
| tests shared by both trees | 166 | 166 | 100% |
| public-only focused tests | 0 | 3 | added for public use |

The application implementation is not embedded only in notebooks. Colab notebooks call the same maintained modules
used by the developer-only CLI and the test suite.

## Application source

The 137 files under [`src/`](../src/) are grouped as follows. Six public modules split large responsibilities from
their original compatibility entry points; they reorganize the same implementation rather than adding a second
pipeline.

| Package | Files | Responsibility |
|---|---:|---|
| [`src/training/`](../src/training/) | 28 | continual SD-LoRA training, validation, persistence and reporting |
| [`src/router/`](../src/router/) | 26 | crop/part routing, taxonomy, prototypes, calibration and abstention |
| [`src/data/`](../src/data/) | 20 | dataset contracts, lineage, releases, public smoke data, OOD splits and integrity |
| [`src/ood/`](../src/ood/) | 14 | OOD scoring, conformal prediction and behavioral acceptance |
| [`src/pipeline/`](../src/pipeline/) | 14 | adapter discovery, release loading, inference payloads and evidence analysis |
| [`src/shared/`](../src/shared/) | 10 | typed contracts and shared serialization/path utilities |
| [`src/workflows/`](../src/workflows/) | 8 | stable training and inference entry points |
| [`src/adapter/`](../src/adapter/) | 4 | adapter configuration, checkpointing and lifecycle helpers |
| [`src/core/`](../src/core/) | 3 | configuration loading and core runtime support |
| [`src/notebook/`](../src/notebook/) | 3 | notebook bootstrap and Git integration |
| [`src/app/`](../src/app/) | 2 | developer and maintenance CLI surface |
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

| Notebook | Audience | Role |
|---|---|---|
| [Notebook 0](https://colab.research.google.com/github/EfeErim/aads-open-world-plant-disease/blob/master/colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb) | user-facing | audits and materializes the grouped runtime dataset |
| [`1_identify_crop_part_with_router.ipynb`](../colab_notebooks/1_identify_crop_part_with_router.ipynb) | support | router building block used by Notebook 8 |
| [Notebook 2](https://colab.research.google.com/github/EfeErim/aads-open-world-plant-disease/blob/master/colab_notebooks/2_train_continual_sd_lora_adapter.ipynb) | user-facing | primary training; calibrates, evaluates, and exports one adapter |
| [Notebook 3](https://colab.research.google.com/github/EfeErim/aads-open-world-plant-disease/blob/master/colab_notebooks/3_validate_exported_adapter_directly.ipynb) | user-facing | reloads and validates an exported adapter |
| [`4_simple_direct_adapter_test_ui.ipynb`](../colab_notebooks/4_simple_direct_adapter_test_ui.ipynb) | internal maintenance | lightweight direct-adapter inspection |
| [`5_calibrate_router_handoff_thresholds.ipynb`](../colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb) | internal maintenance | evaluates and calibrates handoff thresholds |
| [`6_train_all_continual_sd_lora_adapters.ipynb`](../colab_notebooks/6_train_all_continual_sd_lora_adapters.ipynb) | internal maintenance | runs the Notebook 2 path for all eight targets |
| [`7_ood_oe_quality.ipynb`](../colab_notebooks/7_ood_oe_quality.ipynb) | internal maintenance | audits OOD/OE evidence and review decisions |
| [Notebook 8](https://colab.research.google.com/github/EfeErim/aads-open-world-plant-disease/blob/master/colab_notebooks/8_auto_router_adapter_prediction.ipynb) | user-facing | primary inference/demo; routes an image and returns a decision |
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
- the remaining operational, validation, calibration and evidence scripts under [`scripts/`](../scripts/);
- 169 tracked unit, integration, Colab smoke and test-fixture files under [`tests/`](../tests/), including three
  public-only tests for sample data, evidence rebuilding and the models-free demo replay.

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
  [`aads-public-demo-v1.1.2`](https://github.com/EfeErim/aads-open-world-plant-disease/releases/tag/aads-public-demo-v1.1.2);
- private credentials, internal project state and unpublished dataset manifests.

This boundary keeps the repository cloneable without pretending that generated outputs or internal automation are
part of the product implementation.
