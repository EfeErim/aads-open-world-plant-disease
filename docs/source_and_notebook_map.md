# Source and notebook map

What the public repository contains. It was cut from the private project tree at commit
`539397bb72bde59e4b092ac1286b5415fe78dbac` and no longer mirrors it file for file. Colab notebooks call the same
modules as the developer CLI and the test suite.

## Application source

| Package | Responsibility |
|---|---|
| [`src/training/`](../src/training/) | per-target LoRA adapter training (named `continual_sd_lora` for compatibility), validation, persistence and reporting |
| [`src/router/`](../src/router/) | crop/part routing, taxonomy, prototypes, calibration and abstention |
| [`src/data/`](../src/data/) | dataset contracts, lineage, releases, public smoke data, OOD splits and integrity |
| [`src/colab/`](../src/colab/) | Colab bootstrap, widgets, telemetry, checkpoint mirroring and notebook run helpers |
| [`src/shared/`](../src/shared/) | typed contracts, serialization/path/hash utilities and the GitHub Releases client |
| [`src/ood/`](../src/ood/) | OOD scoring, conformal prediction and behavioral acceptance |
| [`src/workflows/`](../src/workflows/) | training and inference entry points and their run stages |
| [`src/pipeline/`](../src/pipeline/) | adapter discovery, release loading, input guard and inference payloads |
| [`src/adapter/`](../src/adapter/) | adapter configuration, checkpointing and lifecycle helpers |
| [`src/core/`](../src/core/) | configuration loading and migrations |
| [`src/app/`](../src/app/) | developer CLI (`aads inference`, `aads training`) |

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
| [`5_calibrate_router_handoff_thresholds.ipynb`](../colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb) | internal maintenance | evaluates and calibrates handoff thresholds |
| [`6_train_all_continual_sd_lora_adapters.ipynb`](../colab_notebooks/6_train_all_continual_sd_lora_adapters.ipynb) | internal maintenance | runs the Notebook 2 path for all eight targets |
| [Notebook 8](https://colab.research.google.com/github/EfeErim/aads-open-world-plant-disease/blob/main/colab_notebooks/8_auto_router_adapter_prediction.ipynb) | user-facing | primary inference/demo; routes an image and returns a decision |

Notebook 8 differs intentionally from the maintained private-project version: its public defaults fetch the immutable
public adapter Release and start in single-image mode. The model and routing implementation it calls remains in
`src/`.

Notebook 2 can generate a deterministic synthetic dataset inside the Colab runtime, so a public user can exercise the
training workflow without the private dataset Release. The sample is smoke-test data, is explicitly ineligible for
production evidence and does not support accuracy claims. Notebook 0 accepts the user's own class-root data.
Notebook 6 is an internal batch workflow whose private inputs are not redistributed.

## Notebook support code

The public repository also includes:

- extracted notebook cell scripts under [`scripts/notebook_cells/`](../scripts/notebook_cells/);
- reusable notebook helpers under [`scripts/notebook_helpers/`](../scripts/notebook_helpers/);
- the operational, validation, calibration, release and evidence scripts under [`scripts/`](../scripts/), listed below;
- unit, integration and Colab smoke tests under [`tests/`](../tests/).

| Script | Role |
|---|---|
| `build_behavioral_acceptance_summary.py` | rebuilds and checks the behavioral-acceptance evidence summary |
| `replay_public_demo.py` | replays the controlled demo from checked-in evidence without models |
| `fetch_public_adapters.py` | downloads and verifies the public adapter Release |
| `manage_adapter_release.py` | builds, verifies and publishes the immutable adapter Release |
| `manage_dataset_release.py` | audits, packages and verifies dataset Releases |
| `build_controlled_demo_manifest.py` | how the 48-row controlled-demo manifest was selected (needs private run outputs) |
| `run_demo_checklist.py` + `demo_checklist/` | Notebook 8 batch demo runner and grading |
| `compare_m2_demo_results.py` | Notebook 8: compares two demo runs |
| `run_router_open_world_validation.py` + `router_open_world/` | Notebook 8 open-world router gate |
| `build_router_prototype_bank.py` | Notebook 8: router prototype bank and taxonomy |
| `calibrate_router_prototype_reconciler.py` | Notebook 8: prototype reconciler calibration |
| `evaluate_router_surface.py`, `calibrate_router_surface.py` | Notebook 5: router threshold evaluation and calibration |
| `evaluate_router_part_surface.py` | Notebook 1: part-routing evaluation |
| `evaluate_dataset_layout.py` | Notebook 0: dataset layout audit |
| `prepare_materialization_dataset.py`, `prune_exact_duplicates.py` | Notebook 0: cleaned working copy before materialization |

## Deliberate exclusions

The following private-tree surfaces are not application source and are not copied:

- `tmp/dataset_repair/`: one-off local repair utilities;
- `runs/**/notebooks/*.executed.ipynb`: generated notebook outputs, not authored notebooks;
- `runs/`, `outputs/` and `.runtime_tmp/`: generated experiments, caches and telemetry;
- training images: redistribution permission is not documented for every source image;
- model binaries in Git history: released separately with checksums in
  [`aads-public-demo-v1.1.2`](https://github.com/EfeErim/aads-open-world-plant-disease/releases/tag/aads-public-demo-v1.1.2);
- private credentials, internal project state and unpublished dataset manifests.
