# AADS

[![CI](https://github.com/EfeErim/bitirmeprojesi/actions/workflows/ci.yml/badge.svg)](https://github.com/EfeErim/bitirmeprojesi/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[Application source](src/)** · **[Training workflow](src/workflows/training.py)** ·
**[Inference workflow](src/workflows/inference.py)** · **[Tests](tests/)** ·
**[Colab notebooks](colab_notebooks/)** ·
**[Methodology and results](docs/methodology_and_results.md)** ·
**[Source map](docs/source_and_notebook_map.md)** ·
**[Adapter release](https://github.com/EfeErim/bitirmeprojesi/releases/tag/aads-public-demo-v1.1.1)**

AADS is my graduation project on open-world plant disease recognition. I built it around a practical failure mode:
a classifier should not force a disease label when the image shows the wrong crop, the wrong plant part, or an
unknown condition.

The repository contains the actual training, routing, adapter inference, OOD calibration and Colab code used by the
project. Model weights are kept in a GitHub Release so the Git history stays usable. Training images are not included
because I cannot document redistribution permission for every source image.

The public tree contains all 130 maintained project files under `src/`, one additional public-sample module, and all
11 authored Colab notebooks. The
[source and notebook map](docs/source_and_notebook_map.md) lists every notebook, module group and deliberate
non-source exclusion.

## System outline

1. The router estimates crop and plant part.
2. The matching crop/part adapter is loaded.
3. The adapter predicts a known class or sends the input to review.
4. Training and evaluation write explicit OOD and readiness reports.

The main implementation is under [`src/`](src/):

- [`src/router/`](src/router/) — crop/part routing and open-set policy
- [`src/training/`](src/training/) — continual SD-LoRA training and calibration
- [`src/pipeline/`](src/pipeline/) — adapter discovery, loading and inference payloads
- [`src/ood/`](src/ood/) — OOD/OE evaluation and readiness logic
- [`src/workflows/`](src/workflows/) — stable training and inference entry points
- [`src/shared/contracts.py`](src/shared/contracts.py) — typed contracts shared across the pipeline

## Results, without overselling them

| Evidence | Result |
|---|---:|
| Fixed controlled demo | 48 / 48 |
| Disease rows in that demo | 36 / 36 |
| Review/abstain rows in that demo | 12 / 12 |
| Adapters passing the separate production-readiness gate | 0 / 8 |

The 48/48 result is a replayable result on one fixed demo set; it is not field accuracy. On separate unknown-input
tests, the adapters still accept too many unsupported cases. Among the five latest tracked artifacts with eligible
same-crop unknown-disease rows, rejection ranges from `0.203` to `0.611`; the other three have no eligible rows in
their selected artifact. The maintained gate requires `1.0`, so these weights are suitable for code review and
controlled experiments, not autonomous diagnosis.

The detailed [methodology and results note](docs/methodology_and_results.md) explains the DINOv3 + LoRA training
design, SAM3/BioCLIP routing, energy/Mahalanobis/kNN OOD scores, Outlier Exposure, conformal prediction, the literature
behind those choices and the latest per-adapter acceptance metrics. The machine-readable 0/8 breakdown is in
[`latest_behavioral_acceptance_summary.json`](evidence/latest_behavioral_acceptance_summary.json).

Verify that summary directly from the eight checked-in per-target records:

```bash
python scripts/build_behavioral_acceptance_summary.py --check
```

The saved evidence is in [`evidence/`](evidence/):

- [`controlled_demo_summary.json`](evidence/controlled_demo_summary.json)
- [`controlled_demo_rows.json`](evidence/controlled_demo_rows.json)
- [`public_asset_manifest.json`](evidence/public_asset_manifest.json)

## Clone and inspect

Python 3.11 is the maintained version.

```bash
git clone https://github.com/EfeErim/bitirmeprojesi.git
cd bitirmeprojesi
python -m venv .venv
```

Activate the environment, then install the runtime and the repository:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m src.app.cli --help
```

The dependency set includes PyTorch, Transformers and PEFT, so installation is not small. To inspect the code without
downloading models, run the structural checks and unit tests:

```bash
python scripts/audit_code_organization.py
python scripts/validate_config_schema.py
python scripts/validate_notebook_imports.py
python -m pytest tests/unit -q
```

Windows users can replace `python` with `.\scripts\python.cmd`.

## Download the released adapters

The eight demo adapter bundles are about 518 MB in total. This command downloads them from the public immutable
Release, verifies every SHA-256 hash and materializes the directory layout expected by the inference code:

```bash
python scripts/fetch_public_adapters.py
```

Full inference additionally needs a compatible GPU setup and access to the gated Hugging Face backbones configured in
[`config/base.json`](config/base.json). Once those are available:

```bash
python -m src.app.cli inference path/to/image.jpg --device cuda
```

[`colab_notebooks/8_auto_router_adapter_prediction.ipynb`](colab_notebooks/8_auto_router_adapter_prediction.ipynb) is
the notebook version of the router-to-adapter path.

## Train with your own data

For a zero-data smoke test, materialize the deterministic synthetic sample:

```bash
python scripts/materialize_public_sample_dataset.py --target tomato__leaf
```

Notebook 2 uses this public sample by default, so cloning the repository does not require access to the private
training dataset Release. The sample exists only to exercise data loading, training, evaluation and export; it is
marked `production_eligible: false` and provides no evidence of model quality. Actual DINOv3 training still requires
access to the gated Hugging Face backbone configured in [`config/base.json`](config/base.json).

For a real experiment, put your own images into this runtime layout:

```text
data/prepared_runtime_datasets/<crop>__<part>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
  ood/*
  oe/*
```

Run the actual training workflow through the CLI:

```bash
python -m src.app.cli training tomato \
  data/prepared_runtime_datasets/tomato__leaf \
  outputs/tomato_leaf \
  --part leaf --device cuda
```

[`colab_notebooks/2_train_continual_sd_lora_adapter.ipynb`](colab_notebooks/2_train_continual_sd_lora_adapter.ipynb)
is the original experiment notebook. It defaults to the synthetic smoke profile; maintainers can explicitly select
the immutable dataset Release, while public users can provide the runtime layout above. See
[`data/README.md`](data/README.md) for the input contract and the limits of the sample data.

## Repository map

- `src/` — application and ML pipeline source
- `scripts/` — validation tools and notebook helpers
- `tests/` — unit, integration and Colab smoke tests
- `config/` — checked-in runtime configuration
- `colab_notebooks/` — training, validation and inference notebooks
- `docs/architecture/` — architecture and code-ownership notes
- `evidence/` — compact, checked-in result records
- `data/`, `models/`, `runs/`, `outputs/` — local or generated content, ignored by Git

For a deeper code map, read
[`docs/architecture/code_organization_map.md`](docs/architecture/code_organization_map.md).

## License

The source code is available under the [MIT License](LICENSE). Released DINOv3-derived adapter bundles also carry the
upstream DINOv3 license inside each bundle.
