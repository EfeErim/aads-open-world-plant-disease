# AADS

[![CI](https://github.com/EfeErim/bitirmeprojesi/actions/workflows/ci.yml/badge.svg)](https://github.com/EfeErim/bitirmeprojesi/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[Source code](src/aads_public/)** · **[Tests](tests/)** · **[Notebooks](notebooks/)** ·
**[Model files](https://github.com/EfeErim/bitirmeprojesi/releases/tag/aads-public-demo-v1.1.1)**

AADS is my graduation project on plant disease recognition.

The main problem I worked on was uncertainty. A normal classifier always returns one of its known labels, even when
the image contains the wrong plant, the wrong plant part, or something that is not a plant at all. In AADS, the model
can return `review` instead of forcing a disease prediction.

The complete project uses Python, PyTorch, DINOv3 and PEFT/LoRA. This repository is a smaller public version containing
the core decision logic, tests, notebooks and a reproducible snapshot of the demo results.

## How it works

1. The router estimates the crop and plant part.
2. The image is sent to the matching crop/part adapter.
3. Unsupported or uncertain inputs are sent to review.
4. Supported inputs receive a disease prediction.

I trained eight adapters for the fruit and leaf targets of apricot, grape, strawberry and tomato. The continual
training objective combines classification loss, feature distillation, replay and adapter regularization.

## Results

| Test | Result |
|---|---:|
| Controlled demo | 48 / 48 |
| Correct disease predictions | 36 / 36 |
| Correct review decisions | 12 / 12 |
| Production-readiness checks | 0 / 8 |

The `48/48` result belongs to one fixed demo set. It is not a claim of field accuracy. Separate held-out tests show
that the adapters still accept too many unknown cases, so none of them is marked production-ready. The recorded
OOD false-positive rates range from `0.283` to `0.873`.

The public evidence files are available here:

- [`controlled_demo_summary.json`](evidence/controlled_demo_summary.json)
- [`controlled_demo_rows.json`](evidence/controlled_demo_rows.json)
- [`public_asset_manifest.json`](evidence/public_asset_manifest.json)

## Run it

```bash
git clone https://github.com/EfeErim/bitirmeprojesi.git
cd bitirmeprojesi
python -m pip install -e ".[dev]"
python -m aads_public replay
python -m pytest
```

`replay` checks the saved 48-row demo record and its hashes. It does not run the full model or require a GPU.

You can also open the
[evidence notebook in Colab](https://colab.research.google.com/github/EfeErim/bitirmeprojesi/blob/master/notebooks/evidence_snapshot.ipynb)
or read the smaller [`continual_objective.ipynb`](notebooks/continual_objective.ipynb) example.

## Source code

The public Python package is in [`src/aads_public/`](src/aads_public/). The main files are:

- [`policy.py`](src/aads_public/policy.py): routing and review rules
- [`training.py`](src/aads_public/training.py): continual-learning loss and replay buffer
- [`evidence.py`](src/aads_public/evidence.py): validation of the saved demo results
- [`release.py`](src/aads_public/release.py): checksum-verified artifact downloads
- [`tests/`](tests): unit tests for the public package

The adapter files are in the
[v1.1.1 release](https://github.com/EfeErim/bitirmeprojesi/releases/tag/aads-public-demo-v1.1.1).
Each file is listed with its size and SHA-256 hash in the public manifest.

More detail is available in the [model card](docs/MODEL_CARD.md) and
[engineering notes](docs/ENGINEERING_NOTES.md).

## Limitations

- The original training datasets are not included because their file-level redistribution permissions are unclear.
- The public package does not contain the full training and inference application.
- The current adapters are suitable for experiments and controlled demos, not autonomous diagnosis or treatment advice.

## License

The public code is available under the [MIT License](LICENSE). The released DINOv3-derived files also include the
original DINOv3 license.
