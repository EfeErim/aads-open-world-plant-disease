# Documentation

The root [README](../README.md) is the shortest entry point. Training and inference are Colab-first; local CLI
wrappers exist for development and CI, not as the normal user workflow.

## Colab workflow

- [Notebook 8: inference and demo](https://colab.research.google.com/github/EfeErim/aads-open-world-plant-disease/blob/master/colab_notebooks/8_auto_router_adapter_prediction.ipynb)
- [Notebook 2: train one adapter](https://colab.research.google.com/github/EfeErim/aads-open-world-plant-disease/blob/master/colab_notebooks/2_train_continual_sd_lora_adapter.ipynb)
- [Notebook 0: audit and prepare a dataset](https://colab.research.google.com/github/EfeErim/aads-open-world-plant-disease/blob/master/colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb)
- [Notebook 3: validate an exported adapter](https://colab.research.google.com/github/EfeErim/aads-open-world-plant-disease/blob/master/colab_notebooks/3_validate_exported_adapter_directly.ipynb)

## Reference

- [Architecture overview](architecture/overview.md)
- [Code organization map](architecture/code_organization_map.md)
- [Complete source and notebook map](source_and_notebook_map.md)
- [Methodology, literature and current results](methodology_and_results.md)
- [Public result records](../evidence/)
- [Dataset layout](../data/README.md)
- [Canonical training workflow](../src/workflows/training.py)
- [Canonical inference workflow](../src/workflows/inference.py)

Generated reports, private dataset manifests and internal project-management notes are intentionally not copied into
the public repository. The maintained implementation is under `src/`; notebooks and scripts call that code instead
of carrying separate model implementations.
