# Data layout

The original research datasets are not distributed with this repository. The intended path is to provide your own
images through [Notebook 0](https://colab.research.google.com/github/EfeErim/aads-open-world-plant-disease/blob/master/colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb)
and train with [Notebook 2](https://colab.research.google.com/github/EfeErim/aads-open-world-plant-disease/blob/master/colab_notebooks/2_train_continual_sd_lora_adapter.ipynb).

For a zero-data plumbing test, Notebook 2 can automatically create 56 deterministic synthetic images across the
`continual`, `val`, `test`, `ood`, and `oe` roles. Its manifest marks the data as `production_eligible: false`: it is
useful for checking that the Colab workflow runs, not for measuring accuracy, OOD performance, or readiness.

Notebook 0 accepts a class-root dataset:

```text
data/class_root_dataset/<dataset_name>/
  <class>/
    <images>
```

For your own uploaded data in Notebook 0, set `DATASET_RELEASE_TAG = ""` so the notebook uses the
audit/materialization path instead of looking for a release-backed dataset.

Training uses the prepared runtime layout:

```text
data/prepared_runtime_datasets/<crop>__<part>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
  ood/*
  oe/*
```

`ood/` contains unknown inputs used for calibration/evaluation. `oe/` is optional Outlier Exposure data and must not
overlap the validation or locked test sets. Keep split membership and provenance fixed when comparing runs.

The preparation and integrity logic is implemented in [`../src/data/`](../src/data/); Notebook 0 is its user-facing
Colab interface.
