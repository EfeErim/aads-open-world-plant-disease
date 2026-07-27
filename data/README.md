# Data layout

Datasets are not distributed with this repository. Add your own images locally; the `.gitignore` keeps them out of
Git by default.

Notebook 0 accepts a class-root dataset:

```text
data/class_root_dataset/<dataset_name>/
  <class>/
    <images>
```

For local data in Notebook 0, set `DATASET_RELEASE_TAG = ""` so it uses the audit/materialization path instead of
looking for a release-backed dataset.

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

The preparation and integrity logic is implemented in [`../src/data/`](../src/data/) and the main dataset notebook is
[`../colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb`](../colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb).
