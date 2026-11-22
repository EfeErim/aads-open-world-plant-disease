# Architecture

What is not obvious from the package layout: how configuration is resolved, which data layouts the code expects,
what an inference call can return, and which output files carry a decision.

## Configuration

`src/core/config_manager.py` loads `config/base.json` from the repository (independent of the working directory) and,
for the `colab` environment, deep-merges `config/colab.json` on top. Both files declare `config_schema_version: 2`;
`validate_config_payload` rejects any other schema version and removed keys instead of upgrading or repairing them.
`src/training/services/config_surface.py` normalizes `training.continual` into the shape the trainer reads.

## Data layouts

Notebook 0 takes a class-root dataset you prepare by hand:

```text
data/class_root_dataset/<dataset_name>/<class>/<images>
```

Set `DATASET_RELEASE_TAG = ""` in Notebook 0 to use your own uploaded data instead of a dataset release. Notebook 0
audits it (exact and near duplicates via pHash, DINOv3 and BioCLIP-2.5 embeddings, source-style grouping, cross-class
label conflicts), splits families 60/20/20, keeps risky but usable samples in `continual` only, and writes
`split_manifest.json`, `human_review_packet.json` and `label_review_summary.json`. It blocks materialization while
cross-class conflicts remain. Training reads the result:

```text
data/prepared_runtime_datasets/<crop>__<part>/
  continual/<class>/*   training split
  val/<class>/*
  test/<class>/*
  ood/*                 unknown inputs for OOD calibration and evaluation
  oe/*                  optional Outlier Exposure pool, disjoint from val, test and ood
```

The loader splits `ood/` into a dev assignment (score and threshold selection) and a held-out test assignment, recorded
in `ood/ood_split_manifest.json`. The router evaluation script (`scripts/evaluate_router_surface.py`) uses a
separate evaluation-only layout under `data/router_eval/` with `id/<crop>/<part>`, `negatives/off_crop`,
`negatives/non_plant`, `ambiguous` and `wrong_part` folders.

## Training

`TrainingWorkflow.run` (`src/workflows/training.py`) prepares loaders and class metadata (`src/workflows/run_setup.py`),
builds the adapter (`src/adapter/independent_crop_adapter.py`) around the trainer
(`src/training/continual_sd_lora.py`), runs epochs through `src/training/session.py`, calibrates OOD, evaluates,
and exports:

```text
<output_dir>/<crop>/<part>/
  continual_sd_lora_adapter/   bundle loaded at inference time
  training_metrics/            training/, validation/, test/, ood_benchmark/ reports
```

When a target has no real `ood/` data, `src/training/services/ood_benchmark.py` runs a leave-one-class-out benchmark
instead. Several JSON files look like decisions; only one is authoritative:

| File | Role |
|---|---|
| `adapter_behavioral_acceptance.json` | the adapter verdict: supported inputs classified correctly and same-crop unsupported diseases rejected |
| `production_readiness.json` | broader deployment diagnostics; does not override the verdict |
| `validation/metric_gate.json`, `test/metric_gate.json` | split-local diagnostics |

Each run also writes `training/experiment_manifest.json` and `training/optimization_record.json`. Runs are only
compared when crop, part, backbone and dataset lineage (`<dataset_key>::<split_manifest_sha256>`) match;
`python -m src.training.services.run_registry --runs-root runs` indexes them under `runs/_index/`.

## Inference

`InferenceWorkflow.predict` (`src/workflows/inference.py`) builds `src/pipeline/router_adapter_runtime.py`, which
routes the image, gates the handoff and loads the adapter from
`models/adapters/<crop>/<part>/continual_sd_lora_adapter/`. The payload status says where a request stopped:

| Status | Meaning |
|---|---|
| `non_plant_rejected` | the optional input guard rejected the image (disabled by default) |
| `unknown_crop` | the router ran but found no supported crop |
| `router_unavailable` | the router failed to load or errored; the next request retries a clean load |
| `router_uncertain` | the primary detection is below the confidence or margin floor, or disagrees with a crop hint |
| `adapter_unavailable` | no adapter bundle for the routed crop/part |

None of these runs the adapter, so they carry no adapter OOD verdict. A plain `crop_hint` is untrusted: the router must
agree. `trust_crop_hint=True` skips the router and reports `trusted_hint_skipped` in the `router` block. `part_hint` is
metadata only. Notebook 1 runs the router alone; Notebook 3 loads one adapter directly without the router.

## Dependency rules

```text
notebooks -> scripts/notebook_cells -> scripts/notebook_helpers -> src
scripts -> src
src/app -> src/workflows -> other src packages -> src/shared
```

`src/` must not import `scripts` or `tests` (enforced by Ruff `TID251`). Notebook cell files call `src/`
and do not hold training, routing, OOD or inference logic of their own.
