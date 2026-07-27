# Architecture Overview

This document explains the current AADS v6 architecture in plain language.

The project is intentionally narrow:

- one maintained training workflow
- one maintained router-driven inference workflow
- one adapter family
- a small set of notebook helper surfaces around those same workflows

If you are new to the repo, read [../../README.md](../../README.md) before this file.

## The Big Picture

At a high level, AADS v6 has four layers:

1. Configuration: load JSON config, merge environment overrides, normalize the public training surface.
2. Training: build one crop adapter, train it, calibrate OOD, and write artifacts.
3. Inference: route an image to the right crop adapter and return a typed prediction payload.
4. Notebook support: thin Colab wrappers for training, inference, telemetry, and smoke testing.

## Important Terms

- `workflow`: the stable app-facing facade that other surfaces should call
- `adapter`: the saved crop-specific bundle produced by training and loaded at inference time
- `router`: the crop-and-part analysis stage used before adapter inference
- `telemetry`: the logs and mirrored artifacts written during notebook runs
- `artifact`: any generated output file such as a JSON report, CSV, plot, or exported adapter
- `runtime dataset`: the split dataset layout used by the training workflow

## Configuration Flow

The configuration path is intentionally simple.

1. `src/core/config_manager.py` loads `config/base.json`.
2. If an environment is requested, such as `colab`, it deep-merges `config/colab.json` on top.
3. `src/core/config_migrations.py` upgrades each config file through the declared `config_schema_version`.
4. `src/training/services/config_surface.py` normalizes the public training surface under `training.continual`.
5. Config files must declare `config_schema_version: 2` and use the canonical `training.continual` / `colab.training.checkpoint_every_n_steps` surface.
6. `src/training/quantization.py` rejects prohibited 4-bit flags before the merged config is used.

Why this matters:

- users edit a small public config surface
- workflow code sees a normalized shape
- unsupported config aliases fail fast instead of being repaired silently

## Training Architecture

The canonical training entrypoint is:

```text
src/workflows/training.py -> TrainingWorkflow.run(...)
```

### End-to-end training flow

1. `TrainingWorkflow.run(...)` receives the crop name, runtime dataset root, output directory, and optional overrides.
2. `src/workflows/training_support.py` prepares the run:
   - config selection
   - loader creation
   - class detection
   - loader-size and split-count summaries
   - supported-class reference-count resolution from `split_manifest.json` or the runtime `continual` split
   - early rejection when a supported class resolves below `100` images unless explicit few-shot research mode is enabled
   - runtime-only class-balance metadata injection before adapter initialization
3. `src/data/loaders.py` creates the training loaders from the runtime dataset. The shipped `training.continual.data.sampler` default is `"auto"`, which keeps shuffle for roughly balanced continual splits and promotes the train loader to weighted sampling when the largest class is at least 1.5x the smallest non-zero class.
4. `src/training/services/class_balance.py` computes the training-side class-support policy and effective-number class-balanced weights. When all supported classes resolve to at least `100` images and at least one class is in the `100-200` band, the trainer applies Cui et al.-style effective-number weighting to the training classifier loss across all supported classes while leaving validation/test loss unweighted for artifact comparability. If the train loader already resolved to weighted sampling, effective-number loss weighting is disabled by default unless `training.continual.class_balance.allow_sampler_and_loss_rebalance=true`. Few-shot research mode can lower the hard floor for ablation runs while recording the production guardrail bypass in artifacts.
5. `src/adapter/independent_crop_adapter.py` exposes the public adapter lifecycle.
5. `src/training/continual_sd_lora.py` owns the actual training engine:
   - backbone loading
   - LoRA wrapping
   - optimization
   - prediction
   - OOD calibration
6. `src/training/session.py` runs epochs and emits observer events for progress, telemetry, and checkpoints.
7. `src/training/services/reporting.py` writes plots, CSV files, JSON metrics, confusion matrices, and readiness artifacts.
8. `src/training/services/ood_benchmark.py` runs the held-out fallback benchmark when real OOD data is missing.
9. The workflow saves the adapter and returns a structured `TrainingWorkflowResult`.

### What the training stack is optimizing

The supported adapter path is continual SD-LoRA only:

- frozen pretrained DINOv3 backbone
- LoRA adapters on selected transformer linear layers
- multi-scale feature fusion
- classifier head over the fused representation
- OOD state saved with the adapter bundle

## Dataset Contracts

There are two dataset contracts and they serve different purposes.

### Contract 1: Notebook 0 prep input

Notebook 0 accepts a flat class-root layout:

```text
<root>/<class>/<images>
```

This is easier for a beginner to prepare by hand.

Notebook 0 uses this contract for duplicate-aware audit before training and defaults to the repo-local staging root `data/class_root_dataset/`. Notebook 0 can also pull a reusable repo-local OOD tree from `data/ood_dataset/<dataset_name>/` when it materializes the runtime dataset.

### Contract 2: Notebook 2 and workflow runtime training layout

Notebook 2, workflow training, and CLI training use a runtime split layout:

```text
<data_dir>/<crop>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
  ood/*
  oe/*
```

This layout is what the code actually trains from.

### The conversion step

Notebook 0 owns the conversion from flat class-root input into the runtime split layout.

It writes:

- `split_manifest.json`


The grouped Notebook 0 prep path uses a 60/20/20 family split target with small-class safeguards. Before materialization, it now also applies conservative metadata-free prep filters:

- source-style proxy grouping from observable cues such as path tokens, stock-site or screenshot hints, aspect/resolution buckets, border/layout traits, web-export naming, and existing source hints
- label-risk triage from strong cross-class conflicts, same-class review clusters, and the grouped representation-review surface
- canonical eval eligibility that requires the family canonical sample and excludes synthetic, eval-quality-risk, source-style-risked, and non-clear label-risk samples

These filters do not delete usable samples. Risky but non-blocking samples stay eligible for `continual` and are kept out of canonical `val`/`test`. Label triage is a heuristic audit and review aid, not a ground-truth relabeling system.

Notebook 0 also writes `human_review_packet.json` plus `label_review_summary.json` and, when interactive review is enabled, pauses only around high-impact audit outcomes. The safe defaults keep uncertain but usable samples train-only, block direct materialization when cross-class conflicts or split blockers remain, and let clean audits continue without image-by-image review. The packet exposes the fixed conservative pHash, DINOv3, and BioCLIP-2.5 thresholds used for evidence generation; it is a review gate, not automatic threshold calibration.

The older non-grouped runtime-layout helper remains effectively 80/10/10.

Important detail:

- the workflow uses the runtime folder name `continual`
- workflow loading maps the logical training split onto that folder
- real `ood/` pools stay in one input tree, but loader construction can write or reuse `ood/ood_split_manifest.json` and expose a manifest-only `ood_dev` assignment plus a held-out final OOD test assignment; `ood_dev` can select the primary OOD score and threshold, while the final assignment is loaded as the normal `ood` loader used by readiness
- optional `oe/` is a separate Outlier Exposure training pool and is not used as final readiness evidence

## Inference Architecture

The canonical inference entrypoint is:

```text
src/workflows/inference.py -> InferenceWorkflow.predict(...)
```

### End-to-end inference flow

1. `InferenceWorkflow` builds `src/pipeline/router_adapter_runtime.py`.
2. The runtime loads config and resolves the adapter root.
3. Unless `trust_crop_hint=True` is supplied with a crop hint, the runtime loads the canonical router surface `src/router/router_pipeline.py`.
4. The router analyzes the image and produces a typed router result with normalized detections.
5. A plain `crop_hint` is treated as untrusted; the router must identify the same crop before the adapter can load.
6. The router keeps one canonical primary detection using router quality ranking when available, otherwise preserving router order.
7. If `inference.input_guard.enabled=true`, the runtime scores the image with the BioCLIP plantness prompt groups in [plantness_input_guard_prompt_groups.md](plantness_input_guard_prompt_groups.md). A non-plant decision returns `non_plant_rejected` and skips adapter inference.
8. Before resolving any adapter, the runtime applies a conservative uncertainty gate on that primary detection using the configured router confidence and router margin thresholds.
9. If the primary crop signal is too weak, ambiguous, or inconsistent with an untrusted crop hint, the payload status is `router_uncertain` and no adapter runs.
10. Otherwise the runtime resolves the crop adapter directory from that primary detection or from the explicitly trusted crop hint.
11. The runtime loads that adapter through `IndependentCropAdapter`.
12. The image is preprocessed to the configured target size.
13. The adapter predicts disease plus calibrated adapter OOD information when an adapter actually runs.
14. `src/pipeline/inference_payloads.py` converts the raw output into the public payload and adds a structured `router` summary block.

If the router runs but cannot identify a supported crop, the payload status is `unknown_crop`.
If the router backend fails to become usable or errors during routing, the payload status is `router_unavailable`.
If router initialization fails, the runtime discards that router instance and retries a clean load on the next request.
`non_plant_rejected`, `unknown_crop`, `router_unavailable`, `router_uncertain`, and `adapter_unavailable` omit adapter-side `ood_analysis` because no calibrated adapter OOD verdict exists on those paths.

### Default adapter resolution

The default deployment path is:

```text
models/adapters/<crop>/<part>/continual_sd_lora_adapter/
```

If no part-specific bundle exists yet, the runtime still falls back to the legacy crop-only layout for older exports.

If `crop_hint` is provided without `trust_crop_hint=True`, the router still runs and must agree before the adapter loads.
If `trust_crop_hint=True` is provided with a crop hint, the router step is skipped. In that case the payload still includes a `router` summary block with status `trusted_hint_skipped` so callers can migrate to the structured router view without losing mirrored crop fields.

`part_hint` is metadata only. It is not a separate adapter selector.

### Router-driven inference vs direct adapter smoke testing

These are intentionally separate surfaces:

- router-driven inference uses the router first, then the crop adapter
- Notebook 1 and `scripts/colab_router_adapter_inference.py` are router-only diagnostic surfaces that stop after crop/part identification and may return `part=unknown` when organ evidence is ambiguous
- Notebook 5 is the router calibration notebook surface over `scripts/evaluate_router_surface.py` and `scripts/calibrate_router_surface.py`
- direct adapter smoke testing loads one adapter directly and bypasses the router

Notebook 3, Notebook 4, and `scripts/colab_adapter_smoke_test.py` are for the second case. Notebook 4 is a smaller widget wrapper over the same direct-adapter helper path.

### Router evidence policy

The maintained router now follows two evidence rules:

- whole-image crop evidence is fused with SAM3 ROI evidence before choosing the crop
- the final exposed `part` label is chosen only from the configured crop-compatible part surface, and ambiguous organ evidence is rejected to `unknown` instead of forcing a confident label
- leaf-like visual cues can only rewrite a broad part label when the classifier also gives explicit leaf support with a positive score margin

This is intentional and crop-agnostic. It follows plant-vision literature that combines local symptom evidence with global plant context instead of trusting one patch alone, for example [Local and Global Feature-Aware Dual-Branch Networks for Plant Disease Recognition (LGNet)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11315374/) and [Deep Learning-Based Phenotyping System With Glocal Description of Plant Anomalies and Symptoms](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2019.01321/full).

The conservative part-resolution rule is also aligned with calibration and selective-prediction literature: ambiguous evidence should not be turned into a confident organ label by heuristic force alone. See [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) and [SelectiveNet: A Deep Neural Network with an Integrated Reject Option](https://proceedings.mlr.press/v97/geifman19a.html).

For offline part-threshold calibration, the maintained eval-only dataset contract is:

```text
data/router_part_eval/<crop>/<part>/*
```

The repo evaluates that surface with `scripts/evaluate_router_part_surface.py`.

For full crop/part handoff calibration, the maintained eval-only dataset contract is:

```text
data/router_eval/
  id/<crop>/<part>/*
  negatives/off_crop/<label>/*
  negatives/non_plant/<label>/*
  ambiguous/<label>/*
  wrong_part/<crop>/<unsupported_part>/*
```

The repo evaluates that surface with `scripts/evaluate_router_surface.py`, which reports crop accuracy, false accepts on negatives, abstention, part precision/recall, unsupported-part emissions, latency, risk-coverage, and threshold-sweep recommendations.

## OOD And Readiness Architecture

OOD behavior is part of the adapter runtime. It is not a separate model family.

The supported path is:

1. train on known classes
2. calibrate OOD after training
3. save the OOD state with the adapter
4. evaluate OOD evidence
5. write a deployment verdict

### Main OOD pieces

- detector logic: `src/ood/continual_ood.py`
- radial normalization: `src/ood/radial_normalization.py`
- SURE+/DS-F1-inspired double scoring: `src/ood/sure_scoring.py`
- conformal prediction: `src/ood/conformal_prediction.py`
- BER loss wrapper: `src/training/ber_loss.py`
- OOD calibration orchestration: `src/training/services/ood_calibration.py`
- readiness metrics and gates: `src/training/services/metrics.py`
- held-out fallback benchmark: `src/training/services/ood_benchmark.py`
- class-balance policy and effective-number weighting: `src/training/services/class_balance.py`

Important current detail:

- the "SURE+" label in this repo is shorthand for a two-threshold semantic-plus-confidence diagnostic path used for DS-F1-style evaluation; the runtime `is_ood` verdict still follows the calibrated primary score method rather than letting the confidence-reject flag override it directly
- conformal mode can now be threshold conformalization on OOD residuals, or standard APS/RAPS set-valued classification
- the energy detector can optionally calibrate a temperature on the calibration split before thresholds are written
- kNN distance scoring supports `cdist`, chunked distance search, and optional FAISS when available

### Why there are multiple JSON artifacts

The workflow writes more than one decision-looking file, but they are not equal.

- `validation/metric_gate.json`: split-local diagnostic gate
- `test/metric_gate.json`: split-local diagnostic gate
- `adapter_behavioral_acceptance.json`: default adapter training/recovery/promotion verdict
- `production_readiness.json`: compatibility and broader deployment diagnostic

Adapter acceptance comes from held-out per-sample decisions in `adapter_behavioral_acceptance.json`: supported ID inputs must classify correctly and same-crop, same-part unsupported diseases must be rejected without a forced supported label. Router-scope far OOD is not an adapter acceptance gate. `production_readiness.json` remains available for compatibility and broader diagnostics but does not override this adapter decision.

## Artifact Flow

### Workflow and CLI training output

Training writes:

```text
<output_dir>/
  <crop>/
    <part>/
      continual_sd_lora_adapter/
      training_metrics/
```

### Run lineage and optimizer-facing records

Each training run now writes two canonical traceability artifacts under:

```text
<output_dir>/<crop>/<part>/training_metrics/training/
  experiment_manifest.json
  optimization_record.json
```

These files do not replace `training/summary.json` or `training/run_context.json`. Instead, they normalize those existing workflow artifacts into a stable experiment-record surface for Pareto analysis.

Current contract:

- `experiment_manifest.json` is the immutable run-lineage record
- `optimization_record.json` is the optimizer-facing parameter and objective record
- runs are only comparable when dataset lineage, crop, part, engine, and backbone match

The dataset-lineage key is:

```text
<dataset_key>::<split_manifest_sha256>
```

This prevents optimizer code from mixing runs that came from different prepared runtime datasets even when the crop name is the same.

### Repo-local run registry

Historical and new training runs can be indexed locally with:

```powershell
.\scripts\python.cmd scripts/index_training_runs.py --runs-root runs
```

That script writes local-generated registry outputs under:

```text
runs/_index/
  trials.jsonl
  latest_registry.json
  pareto_inputs.json
  pareto_frontiers.json
  automatic_wins.md
```

The registry prefers canonical traceability files when they exist and falls back to best-effort reconstruction from older `summary`, `run_context`, readiness, and guided artifacts.

Current automatic behavior:

- when a training run writes canonical traceability artifacts successfully, the repo-local `runs/_index/` registry is refreshed best-effort
- that same refresh now also rebuilds cohort-safe Pareto frontier summaries from the indexed runs
- that same refresh also writes `automatic_wins.md`, a human-readable Markdown summary of Pareto-frontier winners by comparable cohort
- Notebook 2 enables Bayesian proposal refresh by default; standalone registry rebuilds write `bayesian_recommendations.json` only when explicitly requested
- the standalone script remains available when you want to rebuild the registry on demand across existing runs

Phase 2 optimization entrypoint:

```powershell
.\scripts\python.cmd scripts/optimize_training_runs.py --dataset-lineage-key <dataset_key>::<split_manifest_sha256> --crop-name <crop> --part-name <part>
```

That surface reads the canonical run registry, stays inside one comparable cohort, and reports:

- analyze the current Pareto frontier
- Bayesian optimization disabled status by default, or explicit feature-flagged proposal candidates when `--enable-bayesian-optimization` is passed

`training_metrics/` currently contains:

- `training/`
- `validation/`
- `test/`
- `ood_benchmark/`
- `production_readiness.json`

### Notebook 2 output

Notebook 2 writes to three places:

- local outputs under `outputs/colab_notebook_training/`
- repo mirrors under `runs/<crop>/<part>/<RUN_ID>/`
- repo-local telemetry runtime under `outputs/colab_notebook_training/telemetry_runtime/telemetry/<RUN_ID>/`

The artifact roots now also include a `guided/` layer with a human-readable start file plus machine-readable overview/catalog files. This layer organizes the raw artifacts without replacing them.

The repo mirror keeps notebook outputs, telemetry copies, and checkpoint manifests plus only the best checkpoint; rolling checkpoint history remains under the repo-local telemetry runtime root.

Current adapter export detail:

- local notebook export: `outputs/colab_notebook_training/<crop>/<part>/continual_sd_lora_adapter/`
- workflow export: `<output_dir>/<crop>/<part>/continual_sd_lora_adapter/`
- telemetry runtime export: `outputs/colab_notebook_training/telemetry_runtime/telemetry/<RUN_ID>/artifacts/adapter_export/<crop>/<part>/continual_sd_lora_adapter/`

Current notebook naming/detail:

- Notebook 2 exposes `CROP_NAME` and `PART_NAME` at the notebook surface
- Notebook 2 run ids are human-readable and include crop/part/date-time information
- maintained notebooks perform an early update/access check before the main long-running cells
- Notebook 2 no longer requests Google Drive mount access by default
- Notebook 2 sets `ENABLE_BAYESIAN_OPTIMIZATION=True` by default, applies the current adapter cohort's next proposal when one exists, and refreshes `runs/_index/bayesian_recommendations.json` after the run writes traceability.


## Colab Support Layer

The notebook support layer is intentionally thin. The notebooks should wrap the maintained workflows, not replace them.

Maintained Colab helper scripts:

- `scripts/colab_repo_bootstrap.py`
- `scripts/colab_live_telemetry.py`
- `scripts/colab_checkpointing.py`
- `scripts/colab_notebook_helpers.py`
- `scripts/colab_dataset_layout.py`
- `scripts/colab_router_adapter_inference.py`
- `scripts/colab_adapter_smoke_test.py`

These helpers handle:

- repo discovery or auto-clone
- dependency installation
- Drive mount coordination
- repo update and access preflight reporting
- Hugging Face token checks
- dataset materialization
- guided artifact indexing and overview generation
- local output mirroring
- telemetry sync
- checkpoint management
- direct adapter smoke testing

## File-to-Responsibility Map

Use this as the quick code map.

- `src/core/config_manager.py`: config loading, merge, normalization, and alias sync
- `src/workflows/training.py`: canonical training workflow
- `src/workflows/training_support.py`: run preparation and artifact payload shaping
- `src/workflows/inference.py`: canonical inference workflow
- `src/adapter/independent_crop_adapter.py`: public adapter lifecycle
- `src/training/continual_sd_lora.py`: actual adapter training engine
- `src/training/services/metrics.py`: metric thresholds and readiness logic
- `src/training/services/reporting.py`: artifact persistence
- `src/training/services/ood_benchmark.py`: held-out fallback OOD benchmark
- `src/pipeline/router_adapter_runtime.py`: router-plus-adapter inference runtime
- `src/router/router_pipeline.py`: canonical router implementation surface

- `src/shared/contracts.py`: shared result and metadata contracts

## Validation Surfaces

The repo validates the maintained surface through:

- `scripts/validate_notebook_imports.py`
- `scripts/validate_config_schema.py`
- `scripts/validate_router_open_world_preflight.py`
- `scripts/validate_router_open_world_result.py`
- `scripts/evaluate_dataset_layout.py`
- `scripts/evaluate_router_part_surface.py`
- `scripts/benchmark_surfaces.py`
- `scripts/benchmark_runtime_real.py` (opt-in real-model benchmark when assets are available)
- `tests/unit/`
- `tests/integration/`
- `tests/colab/test_smoke_training.py`

## Related Docs

- [../../README.md](../../README.md)
- [../user_guide/colab_training_manual.md](../user_guide/colab_training_manual.md)
- [../user_guide/ood_readiness_guide.md](../user_guide/ood_readiness_guide.md)
- [router_performance_literature_review.md](router_performance_literature_review.md)
- [ood_recommendation.md](ood_recommendation.md)
- [unknown_disease_rejection.md](unknown_disease_rejection.md)
- [../archive/experimental_leave_one_class_out_ood.md](../archive/experimental_leave_one_class_out_ood.md)
