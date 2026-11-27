# Methodology and results

Why AADS is built the way it is, which ideas came from the literature, and what the published evidence does and does
not show. The project is frozen: the results below are final for the `aads-public-demo-v1.1.2` adapters, and the
limitations are properties of the final system, not scheduled work.

## Research question

Plant-disease classifiers are usually evaluated as closed-set systems: every test image belongs to a trained class.
A real input may show the wrong crop, the wrong plant part, an unsupported disease on a supported crop, a non-plant
scene, or a poor image, and a closed-set classifier still has to return one of its labels. AADS asks:

> Can the system return a supported diagnosis when the evidence is strong, and send the image to review when the
> crop, part or disease is outside its supported surface?

That combines open-set recognition, out-of-distribution (OOD) detection and selective prediction.

## System design

```text
image -> crop/part router -> confidence and compatibility gates -> crop__part adapter
      -> supported-class prediction + OOD scores -> answer or review
```

Routing failure, missing adapter, corrupt input, calibration mismatch and OOD rejection stay distinct typed outcomes
instead of being collapsed into a disease label.

## Backbone and adapters

The backbone is `facebook/dinov3-vitl16-pretrain-lvd1689m`, kept frozen. Features from transformer layers 2, 5, 8 and
11 are projected to 768 dimensions and combined with a learned softmax gate. Self-supervised pretraining matters here
because each crop/part dataset is small compared with the backbone's pretraining corpus.

That layer spacing suits a 12-block ViT-B, but ViT-L/16 has 24 blocks (`depth=24` in `vit_large`,
[`dinov3/models/vision_transformer.py`](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/models/vision_transformer.py)).
Every fused feature therefore comes from the first 11 blocks. Blocks 12–24 still run in the forward pass, but their
output never reaches the classifier, and the LoRA weights on them receive no gradient. All eight released adapters
were trained this way, and the choice was not ablated against later layers.

Each crop/part target gets a LoRA adapter: the frozen weight \(W\) is updated as \(W' = W + \frac{\alpha}{r}BA\) with
trainable low-rank \(A, B\). The checked-in configuration ([`config/base.json`](../config/base.json)) applies LoRA to
all transformer linear modules with \(r=24\), \(\alpha=24\) and dropout `0.1`; the released adapters set their
own rank ([released adapter settings](#released-adapter-settings)). An adapter bundle holds the LoRA
weights, classifier, fusion state and OOD calibration. LoRA shrinks what is trained and distributed per target; the
backbone is still needed at inference time.

### Terminology note

Module, class, notebook and artifact names use `continual_sd_lora` (for example `ContinualSDLoRATrainer` and the
`continual_sd_lora_adapter/` bundle directory) because the released bundles and their loaders depend on them. The
names overstate the method:

- the adapter is a standard PEFT `LoraConfig`; it does not implement SD-LoRA's magnitude/direction decomposition;
- "continual" is the training split name and the trainer's ability to grow the classifier head; there is no replay,
  forgetting regularization or forgetting evaluation;
- each crop/part adapter is trained independently on its own data.

The accurate description is **independent per-target LoRA adapters on a frozen DINOv3 backbone**.

## Crop and plant-part routing

1. SAM3 proposes plant and plant-part regions from text prompts.
2. BioCLIP-2.5 scores crop/part prompt ensembles on the regions and on the whole image.
3. A taxonomy rules out impossible crop/part combinations.
4. An optional prototype reconciler compares image embeddings with reviewed target/class prototypes.
5. Confidence, margin and compatibility gates decide whether an adapter handoff is safe.

The public configuration uses a `0.65` router confidence floor, a `0.10` margin floor and weight `0.65` for whole-image
crop context. These are calibrated engineering thresholds, not probabilities.

Symptoms are often local while crop identity is global, which is why region and whole-image evidence are fused (LGNet,
glocal plant-anomaly work). Promptable localization (SAM3; GLIP and Grounding DINO as context) avoids depending on a
fixed detector label set. Following selective-prediction reasoning, a router that forces weak handoffs to raise
apparent accuracy is worse than a conservative one with lower coverage.

## Training and data roles

Base-configuration defaults in `config/base.json`: 16 epochs, batch size 8 with gradient accumulation 4, learning
rate `1.5e-4`, weight decay `0.01`, cosine schedule with 10% warmup, seed 42, LogitNorm loss, RandAugment (2 operations, magnitude 7) and
early stopping on validation loss. [`config/colab.json`](../config/colab.json) overrides these to 12 epochs, batch size
96 without accumulation, learning rate `2e-4`, bf16 and non-deterministic kernels. The released adapters were
trained with per-target recipes that override both; see [released adapter settings](#released-adapter-settings).

LogitNorm was chosen because overconfident logits hurt a system that needs a review path; it was not ablated against
cross-entropy.

Each runtime dataset separates roles: `continual/` fits the adapter, `val/` drives ordinary training decisions,
`test/` measures held-out supported classes, OOD dev evidence may select score methods and thresholds, locked OOD test
evidence is used once for the frozen candidate, and `oe/` is auxiliary training data that must stay disjoint from final
OOD evaluation. Exact hashes, source families and near-duplicate review reduce leakage between them.

## OOD and unknown-disease rejection

The hardest case is a plausible disease of the correct crop and part that the adapter was never trained on. Three
score families are maintained:

- **Energy**: \(E(x) = -T \log \sum_k \exp(z_k(x)/T)\), using the full logit vector; the temperature is calibrated on
  allowed development evidence.
- **Mahalanobis-style distance**: the nearest standardized distance to a class mean with diagonal variance,
  \(d_M(x) = \min_k \sqrt{\sum_j (f_j(x)-\mu_{k,j})^2 / (\sigma_{k,j}^2+\epsilon)}\).
- **Deep kNN**: distance to stored in-distribution feature banks, without a Gaussian assumption.

`primary_score_method: "auto"` compares candidates on development evidence only, never on the locked OOD test slice.
The OOD development target false-positive rate is `0.05`.

In `config/base.json`, Outlier Exposure has weight `0.5` and a uniform target, and BER regularization, ReAct
clipping, the classifier rebalance stage and the plantness input guard are disabled. The released adapters differ:
ReAct is on for 7 of 8, the classifier rebalance stage for 6 of 8, and the Outlier Exposure weight is 0.12–0.3
([released adapter settings](#released-adapter-settings)). None of these was ablated, and nothing here claims they
helped. RAPS conformal prediction is enabled with \(\alpha=0.05\), \(\lambda=0.2\) and `k_reg=1`; it reports
set-valued uncertainty and coverage, but does not replace testing on unknown diseases.

### Released adapter settings

Read from the `adapter_meta.json` of each bundle in release
[`aads-public-demo-v1.1.2`](https://github.com/EfeErim/aads-open-world-plant-disease/releases/tag/aads-public-demo-v1.1.2)
(`gh release download aads-public-demo-v1.1.2 -p '*adapter_meta.json'`). Epochs are the configured maximum; early
stopping on validation loss was on for every target.

| Target | LoRA r / alpha | Epochs | LR | Batch | Label smoothing | OE weight | ReAct | Classifier rebalance | OOD threshold factor | Primary OOD score | Fusion layers |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---|---|
| apricot/fruit | 20/20 | 36 | 0.0001 | 40 | 0.12 | 0.3 | on (0.985) | on (3 ep) | 4.5 | ensemble | 2/5/8/11 |
| apricot/leaf | 20/24 | 24 | 0.00012 | 52 | 0.1 | 0.3 | on (0.985) | on (3 ep) | 4.0 | ensemble | 2/5/8/11 |
| grape/fruit | 24/24 | 32 | 0.0001 | 48 | 0.1 | 0.24 | on (0.985) | on (3 ep) | 3.0 | ensemble | 2/5/8/11 |
| grape/leaf | 24/24 | 30 | 0.0001 | 56 | 0.1 | 0.24 | on (0.985) | on (3 ep) | 3.5 | ensemble | 2/5/8/11 |
| strawberry/fruit | 20/26 | 38 | 5e-05 | 40 | 0.05 | 0.12 | off | off | 2.0 | ensemble | 2/5/8/11 |
| strawberry/leaf | 24/24 | 24 | 0.00012 | 88 | 0.1 | 0.3 | on (0.99) | off | 4.0 | ensemble | 2/5/8/11 |
| tomato/fruit | 24/24 | 30 | 6e-05 | 64 | 0.1 | 0.25 | on (0.985) | on (3 ep) | 3.0 | ensemble | 2/5/8/11 |
| tomato/leaf | 32/24 | 26 | 0.00011 | 88 | 0.1 | 0.25 | on (0.985) | on (3 ep) | 3.5 | ensemble | 2/5/8/11 |

The pipeline is an engineering synthesis of the cited methods, not a paper-faithful reproduction of any one of them.
Selective classification is implemented as post-hoc gates, not a trained SelectiveNet head.

## Results

### 1. Fixed regression replay

Accepted run `20260706T153334Z` on a checksum-pinned 48-row manifest, built by
[`scripts/build_controlled_demo_manifest.py`](../scripts/build_controlled_demo_manifest.py) from an earlier internal
stress run (`20260706T140135Z`). Only rows that passed that run were eligible, and two known regressions (`demo_236`,
`demo_432`) were excluded. The stress run's images and results are private.

| Metric | Result |
|---|---:|
| total / passed / failed | 48 / 48 / 0 |
| answered disease rows | 36 |
| reviewed or abstained rows | 12 |
| negative false accepts | 0 |
| wrong-part disease labels | 0 |

- The 48/48 pass rate follows from the selection rule; this is a regression check, not an accuracy estimate.
- The 36 supported rows cover 12 disease classes; some targets contribute a single class.
- All 12 non-answer rows are router-level rejections (`router_uncertain`), so adapter-level unknown-disease rejection
  is not exercised here.
- In Notebook 8's default path the prototype-reconciler calibration manifest is the graded manifest
  (`scripts/notebook_cells/nb8_cell06_m2_full_demo_run.py`). The public record does not show which calibration
  manifest the frozen run used, so calibration on the graded rows cannot be ruled out.

Source: [`evidence/controlled_demo_summary.json`](../evidence/controlled_demo_summary.json),
[`evidence/controlled_demo_rows.json`](../evidence/controlled_demo_rows.json).

### 2. Per-adapter behavioral acceptance

Each adapter must reach accuracy ≥ 0.93, balanced accuracy ≥ 0.90, macro-F1 ≥ 0.90, at least 30 ID test samples, ID
false-rejection rate ≤ 0.05, at least 30 same-crop unsupported-disease test samples, zero forced supported answers on
that slice, and a same-crop OOD rejection rate of 1.00. Latest tracked artifact per target at source commit
`539397bb72bde59e4b092ac1286b5415fe78dbac`, which is not in this repository's current history; the released bundles
are pinned by the release tag `aads-public-demo-v1.1.2`:

| Target | Accuracy | Balanced acc. | Macro-F1 | ID FRR | Same-crop OOD n | OOD rejection | Forced answers | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `apricot__fruit` | 0.914 | 0.843 | 0.815 | 0.068 | 0 | n/a | 0 | no |
| `apricot__leaf` | 0.940 | 0.939 | 0.940 | 0.082 | 59 | 0.203 | 47 | no |
| `grape__fruit` | 0.920 | 0.883 | 0.873 | 0.063 | 41 | 0.293 | 29 | no |
| `grape__leaf` | 0.910 | 0.900 | 0.899 | 0.034 | 16 | 0.375 | 10 | no |
| `strawberry__fruit` | 0.916 | 0.836 | 0.810 | 0.144 | 0 | n/a | 0 | no |
| `strawberry__leaf` | 0.994 | 0.994 | 0.993 | 0.055 | 0 | n/a | 0 | no |
| `tomato__fruit` | 0.899 | 0.932 | 0.878 | 0.092 | 64 | 0.563 | 28 | no |
| `tomato__leaf` | 0.900 | 0.913 | 0.902 | 0.050 | 18 | 0.611 | 7 | no |

`apricot__leaf` passes the three classification metrics but forces a supported answer on 47 of 59 same-crop
unsupported-disease images; `strawberry__leaf` classifies well but has no eligible same-crop OOD slice and narrowly
misses the false-rejection limit. The result is **0/8 passed**.

The summary [`evidence/latest_behavioral_acceptance_summary.json`](../evidence/latest_behavioral_acceptance_summary.json)
is rebuilt from the eight records in [`evidence/behavioral_acceptance/targets/`](../evidence/behavioral_acceptance/targets/)
by [`scripts/build_behavioral_acceptance_summary.py`](../scripts/build_behavioral_acceptance_summary.py) (`--check`),
which also rejects a record whose internal crop/part does not match its filename. A unit test runs the same check.

### 3. Pooled OOD metrics (three re-evaluated targets)

These pools mix every OOD type, so they are easier than the same-crop slice above. FPR@95 is the fraction of ID test
images scored as OOD when 95% of OOD images are caught.

| Target | OOD n | ID n | AUROC | FPR@95 | Conformal coverage (target 0.95) | Avg. set size |
|---|---:|---:|---:|---:|---:|---:|
| `grape__leaf` | 274 | 324 | 0.855 | 0.543 | 0.978 | 2.01 |
| `strawberry__leaf` | 81 | 491 | 0.986 | 0.057 | 0.994 | 1.00 |
| `tomato__leaf` | 103 | 1204 | 0.920 | 0.264 | 0.934 | 1.17 |

The readiness targets are AUROC ≥ 0.92 and FPR@95 ≤ 0.05. Only `strawberry__leaf` comes close, and its pool contains
no same-crop unsupported-disease images. The other five targets have no comparable public record. Source:
[`evidence/release_inputs/corrected_readiness/`](../evidence/release_inputs/corrected_readiness/).

### 4. Public adapter release

The [release](https://github.com/EfeErim/aads-open-world-plant-disease/releases/tag/aads-public-demo-v1.1.2) holds 64
checksum-pinned files for eight adapter bundles, about 518 MB in total, each with the DINOv3 license notice. Its
manifest records `production_ready: false`. Source:
[`evidence/public_asset_manifest.json`](../evidence/public_asset_manifest.json),
[`docs/evidence/current/demo_release/release_manifest.json`](evidence/current/demo_release/release_manifest.json).

## What the evidence supports

| Claim | Status |
|---|---|
| the frozen pipeline reproduces the recorded replay decisions | supported for that manifest |
| the pipeline is accurate on the replay rows | not established; follows from selection |
| the system generalizes to arbitrary field images | not established; no external field trial |
| adapters are production-ready | false under the maintained gate (0/8) |
| suitable for autonomous diagnosis | not supported |

## Known limitations

- **Near-OOD rejection is weak.** Same-crop unsupported-disease rejection ranges from 0.20 to 0.61 where measured.
- **Under-sampled OOD evidence.** Three targets have no same-crop OOD test rows; `grape__leaf` (16) and `tomato__leaf`
  (18) are below the gate's minimum of 30.
- **Thresholds do not transfer exactly.** OOD thresholds target 5% false rejection on validation data; test ID false
  rejection ranges from 3.4% to 14.4%.
- **Validation data has several roles.** The `val` split drives early stopping, OOD statistics, threshold selection
  and conformal calibration, and the code falls back to training data if `val` is empty.
- **OOD dev/test split is not family-grouped.** It is per-image random within each slice (`src/data/ood_splits.py`), so
  near-duplicates can sit on both sides.
- **Single seed, no ablations.** Seed 42, no confidence intervals; LogitNorm, Outlier Exposure, multi-scale fusion and
  LoRA rank were not ablated. A linear-probe baseline script exists (Notebook 9,
  `scripts/posthoc_linear_probe.py`), but no result from it is recorded here.
- **Not reproducible from public data.** Training images are private and DINOv3 is gated; the evidence can be
  inspected and cross-checked but not regenerated from this repository alone.

## References

- Siméoni et al. (2025), [DINOv3](https://arxiv.org/abs/2508.10104).
- Hu et al. (2021), [LoRA](https://arxiv.org/abs/2106.09685).
- Stevens et al. (2024), [BioCLIP](https://arxiv.org/abs/2311.18803); Gu et al. (2025), [BioCLIP 2](https://arxiv.org/abs/2505.23883).
- Radford et al. (2021), [CLIP](https://arxiv.org/abs/2103.00020).
- Carion et al. (2025), [SAM 3](https://arxiv.org/abs/2511.16719).
- Li et al. (2022), [GLIP](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Grounded_Language-Image_Pre-Training_CVPR_2022_paper.html); Liu et al. (2023), [Grounding DINO](https://arxiv.org/abs/2303.05499).
- Lin et al. (2024), [LGNet: Local and Global Feature-Aware Dual-Branch Networks for Plant Disease Recognition](https://pubmed.ncbi.nlm.nih.gov/39130161/).
- Fuentes et al. (2019), [Glocal Description of Plant Anomalies and Symptoms](https://www.frontiersin.org/articles/10.3389/fpls.2019.01321/full).
- Hendrycks and Gimpel (2017), [A Baseline for Detecting Misclassified and OOD Examples](https://openreview.net/forum?id=Hkg4TI9xl).
- Lee et al. (2018), [Mahalanobis OOD detection](https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html).
- Liu et al. (2020), [Energy-based OOD Detection](https://arxiv.org/abs/2010.03759).
- Sun et al. (2022), [OOD Detection with Deep Nearest Neighbors](https://proceedings.mlr.press/v162/sun22d.html).
- Hendrycks, Mazeika and Dietterich (2019), [Outlier Exposure](https://openreview.net/forum?id=HyxCxhRcY7).
- Vaze et al. (2022), [Open-Set Recognition: a Good Closed-Set Classifier is All You Need?](https://openreview.net/forum?id=5hLP5JY9S2d).
- Bitterwolf et al. (2023), [In or Out? Fixing ImageNet OOD Detection Evaluation](https://proceedings.mlr.press/v202/bitterwolf23a.html).
- Wei et al. (2022), [LogitNorm](https://proceedings.mlr.press/v162/wei22d.html).
- Dong et al. (2024), [Fine-tuning paradigms and unknown plant disease recognition](https://www.nature.com/articles/s41598-024-66958-2).
- Geifman and El-Yaniv (2019), [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html).
- Guo et al. (2017), [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).
- Angelopoulos et al. (2021), [Uncertainty Sets for Image Classifiers using Conformal Prediction](https://openreview.net/forum?id=eNdiU_DbM9).
