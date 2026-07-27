# Methodology, literature and results

This document explains why AADS is built the way it is, which ideas came from the literature, what is implemented in
the repository, and what the current evidence does and does not prove.

The short version is:

- the project is an open-world decision pipeline, not a single image classifier;
- a router first decides whether a supported crop/part adapter should receive the image;
- a frozen DINOv3 backbone is adapted with LoRA for each crop/part target;
- the adapter can classify a supported disease or reject the input through calibrated OOD logic;
- the strongest current evidence is a fully passing fixed 48-row controlled demo;
- the stricter per-adapter behavioral gate is still 0/8, so this is not presented as production-ready diagnosis.

## Research question

Most plant-disease classifiers are evaluated as closed-set systems: every test image is assumed to belong to one of
the disease classes seen during training. That assumption is fragile outside a curated benchmark. A real input may
show:

- the wrong crop;
- the right crop but the wrong plant part;
- a supported crop/part with an unsupported disease;
- a non-plant scene;
- an ambiguous or low-quality image.

A closed-set classifier still has to return one of its known labels. AADS instead asks a harder question:

> Can the system return a supported diagnosis when the evidence is strong, and send the image to review when the
> crop, part or disease is outside its supported surface?

That combines three related research areas:

1. **open-set recognition**, where unseen semantic classes appear at inference time;
2. **out-of-distribution detection**, where a score separates supported and unsupported inputs;
3. **selective prediction**, where the system may abstain instead of forcing a label.

## System design

```text
image
  -> crop/part router
  -> confidence and compatibility gates
  -> matching crop__part adapter
  -> supported-class prediction + OOD scores
  -> answer or review/abstain
```

The canonical code path is:

```text
src/workflows/inference.py
  -> src/pipeline/router_adapter_runtime.py
  -> src/router/router_pipeline.py
  -> src/adapter/continual_adapter.py
  -> src/ood/continual_ood.py
```

The pipeline is deliberately modular. Routing failure, adapter absence, corrupt input, calibration mismatch and OOD
rejection remain distinct typed outcomes instead of being collapsed into a disease label.

## Representation and parameter-efficient adaptation

### DINOv3 backbone

The configured visual backbone is `facebook/dinov3-vitl16-pretrain-lvd1689m`. DINOv3 is used as a strong pretrained
feature extractor; the project does not train a vision transformer from scratch.

The motivation is transfer learning. Large self-supervised vision models learn reusable visual structure before the
plant-disease task is introduced. This matters when the downstream dataset is much smaller than the backbone's
pretraining corpus.

The maintained configuration keeps the backbone frozen and extracts features from transformer layers 2, 5, 8 and 11.
Those features are projected to 768 dimensions and combined with a learned softmax gate.

Implementation:

- [`src/training/continual_sd_lora.py`](../src/training/continual_sd_lora.py)
- [`src/adapter/multi_scale_fusion.py`](../src/adapter/multi_scale_fusion.py)
- [`config/base.json`](../config/base.json)

### LoRA adapters

Full fine-tuning would create a separate large backbone for every crop/part target. AADS instead uses Low-Rank
Adaptation (LoRA): the pretrained weights stay frozen while small trainable low-rank matrices are inserted into
selected transformer linear layers.

If a frozen weight matrix is \(W\), LoRA learns a low-rank update:

\[
W' = W + \frac{\alpha}{r}BA
\]

where \(A\) and \(B\) are trainable and \(r\) is much smaller than the original matrix dimensions.

The checked-in configuration uses:

- all transformer linear modules selected by the maintained resolver;
- rank \(r=24\);
- scaling parameter \(\alpha=24\);
- dropout `0.1`.

The result is one compact adapter per crop/part surface, plus its classifier, fusion state and OOD calibration. LoRA
reduces trainable and distributable task-specific state; it does not make the frozen DINOv3 backbone unnecessary at
inference time.

## Crop and plant-part routing

The router is object-centric rather than a single whole-image crop classifier.

1. SAM3 proposes plant and plant-part regions from text prompts.
2. BioCLIP-2.5 scores crop/part prompt ensembles on regions and whole-image context.
3. A taxonomy restricts impossible crop/part combinations.
4. A prototype reconciler can compare image embeddings with reviewed target/class prototypes.
5. Confidence, margin and compatibility gates decide whether an adapter handoff is safe.

The current public configuration uses a `0.65` router confidence floor and a `0.10` margin floor. Whole-image crop
context is combined with region evidence at weight `0.65`. These are calibrated engineering thresholds, not claims
that model scores are probabilities.

Why combine local and global evidence? Disease symptoms are often local, while crop identity and plant-part context
can be global. Plant-vision studies such as LGNet and glocal plant-anomaly work support treating those signals as
complementary. Open-vocabulary localization work such as GLIP and Grounding DINO supports using language-conditioned
regions rather than depending only on a fixed detector label set.

The project also follows selective-prediction reasoning: a router that increases apparent accuracy by forcing weak
handoffs may be worse than a more conservative router with lower coverage.

Implementation:

- [`src/router/router_pipeline.py`](../src/router/router_pipeline.py)
- [`src/router/sam3_runtime.py`](../src/router/sam3_runtime.py)
- [`src/router/roi_pipeline.py`](../src/router/roi_pipeline.py)
- [`src/router/prototype_reconciler.py`](../src/router/prototype_reconciler.py)
- [`src/router/surface_calibration.py`](../src/router/surface_calibration.py)

## Training objective and data policy

The maintained training defaults are:

- 16 epochs;
- batch size 8 with gradient accumulation 4;
- learning rate `1.5e-4`;
- Adam-compatible weight decay `0.01`;
- cosine schedule with 10% warmup;
- deterministic seed 42;
- LogitNorm loss;
- RandAugment with two operations at magnitude 7;
- early stopping on validation loss.

LogitNorm was chosen because overconfident logits are harmful when the product needs an abstain/review path. It is not
treated as universally superior: cross-entropy and other controlled ablations remain possible.

Runtime data is separated by role:

```text
data/prepared_runtime_datasets/<crop>__<part>/
  continual/<supported class>/*
  val/<supported class>/*
  test/<supported class>/*
  ood/<unsupported evidence>/*
  oe/<auxiliary outliers>/*
```

The roles are intentionally different:

- `continual/` fits the adapter;
- `val/` selects ordinary training behavior;
- `test/` measures held-out supported-class behavior;
- OOD dev evidence may select score/threshold candidates;
- locked OOD test evidence is used once for the frozen candidate;
- `oe/` is auxiliary training evidence and must remain disjoint from final OOD evaluation.

This separation follows a basic model-selection rule: final evidence cannot also be free threshold-tuning data.
Exact hashes, source families and near-duplicate review are used to reduce train/dev/test/OOD leakage.

## OOD and unknown-disease rejection

The hardest rejection case is not a random non-plant image. It is a visually plausible disease of the correct crop
and plant part that the adapter was never trained to diagnose.

AADS maintains three primary score families.

### Energy score

For class logits \(z_k(x)\) and temperature \(T\):

\[
E(x) = -T \log \sum_k \exp(z_k(x)/T)
\]

Energy uses the full logit vector rather than only the largest softmax probability. The implementation calibrates
temperature candidates on allowed development evidence.

### Mahalanobis-style feature distance

For feature vector \(f(x)\), class mean \(\mu_k\) and diagonal class variance \(\sigma_k^2\), the detector measures the
nearest standardized class distance:

\[
d_M(x) = \min_k \sqrt{\sum_j \frac{(f_j(x)-\mu_{k,j})^2}{\sigma_{k,j}^2+\epsilon}}
\]

This asks whether the feature lies near any supported class cluster.

### Deep k-nearest-neighbor distance

The kNN score measures distance from the input embedding to stored in-distribution feature banks. It is
non-parametric and avoids assuming that every class embedding is Gaussian.

### Ensemble and thresholding

The maintained ensemble combines normalized feature-distance and energy evidence. `primary_score_method: "auto"`
means candidate methods can be compared on permitted development evidence. It does not authorize selecting the method
on the same locked OOD test slice later used for the final claim.

The configured OOD development target false-positive rate is `0.05`.

## Outlier Exposure, conformal prediction and optional methods

Outlier Exposure (OE) trains with auxiliary examples that should not map to supported classes. The checked-in
configuration enables an OE objective with weight `0.5` and a uniform target. OE data is not counted as final OOD
evidence.

RAPS conformal prediction is enabled with \(\alpha=0.05\), regularization \(\lambda=0.2\) and `k_reg=1`. Conformal sets
are useful for uncertainty presentation and coverage diagnostics, but they do not remove the need to test semantic
unknown diseases.

Several research mechanisms exist in code or configuration without being active defaults:

- BER regularization: disabled;
- ReAct feature clipping: disabled;
- classifier rebalance stage: disabled;
- plantness input guard: disabled.

Their presence should not be confused with evidence that they improved the published run.

## How the literature influenced the implementation

| Literature signal | Project decision | Status in this repository |
|---|---|---|
| DINOv3: strong self-supervised dense visual features | use a pretrained DINOv3 ViT as the shared backbone | implemented |
| LoRA: low-rank updates on frozen transformer weights | maintain one compact adapter per crop/part target | implemented |
| BioCLIP: biology-aware image/text embeddings | score crop/part prompts and build optional prototypes | implemented |
| SAM3, GLIP and Grounding DINO: promptable/open-vocabulary localization | keep routing object-centric and text-conditioned | implemented with SAM3; GLIP/Grounding DINO are literature context and alternative surfaces |
| local/global plant-vision models | fuse region evidence with whole-image crop context | implemented |
| energy-based OOD detection | keep energy as a first-class score candidate | implemented |
| Mahalanobis feature scoring | model distance from supported class feature statistics | implemented |
| deep kNN OOD detection | keep a non-parametric distance candidate | implemented |
| Outlier Exposure | train against a disjoint auxiliary unknown pool | implemented and configurable |
| LogitNorm | reduce classifier overconfidence | active training default |
| SelectiveNet/selective classification | prefer explicit review over weak forced answers | implemented as post-hoc gates, not a trained SelectiveNet head |
| conformal prediction/RAPS | expose set-valued uncertainty and coverage diagnostics | implemented |
| OOD benchmark criticism | separate realistic near-OOD slices and lock final evidence | implemented in data and acceptance contracts |

The pipeline is an engineering synthesis. It is not claimed as a paper-faithful reproduction of any one cited method.

## Current results

### 1. Fixed controlled demo

The accepted controlled run is `20260706T153334Z` on a checksum-pinned 48-row manifest.

| Metric | Result |
|---|---:|
| total rows | 48 |
| passed | 48 |
| failed | 0 |
| answered disease rows | 36 |
| reviewed or abstained rows | 12 |
| negative false accepts | 0 |
| wrong-part disease labels | 0 |
| runtime | CUDA |
| recorded elapsed time | 7m 1s |

All eight supported crop/part adapter surfaces appear in the demo. The twelve non-answer rows cover unknown crop,
unknown part and non-plant/review behavior.

Source:

- [`evidence/controlled_demo_summary.json`](../evidence/controlled_demo_summary.json)
- [`evidence/controlled_demo_rows.json`](../evidence/controlled_demo_rows.json)

This is strong evidence that the frozen pipeline reproduced the intended decisions on that one manifest. It is not an
estimate of field accuracy.

### 2. Latest tracked per-adapter behavioral acceptance

The production-oriented adapter gate is deliberately stricter than the controlled demo. Each adapter must satisfy all
of the following:

| Gate | Required value |
|---|---:|
| accuracy | at least 0.93 |
| balanced accuracy | at least 0.90 |
| macro-F1 | at least 0.90 |
| ID test samples | at least 30 |
| ID false-rejection rate | at most 0.05 |
| same-crop unsupported-disease test samples | at least 30 |
| forced supported answers on that OOD slice | exactly 0 |
| same-crop OOD rejection rate | 1.00 |

The latest tracked acceptance artifact per target at source commit
`539397bb72bde59e4b092ac1286b5415fe78dbac` gives:

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

Machine-readable values, source paths, source Git blobs and failure reasons are recorded in
[`evidence/latest_behavioral_acceptance_summary.json`](../evidence/latest_behavioral_acceptance_summary.json).
The eight source records are also checked in under
[`evidence/behavioral_acceptance/targets/`](../evidence/behavioral_acceptance/targets/). Rebuild or verify the summary
without access to the private experiment tree:

```bash
python scripts/build_behavioral_acceptance_summary.py
python scripts/build_behavioral_acceptance_summary.py --check
```

The builder verifies that each record's internal crop/part identity matches its filename before aggregation. This
prevents a checksum-valid but misattributed readiness record from being accepted as evidence for another target.

The table explains why a single “accuracy” number is not enough:

- `strawberry__leaf` has excellent closed-set classification but lacks an eligible same-crop OOD test slice in the
  selected artifact and narrowly misses the ID false-rejection gate;
- `apricot__leaf` passes the three classification metrics but forces a supported answer on 47 of 59 same-crop
  unsupported-disease cases;
- `tomato__leaf` is close on ID metrics but rejects only 11 of 18 same-crop unsupported-disease cases;
- three targets have no eligible same-crop OOD rows in their latest tracked artifact, so no rejection rate is reported.

The aggregate production-oriented result is therefore **0/8 passed**, not “nearly production-ready.”

### 3. Public adapter release

The public Release contains 64 checksum-pinned files for eight adapter bundles and is about 518 MB in total. Each
bundle includes LoRA weights, classifier/fusion state, configuration, metadata, a readiness record and the required
DINOv3 license notice.

The downloader:

```bash
python scripts/fetch_public_adapters.py
```

checks SHA-256 before materializing the runtime directory layout.

Source:

- [`evidence/public_asset_manifest.json`](../evidence/public_asset_manifest.json)
- [`docs/evidence/current/demo_release/release_manifest.json`](evidence/current/demo_release/release_manifest.json)
- [public Release](https://github.com/EfeErim/bitirmeprojesi/releases/tag/aads-public-demo-v1.1.2)

The release is for code review and controlled experiments. Its manifest explicitly records
`production_ready: false`.

## Evidence strength

| Claim | Evidence strength | Why |
|---|---|---|
| the repository contains the real implementation | strong | complete `src/`, notebook, script and test surfaces are present and exercised by CI |
| the fixed 48-row demo reproduces the recorded decisions | strong for that manifest | row-level decisions, manifest hash and summary are checked in |
| the system generalizes to arbitrary field images | not established | no representative external field trial or confidence interval is published |
| adapters are production-ready | false under the maintained gate | 0/8 latest tracked acceptance artifacts pass |
| the system is suitable for autonomous diagnosis | not supported | same-crop unknown-disease rejection remains inadequate or under-sampled |

The main achievement is the system design and evidence discipline: typed failure states, reproducible adapter
packages, explicit unknown handling, disjoint evidence roles and a testable end-to-end pipeline. The main research gap
is robust semantic near-OOD rejection under representative field conditions.

## References used in the project

### Representation and adaptation

- Siméoni et al. (2025), [DINOv3](https://arxiv.org/abs/2508.10104).
- Hu et al. (2021), [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- Stevens et al. (2024), [BioCLIP](https://arxiv.org/abs/2311.18803).
- Gu et al. (2025), [BioCLIP 2](https://arxiv.org/abs/2505.23883).
- Radford et al. (2021), [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).

### Routing and plant vision

- Carion et al. (2025), [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719).
- Li et al. (2022), [Grounded Language-Image Pre-Training](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Grounded_Language-Image_Pre-Training_CVPR_2022_paper.html).
- Liu et al. (2023), [Grounding DINO](https://arxiv.org/abs/2303.05499).
- Lin et al. (2024), [Local and Global Feature-Aware Dual-Branch Networks for Plant Disease Recognition](https://pubmed.ncbi.nlm.nih.gov/39130161/).
- Fuentes et al. (2019), [Glocal Description of Plant Anomalies and Symptoms](https://www.frontiersin.org/articles/10.3389/fpls.2019.01321/full).

### OOD, open set and uncertainty

- Hendrycks and Gimpel (2017), [A Baseline for Detecting Misclassified and OOD Examples](https://openreview.net/forum?id=Hkg4TI9xl).
- Lee et al. (2018), [A Simple Unified Framework for OOD Samples and Adversarial Attacks](https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html).
- Liu et al. (2020), [Energy-based Out-of-distribution Detection](https://arxiv.org/abs/2010.03759).
- Sun et al. (2022), [Out-of-Distribution Detection with Deep Nearest Neighbors](https://proceedings.mlr.press/v162/sun22d.html).
- Hendrycks, Mazeika and Dietterich (2019), [Deep Anomaly Detection with Outlier Exposure](https://openreview.net/forum?id=HyxCxhRcY7).
- Vaze et al. (2022), [Open-Set Recognition: a Good Closed-Set Classifier is All You Need?](https://openreview.net/forum?id=5hLP5JY9S2d).
- Bitterwolf et al. (2023), [In or Out? Fixing ImageNet OOD Detection Evaluation](https://proceedings.mlr.press/v202/bitterwolf23a.html).
- Wei et al. (2022), [Mitigating Neural Network Overconfidence with Logit Normalization](https://proceedings.mlr.press/v162/wei22d.html).
- Dong et al. (2024), [The impact of fine-tuning paradigms on unknown plant diseases recognition](https://www.nature.com/articles/s41598-024-66958-2).

### Selective and conformal prediction

- Geifman and El-Yaniv (2019), [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html).
- Guo et al. (2017), [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).
- Angelopoulos and Bates (2021), [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511).
- Angelopoulos et al. (2021), [Uncertainty Sets for Image Classifiers using Conformal Prediction](https://openreview.net/forum?id=eNdiU_DbM9).
- Bates et al. (2021), [Distribution-Free, Risk-Controlling Prediction Sets](https://arxiv.org/abs/2101.02703).
