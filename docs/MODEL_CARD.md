# Model card — AADS controlled-demo adapters v1.1.1

## Summary

AADS is a research system for selective plant-disease recognition. It routes an input to one of eight crop/part
specialists and may return `review` rather than force a disease label. The public artifacts are derived from DINOv3
and PEFT/LoRA training, but they are **not approved for autonomous deployment**.

## Intended use

- graduation-thesis and portfolio review;
- inspection of adapter packaging, evidence, and fail-closed decision contracts;
- controlled demonstrations where a human interprets every output.

## Out of scope

- agricultural treatment or pesticide decisions;
- autonomous production deployment;
- unsupported crops, parts, diseases, or acquisition conditions;
- presenting the 48-row controlled result as field accuracy;
- loading `.pth` files from untrusted or modified sources.

## Evidence

The frozen CUDA run `20260706T153334Z` recorded 48 accepted decisions:

- 36 correct disease answers;
- 12 expected review/abstain decisions;
- 0 negative false accepts;
- 0 wrong-part disease labels.

The public validator recomputes those totals from sanitized row decisions and checks the archived run, manifest, and
row-file identities. It does not rerun the model.

### Independent readiness results

| Target | ID accuracy | Macro-F1 | OOD AUROC | OOD FPR | Deployable |
|---|---:|---:|---:|---:|---|
| `apricot__fruit` | 0.950 | 0.896 | 0.918 | 0.336 | No |
| `apricot__leaf` | 0.813 | 0.812 | 0.635 | 0.873 | No |
| `grape__fruit` | 0.908 | 0.867 | 0.782 | 0.793 | No |
| `grape__leaf` | 0.908 | 0.867 | 0.782 | 0.793 | No |
| `strawberry__fruit` | 0.785 | 0.723 | 0.927 | 0.283 | No |
| `strawberry__leaf` | 0.908 | 0.867 | 0.782 | 0.793 | No |
| `tomato__fruit` | 0.899 | 0.875 | 0.821 | 0.420 | No |
| `tomato__leaf` | 0.899 | 0.875 | 0.821 | 0.420 | No |

All eight production-readiness records have `status=failed` and `deployable=false`. The primary blocker is held-out
OOD false acceptance, not artifact integrity.

## Data and evaluation boundaries

The original datasets are excluded from this public repository because their file-level redistribution licenses are
not sufficiently documented. Consequently, an anonymous reviewer can validate the sanitized decision snapshot and
artifact identities, but cannot independently reproduce training or field evaluation from this repository alone.

The controlled 48-row surface was selected for a customer-style demonstration. It is separate from broader
stress/readiness evidence and has no statistical confidence claim.

## Artifact contents and safe handling

Each target release group contains:

- `adapter_model.safetensors` — LoRA adapter weights;
- `adapter_config.json` — PEFT configuration;
- `classifier.pth` and `fusion.pth` — auxiliary state dictionaries;
- `adapter_meta.json` — class, calibration, and lineage metadata;
- `production_readiness.json` — the failed independent readiness record;
- `README.md` — the target-specific card;
- `DINOV3_LICENSE.md` — the governing DINOv3 terms.

Only download assets listed in the checksum-pinned public manifest. Treat `.pth` files as untrusted pickle containers
unless their digest matches and load them with a framework path equivalent to `torch.load(..., weights_only=True)`.
The compact public package downloads and verifies assets but does not expose an unsafe deserialization helper.

## Risks and limitations

- High OOD false-positive rates can force unknown diseases into supported labels.
- Performance can change with lighting, camera, cultivar, geography, season, and disease stage.
- Abstention reduces unsafe answers but also reduces coverage.
- Dataset provenance and redistribution gaps prevent fully public end-to-end reproduction.
- DINOv3 terms apply separately from the repository MIT license.

## Contact and provenance

Developed by Efe Erim as a graduation-thesis ML/CV engineering project. Source, immutable release identity, and
machine-readable evidence are linked from the repository README.
