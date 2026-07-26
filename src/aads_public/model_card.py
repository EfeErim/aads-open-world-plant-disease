"""Generate concise, evidence-bound model cards for public adapter assets."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

_TARGET_PATTERN = re.compile(r"^[a-z]+__(fruit|leaf)$")


def _read_object(path: Path, label: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _metric(metrics: dict, name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"missing or invalid metric: {name}")
    return float(value)


def build_target_model_card(
    readiness_path: Path,
    adapter_config_path: Path,
    adapter_meta_path: Path,
    *,
    target: str,
) -> str:
    """Build a card without upgrading failed evidence into a readiness claim."""

    if not _TARGET_PATTERN.fullmatch(target):
        raise ValueError("target must use the canonical crop__part form")
    crop, part = target.split("__", maxsplit=1)
    readiness = _read_object(readiness_path, "readiness")
    config = _read_object(adapter_config_path, "adapter config")
    meta = _read_object(adapter_meta_path, "adapter metadata")
    if (meta.get("crop_name"), meta.get("part_name")) != (crop, part):
        raise ValueError("adapter metadata target does not match requested target")
    if readiness.get("passed") is not False or readiness.get("deployable") is not False:
        raise ValueError("public card generator expects an explicitly failed, non-deployable readiness artifact")
    metrics = readiness.get("classification_evidence", {}).get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("readiness classification metrics are missing")
    accuracy = _metric(metrics, "accuracy")
    macro_f1 = _metric(metrics, "macro_f1")
    ood_auroc = _metric(metrics, "ood_auroc")
    ood_fpr = _metric(metrics, "ood_false_positive_rate")
    classification_samples = int(_metric(metrics, "classification_samples"))
    ood_samples = int(_metric(metrics, "ood_samples"))
    base_model = config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model:
        raise ValueError("adapter config has no base model")
    missing = readiness.get("missing_requirements")
    if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
        raise ValueError("readiness missing_requirements must be a list of strings")
    missing_text = ", ".join(f"`{item}`" for item in missing) or "none recorded"
    title = target.replace("__", " / ").title()
    return f"""---
base_model: {base_model}
library_name: peft
license: other
tags:
  - computer-vision
  - dinov3
  - lora
  - controlled-demo
---

# AADS adapter — {title}

SD-LoRA adapter and auxiliary heads for the `{target}` disease-classification target in the AADS v6
graduation project. Developed by Efe Erim. The artifact is published for reproducibility and portfolio
review, not as a production-ready plant-health system.

## Evidence snapshot

| Check | Recorded value |
|---|---:|
| Test accuracy | {accuracy:.3f} |
| Test macro F1 | {macro_f1:.3f} |
| OOD AUROC | {ood_auroc:.3f} |
| OOD false-positive rate | {ood_fpr:.3f} |
| Classification samples | {classification_samples} |
| OOD samples | {ood_samples} |
| Production readiness | **FAILED** |
| Deployable | **No** |

Failed requirements: {missing_text}.

These are frozen evaluation values from the accompanying `production_readiness.json`; they are not a
fresh inference result and do not establish generalization beyond the evaluated dataset.

## Artifact contract

- Base encoder: `{base_model}`
- Adapter format: PEFT LoRA (`adapter_model.safetensors` + `adapter_config.json`)
- Auxiliary heads: `classifier.pth` and `fusion.pth`
- Provenance/runtime metadata: `adapter_meta.json`
- Exact gate details: `production_readiness.json`
- Base-model license copy: `DINOV3_LICENSE.md`

The auxiliary `.pth` files are Python/PyTorch serialization surfaces. Load only the checksummed release
assets and use `torch.load(..., weights_only=True)` with a compatible PyTorch version.

## Intended and out-of-scope use

Intended for code review, controlled demonstrations, and research reproduction inside the documented
AADS pipeline. It must not be used for autonomous diagnosis, treatment decisions, safety-critical
automation, or claims of production readiness. Inputs outside the supported crop/part distribution can
be falsely accepted; the recorded OOD gate failed.

Repository and aggregate limitations: https://github.com/EfeErim/bitirmeprojesi
"""
