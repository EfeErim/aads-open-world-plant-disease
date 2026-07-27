#!/usr/bin/env python3
"""Capture deterministic orchestration benchmarks for supported surfaces."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _IdentityModule(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class _FakeRouter:
    def load_models(self):
        return None

    def analyze_image(self, _image):
        return {"detections": [{"crop": "tomato", "part": "leaf", "crop_confidence": 0.95}]}

    def analyze_image_result(self, image):
        return self.analyze_image(image)

    def is_ready(self):
        return True


class _FakeAdapter:
    def __init__(self, crop_name, device="cpu"):
        self.crop_name = crop_name
        self.device = device

    def load_adapter(self, _adapter_dir):
        return None

    def predict_with_ood(self, _image):
        return {
            "status": "success",
            "disease": {"class_index": 0, "name": "healthy", "confidence": 0.9},
            "ood_analysis": {
                "score_method": "ensemble",
                "primary_score": 0.1,
                "decision_threshold": 0.8,
                "candidate_scores": {"ensemble": 0.1},
                "candidate_thresholds": {"ensemble": 0.8},
                "is_ood": False,
                "calibration_version": 1,
            },
        }


class _FakeTokenizer:
    def __call__(self, prompts):
        return torch.arange(len(prompts), dtype=torch.float32).unsqueeze(1)


class _FakeOpenClipModel:
    def __init__(self):
        self.logit_scale = torch.tensor(0.0)

    def encode_image(self, image_tensor):
        batch = int(image_tensor.shape[0])
        return torch.ones(batch, 2, dtype=torch.float32, device=image_tensor.device)

    def encode_text(self, text_tokens):
        batch = int(text_tokens.shape[0])
        embeds = torch.zeros(batch, 2, dtype=torch.float32, device=text_tokens.device)
        if batch > 0:
            embeds[:, 0] = 1.0
        return embeds


def _time_call(fn):
    started = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, result


def _build_trainer(config_cls, trainer_cls):
    cfg = config_cls(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
    )
    trainer = trainer_cls(cfg)
    trainer.class_to_idx = {"healthy": 0}
    trainer.adapter_model = _IdentityModule()
    trainer.classifier = nn.Linear(4, 1)
    trainer.fusion = _IdentityModule()
    trainable = nn.Parameter(torch.tensor([1.0], requires_grad=True))
    trainer.optimizer = torch.optim.SGD([trainable], lr=0.1)
    trainer.training_step = lambda _batch: (trainable ** 2).sum()  # type: ignore[assignment]
    trainer.forward_logits = lambda images: torch.zeros(images.shape[0], 1)  # type: ignore[assignment]
    trainer._is_initialized = True
    return trainer


def collect_benchmarks() -> dict[str, float]:
    from src.pipeline.router_adapter_runtime import RouterAdapterRuntime
    from src.router.router_pipeline import RouterPipeline
    from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer

    def trainer_factory():
        return _build_trainer(ContinualSDLoRAConfig, ContinualSDLoRATrainer)

    trainer_init_ms, trainer = _time_call(trainer_factory)
    train_batch_ms, _ = _time_call(
        lambda: trainer.train_batch({"images": torch.zeros(1, 3, 8, 8), "labels": torch.zeros(1, dtype=torch.long)})
    )
    snapshot = trainer.snapshot_training_state()
    checkpoint_roundtrip_ms, _ = _time_call(lambda: trainer_factory().restore_training_state(snapshot))
    trainer.class_to_idx = {"healthy": 0, "disease_a": 1}
    trainer.classifier = nn.Linear(4, 2)
    trainer.encode = lambda images: torch.zeros(images.shape[0], 4)  # type: ignore[assignment]
    calibration_loader = [
        {"images": torch.zeros(4, 3, 8, 8), "labels": torch.tensor([0, 1, 0, 1], dtype=torch.long)},
        {"images": torch.zeros(4, 3, 8, 8), "labels": torch.tensor([1, 0, 1, 0], dtype=torch.long)},
    ]
    ood_calibration_stream_ms, _ = _time_call(lambda: trainer.calibrate_ood(calibration_loader))

    runtime_root = Path(".runtime_tmp") / "benchmark_models" / "tomato" / "continual_sd_lora_adapter"
    runtime_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "adapter_meta.json").write_text("{}", encoding="utf-8")

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(runtime_root.parent.parent), "target_size": 224},
        },
        device="cpu",
        adapter_root=runtime_root.parent.parent,
    )
    runtime._build_router = lambda: _FakeRouter()  # type: ignore[assignment]
    runtime._build_adapter = lambda crop_name: cast(Any, _FakeAdapter(crop_name, device="cpu"))  # type: ignore[assignment]

    router_cold_load_ms, _ = _time_call(runtime.load_router)
    warm_inference_ms, _ = _time_call(lambda: runtime.predict(Image.new("RGB", (16, 16), color="green")))

    pipeline = RouterPipeline(
        config={
            "vlm_enabled": True,
            "router": {
                "crop_mapping": {"tomato": {"parts": ["leaf"]}},
                "vlm": {
                    "enabled": True,
                    "crop_labels": ["tomato", "potato"],
                    "part_labels": ["leaf"],
                    "batch_chunk_size": 8,
                    "roi_score_batch_size": 32,
                    "prompt_templates": {
                        "crop": ["{term}", "close {term}"],
                        "part": ["{term}"],
                    },
                },
            },
        },
        device="cpu",
    )
    pipeline.models_loaded = True
    pipeline.actual_pipeline = "sam3"
    pipeline.bioclip_backend = "open_clip"
    pipeline.bioclip = _FakeOpenClipModel()
    pipeline.bioclip_processor = {
        "preprocess": lambda _image: torch.ones(3, 4, 4, dtype=torch.float32),
        "tokenizer": _FakeTokenizer(),
    }
    pipeline._run_sam3_batch = lambda images, prompt, threshold=0.7: [  # type: ignore[assignment]
        {
            "masks": [torch.ones(1, 1, dtype=torch.float32)],
            "boxes": [torch.tensor([0.0, 0.0, 8.0, 8.0])],
            "scores": [torch.tensor(0.95)],
        }
        for _ in images
    ]
    route_batch_8_ms, _ = _time_call(lambda: pipeline.route_batch(torch.zeros(8, 3, 16, 16)))

    return {
        "trainer_init_ms": round(trainer_init_ms, 4),
        "train_batch_ms": round(train_batch_ms, 4),
        "checkpoint_roundtrip_ms": round(checkpoint_roundtrip_ms, 4),
        "ood_calibration_stream_ms": round(ood_calibration_stream_ms, 4),
        "router_cold_load_ms": round(router_cold_load_ms, 4),
        "warm_inference_ms": round(warm_inference_ms, 4),
        "route_batch_8_ms": round(route_batch_8_ms, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture slim-surface orchestration benchmarks.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    benchmarks = collect_benchmarks()
    body = json.dumps(benchmarks, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
