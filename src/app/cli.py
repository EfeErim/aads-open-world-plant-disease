"""Small CLI facade over the canonical workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AADS v6 workflow CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inference = subparsers.add_parser("inference", help="Run router -> adapter inference for a single image")
    inference.add_argument("image", type=Path)
    inference.add_argument("--config-env", default="colab")
    inference.add_argument("--crop", dest="crop_hint")
    inference.add_argument("--part", dest="part_hint")
    inference.add_argument("--trust-crop-hint", action="store_true")
    inference.add_argument("--adapter-root", type=Path)
    inference.add_argument("--device", default="cuda")

    training = subparsers.add_parser("training", help="Run continual adapter training")
    training.add_argument("crop", help="Crop name")
    training.add_argument("--part", dest="part_name")
    training.add_argument("data_dir", type=Path, help="Runtime dataset root")
    training.add_argument("output_dir", type=Path, help="Output directory for adapter assets")
    training.add_argument("--config-env", default="colab")
    training.add_argument("--device", default="cuda")
    training.add_argument("--num-epochs", type=int)
    training.add_argument("--num-workers", type=int)
    training.add_argument("--validation-every-n-epochs", type=int)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    workflow: Any
    if args.command == "inference":
        from src.workflows.inference import InferenceWorkflow

        workflow = InferenceWorkflow(
            environment=args.config_env,
            device=args.device,
            adapter_root=args.adapter_root,
        )
        image = Image.open(args.image).convert("RGB")
        result = workflow.predict(
            image,
            crop_hint=args.crop_hint,
            part_hint=args.part_hint,
            return_ood=True,
            trust_crop_hint=bool(args.trust_crop_hint),
        )
        print(json.dumps(result, indent=2))
        return 0

    from src.workflows.training import TrainingWorkflow

    workflow = TrainingWorkflow(
        environment=args.config_env,
        device=args.device,
    )
    result = workflow.run(
        crop_name=args.crop,
        part_name=args.part_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        validation_every_n_epochs=args.validation_every_n_epochs,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
