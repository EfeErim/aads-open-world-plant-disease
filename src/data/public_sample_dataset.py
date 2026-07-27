"""Deterministic synthetic runtime dataset for public smoke testing.

The generated images are deliberately synthetic and must never be used as model-quality
evidence. They exist only to make the public training path executable without access to
the private research dataset.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

PUBLIC_SAMPLE_SCHEMA = "aads.public_sample_dataset.v1"
PUBLIC_SAMPLE_CLASSES = ("healthy", "synthetic_spot")
PUBLIC_SAMPLE_SPLIT_COUNTS = {"continual": 12, "val": 4, "test": 4}
PUBLIC_SAMPLE_NEGATIVE_COUNTS = {"ood": 8, "oe": 8}


def _target_parts(target: str) -> tuple[str, str]:
    normalized = str(target or "").strip().lower()
    parts = normalized.split("__")
    if len(parts) != 2 or not all(part.replace("_", "").isalnum() for part in parts):
        raise ValueError("Public sample target must use the <crop>__<part> form.")
    return parts[0], parts[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_image(path: Path, *, seed: int, variant: str) -> None:
    rng = random.Random(seed)
    base = (52, 126, 62) if variant == "healthy" else (88, 112, 48)
    image = Image.new("RGB", (96, 96), color=base)
    draw = ImageDraw.Draw(image)
    for _ in range(12):
        x = rng.randint(4, 84)
        y = rng.randint(4, 84)
        radius = rng.randint(2, 8)
        if variant == "healthy":
            color = (rng.randint(65, 110), rng.randint(145, 210), rng.randint(55, 95))
        elif variant == "synthetic_spot":
            color = (rng.randint(115, 175), rng.randint(55, 95), rng.randint(25, 55))
        else:
            color = (rng.randint(35, 220), rng.randint(35, 220), rng.randint(35, 220))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    draw.line((8, 88, 88, 8), fill=(220, 235, 210), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)


def _existing_manifest(target_root: Path, target: str) -> dict[str, Any] | None:
    manifest_path = target_root / "split_manifest.json"
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Public sample manifest must be a JSON object: {manifest_path}")
    if payload.get("schema_version") != PUBLIC_SAMPLE_SCHEMA or payload.get("target") != target:
        raise RuntimeError(f"Existing public sample directory has incompatible metadata: {target_root}")
    for row in payload.get("rows", []):
        relative_path = str(row.get("relative_path") or "")
        sample_path = target_root / relative_path
        if not sample_path.is_file() or _sha256(sample_path) != row.get("sha256"):
            raise RuntimeError(f"Existing public sample dataset failed integrity validation: {relative_path}")
    return payload


def materialize_public_sample_dataset(root: str | Path, *, target: str = "tomato__leaf") -> dict[str, Any]:
    """Create or verify a small rights-safe synthetic runtime dataset."""

    crop, part = _target_parts(target)
    normalized_target = f"{crop}__{part}"
    target_root = Path(root).expanduser().resolve() / normalized_target
    existing = _existing_manifest(target_root, normalized_target)
    if existing is not None:
        return {
            "target": normalized_target,
            "target_root": str(target_root),
            "manifest_path": str(target_root / "split_manifest.json"),
            "created": False,
            "sample_count": len(existing.get("rows", [])),
            "production_eligible": False,
        }
    if target_root.exists() and any(target_root.iterdir()):
        raise RuntimeError(f"Refusing to overwrite a non-sample dataset directory: {target_root}")

    rows: list[dict[str, Any]] = []
    seed = 1000
    for split, count in PUBLIC_SAMPLE_SPLIT_COUNTS.items():
        for class_name in PUBLIC_SAMPLE_CLASSES:
            for index in range(count):
                relative_path = f"{split}/{class_name}/{class_name}_{index:03d}.png"
                sample_path = target_root / relative_path
                _render_image(sample_path, seed=seed, variant=class_name)
                rows.append(
                    {
                        "relative_path": relative_path,
                        "split": split,
                        "class_name": class_name,
                        "sha256": _sha256(sample_path),
                        "source": "deterministic_synthetic_generator",
                    }
                )
                seed += 1
    for split, count in PUBLIC_SAMPLE_NEGATIVE_COUNTS.items():
        for index in range(count):
            relative_path = f"{split}/synthetic_unknown/unknown_{index:03d}.png"
            sample_path = target_root / relative_path
            _render_image(sample_path, seed=seed, variant="unknown")
            rows.append(
                {
                    "relative_path": relative_path,
                    "split": split,
                    "class_name": "synthetic_unknown",
                    "sha256": _sha256(sample_path),
                    "source": "deterministic_synthetic_generator",
                }
            )
            seed += 1

    manifest = {
        "schema_version": PUBLIC_SAMPLE_SCHEMA,
        "target": normalized_target,
        "synthetic": True,
        "production_eligible": False,
        "purpose": "public_training_pipeline_smoke_test_only",
        "classes": list(PUBLIC_SAMPLE_CLASSES),
        "rows": rows,
    }
    manifest_path = target_root / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "target": normalized_target,
        "target_root": str(target_root),
        "manifest_path": str(manifest_path),
        "created": True,
        "sample_count": len(rows),
        "production_eligible": False,
    }
