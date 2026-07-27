#!/usr/bin/env python3
"""Generate artifact manifest for adapter exports, tracking version and metadata."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]


def get_git_commit_hash(repo_root: Path) -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_git_branch(repo_root: Path) -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def build_artifact_manifest(
    adapter_bundle_root: Path,
    crop: str,
    part: str,
    training_start_time: Optional[datetime] = None,
    training_duration_seconds: Optional[float] = None,
    dataset_lineage_key: Optional[str] = None,
    ood_evidence_available: bool = False,
    production_readiness_verdict: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build manifest for exported adapter.

    Args:
        adapter_bundle_root: Root directory containing adapter files
        crop: Crop name (e.g., 'tomato')
        part: Plant part (e.g., 'leaf')
        training_start_time: Training start time
        training_duration_seconds: Training duration
        dataset_lineage_key: Dataset identifier/SHA
        ood_evidence_available: Whether OOD calibration was performed
        production_readiness_verdict: Readiness status (ready/provisional/not-ready)

    Returns:
        Manifest dictionary
    """
    repo_root = ROOT
    now = datetime.now(timezone.utc)
    training_start = training_start_time or now
    duration = training_duration_seconds or 0.0

    # Check for key files in adapter bundle
    has_weights = (adapter_bundle_root / "adapter_model.bin").exists()
    has_classifier = (adapter_bundle_root / "classifier_state.pt").exists()
    has_ood_state = (adapter_bundle_root / "ood_state.json").exists()
    has_config = (adapter_bundle_root / "adapter_config.json").exists()

    manifest = {
        "manifest_version": "1.0",
        "generated_at": now.isoformat(),
        "repo": {
            "commit_hash": get_git_commit_hash(repo_root),
            "branch": get_git_branch(repo_root),
            "root": str(repo_root),
        },
        "adapter": {
            "crop": crop,
            "part": part,
            "bundle_root": str(adapter_bundle_root),
            "files": {
                "adapter_weights": has_weights,
                "classifier_state": has_classifier,
                "ood_state": has_ood_state,
                "config": has_config,
            },
        },
        "training": {
            "start_time": training_start.isoformat(),
            "duration_seconds": duration,
            "dataset_lineage_key": dataset_lineage_key or "unknown",
        },
        "evaluation": {
            "ood_evidence_available": ood_evidence_available,
            "production_readiness_verdict": production_readiness_verdict or "unknown",
        },
    }

    return manifest


def persist_manifest(manifest: Dict[str, Any], output_path: Path) -> None:
    """Write manifest to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Generate artifact manifest")
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--crop", type=str, required=True)
    parser.add_argument("--part", type=str, required=True)
    parser.add_argument("--dataset-key", type=str, default=None)
    parser.add_argument("--ood-available", action="store_true")
    parser.add_argument("--readiness", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_artifact_manifest(
        adapter_bundle_root=args.adapter_root,
        crop=args.crop,
        part=args.part,
        dataset_lineage_key=args.dataset_key,
        ood_evidence_available=args.ood_available,
        production_readiness_verdict=args.readiness,
    )

    persist_manifest(manifest, args.output)
    print(f"✅ Manifest saved to {args.output}")
