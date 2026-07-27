from __future__ import annotations

import importlib.metadata
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from src.shared.hash_utils import sha256_file
from src.shared.json_utils import read_json

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]


def _git_output(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        logger.debug("Git command failed for args=%s: %s", args, exc)
        return ""
    if completed.returncode != 0:
        logger.debug(
            "Git command returned non-zero exit status for args=%s stderr=%s",
            args,
            str(completed.stderr).strip(),
        )
        return ""
    return str(completed.stdout).strip()


def collect_git_context(repo_root: Path) -> Dict[str, Any]:
    return {
        "head": _git_output(repo_root, "rev-parse", "HEAD"),
        "head_short": _git_output(repo_root, "rev-parse", "--short", "HEAD"),
        "branch": _git_output(repo_root, "branch", "--show-current"),
        "is_dirty": bool(_git_output(repo_root, "status", "--porcelain")),
    }


def collect_package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package_name in (
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "peft",
        "accelerate",
        "huggingface-hub",
        "numpy",
        "scikit-learn",
        "opencv-python",
        "Pillow",
    ):
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def collect_dataset_manifest_context(crop_root: Path) -> JsonDict:
    filename = "split_manifest.json"
    manifest_path = crop_root / filename
    payload: JsonDict = {
        "path": str(manifest_path),
        "exists": manifest_path.exists(),
    }
    if manifest_path.exists():
        try:
            manifest_json = read_json(manifest_path, default={}, expect_type=dict)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to read dataset manifest %s: %s", manifest_path, exc)
            manifest_json = {}
        payload.update(
            {
                "sha256": sha256_file(manifest_path),
                "schema_version": manifest_json.get("schema_version"),
                "source_root": manifest_json.get("source_root"),
                "crop_name": manifest_json.get("crop_name"),
                "part_name": manifest_json.get("part_name"),
                "dataset_key": manifest_json.get("dataset_key"),
                "split_policy": manifest_json.get("split_policy"),
                "ood": manifest_json.get("ood"),
            }
        )
    return {filename: payload}


def read_dataset_manifest_payload(crop_root: Path) -> JsonDict:
    manifest_path = crop_root / "split_manifest.json"
    if not manifest_path.exists():
        return {}
    return read_json(manifest_path, default={}, expect_type=dict)


def resolve_part_name(*, runtime_dataset_key: str, manifest_payload: Dict[str, Any]) -> str:
    manifest_part_name = str(manifest_payload.get("part_name", "") or "").strip().lower()
    if manifest_part_name:
        return manifest_part_name
    dataset_key = str(runtime_dataset_key or "").strip().lower()
    if "__" in dataset_key:
        _crop_name, part_name = dataset_key.split("__", 1)
        return str(part_name or "unspecified")
    return "unspecified"


def normalize_part_name(part_name: Optional[str]) -> str:
    normalized = str(part_name or "").strip().lower()
    return normalized or "unspecified"
