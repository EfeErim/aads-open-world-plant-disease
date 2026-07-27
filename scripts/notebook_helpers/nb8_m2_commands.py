"""Notebook 8 M2 subprocess command builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_m2_demo_checklist_command(
    *,
    python_executable: str,
    repo_root: Path,
    manifest_path: Path,
    device: str,
    config_env: str,
    adapter_root_path: Path,
    output_path: Path,
    markdown_output_path: Path,
    analysis_output_path: Path,
    analysis_markdown_output_path: Path,
    batch_size: int,
    adapter_batch_size: int,
    handoff_cache_path: Path,
    demo_limit: Any = None,
    stop_on_dependency_blocker: bool = True,
    refresh_handoff_cache: bool = True,
    enable_prototype_reconciler: bool = False,
    prototype_bank: str = "",
    taxonomy_registry: str = "",
    prototype_min_similarity: Any = None,
    prototype_min_margin: Any = None,
    prototype_min_negative_gap: Any = None,
    prototype_calibration_output_path: Path | None = None,
) -> list[str]:
    command = [
        python_executable,
        str(repo_root / "scripts" / "run_demo_checklist.py"),
        "--no-checklist",
        "--extra-manifest",
        str(manifest_path),
        "--device",
        str(device),
        "--config-env",
        str(config_env),
        "--adapter-root",
        str(adapter_root_path),
        "--output",
        str(output_path),
        "--markdown-output",
        str(markdown_output_path),
        "--analysis-output",
        str(analysis_output_path),
        "--analysis-markdown-output",
        str(analysis_markdown_output_path),
        "--batch-size",
        str(max(1, int(batch_size))),
        "--adapter-batch-size",
        str(max(1, int(adapter_batch_size))),
        "--handoff-cache",
        str(handoff_cache_path),
    ]
    if demo_limit is not None:
        command.extend(["--limit", str(int(demo_limit))])
    if stop_on_dependency_blocker:
        command.append("--stop-on-dependency-blocker")
    if refresh_handoff_cache:
        command.append("--refresh-handoff-cache")
    if enable_prototype_reconciler:
        command.append("--enable-prototype-reconciler")
        if prototype_bank:
            command.extend(["--prototype-bank", str((repo_root / prototype_bank).resolve())])
        if taxonomy_registry:
            command.extend(["--taxonomy-registry", str((repo_root / taxonomy_registry).resolve())])
        if prototype_min_similarity is not None:
            command.extend(["--prototype-min-similarity", str(float(prototype_min_similarity))])
        if prototype_min_margin is not None:
            command.extend(["--prototype-min-margin", str(float(prototype_min_margin))])
        if prototype_min_negative_gap is not None:
            command.extend(["--prototype-min-negative-gap", str(float(prototype_min_negative_gap))])
        if prototype_calibration_output_path and prototype_calibration_output_path.is_file():
            command.extend(["--prototype-calibration-report", str(prototype_calibration_output_path)])
    return command
