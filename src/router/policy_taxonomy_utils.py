#!/usr/bin/env python3
"""Policy/profile and taxonomy utility helpers for VLM router."""

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.router.label_normalization import normalize_part_label
from src.router.runtime_surface import coerce_bool

logger = logging.getLogger(__name__)

__all__ = [
    "deep_merge_dicts",
    "default_policy_graph",
    "build_policy_graph",
    "resolve_requested_profile",
    "apply_runtime_profile",
    "policy_value",
    "policy_enabled",
    "load_taxonomy",
    "load_crop_part_compatibility",
]


def _resolve_taxonomy_path(taxonomy_path: str) -> Optional[Path]:
    path = Path(taxonomy_path)
    if path.is_absolute():
        return path if path.exists() else None

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = Path(__file__).resolve().parent.parent.parent / path
    if repo_candidate.exists():
        return repo_candidate

    return None


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries without mutating inputs."""
    merged = copy.deepcopy(base) if isinstance(base, dict) else {}
    if not isinstance(override, dict):
        return merged

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def default_policy_graph() -> Dict[str, Dict[str, Any]]:
    """Return default policy stage graph for router execution gates."""
    return {
        "roi_filter": {"enabled": True},
        "part_evidence": {"enabled": True},
        "crop_evidence": {"enabled": True},
        "compatibility_fusion": {"enabled": True},
        "part_resolution": {"enabled": True},
        "open_set_gate": {"enabled": True},
        "dedupe": {"enabled": True},
    }


def build_policy_graph(vlm_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build effective policy graph from defaults and config overrides."""
    configured_policy = vlm_config.get("policy_graph", {})
    if not isinstance(configured_policy, dict):
        configured_policy = {}
    return deep_merge_dicts(default_policy_graph(), configured_policy)


def resolve_requested_profile(vlm_config: Dict[str, Any]) -> Optional[str]:
    """Resolve active runtime profile from env override or config."""
    env_profile = str(os.getenv("AADS_ULORA_VLM_PROFILE", "")).strip()
    if env_profile:
        return env_profile

    config_profile = vlm_config.get("profile")
    if isinstance(config_profile, str) and config_profile.strip():
        return config_profile.strip()
    return None


def apply_runtime_profile(
    base_vlm_config: Dict[str, Any],
    profile_name: Optional[str],
    suppress_warning: bool = False,
) -> Tuple[Dict[str, Any], Optional[str], bool]:
    """Apply named profile overlay and return (config, active_profile, changed)."""
    vlm_config = copy.deepcopy(base_vlm_config)
    active_profile: Optional[str] = None

    if not profile_name:
        return vlm_config, active_profile, False

    profiles = base_vlm_config.get("profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}

    profile_cfg = profiles.get(profile_name)
    if not isinstance(profile_cfg, dict):
        if not suppress_warning:
            logger.warning(f"Requested VLM profile '{profile_name}' not found; using base config")
        return vlm_config, active_profile, False

    vlm_config = deep_merge_dicts(base_vlm_config, profile_cfg)
    vlm_config["profile"] = profile_name
    active_profile = profile_name
    logger.info(f"Applied VLM profile: {profile_name}")
    return vlm_config, active_profile, True


def policy_value(
    policy_graph: Dict[str, Dict[str, Any]], vlm_config: Dict[str, Any], stage: str, key: str, default: Any
) -> Any:
    """Read policy value with stage override precedence and config fallback."""
    stage_cfg = policy_graph.get(stage, {})
    if isinstance(stage_cfg, dict) and key in stage_cfg:
        return stage_cfg.get(key)
    return vlm_config.get(key, default)


def policy_enabled(policy_graph: Dict[str, Dict[str, Any]], stage: str, default: bool = True) -> bool:
    """Return effective enabled flag for a policy stage."""
    stage_cfg = policy_graph.get(stage, {})
    if not isinstance(stage_cfg, dict):
        return coerce_bool(default, default=default)
    return coerce_bool(stage_cfg.get("enabled", default), default=default)


def load_taxonomy(taxonomy_path: str) -> Tuple[List[str], List[str]]:
    """Load crop/part labels from taxonomy file."""
    path = _resolve_taxonomy_path(taxonomy_path)
    if path is None:
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load taxonomy from {path}: {e}")
        raise RuntimeError(f"Failed to load taxonomy from {path}") from e

    crops = taxonomy.get("crops", [])
    weeds = taxonomy.get("common_weeds", [])
    ornamentals = taxonomy.get("ornamentals", [])
    all_crops = crops + weeds + ornamentals

    parts_data = taxonomy.get("parts", [])
    if isinstance(parts_data, dict):
        core_parts = parts_data.get("core", [])
        extended_parts = parts_data.get("extended", [])
        parts = core_parts + extended_parts
        logger.info(f"Loaded taxonomy from {path}: {len(all_crops)} crops, {len(parts)} parts (core+extended)")
    else:
        parts = parts_data
        logger.info(f"Loaded taxonomy from {path}: {len(all_crops)} plant types, {len(parts)} part types")

    return all_crops, parts


def load_crop_part_compatibility(taxonomy_path: str) -> Dict[str, List[str]]:
    """Load normalized crop->compatible-parts mapping from taxonomy file."""
    path = _resolve_taxonomy_path(taxonomy_path)
    if path is None:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    compatibility = taxonomy.get("crop_part_compatibility", {})
    if not isinstance(compatibility, dict):
        return {}

    normalized: Dict[str, List[str]] = {}
    for crop_name, parts in compatibility.items():
        if not isinstance(parts, list):
            continue
        crop_key = str(crop_name).strip().lower()
        part_values = list(dict.fromkeys(normalize_part_label(part) for part in parts if normalize_part_label(part)))
        if crop_key and part_values:
            normalized[crop_key] = part_values
    return normalized
