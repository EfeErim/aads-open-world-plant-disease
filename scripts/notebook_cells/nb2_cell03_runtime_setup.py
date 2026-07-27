# Auto-extracted from colab_notebooks/2_train_continual_sd_lora_adapter.ipynb cell 3.
# Keep notebook execute-only cells thin; edit behavior here.

import io
import json
import os
import random
import shutil
import sys
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

# Defer heavy imports until actually needed (train/visualization cells)
matplotlib = None
plt = None
torch = None

def _ensure_torch():
    """Lazy import torch when needed."""
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    return torch

def _ensure_matplotlib():
    """Lazy import and configure matplotlib when needed."""
    global matplotlib, plt
    if matplotlib is None:
        import matplotlib as _matplotlib
        _matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        matplotlib = _matplotlib
        plt = _plt
    return matplotlib, plt

def _infer_part_name_from_dataset_name(dataset_name: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(dataset_name or "").strip())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    tokens = [token for token in normalized.strip("_").split("_") if token]
    if len(tokens) <= 1:
        return "unspecified"
    return "_".join(tokens[1:]) or "unspecified"

# Imports for training notebook setup
from scripts.colab_repo_bootstrap import (
    export_current_colab_notebook,
    install_colab_requirements,
    login_and_check_hf_token,
    mirror_checkpoint_state_to_repo,
    mirror_path_to_repo,
    push_repo_run_to_github,
    resolve_github_token,
    resolve_hf_token,
    resolve_repo_root,
    running_in_colab,
)

from scripts.colab_checkpointing import TrainingCheckpointManager
from scripts.colab_live_telemetry import ColabLiveTelemetry
from src.core.config_manager import ConfigurationManager
from src.shared.adapter_paths import build_adapter_bundle_root
from scripts.colab_notebook_helpers import build_notebook_run_dir, build_notebook_run_id

# Install Colab requirements if running in Colab
IN_COLAB = running_in_colab()
if IN_COLAB:
    install_colab_requirements(ROOT / 'colab_notebooks' / 'requirements_colab.txt', IN_COLAB)

# Allow notebook wrappers to override the exported notebook name and filename.
NOTEBOOK_NAME = str(globals().get('NOTEBOOK_NAME', '2_train_continual_sd_lora_adapter.ipynb'))
NOTEBOOK_FILENAME = str(globals().get('NOTEBOOK_FILENAME', '2_train_continual_sd_lora_adapter.executed.ipynb'))

# Initialize telemetry and run directories
CROP_NAME = globals().get('CROP_NAME', 'tomato')
_requested_part_name = str(globals().get('PART_NAME', '') or '').strip().lower()
if not _requested_part_name or _requested_part_name == 'unspecified':
    _requested_part_name = _infer_part_name_from_dataset_name(globals().get('DATASET_NAME', ''))
PART_NAME = _requested_part_name or 'unspecified'
RUN_ID = build_notebook_run_id(CROP_NAME, PART_NAME)
REPO_RUN_DIR = build_notebook_run_dir(ROOT, CROP_NAME, PART_NAME, RUN_ID)
REPO_NOTEBOOK_OUTPUT_PATH = REPO_RUN_DIR / 'notebooks' / NOTEBOOK_FILENAME
LOCAL_OUTPUT_DIR = ROOT / 'outputs' / 'colab_notebook_training'
REPO_OUTPUT_DIR = REPO_RUN_DIR / 'outputs' / 'colab_notebook_training'
REPO_TELEMETRY_DIR = REPO_RUN_DIR / 'telemetry'
REPO_CHECKPOINT_STATE_DIR = REPO_RUN_DIR / 'checkpoint_state'

LOCAL_TELEMETRY_ROOT = ROOT / 'outputs' / 'colab_notebook_training' / 'telemetry_runtime'
LOCAL_TELEMETRY_SPOOL_ROOT = ROOT / '.runtime_tmp' / 'colab_notebook_training' / 'telemetry_spool'

TELEMETRY = ColabLiveTelemetry(
    notebook_name=NOTEBOOK_NAME,
    run_id=RUN_ID,
    drive_root=LOCAL_TELEMETRY_ROOT,
    local_root=LOCAL_TELEMETRY_SPOOL_ROOT,
)
CHECKPOINT_ROOT = TELEMETRY.drive_run_dir
CHECKPOINT_MANAGER = TrainingCheckpointManager(CHECKPOINT_ROOT, retention=3)
DEVICE = str(ConfigurationManager(config_dir=str(ROOT / 'config'), environment='colab').load_all_configs().get('training', {}).get('continual', {}).get('device', 'cuda'))

LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPO_NOTEBOOK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def rt(message: str, *, phase: str = 'notebook') -> None:
    text = str(message)
    print(text)
    TELEMETRY.emit_log(text, phase=phase, level='info')

def save_run_outputs_to_repo() -> dict[str, str]:
    exports: dict[str, str] = {}
    mirrored_outputs = mirror_path_to_repo(LOCAL_OUTPUT_DIR, REPO_OUTPUT_DIR, exclude_dir_names=("checkpoints", "telemetry_runtime"))
    if mirrored_outputs is not None:
        exports['outputs'] = str(mirrored_outputs)
    telemetry_source = TELEMETRY.drive_run_dir
    mirrored_telemetry = mirror_path_to_repo(telemetry_source, REPO_TELEMETRY_DIR)
    if mirrored_telemetry is not None:
        exports['telemetry'] = str(mirrored_telemetry)
    mirrored_checkpoint_state = mirror_checkpoint_state_to_repo(CHECKPOINT_ROOT, REPO_CHECKPOINT_STATE_DIR)
    if mirrored_checkpoint_state is not None:
        exports['checkpoint_state'] = str(mirrored_checkpoint_state)
    return exports

TELEMETRY.configure_repo_output_export(
    notebook_path=REPO_NOTEBOOK_OUTPUT_PATH,
    export_notebook_fn=export_current_colab_notebook,
)
TELEMETRY.update_latest({'phase': 'bootstrap_ready', 'run_id': RUN_ID})
print(f'[BOOTSTRAP] run_id={RUN_ID} crop={CROP_NAME} part={PART_NAME}')
