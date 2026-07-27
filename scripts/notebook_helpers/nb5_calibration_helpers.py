from __future__ import annotations

import importlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path


def _parse_version(value: str) -> tuple[int, int, int]:
    parts = str(value or "0").split(".")
    parsed = []
    for part in parts[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        parsed.append(int(digits or 0))
    while len(parsed) < 3:
        parsed.append(0)
    return tuple(parsed)


def ensure_router_dependencies_nb5(print_fn=print) -> dict:
    """Install Notebook 5 router dependencies before calibration imports."""
    report = {
        "installed": [],
        "transformers_version_before": "",
        "transformers_version_after": "",
        "open_clip_available_before": False,
        "open_clip_available_after": False,
    }

    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = ""
    report["transformers_version_before"] = transformers_version

    open_clip_available = importlib.util.find_spec("open_clip") is not None
    report["open_clip_available_before"] = open_clip_available

    packages = []
    parsed_transformers = _parse_version(transformers_version) if transformers_version else (0, 0, 0)
    if not ((5, 1, 0) <= parsed_transformers < (5, 2, 0)):
        packages.append("transformers>=5.1.0,<5.2.0")
    if not open_clip_available:
        packages.append("open-clip-torch")

    if packages:
        print_fn("[SETUP] Installing Notebook 5 router dependencies before model load...")
        print_fn("[SETUP] " + " ".join(packages))
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", *packages],
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Notebook 5 dependency install failed with code {completed.returncode}")
        report["installed"] = packages

    try:
        report["transformers_version_after"] = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        report["transformers_version_after"] = ""
    report["open_clip_available_after"] = importlib.util.find_spec("open_clip") is not None

    print_fn(
        "[SETUP] Dependency check: "
        f"transformers={report['transformers_version_after'] or 'missing'} "
        f"open_clip={report['open_clip_available_after']}"
    )
    return report


def run_bootstrap_notebook_nb5(notebook_name: str = "Notebook 5: Router Calibration", require_colab_requirements: bool = True, auto_clone_repo: bool = True) -> dict:
    """Bootstrap Notebook 5."""
    from scripts.colab_notebooks_bootstrap import bootstrap_notebook, print_bootstrap_status
    try:
        from scripts.colab_repo_bootstrap import _ensure_repo_root_for_update_check
        _ensure_repo_root_for_update_check()
    except Exception:
        pass

    # [KONTROL] Ilk hucre: Bootstrap kontrati
    BOOTSTRAP = bootstrap_notebook(
        notebook_name=notebook_name,
        require_colab_requirements=require_colab_requirements,
        auto_clone_repo=auto_clone_repo,
    )
    print_bootstrap_status(BOOTSTRAP)
    return BOOTSTRAP


def run_access_check_nb5(ROOT: Path, print_fn=print) -> dict:
    """Check calibration router model access."""
    from scripts.colab_repo_bootstrap import collect_notebook_access_report, print_notebook_access_report
    from src.core.config_manager import get_config

    CONFIG = get_config(environment='colab')
    ROUTER_VLM_CFG = dict(dict(CONFIG.get('router', {})).get('vlm', {}))
    ROUTER_MODEL_IDS = [
        str(model_id).strip()
        for model_id in list(dict(ROUTER_VLM_CFG.get('model_ids', {})).values())
        if str(model_id).strip()
    ]

    print_fn("[SETUP] Checking model access for router calibration...")
    ACCESS_REPORT = collect_notebook_access_report(repo_root=ROOT, hf_model_ids=ROUTER_MODEL_IDS)
    print_notebook_access_report(ACCESS_REPORT, print_fn=print_fn)
    return ACCESS_REPORT


__all__ = [
    'ensure_router_dependencies_nb5',
    'run_bootstrap_notebook_nb5',
    'run_access_check_nb5',
]
