from __future__ import annotations

from pathlib import Path


def run_access_check_nb2(ROOT: Path, print_fn=print) -> dict:
    """Check training model access."""
    from src.colab.repo_bootstrap import collect_notebook_access_report, print_notebook_access_report
    from src.core.config_manager import get_config

    CONFIG = get_config(environment='colab')
    BACKBONE_MODEL_NAME = str(dict(dict(CONFIG.get('training', {})).get('continual', {})).get('backbone', {}).get('model_name', '')).strip()

    print_fn("[SETUP] Checking model access for training...")
    ACCESS_REPORT = collect_notebook_access_report(
        repo_root=ROOT,
        hf_model_ids=[BACKBONE_MODEL_NAME] if BACKBONE_MODEL_NAME else [],
    )
    print_notebook_access_report(ACCESS_REPORT, print_fn=print_fn)
    if BACKBONE_MODEL_NAME:
        print_fn(f"[CHECK] Default backbone model: {BACKBONE_MODEL_NAME}")
    return ACCESS_REPORT


__all__ = [
    'run_access_check_nb2',
]
