from __future__ import annotations

from pathlib import Path


def run_access_check_nb3(ROOT: Path, print_fn=print) -> dict:
    """Check adapter smoke test model access."""
    from src.colab.repo_bootstrap import collect_notebook_access_report, print_notebook_access_report
    from src.core.config_manager import get_config

    CONFIG = get_config(environment='colab')
    backbone_model = str(
        dict(CONFIG.get('training', {}).get('continual', {})).get('backbone', {}).get('model_name', '')
    ).strip()
    MODEL_IDS = [backbone_model] if backbone_model else []

    print_fn("[SETUP] Checking model access for smoke test...")
    ACCESS_REPORT = collect_notebook_access_report(repo_root=ROOT, hf_model_ids=MODEL_IDS)
    print_notebook_access_report(ACCESS_REPORT, print_fn=print_fn)
    return ACCESS_REPORT


__all__ = [
    'run_access_check_nb3',
]
