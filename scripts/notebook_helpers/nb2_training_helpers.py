from __future__ import annotations

from pathlib import Path


def run_bootstrap_notebook_nb2(notebook_name: str = "Notebook 2: Continual Adapter Training", require_colab_requirements: bool = True, auto_clone_repo: bool = True) -> dict:
    """Bootstrap Notebook 2."""
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


def run_access_check_nb2(ROOT: Path, print_fn=print) -> dict:
    """Check training model access."""
    from scripts.colab_repo_bootstrap import collect_notebook_access_report, print_notebook_access_report
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
        print_fn(f"[KONTROL] Varsayilan backbone modeli: {BACKBONE_MODEL_NAME}")
    return ACCESS_REPORT


__all__ = [
    'run_bootstrap_notebook_nb2',
    'run_access_check_nb2',
]
