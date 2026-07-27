from __future__ import annotations
from pathlib import Path


def run_bootstrap_notebook_nb3(notebook_name: str = "Notebook 3: Adapter Smoke Test", require_colab_requirements: bool = True, auto_clone_repo: bool = True) -> dict:
    """Bootstrap Notebook 3 with repo discovery."""
    from scripts.colab_notebooks_bootstrap import bootstrap_notebook, print_bootstrap_status
    try:
        from scripts.colab_repo_bootstrap import _ensure_repo_root_for_update_check
        repo_root_for_update_check = _ensure_repo_root_for_update_check()
    except Exception:
        repo_root_for_update_check = None

    # [KONTROL] Ilk hucre: Bootstrap kontrati
    BOOTSTRAP = bootstrap_notebook(
        notebook_name=notebook_name,
        require_colab_requirements=require_colab_requirements,
        auto_clone_repo=auto_clone_repo,
    )
    ROOT = BOOTSTRAP["ROOT"]
    print_bootstrap_status(BOOTSTRAP)
    return BOOTSTRAP


def run_access_check_nb3(ROOT: Path, print_fn=print) -> dict:
    """Check adapter smoke test model access."""
    from src.core.config_manager import get_config
    from scripts.colab_repo_bootstrap import collect_notebook_access_report, print_notebook_access_report

    CONFIG = get_config(environment='colab')
    MODEL_IDS = [
        str(m).strip()
        for m in [
            CONFIG.get('training', {}).get('continual', {}).get('base_model_id', ''),
            CONFIG.get('router', {}).get('vlm', {}).get('model_ids', {}).get('fallback', ''),
        ]
        if str(m).strip()
    ]

    print_fn("[SETUP] Checking model access for smoke test...")
    ACCESS_REPORT = collect_notebook_access_report(repo_root=ROOT, hf_model_ids=MODEL_IDS)
    print_notebook_access_report(ACCESS_REPORT, print_fn=print_fn)
    return ACCESS_REPORT


__all__ = [
    'run_bootstrap_notebook_nb3',
    'run_access_check_nb3',
]
