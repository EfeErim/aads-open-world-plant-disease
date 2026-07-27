from __future__ import annotations


def run_bootstrap_notebook_nb4(notebook_name: str = "Notebook 4: Simple Adapter UI", require_colab_requirements: bool = True, auto_clone_repo: bool = True) -> dict:
    """Bootstrap Notebook 4."""
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


__all__ = ['run_bootstrap_notebook_nb4']
