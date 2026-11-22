# Auto-extracted from colab_notebooks/2_train_continual_sd_lora_adapter.ipynb cell 1.
# Keep notebook execute-only cells thin; edit behavior here.

# Bootstrap notebook via helper. Model access is checked in the visible
# Notebook 2 access-check cell, so the first cell stays setup-only.
from src.colab.repo_bootstrap import probe_repo_update_status
try:
    from src.colab.repo_bootstrap import _ensure_repo_root_for_update_check
    repo_root_for_update_check = _ensure_repo_root_for_update_check()
except Exception:
    repo_root_for_update_check = None

# [CHECK] First cell: bootstrap contract
from src.colab.notebooks_bootstrap import run_notebook_bootstrap
BOOTSTRAP = run_notebook_bootstrap("Notebook 2: Continual Adapter Training")
ROOT = BOOTSTRAP["ROOT"]
print("[NEXT] Go to the parameter and adapter selection cell; model access is verified separately in the access-check cell.")
