# Auto-extracted from colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb cell 2.
# Keep notebook execute-only cells thin; edit behavior here.

from src.colab.notebook_bootstrap_helpers import setup_notebook_environment, resolve_github_token

try:
    from src.colab.repo_bootstrap import _ensure_repo_root_for_update_check
    repo_root_for_update_check = _ensure_repo_root_for_update_check()
except Exception:
    repo_root_for_update_check = None

# [CHECK] First cell: bootstrap contract

# Resolve tokens
gh_token = resolve_github_token()
if gh_token:
    print('[GIT] GitHub token found in a Colab secret or env var.')
else:
    print('[GIT] No GitHub token found. Add GH_TOKEN if clone/push needs more than public read access.')

# Bootstrap notebook environment (includes repo finding, cloning, and path setup)
ROOT = setup_notebook_environment(install_requirements=False, print_tokens=True)

# Check for repo updates
try:
    from src.colab.repo_bootstrap import probe_repo_update_status
    update_status = probe_repo_update_status(ROOT)
    relation = str(update_status.get('relation', 'unknown'))
    if relation == 'up_to_date':
        print('[CHECK] Notebook/repo is up to date.')
    elif relation == 'update_available':
        print(f"[CHECK] Update available. Branch={update_status.get('branch', '')}")
    else:
        print(f"[CHECK] Could not read update status: {update_status.get('message', 'no details')}")
except Exception as exc:
    print(f'[CHECK] Update check failed: {exc}')
