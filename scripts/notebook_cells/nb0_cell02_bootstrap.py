# Auto-extracted from colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb cell 2.
# Keep notebook execute-only cells thin; edit behavior here.

from scripts.colab_notebook_bootstrap_helpers import setup_notebook_environment, resolve_github_token
from urllib.parse import urlencode

try:
    from scripts.colab_repo_bootstrap import _ensure_repo_root_for_update_check
    repo_root_for_update_check = _ensure_repo_root_for_update_check()
except Exception:
    repo_root_for_update_check = None

# [KONTROL] Ilk hucre: Bootstrap kontrati


def _build_repo_access_url(base: str, params: dict) -> str:
    """Build repo access URL for freshness check (validation contract)."""
    return f"{base}?{urlencode(params)}"

# Resolve tokens
gh_token = resolve_github_token()
if gh_token:
    print('[GIT] GitHub token Colab secret/env uzerinden bulundu.')
else:
    print('[GIT] GitHub token bulunamadi. Public read disinda clone/push gerekiyorsa GH_TOKEN ekleyin.')

# Bootstrap notebook environment (includes repo finding, cloning, and path setup)
ROOT = setup_notebook_environment(install_requirements=False, print_tokens=True)

# Check for repo updates
try:
    from scripts.colab_repo_bootstrap import probe_repo_update_status
    update_status = probe_repo_update_status(ROOT)
    relation = str(update_status.get('relation', 'unknown'))
    if relation == 'up_to_date':
        print('[KONTROL] Notebook/repo guncel gorunuyor.')
    elif relation == 'update_available':
        print(f"[KONTROL] Guncelleme var. Branch={update_status.get('branch', '')}")
    else:
        print(f"[KONTROL] Guncelleme durumu okunamadi: {update_status.get('message', 'bilgi yok')}")
except Exception as exc:
    print(f'[KONTROL] Guncellik kontrolu basarisiz: {exc}')
