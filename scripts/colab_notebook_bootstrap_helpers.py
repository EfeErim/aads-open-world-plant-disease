"""
Consolidated bootstrap helpers for all Colab notebooks.

This module extracts common repo detection, token resolution, and setup logic
from individual notebook cells, making notebooks cleaner and reducing duplication.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

# =============================================================================
# Token Resolution
# =============================================================================


def _running_in_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        import logging
        logging.exception('Unhandled exception')
        raise
        return False


def _resolve_colab_secret(secret_name: str) -> str:
    """Resolve a secret from Google Colab secrets."""
    if not _running_in_colab():
        return ""
    try:
        from google.colab import userdata
        return str(userdata.get(secret_name) or "").strip()
    except Exception:
        import logging
        logging.exception('Unhandled exception')
        raise
        return ""


def resolve_github_token() -> Optional[str]:
    """Resolve GitHub token from environment or Colab secrets."""
    github_token_names = ("GH_TOKEN", "GITHUB_TOKEN")

    # Check environment first
    for env_name in github_token_names:
        token = str(os.environ.get(env_name, "")).strip()
        if token:
            os.environ.setdefault("GH_TOKEN", token)
            return token

    # Check Colab secrets
    for secret_name in github_token_names:
        token = _resolve_colab_secret(secret_name)
        if token:
            os.environ["GH_TOKEN"] = token
            return token

    return None


def resolve_huggingface_token() -> Optional[str]:
    """Resolve Hugging Face token from environment or Colab secrets."""
    hf_token_names = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")

    # Check environment first
    for env_name in hf_token_names:
        token = str(os.environ.get(env_name, "")).strip()
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            return token

    # Check Colab secrets
    for secret_name in hf_token_names:
        token = _resolve_colab_secret(secret_name)
        if token:
            os.environ["HF_TOKEN"] = token
            return token

    return None


# =============================================================================
# Repo Detection and Bootstrap
# =============================================================================

def _is_repo_root(path: Path) -> bool:
    """Check if a path is a repo root (has src, config, scripts dirs)."""
    return (path / "src").is_dir() and (path / "config").is_dir() and (path / "scripts").is_dir()


def _build_repo_access_url(repo_url: str, token: Optional[str]) -> str:
    """Build repo URL with GitHub token authentication if provided."""
    if not token:
        return repo_url

    parsed = urlsplit(str(repo_url or "").strip())
    if parsed.scheme != "https" or not parsed.netloc:
        return repo_url

    netloc = parsed.netloc.split("@", 1)[-1]
    return urlunsplit((parsed.scheme, f"x-access-token:{token}@{netloc}", parsed.path, parsed.query, parsed.fragment))


def find_repo_root() -> Optional[Path]:
    """Find repo root by checking env vars, cwd, and common locations."""
    # Check explicit env vars
    for raw in (os.environ.get("AADS_REPO_ROOT"), os.environ.get("REPO_ROOT")):
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        if _is_repo_root(candidate):
            return candidate

    # Check cwd and parents
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if _is_repo_root(candidate):
            return candidate

    # Check common Colab locations
    common_candidates = (
        Path("/content/bitirme projesi"),
        Path("/content/bitirmeprojesi"),
        Path("/content/aads_ulora"),
        Path("/content/workspace"),
        Path("/content/project"),
    )
    for candidate in common_candidates:
        if _is_repo_root(candidate):
            return candidate

    # Check if any subdirs of common locations are repo roots
    clone_target = Path("/content/bitirmeprojesi")
    if clone_target.exists() and any(clone_target.iterdir()):
        for child in clone_target.iterdir():
            if child.is_dir() and _is_repo_root(child):
                return child

    return None


def bootstrap_repo_root(
    repo_url: Optional[str] = None,
    clone_target: Optional[Path] = None,
    use_github_token: bool = True,
) -> Path:
    """
    Find or clone repo root, ensuring it's available and sys.path is updated.

    Args:
        repo_url: GitHub repo URL (default from env or hardcoded)
        clone_target: Where to clone if not found (default /content/bitirmeprojesi)
        use_github_token: Whether to use GitHub token for authentication

    Returns:
        Path to repo root

    Raises:
        RuntimeError: If repo bootstrap fails
    """
    if repo_url is None:
        repo_url = os.environ.get("AADS_REPO_URL", "https://github.com/EfeErim/bitirmeprojesi.git")

    if clone_target is None:
        clone_target = Path("/content/bitirmeprojesi")

    # Try to find existing repo
    print("[SETUP] Checking repository checkout...", flush=True)
    repo_root = find_repo_root()
    if repo_root is not None:
        print(f"[SETUP] Repository ready: {repo_root}", flush=True)
        return repo_root

    # Clone repo
    token = resolve_github_token() if use_github_token else None
    clone_url = _build_repo_access_url(repo_url, token)
    clone_target.parent.mkdir(parents=True, exist_ok=True)
    print("[SETUP] Repository not found locally. Cloning the latest demo code...", flush=True)

    completed = subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, str(clone_target)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout)

    if completed.returncode != 0:
        raise RuntimeError(
            "Repo bootstrap failed. Set AADS_REPO_ROOT for an existing checkout or "
            "provide GH_TOKEN/GITHUB_TOKEN for private repo auto-clone."
        )

    # Verify clone was successful
    repo_root = find_repo_root()
    if repo_root is not None:
        print(f"[SETUP] Repository ready: {repo_root}", flush=True)
        return repo_root

    raise RuntimeError(
        "Repo bootstrap failed. Set AADS_REPO_ROOT for existing checkout or "
        "provide GH_TOKEN/GITHUB_TOKEN for private repo auto-clone."
    )


def setup_notebook_environment(
    repo_url: Optional[str] = None,
    install_requirements: bool = True,
    print_tokens: bool = True,
    requirements_file: Optional[str | Path] = None,
) -> Path:
    """
    Complete notebook bootstrap: resolve repo, set sys.path, install requirements.

    Args:
        repo_url: GitHub repo URL (optional, uses default if not provided)
        install_requirements: Whether to install Colab requirements
        print_tokens: Whether to print token resolution status
        requirements_file: Repo-relative requirements file override

    Returns:
        Path to repo root
    """
    # Resolve tokens and print status
    print("[SETUP] Starting notebook environment setup...", flush=True)
    print("[SETUP] Checking Colab access tokens...", flush=True)
    gh_token = resolve_github_token()
    hf_token = resolve_huggingface_token()

    if print_tokens:
        if gh_token:
            print("[GIT] GitHub token resolved from env/secret.")
        else:
            print("[GIT] GitHub token not found. Public read only; set GH_TOKEN for private repos.")

        if hf_token:
            print("[HF] Hugging Face token resolved. Gated model access available.")
        else:
            print("[HF] Hugging Face token not found. Gated models may require setup.")

    # Bootstrap repo
    repo_root = bootstrap_repo_root(repo_url=repo_url)
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Install requirements in Colab if needed
    if install_requirements and _running_in_colab():
        from scripts.colab_repo_bootstrap import install_colab_requirements
        relative_requirements = Path(
            requirements_file or os.environ.get("AADS_COLAB_REQUIREMENTS_FILE", "requirements_colab.txt")
        )
        requirements_path = (repo_root / relative_requirements).resolve()
        try:
            requirements_path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Colab requirements file must stay under the repo root: {relative_requirements}") from exc
        print(f"[SETUP] Dependency profile: {relative_requirements}")
        install_colab_requirements(requirements_path, in_colab=True)

    print("[SETUP] Notebook environment ready.", flush=True)
    return repo_root
