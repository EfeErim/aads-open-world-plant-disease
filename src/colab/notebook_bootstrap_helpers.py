"""One-call environment setup for Notebooks 0, 1 and 16: find or clone the repo, then install requirements."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from src.colab.repo_bootstrap import (
    COLAB_CLONE_TARGET,
    find_repo_root,
    resolve_github_token,
    running_in_colab,
)
from src.colab.repo_bootstrap import resolve_hf_token as resolve_huggingface_token


def _build_repo_access_url(repo_url: str, token: Optional[str]) -> str:
    """Build repo URL with GitHub token authentication if provided."""
    if not token:
        return repo_url

    parsed = urlsplit(str(repo_url or "").strip())
    if parsed.scheme != "https" or not parsed.netloc:
        return repo_url

    netloc = parsed.netloc.split("@", 1)[-1]
    return urlunsplit((parsed.scheme, f"x-access-token:{token}@{netloc}", parsed.path, parsed.query, parsed.fragment))


def bootstrap_repo_root(
    repo_url: Optional[str] = None,
    clone_target: Optional[Path] = None,
    use_github_token: bool = True,
) -> Path:
    """
    Find or clone repo root, ensuring it's available and sys.path is updated.

    Args:
        repo_url: GitHub repo URL (default from env or hardcoded)
        clone_target: Where to clone if not found (default COLAB_CLONE_TARGET)
        use_github_token: Whether to use GitHub token for authentication

    Returns:
        Path to repo root

    Raises:
        RuntimeError: If repo bootstrap fails
    """
    if repo_url is None:
        repo_url = os.environ.get("AADS_REPO_URL", "https://github.com/EfeErim/aads-open-world-plant-disease.git")

    if clone_target is None:
        clone_target = COLAB_CLONE_TARGET

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
    if install_requirements and running_in_colab():
        from src.colab.repo_bootstrap import install_colab_requirements
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
