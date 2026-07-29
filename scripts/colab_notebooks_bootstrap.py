"""
Colab notebook bootstrap utilities.

Centralizes all non-interactive setup code (token resolution, repo discovery,
access checks, imports, config loading) so notebooks remain minimalist.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

NOTEBOOK2_SPARSE_PATHS = (
    "README.md",
    "docs",
    "src",
    "scripts",
    "config",
    "colab_notebooks",
    "requirements.txt",
    "requirements_colab.txt",
    "pyproject.toml",
)


# ============================================================================
# ENVIRONMENT & TOKEN RESOLUTION
# ============================================================================

def resolve_github_token() -> str:
    """Resolve GitHub token from environment or Colab secrets."""
    token_names = ("GH_TOKEN", "GITHUB_TOKEN")

    # Check environment variables first
    for env_name in token_names:
        token = str(os.environ.get(env_name, "")).strip()
        if token:
            os.environ.setdefault("GH_TOKEN", token)
            return token

    # Check Colab secrets
    if _running_in_colab():
        for secret_name in token_names:
            token = _resolve_colab_secret(secret_name)
            if token:
                os.environ["GH_TOKEN"] = token
                return token

    return ""


def resolve_huggingface_token() -> Optional[str]:
    """Resolve HuggingFace token from environment or Colab secrets."""
    token_names = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")

    # Check environment variables
    for env_name in token_names:
        token = str(os.environ.get(env_name, "")).strip()
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            return token

    # Check Colab secrets
    if _running_in_colab():
        for secret_name in token_names:
            token = _resolve_colab_secret(secret_name)
            if token:
                os.environ["HF_TOKEN"] = token
                return token

    return None


def _running_in_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_colab_secret(secret_name: str) -> str:
    """Resolve a secret from Google Colab's userdata."""
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


# ============================================================================
# REPO DISCOVERY & CLONING
# ============================================================================

def _is_repo_root(path: Path) -> bool:
    """Check if path is a valid repo root (has src/, config/, scripts/)."""
    return (
        (path / "src").is_dir()
        and (path / "config").is_dir()
        and (path / "scripts").is_dir()
    )


def _find_repo_root() -> Optional[Path]:
    """Find repo root from environment, current directory, or common locations."""
    # Check environment variables
    for env_var in ("AADS_REPO_ROOT", "REPO_ROOT"):
        raw = os.environ.get(env_var)
        if raw:
            candidate = Path(raw).expanduser().resolve()
            if _is_repo_root(candidate):
                return candidate

    # Check current directory and parents
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

    # Check for nested repo in Colab clone target
    clone_target = Path("/content/bitirmeprojesi")
    if clone_target.exists() and any(clone_target.iterdir()):
        for child in clone_target.iterdir():
            if child.is_dir() and _is_repo_root(child):
                return child

    return None


def _ensure_sparse_checkout_paths(repo_root: Path, paths: tuple[str, ...]) -> None:
    """Add missing sparse-checkout paths for an already cloned Colab repo."""
    if not (repo_root / ".git").exists():
        return
    missing = [path for path in paths if not (repo_root / path).exists()]
    if not missing:
        return
    print(f"[BOOTSTRAP] Adding missing sparse checkout paths: {missing}")
    subprocess.run(
        ["git", "sparse-checkout", "add", *missing],
        cwd=str(repo_root),
        check=True,
    )


def _build_repo_access_url(repo_url: str, token: Optional[str]) -> str:
    """Add authentication token to repo URL if provided."""
    if not token:
        return repo_url

    parsed = urlsplit(str(repo_url or "").strip())
    if parsed.scheme != "https" or not parsed.netloc:
        return repo_url

    netloc = parsed.netloc.split("@", 1)[-1]
    return urlunsplit(
        (parsed.scheme, f"{token}@{netloc}", parsed.path, parsed.query, parsed.fragment)
    )


def ensure_repo_root(
    auto_clone: bool = True,
    clone_target: Optional[Path] = None,
    repo_url: Optional[str] = None,
    github_token: Optional[str] = None,
    sparse_paths: Optional[Sequence[str]] = None,
) -> Path:
    """
    Ensure repo root is available (find or clone).

    Args:
        auto_clone: If True and repo not found, attempt to clone
        clone_target: Target path for cloning (default: /content/bitirmeprojesi)
        repo_url: Repository URL (default: GitHub URL)
        github_token: GitHub token for private repos
        sparse_paths: Optional sparse checkout paths to select after clone

    Returns:
        Path to repo root

    Raises:
        RuntimeError: If repo cannot be found or cloned
    """
    # Try to find existing repo
    repo_root = _find_repo_root()
    if repo_root is not None:
        _ensure_sparse_checkout_paths(repo_root, ("requirements_colab.txt",))
        return repo_root

    if not auto_clone:
        raise RuntimeError(
            "Repo not found. Set AADS_REPO_ROOT or REPO_ROOT environment variable, "
            "or enable auto_clone=True."
        )

    # Setup cloning parameters
    if clone_target is None:
        clone_target = Path("/content/bitirmeprojesi")
    if repo_url is None:
        repo_url = os.environ.get(
            "AADS_REPO_URL", "https://github.com/EfeErim/aads-open-world-plant-disease.git"
        )
    if github_token is None:
        github_token = os.environ.get("GH_TOKEN", "")

    if clone_target.exists() and not _is_repo_root(clone_target):
        shutil.rmtree(clone_target)

    clone_url = _build_repo_access_url(repo_url, github_token)
    clone_target.parent.mkdir(parents=True, exist_ok=True)
    clone_args = ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", clone_url, str(clone_target)]
    print(f"[BOOTSTRAP] Sparse cloning repo to {clone_target}...")
    completed = subprocess.run(
        clone_args,
        check=False,
    )

    if completed.returncode == 0:
        selected_paths = tuple(sparse_paths or (
            "README.md",
            "docs",
            "src",
            "scripts",
            "config",
            "colab_notebooks",
            "requirements.txt",
            "requirements_colab.txt",
            "pyproject.toml",
        ))
        print("[BOOTSTRAP] Selecting source checkout paths...")
        completed = subprocess.run(
            ["git", "sparse-checkout", "set", *selected_paths],
            cwd=str(clone_target),
            check=False,
        )

    if completed.returncode != 0 and clone_target.exists():
        print("[BOOTSTRAP] Sparse checkout failed; falling back to source-only checkout.")
        fallback_paths = [
            "README.md",
            "docs",
            "src",
            "scripts",
            "config",
            "colab_notebooks",
            "requirements.txt",
            "requirements_colab.txt",
            "pyproject.toml",
        ]
        subprocess.run(
            ["git", "sparse-checkout", "set", *fallback_paths],
            cwd=str(clone_target),
            check=False,
        )
    elif completed.returncode != 0:
        raise RuntimeError(
            f"Repo clone failed with code {completed.returncode}. "
            "Set AADS_REPO_ROOT or provide valid GH_TOKEN for private repos."
        )

    # Verify clone
    repo_root = _find_repo_root()
    if repo_root is None:
        raise RuntimeError(
            f"Clone succeeded but repo root not found at {clone_target}. "
            "Check clone output above."
        )

    return repo_root


# ============================================================================
# BOOTSTRAP INITIALIZATION
# ============================================================================

def bootstrap_notebook(
    notebook_name: str = "Notebook",
    require_colab_requirements: bool = True,
    auto_clone_repo: bool = True,
) -> dict:
    """
    Bootstrap a Colab notebook with all necessary setup.

    Performs:
    - Token resolution (GitHub, HuggingFace)
    - Repo discovery/cloning
    - sys.path setup
    - Colab requirements installation

    Args:
        notebook_name: Name of notebook for logging
        require_colab_requirements: If True, install colab_notebooks/requirements_colab.txt
        auto_clone_repo: If True, auto-clone repo if not found

    Returns:
        Dict with bootstrap result and paths:
        {
            "ROOT": Path,
            "IN_COLAB": bool,
            "GH_TOKEN": str,
            "HF_TOKEN": str,
            "bootstrap_status": "ok" | error message
        }
    """
    result = {
        "ROOT": None,
        "IN_COLAB": _running_in_colab(),
        "GH_TOKEN": "",
        "HF_TOKEN": "",
        "bootstrap_status": "initializing",
    }

    try:
        # Resolve tokens
        result["GH_TOKEN"] = resolve_github_token()
        if result["GH_TOKEN"]:
            print("[BOOTSTRAP] GitHub token found.")
        else:
            print("[BOOTSTRAP] GitHub token not found (public read only).")

        result["HF_TOKEN"] = resolve_huggingface_token()
        if result["HF_TOKEN"]:
            print("[BOOTSTRAP] HuggingFace token found.")
        else:
            print("[BOOTSTRAP] HuggingFace token not found (public models only).")

        # Ensure repo root
        notebook2_sparse_paths = (
            NOTEBOOK2_SPARSE_PATHS
            if "Notebook 2" in notebook_name or "Continual Adapter Training" in notebook_name
            else None
        )
        ROOT = ensure_repo_root(
            auto_clone=auto_clone_repo,
            github_token=result["GH_TOKEN"],
            sparse_paths=notebook2_sparse_paths,
        )
        result["ROOT"] = ROOT
        os.chdir(ROOT)
        print(f"[BOOTSTRAP] Repo root: {ROOT}")

        # Setup sys.path
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        # Install colab requirements
        if require_colab_requirements and result["IN_COLAB"]:
            colab_requirements = ROOT / "colab_notebooks" / "requirements_colab.txt"
            if colab_requirements.exists():
                print(f"[BOOTSTRAP] Installing {colab_requirements.name}...")
                print("[BOOTSTRAP] pip install may take several minutes on a fresh runtime.", flush=True)
                completed = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--progress-bar", "on", "-r", str(colab_requirements)],
                    check=False,
                )
                print(f"[BOOTSTRAP] pip install returncode={completed.returncode}", flush=True)

        result["bootstrap_status"] = "ok"
        print(f"[BOOTSTRAP] {notebook_name} bootstrap complete.")

    except Exception as exc:
        result["bootstrap_status"] = str(exc)
        print(f"[BOOTSTRAP] ERROR: {exc}")
        raise

    return result


def print_bootstrap_status(bootstrap_result: dict) -> None:
    """Print bootstrap status and token information."""
    print("\n" + "=" * 70)
    print("BOOTSTRAP STATUS")
    print("=" * 70)
    print(f"Status: {bootstrap_result.get('bootstrap_status', 'unknown')}")
    print(f"Repo Root: {bootstrap_result.get('ROOT', 'N/A')}")
    print(f"In Colab: {bootstrap_result.get('IN_COLAB', False)}")
    print(f"GitHub Token: {'✓' if bootstrap_result.get('GH_TOKEN') else '✗'}")
    print(f"HuggingFace Token: {'✓' if bootstrap_result.get('HF_TOKEN') else '✗'}")
    print("=" * 70 + "\n")
