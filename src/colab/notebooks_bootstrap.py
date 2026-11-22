"""Colab notebook bootstrap: find or clone the repository, resolve tokens and install requirements.

`run_notebook_bootstrap` is the first cell of Notebooks 2, 3 and 9; `setup_notebook_environment` is the first
cell of Notebooks 0 and 1.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from src.colab.git_utils import build_repo_access_url
from src.colab.repo_bootstrap import (
    COLAB_CLONE_TARGET,
    find_repo_root,
    is_repo_root,
    resolve_hf_token,
    running_in_colab,
)
from src.colab.repo_bootstrap import resolve_github_token as resolve_repo_github_token

SOURCE_CHECKOUT_PATHS = (
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


def resolve_github_token() -> str:
    """Resolve a GitHub token from env vars first, then Colab secrets ("" when absent)."""
    return resolve_repo_github_token() or ""


def resolve_huggingface_token() -> Optional[str]:
    """Resolve a Hugging Face token from env vars first, then Colab secrets."""
    return resolve_hf_token()


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


def ensure_repo_root(
    auto_clone: bool = True,
    clone_target: Optional[Path] = None,
    repo_url: Optional[str] = None,
    github_token: Optional[str] = None,
    sparse_paths: Optional[Sequence[str]] = None,
) -> Path:
    """Ensure repo root is available (find or clone)."""
    repo_root = find_repo_root()
    if repo_root is not None:
        _ensure_sparse_checkout_paths(repo_root, ("requirements_colab.txt",))
        return repo_root

    if not auto_clone:
        raise RuntimeError(
            "Repo not found. Set AADS_REPO_ROOT or REPO_ROOT environment variable, "
            "or enable auto_clone=True."
        )

    if clone_target is None:
        clone_target = COLAB_CLONE_TARGET
    if repo_url is None:
        repo_url = os.environ.get(
            "AADS_REPO_URL", "https://github.com/EfeErim/aads-open-world-plant-disease.git"
        )
    if github_token is None:
        github_token = os.environ.get("GH_TOKEN", "")

    if clone_target.exists() and not is_repo_root(clone_target):
        shutil.rmtree(clone_target)

    clone_url = build_repo_access_url(repo_url, github_token)
    clone_target.parent.mkdir(parents=True, exist_ok=True)
    clone_args = ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", clone_url, str(clone_target)]
    print(f"[BOOTSTRAP] Sparse cloning repo to {clone_target}...")
    completed = subprocess.run(
        clone_args,
        check=False,
    )

    if completed.returncode == 0:
        selected_paths = tuple(sparse_paths or SOURCE_CHECKOUT_PATHS)
        print("[BOOTSTRAP] Selecting source checkout paths...")
        completed = subprocess.run(
            ["git", "sparse-checkout", "set", *selected_paths],
            cwd=str(clone_target),
            check=False,
        )

    if completed.returncode != 0 and clone_target.exists():
        print("[BOOTSTRAP] Sparse checkout failed; falling back to source-only checkout.")
        subprocess.run(
            ["git", "sparse-checkout", "set", *SOURCE_CHECKOUT_PATHS],
            cwd=str(clone_target),
            check=False,
        )
    elif completed.returncode != 0:
        raise RuntimeError(
            f"Repo clone failed with code {completed.returncode}. "
            "Set AADS_REPO_ROOT or provide valid GH_TOKEN for private repos."
        )

    repo_root = find_repo_root()
    if repo_root is None:
        raise RuntimeError(
            f"Clone succeeded but repo root not found at {clone_target}. "
            "Check clone output above."
        )

    return repo_root


def bootstrap_notebook(
    notebook_name: str = "Notebook",
    require_colab_requirements: bool = True,
    auto_clone_repo: bool = True,
) -> dict:
    """Bootstrap a Colab notebook with all necessary setup."""
    result = {
        "ROOT": None,
        "IN_COLAB": running_in_colab(),
        "GH_TOKEN": "",
        "HF_TOKEN": "",
        "bootstrap_status": "initializing",
    }

    try:
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

        ROOT = ensure_repo_root(auto_clone=auto_clone_repo, github_token=result["GH_TOKEN"])
        result["ROOT"] = ROOT
        os.chdir(ROOT)
        print(f"[BOOTSTRAP] Repo root: {ROOT}")

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

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


def run_notebook_bootstrap(
    notebook_name: str,
    *,
    require_colab_requirements: bool = True,
    auto_clone_repo: bool = True,
) -> dict:
    """First-cell bootstrap for Notebooks 2, 3 and 9: update check, repo, requirements, status report."""
    try:
        from src.colab.repo_bootstrap import _ensure_repo_root_for_update_check

        _ensure_repo_root_for_update_check()
    except (ImportError, FileNotFoundError):
        # Repo not importable or not found yet: bootstrap_notebook clones it or reports why it cannot.
        pass
    result = bootstrap_notebook(
        notebook_name=notebook_name,
        require_colab_requirements=require_colab_requirements,
        auto_clone_repo=auto_clone_repo,
    )
    print_bootstrap_status(result)
    return result


def bootstrap_repo_root(repo_url: Optional[str] = None) -> Path:
    """Find or clone repo root, ensuring it's available and sys.path is updated."""
    if repo_url is None:
        repo_url = os.environ.get("AADS_REPO_URL", "https://github.com/EfeErim/aads-open-world-plant-disease.git")
    clone_target = COLAB_CLONE_TARGET

    print("[SETUP] Checking repository checkout...", flush=True)
    repo_root = find_repo_root()
    if repo_root is not None:
        print(f"[SETUP] Repository ready: {repo_root}", flush=True)
        return repo_root

    token = resolve_github_token()
    clone_url = build_repo_access_url(repo_url, token)
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
    """Complete notebook bootstrap: resolve repo, set sys.path, install requirements."""
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

    repo_root = bootstrap_repo_root(repo_url=repo_url)
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

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
