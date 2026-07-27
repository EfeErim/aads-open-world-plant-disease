# Auto-extracted from colab_notebooks/4_simple_direct_adapter_test_ui.ipynb cell 1.
# Keep notebook execute-only cells thin; edit behavior here.

import os
import subprocess
import sys
from pathlib import Path


def _configure_colab_git_read_access() -> None:
    token = str(os.environ.get("AADS_GITHUB_RELEASE_READ_TOKEN", "")).strip()
    if not token:
        try:
            from google.colab import userdata

            token = str(userdata.get("AADS_GITHUB_RELEASE_READ_TOKEN") or "").strip()
        except Exception:  # Colab secret access raises provider-specific exceptions.
            token = ""
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    if not token:
        return
    askpass = Path("/tmp/aads_git_read_askpass.sh")
    askpass.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "*Username*) printf '%s\\n' 'x-access-token' ;;\n"
        "*) printf '%s\\n' \"$AADS_GIT_READ_TOKEN\" ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    askpass.chmod(0o700)
    os.environ["GIT_ASKPASS"] = str(askpass)
    os.environ["GIT_ASKPASS_REQUIRE"] = "force"
    os.environ["AADS_GIT_READ_TOKEN"] = token


_configure_colab_git_read_access()


CLONE_TARGET = Path("/content/bitirmeprojesi")
DEFAULT_REPO_URL = "https://github.com/EfeErim/bitirmeprojesi.git"
REPO_URL = os.environ.get("AADS_REPO_URL", DEFAULT_REPO_URL)
REPO_REF = os.environ.get("AADS_REPO_REF", "master")


def _run_git(args: list[str], *, cwd: Path | None = None) -> None:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError("git command failed: git " + " ".join(args))


def _ensure_aads_repo_on_path() -> Path:
    target = CLONE_TARGET
    if target.exists() and not (target / ".git").exists():
        target = CLONE_TARGET.with_name(CLONE_TARGET.name + "_repo")
        print(f"{CLONE_TARGET} exists but is not a git checkout; cloning into {target}.")

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning Notebook 4 repo from {REPO_URL} into {target}...")
        _run_git(["clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(target)])
    elif (target / ".git").exists():
        print(f"Notebook 4 repo already exists: {target}")
        _run_git(["fetch", "--depth", "1", "origin", REPO_REF], cwd=target)
        _run_git(["checkout", REPO_REF], cwd=target)
        _run_git(["reset", "--hard", f"origin/{REPO_REF}"], cwd=target)
    else:
        raise RuntimeError(f"Notebook 4 clone target is not usable: {target}")

    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
    os.chdir(target)
    print(f"Notebook 4 repo ready: {target}")
    return target


repo_root = _ensure_aads_repo_on_path()
ROOT = repo_root
SEARCH_ROOTS = [
    ROOT / "outputs" / "colab_notebook_training",
    ROOT / "outputs" / "colab_notebook_training" / "telemetry_runtime" / "telemetry",
    ROOT / "models" / "adapters",
]

from scripts.colab_repo_bootstrap import (  # noqa: E402
    collect_notebook_access_report,
    install_colab_requirements,
    login_and_check_hf_token,
    print_notebook_access_report,
    resolve_hf_token,
    running_in_colab,
)
from src.core.config_manager import get_config  # noqa: E402

install_colab_requirements(ROOT / "colab_notebooks" / "requirements_colab.txt", running_in_colab())

CONFIG_FOR_ACCESS = get_config(environment="colab")
BACKBONE_MODEL_NAME = str(
    dict(dict(CONFIG_FOR_ACCESS.get("training", {})).get("continual", {}))
    .get("backbone", {})
    .get("model_name", "")
).strip()
ACCESS_REPORT = collect_notebook_access_report(
    repo_root=ROOT,
    hf_model_ids=[BACKBONE_MODEL_NAME] if BACKBONE_MODEL_NAME.strip() else [],
)
print_notebook_access_report(ACCESS_REPORT, print_fn=print)
if BACKBONE_MODEL_NAME:
    print(f"[KONTROL] Adapter backbone modeli: {BACKBONE_MODEL_NAME}")
if resolve_hf_token():
    login_and_check_hf_token(print_fn=print)
else:
    print("[HF] Token bulunamadi. Gated backbone icin Colab secret olarak HF_TOKEN ekleyin ve runtime'i yeniden baslatin.")
