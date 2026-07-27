#!/usr/bin/env python3
"""Shared repository bootstrap helpers for Colab notebooks."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence
from urllib.parse import urlsplit

from src.notebook.git_utils import (
    _chunked,
    _git_current_branch,
    _git_remote_url,
    _realign_local_branch_to_remote,
    _redact_possible_tokens,
    _run_capture,
    _run_git,
    _run_git_ls_remote_with_token,
    _run_git_push_with_token,
    _sanitize_run_text_artifacts,
    _should_exclude_run_file_from_push,
)
from src.notebook.git_utils import (
    _read_json_dict as _read_json_dict,
)
from src.notebook.git_utils import (
    _redact_secret as _redact_secret,
)
from src.notebook.git_utils import (
    mirror_checkpoint_state_to_repo as mirror_checkpoint_state_to_repo,
)
from src.notebook.git_utils import (
    mirror_path_to_repo as mirror_path_to_repo,
)

HF_TOKEN_NAMES = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")
GITHUB_TOKEN_NAMES = ("GH_TOKEN", "GITHUB_TOKEN")
GITHUB_READ_TOKEN_NAMES = ("AADS_GITHUB_RELEASE_READ_TOKEN",)
TORCH_REQUIREMENT_PREFIXES = ("torch", "torchvision", "torchaudio")


def is_repo_root(path: Path) -> bool:
    return (path / "src").is_dir() and (path / "config").is_dir() and (path / "scripts").is_dir()


def maybe_clone_repo() -> Optional[Path]:
    if os.environ.get("AADS_DISABLE_AUTO_CLONE") == "1":
        return None

    repo_url = os.environ.get("AADS_REPO_URL", "https://github.com/EfeErim/bitirmeprojesi.git")
    clone_target = Path(os.environ.get("AADS_REPO_CLONE_TARGET", "/content/bitirmeprojesi")).expanduser()

    if is_repo_root(clone_target):
        return clone_target

    if clone_target.exists() and any(clone_target.iterdir()):
        for child in clone_target.iterdir():
            if child.is_dir() and is_repo_root(child):
                return child
        return None

    clone_target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Repository not found locally. Auto-cloning from: {repo_url}")
    read_token = str(resolve_github_read_token() or "").strip()
    git_env = os.environ.copy()
    git_env["GIT_TERMINAL_PROMPT"] = "0"
    with tempfile.TemporaryDirectory(prefix="aads-git-read-") as temp_dir:
        if read_token:
            askpass = Path(temp_dir) / "askpass.sh"
            askpass.write_text(
                "#!/bin/sh\n"
                'case "$1" in\n'
                "*Username*) printf '%s\\n' 'x-access-token' ;;\n"
                "*) printf '%s\\n' \"$AADS_GIT_READ_TOKEN\" ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            askpass.chmod(0o700)
            git_env["GIT_ASKPASS"] = str(askpass)
            git_env["GIT_ASKPASS_REQUIRE"] = "force"
            git_env["AADS_GIT_READ_TOKEN"] = read_token
        completed = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(clone_target)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=git_env,
        )
    if completed.stdout:
        print(completed.stdout)

    if completed.returncode == 0 and is_repo_root(clone_target):
        return clone_target

    if completed.returncode != 0 and "github.com" in str(repo_url):
        print(
            "Auto-clone failed. If this repository is private, set the read-only "
            "AADS_GITHUB_RELEASE_READ_TOKEN as an env var or Colab secret, or point "
            "AADS_REPO_ROOT to an existing repo checkout."
        )
    return None


def install_colab_requirements(req_path: Path, in_colab: bool) -> None:
    """Install notebook requirements with Colab-safe torch pin handling."""
    req = Path(req_path)
    if not req.exists():
        return

    if not in_colab:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=False)
        return

    filtered = _flatten_colab_safe_requirements(req)
    tmp_req = Path(tempfile.gettempdir()) / "aads_colab_requirements_no_torch.txt"
    tmp_req.parent.mkdir(parents=True, exist_ok=True)
    tmp_req.write_text("\n".join(filtered) + "\n", encoding="utf-8")
    started_at = time.perf_counter()
    print("[SETUP] Installing demo dependencies. A fresh Colab runtime may take several minutes...", flush=True)
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(tmp_req)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        output = str(completed.stdout or "").strip()
        if output:
            print(output)
        raise RuntimeError(
            "Colab dependency installation failed for the filtered requirements set. "
            "See pip output above for details."
        )
    elapsed_seconds = time.perf_counter() - started_at
    print(f"[SETUP] Demo dependencies ready in {elapsed_seconds:.1f}s.", flush=True)


def _requirement_name(requirement_line: str) -> str:
    requirement = str(requirement_line or "").strip().split(";", 1)[0].strip()
    if not requirement:
        return ""
    match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
    return "" if match is None else str(match.group(1)).lower()


def _flatten_colab_safe_requirements(req_path: Path, _seen: Optional[set[Path]] = None) -> list[str]:
    resolved_path = Path(req_path).expanduser().resolve()
    seen = set() if _seen is None else _seen
    if resolved_path in seen:
        return []
    seen.add(resolved_path)

    filtered: list[str] = []
    for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        lowered = stripped.lower()
        if not stripped or stripped.startswith("#"):
            continue
        if lowered.startswith(("-r ", "--requirement ")):
            _, include_path = stripped.split(maxsplit=1)
            nested_path = (resolved_path.parent / include_path.strip()).resolve()
            filtered.extend(_flatten_colab_safe_requirements(nested_path, seen))
            continue
        if _requirement_name(stripped) in TORCH_REQUIREMENT_PREFIXES:
            continue
        filtered.append(stripped)
    return filtered


def resolve_repo_root() -> Path:
    env_candidates = [os.environ.get("AADS_REPO_ROOT"), os.environ.get("REPO_ROOT")]
    for raw in env_candidates:
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        if is_repo_root(candidate):
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if is_repo_root(candidate):
            return candidate

    common_candidates = [
        Path("/content/bitirme projesi"),
        Path("/content/bitirmeprojesi"),
        Path("/content/aads_ulora"),
        Path("/content/drive/MyDrive/bitirme projesi"),
        Path("/content/drive/MyDrive/bitirmeprojesi"),
    ]
    for candidate in common_candidates:
        if is_repo_root(candidate):
            return candidate

    auto_cloned = maybe_clone_repo()
    if auto_cloned is not None:
        return auto_cloned

    raise FileNotFoundError(
        "Repository root not found and auto-clone failed. "
        "Set AADS_REPO_ROOT, or set AADS_REPO_URL/AADS_REPO_CLONE_TARGET. "
        "Private GitHub repos also require GH_TOKEN or GITHUB_TOKEN for auto-clone."
    )


def _ensure_repo_root_for_update_check() -> Path:
    """Resolve the repo root and make it importable before notebook freshness checks."""
    repo_root = resolve_repo_root()
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    return repo_root


def running_in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except Exception:
        return False
    return True


def mount_drive_if_available(force_remount: bool = False) -> None:
    if not running_in_colab():
        return
    try:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=force_remount)
    except Exception as exc:
        print(f"Drive mount skipped: {exc}")


def export_current_colab_notebook(
    destination_path: str | Path,
    *,
    attempts: int = 3,
    retry_delay_sec: float = 1.0,
) -> Optional[Path]:
    """Write the current Colab notebook JSON, including cell outputs, to disk."""
    if not running_in_colab():
        return None

    try:
        from google.colab import _message
    except Exception:
        return None

    payload = None
    max_attempts = max(1, int(attempts))
    delay = max(0.0, float(retry_delay_sec))
    for attempt_index in range(max_attempts):
        try:
            response = _message.blocking_request("get_ipynb", timeout_sec=30)
        except Exception:
            response = None
        candidate = response.get("ipynb") if isinstance(response, dict) else None
        if isinstance(candidate, dict) and candidate:
            payload = candidate
            break
        # Colab occasionally returns an empty payload near runtime teardown.
        # Retry a few times before treating this as a soft failure so finalization can continue.
        if attempt_index + 1 < max_attempts and delay > 0.0:
            time.sleep(delay)

    if not isinstance(payload, dict) or not payload:
        return None

    destination = Path(destination_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def resolve_github_token() -> Optional[str]:
    """Resolve a GitHub token from env vars first, then Colab secrets."""
    for env_name in GITHUB_TOKEN_NAMES:
        token = str(os.environ.get(env_name, "")).strip()
        if token:
            os.environ.setdefault("GH_TOKEN", token)
            return token

    if not running_in_colab():
        return None

    for secret_name in GITHUB_TOKEN_NAMES:
        token = _resolve_colab_secret(secret_name)
        if token:
            os.environ["GH_TOKEN"] = token
            return token

    return None


def resolve_github_read_token() -> Optional[str]:
    """Resolve the private-repository read credential without falling back to a publisher token."""
    for env_name in GITHUB_READ_TOKEN_NAMES:
        token = str(os.environ.get(env_name, "")).strip()
        if token:
            return token

    if not running_in_colab():
        return None

    for secret_name in GITHUB_READ_TOKEN_NAMES:
        token = _resolve_colab_secret(secret_name)
        if token:
            return token

    return None


def push_repo_run_to_github(
    repo_root: str | Path,
    run_id: str,
    *,
    run_relative_dir: Optional[str | Path] = None,
    remote_name: str = "origin",
    branch: Optional[str] = None,
    commit_message: Optional[str] = None,
    token: Optional[str] = None,
    print_fn: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    """Commit and push one mirrored run tree, excluding checkpoint blobs and oversized files."""
    emit = print if print_fn is None else print_fn
    repo = Path(repo_root).expanduser().resolve()
    if run_relative_dir is None:
        run_dir = repo / "runs" / str(run_id)
    else:
        raw_run_relative_dir = str(run_relative_dir).strip().replace("\\", "/")
        if not raw_run_relative_dir or raw_run_relative_dir.startswith("/") or ".." in Path(raw_run_relative_dir).parts:
            raise ValueError(f"Run directory must be a repo-relative path: {run_relative_dir}")
        run_dir = repo / raw_run_relative_dir
    if not is_repo_root(repo):
        raise FileNotFoundError(f"Repository root not found: {repo}")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run export directory not found: {run_dir}")

    resolved_token = str(token or resolve_github_token() or "").strip()
    if not resolved_token:
        raise RuntimeError("GitHub auto-push requires GH_TOKEN or GITHUB_TOKEN in env vars or Colab secrets.")

    resolved_branch = str(branch or os.environ.get("AADS_REPO_PUSH_BRANCH") or _git_current_branch(repo) or "").strip()
    if not resolved_branch:
        raise RuntimeError("Could not determine the target git branch for auto-push.")

    remote_url = _git_remote_url(repo, remote_name)
    relative_run_dir = run_dir.relative_to(repo).as_posix()

    realigned = _realign_local_branch_to_remote(repo, remote_name=remote_name, branch=resolved_branch)
    if realigned:
        emit(
            f"[GIT] Local branch realigned to {remote_name}/{resolved_branch} before secure run push."
        )

    sanitize_report = _sanitize_run_text_artifacts(run_dir, explicit_secrets=[resolved_token, str(resolve_hf_token() or "")])
    emit(
        f"[SECURITY] scanned={sanitize_report['scanned']} redacted_files={sanitize_report['redacted_files']}"
    )
    leaks = list(sanitize_report.get("leaks", []))
    if leaks:
        preview = leaks[:10]
        raise RuntimeError(
            "Push blocked: secret-like patterns are still present after sanitization in "
            f"{len(leaks)} file(s). First matches: {preview}"
        )

    tracked_files: list[str] = []
    skipped_files: list[str] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_to_repo = path.relative_to(repo).as_posix()
        excluded, _reason = _should_exclude_run_file_from_push(path, run_dir=run_dir)
        if excluded:
            skipped_files.append(relative_to_repo)
            continue
        tracked_files.append(relative_to_repo)

    _run_git(["config", "user.name", os.environ.get("AADS_GIT_USER_NAME", "AADS Colab")], cwd=repo)
    _run_git(["config", "user.email", os.environ.get("AADS_GIT_USER_EMAIL", "aads-colab@local")], cwd=repo)

    if tracked_files:
        for chunk in _chunked(tracked_files):
            _run_git(["add", "--sparse", "-f", "--", *chunk], cwd=repo)
    if skipped_files:
        for chunk in _chunked(skipped_files):
            _run_git(["rm", "--cached", "-r", "--ignore-unmatch", "--", *chunk], cwd=repo, check=False)

    staged = _run_git(["diff", "--cached", "--name-only", "--", relative_run_dir], cwd=repo, capture_output=True)
    staged_files = [line.strip() for line in str(staged.stdout or "").splitlines() if line.strip()]
    emit(f"[GIT] Stage prepared for {relative_run_dir}: staged={len(staged_files)} skipped={len(skipped_files)}")
    if not staged_files:
        emit(f"[GIT] No eligible repo mirror changes to push for {relative_run_dir}.")
        return {
            "enabled": True,
            "pushed": False,
            "branch": resolved_branch,
            "remote_name": remote_name,
            "run_dir": str(run_dir),
            "staged_files": [],
            "skipped_files": skipped_files,
        }

    message = str(commit_message or f"Add notebook 2 outputs for run {run_id}")
    _run_git(["commit", "-m", message, "--", relative_run_dir], cwd=repo)
    _run_git_push_with_token(repo=repo, remote_url=remote_url, token=resolved_token, branch=resolved_branch)
    emit(f"[GIT] Pushed {len(staged_files)} file(s) from {relative_run_dir} to {remote_name}/{resolved_branch}.")
    return {
        "enabled": True,
        "pushed": True,
        "branch": resolved_branch,
        "remote_name": remote_name,
        "run_dir": str(run_dir),
        "staged_files": staged_files,
        "skipped_files": skipped_files,
    }


def push_repo_paths_to_github(
    repo_root: str | Path,
    relative_paths: Sequence[str | Path],
    *,
    remote_name: str = "origin",
    branch: Optional[str] = None,
    commit_message: Optional[str] = None,
    token: Optional[str] = None,
    print_fn: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    """Force-add selected repo-relative paths, commit them, and push to GitHub."""
    emit = print if print_fn is None else print_fn
    repo = Path(repo_root).expanduser().resolve()
    if not is_repo_root(repo):
        raise FileNotFoundError(f"Repository root not found: {repo}")

    normalized_paths: List[str] = []
    for raw_path in relative_paths:
        raw_text = str(raw_path).strip().replace("\\", "/")
        if raw_text.startswith("/") or ":" in raw_text.split("/", 1)[0]:
            raise ValueError(f"Path must be repo-relative and stay inside the repo: {raw_path}")
        relative = Path(raw_text).as_posix().strip().strip("/")
        if not relative or relative.startswith("../") or "/../" in relative:
            raise ValueError(f"Path must be repo-relative and stay inside the repo: {raw_path}")
        normalized_paths.append(relative)
    if not normalized_paths:
        raise ValueError("At least one repo-relative path is required.")

    resolved_token = str(token or resolve_github_token() or "").strip()
    if not resolved_token:
        raise RuntimeError("GitHub auto-push requires GH_TOKEN or GITHUB_TOKEN in env vars or Colab secrets.")

    resolved_branch = str(branch or os.environ.get("AADS_REPO_PUSH_BRANCH") or _git_current_branch(repo) or "").strip()
    if not resolved_branch:
        raise RuntimeError("Could not determine the target git branch for auto-push.")

    remote_url = _git_remote_url(repo, remote_name)

    realigned = _realign_local_branch_to_remote(repo, remote_name=remote_name, branch=resolved_branch)
    if realigned:
        emit(
            f"[GIT] Local branch realigned to {remote_name}/{resolved_branch} before secure path push."
        )

    _run_git(["config", "user.name", os.environ.get("AADS_GIT_USER_NAME", "AADS Colab")], cwd=repo)
    _run_git(["config", "user.email", os.environ.get("AADS_GIT_USER_EMAIL", "aads-colab@local")], cwd=repo)
    _run_git(["add", "-A", "-f", "--", *normalized_paths], cwd=repo)

    staged = _run_git(["diff", "--cached", "--name-only", "--", *normalized_paths], cwd=repo, capture_output=True)
    staged_files = [line.strip() for line in str(staged.stdout or "").splitlines() if line.strip()]
    if not staged_files:
        emit(f"[GIT] No eligible changes to push for: {', '.join(normalized_paths)}.")
        return {
            "enabled": True,
            "pushed": False,
            "branch": resolved_branch,
            "remote_name": remote_name,
            "paths": normalized_paths,
            "staged_files": [],
        }

    message = str(commit_message or f"Add generated repo assets: {', '.join(normalized_paths)}")
    _run_git(["commit", "-m", message, "--", *normalized_paths], cwd=repo)
    _run_git_push_with_token(repo=repo, remote_url=remote_url, token=resolved_token, branch=resolved_branch)
    emit(
        f"[GIT] Pushed {len(staged_files)} file(s) from "
        f"{', '.join(normalized_paths)} to {remote_name}/{resolved_branch}."
    )
    return {
        "enabled": True,
        "pushed": True,
        "branch": resolved_branch,
        "remote_name": remote_name,
        "paths": normalized_paths,
        "staged_files": staged_files,
    }


def resolve_hf_token() -> Optional[str]:
    """Resolve a Hugging Face token from env vars first, then Colab secrets."""
    for env_name in HF_TOKEN_NAMES:
        token = str(os.environ.get(env_name, "")).strip()
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            return token

    if not running_in_colab():
        return None

    for secret_name in HF_TOKEN_NAMES:
        token = _resolve_colab_secret(secret_name)
        if token:
            os.environ["HF_TOKEN"] = token
            return token

    return None


def _resolve_colab_secret(secret_name: str) -> str:
    if not running_in_colab():
        return ""

    try:
        from google.colab import userdata
    except Exception:
        return ""

    try:
        return str(userdata.get(secret_name) or "").strip()
    except Exception:
        return ""


def probe_repo_update_status(
    repo_root: str | Path,
    *,
    remote_name: str = "origin",
    branch: Optional[str] = None,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    if not is_repo_root(repo):
        return {"status": "unavailable", "message": f"Repository root not found: {repo}"}

    resolved_branch = str(branch or _git_current_branch(repo) or "").strip()
    if not resolved_branch:
        return {"status": "unavailable", "message": "Current git branch could not be determined."}

    local_head_completed = _run_git(["rev-parse", "HEAD"], cwd=repo, check=False, capture_output=True)
    local_head = str(local_head_completed.stdout or "").strip()
    if local_head_completed.returncode != 0 or not local_head:
        return {"status": "unavailable", "branch": resolved_branch, "message": "Local HEAD could not be resolved."}

    remote_completed = _run_capture(["git", "ls-remote", remote_name, f"refs/heads/{resolved_branch}"], cwd=repo)
    remote_stdout = str(remote_completed.stdout or "").strip()
    if remote_completed.returncode != 0 or not remote_stdout:
        return {
            "status": "unavailable",
            "branch": resolved_branch,
            "local_head": local_head,
            "message": "Remote branch information could not be read.",
            "detail": remote_stdout,
        }

    remote_head = remote_stdout.split()[0].strip()
    update_available = bool(remote_head and remote_head != local_head)
    return {
        "status": "ok",
        "branch": resolved_branch,
        "local_head": local_head,
        "remote_head": remote_head,
        "update_available": update_available,
        "relation": "update_available" if update_available else "up_to_date",
    }


def probe_github_repo_access(
    *,
    repo_url: Optional[str] = None,
    repo_root: Optional[str | Path] = None,
    token: Optional[str] = None,
) -> dict[str, Any]:
    resolved_repo_url = str(repo_url or "").strip()
    if not resolved_repo_url and repo_root is not None:
        repo = Path(repo_root).expanduser().resolve()
        if is_repo_root(repo):
            try:
                resolved_repo_url = _git_remote_url(repo, "origin")
            except Exception:
                resolved_repo_url = ""
    if not resolved_repo_url:
        resolved_repo_url = str(os.environ.get("AADS_REPO_URL", "")).strip()

    if not resolved_repo_url:
        return {"status": "unavailable", "message": "Repository URL could not be determined."}

    resolved_token = str(token or resolve_github_token() or "").strip()
    anonymous_probe = _run_capture(["git", "ls-remote", resolved_repo_url, "HEAD"])
    anonymous_ok = anonymous_probe.returncode == 0 and bool(str(anonymous_probe.stdout or "").strip())

    token_ok = anonymous_ok
    token_detail = _redact_possible_tokens(str(anonymous_probe.stdout or "").strip(), resolved_token)
    if not anonymous_ok and resolved_token:
        token_probe = _run_git_ls_remote_with_token(remote_url=resolved_repo_url, token=resolved_token, ref="HEAD")
        token_ok = token_probe.returncode == 0 and bool(str(token_probe.stdout or "").strip())
        token_detail = _redact_possible_tokens(str(token_probe.stdout or "").strip(), resolved_token)

    if anonymous_ok:
        read_access_mode = "public"
    elif resolved_token and token_ok:
        read_access_mode = "token_required"
    else:
        read_access_mode = "unavailable"

    parsed = urlsplit(resolved_repo_url)
    has_embedded_auth = "@" in str(parsed.netloc or "")
    push_ready = bool(resolved_token or has_embedded_auth or parsed.scheme == "ssh")
    return {
        "status": "ok" if read_access_mode != "unavailable" else "unavailable",
        "repo_url": resolved_repo_url,
        "token_present": bool(resolved_token),
        "read_access_mode": read_access_mode,
        "anonymous_read_access": bool(anonymous_ok),
        "token_read_access": bool(token_ok),
        "push_requires_auth": True,
        "push_ready": bool(push_ready),
        "detail": token_detail if token_detail else _redact_possible_tokens(str(anonymous_probe.stdout or "").strip(), resolved_token),
    }


def probe_hf_model_access(
    model_ids: Sequence[str],
    *,
    token: Optional[str] = None,
) -> dict[str, Any]:
    resolved_model_ids = [str(model_id).strip() for model_id in list(model_ids or []) if str(model_id).strip()]
    if not resolved_model_ids:
        return {"status": "skipped", "model_ids": [], "access_mode": "not_checked"}

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return {
            "status": "unavailable",
            "model_ids": resolved_model_ids,
            "access_mode": "unavailable",
            "message": f"huggingface_hub import failed: {exc}",
        }

    resolved_token = str(token or resolve_hf_token() or "").strip()
    api_anon = HfApi()
    api_auth = HfApi(token=resolved_token) if resolved_token else None
    per_model: list[dict[str, Any]] = []
    for model_id in resolved_model_ids:
        anonymous_ok = False
        token_ok = False
        anonymous_detail = ""
        token_detail = ""
        try:
            api_anon.model_info(model_id)
            anonymous_ok = True
        except Exception as exc:
            anonymous_detail = f"{exc.__class__.__name__}: {exc}"
        if anonymous_ok:
            token_ok = True
        elif api_auth is not None:
            try:
                api_auth.model_info(model_id)
                token_ok = True
            except Exception as exc:
                token_detail = f"{exc.__class__.__name__}: {exc}"
        access_mode = "public" if anonymous_ok else "token_required" if token_ok else "unavailable"
        per_model.append(
            {
                "model_id": model_id,
                "access_mode": access_mode,
                "anonymous_ok": anonymous_ok,
                "token_ok": token_ok,
                "detail": token_detail or anonymous_detail,
            }
        )

    overall_mode = "public"
    if any(item["access_mode"] == "unavailable" for item in per_model):
        overall_mode = "unavailable"
    elif any(item["access_mode"] == "token_required" for item in per_model):
        overall_mode = "token_required"

    return {
        "status": "ok" if overall_mode != "unavailable" else "unavailable",
        "model_ids": resolved_model_ids,
        "token_present": bool(resolved_token),
        "access_mode": overall_mode,
        "requires_token_for_any": any(item["access_mode"] == "token_required" for item in per_model),
        "per_model": per_model,
    }


def collect_notebook_access_report(
    *,
    repo_root: Optional[str | Path] = None,
    repo_url: Optional[str] = None,
    hf_model_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    resolved_repo_root = Path(repo_root).expanduser().resolve() if repo_root is not None else None
    github = probe_github_repo_access(repo_url=repo_url, repo_root=resolved_repo_root)
    updates = (
        probe_repo_update_status(resolved_repo_root)
        if resolved_repo_root is not None and is_repo_root(resolved_repo_root)
        else {"status": "unavailable", "message": "Repository root is not available yet."}
    )
    huggingface = probe_hf_model_access(list(hf_model_ids or []))
    return {
        "github": github,
        "repo_updates": updates,
        "huggingface": huggingface,
    }


def print_notebook_access_report(
    report: dict[str, Any],
    *,
    print_fn: Optional[Callable[[str], None]] = None,
) -> None:
    emit = print if print_fn is None else print_fn
    github = dict(report.get("github", {}))
    updates = dict(report.get("repo_updates", {}))
    huggingface = dict(report.get("huggingface", {}))

    relation = str(updates.get("relation", "unknown"))
    if relation == "up_to_date":
        emit("[CHECK] Repository appears up to date.")
    elif relation == "update_available":
        emit(f"[CHECK] Repository update available. Branch={updates.get('branch', '')}")
    else:
        emit(f"[CHECK] Repository update status could not be read: {updates.get('message', 'no details')}")

    read_access_mode = str(github.get("read_access_mode", "unavailable"))
    if read_access_mode == "public":
        emit("[CHECK] GitHub read access is public; clone and pull do not require an additional token.")
    elif read_access_mode == "token_required":
        emit("[CHECK] GitHub read access requires a token; set GH_TOKEN for a private repository.")
    else:
        emit("[CHECK] GitHub read access could not be verified.")

    if bool(github.get("push_ready")):
        emit("[CHECK] GitHub push credentials are available.")
    else:
        emit("[CHECK] GitHub push requires additional authentication.")

    hf_mode = str(huggingface.get("access_mode", "not_checked"))
    if hf_mode == "public":
        emit("[CHECK] Required Hugging Face models are available with anonymous access.")
    elif hf_mode == "token_required":
        emit("[CHECK] At least one Hugging Face model requires a token; use a Colab secret.")
    elif hf_mode == "not_checked":
        emit("[CHECK] Hugging Face model access was not checked separately for this notebook.")
    else:
        emit("[CHECK] Hugging Face model access could not be verified.")


def login_and_check_hf_token(*, print_fn: Optional[Callable[[str], None]] = None) -> bool:
    """Authenticate once and validate the token with a lightweight identity lookup."""
    emit = print if print_fn is None else print_fn
    token = resolve_hf_token()
    if not token:
        emit(
            "[HF] Token not found. Before inference or training, define HF_TOKEN "
            "as a Colab secret or environment variable."
        )
        return False

    try:
        from huggingface_hub import HfApi, login
    except Exception as exc:
        emit(f"[HF] huggingface_hub could not be imported: {exc}")
        return False

    try:
        login(token=token, add_to_git_credential=False)
        profile = dict(HfApi(token=token).whoami() or {})
        username = str(
            profile.get("name")
            or profile.get("fullname")
            or profile.get("email")
            or profile.get("user")
            or "authenticated user"
        )
        emit(f"[HF] Identity verified: {username}")
        return True
    except Exception as exc:
        emit(f"[HF] Identity verification failed: {exc}")
        return False
