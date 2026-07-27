"""Git, mirroring, and redaction helpers for Colab notebook bootstrap code."""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

SENSITIVE_TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml", ".csv", ".ini"}
MAX_PUSH_FILE_SIZE_BYTES = 95 * 1024 * 1024
EXCLUDED_PUSH_SUFFIXES = {".pt"}
EXCLUDED_PUSH_PATH_PARTS = ("checkpoint_state/checkpoints",)


def _read_json_dict(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=check,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
        text=True,
    )


def _chunked(items: list[str], size: int = 200) -> list[list[str]]:
    if size <= 0:
        return [items]
    return [items[index : index + size] for index in range(0, len(items), size)]


def _git_current_branch(repo_root: Path) -> str:
    completed = _run_git(["branch", "--show-current"], cwd=repo_root, capture_output=True)
    return str(completed.stdout or "").strip()


def _git_remote_url(repo_root: Path, remote_name: str) -> str:
    completed = _run_git(["remote", "get-url", remote_name], cwd=repo_root, capture_output=True)
    return str(completed.stdout or "").strip()


def _build_authenticated_remote_url(repo_url: str, token: str) -> str:
    parsed = urlsplit(str(repo_url or "").strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError(
            "GitHub auto-push currently supports only HTTPS remotes. "
            "Set origin to an https:// URL or disable auto-push."
        )
    netloc = parsed.netloc.split("@", 1)[-1]
    return urlunsplit((parsed.scheme, f"x-access-token:{token}@{netloc}", parsed.path, parsed.query, parsed.fragment))


def _clean_https_remote_url(repo_url: str) -> str:
    parsed = urlsplit(str(repo_url or "").strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError(
            "GitHub auto-push currently supports only HTTPS remotes. "
            "Set origin to an https:// URL or disable auto-push."
        )
    netloc = parsed.netloc.split("@", 1)[-1]
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _redact_secret(value: str, secret: str) -> str:
    redacted = str(value or "")
    if secret:
        redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _redact_possible_tokens(value: str, secret: Optional[str] = None) -> str:
    redacted = _redact_secret(str(value or ""), str(secret or ""))
    redacted = re.sub(r"\bgh[pousr]_[A-Za-z0-9_]{10,}\b", "<redacted>", redacted)
    redacted = re.sub(r"\bgithub_pat_[A-Za-z0-9_]{10,}\b", "<redacted>", redacted)
    redacted = re.sub(
        r"AUTHORIZATION\s*:\s*basic\s+[A-Za-z0-9+/=]+",
        "AUTHORIZATION: <redacted>",
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"AUTHORIZATION\s*:\s*bearer\s+[A-Za-z0-9_\-.=+/]+",
        "AUTHORIZATION: <redacted>",
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(r"x-access-token:[^\s@/]+", "x-access-token:<redacted>", redacted, flags=re.IGNORECASE)
    redacted = re.sub(
        r"https://[^\s:@]+:[^\s@]+@github\.com",
        "https://<redacted>@github.com",
        redacted,
        flags=re.IGNORECASE,
    )
    return redacted


def _redact_possible_tokens_with_secrets(value: str, secrets: Sequence[str]) -> str:
    redacted = str(value or "")
    for secret in [str(item).strip() for item in list(secrets or []) if str(item).strip()]:
        redacted = _redact_possible_tokens(redacted, secret)
    return _redact_possible_tokens(redacted, None)


def _contains_possible_tokens(value: str) -> bool:
    text = str(value or "")
    patterns = (
        r"\bgh[pousr]_[A-Za-z0-9_]{10,}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{10,}\b",
        r"AUTHORIZATION\s*:\s*(?:basic|bearer)\s+[A-Za-z0-9_\-.=+/]+",
        r"x-access-token:[^\s@/]+",
        r"https://[^\s:@]+:[^\s@]+@github\.com",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _redact_nested_obj(value: Any, explicit_secrets: Sequence[str]) -> Any:
    if isinstance(value, dict):
        return {key: _redact_nested_obj(item, explicit_secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_nested_obj(item, explicit_secrets) for item in value]
    if isinstance(value, str):
        return _redact_possible_tokens_with_secrets(value, explicit_secrets)
    return value


def _sanitize_run_text_artifacts(run_dir: Path, *, explicit_secrets: Sequence[str]) -> dict[str, Any]:
    scanned = 0
    redacted_files = 0
    leaks: list[str] = []

    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SENSITIVE_TEXT_SUFFIXES:
            continue

        scanned += 1
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        redacted_text = raw_text
        suffix = path.suffix.lower()

        if suffix == ".json":
            try:
                payload = json.loads(raw_text)
                redacted_payload = _redact_nested_obj(payload, explicit_secrets)
                redacted_text = json.dumps(redacted_payload, ensure_ascii=False, indent=2) + "\n"
            except Exception:
                redacted_text = _redact_possible_tokens_with_secrets(raw_text, explicit_secrets)
        elif suffix == ".jsonl":
            redacted_lines: list[str] = []
            for raw_line in raw_text.splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    redacted_lines.append(raw_line)
                    continue
                try:
                    parsed_line = json.loads(stripped)
                    redacted_line_payload = _redact_nested_obj(parsed_line, explicit_secrets)
                    redacted_lines.append(json.dumps(redacted_line_payload, ensure_ascii=False))
                except Exception:
                    redacted_lines.append(_redact_possible_tokens_with_secrets(raw_line, explicit_secrets))
            redacted_text = "\n".join(redacted_lines)
            if raw_text.endswith("\n"):
                redacted_text += "\n"
        else:
            redacted_text = _redact_possible_tokens_with_secrets(raw_text, explicit_secrets)

        if redacted_text != raw_text:
            path.write_text(redacted_text, encoding="utf-8")
            redacted_files += 1

        if _contains_possible_tokens(redacted_text):
            leaks.append(path.as_posix())

    return {
        "scanned": scanned,
        "redacted_files": redacted_files,
        "leaks": leaks,
    }


def _should_exclude_run_file_from_push(path: Path, *, run_dir: Path) -> tuple[bool, str]:
    relative_to_run = path.relative_to(run_dir).as_posix().lower()
    if path.suffix.lower() in EXCLUDED_PUSH_SUFFIXES:
        return True, "excluded_suffix"
    if any(part in relative_to_run for part in EXCLUDED_PUSH_PATH_PARTS):
        return True, "excluded_checkpoint_path"
    try:
        if path.stat().st_size > MAX_PUSH_FILE_SIZE_BYTES:
            return True, "excluded_large_file"
    except OSError:
        return True, "excluded_unreadable"
    return False, "tracked"


def _run_capture(
    args: list[str],
    *,
    cwd: Optional[Path] = None,
    timeout_sec: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=None if cwd is None else str(cwd),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=max(1.0, float(timeout_sec)),
    )


def _realign_local_branch_to_remote(repo: Path, *, remote_name: str, branch: str) -> bool:
    remote_probe = _run_capture(["git", "ls-remote", "--heads", remote_name, f"refs/heads/{branch}"], cwd=repo)
    if remote_probe.returncode != 0 or not str(remote_probe.stdout or "").strip():
        return False

    fetch_completed = _run_git(["fetch", remote_name, branch], cwd=repo, check=False, capture_output=True)
    if fetch_completed.returncode != 0:
        output = str(fetch_completed.stdout or "").strip() or "fetch failed"
        raise RuntimeError(f"Git fetch failed before secure push: {output}")

    soft_reset = _run_git(["reset", "--soft", f"{remote_name}/{branch}"], cwd=repo, check=False, capture_output=True)
    if soft_reset.returncode != 0:
        output = str(soft_reset.stdout or "").strip() or "soft reset failed"
        raise RuntimeError(f"Git reset --soft failed before secure push: {output}")

    unstage_reset = _run_git(["reset"], cwd=repo, check=False, capture_output=True)
    if unstage_reset.returncode != 0:
        output = str(unstage_reset.stdout or "").strip() or "unstage reset failed"
        raise RuntimeError(f"Git reset failed before secure push: {output}")

    return True


def _run_git_ls_remote_with_token(*, remote_url: str, token: str, ref: str = "HEAD") -> subprocess.CompletedProcess[str]:
    clean_remote_url = _clean_https_remote_url(remote_url)
    auth_value = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
    env = os.environ.copy()
    try:
        config_count = int(str(env.get("GIT_CONFIG_COUNT", "0") or "0"))
    except ValueError:
        config_count = 0
    env["GIT_CONFIG_COUNT"] = str(config_count + 1)
    env[f"GIT_CONFIG_KEY_{config_count}"] = "http.https://github.com/.extraheader"
    env[f"GIT_CONFIG_VALUE_{config_count}"] = f"AUTHORIZATION: basic {auth_value}"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    return subprocess.run(
        ["git", "ls-remote", clean_remote_url, ref],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )


def _run_git_push_with_token(*, repo: Path, remote_url: str, token: str, branch: str) -> None:
    clean_remote_url = _clean_https_remote_url(remote_url)
    auth_value = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
    env = os.environ.copy()
    try:
        config_count = int(str(env.get("GIT_CONFIG_COUNT", "0") or "0"))
    except ValueError:
        config_count = 0
    env["GIT_CONFIG_COUNT"] = str(config_count + 1)
    env[f"GIT_CONFIG_KEY_{config_count}"] = "http.https://github.com/.extraheader"
    env[f"GIT_CONFIG_VALUE_{config_count}"] = f"AUTHORIZATION: basic {auth_value}"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    completed = subprocess.run(
        ["git", "push", clean_remote_url, f"HEAD:{branch}"],
        cwd=str(repo),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        output = _redact_secret(str(completed.stdout or ""), token)
        output_lower = output.lower()
        permission_hint = ""
        if (
            "permission to " in output_lower
            or "denied to" in output_lower
            or "http 403" in output_lower
            or "returned error: 403" in output_lower
        ):
            permission_hint = (
                " The token appears to lack write access to this repository. "
                "Use a GitHub token from an account or PAT that can push to the target repo, "
                "or set AUTO_PUSH_TO_GITHUB=False to keep the run local."
            )
        raise RuntimeError(
            f"GitHub push failed with exit code {completed.returncode}. "
            f"Output:\n{output.strip() or '<no output>'}{permission_hint}"
        )


def _build_repo_access_url(repo_url: str, token: Optional[str]) -> str:
    cleaned_url = str(repo_url or "").strip()
    if not cleaned_url or not token:
        return cleaned_url
    try:
        return _build_authenticated_remote_url(cleaned_url, token)
    except RuntimeError:
        return cleaned_url


def mirror_path_to_repo(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    exclude_dir_names: tuple[str, ...] = ("checkpoints",),
) -> Optional[Path]:
    """Copy a file or directory tree into the repo, optionally skipping directories by name."""
    source = Path(source_path).expanduser()
    if not source.exists():
        return None

    destination = Path(destination_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.is_file():
        shutil.copy2(source, destination)
        return destination

    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)

    excluded = set(str(name) for name in exclude_dir_names)

    def _ignore(current_dir: str, names: list[str]) -> set[str]:
        current = Path(current_dir)
        ignored: set[str] = set()
        for name in names:
            if name in excluded and (current / name).is_dir():
                ignored.add(name)
        return ignored

    shutil.copytree(source, destination, ignore=_ignore)
    return destination


def mirror_checkpoint_state_to_repo(
    source_root: str | Path,
    destination_root: str | Path,
) -> Optional[Path]:
    """Copy checkpoint metadata plus the mirrored best checkpoint only."""
    source = Path(source_root).expanduser()
    if not source.exists():
        return None

    destination = Path(destination_root).expanduser()
    mirrored_root = mirror_path_to_repo(source, destination, exclude_dir_names=("checkpoints",))
    if mirrored_root is None:
        return None

    source_checkpoints_dir = source / "checkpoints"
    if not source_checkpoints_dir.exists():
        return mirrored_root

    best_manifest = _read_json_dict(source / "best_checkpoint.json")
    best_name = str(best_manifest.get("name") or "").strip()
    source_best_path = source / "checkpoints" / "best"

    manifest_path = str(best_manifest.get("path") or "").strip()
    if manifest_path:
        candidate = Path(manifest_path).expanduser()
        if candidate.exists():
            source_best_path = candidate
        else:
            raise RuntimeError(f"Best checkpoint path from manifest was not found: {candidate}")
    elif best_name:
        named_candidate = source_checkpoints_dir / best_name
        if named_candidate.exists():
            source_best_path = named_candidate

    if not source_best_path.exists():
        if any(source_checkpoints_dir.iterdir()):
            raise RuntimeError(
                "Best checkpoint could not be resolved from checkpoint_state metadata. "
                "Check best_checkpoint.json and the checkpoint directory contents."
            )
        return mirrored_root

    destination_checkpoints_dir = destination / "checkpoints"
    destination_best_name = best_name or source_best_path.name or "best"
    destination_best_path = destination_checkpoints_dir / destination_best_name
    mirror_path_to_repo(source_best_path, destination_best_path, exclude_dir_names=())

    if best_manifest:
        best_manifest["path"] = str(destination_best_path)
        (destination / "best_checkpoint.json").write_text(
            json.dumps(best_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (destination / "latest_checkpoint.json").write_text(
            json.dumps(best_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (destination / "checkpoint_index.json").write_text(
            json.dumps([best_manifest], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return mirrored_root
