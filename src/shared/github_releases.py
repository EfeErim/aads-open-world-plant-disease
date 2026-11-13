"""Minimal GitHub Releases REST client and token lookup shared by the adapter and dataset releases."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

GITHUB_API_VERSION = "2026-03-10"
GITHUB_API_ROOT = "https://api.github.com"
READ_TOKEN_NAMES = ("AADS_GITHUB_RELEASE_READ_TOKEN", "AADS_GITHUB_RELEASE_CI_READ_TOKEN")
WRITE_TOKEN_NAMES = ("AADS_GITHUB_RELEASE_WRITE_TOKEN",)
REPOSITORY_ENV = "GITHUB_RELEASE_REPOSITORY"
COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def resolve_token(*, write: bool) -> str | None:
    for name in WRITE_TOKEN_NAMES if write else READ_TOKEN_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    return None


class GitHubReleaseClient:
    """Minimal secret-redacting GitHub REST client for release operations."""

    def __init__(self, token: str | None = None) -> None:
        import requests

        self._requests = requests
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        headers = dict(self._headers)
        headers.update(kwargs.pop("headers", {}))
        response = self._requests.request(method, url, headers=headers, timeout=120, **kwargs)
        if not response.ok:
            try:
                message = str(response.json().get("message") or "request failed")
            except (TypeError, ValueError):
                message = "request failed"
            raise RuntimeError(f"GitHub API {method} {url.split('?')[0]} failed ({response.status_code}): {message}")
        return response

    def get_repository(self, repository: str) -> dict[str, Any]:
        return dict(self._request("GET", f"{GITHUB_API_ROOT}/repos/{repository}").json())

    def immutable_releases_enabled(self, repository: str) -> bool:
        response = self._request("GET", f"{GITHUB_API_ROOT}/repos/{repository}/immutable-releases")
        return bool(response.json().get("enabled"))

    def create_draft(self, repository: str, tag: str, commit: str, *, name: str, body: str) -> dict[str, Any]:
        payload = {
            "tag_name": tag,
            "target_commitish": commit,
            "name": name,
            "body": body,
            "draft": True,
            "prerelease": False,
            "make_latest": "false",
        }
        return dict(self._request("POST", f"{GITHUB_API_ROOT}/repos/{repository}/releases", json=payload).json())

    def get_release(self, repository: str, release_id: int) -> dict[str, Any]:
        return dict(self._request("GET", f"{GITHUB_API_ROOT}/repos/{repository}/releases/{release_id}").json())

    def find_release_by_tag(self, repository: str, tag: str) -> dict[str, Any] | None:
        response = self._request("GET", f"{GITHUB_API_ROOT}/repos/{repository}/releases", params={"per_page": 100})
        for release in response.json():
            if str(release.get("tag_name") or "") == tag:
                return dict(release)
        return None

    def upload_asset(self, upload_url: str, source: Path, asset_name: str) -> dict[str, Any]:
        url = upload_url.split("{")[0]
        with source.open("rb") as handle:
            response = self._request(
                "POST",
                url,
                params={"name": asset_name},
                headers={"Content-Type": "application/octet-stream"},
                data=handle,
            )
        return dict(response.json())

    def download_asset(self, repository: str, asset_id: int, destination: Path) -> None:
        response = self._request(
            "GET",
            f"{GITHUB_API_ROOT}/repos/{repository}/releases/assets/{asset_id}",
            headers={"Accept": "application/octet-stream"},
            stream=True,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    def publish(self, repository: str, release_id: int) -> dict[str, Any]:
        return dict(
            self._request(
                "PATCH",
                f"{GITHUB_API_ROOT}/repos/{repository}/releases/{release_id}",
                json={"draft": False, "make_latest": "false"},
            ).json()
        )

    def tag_commit_sha(self, repository: str, tag: str) -> str:
        response = self._request("GET", f"{GITHUB_API_ROOT}/repos/{repository}/git/ref/tags/{quote(tag, safe='')}")
        commit = str((response.json().get("object") or {}).get("sha") or "")
        if not COMMIT_SHA_PATTERN.fullmatch(commit):
            raise RuntimeError("Published GitHub Release tag did not resolve to a commit SHA.")
        return commit
