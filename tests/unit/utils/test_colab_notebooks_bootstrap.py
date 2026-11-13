import subprocess
from pathlib import Path

import pytest

from src.colab import notebooks_bootstrap as nb_bootstrap


def _make_repo(root: Path) -> Path:
    for name in ("src", "config", "scripts"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def _completed(returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode)


def test_ensure_repo_root_returns_existing_checkout_without_cloning(tmp_path: Path, monkeypatch):
    repo = _make_repo(tmp_path / "repo")
    monkeypatch.setattr(nb_bootstrap, "find_repo_root", lambda: repo)
    monkeypatch.setattr(nb_bootstrap.subprocess, "run", lambda *a, **k: pytest.fail("must not run git"))

    assert nb_bootstrap.ensure_repo_root() == repo


def test_ensure_repo_root_adds_missing_sparse_paths_to_existing_clone(tmp_path: Path, monkeypatch):
    repo = _make_repo(tmp_path / "repo")
    (repo / ".git").mkdir()
    calls = []
    monkeypatch.setattr(nb_bootstrap, "find_repo_root", lambda: repo)
    monkeypatch.setattr(nb_bootstrap.subprocess, "run", lambda args, **_k: calls.append(args) or _completed())

    nb_bootstrap.ensure_repo_root()

    assert calls == [["git", "sparse-checkout", "add", "requirements_colab.txt"]]


def test_ensure_repo_root_refuses_to_clone_when_auto_clone_is_off(monkeypatch):
    monkeypatch.setattr(nb_bootstrap, "find_repo_root", lambda: None)

    with pytest.raises(RuntimeError, match="Repo not found"):
        nb_bootstrap.ensure_repo_root(auto_clone=False)


def test_ensure_repo_root_sparse_clones_source_paths(tmp_path: Path, monkeypatch):
    clone_target = tmp_path / "clone"
    found = iter([None, _make_repo(tmp_path / "cloned")])
    calls = []
    monkeypatch.setattr(nb_bootstrap, "find_repo_root", lambda: next(found))
    monkeypatch.setattr(nb_bootstrap.subprocess, "run", lambda args, **_k: calls.append(args) or _completed())

    root = nb_bootstrap.ensure_repo_root(
        clone_target=clone_target, repo_url="https://github.com/example/repo.git", github_token=""
    )

    assert root == tmp_path / "cloned"
    assert calls[0][:2] == ["git", "clone"]
    assert calls[0][-2:] == ["https://github.com/example/repo.git", str(clone_target)]
    assert calls[1] == ["git", "sparse-checkout", "set", *nb_bootstrap.SOURCE_CHECKOUT_PATHS]


def test_ensure_repo_root_raises_when_clone_fails(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(nb_bootstrap, "find_repo_root", lambda: None)
    monkeypatch.setattr(nb_bootstrap.subprocess, "run", lambda *a, **k: _completed(128))

    with pytest.raises(RuntimeError, match="Repo clone failed with code 128"):
        nb_bootstrap.ensure_repo_root(clone_target=tmp_path / "missing", github_token="")


def test_build_repo_access_url_only_adds_tokens_to_https_urls():
    assert nb_bootstrap._build_repo_access_url("https://github.com/a/b.git", "tok") == "https://tok@github.com/a/b.git"
    assert nb_bootstrap._build_repo_access_url("git@github.com:a/b.git", "tok") == "git@github.com:a/b.git"
    assert nb_bootstrap._build_repo_access_url("https://github.com/a/b.git", None) == "https://github.com/a/b.git"


def test_bootstrap_notebook_puts_repo_on_sys_path_and_skips_pip_outside_colab(tmp_path: Path, monkeypatch, capsys):
    repo = _make_repo(tmp_path / "repo")
    chdirs = []
    monkeypatch.setattr(nb_bootstrap, "resolve_github_token", lambda: "")
    monkeypatch.setattr(nb_bootstrap, "resolve_huggingface_token", lambda: "hf")
    monkeypatch.setattr(nb_bootstrap, "running_in_colab", lambda: False)
    monkeypatch.setattr(nb_bootstrap, "ensure_repo_root", lambda **_k: repo)
    monkeypatch.setattr(nb_bootstrap.os, "chdir", chdirs.append)
    monkeypatch.setattr(nb_bootstrap.sys, "path", [])
    monkeypatch.setattr(nb_bootstrap.subprocess, "run", lambda *a, **k: pytest.fail("pip must not run"))

    result = nb_bootstrap.bootstrap_notebook("Notebook 2")

    assert result["bootstrap_status"] == "ok"
    assert result["ROOT"] == repo
    assert result["HF_TOKEN"] == "hf"
    assert chdirs == [repo]
    assert nb_bootstrap.sys.path == [str(repo)]
    nb_bootstrap.print_bootstrap_status(result)
    assert "Status: ok" in capsys.readouterr().out


def test_bootstrap_notebook_records_and_reraises_failures(monkeypatch):
    monkeypatch.setattr(nb_bootstrap, "resolve_github_token", lambda: "")
    monkeypatch.setattr(nb_bootstrap, "resolve_huggingface_token", lambda: None)
    monkeypatch.setattr(nb_bootstrap, "running_in_colab", lambda: False)

    def _fail(**_kwargs):
        raise RuntimeError("no repo")

    monkeypatch.setattr(nb_bootstrap, "ensure_repo_root", _fail)

    with pytest.raises(RuntimeError, match="no repo"):
        nb_bootstrap.bootstrap_notebook()
