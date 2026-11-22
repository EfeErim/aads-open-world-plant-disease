import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from src.colab import git_utils
from src.colab import repo_bootstrap as bootstrap


def test_resolve_hf_token_reads_colab_secret(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(bootstrap, "running_in_colab", lambda: True)

    fake_colab = ModuleType("google.colab")
    fake_colab.userdata = SimpleNamespace(get=lambda name: "secret-token" if name == "HF_TOKEN" else None)
    fake_google = ModuleType("google")
    fake_google.colab = fake_colab

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)

    assert bootstrap.resolve_hf_token() == "secret-token"
    assert os.environ["HF_TOKEN"] == "secret-token"


def test_resolve_github_read_token_never_falls_back_to_publisher_env(monkeypatch):
    monkeypatch.delenv("AADS_GITHUB_RELEASE_READ_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "publisher-token")
    monkeypatch.setattr(bootstrap, "running_in_colab", lambda: False)

    assert bootstrap.resolve_github_read_token() is None


def test_maybe_clone_repo_uses_askpass_without_persisting_read_token(tmp_path: Path, monkeypatch):
    clone_target = tmp_path / "bitirmeprojesi"
    monkeypatch.delenv("AADS_DISABLE_AUTO_CLONE", raising=False)
    monkeypatch.setenv("AADS_REPO_URL", "https://github.com/EfeErim/aads-open-world-plant-disease.git")
    monkeypatch.setenv("AADS_REPO_CLONE_TARGET", str(clone_target))
    monkeypatch.setattr(bootstrap, "resolve_github_read_token", lambda: "read-secret")

    calls: list[list[str]] = []
    observed_env: dict[str, str] = {}
    askpass_text = ""
    askpass_path: Path | None = None

    def fake_run(command, check=False, stdout=None, stderr=None, text=None, env=None):
        nonlocal askpass_path, askpass_text
        calls.append(list(command))
        observed_env.update(dict(env or {}))
        askpass_path = Path(observed_env["GIT_ASKPASS"])
        askpass_text = askpass_path.read_text(encoding="utf-8")
        clone_target.mkdir(parents=True, exist_ok=True)
        (clone_target / "src").mkdir(exist_ok=True)
        (clone_target / "config").mkdir(exist_ok=True)
        (clone_target / "scripts").mkdir(exist_ok=True)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    resolved = bootstrap.maybe_clone_repo()

    assert resolved == clone_target
    assert calls
    assert calls[0][:3] == ["git", "clone", "--depth"]
    assert calls[0][4] == "https://github.com/EfeErim/aads-open-world-plant-disease.git"
    assert "read-secret" not in " ".join(calls[0])
    assert "read-secret" not in askpass_text
    assert observed_env["AADS_GIT_READ_TOKEN"] == "read-secret"
    assert observed_env["GIT_TERMINAL_PROMPT"] == "0"
    assert askpass_path is not None and not askpass_path.exists()


def test_export_current_colab_notebook_treats_request_errors_as_soft_failures(tmp_path: Path, monkeypatch):
    target = tmp_path / "executed.ipynb"
    monkeypatch.setattr(bootstrap, "running_in_colab", lambda: True)
    monkeypatch.setattr(bootstrap.time, "sleep", lambda *_args, **_kwargs: None)

    responses = iter(
        [
            RuntimeError("runtime tearing down"),
            {"ipynb": {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}},
        ]
    )

    def _blocking_request(*_args, **_kwargs):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    fake_colab = ModuleType("google.colab")
    fake_colab._message = SimpleNamespace(blocking_request=_blocking_request)
    fake_google = ModuleType("google")
    fake_google.colab = fake_colab

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)

    exported = bootstrap.export_current_colab_notebook(target, attempts=2, retry_delay_sec=0.0)

    assert exported == target
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["nbformat_minor"] == 5


def test_flatten_colab_safe_requirements_expands_nested_files_and_skips_torch_family(tmp_path: Path):
    root_req = tmp_path / "requirements_colab.txt"
    base_req = tmp_path / "requirements.txt"
    nested_req = tmp_path / "extras.txt"

    base_req.write_text(
        "\n".join(
            [
                "torch~=2.10.0",
                "torchvision~=0.25.0",
                "transformers~=5.1.0",
                "-r extras.txt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    nested_req.write_text(
        "\n".join(
            [
                "open-clip-torch~=3.2.0",
                "torchaudio~=2.10.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    root_req.write_text("-r requirements.txt\npsutil>=5.9.0\n", encoding="utf-8")

    flattened = bootstrap._flatten_colab_safe_requirements(root_req)

    assert flattened == [
        "transformers~=5.1.0",
        "open-clip-torch~=3.2.0",
        "psutil>=5.9.0",
    ]


def test_install_colab_requirements_raises_when_filtered_pip_install_fails(tmp_path: Path, monkeypatch):
    req = tmp_path / "requirements_colab.txt"
    req.write_text("open-clip-torch~=3.2.0\n", encoding="utf-8")

    def fake_run(command, check=False, stdout=None, stderr=None, text=None):
        return subprocess.CompletedProcess(command, 1, stdout="pip failure details")

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    try:
        bootstrap.install_colab_requirements(req, in_colab=True)
    except RuntimeError as exc:
        assert "Colab dependency installation failed" in str(exc)
    else:
        raise AssertionError("Expected install_colab_requirements to raise on pip failure")


def test_login_and_check_hf_token_validates_identity(monkeypatch):
    calls: dict[str, object] = {}
    lines: list[str] = []

    monkeypatch.setattr(bootstrap, "resolve_hf_token", lambda: "hf-secret")

    fake_hf = ModuleType("huggingface_hub")

    def fake_login(*, token, add_to_git_credential):
        calls["login"] = (token, add_to_git_credential)

    class FakeHfApi:
        def __init__(self, token):
            calls["api_token"] = token

        def whoami(self):
            return {"name": "tester"}

    fake_hf.login = fake_login
    fake_hf.HfApi = FakeHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    assert bootstrap.login_and_check_hf_token(print_fn=lines.append) is True
    assert calls["login"] == ("hf-secret", False)
    assert calls["api_token"] == "hf-secret"
    assert lines == ["[HF] Identity verified: tester"]


def test_mirror_checkpoint_state_to_repo_copies_only_best_checkpoint(tmp_path: Path):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "repo" / "runs" / "run_1" / "checkpoint_state"

    best_source = source_root / "checkpoints" / "ckpt_best"
    other_source = source_root / "checkpoints" / "ckpt_other"
    best_source.mkdir(parents=True, exist_ok=True)
    other_source.mkdir(parents=True, exist_ok=True)
    (best_source / "checkpoint.pt").write_text("best", encoding="utf-8")
    (best_source / "adapter_meta.json").write_text("{}", encoding="utf-8")
    (other_source / "checkpoint.pt").write_text("other", encoding="utf-8")

    (source_root / "best_checkpoint.json").parent.mkdir(parents=True, exist_ok=True)
    (source_root / "best_checkpoint.json").write_text(
        json.dumps({"name": "ckpt_best", "path": str(best_source)}) + "\n",
        encoding="utf-8",
    )
    (source_root / "latest_checkpoint.json").write_text(
        json.dumps({"name": "ckpt_other", "path": str(other_source)}) + "\n",
        encoding="utf-8",
    )
    (source_root / "checkpoint_index.json").write_text(
        json.dumps(
            [
                {"name": "ckpt_best", "path": str(best_source)},
                {"name": "ckpt_other", "path": str(other_source)},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    mirrored_root = git_utils.mirror_checkpoint_state_to_repo(source_root, destination_root)

    assert mirrored_root == destination_root
    best_checkpoint_file = destination_root / "checkpoints" / "ckpt_best" / "checkpoint.pt"
    assert best_checkpoint_file.exists()
    assert best_checkpoint_file.read_text(encoding="utf-8") == "best"
    assert not (destination_root / "checkpoints" / "best").exists()
    assert not (destination_root / "checkpoints" / "ckpt_other").exists()

    mirrored_best = git_utils._read_json_dict(destination_root / "best_checkpoint.json")
    mirrored_latest = git_utils._read_json_dict(destination_root / "latest_checkpoint.json")
    mirrored_index = (destination_root / "checkpoint_index.json").read_text(encoding="utf-8")

    assert mirrored_best["path"] == str(destination_root / "checkpoints" / "ckpt_best")
    assert mirrored_latest["path"] == str(destination_root / "checkpoints" / "ckpt_best")
    assert "ckpt_other" not in mirrored_index


def test_push_repo_paths_to_github_redacts_token_from_push_errors(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    for name in ("src", "config", "scripts"):
        (repo_root / name).mkdir(parents=True, exist_ok=True)
    runtime_file = repo_root / "data" / "prepared_runtime_datasets" / "grape__fruit" / "split_manifest.json"
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(bootstrap, "resolve_github_token", lambda: "gh-secret")

    def fake_run(command, cwd=None, check=True, stdout=None, stderr=None, text=None, env=None, timeout=None):
        args = list(command[1:])
        if args == ["branch", "--show-current"]:
            return subprocess.CompletedProcess(command, 0, stdout="master\n")
        if args == ["remote", "get-url", "origin"]:
            return subprocess.CompletedProcess(command, 0, stdout="https://github.com/EfeErim/aads-open-world-plant-disease.git\n")
        if args[:4] == ["diff", "--cached", "--name-only", "--"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="data/prepared_runtime_datasets/grape__fruit/split_manifest.json\n",
            )
        if args and args[0] == "push":
            return subprocess.CompletedProcess(command, 128, stdout="remote rejected gh-secret\n")
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    try:
        bootstrap.push_repo_paths_to_github(
            repo_root,
            ["data/prepared_runtime_datasets/grape__fruit"],
            print_fn=lambda _: None,
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "GitHub push failed" in message
        assert "gh-secret" not in message
        assert "<redacted>" in message
    else:
        raise AssertionError("Expected push failure")


def test_probe_repo_update_status_reports_available_update(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    for name in ("src", "config", "scripts"):
        (repo_root / name).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(bootstrap, "_git_current_branch", lambda _repo: "master")
    monkeypatch.setattr(
        bootstrap,
        "_run_git",
        lambda args, cwd, check=True, capture_output=False: subprocess.CompletedProcess(
            ["git", *args],
            0,
            stdout="local-sha\n",
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "_run_capture",
        lambda args, cwd=None, timeout_sec=30.0: subprocess.CompletedProcess(
            args,
            0,
            stdout="remote-sha\trefs/heads/master\n",
        ),
    )

    report = bootstrap.probe_repo_update_status(repo_root)

    assert report["status"] == "ok"
    assert report["branch"] == "master"
    assert report["local_head"] == "local-sha"
    assert report["remote_head"] == "remote-sha"
    assert report["update_available"] is True
    assert report["relation"] == "update_available"
