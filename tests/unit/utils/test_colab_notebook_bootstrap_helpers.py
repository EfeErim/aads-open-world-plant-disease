from pathlib import Path

from scripts import colab_notebook_bootstrap_helpers as bootstrap_helpers


def test_presentation_colab_profile_keeps_peft_compatible_torchao_pin():
    repo_root = Path(__file__).resolve().parents[3]
    requirements = (repo_root / "requirements_presentation_colab.txt").read_text(encoding="utf-8").splitlines()

    assert "torchao==0.17.0" in requirements


def test_setup_notebook_environment_uses_repo_relative_requirements_override(tmp_path: Path, monkeypatch, capsys):
    requirements_path = tmp_path / "requirements_presentation_colab.txt"
    requirements_path.write_text("open-clip-torch~=3.2.0\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(bootstrap_helpers, "resolve_github_token", lambda: "")
    monkeypatch.setattr(bootstrap_helpers, "resolve_huggingface_token", lambda: "")
    monkeypatch.setattr(bootstrap_helpers, "bootstrap_repo_root", lambda repo_url=None: tmp_path)
    monkeypatch.setattr(bootstrap_helpers, "_running_in_colab", lambda: True)
    monkeypatch.setattr(bootstrap_helpers.os, "chdir", lambda _path: None)
    monkeypatch.setenv("AADS_COLAB_REQUIREMENTS_FILE", requirements_path.name)
    monkeypatch.setattr(
        "scripts.colab_repo_bootstrap.install_colab_requirements",
        lambda path, in_colab: calls.append((path, in_colab)),
    )

    root = bootstrap_helpers.setup_notebook_environment(print_tokens=False)

    assert root == tmp_path
    assert calls == [(requirements_path.resolve(), True)]
    assert "[SETUP] Dependency profile: requirements_presentation_colab.txt" in capsys.readouterr().out


def test_setup_notebook_environment_rejects_requirements_outside_repo(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(bootstrap_helpers, "resolve_github_token", lambda: "")
    monkeypatch.setattr(bootstrap_helpers, "resolve_huggingface_token", lambda: "")
    monkeypatch.setattr(bootstrap_helpers, "bootstrap_repo_root", lambda repo_url=None: tmp_path)
    monkeypatch.setattr(bootstrap_helpers, "_running_in_colab", lambda: True)
    monkeypatch.setattr(bootstrap_helpers.os, "chdir", lambda _path: None)

    try:
        bootstrap_helpers.setup_notebook_environment(
            print_tokens=False,
            requirements_file="../outside.txt",
        )
    except ValueError as exc:
        assert "must stay under the repo root" in str(exc)
    else:
        raise AssertionError("Expected an out-of-repo requirements file to be rejected")
