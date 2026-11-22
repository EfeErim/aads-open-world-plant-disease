from pathlib import Path

from src.colab import notebooks_bootstrap
from src.colab import repo_bootstrap as colab_repo_bootstrap


def test_run_notebook_bootstrap_does_not_fail_when_update_check_is_unavailable(monkeypatch):
    bootstrap_result = {
        "ROOT": Path("/content/bitirmeprojesi"),
        "IN_COLAB": True,
        "GH_TOKEN": "",
        "HF_TOKEN": "",
        "bootstrap_status": "ok",
    }
    calls = []
    monkeypatch.setattr(
        notebooks_bootstrap,
        "bootstrap_notebook",
        lambda **kwargs: calls.append(kwargs) or bootstrap_result,
    )
    monkeypatch.setattr(notebooks_bootstrap, "print_bootstrap_status", lambda _result: None)
    monkeypatch.setattr(
        colab_repo_bootstrap,
        "_ensure_repo_root_for_update_check",
        lambda: (_ for _ in ()).throw(ImportError("update check unavailable")),
    )

    result = notebooks_bootstrap.run_notebook_bootstrap("Notebook 2", require_colab_requirements=False)

    assert result == bootstrap_result
    assert calls == [{"notebook_name": "Notebook 2", "require_colab_requirements": False, "auto_clone_repo": True}]
