import sys
from pathlib import Path
from types import ModuleType

from scripts import colab_repo_bootstrap
from scripts.notebook_helpers import nb2_training_helpers


def test_run_bootstrap_notebook_nb2_does_not_fail_when_update_check_is_unavailable(monkeypatch):
    fake_bootstrap_module = ModuleType("scripts.colab_notebooks_bootstrap")
    bootstrap_result = {
        "ROOT": Path("/content/bitirmeprojesi"),
        "IN_COLAB": True,
        "GH_TOKEN": "",
        "HF_TOKEN": "",
        "bootstrap_status": "ok",
    }
    fake_bootstrap_module.bootstrap_notebook = lambda **_kwargs: bootstrap_result
    fake_bootstrap_module.print_bootstrap_status = lambda _result: None

    monkeypatch.setitem(sys.modules, "scripts.colab_notebooks_bootstrap", fake_bootstrap_module)
    monkeypatch.setattr(
        colab_repo_bootstrap,
        "_ensure_repo_root_for_update_check",
        lambda: (_ for _ in ()).throw(ImportError("update check unavailable")),
    )

    result = nb2_training_helpers.run_bootstrap_notebook_nb2(require_colab_requirements=False)

    assert result == bootstrap_result
