import sys
import types

from src.training.services import trainer_runtime


def test_resolve_auto_model_factory_imports_transformers_late(monkeypatch):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModel = object()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    assert trainer_runtime._resolve_auto_model_factory(None) is fake_transformers.AutoModel
