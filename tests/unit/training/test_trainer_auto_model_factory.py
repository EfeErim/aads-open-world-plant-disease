import sys
import types

from src.training import continual_sd_lora


def test_resolve_auto_model_factory_imports_transformers_late(monkeypatch):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModel = object()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    assert continual_sd_lora._resolve_auto_model_factory(None) is fake_transformers.AutoModel
