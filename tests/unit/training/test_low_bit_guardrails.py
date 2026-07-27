import pytest

from src.training.quantization import (
    assert_no_prohibited_4bit_flags,
    find_prohibited_4bit_flags,
)


def test_find_prohibited_flags_detects_low_bit_settings():
    hits = find_prohibited_4bit_flags({'quantization': {'load_in_4bit': True}})
    assert hits


def test_find_prohibited_flags_detects_notebook_and_nested_surfaces():
    config = {
        'base': {'training': {'continual': {'adapter': {'target_modules_strategy': 'all_linear_transformer'}}}},
        'colab': {'injected': {'bnb_4bit_quant_type': 'nf4'}},
        'notebook_runtime': [{'config_patch': {'load_in_4bit': False}}],
    }
    hits = find_prohibited_4bit_flags(config)
    assert any('colab.injected.bnb_4bit_quant_type' in path for path in hits)
    assert any('notebook_runtime' in path for path in hits)


def test_assert_no_prohibited_flags_rejects_low_bit_settings():
    with pytest.raises(ValueError):
        assert_no_prohibited_4bit_flags({'nf4_enabled': True})
