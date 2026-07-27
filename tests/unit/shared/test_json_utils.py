from pathlib import Path

from src.shared.json_utils import deep_merge, read_json_dict, write_json


def test_deep_merge_keeps_inputs_immutable():
    base = {"training": {"continual": {"seed": 42, "optimization": {"mixed_precision": "auto"}}}}
    override = {"training": {"continual": {"optimization": {"mixed_precision": "bf16"}}}}

    merged = deep_merge(base, override)

    assert merged["training"]["continual"]["optimization"]["mixed_precision"] == "bf16"
    assert base["training"]["continual"]["optimization"]["mixed_precision"] == "auto"


def test_write_and_read_json_dict_roundtrip(tmp_path: Path):
    path = tmp_path / "payload.json"
    payload = {"status": "ok", "metrics": {"accuracy": 0.95}}

    write_json(path, payload, ensure_ascii=False)
    loaded = read_json_dict(path)

    assert loaded == payload
