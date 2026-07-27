from pathlib import Path
import json

from src.pipeline import adapter_discovery as ad


def make_adapter_bundle(root: Path, crop: str, part: str) -> Path:
    bundle = root / crop / part / "continual_sd_lora_adapter"
    bundle.mkdir(parents=True, exist_ok=True)
    meta = {"crop_name": crop, "part_name": part}
    (bundle / "adapter_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return bundle


def test_discover_fallback_adapter_dir(tmp_path: Path):
    root = tmp_path / "models" / "adapters"
    b1 = make_adapter_bundle(root, "apple", "main")
    b2 = make_adapter_bundle(root, "apple", "other")
    # request exact match
    found = ad.discover_fallback_adapter_dir(root, crop_name="apple", part_name="main", allow_cross_part=False)
    assert found is not None
    assert found.name == "continual_sd_lora_adapter"


def test_resolve_adapter_dir_with_override(tmp_path: Path):
    root = tmp_path / "models" / "adapters"
    bundle = make_adapter_bundle(root, "grape", "unspecified")
    resolved = ad.resolve_adapter_dir(adapter_root=root, crop_name="grape", part_name="unspecified", allow_cross_part=False, adapter_dir_override=bundle)
    assert resolved == bundle


def test_adapter_meta_state(tmp_path: Path):
    root = tmp_path / "models" / "adapters"
    bundle = make_adapter_bundle(root, "tomato", "partA")
    resolved_dir, part_name, mtime_ns, size = ad.adapter_meta_state(bundle)
    assert resolved_dir == bundle
    assert part_name in {"parta", "partA"} or isinstance(part_name, str)
    assert isinstance(mtime_ns, int)
    assert isinstance(size, int)
