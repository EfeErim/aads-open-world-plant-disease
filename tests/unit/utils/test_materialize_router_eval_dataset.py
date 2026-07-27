import json
from pathlib import Path

from scripts.materialize_router_eval_dataset import materialize_router_eval_datasets


def _write_image(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def test_materialize_router_eval_datasets_builds_expected_layout_and_manifests(tmp_path: Path):
    runtime_root = tmp_path / "data" / "prepared_runtime_datasets"
    ood_root = tmp_path / "data" / "ood_dataset" / "final"
    dev_root = tmp_path / "data" / "router_eval"
    holdout_root = tmp_path / "data" / "router_eval_holdout"

    _write_image(runtime_root / "tomato__leaf" / "val" / "healthy" / "dev_leaf.jpg", b"dev-leaf")
    _write_image(runtime_root / "tomato__leaf" / "test" / "healthy" / "holdout_leaf.jpg", b"holdout-leaf")
    _write_image(runtime_root / "grape__fruit" / "val" / "healthy" / "dev_fruit.jpg", b"dev-fruit")
    _write_image(runtime_root / "grape__fruit" / "test" / "healthy" / "holdout_fruit.jpg", b"holdout-fruit")
    _write_image(runtime_root / "strawberry__fruit" / "val" / "healthy" / "skip.jpg", b"skip")

    _write_image(ood_root / "tomato__leaf_ood_final" / "non_plant_misc" / "tool.jpg", b"non-plant")
    _write_image(ood_root / "tomato__leaf_ood_final" / "off_crop_secondary" / "pepper.jpg", b"off-crop")
    _write_image(ood_root / "tomato__leaf_ood_final" / "scene_context_leak_check" / "scene.jpg", b"scene")
    _write_image(ood_root / "tomato__leaf_ood_final" / "unsupported_tomato_unknowns" / "root.jpg", b"wrong")

    result = materialize_router_eval_datasets(
        repo_root=tmp_path,
        runtime_root=runtime_root,
        ood_root=ood_root,
        dev_root=dev_root,
        holdout_root=holdout_root,
        seed=123,
        force=False,
    )

    assert result["dev_count"] == 6
    assert result["holdout_count"] == 2
    assert (dev_root / "id" / "tomato" / "leaf").is_dir()
    assert (dev_root / "id" / "grape" / "fruit").is_dir()
    assert not (dev_root / "id" / "strawberry" / "fruit").exists()
    assert (dev_root / "negatives" / "non_plant").is_dir()
    assert (dev_root / "negatives" / "off_crop").is_dir()
    assert (dev_root / "ambiguous").is_dir()
    assert (dev_root / "wrong_part" / "tomato" / "unsupported_unknown").is_dir()

    manifest = json.loads((dev_root / "router_eval_manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"]["duplicate_sha256_count"] == 0
    assert manifest["summary"]["counts_by_group"] == {
        "ambiguous": 1,
        "id": 2,
        "non_plant": 1,
        "off_crop": 1,
        "wrong_part": 1,
    }
    assert all("source_path" in entry and "sha256" in entry for entry in manifest["entries"])
