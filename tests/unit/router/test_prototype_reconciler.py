from pathlib import Path

from PIL import Image

import src.router.prototype_reconciler as prototype_reconciler
from src.router.prototype_bank import build_prototype_bank
from src.router.prototype_reconciler import PrototypeMatch, nearest_target, reconcile_router_handoff, taxonomy_relation
from src.router.taxonomy_registry import build_taxonomy_registry


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def test_reconciler_accepts_router_when_prototype_agrees(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    image_path = dataset_root / "tomato__leaf" / "train" / "healthy" / "a.png"
    _write_image(image_path, (20, 90, 40))
    prototype_payload = build_prototype_bank(dataset_root=dataset_root, created_at="20260617T000000Z")
    registry_payload = build_taxonomy_registry(dataset_root=dataset_root, adapter_root=None, created_at="20260617T000000Z")

    decision = reconcile_router_handoff(
        image_path=image_path,
        router_crop="tomato",
        router_part="leaf",
        router_status="ok",
        prototype_payload=prototype_payload,
        registry_payload=registry_payload,
        min_similarity=0.1,
        min_margin=0.0,
    )

    assert decision.decision == "accept_router"
    assert decision.crop == "tomato"
    assert decision.part == "leaf"
    assert decision.to_payload()["reconciled_crop"] == "tomato"


def test_nearest_target_uses_class_prototypes_when_available(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    tomato_leaf = dataset_root / "tomato__leaf" / "train" / "late_blight" / "a.png"
    tomato_fruit = dataset_root / "tomato__fruit" / "train" / "healthy" / "b.png"
    _write_image(tomato_leaf, (20, 90, 40))
    _write_image(tomato_fruit, (190, 30, 30))
    prototype_payload = build_prototype_bank(dataset_root=dataset_root, created_at="20260617T000000Z")

    match = nearest_target(tomato_leaf, prototype_payload)

    assert match.target_id == "tomato__leaf"
    assert match.class_label == "late_blight"
    assert match.prototype_level == "class"


def test_reconciler_corrects_unknown_router_from_strong_prototype(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    tomato_image = dataset_root / "tomato__fruit" / "train" / "healthy" / "a.png"
    grape_image = dataset_root / "grape__fruit" / "train" / "healthy" / "b.png"
    _write_image(tomato_image, (190, 30, 30))
    _write_image(grape_image, (40, 20, 120))
    prototype_payload = build_prototype_bank(dataset_root=dataset_root, created_at="20260617T000000Z")
    registry_payload = build_taxonomy_registry(dataset_root=dataset_root, adapter_root=None, created_at="20260617T000000Z")

    decision = reconcile_router_handoff(
        image_path=tomato_image,
        router_crop="unknown",
        router_part="fruit",
        router_status="unknown_crop",
        prototype_payload=prototype_payload,
        registry_payload=registry_payload,
        min_similarity=0.1,
        min_margin=0.01,
    )

    assert decision.decision == "use_prototype"
    assert decision.crop == "tomato"
    assert decision.part == "fruit"
    assert decision.reason == "prototype_corrected_router_handoff"


def test_reconciler_abstains_on_part_conflict(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    image_path = dataset_root / "tomato__leaf" / "train" / "healthy" / "a.png"
    fruit_path = dataset_root / "tomato__fruit" / "train" / "healthy" / "b.png"
    _write_image(image_path, (20, 90, 40))
    _write_image(fruit_path, (190, 30, 30))
    prototype_payload = build_prototype_bank(dataset_root=dataset_root, created_at="20260617T000000Z")
    registry_payload = build_taxonomy_registry(dataset_root=dataset_root, adapter_root=None, created_at="20260617T000000Z")

    decision = reconcile_router_handoff(
        image_path=image_path,
        router_crop="tomato",
        router_part="fruit",
        router_status="ok",
        prototype_payload=prototype_payload,
        registry_payload=registry_payload,
        min_similarity=0.1,
        min_margin=0.0,
    )

    assert decision.decision == "abstain"
    assert decision.reason == "part_conflict"


def test_reconciler_overrides_untrusted_unsupported_router_crop(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    tomato_image = dataset_root / "tomato__fruit" / "train" / "healthy" / "a.png"
    grape_image = dataset_root / "grape__fruit" / "train" / "healthy" / "b.png"
    _write_image(tomato_image, (190, 30, 30))
    _write_image(grape_image, (40, 20, 120))
    prototype_payload = build_prototype_bank(dataset_root=dataset_root, created_at="20260617T000000Z")
    registry_payload = build_taxonomy_registry(dataset_root=dataset_root, adapter_root=None, created_at="20260617T000000Z")

    decision = reconcile_router_handoff(
        image_path=tomato_image,
        router_crop="eggplant",
        router_part="unknown",
        router_status="router_uncertain",
        prototype_payload=prototype_payload,
        registry_payload=registry_payload,
        min_similarity=0.1,
        min_margin=0.01,
    )

    assert decision.decision == "use_prototype"
    assert decision.crop == "tomato"
    assert decision.part == "fruit"
    assert decision.taxonomy_relation == "distant_or_unknown"
    assert decision.reason == "prototype_overrode_untrusted_router_handoff"


def test_reconciler_applies_target_policy_before_global_thresholds(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    tomato_leaf = dataset_root / "tomato__leaf" / "train" / "healthy" / "a.png"
    grape_leaf = dataset_root / "grape__leaf" / "train" / "healthy" / "b.png"
    _write_image(tomato_leaf, (20, 90, 40))
    _write_image(grape_leaf, (40, 20, 120))
    prototype_payload = build_prototype_bank(dataset_root=dataset_root, created_at="20260617T000000Z")
    registry_payload = build_taxonomy_registry(dataset_root=dataset_root, adapter_root=None, created_at="20260617T000000Z")

    decision = reconcile_router_handoff(
        image_path=tomato_leaf,
        router_crop="tomato",
        router_part="leaf",
        router_status="ok",
        prototype_payload=prototype_payload,
        registry_payload=registry_payload,
        min_similarity=0.1,
        min_margin=0.0,
        target_policies={
            "tomato__leaf": {
                "selected_policy": {
                    "min_similarity": 0.1,
                    "min_margin": 0.0,
                    "min_negative_gap": 1.0,
                }
            }
        },
    )

    assert decision.decision == "abstain"
    assert decision.reason == "negative_prototype_too_close"
    assert decision.min_negative_gap == 1.0


def test_reconciler_uses_curated_hard_negative_gap(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    tomato_positive = dataset_root / "tomato__fruit" / "train" / "healthy" / "a.png"
    grape_positive = dataset_root / "grape__fruit" / "train" / "healthy" / "b.png"
    query_image = tmp_path / "query.png"
    hard_negative = tmp_path / "hard_negative.png"
    _write_image(tomato_positive, (190, 30, 30))
    _write_image(grape_positive, (40, 20, 120))
    _write_image(query_image, (190, 30, 30))
    _write_image(hard_negative, (190, 30, 30))
    prototype_payload = build_prototype_bank(dataset_root=dataset_root, created_at="20260617T000000Z")
    prototype_payload["hard_negative_prototypes"] = {
        "tomato__fruit": {
            "negative_for_target_id": "tomato__fruit",
            "sample_count": 1,
            "centroid": prototype_payload["target_prototypes"]["tomato__fruit"]["centroid"],
        }
    }
    registry_payload = build_taxonomy_registry(dataset_root=dataset_root, adapter_root=None, created_at="20260617T000000Z")

    decision = reconcile_router_handoff(
        image_path=query_image,
        router_crop="tomato",
        router_part="fruit",
        router_status="ok",
        prototype_payload=prototype_payload,
        registry_payload=registry_payload,
        min_similarity=0.1,
        min_margin=0.0,
        min_negative_gap=0.01,
    )

    assert decision.decision == "abstain"
    assert decision.reason == "negative_prototype_too_close"
    assert decision.to_payload()["prototype_hard_negative_target"] == "tomato__fruit"
    assert decision.to_payload()["prototype_hard_negative_gap"] == 0.0


def test_reconciler_uses_manifest_exact_class_rescue_when_calibration_policy_missing(monkeypatch, tmp_path: Path):
    query_image = tmp_path / "query.png"
    _write_image(query_image, (20, 90, 40))

    def fake_nearest_target(_image_path, _prototype_payload):
        return PrototypeMatch(
            target_id="tomato__leaf",
            crop="tomato",
            part="leaf",
            similarity=0.64,
            distance=0.56,
            margin=0.038,
            class_label="bacterial_spot_leaf",
            prototype_level="class",
            hard_negative_target_id="tomato__leaf",
            hard_negative_similarity=0.64,
            hard_negative_gap=0.0,
        )

    monkeypatch.setattr(prototype_reconciler, "nearest_target", fake_nearest_target)

    decision = reconcile_router_handoff(
        image_path=query_image,
        router_crop="potato",
        router_part="leaf",
        router_status="unknown_crop",
        prototype_payload={},
        registry_payload={"targets": [{"target_id": "tomato__leaf", "crop_canonical_name": "tomato"}]},
        min_similarity=0.2,
        min_margin=0.03,
        min_negative_gap=0.06,
        target_policies={
            "__requires_selected_policy__": True,
            "tomato__leaf": {
                "status": "no_eligible_policy",
                "selected_policy": None,
            },
        },
        expected_class_label="bacterial_spot_leaf",
    )

    assert decision.decision == "use_prototype"
    assert decision.reason == "prototype_overrode_untrusted_router_handoff"
    assert decision.min_margin == 0.02
    assert decision.min_negative_gap == 0.0


def test_reconciler_blocks_below_calibrated_untrusted_margin_floor(monkeypatch, tmp_path: Path):
    query_image = tmp_path / "query.png"
    _write_image(query_image, (190, 30, 30))

    def fake_nearest_target(_image_path, _prototype_payload):
        return PrototypeMatch(
            target_id="apricot__fruit",
            crop="apricot",
            part="fruit",
            similarity=0.62,
            distance=0.61,
            margin=0.016,
            class_label="shot_hole_fruit",
            prototype_level="class",
        )

    monkeypatch.setattr(prototype_reconciler, "nearest_target", fake_nearest_target)

    decision = reconcile_router_handoff(
        image_path=query_image,
        router_crop="eggplant",
        router_part="unknown",
        router_status="unknown_crop",
        prototype_payload={},
        registry_payload={
            "targets": [
                {"target_id": "tomato__fruit", "crop_canonical_name": "tomato"},
                {"target_id": "apricot__fruit", "crop_canonical_name": "apricot"},
            ]
        },
        min_similarity=0.2,
        min_margin=0.03,
        target_policies={
            "apricot__fruit": {
                "negative_mode": "none",
                "class_policies": {
                    "shot_hole_fruit": {
                        "selected_policy": {
                            "min_similarity": 0.2,
                            "min_margin": 0.0,
                            "min_negative_gap": 0.0,
                        }
                    }
                },
            }
        },
    )

    assert decision.decision == "abstain"
    assert decision.reason == "prototype_evidence_weak"
    assert decision.min_margin == 0.02


def test_taxonomy_relation_detects_same_family(tmp_path: Path):
    registry_payload = {
        "targets": [
            {
                "target_id": "apricot__leaf",
                "crop_canonical_name": "apricot",
                "common_names": ["apricot"],
                "synonyms": [],
                "genus": "Prunus",
                "family": "Rosaceae",
            },
            {
                "target_id": "strawberry__leaf",
                "crop_canonical_name": "strawberry",
                "common_names": ["strawberry"],
                "synonyms": [],
                "genus": "Fragaria",
                "family": "Rosaceae",
            },
        ]
    }
    taxonomy = {entry["target_id"]: entry for entry in registry_payload["targets"]}

    assert taxonomy_relation("strawberry", "apricot__leaf", taxonomy) == "same_family"
