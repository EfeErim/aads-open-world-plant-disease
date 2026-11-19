from pathlib import Path

from PIL import Image

from scripts.calibrate_router_prototype_reconciler import ScoredRow, calibrate, score_manifest
from src.router.prototype_bank import build_prototype_bank, write_prototype_bank


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def test_score_manifest_and_calibrate_selects_policy(tmp_path: Path):
    repo_root = tmp_path
    dataset_root = repo_root / "data" / "prepared_runtime_datasets"
    tomato_image = dataset_root / "tomato__fruit" / "train" / "healthy" / "a.png"
    grape_image = dataset_root / "grape__fruit" / "train" / "healthy" / "b.png"
    _write_image(tomato_image, (190, 30, 30))
    _write_image(grape_image, (40, 20, 120))
    prototype_payload = build_prototype_bank(dataset_root=dataset_root, created_at="20260617T000000Z")
    prototype_path = write_prototype_bank(prototype_payload, tmp_path / "prototype_bank.json")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "image_id,source,expected_target,expected_behavior,notes",
                "demo_001,local_test_pool:data/prepared_runtime_datasets/tomato__fruit/train/healthy/a.png,tomato__fruit,answer,",
                "demo_002,local_test_pool:data/prepared_runtime_datasets/grape__fruit/train/healthy/b.png,grape__fruit,answer,",
                "demo_003,local_test_pool:data/prepared_runtime_datasets/tomato__fruit/train/healthy/a.png,non_plant,abstain,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = score_manifest(manifest_path=manifest, prototype_bank_path=prototype_path, repo_root=repo_root)
    result = calibrate(
        rows,
        similarity_grid=(0.1,),
        margin_grid=(0.0,),
        min_precision=1.0,
        min_coverage=1.0,
    )

    assert len(rows) == 3
    assert rows[0].prototype_class_label == "healthy"
    assert rows[0].prototype_level == "class"
    assert result["selected_policy"] is None
    assert result["best_candidate"]["supported_precision"] == 1.0
    assert result["best_candidate"]["supported_coverage"] == 1.0
    assert result["best_candidate"]["non_plant_false_accept_count"] == 1
    assert result["target_policies"]["grape__fruit"]["status"] == "no_eligible_policy"


def test_calibrate_blocks_unknown_crop_false_accept_by_default():
    rows = [
        ScoredRow(
            image_id="demo_001",
            expected_target="tomato__fruit",
            expected_behavior="answer",
            predicted_target="tomato__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image="a.png",
            status="ok",
        ),
        ScoredRow(
            image_id="demo_002",
            expected_target="unknown_crop",
            expected_behavior="abstain",
            predicted_target="tomato__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image="b.png",
            status="ok",
        ),
    ]

    result = calibrate(
        rows,
        similarity_grid=(0.1,),
        margin_grid=(0.0,),
        min_precision=1.0,
        min_coverage=1.0,
    )

    assert result["selected_policy"] is None
    assert result["best_candidate"]["supported_precision"] == 1.0
    assert result["best_candidate"]["negative_false_accept_count"] == 1
    assert result["best_candidate"]["non_plant_false_accept_count"] == 0


def test_calibrate_rejects_target_policy_with_cross_part_supported_false_accept():
    rows = [
        ScoredRow(
            image_id=f"demo_{index:03d}",
            expected_target="apricot__leaf",
            expected_behavior="answer",
            predicted_target="apricot__leaf",
            similarity=0.7,
            margin=0.04,
            resolved_image=f"{index}.png",
            status="ok",
            prototype_class_label="shot_hole_leaf",
        )
        for index in range(1, 11)
    ]
    rows.append(
        ScoredRow(
            image_id="demo_011",
            expected_target="apricot__fruit",
            expected_behavior="answer",
            predicted_target="apricot__leaf",
            similarity=0.7,
            margin=0.04,
            resolved_image="11.png",
            status="ok",
            prototype_class_label="shot_hole_leaf",
        )
    )

    result = calibrate(
        rows,
        similarity_grid=(0.1,),
        margin_grid=(0.0,),
        min_precision=0.985,
        min_coverage=1.0,
        target_min_precision=1.0,
        target_max_supported_wrong=1,
        target_max_cross_part_supported_wrong=0,
    )

    apricot_policy = result["target_policies"]["apricot__leaf"]
    assert apricot_policy["status"] == "no_eligible_policy"
    assert "supported_cross_part_wrong_above_target" in apricot_policy["failure_reasons"]
    validation = apricot_policy["best_candidate"]["full_set_validation"]
    assert validation["supported_cross_part_wrong"] == 1
    assert validation["supported_cross_part_wrong_image_ids"] == ["demo_011"]


def test_calibrate_emits_exact_class_rescue_policy_below_class_min():
    rows = [
        ScoredRow(
            image_id=f"demo_{index:03d}",
            expected_target="tomato__leaf",
            expected_class="late_blight",
            expected_behavior="answer",
            predicted_target="tomato__leaf",
            similarity=0.7,
            margin=0.04,
            resolved_image=f"{index}.png",
            status="ok",
            prototype_class_label="late_blight",
        )
        for index in range(1, 4)
    ]
    rows.append(
        ScoredRow(
            image_id="demo_004",
            expected_target="tomato__leaf",
            expected_class="leaf_mold",
            expected_behavior="answer",
            predicted_target="grape__leaf",
            similarity=0.7,
            margin=0.04,
            resolved_image="4.png",
            status="ok",
            prototype_class_label="leaf_mold",
        )
    )

    result = calibrate(
        rows,
        similarity_grid=(0.1,),
        margin_grid=(0.0,),
        min_precision=0.985,
        min_coverage=1.0,
        target_min_precision=0.98,
        target_max_supported_wrong=0,
        target_class_min_accepted=5,
    )

    class_policy = result["target_policies"]["tomato__leaf"]["class_policies"]["late_blight"]
    assert class_policy["status"] == "no_eligible_policy"
    rescue = class_policy["exact_class_rescue_policy"]
    assert rescue["allow_expected_class_rescue"] is True
    assert rescue["ignore_hard_negative_gap"] is True
    assert rescue["exact_class_supported_correct"] == 3
    assert rescue["exact_class_supported_wrong"] == 0
