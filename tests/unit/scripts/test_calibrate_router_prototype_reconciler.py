from pathlib import Path

from PIL import Image

from scripts.calibrate_router_prototype_reconciler import ScoredRow, calibrate, has_runtime_policy, score_manifest
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


def test_calibrate_emits_target_policy_for_safe_supported_rows():
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
            expected_target="grape__fruit",
            expected_behavior="answer",
            predicted_target="grape__fruit",
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

    assert result["selected_policy"]["supported_precision"] == 1.0
    assert result["target_policies"]["tomato__fruit"]["status"] == "target_specific"


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


def test_calibrate_can_select_target_policies_without_global_negatives():
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
        target_policy_negative_mode="none",
    )

    assert result["selected_policy"] is None
    assert result["target_policies"]["tomato__fruit"]["status"] == "target_specific"
    assert result["target_policies"]["tomato__fruit"]["negative_mode"] == "none"


def test_calibrate_selects_near_miss_target_policy_with_target_constraints():
    rows = [
        ScoredRow(
            image_id=f"demo_{index:03d}",
            expected_target="grape__fruit",
            expected_behavior="answer",
            predicted_target="grape__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image=f"{index}.png",
            status="ok",
        )
        for index in range(1, 55)
    ]
    rows.append(
        ScoredRow(
            image_id="demo_055",
            expected_target="grape__fruit",
            expected_behavior="answer",
            predicted_target="grape__leaf",
            similarity=0.7,
            margin=0.04,
            resolved_image="55.png",
            status="ok",
        )
    )

    result = calibrate(
        rows,
        similarity_grid=(0.1,),
        margin_grid=(0.0,),
        min_precision=0.985,
        min_coverage=1.0,
        target_min_precision=0.98,
        target_max_supported_wrong=1,
    )

    assert result["selected_policy"] is None
    grape_policy = result["target_policies"]["grape__fruit"]
    assert grape_policy["status"] == "target_specific"
    assert grape_policy["selected_policy"]["supported_precision"] == 0.981818
    assert grape_policy["selected_policy"]["supported_wrong"] == 1
    assert grape_policy["selected_policy"]["supported_wrong_rows"] == [
        {
            "image_id": "demo_055",
            "expected_target": "grape__fruit",
            "predicted_target": "grape__leaf",
            "prototype_class_label": None,
            "prototype_level": "target",
            "similarity": 0.7,
            "margin": 0.04,
        }
    ]


def test_calibrate_rejects_noisy_target_policy_with_target_constraints():
    rows = [
        ScoredRow(
            image_id=f"demo_{index:03d}",
            expected_target="tomato__leaf",
            expected_behavior="answer",
            predicted_target="tomato__leaf" if index <= 92 else "tomato__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image=f"{index}.png",
            status="ok",
            prototype_class_label="clean_leaf" if index <= 92 else "fruit_confuser",
        )
        for index in range(1, 109)
    ]

    result = calibrate(
        rows,
        similarity_grid=(0.1,),
        margin_grid=(0.0,),
        min_precision=0.985,
        min_coverage=1.0,
        target_min_precision=0.98,
        target_max_supported_wrong=1,
    )

    tomato_policy = result["target_policies"]["tomato__leaf"]
    assert tomato_policy["status"] == "no_eligible_policy"
    assert "supported_precision_below_target" in tomato_policy["failure_reasons"]
    assert "supported_wrong_above_target" in tomato_policy["failure_reasons"]
    assert tomato_policy["class_policies"]["clean_leaf"]["status"] == "class_specific"
    assert tomato_policy["class_policies"]["clean_leaf"]["selected_policy"]["supported_correct"] == 92
    assert has_runtime_policy(result)


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


def test_calibrate_rejects_class_policy_with_cross_part_supported_false_accept():
    rows = [
        ScoredRow(
            image_id=f"demo_{index:03d}",
            expected_target="apricot__leaf",
            expected_behavior="answer",
            predicted_target="apricot__fruit" if index == 1 else "apricot__leaf",
            similarity=0.7,
            margin=0.04,
            resolved_image=f"{index}.png",
            status="ok",
            prototype_class_label="fruit_confuser" if index == 1 else "shot_hole_leaf",
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
        target_min_precision=0.98,
        target_max_supported_wrong=1,
        target_max_cross_part_supported_wrong=0,
        target_class_min_accepted=5,
    )

    apricot_policy = result["target_policies"]["apricot__leaf"]
    assert apricot_policy["status"] == "no_eligible_policy"
    class_policy = apricot_policy["class_policies"]["shot_hole_leaf"]
    assert class_policy["status"] == "no_eligible_policy"
    assert class_policy["best_candidate"]["supported_cross_part_wrong"] == 1
    assert "supported_cross_part_wrong_above_class_target" in class_policy["failure_reasons"]


def test_calibrate_allows_clean_fruit_class_policy_part_conflict_override():
    rows = [
        ScoredRow(
            image_id=f"demo_{index:03d}",
            expected_target="grape__fruit",
            expected_behavior="answer",
            predicted_target="grape__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image=f"{index}.png",
            status="ok",
            prototype_class_label="mildew_fruit",
        )
        for index in range(1, 8)
    ]
    rows.append(
        ScoredRow(
            image_id="demo_008",
            expected_target="grape__fruit",
            expected_behavior="answer",
            predicted_target="grape__leaf",
            similarity=0.7,
            margin=0.04,
            resolved_image="8.png",
            status="ok",
            prototype_class_label="leaf_confuser",
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
        target_max_cross_part_supported_wrong=0,
        target_class_min_accepted=5,
    )

    grape_policy = result["target_policies"]["grape__fruit"]
    assert grape_policy["status"] == "no_eligible_policy"
    class_policy = grape_policy["class_policies"]["mildew_fruit"]
    assert class_policy["status"] == "class_specific"
    assert class_policy["selected_policy"]["supported_cross_part_wrong"] == 0
    assert class_policy["selected_policy"]["allow_part_conflict_override"] is True


def test_calibrate_allows_clean_fruit_target_policy_part_conflict_override():
    rows = [
        ScoredRow(
            image_id=f"demo_{index:03d}",
            expected_target="strawberry__fruit",
            expected_behavior="answer",
            predicted_target="strawberry__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image=f"{index}.png",
            status="ok",
            prototype_class_label="healthy_fruit",
        )
        for index in range(1, 11)
    ]

    result = calibrate(
        rows,
        similarity_grid=(0.1,),
        margin_grid=(0.0,),
        negative_gap_grid=(0.0,),
        min_precision=0.985,
        min_coverage=0.8,
        target_min_precision=0.98,
        target_max_supported_wrong=1,
        target_max_cross_part_supported_wrong=0,
        target_class_min_accepted=5,
    )

    selected = result["target_policies"]["strawberry__fruit"]["selected_policy"]
    assert selected["allow_part_conflict_override"] is True
    assert selected["full_set_validation"]["supported_cross_part_wrong"] == 0


def test_calibrate_blocks_fruit_target_policy_part_conflict_override_when_full_set_has_cross_part_wrong():
    rows = [
        ScoredRow(
            image_id=f"demo_{index:03d}",
            expected_target="strawberry__fruit",
            expected_behavior="answer",
            predicted_target="strawberry__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image=f"{index}.png",
            status="ok",
            prototype_class_label="healthy_fruit",
        )
        for index in range(1, 11)
    ]
    rows.append(
        ScoredRow(
            image_id="demo_011",
            expected_target="strawberry__leaf",
            expected_behavior="answer",
            predicted_target="strawberry__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image="11.png",
            status="ok",
            prototype_class_label="healthy_fruit",
        )
    )

    result = calibrate(
        rows,
        similarity_grid=(0.1,),
        margin_grid=(0.0,),
        negative_gap_grid=(0.0,),
        min_precision=0.985,
        min_coverage=0.8,
        target_min_precision=0.98,
        target_max_supported_wrong=1,
        target_max_cross_part_supported_wrong=0,
        target_class_min_accepted=5,
    )

    policy = result["target_policies"]["strawberry__fruit"]
    assert policy["status"] == "no_eligible_policy"
    assert policy["selected_policy"] is None
    assert "supported_cross_part_wrong_above_target" in policy["failure_reasons"]


def test_calibrate_does_not_allow_leaf_class_policy_part_conflict_override():
    rows = [
        ScoredRow(
            image_id=f"demo_{index:03d}",
            expected_target="grape__leaf",
            expected_behavior="answer",
            predicted_target="grape__leaf",
            similarity=0.7,
            margin=0.04,
            resolved_image=f"{index}.png",
            status="ok",
            prototype_class_label="mildew_leaf",
        )
        for index in range(1, 8)
    ]
    rows.append(
        ScoredRow(
            image_id="demo_008",
            expected_target="grape__leaf",
            expected_behavior="answer",
            predicted_target="grape__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image="8.png",
            status="ok",
            prototype_class_label="fruit_confuser",
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
        target_max_cross_part_supported_wrong=0,
        target_class_min_accepted=5,
    )

    grape_policy = result["target_policies"]["grape__leaf"]
    assert grape_policy["status"] == "no_eligible_policy"
    class_policy = grape_policy["class_policies"]["mildew_leaf"]
    assert class_policy["status"] == "class_specific"
    assert "allow_part_conflict_override" not in class_policy["selected_policy"]


def test_calibrate_class_policy_counts_supported_false_accepts():
    rows = [
        ScoredRow(
            image_id=f"demo_{index:03d}",
            expected_target="tomato__leaf",
            expected_behavior="answer",
            predicted_target="tomato__leaf",
            similarity=0.7,
            margin=0.04,
            resolved_image=f"{index}.png",
            status="ok",
            prototype_class_label="late_blight",
        )
        for index in range(1, 8)
    ]
    rows.extend(
        [
            ScoredRow(
                image_id="demo_008",
                expected_target="grape__leaf",
                expected_behavior="answer",
                predicted_target="tomato__leaf",
                similarity=0.7,
                margin=0.04,
                resolved_image="8.png",
                status="ok",
                prototype_class_label="late_blight",
            ),
            ScoredRow(
                image_id="demo_009",
                expected_target="tomato__leaf",
                expected_behavior="answer",
                predicted_target="tomato__fruit",
                similarity=0.7,
                margin=0.04,
                resolved_image="9.png",
                status="ok",
                prototype_class_label="fruit_confuser",
            ),
        ]
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

    tomato_policy = result["target_policies"]["tomato__leaf"]
    assert tomato_policy["status"] == "no_eligible_policy"
    class_policy = tomato_policy["class_policies"]["late_blight"]
    assert class_policy["status"] == "no_eligible_policy"
    assert class_policy["best_candidate"]["supported_wrong"] == 1
    assert "supported_wrong_above_class_target" in class_policy["failure_reasons"]


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


def test_calibrate_blocks_exact_class_rescue_policy_with_class_false_accept():
    rows = [
        ScoredRow(
            image_id="demo_001",
            expected_target="tomato__leaf",
            expected_class="late_blight",
            expected_behavior="answer",
            predicted_target="tomato__leaf",
            similarity=0.7,
            margin=0.04,
            resolved_image="1.png",
            status="ok",
            prototype_class_label="late_blight",
        ),
        ScoredRow(
            image_id="demo_002",
            expected_target="tomato__leaf",
            expected_class="leaf_mold",
            expected_behavior="answer",
            predicted_target="tomato__leaf",
            similarity=0.7,
            margin=0.04,
            resolved_image="2.png",
            status="ok",
            prototype_class_label="late_blight",
        ),
        ScoredRow(
            image_id="demo_003",
            expected_target="tomato__leaf",
            expected_class="late_blight",
            expected_behavior="answer",
            predicted_target="tomato__leaf",
            similarity=0.7,
            margin=0.04,
            resolved_image="3.png",
            status="ok",
            prototype_class_label="late_blight",
        ),
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
        ),
    ]

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
    assert class_policy["exact_class_rescue_policy"] is None


def test_calibrate_can_allow_negative_false_accepts_explicitly():
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
        max_negative_false_accepts=1,
        max_negative_false_accept_rate=1.0,
    )

    assert result["selected_policy"]["negative_false_accept_count"] == 1
