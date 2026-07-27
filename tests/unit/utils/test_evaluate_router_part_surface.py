from pathlib import Path

from scripts import evaluate_router_part_surface as eval_script


def test_discover_eval_samples_normalizes_part_alias_and_filters_images(tmp_path: Path):
    (tmp_path / "tomato" / "whole").mkdir(parents=True)
    (tmp_path / "tomato" / "whole" / "img1.png").write_bytes(b"png")
    (tmp_path / "tomato" / "whole" / "notes.txt").write_text("skip", encoding="utf-8")

    samples = eval_script.discover_eval_samples(tmp_path)

    assert samples == [
        {
            "image_path": tmp_path / "tomato" / "whole" / "img1.png",
            "expected_crop": "tomato",
            "expected_part": "whole plant",
        }
    ]


def test_sweep_part_thresholds_prefers_zero_unsupported_then_precision():
    samples = [
        {
            "expected_crop": "tomato",
            "expected_part": "fruit",
            "crop_correct": True,
            "compatible_parts": ["leaf", "fruit", "whole plant"],
            "raw_part_label": "fruit",
            "raw_part_confidence": 0.46,
            "raw_part_second_confidence": 0.21,
            "raw_part_margin": 0.25,
            "part_unknown_confidence": 0.11,
        },
        {
            "expected_crop": "tomato",
            "expected_part": "leaf",
            "crop_correct": True,
            "compatible_parts": ["leaf", "fruit", "whole plant"],
            "raw_part_label": "leaf",
            "raw_part_confidence": 0.38,
            "raw_part_second_confidence": 0.30,
            "raw_part_margin": 0.08,
            "part_unknown_confidence": 0.12,
        },
    ]

    sweep = eval_script.sweep_part_thresholds(
        samples,
        min_confidence_grid=[0.30, 0.40],
        margin_grid=[0.05, 0.10],
    )

    assert sweep["sample_count"] == 2
    assert sweep["recommended"] == {
        "min_confidence": 0.3,
        "margin": 0.05,
        "sample_count": 2,
        "non_unknown_precision": 1.0,
        "non_unknown_recall": 1.0,
        "abstention_rate": 0.0,
        "unsupported_part_emission": 0,
        "confusion_matrix": {
            "fruit": {"fruit": 1},
            "leaf": {"leaf": 1},
        },
    }


def test_summarize_predictions_marks_crop_mismatch_outside_part_confusion():
    summary = eval_script.summarize_predictions(
        [
            {
                "expected_part": "fruit",
                "predicted_part": "fruit",
                "crop_correct": True,
                "compatible_parts": ["fruit", "leaf"],
            },
            {
                "expected_part": "leaf",
                "predicted_part": "unknown",
                "crop_correct": False,
                "compatible_parts": ["fruit", "leaf"],
            },
        ]
    )

    assert summary == {
        "sample_count": 2,
        "non_unknown_precision": 1.0,
        "non_unknown_recall": 0.5,
        "abstention_rate": 0.0,
        "unsupported_part_emission": 0,
        "confusion_matrix": {
            "fruit": {"fruit": 1},
            "leaf": {"crop_mismatch": 1},
        },
    }
