from pathlib import Path

from scripts import evaluate_router_surface as eval_script
from src.shared.contracts import RouterAnalysisResult, RouterDetection


def test_discover_eval_samples_parses_full_router_surface(tmp_path: Path):
    paths = [
        tmp_path / "id" / "tomato" / "leaf" / "id.png",
        tmp_path / "negatives" / "off_crop" / "potato_leaf" / "neg.jpg",
        tmp_path / "negatives" / "non_plant" / "tool" / "tool.webp",
        tmp_path / "ambiguous" / "mixed_crop" / "amb.bmp",
        tmp_path / "wrong_part" / "tomato" / "root" / "root.png",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    (tmp_path / "id" / "tomato" / "leaf" / "notes.txt").write_text("skip", encoding="utf-8")

    samples = eval_script.discover_eval_samples(tmp_path)

    assert [sample["group"] for sample in samples] == [
        "id",
        "off_crop",
        "non_plant",
        "ambiguous",
        "wrong_part",
    ]
    assert samples[0]["expected_crop"] == "tomato"
    assert samples[0]["expected_part"] == "leaf"
    assert samples[-1]["unsupported_part"] == "root"


def test_summarize_predictions_reports_crop_negative_and_part_metrics():
    rows = [
        {
            "group": "id",
            "handoff_crop": True,
            "crop_correct": True,
            "part_correct": True,
            "part_abstained": False,
            "unsupported_part_emitted": False,
            "latency_ms": 10.0,
        },
        {
            "group": "id",
            "handoff_crop": True,
            "crop_correct": True,
            "part_correct": False,
            "part_abstained": True,
            "unsupported_part_emitted": False,
            "latency_ms": 20.0,
        },
        {
            "group": "off_crop",
            "handoff_crop": True,
            "crop_correct": False,
            "part_correct": False,
            "part_abstained": False,
            "unsupported_part_emitted": True,
            "latency_ms": 30.0,
        },
        {
            "group": "non_plant",
            "handoff_crop": False,
            "crop_correct": False,
            "part_correct": False,
            "part_abstained": True,
            "unsupported_part_emitted": False,
            "latency_ms": 40.0,
        },
        {
            "group": "wrong_part",
            "handoff_crop": True,
            "crop_correct": True,
            "part_correct": False,
            "part_abstained": True,
            "unsupported_part_emitted": False,
            "latency_ms": 50.0,
        },
    ]

    summary = eval_script.summarize_predictions(rows)

    assert summary["crop_accuracy"] == 1.0
    assert summary["negative_false_accept_rate"] == 0.5
    assert summary["part_non_unknown_precision"] == 1.0
    assert summary["part_recall"] == 0.5
    assert summary["unsupported_part_emissions"] == 1
    assert summary["wrong_part_rejection_rate"] == 1.0
    assert summary["mean_latency_ms"] == 30.0


def test_sample_from_analysis_marks_negative_false_accept_and_unsupported_part():
    item = {
        "image_path": Path("sample.png"),
        "group": "off_crop",
        "expected_crop": "unknown",
        "expected_part": "unknown",
        "expected_handoff": False,
    }
    analysis = RouterAnalysisResult(
        status="ok",
        primary_detection=RouterDetection(
            crop="tomato",
            part="root",
            crop_confidence=0.91,
            part_confidence=0.8,
        ),
        detections_count=1,
    )

    sample = eval_script.sample_from_analysis(
        item=item,
        analysis=analysis,
        latency_ms=12.3,
        config={"router": {"crop_mapping": {"tomato": {"parts": ["leaf", "fruit"]}}}},
    )

    assert sample["handoff_crop"] is True
    assert sample["unsupported_part_emitted"] is True
    assert sample["predicted_crop"] == "tomato"
    assert sample["predicted_part"] == "root"
    assert sample["top_crop_candidates"][0]["crop"] == "tomato"


def test_sample_from_analysis_preserves_router_diagnostics():
    item = {
        "image_path": Path("sample.png"),
        "group": "id",
        "expected_crop": "tomato",
        "expected_part": "leaf",
        "expected_handoff": True,
    }
    analysis = RouterAnalysisResult(
        status="ok",
        message="Part abstained for crop=tomato",
        detections=[
            RouterDetection(
                crop="tomato",
                part="unknown",
                crop_confidence=0.91,
                part_confidence=0.50,
                quality_score=0.77,
                metadata={
                    "raw_part_label": "fruit",
                    "raw_part_confidence": 0.41,
                    "raw_part_second_confidence": 0.20,
                    "part_unknown_confidence": 0.46,
                    "raw_part_margin": 0.21,
                    "part_rejection_reason": "unknown_confidence (0.4600) >= confidence (0.4100)",
                },
            ),
            RouterDetection(
                crop="pepper",
                part="leaf",
                crop_confidence=0.81,
                part_confidence=0.70,
                quality_score=0.65,
            ),
        ],
        detections_count=2,
    )

    sample = eval_script.sample_from_analysis(
        item=item,
        analysis=analysis,
        latency_ms=14.0,
        config={"router": {"crop_mapping": {"tomato": {"parts": ["leaf", "fruit"]}}}},
    )

    assert sample["top_crop_candidates"] == [
        {
            "crop": "tomato",
            "part": "unknown",
            "crop_confidence": 0.91,
            "part_confidence": 0.5,
            "quality_score": 0.77,
        },
        {
            "crop": "pepper",
            "part": "leaf",
            "crop_confidence": 0.81,
            "part_confidence": 0.7,
            "quality_score": 0.65,
        },
    ]
    assert sample["crop_confidence_margin"] == 0.1
    assert sample["raw_part_label"] == "fruit"
    assert sample["raw_part_confidence"] == 0.41
    assert sample["part_unknown_confidence"] == 0.46
    assert sample["part_rejection_reason"] == "unknown_confidence (0.4600) >= confidence (0.4100)"
    assert sample["detection_count"] == 2
    assert sample["latency_ms"] == 14.0


def test_threshold_sweep_prefers_low_false_accept_then_accuracy():
    rows = [
        {"group": "id", "handoff_crop": True, "crop_correct": True, "part_correct": True, "part_abstained": False, "crop_confidence": 0.9},
        {"group": "id", "handoff_crop": True, "crop_correct": True, "part_correct": True, "part_abstained": False, "crop_confidence": 0.7},
        {"group": "off_crop", "handoff_crop": True, "crop_correct": False, "part_correct": False, "part_abstained": False, "crop_confidence": 0.6},
    ]

    sweep = eval_script.threshold_sweep(rows, thresholds=[0.0, 0.65])

    assert sweep["recommended"]["threshold"] == 0.65
    assert sweep["recommended"]["negative_false_accept_rate"] == 0.0
    assert sweep["recommended"]["crop_accuracy"] == 1.0
