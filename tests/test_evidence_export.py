import json
from pathlib import Path

from aads_public.evidence import EXPECTED_MANIFEST_SHA256, EXPECTED_RUN_ID
from aads_public.evidence_export import export_public_evidence


def test_export_removes_private_paths_and_recomputes_decisions(tmp_path: Path) -> None:
    source_summary = tmp_path / "summary.json"
    source_run = tmp_path / "run.json"
    destination_summary = tmp_path / "public-summary.json"
    destination_rows = tmp_path / "public-rows.json"
    source_summary.write_text(
        json.dumps(
            {
                "created_at": EXPECTED_RUN_ID,
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                "runner_exit_code": 0,
            }
        ),
        encoding="utf-8",
    )
    source_run.write_text(
        json.dumps(
            {
                "device": "cuda",
                "rows": [
                    {
                        "image_id": "demo_001",
                        "source": "private/image.jpg",
                        "resolved_image": "/content/private/image.jpg",
                        "expected_target": "tomato__leaf",
                        "expected_class": "late_blight",
                        "actual_status": "success",
                        "predicted_crop": "tomato",
                        "predicted_part": "leaf",
                        "predicted_disease": "late_blight",
                        "pass_fail": "pass",
                        "notes": "private note",
                    },
                    {
                        "image_id": "demo_002",
                        "source": "private/nonplant.jpg",
                        "expected_target": "non_plant",
                        "expected_class": "non_plant",
                        "actual_status": "router_uncertain",
                        "predicted_disease": "",
                        "pass_fail": "pass",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summary, rows = export_public_evidence(
        source_summary_path=source_summary,
        source_rows_path=source_run,
        destination_summary_path=destination_summary,
        destination_rows_path=destination_rows,
    )

    serialized = destination_rows.read_text(encoding="utf-8")
    assert "private/image.jpg" not in serialized
    assert "/content/" not in serialized
    assert "private note" not in serialized
    assert summary["total"] == 2
    assert summary["answered"] == 1
    assert summary["reviewed_or_abstained"] == 1
    assert rows["rows"][1]["expected_outcome"] == "review"
    assert rows["rows"][1]["actual_outcome"] == "review"
