import csv
from pathlib import Path

from scripts.build_m2_problem_only_manifest import build_problem_only_manifest
from scripts.run_demo_checklist import parse_manifest_rows


def test_build_problem_only_manifest_from_router_audit(tmp_path: Path):
    audit = tmp_path / "hard_example_audit.csv"
    audit.write_text(
        "\n".join(
            [
                "image_id,source,expected_target,expected_crop,expected_part,expected_class,expected_behavior,"
                "notes,resolved_image,reconcile_reason,review_decision,review_notes",
                (
                    "demo_145,staged_external:docs/demo_assets/m2_full_image_set/images/demo_145.jpg,"
                    "apricot__fruit,apricot,fruit,kayisi_meyve,old behavior,old notes,"
                    "/content/bitirmeprojesi/docs/demo_assets/m2_full_image_set/images/demo_145.jpg,"
                    "prototype_evidence_weak,add_prototype_positive,same-target positive"
                ),
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "problem_only.csv"

    summary = build_problem_only_manifest(audit_csv=audit, output=output)

    assert summary["row_count"] == 1
    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["image_id"] == "demo_145"
    assert rows[0]["expected_behavior"] == "router/prototype hard example; disease answer or review expected"
    assert "prototype_evidence_weak" in rows[0]["notes"]
    parsed = parse_manifest_rows(output)
    assert parsed[0].expected_target == "apricot__fruit"
    assert parsed[0].expected_class == "kayisi_meyve"
