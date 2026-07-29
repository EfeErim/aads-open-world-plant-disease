import json
from pathlib import Path

import pytest

from scripts.replay_public_demo import build_replay_payload, format_replay, load_demo_evidence


def test_public_demo_replay_matches_checked_in_evidence():
    summary, rows = load_demo_evidence()

    payload = build_replay_payload(summary, rows)

    assert payload["fresh_inference"] is False
    assert payload["summary"] == {
        "passed": 48,
        "total": 48,
        "answered": 36,
        "reviewed_or_abstained": 12,
        "negative_false_accepts": 0,
        "wrong_part_disease_labels": 0,
        "production_ready": False,
    }
    assert [row["actual_outcome"] for row in payload["examples"]] == ["answer", "review", "review"]
    assert "production safety gate: not passed" in format_replay(payload)


def test_public_demo_replay_rejects_mismatched_row_count(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    rows_path = tmp_path / "rows.json"
    summary_path.write_text(json.dumps({"run_id": "run", "total": 2, "passed": 1}), encoding="utf-8")
    rows_path.write_text(
        json.dumps({"run_id": "run", "rows": [{"passed": True}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row count"):
        load_demo_evidence(summary_path, rows_path)
