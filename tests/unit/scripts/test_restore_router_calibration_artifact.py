import json
from pathlib import Path

import pytest

from scripts.restore_router_calibration_artifact import restore_latest_router_calibration


def test_restore_latest_router_calibration_copies_newest_published_run(tmp_path: Path):
    published_root = tmp_path / "runs" / "_index" / "router_calibration"
    older = published_root / "20260613T010000Z" / "router_calibration.json"
    newer = published_root / "20260614T010000Z" / "router_calibration.json"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_text(json.dumps({"run": "older"}), encoding="utf-8")
    newer.write_text(json.dumps({"run": "newer"}), encoding="utf-8")
    output = tmp_path / ".runtime_tmp" / "router_calibration.json"

    summary = restore_latest_router_calibration(published_root, output)

    assert summary["published_run"] == "20260614T010000Z"
    assert summary["source"] == str(newer)
    assert summary["output"] == str(output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"run": "newer"}


def test_restore_latest_router_calibration_reports_missing_published_runs(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No published router_calibration.json"):
        restore_latest_router_calibration(tmp_path / "missing", tmp_path / "out.json")
