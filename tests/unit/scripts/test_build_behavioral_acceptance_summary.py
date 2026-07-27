from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "build_behavioral_acceptance_summary.py"
SPEC = importlib.util.spec_from_file_location("build_behavioral_acceptance_summary", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tracked_behavioral_summary_is_rebuilt_from_public_target_records() -> None:
    repo_root = Path(__file__).parents[3]
    expected = MODULE.build_summary(repo_root / "evidence")
    tracked = json.loads(
        (repo_root / "evidence" / "latest_behavioral_acceptance_summary.json").read_text(encoding="utf-8")
    )

    assert tracked == expected
    assert tracked["passed_targets"] == 0
    assert tracked["total_targets"] == 8
    assert all(Path(repo_root, value["source_path"]).is_file() for value in tracked["targets"].values())


def test_behavioral_summary_rejects_misattributed_target_record(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[3]
    evidence_root = tmp_path / "evidence"
    source_root = repo_root / "evidence" / "behavioral_acceptance" / "targets"
    target_root = evidence_root / "behavioral_acceptance" / "targets"
    target_root.mkdir(parents=True)
    for source_path in source_root.glob("*.json"):
        target_root.joinpath(source_path.name).write_bytes(source_path.read_bytes())
    corrupt_path = target_root / "grape__leaf.json"
    corrupt = json.loads(corrupt_path.read_text(encoding="utf-8"))
    corrupt["context"]["part_name"] = "fruit"
    corrupt_path.write_text(json.dumps(corrupt), encoding="utf-8")

    with pytest.raises(ValueError, match="target mismatch"):
        MODULE.build_summary(evidence_root)
