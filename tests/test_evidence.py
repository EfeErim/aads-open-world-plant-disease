import json
from pathlib import Path

import pytest

from aads_public.cli import _default_evidence
from aads_public.evidence import load_acceptance_report

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "controlled_demo_summary.json"


def test_frozen_controlled_demo_passes() -> None:
    report = load_acceptance_report(EVIDENCE)
    assert report.controlled_demo_passed
    assert (report.total, report.answered, report.reviewed) == (48, 36, 12)
    assert report.production_ready is False


def test_packaged_and_repository_evidence_are_identical() -> None:
    assert _default_evidence().read_bytes() == EVIDENCE.read_bytes()


def test_evidence_rejects_inconsistent_totals(tmp_path: Path) -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    payload["passed"] = 47
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="passed \\+ failed"):
        load_acceptance_report(path)


def test_evidence_rejects_production_claim(tmp_path: Path) -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    payload["production_ready"] = True
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="production readiness"):
        load_acceptance_report(path)
