import hashlib
import json
from pathlib import Path

import pytest

from aads_public.cli import _default_evidence
from aads_public.evidence import load_acceptance_report

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "controlled_demo_summary.json"
ROWS = ROOT / "evidence" / "controlled_demo_rows.json"


def _copy_evidence(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    summary = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    summary_path = tmp_path / EVIDENCE.name
    rows_path = tmp_path / ROWS.name
    rows_path.write_bytes(ROWS.read_bytes())
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path, rows_path, summary, rows


def test_frozen_controlled_demo_validates_all_rows_and_identity() -> None:
    report = load_acceptance_report(EVIDENCE)
    assert report.controlled_demo_passed
    assert report.identity_verified
    assert report.rows_verified
    assert (report.total, report.answered, report.reviewed) == (48, 36, 12)
    assert report.production_ready is False


def test_packaged_and_repository_evidence_are_identical() -> None:
    package_root = _default_evidence().parent
    assert _default_evidence().read_bytes() == EVIDENCE.read_bytes()
    assert (package_root / ROWS.name).read_bytes() == ROWS.read_bytes()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("manifest_sha256", "0" * 64, "manifest identity"),
        ("run_id", "different-run", "run_id"),
        ("source_surface", "stress", "source_surface"),
        ("runtime", "cpu", "CUDA runtime"),
        ("runner_exit_code", 1, "did not exit successfully"),
        ("source_summary_sha256", "0" * 64, "source summary identity"),
        ("source_rows_sha256", "0" * 64, "source row identity"),
        ("production_ready", True, "production_ready"),
    ],
)
def test_evidence_rejects_tampered_identity(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    summary_path, rows_path, summary, _ = _copy_evidence(tmp_path)
    summary[field] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_acceptance_report(summary_path, rows_path)


def test_evidence_requires_explicit_false_production_status(tmp_path: Path) -> None:
    summary_path, rows_path, summary, _ = _copy_evidence(tmp_path)
    summary.pop("production_ready")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="production_ready"):
        load_acceptance_report(summary_path, rows_path)


def test_evidence_rejects_row_file_digest_mismatch(tmp_path: Path) -> None:
    summary_path, rows_path, _, rows = _copy_evidence(tmp_path)
    rows["rows"][0]["passed"] = False
    rows_path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        load_acceptance_report(summary_path, rows_path)


def test_evidence_rejects_coordinated_row_and_digest_tampering(tmp_path: Path) -> None:
    summary_path, rows_path, summary, rows = _copy_evidence(tmp_path)
    rows["rows"][0]["passed"] = False
    rows_content = json.dumps(rows).encode()
    rows_path.write_bytes(rows_content)
    summary["rows_file_sha256"] = hashlib.sha256(rows_content).hexdigest()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen public snapshot"):
        load_acceptance_report(summary_path, rows_path)


def test_evidence_rejects_coordinated_class_claim_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path, rows_path, summary, rows = _copy_evidence(tmp_path)
    answer = next(row for row in rows["rows"] if row["actual_outcome"] == "answer")
    answer["predicted_class"] = "different_class"
    rows_content = json.dumps(rows).encode()
    rows_path.write_bytes(rows_content)
    digest = hashlib.sha256(rows_content).hexdigest()
    summary["rows_file_sha256"] = digest
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr("aads_public.evidence.EXPECTED_PUBLIC_ROWS_SHA256", digest)

    with pytest.raises(ValueError, match="class_correct is inconsistent"):
        load_acceptance_report(summary_path, rows_path)


def test_evidence_rejects_empty_per_target_even_when_totals_match(tmp_path: Path) -> None:
    summary_path, rows_path, summary, _ = _copy_evidence(tmp_path)
    summary["per_target"] = {}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="per_target"):
        load_acceptance_report(summary_path, rows_path)


def test_evidence_rejects_hidden_row_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path, rows_path, summary, rows = _copy_evidence(tmp_path)
    rows["rows"][0]["source_path"] = "private/image.jpg"
    rows_content = json.dumps(rows).encode()
    rows_path.write_bytes(rows_content)
    digest = hashlib.sha256(rows_content).hexdigest()
    summary["rows_file_sha256"] = digest
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr("aads_public.evidence.EXPECTED_PUBLIC_ROWS_SHA256", digest)

    with pytest.raises(ValueError, match="unexpected fields"):
        load_acceptance_report(summary_path, rows_path)
