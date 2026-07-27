"""Validate the two pinned, public notebook contracts without Jupyter dependencies."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

from aads_public.evidence import load_acceptance_report

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"
RELEASE_TAG = "aads-public-demo-v1.1.1"
EXPECTED_NOTEBOOKS = frozenset({"continual_objective.ipynb", "evidence_snapshot.ipynb"})


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("nbformat") != 4 or not isinstance(payload.get("cells"), list):
        raise ValueError(f"{path.name} is not a valid notebook v4 document")
    return payload


def _saved_stdout(cell: dict) -> str:
    chunks: list[str] = []
    for output in cell.get("outputs") or []:
        if output.get("output_type") == "stream" and output.get("name") == "stdout":
            text = output.get("text") or []
            chunks.extend(text if isinstance(text, list) else [str(text)])
    return "".join(chunks)


def _code_cells(payload: dict) -> list[dict]:
    return [cell for cell in payload["cells"] if cell.get("cell_type") == "code"]


def _validate_pinning(path: Path, payload: dict) -> None:
    serialized = json.dumps(payload)
    if f".git@{RELEASE_TAG}" not in serialized:
        raise ValueError(f"{path.name} does not install the immutable public tag")
    if "bitirmeprojesi.git\"" in serialized or "/master/evidence/" in serialized:
        raise ValueError(f"{path.name} contains a mutable repository reference")


def _validate_evidence_output(payload: dict) -> None:
    report = load_acceptance_report(
        ROOT / "evidence" / "controlled_demo_summary.json",
        ROOT / "evidence" / "controlled_demo_rows.json",
    )
    expected = (
        "Manifest identity: PASS\n"
        f"Sanitized decisions: {report.passed}/{report.total} PASS\n"
        f"Disease answers: {report.answered}\n"
        f"Safe review/abstain: {report.reviewed}\n"
        f"Production ready: {report.production_ready}\n"
    )
    cells = _code_cells(payload)
    if len(cells) != 2 or _saved_stdout(cells[1]) != expected:
        raise ValueError("evidence_snapshot.ipynb saved output is stale")


def _validate_objective_output(payload: dict) -> None:
    cells = _code_cells(payload)
    if len(cells) != 2:
        raise ValueError("continual_objective.ipynb must contain install and walkthrough cells")
    source = "".join(cells[1].get("source") or [])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(source, "continual_objective.ipynb", "exec"), {})
    if _saved_stdout(cells[1]) != output.getvalue():
        raise ValueError("continual_objective.ipynb saved output is stale")


def main() -> int:
    paths = sorted(NOTEBOOK_ROOT.glob("*.ipynb"))
    if {path.name for path in paths} != EXPECTED_NOTEBOOKS:
        raise ValueError("public notebook set does not match the maintained two-notebook contract")
    for path in paths:
        payload = _load(path)
        _validate_pinning(path, payload)
        if path.name == "evidence_snapshot.ipynb":
            _validate_evidence_output(payload)
        else:
            _validate_objective_output(payload)
    print(f"Public notebooks: {len(paths)}/{len(EXPECTED_NOTEBOOKS)} valid, pinned, outputs current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
