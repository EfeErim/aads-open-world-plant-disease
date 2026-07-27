from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_code_organization import build_report, categorize_path, main


def test_categorize_path_maps_repo_surfaces() -> None:
    root = Path("repo").resolve()

    assert categorize_path(root / "src" / "workflows" / "training.py", root) == "workflow"
    assert categorize_path(root / "src" / "shared" / "json_utils.py", root) == "shared"
    assert categorize_path(root / "scripts" / "notebook_cells" / "nb2_cell09_training.py", root) == "notebook_cell"
    assert categorize_path(root / "scripts" / "validate_config_schema.py", root) == "validation"
    assert categorize_path(root / "tests" / "unit" / "test_example.py", root) == "test"


def test_build_report_fails_when_src_imports_scripts(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    scripts_dir = tmp_path / "scripts"
    tests_dir = tmp_path / "tests"
    src_dir.mkdir()
    scripts_dir.mkdir()
    tests_dir.mkdir()
    (src_dir / "bad.py").write_text("import scripts.some_helper\n", encoding="utf-8")
    (scripts_dir / "some_helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tests_dir / "test_placeholder.py").write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")

    report = build_report(tmp_path)

    assert report["status"] == "fail"
    assert report["error_count"] == 1
    assert report["findings"][0]["path"] == "src/bad.py"
    assert "must not import `scripts`" in report["findings"][0]["message"]


def test_build_report_passes_and_warns_for_long_script(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    scripts_dir = tmp_path / "scripts"
    src_dir.mkdir()
    scripts_dir.mkdir()
    (src_dir / "ok.py").write_text("VALUE = 1\n", encoding="utf-8")
    (scripts_dir / "large_wrapper.py").write_text("\n".join(["VALUE = 1"] * 801), encoding="utf-8")

    report = build_report(tmp_path)

    assert report["status"] == "pass"
    assert report["error_count"] == 0
    assert report["warning_count"] == 1
    assert report["category_counts"]["cli"] == 1
    assert report["size_buckets"]["over_500"] == 1


def test_main_writes_json_report(tmp_path: Path, capsys) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "ok.py").write_text("VALUE = 1\n", encoding="utf-8")
    output_path = tmp_path / "audit.json"

    exit_code = main(["--root", str(tmp_path), "--output", str(output_path)])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert "code_organization status=pass" in capsys.readouterr().out
