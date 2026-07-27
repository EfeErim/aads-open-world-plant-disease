from pathlib import Path

from scripts.optimize_ood_oe_datasets import analyze_oe_dataset, analyze_ood_dataset, main


def _write_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_dataset_analysis_handles_missing_roots(tmp_path: Path):
    ood = analyze_ood_dataset(tmp_path / "missing_ood")
    oe = analyze_oe_dataset(tmp_path / "missing_oe")

    assert ood["total_files"] == 0
    assert ood["folders_analyzed"] == {}
    assert oe["total_files"] == 0
    assert oe["slice_distribution"] == {}


def test_oe_analysis_reports_duplicates_and_slice_imbalance(tmp_path: Path):
    root = tmp_path / "oe"
    _write_file(root / "tomato__leaf" / "slice_a" / "a.jpg", b"same")
    _write_file(root / "tomato__leaf" / "slice_a" / "b.jpg", b"same")
    _write_file(root / "tomato__leaf" / "slice_a" / "c.jpg", b"unique-a")
    _write_file(root / "tomato__leaf" / "slice_b" / "d.jpg", b"unique-b")

    analysis = analyze_oe_dataset(root)

    assert analysis["total_files"] == 4
    assert analysis["duplicates_found"] == 1
    assert any(item["type"] == "duplicate_files" for item in analysis["recommendations"])
    assert any(item["type"] == "slice_imbalance" for item in analysis["recommendations"])


def test_main_uses_supplied_root_and_writes_report(tmp_path: Path):
    _write_file(tmp_path / "data" / "ood_dataset" / "final" / "plant" / "a.jpg", b"same")
    _write_file(tmp_path / "data" / "ood_dataset" / "final" / "plant" / "b.jpg", b"same")

    report = main(tmp_path)

    assert report["summary"]["ood_duplicates"] == 1
    assert (tmp_path / "outputs" / "ood_oe_optimization_report.json").exists()
