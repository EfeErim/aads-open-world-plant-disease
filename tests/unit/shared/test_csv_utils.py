import zipfile
from pathlib import Path

from src.shared.csv_utils import read_csv_preview, read_csv_rows, read_csv_rows_from_source


def test_read_csv_rows_reads_plain_file(tmp_path: Path):
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    rows = read_csv_rows(csv_path)

    assert rows == [{"a": "1", "b": "2"}]


def test_read_csv_rows_from_source_reads_zip_member(tmp_path: Path):
    zip_path = tmp_path / "rows.zip"
    payload = "a,b\n3,4\n"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("run/artifacts/data_prep_artifacts/example.csv", payload)

    rows = read_csv_rows_from_source(zip_path, zip_member_suffix="example.csv")

    assert rows == [{"a": "3", "b": "4"}]


def test_read_csv_preview_returns_count_and_selected_fields(tmp_path: Path):
    csv_path = tmp_path / "preview.csv"
    csv_path.write_text("a,b,c\n1,2,\n3,,4\n", encoding="utf-8-sig")

    row_count, preview = read_csv_preview(csv_path, max_rows=1, fields=("a", "c"))

    assert row_count == 2
    assert preview == [{"a": "1"}]