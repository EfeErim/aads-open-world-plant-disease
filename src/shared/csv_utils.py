from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def read_csv_rows(path: Path, *, encoding: str = "utf-8-sig") -> List[Dict[str, str]]:
    p = Path(path)
    if not p.is_file() or p.stat().st_size <= 0:
        return []
    with p.open("r", encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_rows_from_source(
    source_path: Path,
    *,
    zip_member_suffix: Optional[str] = None,
    encoding: str = "utf-8-sig",
) -> List[Dict[str, str]]:
    source_path = Path(source_path)
    if source_path.suffix.lower() == ".zip":
        return _read_csv_from_zip(source_path, zip_member_suffix, encoding)
    return read_csv_rows(source_path, encoding=encoding)


def _read_csv_from_zip(source_path: Path, zip_member_suffix: str | None, encoding: str) -> List[Dict[str, str]]:
    if not zip_member_suffix:
        raise ValueError("zip_member_suffix is required when reading rows from a zip archive")
    with zipfile.ZipFile(source_path) as archive:
        members = [name for name in archive.namelist() if name.endswith(f"/{zip_member_suffix}")]
        if not members:
            raise FileNotFoundError(f"No {zip_member_suffix} found in zip: {source_path}")
        if len(members) > 1:
            raise RuntimeError(f"Zip contains multiple {zip_member_suffix} files: {members}")
        payload = archive.read(members[0]).decode(encoding)
    return list(csv.DictReader(payload.splitlines()))


def read_csv_preview(
    path: Path,
    *,
    max_rows: int,
    fields: Sequence[str],
    encoding: str = "utf-8-sig",
) -> tuple[int, List[Dict[str, Any]]]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return 0, []
    preview: List[Dict[str, Any]] = []
    row_count = 0
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            if len(preview) >= max_rows:
                continue
            selected = _select_nonempty_fields(row, fields)
            if selected:
                preview.append(selected)
    return row_count, preview


def _select_nonempty_fields(row: Dict[str, str], fields: Sequence[str]) -> Dict[str, str]:
    """Return a dict of field->value for fields where the value is non-empty after strip()."""
    return {
        field: row.get(field, "")
        for field in fields
        if str(row.get(field, "")).strip()
    }


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[Any]], *, encoding: str = "utf-8") -> Path:
    """Write rows to a CSV file with explicit headers.

    Args:
        path: Output file path (parent directories created if needed).
        headers: Column names.
        rows: Sequence of rows, each with values matching headers.
        encoding: Text encoding (default utf-8).

    Returns:
        The resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(headers))
        for row in rows:
            writer.writerow(list(row))
    return path


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], *, encoding: str = "utf-8") -> Path:
    """Write dict rows to a CSV file with auto-discovered headers.

    Derives headers from all keys across all rows, sorts them, and writes with DictWriter.

    Args:
        path: Output file path (parent directories created if needed).
        rows: Sequence of dict rows.
        encoding: Text encoding (default utf-8).

    Returns:
        The resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding=encoding)
        return path
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_csv_rows_with_order(
    path: Path, rows: Sequence[Dict[str, Any]], *, preferred_headers: Sequence[str], encoding: str = "utf-8"
) -> Path:
    """Write dict rows preserving preferred column order with fallback to sorted extras.

    Args:
        path: Output file path (parent directories created if needed).
        rows: Sequence of dict rows.
        preferred_headers: Column names to prioritize (in order).
        encoding: Text encoding (default utf-8).

    Returns:
        The resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized_rows = [dict(row) for row in rows]
    preferred_header_set = set(preferred_headers)
    extra_headers = sorted(
        {
            str(key)
            for row in materialized_rows
            for key in row.keys()
            if str(key) not in preferred_header_set
        }
    )
    headers = [*preferred_headers, *extra_headers]
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in materialized_rows:
            writer.writerow({header: row.get(header, "") for header in headers})
    return path
