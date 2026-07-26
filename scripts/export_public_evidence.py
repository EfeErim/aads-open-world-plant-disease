"""Export sanitized row-level evidence from an archived controlled-demo run."""

from __future__ import annotations

import argparse
from pathlib import Path

from aads_public.evidence_export import export_public_evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_summary", type=Path)
    parser.add_argument("source_rows", type=Path)
    parser.add_argument("destination_summary", type=Path)
    parser.add_argument("destination_rows", type=Path)
    args = parser.parse_args()
    summary, rows = export_public_evidence(
        source_summary_path=args.source_summary,
        source_rows_path=args.source_rows,
        destination_summary_path=args.destination_summary,
        destination_rows_path=args.destination_rows,
    )
    print(f"Exported {summary['passed']}/{summary['total']} accepted rows; sanitized rows={len(rows['rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
