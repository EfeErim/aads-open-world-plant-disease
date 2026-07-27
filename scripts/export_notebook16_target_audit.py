"""Export a full missed-wrong audit table for a Notebook 16 target."""

from __future__ import annotations

import argparse
import json
import re
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.notebook16_failure_analysis import (  # noqa: E402
    DEFAULT_FOCUS_TARGET,
    DEFAULT_SOURCE_REPORT,
    DEFAULT_TARGET_AUDIT_CSV_OUTPUT,
    DEFAULT_TARGET_AUDIT_JSON_OUTPUT,
    DEFAULT_TARGET_AUDIT_MARKDOWN_OUTPUT,
    build_target_missed_wrong_audit,
    load_notebook16_report,
    write_target_missed_wrong_audit_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_SOURCE_REPORT), help="Notebook 16 multi-target report JSON.")
    parser.add_argument("--target-id", default=DEFAULT_FOCUS_TARGET, help="Target id to audit, such as tomato__leaf.")
    parser.add_argument("--json-output", default=str(DEFAULT_TARGET_AUDIT_JSON_OUTPUT), help="Audit JSON output.")
    parser.add_argument("--csv-output", default=str(DEFAULT_TARGET_AUDIT_CSV_OUTPUT), help="Audit CSV output.")
    parser.add_argument("--markdown-output", default=str(DEFAULT_TARGET_AUDIT_MARKDOWN_OUTPUT), help="Audit report output.")
    parser.add_argument(
        "--contact-sheet-output",
        default=".runtime_tmp/tomato_leaf_missed_wrong_contact_sheet.jpg",
        help="Optional contact-sheet image output. Pass an empty string to skip.",
    )
    parser.add_argument(
        "--packet-output-dir",
        default=".runtime_tmp/tomato_leaf_missed_wrong_packets",
        help="Optional confusion-pair packet directory. Pass an empty string to skip.",
    )
    parser.add_argument(
        "--review-index-output",
        default=".runtime_tmp/tomato_leaf_missed_wrong_packets/index.html",
        help="Optional static HTML review index. Pass an empty string to skip.",
    )
    parser.add_argument("--repo-root", default=".", help="Repo root for local file existence checks.")
    return parser


def write_contact_sheet(
    rows: list[dict],
    output: str | Path,
    *,
    repo_root: str | Path = ".",
    title: str = "",
) -> int:
    """Write a thumbnail contact sheet for local audit rows."""
    if not output:
        return 0
    from PIL import Image, ImageDraw

    repo_root_path = Path(repo_root)
    thumbs = []
    for row in rows:
        local_path = row.get("local_path")
        if not local_path:
            continue
        image_path = repo_root_path / str(local_path)
        if not image_path.is_file():
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue
        image.thumbnail((220, 160))
        thumbs.append((row, image.copy()))

    cols = 4
    cell_width = 320
    cell_height = 245
    title_height = 42 if title else 0
    row_count = max(1, (len(thumbs) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cell_width, title_height + row_count * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((8, 12), title, fill=(0, 0, 0))
    for index, (row, image) in enumerate(thumbs):
        x = (index % cols) * cell_width
        y = title_height + (index // cols) * cell_height
        sheet.paste(image, (x + (220 - image.width) // 2, y + 5))
        label = (
            f"#{row.get('rank')} GT: {row.get('expected_label')}\n"
            f"Pred: {row.get('diagnosis')}\n"
            f"conf: {float(row.get('full_confidence') or 0.0):.4f}"
        )
        draw.multiline_text((x + 8, y + 170), label, fill=(0, 0, 0), spacing=2)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)
    return len(thumbs)


def write_grouped_packets(payload: dict, output_dir: str | Path, *, repo_root: str | Path = ".") -> dict:
    """Write per-confusion CSV and contact-sheet packets for review."""
    if not output_dir:
        return {"packet_dir": "", "packet_count": 0, "rows_written": 0}
    from src.shared.csv_utils import write_csv_rows_with_order

    packet_root = Path(output_dir)
    packet_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = {}
    for row in payload.get("rows", []):
        grouped.setdefault(str(row.get("confusion_pair") or "unknown"), []).append(row)

    packets = []
    for index, (confusion_pair, rows) in enumerate(
        sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])),
        start=1,
    ):
        slug = _slugify_confusion_pair(confusion_pair)
        packet_dir = packet_root / f"{index:02d}_{slug}"
        packet_dir.mkdir(parents=True, exist_ok=True)
        csv_path = packet_dir / "review_rows.csv"
        sheet_path = packet_dir / "contact_sheet.jpg"
        write_csv_rows_with_order(
            csv_path,
            rows,
            preferred_headers=(
                "rank",
                "expected_label",
                "diagnosis",
                "full_confidence",
                "local_path",
                "review_decision",
                "review_notes",
            ),
            encoding="utf-8-sig",
        )
        contact_sheet_count = write_contact_sheet(
            rows,
            sheet_path,
            repo_root=repo_root,
            title=f"{confusion_pair} ({len(rows)} rows)",
        )
        (packet_dir / "README.md").write_text(
            "\n".join(
                [
                    f"# `{confusion_pair}`",
                    "",
                    f"- rows: `{len(rows)}`",
                    f"- contact sheet items: `{contact_sheet_count}`",
                    "- Fill `review_decision` only after visual/domain review.",
                    "- Suggested decisions: `keep`, `remove_from_test`, `relabel:<class>`.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        packets.append(
            {
                "confusion_pair": confusion_pair,
                "rows": len(rows),
                "packet_dir": str(packet_dir),
                "csv": str(csv_path),
                "contact_sheet": str(sheet_path),
            }
        )
    summary = {
        "packet_dir": str(packet_root),
        "packet_count": len(packets),
        "rows_written": sum(int(packet["rows"]) for packet in packets),
        "packets": packets,
    }
    (packet_root / "packet_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (packet_root / "README.md").write_text(
        "\n".join(
            [
                "# Notebook 16 Target Audit Packets",
                "",
                f"Target: `{payload.get('target_id')}`",
                f"Missed-wrong rows: `{payload.get('missed_wrong_count')}`",
                f"Confusion-pair packets: `{len(packets)}`",
                "",
                "## Review Workflow",
                "",
                "1. Open `index.html` and review packet contact sheets in descending confusion count.",
                "2. Fill `review_decision` in the relevant `review_rows.csv` files or in the full audit CSV.",
                "3. Use only these decisions unless the apply script is intentionally extended:",
                "   - `keep`: keep the sample and label unchanged.",
                "   - `remove_from_test`: move the sample to quarantine, not delete it.",
                "   - `relabel:<class>`: move the sample to `test/<class>/`.",
                "4. Run `scripts/apply_notebook16_target_audit_decisions.py --packet-dir .runtime_tmp/tomato_leaf_missed_wrong_packets --require-reviewed` before applying.",
                "5. Use `--apply` only after the dry-run summary is correct.",
                "",
                "Do not promote runtime review-gate policy from these packets alone; rerun Notebook 16 after any data changes.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def write_review_index(payload: dict, packet_summary: dict, output: str | Path) -> str:
    """Write a static HTML index for grouped review packets."""
    if not output:
        return ""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    packet_root = Path(str(packet_summary.get("packet_dir") or output_path.parent))
    cards = []
    for packet in packet_summary.get("packets", []):
        contact_sheet = Path(str(packet.get("contact_sheet") or ""))
        csv_path = Path(str(packet.get("csv") or ""))
        try:
            image_ref = contact_sheet.relative_to(packet_root).as_posix()
        except ValueError:
            image_ref = contact_sheet.as_posix()
        try:
            csv_ref = csv_path.relative_to(packet_root).as_posix()
        except ValueError:
            csv_ref = csv_path.as_posix()
        cards.append(
            "\n".join(
                [
                    '<section class="packet">',
                    f"<h2>{escape(str(packet.get('confusion_pair') or 'unknown'))}</h2>",
                    f"<p><strong>Rows:</strong> {int(packet.get('rows') or 0)}</p>",
                    f'<p><a href="{escape(csv_ref)}">review_rows.csv</a></p>',
                    f'<img src="{escape(image_ref)}" alt="{escape(str(packet.get("confusion_pair") or "packet"))} contact sheet">',
                    "</section>",
                ]
            )
        )
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    header {{ margin-bottom: 24px; }}
    .summary {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 16px 0; }}
    .summary div {{ border: 1px solid #d7dde5; padding: 10px 12px; border-radius: 6px; }}
    .packet {{ border-top: 1px solid #d7dde5; padding: 18px 0 26px; }}
    .packet h2 {{ font-size: 18px; margin: 0 0 8px; }}
    .packet img {{ max-width: 100%; height: auto; border: 1px solid #d7dde5; }}
    a {{ color: #0f5ea8; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>Static review index for grouped Notebook 16 missed-wrong audit packets.</p>
    <div class="summary">
      <div><strong>Target</strong><br>{target}</div>
      <div><strong>Wrong</strong><br>{wrong}</div>
      <div><strong>Missed wrong</strong><br>{missed}</div>
      <div><strong>Packets</strong><br>{packets}</div>
      <div><strong>Rows in packets</strong><br>{packet_rows}</div>
    </div>
  </header>
  {cards}
</body>
</html>
""".format(
        title=escape(f"{payload.get('target_id')} Missed-Wrong Review Packets"),
        target=escape(str(payload.get("target_id") or "")),
        wrong=int(payload.get("wrong_count") or 0),
        missed=int(payload.get("missed_wrong_count") or 0),
        packets=int(packet_summary.get("packet_count") or 0),
        packet_rows=int(packet_summary.get("rows_written") or 0),
        cards="\n".join(cards),
    )
    output_path.write_text(html, encoding="utf-8")
    return str(output_path)


def _slugify_confusion_pair(value: str) -> str:
    text = value.replace(" -> ", "__to__")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text[:96] or "unknown"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    rows = load_notebook16_report(input_path)
    payload = build_target_missed_wrong_audit(
        rows,
        target_id=args.target_id,
        source_report=input_path,
        repo_root=args.repo_root,
    )
    write_target_missed_wrong_audit_outputs(
        payload,
        json_output=args.json_output,
        csv_output=args.csv_output,
        markdown_output=args.markdown_output,
    )
    contact_sheet_count = write_contact_sheet(
        payload.get("rows", []),
        args.contact_sheet_output,
        repo_root=args.repo_root,
        title=f"{payload['target_id']} missed-wrong audit ({payload['missed_wrong_count']} rows)",
    )
    packet_summary = write_grouped_packets(payload, args.packet_output_dir, repo_root=args.repo_root)
    review_index_output = write_review_index(payload, packet_summary, args.review_index_output)
    print(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "target_id": payload["target_id"],
                "wrong_count": payload["wrong_count"],
                "missed_wrong_count": payload["missed_wrong_count"],
                "local_available_count": payload["local_available_count"],
                "json_output": str(args.json_output),
                "csv_output": str(args.csv_output),
                "markdown_output": str(args.markdown_output),
                "contact_sheet_output": str(args.contact_sheet_output),
                "contact_sheet_count": contact_sheet_count,
                "packet_output_dir": str(args.packet_output_dir),
                "packet_count": packet_summary.get("packet_count", 0),
                "packet_rows_written": packet_summary.get("rows_written", 0),
                "review_index_output": review_index_output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
