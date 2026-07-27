"""Audit repo-wide code organization boundaries.

This guard keeps the shared platform model explicit: durable code lives under
`src/`, operational wrappers live under `scripts/`, and notebooks stay thin.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(".runtime_tmp/code_organization_audit.json")


@dataclass(frozen=True)
class FileRecord:
    path: str
    category: str
    line_count: int
    function_count: int = 0
    class_count: int = 0
    max_function_lines: int = 0
    traits: tuple[str, ...] = ()
    marker_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class Finding:
    severity: str
    path: str
    message: str


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_python_files(root: Path) -> Iterable[Path]:
    for base in ("src", "scripts", "tests"):
        base_path = root / base
        if not base_path.exists():
            continue
        yield from sorted(
            path
            for path in base_path.rglob("*.py")
            if "__pycache__" not in path.parts
        )


def iter_notebooks(root: Path) -> Iterable[Path]:
    notebook_root = root / "colab_notebooks"
    if not notebook_root.exists():
        return
    yield from sorted(notebook_root.glob("*.ipynb"))


def categorize_path(path: Path, root: Path) -> str:
    rel = _relative(path, root)
    if rel.startswith("src/core/"):
        return "core"
    if rel.startswith("src/shared/"):
        return "shared"
    if rel.startswith("src/workflows/"):
        return "workflow"
    if rel.startswith("src/pipeline/"):
        return "runtime"
    if rel.startswith(("src/data/", "src/ood/", "src/router/", "src/adapter/")):
        return "domain"
    if rel.startswith("src/training/services/"):
        return "service"
    if rel.startswith("src/training/"):
        return "domain"
    if rel.startswith("src/app/"):
        return "cli"
    if rel.startswith("src/"):
        return "domain"
    if rel.startswith("scripts/notebook_cells/"):
        return "notebook_cell"
    if rel.startswith("scripts/notebook_helpers/"):
        return "notebook_helper"
    if rel.startswith(("scripts/validate_", "scripts/check_", "scripts/monitor_")):
        return "validation"
    if rel.startswith("scripts/"):
        return "cli"
    if rel.startswith("tests/"):
        return "test"
    if rel.startswith("colab_notebooks/"):
        return "notebook"
    return "other"


def _line_count(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except UnicodeDecodeError:
        return len(path.read_text(encoding="utf-8-sig").splitlines())


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def _parse_python(path: Path) -> ast.AST | None:
    try:
        return ast.parse(_read_text(path), filename=str(path))
    except SyntaxError:
        return None


def _imported_top_level_modules(path: Path) -> list[str]:
    tree = _parse_python(path)
    if tree is None:
        return []
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module.split(".", 1)[0])
    return modules


def _python_metrics(path: Path) -> tuple[int, int, int]:
    tree = _parse_python(path)
    if tree is None:
        return 0, 0, 0
    function_nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    class_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    function_lengths = [
        int(getattr(node, "end_lineno", node.lineno) - node.lineno + 1)
        for node in function_nodes
        if hasattr(node, "lineno")
    ]
    return len(function_nodes), len(class_nodes), max(function_lengths, default=0)


def _marker_counts(path: Path) -> dict[str, int]:
    text = _read_text(path).lower()
    markers = ("legacy", "fallback", "compat", "workaround", "temporary", "duplicate", "deprecated")
    return {marker: count for marker in markers if (count := text.count(marker))}


def _script_traits(path: Path) -> tuple[str, ...]:
    text = _read_text(path).lower()
    imports = set(_imported_top_level_modules(path))
    checks = (
        ("argparse", "argparse" in imports or "argparse." in text),
        ("subprocess", "subprocess" in imports or "subprocess." in text),
        ("git", '"git"' in text or "'git'" in text or "git " in text),
        ("copy_move", "shutil" in imports or ".copy" in text or ".move" in text),
        ("publishing", "publish" in text or "push" in text),
        ("notebook", ".ipynb" in text or "notebook" in text),
        ("report_io", "json." in text or "write_text" in text or "csv." in text),
    )
    return tuple(name for name, present in checks if present)


def _audit_import_boundaries(path: Path, root: Path) -> list[Finding]:
    rel = _relative(path, root)
    imports = set(_imported_top_level_modules(path))
    findings: list[Finding] = []
    if rel.startswith("src/"):
        forbidden = sorted(imports.intersection({"scripts", "tests", "colab_notebooks"}))
        for module in forbidden:
            findings.append(
                Finding(
                    severity="error",
                    path=rel,
                    message=f"`src` modules must not import `{module}`; move shared logic into `src` instead.",
                )
            )
    if rel.startswith("src/shared/"):
        forbidden_shared = sorted(imports.intersection({"torch", "transformers", "open_clip", "PIL"}))
        for module in forbidden_shared:
            findings.append(
                Finding(
                    severity="warning",
                    path=rel,
                    message=f"`src/shared` imports heavy/domain dependency `{module}`; keep shared utilities lightweight.",
                )
            )
    return findings


def _audit_size(path: Path, root: Path, line_count: int) -> list[Finding]:
    rel = _relative(path, root)
    category = categorize_path(path, root)
    findings: list[Finding] = []
    if category in {"cli", "notebook_cell", "notebook_helper", "validation"} and line_count > 800:
        findings.append(
            Finding(
                severity="warning",
                path=rel,
                message=(
                    f"{category} file has {line_count} lines; consider extracting reusable logic into `src` "
                    "or smaller testable helpers."
                ),
            )
        )
    return findings


def _file_record(path: Path, root: Path) -> FileRecord:
    function_count, class_count, max_function_lines = _python_metrics(path)
    return FileRecord(
        path=_relative(path, root),
        category=categorize_path(path, root),
        line_count=_line_count(path),
        function_count=function_count,
        class_count=class_count,
        max_function_lines=max_function_lines,
        traits=_script_traits(path) if _relative(path, root).startswith("scripts/") else (),
        marker_counts=_marker_counts(path),
    )


def _audit_notebook(path: Path, root: Path, max_code_cell_lines: int) -> tuple[FileRecord, list[Finding]]:
    rel = _relative(path, root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return FileRecord(path=rel, category="notebook", line_count=0), [
            Finding(severity="error", path=rel, message=f"Notebook JSON could not be parsed: {exc}")
        ]
    code_lines = 0
    findings: list[Finding] = []
    for index, cell in enumerate(payload.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source_lines = source if isinstance(source, list) else str(source).splitlines()
        cell_line_count = len(source_lines)
        code_lines += cell_line_count
        if cell_line_count > max_code_cell_lines:
            findings.append(
                Finding(
                    severity="warning",
                    path=rel,
                    message=(
                        f"Code cell {index} has {cell_line_count} lines; move durable logic into "
                        "`scripts/notebook_helpers` or `src`."
                    ),
                )
            )
    return FileRecord(path=rel, category="notebook", line_count=code_lines), findings


def build_report(root: Path = REPO_ROOT, *, max_code_cell_lines: int = 120) -> dict[str, Any]:
    root = root.resolve()
    files: list[FileRecord] = []
    findings: list[Finding] = []

    for path in iter_python_files(root):
        record = _file_record(path, root)
        files.append(record)
        findings.extend(_audit_import_boundaries(path, root))
        findings.extend(_audit_size(path, root, record.line_count))

    for path in iter_notebooks(root):
        record, notebook_findings = _audit_notebook(path, root, max_code_cell_lines)
        files.append(record)
        findings.extend(notebook_findings)

    category_counts: dict[str, int] = {}
    for record in files:
        category_counts[record.category] = category_counts.get(record.category, 0) + 1
    size_buckets = {
        "over_500": sum(1 for record in files if record.line_count > 500),
        "over_1000": sum(1 for record in files if record.line_count > 1000),
        "over_2000": sum(1 for record in files if record.line_count > 2000),
    }
    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    return {
        "status": "fail" if error_count else "pass",
        "file_count": len(files),
        "category_counts": dict(sorted(category_counts.items())),
        "size_buckets": size_buckets,
        "error_count": error_count,
        "warning_count": warning_count,
        "files": [asdict(record) for record in files],
        "findings": [asdict(finding) for finding in findings],
    }


def write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    files = sorted(
        report["files"],
        key=lambda item: (-int(item.get("line_count", 0)), str(item.get("path", ""))),
    )
    lines = [
        "# Python Surface Audit",
        "",
        "Generated by `scripts/audit_code_organization.py --markdown-output`.",
        "",
        "## Summary",
        "",
        f"- status: `{report['status']}`",
        f"- files: `{report['file_count']}`",
        f"- errors: `{report['error_count']}`",
        f"- warnings: `{report['warning_count']}`",
        f"- size_buckets: `{json.dumps(report.get('size_buckets', {}), sort_keys=True)}`",
        "",
        "## Largest Python And Notebook Surfaces",
        "",
        "| path | category | lines | functions | classes | max_function_lines | traits | markers |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for item in files[:20]:
        markers = dict(item.get("marker_counts") or {})
        marker_summary = ", ".join(f"{key}:{value}" for key, value in sorted(markers.items())) if markers else ""
        lines.append(
            f"| {item.get('path', '')} | {item.get('category', '')} | {item.get('line_count', 0)} | "
            f"{item.get('function_count', 0)} | {item.get('class_count', 0)} | "
            f"{item.get('max_function_lines', 0)} | {', '.join(item.get('traits') or [])} | "
            f"{marker_summary} |"
        )

    lines.extend(["", "## Current Refactor Warnings", ""])
    if report["findings"]:
        lines.extend(f"- `{finding['severity']}` `{finding['path']}`: {finding['message']}" for finding in report["findings"])
    else:
        lines.append("- No findings.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--max-code-cell-lines", type=int, default=120)
    parser.add_argument("--no-write", action="store_true", help="Print the summary without writing the JSON report.")
    args = parser.parse_args(argv)

    report = build_report(args.root, max_code_cell_lines=args.max_code_cell_lines)
    if not args.no_write:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = args.root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown_output:
        markdown_output = args.markdown_output
        if not markdown_output.is_absolute():
            markdown_output = args.root / markdown_output
        write_markdown_report(report, markdown_output)

    print(
        "code_organization "
        f"status={report['status']} files={report['file_count']} "
        f"errors={report['error_count']} warnings={report['warning_count']}"
    )
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
