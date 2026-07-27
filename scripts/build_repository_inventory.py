from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
DISPOSITIONS = {"keep", "archive", "delete", "generated-local", "quarantine", "unclassified"}
GENERATED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "checkpoints",
    "logs",
    "outputs",
    "runs",
    "wandb",
}
CACHE_PARTS = {".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__", "wandb"}
BINARY_SUFFIXES = {
    ".7z", ".bin", ".docx", ".gif", ".gz", ".h5", ".hdf5", ".jpeg", ".jpg", ".onnx",
    ".pdf", ".pickle", ".pkl", ".png", ".pt", ".pth", ".tar", ".webp", ".xlsx", ".zip",
}
TEXT_SUFFIXES = {
    ".cfg", ".csv", ".ini", ".ipynb", ".json", ".md", ".py", ".toml", ".tsv", ".txt", ".yaml", ".yml",
}
REFERENCE_SUFFIXES = TEXT_SUFFIXES - {".csv", ".tsv"}
PATH_TOKEN_RE = re.compile(
    r"(?P<path>(?:\.?\.?/)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_ .@+()\[\]-]+)+\.[A-Za-z0-9]{1,12})"
)
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#?]+)")
SUBPROCESS_RE = re.compile(r"(?:subprocess\.(?:run|Popen|call|check_call|check_output)|os\.system)\s*\(")
CONFIG_RE = re.compile(r"(?:config/[^\s'\"`),]+|[^\s'\"`),]+\.(?:json|ya?ml|toml))")
OUTPUT_DIRS = {"docs/demo_results", "docs/ablation_results", "models/adapters", "outputs", "runs", ".runtime_tmp"}
METADATA_ONLY_ROOTS = {"data", "models", "outputs", "runs", ".runtime_tmp"}
METADATA_ONLY_SIZE_BYTES = 10 * 1024 * 1024


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return result.stdout


def git_paths(root: Path, *args: str) -> set[str]:
    output = subprocess.run(
        ["git", *args, "-z"], cwd=root, check=True, capture_output=True
    ).stdout
    return {part.decode("utf-8", errors="surrogateescape").replace("\\", "/") for part in output.split(b"\0") if part}


def git_index_entries(root: Path) -> dict[str, str]:
    output = subprocess.run(
        ["git", "ls-files", "--stage", "-z"], cwd=root, check=True, capture_output=True
    ).stdout
    entries: dict[str, str] = {}
    for raw_entry in output.split(b"\0"):
        if not raw_entry:
            continue
        metadata, raw_path = raw_entry.split(b"\t", maxsplit=1)
        _mode, object_id, stage = metadata.decode("ascii").split()
        if stage == "0":
            entries[raw_path.decode("utf-8", errors="surrogateescape").replace("\\", "/")] = object_id
    return entries


def hash_git_objects(root: Path, object_ids: Iterable[str]) -> dict[str, tuple[str, int]]:
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"], cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    assert process.stdin is not None and process.stdout is not None
    results: dict[str, tuple[str, int]] = {}
    try:
        for object_id in dict.fromkeys(object_ids):
            process.stdin.write(object_id.encode("ascii") + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().decode("ascii").strip().split()
            if len(header) != 3 or header[1] != "blob":
                raise RuntimeError(f"Unable to read Git blob {object_id}: {' '.join(header)}")
            size = int(header[2])
            digest = hashlib.sha256()
            remaining = size
            while remaining:
                chunk = process.stdout.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError(f"Unexpected EOF while reading Git blob {object_id}")
                digest.update(chunk)
                remaining -= len(chunk)
            if process.stdout.read(1) != b"\n":
                raise RuntimeError(f"Missing delimiter after Git blob {object_id}")
            results[object_id] = (digest.hexdigest(), size)
    finally:
        process.stdin.close()
        process.wait()
    return results


def git_object_sizes(root: Path, object_ids: Iterable[str]) -> dict[str, int]:
    unique_ids = list(dict.fromkeys(object_ids))
    result = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        cwd=root,
        input="".join(f"{object_id}\n" for object_id in unique_ids),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    sizes: dict[str, int] = {}
    for line in result.stdout.splitlines():
        object_id, object_type, raw_size = line.split()
        if object_type != "blob":
            raise RuntimeError(f"Expected Git blob for {object_id}, found {object_type}")
        sizes[object_id] = int(raw_size)
    return sizes


def sha256_file(path: Path) -> tuple[str | None, str | None]:
    digest = hashlib.sha256()
    try:
        with _io_path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return digest.hexdigest(), None


def is_metadata_only(relative: str, size_bytes: int) -> bool:
    path = PurePosixPath(relative)
    return (
        bool(path.parts and path.parts[0] in METADATA_ONLY_ROOTS)
        or path.suffix.lower() in BINARY_SUFFIXES
        or size_bytes >= METADATA_ONLY_SIZE_BYTES
    )


def inspect_file(path: Path, root: Path) -> tuple[int | None, str | None, str | None]:
    try:
        size = _io_path(path).stat().st_size
    except OSError as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    relative = path.relative_to(root).as_posix()
    if is_metadata_only(relative, size):
        return size, None, None
    file_hash, error = sha256_file(path)
    return size, file_hash, error


def _io_path(path: Path) -> Path:
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def iter_repository_files(root: Path, excluded_relative_paths: set[str]) -> Iterable[Path]:
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(root)
        if relative_directory == Path("."):
            dirnames[:] = [name for name in dirnames if name not in {".git", ".venv"}]
        dirnames.sort(key=str.casefold)
        filenames.sort(key=str.casefold)
        for filename in filenames:
            path = directory_path / filename
            relative = path.relative_to(root).as_posix()
            if relative in excluded_relative_paths or path.is_symlink():
                continue
            yield path


def classify_disposition(relative: str, tracked: bool, ignored: bool, protected: bool) -> tuple[str, str]:
    parts = set(PurePosixPath(relative).parts)
    if protected:
        return "keep", "Protected by the Phase 0 worktree contract."
    if ignored or parts & GENERATED_PARTS or relative.startswith("data/prepared_runtime_datasets/"):
        return "generated-local", "Local/generated surface excluded from maintained source."
    if relative.startswith(("tmp/", ".tmp")):
        return "quarantine", "Temporary content; retain until a later reviewed quarantine phase."
    if tracked:
        return "keep", "Tracked maintained or evidence file; no Phase 1 retention change."
    return "unclassified", "Untracked path requires an explicit later-phase retention decision."


def infer_purpose(relative: str) -> str:
    path = PurePosixPath(relative)
    if relative.startswith("tests/"):
        return "test evidence"
    if relative.startswith("docs/") or path.suffix.lower() == ".md":
        return "documentation or recorded evidence"
    if relative.startswith("colab_notebooks/"):
        return "notebook orchestration"
    if relative.startswith("scripts/"):
        return "operational script or notebook helper"
    if relative.startswith("src/"):
        return "maintained application source"
    if relative.startswith("config/"):
        return "versioned configuration"
    if relative.startswith("data/"):
        return "dataset, manifest, provenance, or fixture"
    if relative.startswith("runs/"):
        return "local run artifact"
    if relative.startswith("models/"):
        return "model or adapter artifact"
    if relative.startswith(".ai/") or relative.startswith(".agents/"):
        return "repository AI instruction"
    return "repository support file"


def infer_owner(relative: str) -> str:
    first = PurePosixPath(relative).parts[0]
    return {
        "src": "application-platform",
        "scripts": "operations",
        "tests": "testing",
        "docs": "documentation",
        "colab_notebooks": "colab-workflows",
        "config": "configuration",
        "data": "data-governance",
        "runs": "runtime-artifacts",
        "models": "model-artifacts",
        ".ai": "ai-engineering",
        ".agents": "ai-discovery",
    }.get(first, "repository-maintenance")


def infer_outputs(relative: str) -> list[str]:
    return [output for output in sorted(OUTPUT_DIRS) if relative == output or relative.startswith(f"{output}/")]


def read_text(path: Path, size: int) -> str | None:
    if size > 5 * 1024 * 1024 or path.suffix.lower() not in REFERENCE_SUFFIXES:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def python_imports(relative: str, text: str) -> list[str]:
    if not relative.endswith(".py"):
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return sorted(imports)


def resolve_reference(source: str, candidate: str, known_paths: set[str]) -> str | None:
    candidate = candidate.strip().replace("\\", "/")
    if not candidate or candidate.startswith(("http://", "https://", "#")):
        return None
    direct = PurePosixPath(candidate).as_posix().lstrip("./")
    source_relative = (PurePosixPath(source).parent / candidate).as_posix()
    for value in (direct, source_relative):
        normalized = str(PurePosixPath(value))
        if normalized in known_paths:
            return normalized
    return None


def build_reference_graph(root: Path, records: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    known_paths = {record["path"] for record in records}
    edges: set[tuple[str, str, str]] = set()
    unresolved: set[tuple[str, str, str]] = set()
    for record in records:
        if record["disposition"] in {"generated-local", "quarantine"} or record["hash_policy"] == "metadata-only":
            continue
        relative = record["path"]
        text = read_text(root / relative, record["size_bytes"])
        if text is None:
            continue
        candidates: list[tuple[str, str]] = []
        if relative.endswith(".py"):
            candidates.extend(("import", module.replace(".", "/") + ".py") for module in python_imports(relative, text))
        if relative.endswith(".md"):
            candidates.extend(("markdown", match) for match in MARKDOWN_LINK_RE.findall(text))
        if relative.endswith(".ipynb"):
            candidates.extend(("notebook", match.group("path")) for match in PATH_TOKEN_RE.finditer(text))
        if SUBPROCESS_RE.search(text):
            candidates.extend(("subprocess", match.group("path")) for match in PATH_TOKEN_RE.finditer(text))
        candidates.extend(("config", match) for match in CONFIG_RE.findall(text))
        for candidate in candidates:
            kind, raw_target = candidate
            target = resolve_reference(relative, raw_target, known_paths)
            if target:
                edges.add((kind, relative, target))
            elif "/" in raw_target or kind in {"markdown", "notebook", "subprocess"}:
                unresolved.add((kind, relative, raw_target.strip()))
    return (
        [{"kind": kind, "source": source, "target": target} for kind, source, target in sorted(edges)],
        [{"kind": kind, "source": source, "reference": target} for kind, source, target in sorted(unresolved)],
    )


def build_inventory(root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    output_path = output_path.resolve()
    excluded = {output_path.relative_to(root).as_posix()}
    index_entries = git_index_entries(root)
    tracked_paths = set(index_entries)
    untracked_paths = git_paths(root, "ls-files", "--others", "--exclude-standard")
    ignored_paths = git_paths(root, "ls-files", "--others", "--ignored", "--exclude-standard")
    status_paths = git_paths(root, "diff", "--name-only") | git_paths(root, "diff", "--cached", "--name-only")
    protected_roots = _load_protected_roots(root)
    records: list[dict[str, Any]] = []
    scan_errors: list[dict[str, str]] = []
    totals_by_suffix: Counter[str] = Counter()
    bytes_by_suffix: Counter[str] = Counter()
    paths = list(iter_repository_files(root, excluded))
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
        inspections = executor.map(lambda path: inspect_file(path, root), paths)
        path_inspections = zip(paths, inspections, strict=True)
        for path, inspection in path_inspections:
            relative = path.relative_to(root).as_posix()
            size, file_hash, inspection_error = inspection
            if size is None or inspection_error:
                scan_errors.append({"path": relative, "error": inspection_error or "unknown inspection error"})
                continue
            tracked = relative in tracked_paths
            ignored = relative in ignored_paths
            protected = any(
                relative == item.rstrip("/") or relative.startswith(item.rstrip("/") + "/")
                for item in protected_roots
            )
            disposition, decision_reason = classify_disposition(relative, tracked, ignored, protected)
            suffix = path.suffix.lower() or "[no-extension]"
            totals_by_suffix[suffix] += 1
            bytes_by_suffix[suffix] += size
            metadata_only = is_metadata_only(relative, size)
            record: dict[str, Any] = {
                "path": relative,
                "present": True,
                "folder": PurePosixPath(relative).parent.as_posix(),
                "extension": PurePosixPath(relative).suffix.lower(),
                "size_bytes": size,
                "tracked": tracked,
                "untracked": relative in untracked_paths,
                "disposition": disposition,
                "hash_policy": "metadata-only" if metadata_only else "sha256",
            }
            if not metadata_only:
                record.update(
                    {
                        "purpose": infer_purpose(relative),
                        "owner": infer_owner(relative),
                        "consumers": [],
                        "outputs": infer_outputs(relative),
                        "sha256": file_hash,
                        "lineage": {
                            "tracked": tracked,
                            "untracked": relative in untracked_paths,
                            "ignored": ignored,
                            "worktree_modified": relative in status_paths,
                        },
                        "decision_reason": decision_reason,
                        "canonical_replacement": None,
                    }
                )
            records.append(record)
    record_paths = {record["path"] for record in records}
    missing_tracked_paths = sorted(tracked_paths - record_paths - excluded)
    object_sizes = git_object_sizes(root, (index_entries[path] for path in missing_tracked_paths))
    hash_required_paths = [
        path
        for path in missing_tracked_paths
        if not is_metadata_only(path, object_sizes[index_entries[path]])
    ]
    missing_object_data = hash_git_objects(root, (index_entries[path] for path in hash_required_paths))
    for relative in missing_tracked_paths:
        object_id = index_entries[relative]
        size = object_sizes[object_id]
        metadata_only = is_metadata_only(relative, size)
        file_hash = None if metadata_only else missing_object_data[object_id][0]
        record = {
            "path": relative,
            "present": False,
            "folder": PurePosixPath(relative).parent.as_posix(),
            "extension": PurePosixPath(relative).suffix.lower(),
            "size_bytes": size,
            "tracked": True,
            "untracked": False,
            "disposition": "keep",
            "hash_policy": "metadata-only" if metadata_only else "sha256",
        }
        if not metadata_only:
            record.update(
                {
                    "purpose": infer_purpose(relative),
                    "owner": infer_owner(relative),
                    "consumers": [],
                    "outputs": infer_outputs(relative),
                    "sha256": file_hash,
                    "lineage": {
                        "tracked": True,
                        "untracked": False,
                        "ignored": False,
                        "worktree_modified": relative in status_paths,
                        "index_blob": object_id,
                        "materialized": False,
                    },
                    "decision_reason": "Tracked Git index entry is absent from the sparse or modified working tree.",
                    "canonical_replacement": None,
                }
            )
        records.append(record)
    records.sort(key=lambda record: record["path"].casefold())
    missing_tracked: list[str] = []
    graph, unresolved = build_reference_graph(root, records)
    consumers: defaultdict[str, set[str]] = defaultdict(set)
    for edge in graph:
        consumers[edge["target"]].add(edge["source"])
    for record in records:
        if record["hash_policy"] == "sha256":
            record["consumers"] = sorted(consumers[record["path"]])
    dispositions = Counter(record["disposition"] for record in records)
    categories = _category_totals(records)
    reference_kind_counts = Counter(edge["kind"] for edge in graph)
    head = _run_git(root, "rev-parse", "HEAD").strip()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "root": root.as_posix(),
        "base_commit": head,
        "snapshot": {"before_commit": head, "after_commit": head, "files_deleted": 0, "files_moved": 0},
        "excluded_environment_roots": [".git/", ".venv/"],
        "excluded_self_paths": sorted(excluded),
        "summary": {
            "file_count": len(records),
            "size_bytes": sum(record["size_bytes"] for record in records),
            "materialized_file_count": sum(record.get("present", True) for record in records),
            "materialized_size_bytes": sum(
                record["size_bytes"] for record in records if record.get("present", True)
            ),
            "nonmaterialized_tracked_file_count": len(missing_tracked_paths),
            "tracked_file_count": len(tracked_paths),
            "tracked_files_missing": missing_tracked,
            "unclassified_count": dispositions["unclassified"],
            "unresolved_reference_count": len(unresolved),
            "reference_kind_counts": dict(sorted(reference_kind_counts.items())),
            "scan_error_count": len(scan_errors),
            "metadata_only_count": sum(record["hash_policy"] == "metadata-only" for record in records),
            "sha256_count": sum(record["hash_policy"] == "sha256" for record in records),
            "disposition_counts": dict(sorted(dispositions.items())),
            "category_totals": categories,
            "suffix_file_counts": dict(sorted(totals_by_suffix.items())),
            "suffix_size_bytes": dict(sorted(bytes_by_suffix.items())),
        },
        "reference_graph": graph,
        "unresolved_references": unresolved,
        "scan_errors": scan_errors,
        "files": records,
    }


def _load_protected_roots(root: Path) -> list[str]:
    state_path = root / "docs/repository_simplification_run_state.json"
    if not state_path.exists():
        return []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return [value.replace("\\", "/") for value in state.get("phases", {}).get("0", {}).get("protected_paths", []) if not re.match(r"^[A-Za-z]:/", value)]


def _category_totals(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    categories = {
        "dataset": lambda path: path.startswith("data/"),
        "run": lambda path: path.startswith("runs/"),
        "binary": lambda path: PurePosixPath(path).suffix.lower() in BINARY_SUFFIXES,
        "markdown": lambda path: path.endswith(".md"),
        "python": lambda path: path.endswith(".py"),
        "notebook": lambda path: path.endswith(".ipynb"),
        "cache": lambda path: bool(set(PurePosixPath(path).parts) & CACHE_PARTS),
    }
    result: dict[str, dict[str, int]] = {}
    for name, predicate in categories.items():
        selected = [record for record in records if predicate(record["path"])]
        result[name] = {"files": len(selected), "bytes": sum(record["size_bytes"] for record in selected)}
    materialized = [record for record in records if record.get("present", True)]
    result["disk"] = {"files": len(materialized), "bytes": sum(record["size_bytes"] for record in materialized)}
    return result


def write_summary(path: Path, inventory: dict[str, Any], inventory_relative: str) -> None:
    summary = inventory["summary"]
    category_rows = "\n".join(
        f"| `{name}` | {values['files']:,} | {values['bytes']:,} |"
        for name, values in summary["category_totals"].items()
    )
    disposition_rows = "\n".join(
        f"| `{name}` | {count:,} |" for name, count in summary["disposition_counts"].items()
    )
    text = f"""# Repository Inventory Summary

This is the tracked Phase 1 summary. The complete file-level inventory remains local at `{inventory_relative}`.

- Snapshot commit: `{inventory['base_commit']}`
- Generated at: `{inventory['generated_at']}`
- Files: `{summary['file_count']:,}` (`{summary['size_bytes']:,}` bytes)
- Materialized working-tree files: `{summary['materialized_file_count']:,}` (`{summary['materialized_size_bytes']:,}` bytes)
- Non-materialized tracked files: `{summary['nonmaterialized_tracked_file_count']:,}`
- Git-tracked files: `{summary['tracked_file_count']:,}`; missing from inventory: `{len(summary['tracked_files_missing'])}`
- Unclassified files: `{summary['unclassified_count']:,}`
- Unresolved references: `{summary['unresolved_reference_count']:,}`
- Concurrent generated-file scan errors: `{summary['scan_error_count']:,}`
- Metadata-only files: `{summary['metadata_only_count']:,}`; SHA-256 files: `{summary['sha256_count']:,}`
- Reference edges: `{sum(summary['reference_kind_counts'].values()):,}` across `{', '.join(f'{name}={count:,}' for name, count in summary['reference_kind_counts'].items())}`
- Deleted files: `0`; moved files: `0`

## Surface Totals

| Surface | Files | Bytes |
|---|---:|---:|
{category_rows}

## Dispositions

| Disposition | Files |
|---|---:|
{disposition_rows}

The before and after baselines intentionally point to the same commit because Phase 1 is measurement-only. `unclassified` and unresolved references are explicit review queues, not implicit deletion approval.
"""
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the complete Phase 1 repository inventory.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path(".runtime_tmp/repo_cleanup/active/repository_inventory.json"))
    parser.add_argument("--summary", type=Path, default=Path("docs/repository_inventory_summary.md"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    summary_path = args.summary if args.summary.is_absolute() else root / args.summary
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(root, output)
    output.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary(summary_path, inventory, output.relative_to(root).as_posix())
    summary = inventory["summary"]
    status = "pass" if not summary["tracked_files_missing"] else "fail"
    print(
        f"repository_inventory status={status} files={summary['file_count']} "
        f"tracked={summary['tracked_file_count']} unclassified={summary['unclassified_count']} "
        f"unresolved_references={summary['unresolved_reference_count']}"
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
