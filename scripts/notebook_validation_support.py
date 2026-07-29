"""Shared helpers for notebook surface validation."""

from __future__ import annotations

import builtins
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _safe_print(*args, **kwargs):
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        converted = [str(a).encode("ascii", errors="replace").decode("ascii") for a in args]
        builtins.print(*converted, **kwargs)


print = _safe_print

PARAMETER_CAPTURE = 'with TELEMETRY.capture_cell_output("Cell 3: Parameters"):'
ACCESS_CHECK_CAPTURE = (
    'with TELEMETRY.capture_cell_output("Cell 3b: Guncelleme ve Erisim Kontrolu"):'
)
REPO_BOOTSTRAP_REQUIRED = (
    "from pathlib import Path",
    "CLONE_TARGET = Path('/content/bitirmeprojesi')",
    "REPO_URL = os.environ.get('AADS_REPO_URL'",
    "['git', 'clone', '--depth', '1'",
)
UPDATE_CHECK_REQUIRED = (
    "repo_root_for_update_check = _ensure_repo_root_for_update_check()",
    "def _build_repo_access_url(",
    "from scripts.colab_repo_bootstrap import probe_repo_update_status",
    "[KONTROL] Ilk hucre:",
)
DRIVE_REPO_BOOTSTRAP_FORBIDDEN = (
    "Path('/content/drive/MyDrive/bitirme projesi')",
    "Path('/content/drive/MyDrive/bitirmeprojesi')",
    "def _mount_drive_inline()",
    "mount_drive_if_available",
)


def _assert_contains(source: str, snippet: str, message: str) -> None:
    assert snippet in source, message.format(snippet=snippet)


def _assert_not_contains(source: str, snippet: str, message: str) -> None:
    assert snippet not in source, message.format(snippet=snippet)


def _assert_contains_all(source: str, snippets: tuple[str, ...], message: str) -> None:
    for snippet in snippets:
        _assert_contains(source, snippet, message)


def _assert_not_contains_all(source: str, snippets: tuple[str, ...], message: str) -> None:
    for snippet in snippets:
        _assert_not_contains(source, snippet, message)


@dataclass(frozen=True)
class NotebookSources:
    notebook_path: Path
    code_cells: tuple[str, ...]
    full_source: str
    first_code_source: str


def _load_notebook_sources_from_path(notebook_path: Path) -> NotebookSources:
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell_runner_pattern = re.compile(r"run_cell_script\((['\"])(?P<name>[^'\"]+)\1,\s*globals\(\)\)")

    def _expand_cell_source(source: str) -> str:
        matches = tuple(cell_runner_pattern.finditer(source))
        if not matches:
            return source
        expanded_parts = [source]
        for match in matches:
            script_path = ROOT / "scripts" / "notebook_cells" / match.group("name")
            assert script_path.is_file(), f"Notebook cell script was not found: {script_path}"
            expanded_parts.append(script_path.read_text(encoding="utf-8"))
        return "\n".join(expanded_parts)

    code_cells = tuple(
        _expand_cell_source("".join(cell.get("source", [])))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    assert code_cells, f"{notebook_path.name} code cells were not found"
    return NotebookSources(
        notebook_path=notebook_path,
        code_cells=code_cells,
        full_source="\n\n".join(code_cells),
        first_code_source=code_cells[0],
    )


def _load_notebook_sources(notebook_name: str) -> NotebookSources:
    return _load_notebook_sources_from_path(ROOT / "colab_notebooks" / notebook_name)


def _find_code_cell_source(sources: NotebookSources, marker: str, missing_message: str) -> str:
    for source in sources.code_cells:
        if marker in source:
            return source
    raise AssertionError(missing_message)


def _assert_code_cells_compile(sources: NotebookSources, notebook_label: str) -> None:
    for index, source in enumerate(sources.code_cells, start=1):
        try:
            compile(source, f"{sources.notebook_path}:code_cell_{index}", "exec")
        except SyntaxError as exc:
            raise AssertionError(
                f"{notebook_label} code cell {index} has invalid Python syntax: "
                f"line {exc.lineno}, offset {exc.offset}: {exc.msg}"
            ) from exc


def _assert_repo_bootstrap_contract(first_code_source: str, notebook_label: str) -> None:
    _assert_contains(
        first_code_source,
        "def _ensure_aads_repo_on_path():",
        f"{notebook_label} first code cell should make repo scripts importable before runner import: {{snippet}}",
    )
    assert first_code_source.index("def _ensure_aads_repo_on_path():") < first_code_source.index(
        "from scripts.notebook_helpers.cell_script_runner import run_cell_script"
    ), f"{notebook_label} first code cell imports the cell runner before repo path bootstrap"
    _assert_contains_all(
        first_code_source,
        REPO_BOOTSTRAP_REQUIRED,
        f"{notebook_label} first code cell is missing required GitHub bootstrap: {{snippet}}",
    )
    assert first_code_source.index("from pathlib import Path") < first_code_source.index(
        "CLONE_TARGET = Path('/content/bitirmeprojesi')"
    ), f"{notebook_label} first code cell uses Path before importing it"
    _assert_not_contains_all(
        first_code_source,
        DRIVE_REPO_BOOTSTRAP_FORBIDDEN,
        f"{notebook_label} first code cell should not use Drive for repo bootstrap: {{snippet}}",
    )


def _assert_clone_bootstrap_contract(first_code_source: str, notebook_label: str) -> None:
    _assert_contains(
        first_code_source,
        "def _ensure_aads_repo_on_path():",
        f"{notebook_label} first code cell should define the clone bootstrap: {{snippet}}",
    )
    _assert_contains_all(
        first_code_source,
        (
            "from pathlib import Path",
            "DEFAULT_REPO_URL = 'https://github.com/EfeErim/aads-open-world-plant-disease.git'",
            "REPO_URL = os.environ.get('AADS_REPO_URL', DEFAULT_REPO_URL)",
            "CLONE_TARGET = Path('/content/bitirmeprojesi')",
            "subprocess.run(",
            "'clone', '--depth', '1', '--branch', REPO_REF",
            "Notebook 4 repo ready:",
        ),
        f"{notebook_label} first code cell is missing required clone bootstrap: {{snippet}}",
    )
    assert first_code_source.index("from pathlib import Path") < first_code_source.index(
        "CLONE_TARGET = Path('/content/bitirmeprojesi')"
    ), f"{notebook_label} first code cell uses Path before importing it"
    _assert_not_contains_all(
        first_code_source,
        (
            "https://api.github.com/repos/",
            "DOWNLOAD_MANIFEST",
            "raw.githubusercontent.com",
            "manifest_text = response.read().decode('utf-8')",
            "urllib.request",
        ),
        f"{notebook_label} first code cell should clone instead of downloading raw source files: {{snippet}}",
    )


def _assert_update_check_contract(
    first_code_source: str,
    notebook_label: str,
    *,
    forbid_drive_bootstrap: bool,
) -> None:
    _assert_contains(
        first_code_source,
        "def _ensure_aads_repo_on_path():",
        f"{notebook_label} first code cell should make repo scripts importable before runner import: {{snippet}}",
    )
    assert first_code_source.index("def _ensure_aads_repo_on_path():") < first_code_source.index(
        "from scripts.notebook_helpers.cell_script_runner import run_cell_script"
    ), f"{notebook_label} first code cell imports the cell runner before repo path bootstrap"
    _assert_contains_all(
        first_code_source,
        UPDATE_CHECK_REQUIRED,
        f"{notebook_label} first code cell is missing required freshness check: {{snippet}}",
    )
    if forbid_drive_bootstrap:
        _assert_not_contains_all(
            first_code_source,
            DRIVE_REPO_BOOTSTRAP_FORBIDDEN,
            f"{notebook_label} first code cell should stay repo-first without Drive bootstrap: {{snippet}}",
        )


def gate_label(step_id: str, name: str) -> str:
    return f"[{step_id}] {name}"


@dataclass(frozen=True)
class ValidationCheck:
    result_name: str
    step_id: str
    description: str
    success_message: str
    failure_prefix: str
    callback: Callable[[], None]
    validation_group: str
    requires_runtime_dependencies: bool = True


VALIDATION_GROUP_ORDER = (
    "shared-prerequisite",
    "customer-notebook-support",
    "customer-facing-notebooks",
    "internal-maintenance-notebooks",
    "historical-report-only-notebooks",
)


def _run_check(check: ValidationCheck, *, leading_newline: bool = False) -> bool:
    prefix = "\n" if leading_newline else ""
    print(f"{prefix}Testing {gate_label(check.step_id, check.description)}...")
    try:
        check.callback()
    except Exception as exc:
        detail = str(exc).strip()
        failure_message = check.failure_prefix if not detail else f"{check.failure_prefix}: {detail}"
        print(f"FAIL {gate_label(check.step_id, failure_message)}")
        return False

    print(f"PASS {gate_label(check.step_id, check.success_message)}")
    return True
