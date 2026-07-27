from __future__ import annotations

from pathlib import Path
from typing import MutableMapping


NOTEBOOK_CELL_SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "notebook_cells"


def run_cell_script(script_name: str, notebook_globals: MutableMapping[str, object]) -> None:
    """Execute a repo-maintained notebook cell script in the notebook namespace."""
    script_path = (NOTEBOOK_CELL_SCRIPT_ROOT / script_name).resolve()
    root = NOTEBOOK_CELL_SCRIPT_ROOT.resolve()
    try:
        script_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Notebook cell script must stay under {root}: {script_name}") from exc
    if not script_path.is_file():
        raise FileNotFoundError(f"Notebook cell script not found: {script_path}")

    notebook_globals.setdefault("__notebook_cell_script_root__", str(root))
    notebook_globals["__notebook_cell_script__"] = str(script_path)
    code = script_path.read_text(encoding="utf-8")
    exec(compile(code, str(script_path), "exec"), notebook_globals)


__all__ = ["run_cell_script"]
