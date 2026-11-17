"""Structural checks for authored notebooks without importing model dependencies."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _sources():
    for path in sorted((ROOT / "colab_notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                yield f"{path.name}:{index}", "".join(cell["source"])


def _trees():
    for label, source in _sources():
        yield label, ast.parse(source, filename=label)
    for folder in ("notebook_cells", "notebook_helpers"):
        for path in sorted((ROOT / "scripts" / folder).glob("*.py")):
            yield str(path.relative_to(ROOT)), ast.parse(path.read_text(encoding="utf-8-sig"))


def test_notebook_code_compiles():
    sources = list(_sources())
    assert sources, "No authored notebook code found"
    for label, source in sources:
        compile(source, label, "exec")


def test_referenced_cell_scripts_exist():
    for label, tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id == "run_cell_script" and node.args:
                name = node.args[0]
                if isinstance(name, ast.Constant) and isinstance(name.value, str):
                    assert (ROOT / "scripts/notebook_cells" / name.value).is_file(), (label, name.value)


def test_repository_imports_resolve():
    for label, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                if module.split(".")[0] not in {"src", "scripts", "experiments"}:
                    continue
                path = ROOT.joinpath(*module.split("."))
                assert path.with_suffix(".py").is_file() or path.is_dir(), (label, module)


def test_notebooks_do_not_embed_tokens_in_remote_urls():
    forbidden = ("_repo_url_with_token", "x-access-token:", "{token}@")
    sources = list(_sources())
    sources.extend((str(path), path.read_text(encoding="utf-8-sig"))
                   for path in (ROOT / "scripts/notebook_cells").glob("*.py"))
    for label, source in sources:
        for fragment in forbidden:
            assert fragment not in source, (label, fragment)
