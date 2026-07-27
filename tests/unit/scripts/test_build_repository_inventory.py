from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.build_repository_inventory import build_inventory, main


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "inventory@example.invalid")
    _git(tmp_path, "config", "user.name", "Inventory Test")
    (tmp_path / "docs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "docs" / "guide.md").write_text("See [module](../src/module.py).\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "fixture")
    return tmp_path


def test_inventory_contains_every_tracked_file_and_reference(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    output = root / ".runtime_tmp" / "repo_cleanup" / "active" / "repository_inventory.json"

    inventory = build_inventory(root, output)

    assert inventory["summary"]["tracked_files_missing"] == []
    assert inventory["summary"]["tracked_file_count"] == 2
    assert {record["path"] for record in inventory["files"]} == {"docs/guide.md", "src/module.py"}
    assert {record["disposition"] for record in inventory["files"]} <= {
        "keep", "archive", "delete", "generated-local", "quarantine", "unclassified"
    }
    assert {tuple(edge.values()) for edge in inventory["reference_graph"]} >= {
        ("markdown", "docs/guide.md", "src/module.py")
    }
    assert inventory["summary"]["reference_kind_counts"]["markdown"] == 1
    assert inventory["summary"]["sha256_count"] == 2


def test_binary_and_large_surfaces_are_metadata_only(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "data").mkdir()
    (root / "data" / "sample.jpg").write_bytes(b"not-an-image")

    inventory = build_inventory(root, root / ".runtime_tmp/repository_inventory.json")
    record = next(item for item in inventory["files"] if item["path"] == "data/sample.jpg")

    assert record["hash_policy"] == "metadata-only"
    assert set(record) == {
        "path", "present", "folder", "extension", "size_bytes", "tracked", "untracked", "disposition", "hash_policy"
    }
    assert record["size_bytes"] == 12
    assert record["extension"] == ".jpg"
    assert record["folder"] == "data"


def test_main_writes_full_inventory_and_short_summary(tmp_path: Path, capsys) -> None:
    root = _repo(tmp_path)
    (root / "scratch.txt").write_text("local\n", encoding="utf-8")

    exit_code = main(["--root", str(root)])

    assert exit_code == 0
    inventory_path = root / ".runtime_tmp" / "repo_cleanup" / "active" / "repository_inventory.json"
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert payload["summary"]["unclassified_count"] == 1
    assert (root / "docs" / "repository_inventory_summary.md").is_file()
    assert "repository_inventory status=pass" in capsys.readouterr().out
