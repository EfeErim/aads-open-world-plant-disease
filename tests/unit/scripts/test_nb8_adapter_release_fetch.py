from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CELL = ROOT / "scripts/notebook_cells/nb8_cell04_adapter_release_fetch.py"


def test_notebook8_release_fetch_loads_only_the_read_only_colab_secret() -> None:
    source = CELL.read_text(encoding="utf-8")

    assert 'userdata.get("AADS_GITHUB_RELEASE_READ_TOKEN")' in source
    assert 'os.environ["AADS_GITHUB_RELEASE_READ_TOKEN"]' in source
    assert "AADS_GITHUB_RELEASE_WRITE_TOKEN" not in source
