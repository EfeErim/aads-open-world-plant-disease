from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "collect_image_search_candidates.py"
SPEC = importlib.util.spec_from_file_location("collect_image_search_candidates", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_bing_results_preserves_provenance_and_deduplicates() -> None:
    payload = """
    <a class="iusc" m='{"murl":"https://images.test/a.jpg","purl":"https://source.test/a","t":"A"}'></a>
    <a class="iusc" m='{"murl":"https://images.test/a.jpg","purl":"https://source.test/duplicate"}'></a>
    <a class="iusc" m='{"murl":"https://images.test/b.jpg","purl":"https://source.test/b","t":"B"}'></a>
    """

    assert MODULE.parse_bing_results(payload) == [
        {
            "image_url": "https://images.test/a.jpg",
            "source_page_url": "https://source.test/a",
            "title": "A",
        },
        {
            "image_url": "https://images.test/b.jpg",
            "source_page_url": "https://source.test/b",
            "title": "B",
        },
    ]


def test_parse_duckduckgo_results_preserves_provenance() -> None:
    payload = {
        "results": [
            {"image": "https://images.test/a.jpg", "url": "https://source.test/a", "title": "A"},
            {"image": "https://images.test/a.jpg", "url": "https://source.test/duplicate"},
        ]
    }

    assert MODULE.parse_duckduckgo_results(payload) == [
        {
            "image_url": "https://images.test/a.jpg",
            "source_page_url": "https://source.test/a",
            "title": "A",
        }
    ]


def test_decode_image_rejects_excessive_pixel_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "MAX_IMAGE_PIXELS", 100)
    payload = io.BytesIO()
    Image.new("RGB", (11, 10), "red").save(payload, format="PNG")

    with pytest.raises(ValueError, match="pixel count"):
        MODULE.decode_image(payload.getvalue())
