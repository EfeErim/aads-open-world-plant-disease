#!/usr/bin/env python3
"""Collect provenance-preserving image-search candidates for manual OOD review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw

USER_AGENT = "Mozilla/5.0"
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    filename: str
    query: str
    image_url: str
    source_page_url: str
    title: str
    sha256: str
    width: int
    height: int
    review_status: str = "candidate_pending_manual_review"


def parse_bing_results(payload: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    soup = BeautifulSoup(payload, "html.parser")
    for element in soup.select("a.iusc[m]"):
        try:
            metadata = json.loads(html.unescape(str(element.get("m") or "")))
        except json.JSONDecodeError:
            continue
        image_url = str(metadata.get("murl") or "").strip()
        page_url = str(metadata.get("purl") or "").strip()
        if not image_url.startswith(("http://", "https://")) or image_url in seen:
            continue
        seen.add(image_url)
        rows.append(
            {
                "image_url": image_url,
                "source_page_url": page_url,
                "title": str(metadata.get("t") or metadata.get("desc") or "").strip(),
            }
        )
    return rows


def parse_duckduckgo_results(payload: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in payload.get("results", []):
        if not isinstance(raw, dict):
            continue
        image_url = str(raw.get("image") or "").strip()
        if not image_url.startswith(("http://", "https://")) or image_url in seen:
            continue
        seen.add(image_url)
        rows.append(
            {
                "image_url": image_url,
                "source_page_url": str(raw.get("url") or "").strip(),
                "title": str(raw.get("title") or "").strip(),
            }
        )
    return rows


def fetch_results(session: requests.Session, query: str) -> list[dict[str, str]]:
    landing = session.get("https://duckduckgo.com/", params={"q": query}, timeout=30)
    landing.raise_for_status()
    match = re.search(r"vqd=([0-9-]+)", landing.text)
    if match is None:
        raise RuntimeError("DuckDuckGo image-search token was not found")
    response = session.get(
        "https://duckduckgo.com/i.js",
        params={"q": query, "vqd": match.group(1), "o": "json", "f": ",,,", "p": "1"},
        headers={"Referer": "https://duckduckgo.com/"},
        timeout=30,
    )
    response.raise_for_status()
    return parse_duckduckgo_results(response.json())


def decode_image(payload: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(payload)) as image:
        if image.format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError(f"unsupported image format: {image.format}")
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError("image pixel count exceeds safety limit")
        image.load()
    if width < 128 or height < 128:
        raise ValueError("image is too small")
    return width, height


def download_image(result: dict[str, str]) -> tuple[bytes | None, str]:
    try:
        with requests.get(
            result["image_url"], headers={"User-Agent": USER_AGENT}, timeout=(5, 10), stream=True
        ) as response:
            response.raise_for_status()
            content_length = int(response.headers.get("Content-Length") or 0)
            if content_length > MAX_DOWNLOAD_BYTES:
                return None, "download_failed"
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    return None, "download_failed"
                chunks.append(chunk)
            return b"".join(chunks), ""
    except requests.RequestException:
        return None, "download_failed"


def collect(query: str, output_dir: Path, *, limit: int) -> tuple[list[Candidate], dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    results = fetch_results(session, query)
    accepted: list[Candidate] = []
    seen_hashes: set[str] = set()
    stats = {"search_results": len(results), "download_failed": 0, "invalid_image": 0, "duplicate_hash": 0}
    batch_size = 16
    for offset in range(0, len(results), batch_size):
        batch = results[offset : offset + batch_size]
        with ThreadPoolExecutor(max_workers=8) as executor:
            downloads = list(executor.map(download_image, batch))
        for result, (payload, failure) in zip(batch, downloads, strict=True):
            if len(accepted) >= limit:
                break
            if payload is None:
                assert failure
                stats["download_failed"] += 1
                continue
            try:
                width, height = decode_image(payload)
            except (OSError, ValueError):
                stats["invalid_image"] += 1
                continue
            digest = hashlib.sha256(payload).hexdigest()
            if digest in seen_hashes:
                stats["duplicate_hash"] += 1
                continue
            seen_hashes.add(digest)
            candidate_id = f"c{len(accepted) + 1:03d}"
            filename = f"{candidate_id}_{digest[:12]}.jpg"
            (output_dir / filename).write_bytes(payload)
            accepted.append(
                Candidate(
                    candidate_id=candidate_id,
                    filename=filename,
                    query=query,
                    image_url=result["image_url"],
                    source_page_url=result["source_page_url"],
                    title=result["title"],
                    sha256=digest,
                    width=width,
                    height=height,
                )
            )
        if len(accepted) >= limit:
            break
    stats["accepted_candidates"] = len(accepted)
    return accepted, stats


def write_manifest(rows: list[Candidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_contact_sheet(rows: list[Candidate], image_dir: Path, output: Path) -> None:
    thumb_width, thumb_height, columns = 220, 180, 5
    canvas = Image.new(
        "RGB",
        (columns * thumb_width, ((len(rows) + columns - 1) // columns) * thumb_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        with Image.open(image_dir / row.filename) as source:
            image = source.convert("RGB")
            image.thumbnail((thumb_width - 8, thumb_height - 35))
        x = (index % columns) * thumb_width
        y = (index // columns) * thumb_height
        canvas.paste(image, (x + 4, y + 4))
        draw.text((x + 4, y + thumb_height - 27), f"{row.candidate_id} {row.width}x{row.height}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, stats = collect(args.query, args.output_dir, limit=args.limit)
    if not rows:
        raise RuntimeError("No valid image candidates were collected")
    write_manifest(rows, args.manifest)
    write_contact_sheet(rows, args.output_dir, args.contact_sheet)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    print(f"manifest={args.manifest} contact_sheet={args.contact_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
