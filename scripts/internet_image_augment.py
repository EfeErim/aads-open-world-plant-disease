#!/usr/bin/env python3
"""Download internet image candidates for weak classes while guarding against leakage.

Usage: set env vars `BING_SEARCH_KEY` and `BING_SEARCH_ENDPOINT` to enable automatic
image URL fetching from Bing Image Search. Otherwise the script emits per-class query
CSV files you can review and populate manually.

The script downloads candidates into `outputs/internet_image_candidates/<class>/staging/`,
computes exact hashes and perceptual hashes, flags web/screenshot sources, and moves
risky items into a quarantine folder to avoid accidental leakage into eval/train sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from PIL import Image, ImageOps
import numpy as np

from scripts.prepare_grouped_runtime_dataset import SOURCE_LIKE_WEBSITE_KEYWORDS, EVAL_RISK_KEYWORDS


PHASH_SIZE = 8


def _compute_exact_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compute_phash_hex(image: Image.Image) -> str:
    img = ImageOps.grayscale(image).resize((32, 32))
    pixels = np.asarray(img, dtype=np.float32)
    dct = np.fft.fft2(pixels)
    low_freq = np.abs(dct[: PHASH_SIZE + 1, : PHASH_SIZE + 1])
    med = np.median(low_freq[1:, 1:])
    bits = low_freq > med
    packed = 0
    for bit in bits.flatten()[: PHASH_SIZE * PHASH_SIZE]:
        packed = (packed << 1) | int(bool(bit))
    width = PHASH_SIZE * PHASH_SIZE // 4
    return f"{packed:0{width}x}"


def _has_web_like_source(path_like: str) -> bool:
    normalized = str(path_like or "").lower()
    return any(keyword in normalized for keyword in SOURCE_LIKE_WEBSITE_KEYWORDS)


def _has_eval_risk(path_like: str) -> bool:
    normalized = str(path_like or "").lower()
    return any(keyword in normalized for keyword in EVAL_RISK_KEYWORDS)


def fetch_bing_image_urls(query: str, count: int = 50) -> List[str]:
    key = os.environ.get("BING_SEARCH_KEY")
    endpoint = os.environ.get("BING_SEARCH_ENDPOINT")
    if not key or not endpoint:
        raise RuntimeError("BING_SEARCH_KEY and BING_SEARCH_ENDPOINT must be set for Bing queries")
    headers = {"Ocp-Apim-Subscription-Key": key}
    params = {"q": query, "count": min(150, int(count)), "imageType": "Photo"}
    url = endpoint.rstrip("/") + "/v7.0/images/search"
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    urls: List[str] = []
    for item in data.get("value", []):
        content_url = item.get("contentUrl") or item.get("thumbnailUrl")
        if content_url:
            urls.append(content_url)
    return urls


def fetch_duckduckgo_image_urls(query: str, count: int = 50) -> List[str]:
    # Lightweight DuckDuckGo image search fallback that doesn't require an API key.
    # Returns a list of image URLs. This relies on the public JSON endpoint used by
    # DuckDuckGo's frontend and may be rate-limited.
    try:
        session = requests.Session()
        params = {"q": query}
        resp = session.get("https://duckduckgo.com/", params=params, timeout=20)
        resp.raise_for_status()
        # extract vqd token
        text = resp.text
        token_idx = text.find("vqd='")
        if token_idx == -1:
            token_idx = text.find('vqd=\"')
        if token_idx == -1:
            return []
        token = None
        try:
            token = text[token_idx + 5 : text.index("'", token_idx + 5)]
        except Exception:
            try:
                token = text[token_idx + 5 : text.index('"', token_idx + 5)]
            except Exception:
                token = None
        if not token:
            return []
        img_url = "https://duckduckgo.com/i.js"
        urls: List[str] = []
        params = {"l": "us-en", "o": "json", "q": query, "vqd": token}
        while len(urls) < int(count):
            r = session.get(img_url, params=params, timeout=20, headers={"referer": "https://duckduckgo.com/"})
            r.raise_for_status()
            data = r.json()
            for item in data.get("results", []):
                u = item.get("image") or item.get("thumbnail") or item.get("url")
                if u:
                    urls.append(u)
                    if len(urls) >= int(count):
                        break
            next_url = data.get("next")
            if not next_url:
                break
            # next is a relative path; call again
            img_url = "https://duckduckgo.com" + next_url
            params = {}
        return urls
    except Exception:
        return []


def fetch_wikimedia_image_urls(query: str, count: int = 50) -> List[str]:
    try:
        api = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": str(min(50, int(count))),
            "prop": "imageinfo",
            "iiprop": "url",
        }
        resp = requests.get(api, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        urls: List[str] = []
        for page in pages.values():
            imageinfo = page.get("imageinfo")
            if imageinfo and isinstance(imageinfo, list):
                ii = imageinfo[0]
                url = ii.get("url")
                if url:
                    urls.append(url)
                    if len(urls) >= int(count):
                        break
        return urls
    except Exception:
        return []


def download_url(url: str, dest: Path, timeout: int = 30) -> Optional[Path]:
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(1024 * 16):
                if not chunk:
                    continue
                fh.write(chunk)
        return dest
    except Exception:
        return None


def stage_candidates_for_class(class_name: str, queries: Iterable[str], max_per_query: int, out_root: Path) -> Path:
    class_dir = out_root / class_name
    staging = class_dir / "staging"
    quarantine = class_dir / "quarantine"
    class_dir.mkdir(parents=True, exist_ok=True)
    (staging).mkdir(parents=True, exist_ok=True)
    (quarantine).mkdir(parents=True, exist_ok=True)

    candidates: List[Dict[str, Any]] = []
    for q in queries:
        urls: List[str] = []
        try:
            urls = fetch_bing_image_urls(q, count=max_per_query)
        except Exception:
            # Try DuckDuckGo fallback (no API key required)
            try:
                urls = fetch_duckduckgo_image_urls(q, count=max_per_query)
            except Exception:
                urls = []
        if not urls:
            # Try Wikimedia Commons fallback
            try:
                urls = fetch_wikimedia_image_urls(q, count=max_per_query)
            except Exception:
                urls = []
        if not urls:
            # write the query out for manual follow-up if no urls found
            continue
        for idx, url in enumerate(urls[:max_per_query]):
            fname = f"candidate_{int(time.time())}_{idx}.jpg"
            dest = staging / fname
            path = download_url(url, dest)
            if not path:
                continue
            try:
                with Image.open(path) as img:
                    image = ImageOps.exif_transpose(img.convert("RGB"))
                    phash = _compute_phash_hex(image)
                    ex_hash = _compute_exact_hash(path)
                    width, height = image.size
            except Exception:
                # Move unreadable file to quarantine
                path.replace(quarantine / path.name)
                continue
            risk_web = _has_web_like_source(url)
            risk_eval = _has_eval_risk(url)
            if risk_web or risk_eval:
                path.replace(quarantine / path.name)
                candidates.append(
                    {
                        "url": url,
                        "relative_path": str((quarantine / path.name).relative_to(out_root)),
                        "quarantined": True,
                        "web_like": bool(risk_web),
                        "eval_risk": bool(risk_eval),
                        "width": int(width),
                        "height": int(height),
                        "phash": phash,
                        "exact_hash": ex_hash,
                    }
                )
            else:
                candidates.append(
                    {
                        "url": url,
                        "relative_path": str(path.relative_to(out_root)),
                        "quarantined": False,
                        "web_like": False,
                        "eval_risk": False,
                        "width": int(width),
                        "height": int(height),
                        "phash": phash,
                        "exact_hash": ex_hash,
                    }
                )
    # write manifest
    manifest_path = class_dir / "internet_candidates.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(candidates, fh, ensure_ascii=False, indent=2)
    return class_dir


def build_queries_for_class(class_name: str, crop_name: Optional[str] = None) -> List[str]:
    base = [class_name, f"{class_name} plant", f"{class_name} disease"]
    if crop_name:
        base = [f"{class_name} {crop_name}", f"{class_name} {crop_name} plant disease"] + base
    # prefer high-resolution photos
    return [q + " photo" for q in base]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classes", nargs="+", required=True, help="Class names to fetch candidates for.")
    p.add_argument("--crop", type=str, default="", help="Crop name to include in queries.")
    p.add_argument("--out-root", type=Path, default=Path("outputs") / "internet_image_candidates")
    p.add_argument("--max-per-query", type=int, default=30)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    out_root: Path = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for cls in args.classes:
        queries = build_queries_for_class(cls, crop_name=args.crop or None)
        print(f"[AUG] staging candidates for class {cls}")
        class_dir = stage_candidates_for_class(cls, queries, args.max_per_query, out_root)
        print(f"[AUG] wrote manifest: {class_dir / 'internet_candidates.json'}")

    print("[AUG] Done. Review manifests and quarantine before merging into prepared datasets.")
    print("[AUG] If you do not have Bing keys, set BING_SEARCH_KEY and BING_SEARCH_ENDPOINT to enable automated fetching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
