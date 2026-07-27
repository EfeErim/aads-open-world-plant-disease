#!/usr/bin/env python3
"""Build the expanded M2 internet image set from disease-focused internet photos."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IMAGE_EXT = ".jpg"


@dataclass(frozen=True)
class ImagePlan:
    expected_target: str
    taxon_name: str
    query: str
    count: int
    expected_behavior: str
    notes: str
    fallback_queries: tuple[str, ...] = ()


SUPPORTED_BEHAVIOR = "internet disease-focused supported crop/part; disease answer or review expected, no crash"
UNKNOWN_BEHAVIOR = "internet diseased or damaged supported crop/part; unknown, OOD, or review acceptable"
UNSUPPORTED_BEHAVIOR = "internet unsupported crop/part or irrelevant subject; unknown/review expected, no disease label"

PLAN: tuple[ImagePlan, ...] = (
    ImagePlan(
        "tomato__fruit",
        "Solanum lycopersicum",
        "tomato anthracnose fruit",
        10,
        SUPPORTED_BEHAVIOR,
        "disease-focused tomato fruit",
        ("tomato blossom end rot fruit", "tomato late blight fruit", "tomato fruit disease", "tomato rot fruit"),
    ),
    ImagePlan(
        "tomato__leaf",
        "Solanum lycopersicum",
        "tomato late blight leaf",
        10,
        SUPPORTED_BEHAVIOR,
        "disease-focused tomato leaf",
        ("tomato early blight leaf", "tomato septoria leaf spot", "tomato powdery mildew leaf"),
    ),
    ImagePlan(
        "strawberry__fruit",
        "Fragaria x ananassa",
        "strawberry botrytis fruit",
        10,
        SUPPORTED_BEHAVIOR,
        "disease-focused strawberry fruit",
        ("strawberry gray mold fruit", "strawberry anthracnose fruit", "strawberry fruit disease"),
    ),
    ImagePlan(
        "strawberry__leaf",
        "Fragaria x ananassa",
        "strawberry leaf scorch",
        10,
        SUPPORTED_BEHAVIOR,
        "disease-focused strawberry leaf",
        ("strawberry leaf spot", "strawberry powdery mildew leaf", "strawberry leaf disease"),
    ),
    ImagePlan(
        "grape__fruit",
        "Vitis vinifera",
        "grape powdery mildew fruit",
        10,
        SUPPORTED_BEHAVIOR,
        "disease-focused grape fruit",
        ("grape botrytis fruit", "grape black rot fruit", "grape fruit disease"),
    ),
    ImagePlan(
        "grape__leaf",
        "Vitis vinifera",
        "grape downy mildew leaf",
        10,
        SUPPORTED_BEHAVIOR,
        "disease-focused grape leaf",
        ("grape powdery mildew leaf", "grape esca leaf", "grape leafroll virus"),
    ),
    ImagePlan(
        "apricot__fruit",
        "Prunus armeniaca",
        "plum pox apricot fruit",
        10,
        SUPPORTED_BEHAVIOR,
        "disease-focused apricot fruit",
        ("apricot brown rot fruit", "monilinia apricot fruit", "Monilinia fructicola", "apricot fruit disease"),
    ),
    ImagePlan(
        "apricot__leaf",
        "Prunus armeniaca",
        "shot hole disease apricot leaf",
        10,
        SUPPORTED_BEHAVIOR,
        "disease-focused apricot leaf",
        ("Wilsonomyces carpophilus", "apricot leaf spot disease", "apricot leaf disease"),
    ),
    ImagePlan(
        "tomato__leaf",
        "Solanum lycopersicum",
        "tomato diseased plant",
        2,
        UNKNOWN_BEHAVIOR,
        "disease-focused uncertain tomato leaf",
        ("tomato plant disease",),
    ),
    ImagePlan(
        "grape__leaf",
        "Vitis vinifera",
        "diseased grape leaf",
        2,
        UNKNOWN_BEHAVIOR,
        "disease-focused uncertain grape leaf",
        ("grape leaf disease",),
    ),
    ImagePlan(
        "strawberry__fruit",
        "Fragaria x ananassa",
        "diseased strawberry fruit",
        2,
        UNKNOWN_BEHAVIOR,
        "disease-focused uncertain strawberry fruit",
        ("strawberry disease",),
    ),
    ImagePlan(
        "apricot__leaf",
        "Prunus armeniaca",
        "diseased apricot leaf",
        2,
        UNKNOWN_BEHAVIOR,
        "disease-focused uncertain apricot leaf",
        ("apricot disease",),
    ),
    ImagePlan("unknown_crop", "Malus domestica", "apple scab leaf", 2, UNSUPPORTED_BEHAVIOR, "diseased unsupported apple leaf"),
    ImagePlan(
        "unknown_crop",
        "Capsicum annuum",
        "pepper bacterial spot fruit",
        2,
        UNSUPPORTED_BEHAVIOR,
        "diseased unsupported pepper fruit",
        ("pepper disease fruit", "pepper leaf spot"),
    ),
    ImagePlan("non_plant", "Fungi", "mold fungus", 4, UNSUPPORTED_BEHAVIOR, "non-plant disease-like distractor"),
)


def _json_get(url: str, *, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "aads-m2-demo-set-builder/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, output_path: Path, *, timeout: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "aads-m2-demo-set-builder/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    if len(payload) < 1024:
        raise ValueError(f"Downloaded file is too small from {url}")
    output_path.write_bytes(payload)


def _photo_url(photo: dict[str, Any]) -> str:
    url = str(photo.get("url") or "")
    if not url:
        return ""
    return url.replace("/square.", "/large.").replace("square.jpg", "large.jpg")


def _fetch_observations(plan: ImagePlan, *, page: int, per_page: int, timeout: int) -> list[dict[str, Any]]:
    return _fetch_observations_for_query(plan, query=plan.query, page=page, per_page=per_page, timeout=timeout)


def _fetch_observations_for_query(
    plan: ImagePlan,
    *,
    query: str,
    page: int,
    per_page: int,
    timeout: int,
) -> list[dict[str, Any]]:
    payload = {
        "photos": "true",
        "per_page": per_page,
        "page": page,
        "order_by": "created_at",
        "order": "desc",
    }
    if query:
        payload["q"] = query
    elif plan.taxon_name:
        payload["taxon_name"] = plan.taxon_name
    params = urllib.parse.urlencode(
        payload
    )
    url = f"https://api.inaturalist.org/v1/observations?{params}"
    return list(_json_get(url, timeout=timeout).get("results", []))


def build_image_set(
    *,
    output_dir: Path,
    manifest_path: Path,
    start_id: int,
    per_page: int,
    max_pages: int,
    timeout: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen_photo_ids: set[str] = set()
    next_id = int(start_id)

    for plan in PLAN:
        collected_for_plan = 0
        for query in (plan.query, *plan.fallback_queries):
            for page in range(1, max_pages + 1):
                if collected_for_plan >= plan.count:
                    break
                try:
                    observations = _fetch_observations_for_query(
                        plan,
                        query=query,
                        page=page,
                        per_page=per_page,
                        timeout=timeout,
                    )
                except Exception as exc:
                    print(
                        f"[WARN] query failed target={plan.expected_target} "
                        f"query={query!r} page={page}: {exc}"
                    )
                    break
                if not observations:
                    break
                for observation in observations:
                    if collected_for_plan >= plan.count:
                        break
                    photos = observation.get("observation_photos") or []
                    for photo_record in photos:
                        photo = photo_record.get("photo") or {}
                        photo_id = str(photo.get("id") or photo.get("uuid") or photo.get("url") or "")
                        if not photo_id or photo_id in seen_photo_ids:
                            continue
                        source_url = _photo_url(photo)
                        if not source_url:
                            continue
                        image_id = f"demo_{next_id:03d}"
                        target_dir = output_dir / plan.expected_target
                        target_dir.mkdir(parents=True, exist_ok=True)
                        filename = f"{image_id}_{plan.expected_target.replace('__', '_')}{IMAGE_EXT}"
                        output_path = target_dir / filename
                        try:
                            _download(source_url, output_path, timeout=timeout)
                        except Exception as exc:
                            print(f"[WARN] download failed {source_url}: {exc}")
                            continue
                        seen_photo_ids.add(photo_id)
                        collected_for_plan += 1
                        rows.append(
                            {
                                "image_id": image_id,
                                "source": f"staged_external:{output_path.as_posix()}",
                                "expected_target": plan.expected_target,
                                "expected_crop": plan.expected_target.split("__", 1)[0]
                                if "__" in plan.expected_target
                                else "",
                                "expected_part": plan.expected_target.split("__", 1)[1]
                                if "__" in plan.expected_target
                                else "",
                                "expected_class": "",
                                "expected_behavior": plan.expected_behavior,
                                "notes": plan.notes,
                                "origin_url": str(observation.get("uri") or ""),
                                "photo_url": source_url,
                                "taxon_name": plan.taxon_name,
                                "query": query,
                                "license_code": str(photo.get("license_code") or ""),
                                "attribution": str(photo.get("attribution") or ""),
                            }
                        )
                        print(f"[OK] {image_id} {plan.expected_target} <- {source_url}")
                        next_id += 1
                        if sleep_seconds > 0:
                            time.sleep(sleep_seconds)
                        break
            if collected_for_plan >= plan.count:
                break

        if collected_for_plan < plan.count:
            print(
                f"[WARN] collected {collected_for_plan}/{plan.count} for "
                f"{plan.expected_target} taxon={plan.taxon_name!r} query={plan.query!r}"
            )

    fieldnames = [
        "image_id",
        "source",
        "expected_target",
        "expected_crop",
        "expected_part",
        "expected_class",
        "expected_behavior",
        "notes",
        "origin_url",
        "photo_url",
        "taxon_name",
        "query",
        "license_code",
        "attribution",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "schema_version": "v1_m2_internet_image_set",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "target_count": sum(plan.count for plan in PLAN),
        "downloaded_count": len(rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".runtime_tmp/final_demo_images/internet_expansion"))
    parser.add_argument("--manifest", type=Path, default=Path(".runtime_tmp/m2_internet_image_set_manifest.csv"))
    parser.add_argument("--summary", type=Path, default=Path(".runtime_tmp/m2_internet_image_set_summary.json"))
    parser.add_argument("--start-id", type=int, default=49)
    parser.add_argument("--per-page", type=int, default=80)
    parser.add_argument("--max-pages", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_image_set(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        start_id=args.start_id,
        per_page=args.per_page,
        max_pages=args.max_pages,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["downloaded_count"] == summary["target_count"] else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
