#!/usr/bin/env python3
"""Audit existing tracked OE/OOD pools against the Turkey disease catalog."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_turkey_adapter_ood_catalog import ACTIVE_RANK_LIMIT, COVERAGE_TIERS, load_catalog  # noqa: E402

from src.data.dataset_release_runtime import DEFAULT_MATERIALIZED_DATASET_ROOT  # noqa: E402

ALIASES: dict[str, tuple[str, ...]] = {
    "tomato_bacterial_canker": ("bacterial_canker", "clavibacter", "bacterial_wilt"),
    "tomato_alternaria_fruit_rot": ("alternaria", "early_blight"),
    "tomato_fruit_cracking": ("fruit_cracking", "fruit_crack"),
    "tomato_buckeye_rot": ("buckeye", "phytophthora_fruit"),
    "grape_phomopsis": ("phomopsis", "dead_arm", "olukol"),
    "grape_black_rot": ("black_rot", "guignardia", "bidwellii"),
    "grape_gray_mold": ("gray_mold", "grey_mold", "botrytis"),
    "grape_bacterial_crown_gall": ("crown_gall", "agrobacterium", "allorhizobium"),
    "grape_sour_rot": ("sour_rot",),
    "grape_ripe_rot": ("ripe_rot", "colletotrichum"),
    "grape_blue_mold": ("blue_mold", "penicillium"),
    "grape_aspergillus_rot": ("aspergillus",),
    "strawberry_leaf_blight": ("leaf_blight", "phomopsis"),
    "strawberry_angular_leaf_spot": ("angular_leaf", "xanthomonas_fragariae"),
    "strawberry_anthracnose": ("anthracnose", "colletotrichum"),
    "strawberry_gray_mold": ("gray_mold", "grey_mold", "botrytis"),
    "strawberry_verticillium_wilt": ("verticillium",),
    "strawberry_fusarium_wilt": ("fusarium",),
    "strawberry_crown_rot": ("crown_rot", "phytophthora_cactorum"),
    "strawberry_leather_rot": ("leather_rot", "phytophthora_cactorum"),
    "strawberry_rhizopus_rot": ("rhizopus",),
    "strawberry_mucor_rot": ("mucor",),
    "strawberry_alternaria_rot": ("alternaria",),
    "strawberry_phomopsis_rot": ("phomopsis",),
    "strawberry_gnomonia_rot": ("gnomonia",),
    "strawberry_bacterial_angular_spot": ("angular_leaf", "xanthomonas_fragariae"),
    "apricot_bacterial_canker": ("bacterial_canker", "pseudomonas_syringae"),
    "apricot_powdery_mildew": ("powdery_mildew", "podosphaera"),
    "apricot_rust": ("rust", "tranzschelia"),
    "apricot_leaf_curl": ("leaf_curl", "taphrina"),
    "apricot_alternaria_spot": ("alternaria",),
    "apricot_verticillium_wilt": ("verticillium",),
    "apricot_cytospora_canker": ("cytospora",),
    "apricot_bacterial_spot": ("bacterial_spot", "xanthomonas_arboricola"),
    "apricot_rhizopus_rot": ("rhizopus",),
    "apricot_alternaria_rot": ("alternaria",),
    "apricot_blue_mold": ("blue_mold", "penicillium"),
    "apricot_sour_rot": ("sour_rot", "geotrichum"),
}


def normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return "_".join("".join(char if char.isalnum() else " " for char in decomposed).split())


def materialized_paths(dataset_root: Path) -> list[str]:
    if not dataset_root.is_dir():
        return []
    return [f"/{path.relative_to(dataset_root).as_posix()}" for path in dataset_root.rglob("*") if path.is_file()]


def target_part_matches(path: str, target: str) -> bool:
    normalized = normalize(path.replace("\\", "/").split("/oe/", 1)[-1])
    normalized = normalized.replace(normalize(target), "")
    if target.endswith("__fruit"):
        return not ("leaf" in normalized and "fruit" not in normalized)
    if target.endswith("__leaf"):
        return not ("fruit" in normalized and "leaf" not in normalized)
    return True


def path_matches(path: str, disease_id: str, target: str = "") -> bool:
    normalized = normalize(path)
    return target_part_matches(path, target) and any(
        normalize(alias) in normalized for alias in ALIASES.get(disease_id, ())
    )


def staged_counts(manifest_root: Path | None) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = {}
    if manifest_root is None or not manifest_root.is_dir():
        return counts
    for path in sorted(manifest_root.glob("*_manifest.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (str(row.get("target") or ""), str(row.get("disease_id") or ""), str(row.get("role") or ""))
                counts[key] = counts.get(key, 0) + 1
    return counts


def audit(
    payload: dict[str, Any],
    *,
    dataset_root: Path = DEFAULT_MATERIALIZED_DATASET_ROOT,
    staged_manifest_root: Path | None = None,
) -> list[dict[str, Any]]:
    paths = materialized_paths(dataset_root)
    staged = staged_counts(staged_manifest_root)
    rows: list[dict[str, Any]] = []
    for target, diseases in sorted(payload["targets"].items()):
        oe_paths = [path for path in paths if f"/{target}/oe/" in path]
        split_manifest = dataset_root / target / "ood" / "ood_split_manifest.json"
        try:
            split_entries = json.loads(split_manifest.read_text(encoding="utf-8")).get("entries", {})
        except (OSError, ValueError):
            split_entries = {}
        for disease in diseases:
            if disease["status"] != "out_of_distribution" or disease["rank"] > ACTIVE_RANK_LIMIT:
                continue
            disease_id = disease["disease_id"]
            oe_count = sum(path_matches(path, disease_id, target) for path in oe_paths) + staged.get(
                (target, disease_id, "oe_train"), 0
            )
            dev_count = sum(
                path_matches(relative_path, disease_id, target) and entry.get("split") == "dev"
                for relative_path, entry in split_entries.items()
            ) + staged.get((target, disease_id, "ood_dev"), 0)
            test_count = sum(
                path_matches(relative_path, disease_id, target) and entry.get("split") == "test"
                for relative_path, entry in split_entries.items()
            ) + staged.get((target, disease_id, "ood_test"), 0)
            role_counts = {"oe_train": oe_count, "ood_dev": dev_count, "ood_test": test_count}
            tier_complete = {
                tier: all(role_counts[role] >= target_count for role, target_count in targets.items())
                for tier, targets in COVERAGE_TIERS.items()
            }
            coverage_tier = next(
                (tier for tier in ("strong", "good", "minimum") if tier_complete[tier]),
                "below_minimum",
            )
            rows.append(
                {
                    "target": target,
                    "disease_id": disease_id,
                    "name_tr": disease["name_tr"],
                    "existing_oe_train": oe_count,
                    "existing_ood_dev": dev_count,
                    "existing_ood_test": test_count,
                    "minimum_oe_gap_to_10": max(0, 10 - oe_count),
                    "minimum_ood_dev_gap_to_5": max(0, 5 - dev_count),
                    "minimum_ood_test_gap_to_5": max(0, 5 - test_count),
                    "good_oe_gap_to_20": max(0, 20 - oe_count),
                    "good_ood_dev_gap_to_10": max(0, 10 - dev_count),
                    "good_ood_test_gap_to_10": max(0, 10 - test_count),
                    "oe_gap_to_30": max(0, 30 - oe_count),
                    "ood_dev_gap_to_15": max(0, 15 - dev_count),
                    "ood_test_gap_to_15": max(0, 15 - test_count),
                    "coverage_tier": coverage_tier,
                    "minimum_coverage_complete": tier_complete["minimum"],
                    "good_coverage_complete": tier_complete["good"],
                    "strong_coverage_complete": tier_complete["strong"],
                    "coverage_complete": tier_complete["minimum"],
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    minimum = sum(bool(row["minimum_coverage_complete"]) for row in rows)
    good = sum(bool(row["good_coverage_complete"]) for row in rows)
    strong = sum(bool(row["strong_coverage_complete"]) for row in rows)
    lines = [
        "# Türkiye Adaptör OOD Mevcut Veri Kapsamı",
        "",
        f"Kesin OOD hastalığı: **{len(rows)}** · minimum 10/5/5: **{minimum}** · "
        f"iyi 20/10/10: **{good}** · güçlü 30/15/15: **{strong}**",
        "",
        "> Sayımlar izlenen dosya yolları ve dondurulmuş `ood_split_manifest.json` atamalarından türetilir. Hastalık adı görünmeyen genel dilimler bu hastalık bazlı sayımlara dahil edilmez.",
        "",
        "| Adaptör | Hastalık | OE | Dev | Test | Minimum eksik (10/5/5) | Seviye |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        gaps = (
            f"{row['minimum_oe_gap_to_10']}/{row['minimum_ood_dev_gap_to_5']}/"
            f"{row['minimum_ood_test_gap_to_5']}"
        )
        lines.append(
            f"| `{row['target']}` | {row['name_tr']} | {row['existing_oe_train']} | {row['existing_ood_dev']} | "
            f"{row['existing_ood_test']} | {gaps} | `{row['coverage_tier']}` |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("docs/research/turkey_adapter_disease_catalog.json"))
    parser.add_argument("--csv", type=Path, default=Path("docs/research/turkey_adapter_ood_coverage.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/research/turkey_adapter_ood_coverage.md"))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_MATERIALIZED_DATASET_ROOT)
    parser.add_argument(
        "--staged-manifest-root",
        type=Path,
        default=Path("data/internet_image_candidates/turkey_adapter_ood"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = audit(
        load_catalog(args.catalog),
        dataset_root=args.dataset_root,
        staged_manifest_root=args.staged_manifest_root,
    )
    write_csv(rows, args.csv)
    write_markdown(rows, args.markdown)
    print(
        f"PASS: ood_diseases={len(rows)} "
        f"minimum={sum(row['minimum_coverage_complete'] for row in rows)} "
        f"good={sum(row['good_coverage_complete'] for row in rows)} "
        f"strong={sum(row['strong_coverage_complete'] for row in rows)}"
    )
    print(f"csv={args.csv} markdown={args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
