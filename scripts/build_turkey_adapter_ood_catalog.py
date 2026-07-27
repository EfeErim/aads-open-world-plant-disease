#!/usr/bin/env python3
"""Validate and render the Turkey-specific adapter disease/OOD catalog."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

VALID_STATUSES = {"in_distribution", "out_of_distribution", "ambiguous", "organ_mismatch"}
VALID_EVIDENCE = {"high", "medium", "limited"}
ACTIVE_RANK_LIMIT = 8
COVERAGE_TIERS = {
    "minimum": {"oe_train": 10, "ood_dev": 5, "ood_test": 5},
    "good": {"oe_train": 20, "ood_dev": 10, "ood_test": 10},
    "strong": {"oe_train": 30, "ood_dev": 15, "ood_test": 15},
}
ROLE_TARGETS = COVERAGE_TIERS["minimum"]


def load_catalog(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "v1_turkey_adapter_disease_catalog":
        raise ValueError("unsupported catalog schema")
    sources = payload.get("sources") or {}
    targets = payload.get("targets") or {}
    if len(targets) != 8:
        raise ValueError(f"expected 8 targets, found {len(targets)}")
    for target, diseases in targets.items():
        if len(diseases) != 10:
            raise ValueError(f"{target}: expected 10 diseases, found {len(diseases)}")
        if [row.get("rank") for row in diseases] != list(range(1, 11)):
            raise ValueError(f"{target}: ranks must be exactly 1..10")
        disease_ids = [str(row.get("disease_id") or "") for row in diseases]
        if len(set(disease_ids)) != len(disease_ids) or not all(disease_ids):
            raise ValueError(f"{target}: disease_id values must be non-empty and unique")
        for row in diseases:
            if row.get("status") not in VALID_STATUSES:
                raise ValueError(f"{target}/{row.get('disease_id')}: invalid status")
            if row.get("evidence") not in VALID_EVIDENCE:
                raise ValueError(f"{target}/{row.get('disease_id')}: invalid evidence")
            if row.get("status") == "in_distribution" and not row.get("id_class"):
                raise ValueError(f"{target}/{row.get('disease_id')}: ID row is missing id_class")
            unknown_sources = sorted(set(row.get("sources") or []) - set(sources))
            if unknown_sources:
                raise ValueError(f"{target}/{row.get('disease_id')}: unknown sources {unknown_sources}")
    return payload


def flat_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target, diseases in sorted(payload["targets"].items()):
        for disease in diseases:
            rows.append(
                {
                    "target_id": target,
                    "rank": disease["rank"],
                    "disease_id": disease["disease_id"],
                    "name_tr": disease["name_tr"],
                    "pathogen": disease["pathogen"],
                    "evidence": disease["evidence"],
                    "distribution_status": disease["status"],
                    "active_collection_scope": disease["rank"] <= ACTIVE_RANK_LIMIT,
                    "id_class": disease.get("id_class") or "",
                    "source_ids": ";".join(disease["sources"]),
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Türkiye Adaptör Hastalık Kataloğu",
        "",
        "> Bu sıralama ulusal prevalans sayımı değildir. Resmî önem, Türkiye saha bulguları ve ekonomik önem birlikte değerlendirilmiştir.",
        "",
    ]
    for target, diseases in sorted(payload["targets"].items()):
        counts = Counter(row["status"] for row in diseases)
        lines.extend(
            [
                f"## `{target}`",
                "",
                f"ID: **{counts['in_distribution']}** · OOD adayı: **{counts['out_of_distribution']}** · Belirsiz/organ dışı: **{counts['ambiguous'] + counts['organ_mismatch']}**",
                "",
                "| Sıra | Hastalık | Etmen | Kanıt | Durum | Mevcut sınıf |",
                "|---:|---|---|---|---|---|",
            ]
        )
        for row in diseases:
            lines.append(
                f"| {row['rank']} | {row['name_tr']} | {row['pathogen']} | {row['evidence']} | "
                f"`{row['status']}` | `{row.get('id_class') or '-'}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## OOD/OE veri yeterlilik seviyeleri",
            "",
            "Her doğrulanmış OOD hastalığı için aktif kabul eşiği minimum seviyedir; daha yüksek seviyeler "
            "veri bulunabildiğinde kalite artış hedefidir:",
            "",
            "- Minimum kabul: `10 OE + 5 OOD-dev + 5 OOD-test = 20`",
            "- İyi seviye: `20 OE + 10 OOD-dev + 10 OOD-test = 40`",
            "- Güçlü hedef: `30 OE + 15 OOD-dev + 15 OOD-test = 60`",
            "",
            "Sayıyı tamamlamak için yanlış organ, belirsiz hastalık, semantik kopya veya rol sızıntısı kabul edilmez. "
            "OE yalnız eğitim/takviye havuzudur; OOD-dev eşik seçimi, OOD-test ise kilitli değerlendirme içindir.",
            "",
            "Aktif koleksiyon kapsamı her adaptörde Türkiye sıralamasındaki ilk 8 hastalıktır. Rank 9-10 kayıtları "
            "araştırma geçmişi ve mevcut ek veri olarak katalogda korunur, fakat minimum tamamlama hesabına girmez.",
            "",
            "## Kaynaklar",
            "",
        ]
    )
    for source_id, source in payload["sources"].items():
        lines.append(f"- `{source_id}` — [{source['title']}]({source['url']}) ({source['kind']})")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_collection_plan(payload: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for target, diseases in sorted(payload["targets"].items()):
        for disease in diseases:
            if disease["status"] != "out_of_distribution" or disease["rank"] > ACTIVE_RANK_LIMIT:
                continue
            for role, target_count in ROLE_TARGETS.items():
                rows.append(
                    {
                        "target": target,
                        "disease_id": disease["disease_id"],
                        "name_tr": disease["name_tr"],
                        "pathogen": disease["pathogen"],
                        "role": role,
                        "ood_type": "same_crop_unsupported_disease",
                        "target_count": target_count,
                        "minimum_target_count": COVERAGE_TIERS["minimum"][role],
                        "good_target_count": COVERAGE_TIERS["good"][role],
                        "strong_target_count": COVERAGE_TIERS["strong"][role],
                        "accepted_count": 0,
                        "remaining_count": target_count,
                        "review_status": "collection_pending",
                    }
                )
    write_csv(rows, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("docs/research/turkey_adapter_disease_catalog.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/research/turkey_adapter_disease_catalog.md"))
    parser.add_argument("--csv", type=Path, default=Path("docs/research/turkey_adapter_disease_catalog.csv"))
    parser.add_argument("--collection-plan", type=Path, default=Path("data/internet_image_candidates/turkey_adapter_ood/collection_plan.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_catalog(args.catalog)
    rows = flat_rows(payload)
    write_csv(rows, args.csv)
    write_markdown(payload, args.markdown)
    write_collection_plan(payload, args.collection_plan)
    ood_count = sum(row["distribution_status"] == "out_of_distribution" for row in rows)
    print(f"PASS: targets=8 diseases={len(rows)} ood_diseases={ood_count}")
    print(f"markdown={args.markdown} csv={args.csv} collection_plan={args.collection_plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
