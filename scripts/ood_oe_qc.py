import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OOD_REPORT = ROOT / "data" / "ood_dataset" / "_ood_optimization_report.json"
OE_SUMMARY = ROOT / "data" / "oe_dataset" / "_oe_summary.json"
OOD_FINAL_DIR = ROOT / "data" / "ood_dataset" / "final"
OE_DIR = ROOT / "data" / "oe_dataset"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def load_json(p: Path):
    # support files with BOM
    with p.open(encoding="utf-8-sig") as f:
        return json.load(f)


def find_final_folder_for_source(source_name: str):
    # heuristic: split source into words and find a final folder that contains them
    parts = source_name.split("_")
    folders = list(OOD_FINAL_DIR.iterdir()) if OOD_FINAL_DIR.exists() else []
    for folder in folders:
        name = folder.name.lower()
        if all(any(part.lower() in name for part in [parts[i], parts[i+1]] if i+1 < len(parts))
               for i in range(0, len(parts)-1, 2)):
            return folder
    # fallback: return any folder that contains first part
    for folder in folders:
        if parts[0].lower() in folder.name.lower():
            return folder
    return None


def sample_files(folder: Path, n=5):
    files = [p for p in folder.rglob("*.*") if p.is_file()]
    if not files:
        return []
    return [str(p.relative_to(ROOT)) for p in random.sample(files, min(n, len(files)))]


def analyze_ood(report):
    flagged = []
    lines = []
    for rec in report:
        copied = rec.get("copied_images", 0)
        slices = rec.get("source_slices_after_dedup", {}) or {}
        # count problematic slice types
        problem_keys = [k for k in slices.keys() if any(tok in k for tok in ["unsupported", "off_crop", "other_crop", "non_plant", "blur", "occlusion", "failure"]) ]
        problem_count = sum(slices.get(k, 0) for k in problem_keys)
        pct_problem = (problem_count / copied) if copied else 0.0
        low_count_flag = copied < 200
        high_problem_flag = pct_problem > 0.25
        folder = find_final_folder_for_source(rec.get("source", ""))
        samples = sample_files(folder) if folder else []
        lines.append({
            "source": rec.get("source"),
            "copied": copied,
            "problem_count": problem_count,
            "pct_problem": round(pct_problem, 3),
            "flags": {
                "low_count": low_count_flag,
                "high_problem_ratio": high_problem_flag
            },
            "samples": samples,
            "final_folder": str(folder.relative_to(ROOT)) if folder else None,
            "slices": slices,
        })
        if low_count_flag or high_problem_flag:
            flagged.append(lines[-1])
    return flagged, lines


def analyze_oe(summary):
    flagged = []
    lines = []
    for name, info in summary.items():
        images = info.get("images", 0)
        slices = info.get("slices", {}) or {}
        # low diversity if single slice equals total images
        single_slice_eq = any(v == images for v in slices.values()) and len(slices) == 1
        folder = OE_DIR
        lines.append({
            "name": name,
            "images": images,
            "slices": slices,
            "single_slice_all": single_slice_eq
        })
        if single_slice_eq or images < 200:
            flagged.append(lines[-1])
    return flagged, lines


def write_report(ood_flagged, ood_all, oe_flagged, oe_all):
    out = OUT_DIR / "ood_oe_qc_report.md"
    with out.open("w", encoding="utf-8") as f:
        f.write("# OOD / OE QC Report\n\n")
        f.write("## Flagged OOD sources\n\n")
        for r in ood_flagged:
            f.write(f"- **{r['source']}**: copied={r['copied']}, problem_count={r['problem_count']}, pct_problem={r['pct_problem']}\n")
            if r['final_folder']:
                f.write(f"  - final_folder: {r['final_folder']}\n")
            if r['samples']:
                f.write("  - sample files:\n")
                for s in r['samples']:
                    f.write(f"    - {s}\n")
        f.write("\n## All OOD sources (summary)\n\n")
        for r in ood_all:
            f.write(f"- {r['source']}: copied={r['copied']}, pct_problem={r['pct_problem']}\n")

        f.write("\n## Flagged OE entries\n\n")
        for e in oe_flagged:
            f.write(f"- **{e['name']}**: images={e['images']}, single_slice_all={e['single_slice_all']}\n")

        f.write("\n## OE summary\n\n")
        for e in oe_all:
            f.write(f"- {e['name']}: images={e['images']}\n")

    print(f"Report written to: {out}")


def main():
    ood = load_json(OOD_REPORT) if OOD_REPORT.exists() else []
    oe = load_json(OE_SUMMARY) if OE_SUMMARY.exists() else {}
    ood_flagged, ood_all = analyze_ood(ood)
    oe_flagged, oe_all = analyze_oe(oe)
    write_report(ood_flagged, ood_all, oe_flagged, oe_all)


if __name__ == "__main__":
    main()
