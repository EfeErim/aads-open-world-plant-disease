#!/usr/bin/env python3
"""
Fetch recent arXiv papers for repo-relevant keywords and refresh the guide.

Intended to be run from CI on a schedule. Does NOT push by itself; the workflow
commits and pushes the updated guide to the current branch.

Usage:
  python scripts/update_sota_references.py --output docs/SOTA_AUTOMATION_GUIDE.md

"""
import argparse
import datetime
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

QUERIES = [
    "out-of-distribution detection",
    "energy based ood",
    "mahalanobis ood",
    "logitnorm",
    "selective prediction",
    "segment anything",
    "sam segmentation",
    "bioclip",
    "router calibration",
    "conformal prediction",
]


ARXIV_API = "http://export.arxiv.org/api/query"
CANDIDATE_SECTION_BEGIN = "<!-- BEGIN SOTA AUTOMATION CANDIDATES -->"
CANDIDATE_SECTION_END = "<!-- END SOTA AUTOMATION CANDIDATES -->"
DEFAULT_OPPORTUNITY_ROOTS = (".github", "scripts", "src", "tests", "docs", "config", "skills")
TEXT_FILE_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_DIRS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".runtime_tmp", ".venv", "__pycache__"}
IMPROVEMENT_MARKER_PATTERN = re.compile(r"\b(?P<tag>TODO|FIXME|HACK|XXX|BUG)\b\s*[:\-]\s*(?P<detail>.*)")
SRC_TO_SCRIPTS_IMPORT_PATTERN = re.compile(r"^\s*(?:from\s+scripts(?:\.|\s+import\b)|import\s+scripts(?:\.|\s|$))")
RELEVANCE_TERMS = (
    "adapter",
    "bioclip",
    "calibration",
    "conformal",
    "disease",
    "energy score",
    "energy-based",
    "mahalanobis",
    "ood",
    "out-of-distribution",
    "plant",
    "risk-coverage",
    "router",
    "selective prediction",
    "segment anything",
)
QUERY_RELEVANCE_TERMS = {
    "out-of-distribution detection": ("out-of-distribution", "ood"),
    "energy based ood": ("energy-based", "energy based", "energy score", "out-of-distribution", "ood"),
    "mahalanobis ood": ("mahalanobis", "out-of-distribution", "ood"),
    "logitnorm": ("logitnorm", "logit normalization", "out-of-distribution", "ood"),
    "selective prediction": ("selective prediction", "risk-coverage", "abstention"),
    "segment anything": ("segment anything", "sam 2", "sam2", "sam 3", "sam3"),
    "sam segmentation": ("segment anything", "sam 2", "sam2", "sam 3", "sam3"),
    "bioclip": ("bioclip",),
    "router calibration": ("router calibration", "calibrated router", "router handoff"),
    "conformal prediction": ("conformal prediction",),
}
TITLE_SCOPED_QUERIES = {
    "out-of-distribution detection",
    "energy based ood",
    "mahalanobis ood",
    "logitnorm",
    "selective prediction",
    "segment anything",
    "sam segmentation",
    "router calibration",
    "conformal prediction",
}
VISUAL_CONTEXT_TERMS = ("image", "visual", "vision", "pixel", "segmentation", "plant", "crop", "leaf")
BIOCLIP_CONTEXT_TERMS = ("plant", "crop", "leaf", "disease", "flora", "botanical", "vegetation")


def query_arxiv(q, max_results=5):
    params = {
        "search_query": f"all:{q}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    r = requests.get(ARXIV_API, params=params, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        link = entry.find("atom:id", ns).text.strip()
        published = entry.find("atom:published", ns).text.strip()
        authors = [a.find("atom:name", ns).text.strip() for a in entry.findall("atom:author", ns)]
        items.append({"title": title, "summary": summary, "link": link, "published": published, "authors": authors})
    return items


def load_existing_titles(guide_path):
    if not os.path.exists(guide_path):
        return set()
    text = open(guide_path, "r", encoding="utf-8").read()
    titles = set()
    # crude heuristic: find lines in Literature Anchors tables or bullets that look like titles
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) < 200 and (line.startswith("|") or line.startswith("- ") or line.startswith("* ")):
            # remove table pipes
            cleaned = re.sub(r"[|*\-]", "", line).strip()
            if cleaned and len(cleaned) > 10:
                titles.add(cleaned.lower())
    return titles


def is_relevant_candidate(item):
    title = item.get("title", "").lower()
    text = " ".join([title, item.get("summary", "")]).lower()
    query = item.get("query", "")
    query_terms = QUERY_RELEVANCE_TERMS.get(query)
    if query_terms:
        haystack = title if query in TITLE_SCOPED_QUERIES else text
        if not any(_contains_term(haystack, term) for term in query_terms):
            return False
        if query == "bioclip":
            return any(term in text for term in BIOCLIP_CONTEXT_TERMS)
        return query == "router calibration" or any(_contains_term(text, term) for term in VISUAL_CONTEXT_TERMS)
    for term in RELEVANCE_TERMS:
        if _contains_term(text, term):
            return True
    return False


def _contains_term(text, term):
    if term == "ood":
        return re.search(r"\bood\b", text) is not None
    pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def summarize_query_error(error):
    error_text = str(error)
    if "WinError 10013" in error_text:
        return "network access blocked by local permissions"
    if "timed out" in error_text.lower():
        return "query timed out"
    return error_text.splitlines()[0][:200]


def infer_repo_action_hint(item):
    """Map a literature candidate to a concrete review action for this repo."""
    query = item.get("query", "").lower()
    title = item.get("title", "").lower()
    summary = item.get("summary", "").lower()
    text = " ".join([query, title, summary])

    if "fine-grained" in text and ("ood" in text or "out-of-distribution" in text):
        return (
            "Audit OOD readiness for near-OOD slices, background/style confounders, and feature-space detector "
            "separation before adopting a new architecture."
        )
    if "bioclip" in text or ("plant" in text and "self-supervised" in text):
        return (
            "Review plant-domain representation and augmentation policy; avoid transformations that erase subtle "
            "leaf/symptom cues unless a local ablation proves they help."
        )
    if "selective prediction" in text or "risk-coverage" in text:
        return "Check router abstention with risk-coverage curves instead of relying on top-1 accuracy alone."
    if "calibration" in text:
        return "Route to router/OOD calibration checks and compare threshold recommendations against current artifacts."
    if "conformal" in text:
        return "Review exchangeability and split boundaries before promoting conformal guarantees into readiness policy."
    if "segment anything" in text or "sam" in text:
        return "Validate segmentation-assisted routing on current crop/part eval data before changing router prompts."
    return "Review for a concrete change to ML method, evaluation policy, data curation, or guard behavior before promotion."


def render_candidate_section(found, generated_at, query_errors=None, extra_sections=None):
    query_errors = query_errors or []
    extra_sections = extra_sections or []
    out_lines = [
        CANDIDATE_SECTION_BEGIN,
        "#### Latest Automated Candidate Scan",
        "",
        f"Generated: `{generated_at}`",
        "",
        "These are machine-collected literature candidates for human review. They are not accepted repo guidance until a maintainer promotes them into the relevant Literature Anchors table above.",
        "",
    ]
    if query_errors:
        out_lines.append("Candidate scan could not query all configured sources:")
        out_lines.append("")
        for query, error in query_errors:
            out_lines.append(f"- `{query}`: {summarize_query_error(error)}")
        out_lines.append("")

    if not found and not query_errors:
        out_lines.append("No new papers found for configured queries.")
    elif found:
        for it in found:
            out_lines.append(f"##### {it['title']}")
            out_lines.append("")
            out_lines.append(f"- Query: `{it['query']}`")
            out_lines.append(f"- Published: `{it['published']}`")
            out_lines.append(f"- Authors: {', '.join(it['authors'])}")
            out_lines.append(f"- Link: {it['link']}")
            out_lines.append(f"- Repo action hint: {infer_repo_action_hint(it)}")
            out_lines.append(f"- Review note: {it['summary']}")
            out_lines.append("")
    for section in extra_sections:
        out_lines.extend(["", section])
    out_lines.extend(["", CANDIDATE_SECTION_END])
    return "\n".join(out_lines)


def _iter_text_files(repo_root, roots):
    repo_root = Path(repo_root)
    for root_name in roots:
        root = repo_root / root_name
        if not root.exists():
            continue
        files = [root] if root.is_file() else root.rglob("*")
        for path in files:
            if not path.is_file():
                continue
            relative_parts = path.relative_to(repo_root).parts
            if any(part in SKIP_DIRS for part in relative_parts):
                continue
            if path.suffix.lower() not in TEXT_FILE_SUFFIXES:
                continue
            yield path


def _relative_posix(path, repo_root):
    return Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()


def scan_repo_opportunities(repo_root, max_items=25, roots=DEFAULT_OPPORTUNITY_ROOTS):
    """Find lightweight repo-local bug, weak-point, and suboptimal-code signals."""
    repo_root = Path(repo_root)
    opportunities = []

    stale_candidate_path = repo_root / "docs" / "SOTA_AUTOMATION_UPDATES.md"
    if stale_candidate_path.exists():
        opportunities.append(
            {
                "kind": "weak_point",
                "file": "docs/SOTA_AUTOMATION_UPDATES.md",
                "line": 1,
                "summary": "stale standalone SOTA candidate report exists; guide is the canonical target",
            }
        )

    workflow_path = repo_root / ".github" / "workflows" / "sota_auto_update.yml"
    if workflow_path.exists():
        workflow_text = workflow_path.read_text(encoding="utf-8", errors="replace")
        if "SOTA_AUTOMATION_UPDATES.md" in workflow_text:
            opportunities.append(
                {
                    "kind": "bug",
                    "file": ".github/workflows/sota_auto_update.yml",
                    "line": workflow_text[: workflow_text.index("SOTA_AUTOMATION_UPDATES.md")].count("\n") + 1,
                    "summary": "workflow writes standalone update report instead of the canonical guide",
                }
            )

    for path in _iter_text_files(repo_root, roots):
        if len(opportunities) >= max_items:
            break
        rel_path = _relative_posix(path, repo_root)
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if len(opportunities) >= max_items:
                break
            if path.suffix.lower() == ".py":
                opportunities.extend(_scan_python_line_for_code_quality(rel_path, line_number, line))
                if len(opportunities) >= max_items:
                    break
            if path.suffix.lower() == ".py" and not line.lstrip().startswith("#"):
                continue
            marker = IMPROVEMENT_MARKER_PATTERN.search(line)
            if not marker:
                continue
            detail = marker.group("detail").strip(" -:#")
            summary = f"{marker.group('tag').upper()} marker"
            if detail:
                summary = f"{summary}: {detail[:160]}"
            opportunities.append(
                {
                    "kind": "improvement",
                    "file": rel_path,
                    "line": line_number,
                    "summary": summary,
                }
            )
            if len(opportunities) >= max_items:
                break

    return opportunities[:max_items]


def _scan_python_line_for_code_quality(rel_path, line_number, line):
    """Return high-signal code-quality opportunities for one Python line."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return []

    opportunities = []
    if rel_path.startswith("src/") and SRC_TO_SCRIPTS_IMPORT_PATTERN.search(line):
        opportunities.append(
            {
                "kind": "bug",
                "file": rel_path,
                "line": line_number,
                "summary": "src imports scripts; durable logic must not depend on orchestration surfaces",
            }
        )
    if re.search(r"\bexcept\s+(?:Exception|BaseException)\s*:\s*pass(?:\s*#.*)?$", line) or re.search(
        r"\bexcept\s*:\s*pass(?:\s*#.*)?$",
        line,
    ):
        opportunities.append(
            {
                "kind": "weak_point",
                "file": rel_path,
                "line": line_number,
                "summary": "broad exception is swallowed with pass; preserve error context or narrow the exception",
            }
        )
    return opportunities


def render_repo_opportunity_scan(opportunities):
    out_lines = ["#### Repo Bug / Weak Point / Suboptimal Code Scan", ""]
    if not opportunities:
        out_lines.append("No lightweight repo-local bug, weak-point, or suboptimal-code signals found in the configured roots.")
        return "\n".join(out_lines)

    out_lines.append("Machine-collected candidates for triage. Review before treating any item as a confirmed defect.")
    out_lines.append("")
    for item in opportunities:
        out_lines.append(f"- `{item['kind']}` [{item['file']}:{item['line']}]: {item['summary']}")
    return "\n".join(out_lines)


def update_guide_with_candidate_section(guide_text, candidate_section):
    if CANDIDATE_SECTION_BEGIN in guide_text and CANDIDATE_SECTION_END in guide_text:
        pattern = re.compile(
            rf"{re.escape(CANDIDATE_SECTION_BEGIN)}.*?{re.escape(CANDIDATE_SECTION_END)}",
            flags=re.DOTALL,
        )
        return pattern.sub(lambda _match: candidate_section, guide_text)

    phase_2_heading = "### Phase 2 Checklist"
    if phase_2_heading in guide_text:
        return guide_text.replace(phase_2_heading, f"{candidate_section}\n\n{phase_2_heading}", 1)

    return guide_text.rstrip() + "\n\n" + candidate_section + "\n"


def should_preserve_existing_candidate_section(successful_query_count, query_errors):
    """Keep the last reviewer scan when every configured source is temporarily unavailable."""
    return successful_query_count == 0 and bool(query_errors)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="docs/SOTA_AUTOMATION_GUIDE.md")
    p.add_argument("--guide", default="docs/SOTA_AUTOMATION_GUIDE.md")
    p.add_argument("--max-per-query", type=int, default=5)
    p.add_argument("--max-opportunities", type=int, default=25)
    p.add_argument("--skip-repo-scan", action="store_true")
    args = p.parse_args()

    existing = load_existing_titles(args.guide)

    found = []
    found_titles = set()
    query_errors = []
    successful_query_count = 0
    for q in QUERIES:
        try:
            items = query_arxiv(q, max_results=args.max_per_query)
        except Exception as e:
            print(f"query failed for {q}: {e}")
            query_errors.append((q, str(e)))
            continue
        successful_query_count += 1
        for it in items:
            key = it["title"].lower()
            if key in existing or key in found_titles:
                continue
            candidate = {"query": q, **it}
            if not is_relevant_candidate(candidate):
                continue
            found.append(candidate)
            found_titles.add(key)

    now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    output_path = Path(args.output)
    guide_path = Path(args.guide)
    extra_sections = []
    if not args.skip_repo_scan:
        opportunities = scan_repo_opportunities(Path.cwd(), max_items=args.max_opportunities)
        extra_sections.append(render_repo_opportunity_scan(opportunities))
    candidate_section = render_candidate_section(found, now, query_errors=query_errors, extra_sections=extra_sections)

    if output_path == guide_path:
        guide_text = guide_path.read_text(encoding="utf-8")
        if should_preserve_existing_candidate_section(successful_query_count, query_errors):
            print("Preserved existing SOTA candidate scan because all configured queries failed")
            return
        output_text = update_guide_with_candidate_section(guide_text, candidate_section)
    else:
        output_text = candidate_section

    os.makedirs(output_path.parent, exist_ok=True)
    output_path.write_text(output_text.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote SOTA candidate scan to {args.output} (found {len(found)} items)")


if __name__ == "__main__":
    main()
