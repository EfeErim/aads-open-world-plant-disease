#!/usr/bin/env python3
"""Repair script to make exception handling explicit and safer.

It scans tracked Python files (via `git ls-files`) and replaces bare
`except:` or `except Exception:` occurrences with a pattern that
binds the exception, logs it, and re-raises. It avoids touching
notebook cells, tests, and generated `.runtime_tmp` content.

Run with: python scripts/repair_safety.py
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_PATTERNS = [
    ".runtime_tmp",
    "scripts/notebook_cells",
    "tests",
]

RE_EXCEPT = re.compile(r"^(?P<indent>\s*)except\s*(?P<typ>Exception)?\s*:\s*$")


def should_exclude(p: Path) -> bool:
    s = str(p).replace('\\', '/')
    return any(pat in s for pat in EXCLUDE_PATTERNS)


def repair_file(p: Path) -> bool:
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    changed = False
    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = RE_EXCEPT.match(line)
        if m:
            indent = m.group('indent')
            # Replace with bound exception, log and re-raise
            out_lines.append(f"{indent}except Exception as exc:")
            out_lines.append(f"{indent}    import logging")
            out_lines.append(f"{indent}    logging.exception('Unhandled exception')")
            out_lines.append(f"{indent}    raise")
            changed = True
            i += 1
            continue
        out_lines.append(line)
        i += 1

    if changed:
        p.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    proc = subprocess.run(["git", "ls-files", "*.py"], cwd=str(ROOT), capture_output=True, text=True)
    files = [ROOT / s for s in proc.stdout.splitlines() if s]
    modified = []
    for f in files:
        if should_exclude(f):
            continue
        try:
            if repair_file(f):
                modified.append(str(f))
        except Exception as exc:
            print(f"Failed to process {f}: {exc}")

    if modified:
        print("Modified files:")
        for m in modified:
            print(m)
    else:
        print("No changes made.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
