#!/usr/bin/env python3
"""Parse collegebase_essays.txt (exported UI labels + essays) into CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def parse_essays(text: str) -> list[dict]:
    lines = text.splitlines()
    out: list[dict] = []
    i = 0
    n = len(lines)

    while i < n:
        if lines[i].strip() != "School logo":
            i += 1
            continue

        i += 1
        if i >= n or lines[i].strip() != "School":
            continue
        i += 1
        if i >= n:
            break
        school = lines[i].strip()
        i += 1

        if i < n and lines[i].strip() == "Essay type icon":
            i += 1
        if i < n and lines[i].strip() == "Essay Type":
            i += 1
        if i >= n:
            break
        type_raw = lines[i].strip()
        i += 1

        if i < n and lines[i].strip() == "Category icon":
            i += 1
        if i < n and lines[i].strip() == "Category":
            i += 1
        if i >= n:
            break
        category = lines[i].strip()
        i += 1

        while i < n and lines[i].strip() == "":
            i += 1

        prompt: str | None = None
        if i < n and lines[i].strip() == "Prompt":
            i += 1
            prompt_lines: list[str] = []
            while i < n and lines[i].strip() != "Essay":
                prompt_lines.append(lines[i])
                i += 1
            prompt = "\n".join(prompt_lines).strip() or None

        if i >= n or lines[i].strip() != "Essay":
            continue
        i += 1

        body_lines: list[str] = []
        while i < n and lines[i].strip() != "Word count":
            body_lines.append(lines[i])
            i += 1
        body = "\n".join(body_lines).strip()

        word_count: int | None = None
        if i < n and lines[i].strip() == "Word count":
            i += 1
            if i < n:
                m = re.match(r"^(\d+)\s*words?\s*$", lines[i].strip(), re.I)
                if m:
                    word_count = int(m.group(1))
                i += 1

        parts = [type_raw, category]
        if prompt:
            parts.append(prompt)
        essay_type_col = " — ".join(parts)

        meta = {
            "source": "collegebase",
            "essay_type_raw": type_raw,
            "category": category,
            "prompt": prompt,
            "word_count": word_count,
        }

        out.append(
            {
                "school": school,
                "grade": "",
                "essay_type": essay_type_col,
                "essay_text": body,
                "metadata": json.dumps(meta, ensure_ascii=False),
            }
        )

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input",
        default="collegebase_essays.txt",
        help="Path to pasted collegebase export",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="collegebase_essays.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_file():
        print(f"Not found: {src}", file=sys.stderr)
        sys.exit(1)

    text = src.read_text(encoding="utf-8", errors="replace")
    rows = parse_essays(text)

    fieldnames = ["school", "grade", "essay_type", "essay_text", "metadata"]
    with Path(args.output).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
