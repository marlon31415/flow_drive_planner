#!/usr/bin/env python3
"""Convert a CSV file into a Markdown table.

extract_score.py calls csv_to_markdown() after every score append, so a run's table is written
without a separate step. The CLI is for rendering CSVs written before that, or by hand:

    python csv_to_markdown_table.py simulation_scores/<run>.csv -o output.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _escape_markdown_cell(value: str) -> str:
    """Escape markdown table separators and normalize whitespace."""
    return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ").strip()


def csv_to_markdown(input_csv: Path) -> str:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Input file is empty: {input_csv}")

    header = [_escape_markdown_cell(cell) for cell in rows[0]]
    body = [[_escape_markdown_cell(cell) for cell in row] for row in rows[1:]]

    markdown_lines = []
    markdown_lines.append("| " + " | ".join(header) + " |")
    markdown_lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for row in body:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]
        markdown_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(markdown_lines) + "\n"


def default_output_path(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CSV to a Markdown table.")
    parser.add_argument("input_csv", type=Path, help="Path to the CSV file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to output Markdown file (default: <input_basename>.md)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv: Path = args.input_csv
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV file not found: {input_csv}")

    output_md = (
        args.output if args.output is not None else default_output_path(input_csv)
    )

    markdown_content = csv_to_markdown(input_csv)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown_content, encoding="utf-8")

    print(f"Wrote markdown table to: {output_md}")


if __name__ == "__main__":
    main()
