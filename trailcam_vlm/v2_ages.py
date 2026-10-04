"""Copies of an export with teenagers counted in age_unknown, for scoring with version 2's evaluation script.

Version 2 and its validation labels have no teenager category: the labels
count teenagers in ``age_unclear``, which ``python -m trailcam.evaluate``
compares with the export column ``age_unknown``. This tool reports teenagers
in ``teens``, so for that comparison ``age_unknown`` must include them::

    python -m trailcam_vlm.v2_ages "exports\\export_DATE"

writes ``export_DATE_v2ages.csv`` and ``export_DATE_events_v2ages.csv`` next
to the originals. Only ``age_unknown`` changes (teens + age_unknown; blank
stays blank); every other column, including ``teens``, is copied unchanged.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .export import write_csv

SUFFIX = "_v2ages"


def teens_in_unknown(row):
    """The row with teens added to age_unknown (unchanged when either is blank)."""
    teens, unknown = (row.get("teens") or "").strip(), (row.get("age_unknown") or "").strip()
    if teens.isdigit() and unknown.isdigit():
        return {**row, "age_unknown": str(int(teens) + int(unknown))}
    return dict(row)


def convert(source):
    """Write the v2-style copy of one CSV; returns its path."""
    source = Path(source)
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames or [], list(reader)
    if "teens" not in fields or "age_unknown" not in fields:
        raise ValueError(f"{source} has no teens and age_unknown columns; is it an images or events CSV of this tool?")
    if "age_unclear" in fields:
        raise ValueError(f"{source} was written before teenagers became a separate category: its age_unknown already "
                         "includes teens. Use it as it is.")
    target = source.with_name(source.stem + SUFFIX + source.suffix)
    # The values were already made spreadsheet-safe when the export was written; copy them as they are.
    write_csv(target, [teens_in_unknown(r) for r in rows], fields, escape=False)
    return target


def sources(path):
    """The images and events CSVs of an export folder, or the given CSV."""
    path = Path(path).resolve()
    if path.is_dir():
        found = [path / f"{path.name}.csv", path / f"{path.name}_events.csv"]
        found = [p for p in found if p.is_file()]
        if not found:
            raise ValueError(f"No {path.name}.csv or {path.name}_events.csv in {path}")
        return found
    if path.suffix.lower() == ".csv" and path.is_file():
        if path.stem.lower().endswith(SUFFIX.lower()):
            raise ValueError(f"{path} is already a converted copy")
        return [path]
    raise ValueError(f"Not an export folder or CSV file: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m trailcam_vlm.v2_ages",
        description="Copy an export with teenagers counted in age_unknown, as version 2 and its labels count them, "
                    "for scoring with version 2's 'python -m trailcam.evaluate'.")
    parser.add_argument("paths", nargs="+", help="Export folders (export_DATE) or images/events CSV files")
    args = parser.parse_args(argv)
    try:
        for path in args.paths:
            for source in sources(path):
                print(f"Wrote {convert(source)}")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
