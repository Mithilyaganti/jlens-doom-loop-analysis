#!/usr/bin/env python
"""Print the CPU loop-detector audit. Does not write paper-1 result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jspace.detector_audit import audit_report, human_audit_header, load_fixtures  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "detector_loops.jsonl",
    )
    parser.add_argument(
        "--example-html",
        type=Path,
        default=ROOT / "results" / "exp2" / "example_looping_trace.html",
    )
    parser.add_argument(
        "--write-human-sheet",
        type=Path,
        default=None,
        help="Write the empty human-audit CSV (header only) to this path.",
    )
    args = parser.parse_args()
    fixtures = load_fixtures(args.fixtures)
    report = audit_report(fixtures, example_html=args.example_html)
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if args.write_human_sheet is not None:
        args.write_human_sheet.parent.mkdir(parents=True, exist_ok=True)
        args.write_human_sheet.write_text(human_audit_header(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
