#!/usr/bin/env python
"""Score pre-onset doom-loop forecasts from an existing baseline pass.

Reads the trace layout written by scripts/02_baseline_pass.py and writes

    results/forecast/{slug}_summary.json
    results/forecast/{slug}_report.md

Does not generate tokens and does not modify the generation scripts.
Default slug is the LFM2 checkout this report was built against. Point
--slug at qwen3.5-4b once that baseline pass has landed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jspace.forecasting.run import render_report, summarize_results, write_summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-onset doom-loop forecast report")
    parser.add_argument("--slug", default="lfm2-2.6b")
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = summarize_results(
        args.results,
        args.slug,
        n_boot=args.n_boot,
        seed=args.seed,
        window_len=args.window,
    )
    out = Path(args.results) / "forecast"
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{args.slug}_summary.json"
    md_path = out / f"{args.slug}_report.md"
    write_summary(json_path, summary)
    md_path.write_text(render_report(summary), encoding="utf-8")
    print(f"data_status={summary.get('data_status')} metrics_computed={summary.get('metrics_computed')}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
