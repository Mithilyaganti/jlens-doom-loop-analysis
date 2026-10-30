"""Pre-onset doom-loop forecasting.

The question this package is built to answer is whether a trace announces a
doom loop before onset, and how many tokens ahead. Onset is the Antidoom
trigger: the first token of the first repeated span, under the character rule
of at least 4 repeats and at least 60 repeated characters.

Qwen baseline traces are the same files ``scripts/02_baseline_pass.py`` already
writes (``meta.json`` plus ``tokens.pt``). This package does not generate them.
"""

from jspace.forecasting.run import analyze_traces, render_report, summarize_results, write_summary

__all__ = ["analyze_traces", "render_report", "summarize_results", "write_summary"]
