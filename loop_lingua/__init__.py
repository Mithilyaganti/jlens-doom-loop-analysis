"""Loop-Lingua: CPU-side tools for a cross-lingual doom-loop study.

Languages in the matched set: English, Hindi, Telugu, and romanized Telugu.
Code-mixed Tenglish is a hand-edit placeholder, not a generated arm.

Paper-1 (``jspace/``) is not modified. D0 calls a verbatim copy of its
Antidoom rule. The offset fix lives here.
"""

from __future__ import annotations

from loop_lingua.detectors import Detection, LoopHit, detect, detect_all
from loop_lingua.drift import measure_drift, script_profile
from loop_lingua.fertility import measure_fertility
from loop_lingua.offsets import TokenOffsetMap, map_token_ids, map_token_strings
from loop_lingua.pilot import PilotConfig, plan_go_nogo, run_harness
from loop_lingua.power import analysis_table, power_at_n
from loop_lingua.prompts import build_matched_set, select_item_ids
from loop_lingua.transliterate import transliterate_telugu

__version__ = "0.1.0"

__all__ = [
    "Detection",
    "LoopHit",
    "PilotConfig",
    "TokenOffsetMap",
    "analysis_table",
    "build_matched_set",
    "detect",
    "detect_all",
    "map_token_ids",
    "map_token_strings",
    "measure_drift",
    "measure_fertility",
    "plan_go_nogo",
    "power_at_n",
    "run_harness",
    "script_profile",
    "select_item_ids",
    "transliterate_telugu",
]
