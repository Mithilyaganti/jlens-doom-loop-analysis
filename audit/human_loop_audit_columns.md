# Human loop-audit sheet

`audit/human_loop_audit_sheet.csv` is a header-only template. It has no filled judgments. Do not treat a blank cell as a label.

| Column | Who fills it | Meaning |
|---|---|---|
| `auditor` | human | Name or initials of the person who read the trace. |
| `trace_id` | human | Prompt id or fixture id of the text that was read. |
| `source_path` | human | Repo path or run artifact the text came from. |
| `char_len` | human | Character length of the text that was read, counted by the auditor. |
| `sampler_hit` | human, from the detector | `1` or `0` from `find_inner_repetition` on that full text. Leave blank until the full text is in hand. |
| `exhaustive_hit` | human, from the audit script | `1` or `0` from the exhaustive period scan on that same full text. |
| `human_is_loop` | human | `1` if the auditor judges the text to be a doom loop, `0` otherwise. This column is the judgment. It is empty in the template. |
| `human_notes` | human | What repeated, where it started, and whether the repetition is a reasoning loop or a copied formula. |
| `disagreement_reason` | human | Filled only when the human label disagrees with the sampler or the exhaustive scan. |

The CPU audit script can print sampler and exhaustive labels for fixtures. Those machine labels are not copied into this sheet, because a fixture label is not a human judgment.
