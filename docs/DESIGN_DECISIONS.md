# Design decisions: multiple-comparison adjustments

## Choice

Add Holm (step-down, family-wise) and Benjamini-Hochberg (FDR) as pure functions of a p-value list. The caller decides the family. Non-finite p-values are left as NaN and are not counted in `m`, matching `mcnemar_exact`, which returns NaN when there are no discordant pairs.

## Alternatives considered

- Bonferroni only. Rejected as the sole method. Holm is uniformly at least as powerful and uses the same family-wise error target. BH is the FDR alternative a reviewer will ask for when several paired contrasts are reported. Shipping both does not pick one for the paper.
- Running either adjustment on `results/exp3/stats.json` and committing a new p-value. Rejected. That file has one McNemar contrast. A one-test family does not change the p-value, and building a larger family from contrasts that were not saved as p-values would invent inputs.
- Editing `scripts/05_exp3_causal.py` so the next GPU run prints adjusted p-values. Rejected in this change. The functions are available; wiring them into the locked analysis plan is a separate protocol decision.

## Sort stability

Ties use `mergesort` so equal p-values keep their relative order. The adjustment formulas then make tied inputs agree after the cumulative step, so the sort tie-break does not change the returned values.
