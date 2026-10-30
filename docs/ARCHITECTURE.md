# Architecture: Holm and Benjamini-Hochberg

Both functions live in `jspace/stats.py`, next to `mcnemar_exact`. They take a sequence of p-values and return a numpy array of the same length.

```
caller-supplied p-values
        │
        ├─ drop non-finite entries from the family (leave them NaN)
        ├─ sort ascending (stable)
        ├─ Holm: cumulative max of (m - i + 1) * p_(i), clipped at 1
        └─ BH: (m / i) * p_(i), clipped at 1, then monotone from the right
        │
        ▼
adjusted values in the original order
```

`scripts/05_exp3_causal.py` does not call them. Nothing in `results/` is rewritten.
