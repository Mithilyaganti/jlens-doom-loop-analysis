# Process: CPU CI and notebook lint

## Steps

1. Parsed every code cell in `notebooks/` with CPython after stripping only column-0 magics.
2. Recorded the five SyntaxErrors below.
3. Replaced those indented magics with `subprocess.run`. Did not edit `scripts/` experiment files or `results/`.
4. Re-ran the lint, `tests/test_notebook_lint.py`, and `tests/test_stats.py`.

## Verified before the notebook edit

```bash
python scripts/lint_notebooks.py
```

```
5 notebook syntax error(s):
/workspace/notebooks/colab_antidoom_baseline.ipynb:cell 2: line 16: invalid syntax
/workspace/notebooks/colab_antidoom_baseline.ipynb:cell 7: line 6: invalid decimal literal
/workspace/notebooks/colab_lfm2_26b_baseline.ipynb:cell 2: line 13: invalid syntax
/workspace/notebooks/colab_lfm2_26b_baseline.ipynb:cell 5: line 20: invalid syntax
/workspace/notebooks/colab_qwen35_4b_a1_baseline.ipynb:cell 3: line 34: invalid syntax
```

Exit code 1.

## Verified after the notebook edit

```bash
python scripts/lint_notebooks.py
```

```
ok: 4 notebooks
```

```bash
python -m pytest tests/test_notebook_lint.py tests/test_stats.py -q
```

```
...............                                                          [100%]
15 passed in 0.14s
```

`python3 -c` `json.loads` on each notebook exited 0.

## Assumed

Colab is what surfaced the error. This change did not open a Colab session. The messages above are CPython `SyntaxError` text on the notebook source when indented magics are not rewritten. IPython 7.34's transformer accepts those same cells, which is why the lint does not use it.

## Not run

The GitHub Actions workflow itself (no `workflow_dispatch` from this environment). Torch tests. Any notebook execution.
