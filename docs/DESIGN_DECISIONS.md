# Design decisions: CPU CI and notebook lint

## What failed

Five code cells contain a `!` or `%` line magic indented under `if`, `else`, or `try`. CPython's `ast.parse` reports `invalid syntax` or `invalid decimal literal` on those lines. The lint found:

- `notebooks/colab_antidoom_baseline.ipynb` cell 2 line 16 (`!git clone` under `if`)
- same notebook cell 7 line 6 (`!python` under `else`)
- `notebooks/colab_lfm2_26b_baseline.ipynb` cell 2 line 13 (`%pip` under `try`)
- same notebook cell 5 line 20 (`!python` under `else`)
- `notebooks/colab_qwen35_4b_a1_baseline.ipynb` cell 3 line 34 (`%pip` under `try`)

Colab's editor has treated `!` that is not at column 0 as a syntax error (colabtools issue 3507). Local IPython 7.34 rewrites indented magics, so a stock IPython transform is the wrong check: it hides the lines Colab rejects.

## Choice

Strip a magic only when it starts at column 0, then `ast.parse` the cell. Replace the five indented magics with `subprocess.run` so the cell is Python on both Colab and CPython. Leave top-level `%pip` and `!python` lines in place. Those are valid Colab and the lint accepts them.

## Alternatives considered

- Running the notebooks in Colab from this change. Rejected. That spends a GPU runtime and is outside the CPU limit.
- Installing CPU torch so CI can run `tests/test_generation_logic.py` and the intervention tests. Rejected for this workflow. Those tests import torch at collection time, and the failure this change is aimed at is the notebook syntax error plus a CPU stats suite that already exists.
- Rewriting experiment scripts that the notebooks call. Rejected. The syntax error is in the notebook cells.

## Extra import in one cell

`notebooks/colab_lfm2_26b_baseline.ipynb` cell 5 called `os.chdir` without importing `os`. The cell was already being edited to remove `!python`, so the import was added there. That is a notebook runtime fix, not a change to the experiment script.
