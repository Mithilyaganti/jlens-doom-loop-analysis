# Architecture: CPU CI and notebook lint

```
notebooks/*.ipynb
        │
        ▼
jspace/notebook_lint.py
  column-0 ! and % lines, including backslash continuations → pass
  indented ! and % left untouched
        │
        ▼
ast.parse
        │
        ▼
scripts/lint_notebooks.py          exit 1 on the first syntax error
.github/workflows/cpu-tests.yml    ubuntu-latest
  pytest tests/test_stats.py
  pytest tests/test_notebook_lint.py
  python scripts/lint_notebooks.py
```

`jspace/stats.py` is the other CPU suite in the workflow. It does not import torch. The workflow does not install torch, bitsandbytes, or a GPU driver.

Notebook edits are limited to cells whose indented magics were the syntax errors. Experiment scripts and `results/` are untouched.
