"""LeanScope: hidden-lean detection with a J-lens (paper 2).

The package is the shared study code, not part of paper 1. Records use one
schema (:class:`leanscope.schema.Record`). Data builds the items, detection
reads residuals at P0 and P1, and the causal and panel modules steer that
detector direction and open an alternatives panel when its score fires.

CPU tests use a fake model. Nothing here downloads Qwen or calls a paid API.
"""

from leanscope.schema import LeanCondition, Record, Scenario

__all__ = ["LeanCondition", "Record", "Scenario"]

__version__ = "0.1.0"
