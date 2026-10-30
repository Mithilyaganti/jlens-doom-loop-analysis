"""LeanScope data layer: scenarios, lean conditions, locked splits, labels.

Paper 2 on the shared J-lens probe library. This package builds the items and
reads the behavioural labels. It does not train detectors.
"""

from leanscope.schema import LeanCondition, Record, Scenario

__all__ = ["LeanCondition", "Record", "Scenario"]

__version__ = "0.1.0"
