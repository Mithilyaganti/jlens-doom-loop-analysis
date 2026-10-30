"""LeanScope: hidden-lean detection with J-lens (paper 2).

The modules here are the causal and intervention side of the study. Detectors
plug in later by supplying a direction function and a lean score. Simulation
outputs are a Tier A pre-test, not evidence about people.
"""

__all__ = ["schema", "prompts", "causal", "panel", "simulate"]
