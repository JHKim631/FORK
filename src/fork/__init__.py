"""FORK: a precision objective for rubric-based reinforcement learning.

The reward combines rubric recall (the standard weighted rubric coverage) with
response precision — the fraction of response units that are necessary and
factually correct — through their F-beta score. See the paper for the method
and README.md for how to train and evaluate with it.
"""

__version__ = "0.1.0"
