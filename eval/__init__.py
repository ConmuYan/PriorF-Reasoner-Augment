"""Task 5 unified head-scoring package.

The only public surface is the canonical ``score_head`` predict_proba +
metrics entrypoint and its strictly-typed input/output/provenance contracts.
"""

from eval.head_scoring import (
    CheckpointProvenance,
    ClsHead,
    HeadScoringInputs,
    HeadScoringSample,
    ScorerReport,
    score_head,
)

__all__ = [
    "CheckpointProvenance",
    "ClsHead",
    "HeadScoringInputs",
    "HeadScoringSample",
    "ScorerReport",
    "score_head",
]
