from __future__ import annotations

from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = ("FusionInputs", "fuse_probabilities")

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class FusionInputs(BaseModel):
    """Validated probability-level fusion inputs.

    This module is intentionally narrow: it only validates already-materialized
    probability vectors and combines them with a frozen ``alpha``.  It does not
    tune ``alpha``, read artifacts, or inspect population metadata.
    """

    model_config = _STRICT_MODEL_CONFIG

    teacher_probs: tuple[float, ...] = Field(min_length=1)
    student_probs: tuple[float, ...] = Field(min_length=1)
    alpha: float = Field(ge=0.0, le=1.0)

    @field_validator("teacher_probs", "student_probs")
    @classmethod
    def _probabilities_must_be_finite_and_bounded(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        for value in values:
            if not isfinite(float(value)):
                raise ValueError("probabilities must be finite")
            if value < 0.0 or value > 1.0:
                raise ValueError("probabilities must be within [0.0, 1.0]")
        return values

    @model_validator(mode="after")
    def _teacher_and_student_lengths_must_match(self) -> "FusionInputs":
        if len(self.teacher_probs) != len(self.student_probs):
            raise ValueError("teacher_probs and student_probs must have identical lengths")
        return self


def fuse_probabilities(*, teacher_probs: tuple[float, ...], student_probs: tuple[float, ...], alpha: float) -> tuple[float, ...]:
    """Return the frozen-alpha convex combination of teacher and student probabilities."""

    validated = FusionInputs(
        teacher_probs=teacher_probs,
        student_probs=student_probs,
        alpha=alpha,
    )
    one_minus_alpha = 1.0 - validated.alpha
    return tuple(
        (one_minus_alpha * teacher_prob) + (validated.alpha * student_prob)
        for teacher_prob, student_prob in zip(validated.teacher_probs, validated.student_probs)
    )
