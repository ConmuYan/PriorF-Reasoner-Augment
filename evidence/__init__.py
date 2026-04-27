"""Task 3 Evidence Card, prompt, and strict output contracts."""

from evidence.evidence_schema import (
    EvidenceAblationMask,
    EvidenceCard,
    DiscrepancySummary,
    StudentVisibleNeighborSummary,
    TaskInstruction,
    TeacherSummary,
    build_evidence_card,
    build_student_evidence_card,
)
from evidence.output_schema import PredLabel, StrictOutput, canonical_serialize, parse_strict
from evidence.prompt_builder import ChatMessage, FewShotExample, PromptBundle, PromptMode, ThinkingMode, build_prompt

__all__ = [
    "ChatMessage",
    "DiscrepancySummary",
    "EvidenceAblationMask",
    "EvidenceCard",
    "FewShotExample",
    "PredLabel",
    "PromptBundle",
    "PromptMode",
    "StrictOutput",
    "StudentVisibleNeighborSummary",
    "TaskInstruction",
    "TeacherSummary",
    "ThinkingMode",
    "build_evidence_card",
    "build_student_evidence_card",
    "build_prompt",
    "canonical_serialize",
    "parse_strict",
]
