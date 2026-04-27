"""Fail-closed prompt leakage audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from evidence.leakage_policy import STUDENT_VISIBLE_FORBIDDEN_FIELDS, STUDENT_VISIBLE_FORBIDDEN_PHRASES


@dataclass(frozen=True)
class PromptAuditViolation:
    location: str
    needle: str
    kind: str


@dataclass(frozen=True)
class PromptAuditResult:
    checked: int
    violations: tuple[PromptAuditViolation, ...]

    @property
    def passed(self) -> bool:
        return not self.violations


def audit_text(text: str, *, location: str = "text") -> PromptAuditResult:
    lowered = text.lower()
    violations: list[PromptAuditViolation] = []
    for field in STUDENT_VISIBLE_FORBIDDEN_FIELDS:
        if field.lower() in lowered:
            violations.append(PromptAuditViolation(location=location, needle=field, kind="field"))
    for phrase in STUDENT_VISIBLE_FORBIDDEN_PHRASES:
        if phrase.lower() in lowered:
            violations.append(PromptAuditViolation(location=location, needle=phrase, kind="phrase"))
    return PromptAuditResult(checked=1, violations=tuple(violations))


def audit_prompt_bundle(bundle, *, include_assistant_target: bool = True) -> PromptAuditResult:
    violations: list[PromptAuditViolation] = []
    checked = 0
    for idx, message in enumerate(bundle.messages):
        if message.role == "assistant" and not include_assistant_target:
            continue
        result = audit_text(message.content, location=f"messages[{idx}].{message.role}")
        checked += result.checked
        violations.extend(result.violations)
    return PromptAuditResult(checked=checked, violations=tuple(violations))


def assert_prompt_audit_passes(result: PromptAuditResult) -> None:
    if result.violations:
        rendered = ", ".join(
            f"{violation.location}:{violation.kind}:{violation.needle}"
            for violation in result.violations
        )
        raise ValueError(f"student-visible prompt leakage audit failed: {rendered}")
