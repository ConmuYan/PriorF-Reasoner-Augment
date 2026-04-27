from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from scripts import run_generation_audit

REPO_ROOT = Path(__file__).resolve().parents[1]
_SUBPROCESS_ENV = {
    **os.environ,
    "MKL_SERVICE_FORCE_INTEL": "1",
}


def test_contains_structural_signal_detects_expected_tokens() -> None:
    assert run_generation_audit._contains_structural_signal("branch_gap and neighbor profile both matter")
    assert run_generation_audit._contains_structural_signal("Route_Hint is high discrepancy")
    assert not run_generation_audit._contains_structural_signal("plain natural language without schema cues")


def test_semantic_usefulness_summary_handles_empty_and_parsed_samples() -> None:
    empty_summary = run_generation_audit._semantic_usefulness_summary([], graph_regime="transductive_standard")
    assert empty_summary["review_sample_count"] == 0
    assert empty_summary["parsed_review_sample_count"] == 0

    summary = run_generation_audit._semantic_usefulness_summary(
        [
            {
                "parsed_output": {"label": "fraud"},
                "semantic_checks": {
                    "rationale_has_structural_signal": True,
                    "evidence_has_structural_signal": False,
                    "pattern_hint_mentions_graph_regime": True,
                },
            },
            {
                "parsed_output": {"label": "benign"},
                "semantic_checks": {
                    "rationale_has_structural_signal": False,
                    "evidence_has_structural_signal": True,
                    "pattern_hint_mentions_graph_regime": False,
                },
            },
        ],
        graph_regime="transductive_standard",
    )
    assert summary["review_sample_count"] == 2
    assert summary["parsed_review_sample_count"] == 2
    assert summary["rationale_structural_signal_rate"] == 0.5
    assert summary["evidence_structural_signal_rate"] == 0.5
    assert summary["pattern_hint_mentions_graph_regime_rate"] == 0.5


def test_generation_audit_help_succeeds() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_generation_audit.py", "--help"],
        cwd=REPO_ROOT,
        env=_SUBPROCESS_ENV,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--teacher-export" in completed.stdout
    assert "--review-sample-size" in completed.stdout
