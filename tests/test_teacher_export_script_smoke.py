"""Smoke tests for teacher-export operator scripts.

These tests do not run full teacher inference.  They only lock the CLI import
surface so operator-facing commands fail closed on real argument errors rather
than failing early because the script cannot import project or teacher-source
modules.
"""

from __future__ import annotations

import subprocess
import sys


def test_generate_teacher_exports_help_imports_without_pythonpath():
    completed = subprocess.run(
        [sys.executable, "scripts/generate_teacher_exports.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--dataset" in completed.stdout
    assert "--device" in completed.stdout
