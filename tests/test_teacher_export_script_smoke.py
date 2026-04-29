"""Smoke tests for teacher-export operator scripts.

These tests do not run full teacher inference.  They only lock the CLI import
surface so operator-facing commands fail closed on real argument errors rather
than failing early because the script cannot import project or teacher-source
modules.
"""

from __future__ import annotations

import subprocess
import sys

from scripts.generate_teacher_exports import _model_hyperparams_for_inference


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


def test_seed42_full_summary_is_filtered_to_inference_hyperparams(tmp_path):
    import numpy as np
    from scipy.io import savemat

    mat_path = tmp_path / "canonical.mat"
    savemat(mat_path, {"x": np.zeros((3, 25), dtype=np.float32)})
    summary = {
        "config": {
            "model": {
                "mlp_hidden": 64,
                "gnn_hidden": 64,
                "out_hidden": 64,
                "num_relations": 3,
                "dropout": 0.3,
                "focal_alpha": 0.25,
                "lambda_sdcl": 0.1,
                "use_asda": True,
            }
        }
    }

    hyperparams = _model_hyperparams_for_inference(summary, mat_path)

    assert hyperparams == {
        "in_dim": 26,
        "mlp_hidden": 64,
        "gnn_hidden": 64,
        "out_hidden": 64,
        "num_relations": 3,
        "dropout": 0.3,
        "use_asda": True,
        "use_mlp_branch": True,
        "use_gnn_branch": True,
        "proj_dim": 64,
    }
