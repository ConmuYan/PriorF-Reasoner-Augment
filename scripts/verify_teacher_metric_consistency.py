"""Diagnostic: verify teacher metric consistency across methods.

NON-CANONICAL / NON-GATED / NON-FORMAL. This script is a one-shot diagnostic.
It reads nothing under outputs/ and writes nothing under outputs/; it does not
produce parquet, does not drive any gate, and is not referenced by canonical
code paths. Its only purpose is to answer one empirical question:

    Was ``assets/teacher/{dataset}/model_summary.json::best_metric`` computed
    via ``sklearn.metrics.auc(recall_curve, precision_curve)`` (trapezoidal
    AUPRC) rather than ``average_precision_score`` (step-wise AUPRC), such that
    the observed mismatch with the teacher-inference-bridge validation AUROC is
    explained by the metric-method gap and not by a bridge bug?

For each of {amazon, yelpchi} this script:
  1. Loads canonical .mat (produced by scripts/legacy_mat_to_canonical.py).
  2. Runs priorf_teacher.inference.run_teacher_inference with the same
     arguments as scripts/generate_teacher_exports.py (zero drift).
  3. Extracts validation (ground_truth_label, teacher_prob) pairs.
  4. Computes three metrics on that validation split:
       * AUROC                   -> sklearn.metrics.roc_auc_score
       * AUPRC_trapezoidal       -> sklearn.metrics.auc(recall, precision)
                                    (matches priorf_gnn/lghgcl/metrics.py:116-117)
       * AUPRC_step              -> sklearn.metrics.average_precision_score
  5. Prints a plain-text table to stdout. No file writes.

Expected outcome if the diagnosis in status_package/general.md is correct:
    dataset   AUROC    AUPRC_trap   AUPRC_step   summary.best_metric
    amazon    ~0.97    ~0.83        < AUPRC_trap 0.8318
    yelpchi   ~0.75    ~0.84        < AUPRC_trap 0.8421   (<-- AUPRC_trap ~ summary value)

This script intentionally duplicates a small amount of wiring from
generate_teacher_exports.py rather than importing from it, because
generate_teacher_exports.py performs gated writes and must not be imported
from a diagnostic entrypoint.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    auc as _sk_auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIORF_GNN_ROOT = REPO_ROOT / "priorf_gnn"
for import_root in (REPO_ROOT, PRIORF_GNN_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

from priorf_teacher.inference import run_teacher_inference
from priorf_teacher.schema import DatasetName, GraphRegime

# Must stay in lock-step with scripts/generate_teacher_exports.py::DATASETS.
# We deliberately do NOT import that dict to keep this diagnostic script
# independent of the gated-write entrypoint.
DATASETS: dict[str, dict] = {
    "amazon": {
        "legacy_mat": "assets/data/Amazon.mat",
        "canonical_mat": "assets/data/Amazon_canonical.mat",
        "checkpoint": "assets/teacher/amazon/best_model.pt",
        "model_summary": "assets/teacher/amazon/model_summary.json",
        "dataset_enum": DatasetName.AMAZON,
    },
    "yelpchi": {
        "legacy_mat": "assets/data/YelpChi.mat",
        "canonical_mat": "assets/data/YelpChi_canonical.mat",
        "checkpoint": "assets/teacher/yelpchi/best_model.pt",
        "model_summary": "assets/teacher/yelpchi/model_summary.json",
        "dataset_enum": DatasetName.YELPCHI,
    },
}


def _compute_metrics(
    labels: np.ndarray, probs: np.ndarray
) -> tuple[float, float, float]:
    """Return (AUROC, AUPRC_trapezoidal, AUPRC_step) on a single population."""
    if labels.ndim != 1 or probs.ndim != 1:
        raise ValueError("labels/probs must be 1-D")
    if labels.shape != probs.shape:
        raise ValueError("labels/probs shape mismatch")
    unique = np.unique(labels)
    if unique.size < 2:
        raise ValueError(
            f"single-class population (labels={unique.tolist()}); "
            "AUROC/AUPRC undefined"
        )

    auroc = float(roc_auc_score(labels, probs))
    # AUPRC_trap: mirrors priorf_gnn/lghgcl/metrics.py:116-117 exactly.
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
    auprc_trap = float(_sk_auc(recall_curve, precision_curve))
    # AUPRC_step: the sklearn-recommended AUPRC estimator.
    auprc_step = float(average_precision_score(labels, probs))
    return auroc, auprc_trap, auprc_step


def _summary_best_metric(model_summary_path: str) -> float | None:
    try:
        summary = json.loads(Path(model_summary_path).read_text())
    except FileNotFoundError:
        return None
    val = summary.get("best_metric")
    return float(val) if val is not None else None


def run_one(dataset_key: str, device: str) -> dict[str, float | str | None]:
    cfg = DATASETS[dataset_key]
    print(f"\n=== {dataset_key.upper()} ===", flush=True)

    summary_raw = json.loads(Path(cfg["model_summary"]).read_text())
    model_hyperparams = summary_raw["config"]["model"]

    print("  [1/2] running teacher inference (full-graph forward) ...", flush=True)
    pop_records = run_teacher_inference(
        dataset_name=cfg["dataset_enum"],
        legacy_mat_path=cfg["legacy_mat"],
        canonical_mat_path=cfg["canonical_mat"],
        checkpoint_path=cfg["checkpoint"],
        model_hyperparams=model_hyperparams,
        teacher_model_name="LGHGCLNetV2",
        teacher_checkpoint=cfg["checkpoint"],
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        device=device,
    )

    val_records = pop_records.get("validation", [])
    if not val_records:
        raise RuntimeError(f"{dataset_key}: no validation records from inference")

    labels = np.asarray(
        [int(r.ground_truth_label) for r in val_records], dtype=np.int64
    )
    probs = np.asarray(
        [float(r.teacher_prob) for r in val_records], dtype=np.float64
    )
    n_total = int(labels.size)
    n_pos = int(labels.sum())
    pos_rate = float(n_pos / n_total) if n_total else float("nan")

    print(f"       validation size   = {n_total}", flush=True)
    print(f"       validation n_pos  = {n_pos}", flush=True)
    print(f"       validation pos %  = {pos_rate:.4f}", flush=True)

    print("  [2/2] computing metrics ...", flush=True)
    auroc, auprc_trap, auprc_step = _compute_metrics(labels, probs)
    best_metric = _summary_best_metric(cfg["model_summary"])

    return {
        "dataset": dataset_key,
        "n_val": n_total,
        "pos_rate": pos_rate,
        "auroc": auroc,
        "auprc_trap": auprc_trap,
        "auprc_step": auprc_step,
        "summary_best_metric": best_metric,
    }


def _fmt(v: float | None) -> str:
    if v is None:
        return "   n/a"
    return f"{v:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only: recompute (AUROC, AUPRC_trap, AUPRC_step) on the "
            "validation split of Amazon/YelpChi using the teacher inference "
            "bridge. No artifact is written."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["amazon", "yelpchi"],
        choices=sorted(DATASETS.keys()),
        help="Which datasets to verify (default: both).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device string passed to run_teacher_inference.",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for key in args.datasets:
        rows.append(run_one(key, device=args.device))

    print("\n" + "=" * 84, flush=True)
    print("VALIDATION METRIC CONSISTENCY (bridge-recomputed vs model_summary)", flush=True)
    print("=" * 84, flush=True)
    header = (
        f"{'dataset':<10} {'n_val':>7} {'pos%':>7}  "
        f"{'AUROC':>8} {'AUPRC_trap':>11} {'AUPRC_step':>11}  "
        f"{'summary.best_metric':>20}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in rows:
        line = (
            f"{r['dataset']:<10} "
            f"{r['n_val']:>7d} "
            f"{r['pos_rate']:>7.4f}  "
            f"{_fmt(r['auroc']):>8} "
            f"{_fmt(r['auprc_trap']):>11} "
            f"{_fmt(r['auprc_step']):>11}  "
            f"{_fmt(r['summary_best_metric']):>20}"
        )
        print(line, flush=True)
    print("=" * 84, flush=True)
    print(
        "interpretation keys:\n"
        "  * if AUPRC_trap ~= summary.best_metric, the original training metric\n"
        "    is val AUPRC computed via trapezoidal auc(recall, precision) and\n"
        "    the bridge is fully consistent with the original training harness.\n"
        "  * if AUPRC_step < AUPRC_trap (same dataset), trapezoidal AUPRC is\n"
        "    over-estimating relative to sklearn's recommended estimator.\n"
        "  * AUROC is the bridge's independent measure and should be trusted\n"
        "    for gating decisions unless a well-justified alternative is chosen.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
