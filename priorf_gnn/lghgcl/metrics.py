"""Evaluation metrics for LG-HGCL."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def recall_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_pcts: list[float] | None = None,
) -> dict[str, float]:
    """Recall@K% — fraction of true positives captured in the top K% by score.

    Args:
        y_true: Binary ground truth labels.
        y_score: Predicted probability scores.
        k_pcts: List of percentages (e.g. [0.01, 0.05] for top 1% and 5%).

    Returns:
        Dict mapping "Recall@1%" etc. to recall values.
    """
    if k_pcts is None:
        k_pcts = [0.01, 0.05]
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    total_pos = y_true.sum()
    if total_pos == 0:
        return {f"Recall@{int(p * 100)}%": 0.0 for p in k_pcts}

    order = np.argsort(-y_score)
    result: dict[str, float] = {}
    for p in k_pcts:
        k = max(1, int(np.ceil(n * p)))
        top_k_labels = y_true[order[:k]]
        result[f"Recall@{int(p * 100)}%"] = float(top_k_labels.sum() / total_pos)
    return result


def ndcg_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 100,
) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Measures ranking quality: whether the model places true positives
    at the very top of the ranked list.

    Args:
        y_true: Binary ground truth labels.
        y_score: Predicted probability scores.
        k: Number of positions to evaluate.

    Returns:
        NDCG@K value in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score)
    n = len(y_true)
    k = min(k, n)
    if k == 0 or y_true.sum() == 0:
        return 0.0

    order = np.argsort(-y_score)
    dcg = np.sum(y_true[order[:k]] / np.log2(np.arange(2, k + 2)))

    ideal_order = np.argsort(-y_true)
    idcg = np.sum(y_true[ideal_order[:k]] / np.log2(np.arange(2, k + 2)))

    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_performance(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float | np.ndarray]:
    """Calculate comprehensive evaluation metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_score: Predicted probability scores.
        threshold: Classification threshold. If None, find optimal F1 threshold.

    Returns:
        Dictionary containing:
        - AUROC
        - AUPRC
        - F1
        - Precision
        - Recall
        - Confusion Matrix
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # 1. AUROC
    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = 0.0

    # 2. AUPRC
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
        auprc = auc(recall_curve, precision_curve)
        # sklearn average_precision_score is also a good approximation
        # auprc = average_precision_score(y_true, y_score)
    except ValueError:
        auprc = 0.0

    # 3. Determine Threshold for Binary Metrics
    if threshold is None:
        # Find threshold that maximizes F1 via fine-grained search
        best_f1 = 0.0
        best_thresh = 0.5
        # Two-pass: coarse percentile scan then fine search around best
        candidates = np.percentile(y_score, np.arange(1, 100, 1))
        candidates = np.unique(candidates)
        for th in candidates:
            y_pred_tmp = (y_score >= th).astype(int)
            f1_tmp = f1_score(y_true, y_pred_tmp, zero_division=0)
            if f1_tmp > best_f1:
                best_f1 = f1_tmp
                best_thresh = th
        # Fine search: 200 points around best region
        lo = max(best_thresh - 0.05, y_score.min())
        hi = min(best_thresh + 0.05, y_score.max())
        for th in np.linspace(lo, hi, 200):
            y_pred_tmp = (y_score >= th).astype(int)
            f1_tmp = f1_score(y_true, y_pred_tmp, zero_division=0)
            if f1_tmp > best_f1:
                best_f1 = f1_tmp
                best_thresh = th
        threshold = best_thresh

    y_pred = (y_score >= threshold).astype(int)

    # 4. Binary Metrics
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # G-means: geometric mean of sensitivity (recall for positive) and specificity (recall for negative)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        gmeans = float(np.sqrt(sensitivity * specificity))
    else:
        gmeans = 0.0

    # 5. Recall@K% & NDCG@K
    rak = recall_at_k(y_true, y_score)
    ndcg = ndcg_at_k(y_true, y_score, k=100)

    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "F1-Macro": float(f1_macro),
        "G-means": gmeans,
        "Precision": float(precision),
        "Recall": float(recall),
        **rak,
        "NDCG@100": ndcg,
        "ConfusionMatrix": cm.tolist(),
        "Threshold": float(threshold),
    }
