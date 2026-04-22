"""Visualization utilities for LG-HGCL.

Provides:
  - t-SNE / UMAP node embedding visualization
  - HSD distribution analysis (fraud vs normal)
  - Edge type distribution statistics
  - Training loss curve plotting
  - SHAP feature importance analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

from lghgcl.logging_utils import get_logger

logger = get_logger(__name__)

# Consistent styling
sns.set_theme(style="whitegrid", font_scale=1.2)
FRAUD_COLOR = "#e74c3c"
NORMAL_COLOR = "#3498db"


# ------------------------------------------------------------------
#  1. t-SNE / UMAP Embedding Visualization
# ------------------------------------------------------------------


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    max_samples: int = 5000,
    perplexity: float = 30.0,
    title: str = "t-SNE Embedding Visualization",
) -> None:
    """t-SNE 2D scatter of node embeddings coloured by fraud/normal."""
    from sklearn.manifold import TSNE

    n = len(labels)
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(labels) - 1), random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl, color, name in [(0, NORMAL_COLOR, "Normal"), (1, FRAUD_COLOR, "Fraud")]:
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=name, alpha=0.5, s=8, edgecolors="none")
    ax.set_title(title)
    ax.legend(markerscale=3)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    _save_fig(fig, out_path)


def plot_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    max_samples: int = 5000,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    title: str = "UMAP Embedding Visualization",
) -> None:
    """UMAP 2D scatter of node embeddings."""
    import umap

    n = len(labels)
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl, color, name in [(0, NORMAL_COLOR, "Normal"), (1, FRAUD_COLOR, "Fraud")]:
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=name, alpha=0.5, s=8, edgecolors="none")
    ax.set_title(title)
    ax.legend(markerscale=3)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    _save_fig(fig, out_path)


# ------------------------------------------------------------------
#  2. HSD Distribution Analysis
# ------------------------------------------------------------------


def plot_hsd_distribution(
    hsd: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    title: str = "HSD Distribution: Fraud vs Normal",
) -> None:
    """Overlaid histogram + KDE of HSD scores for fraud vs normal nodes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    fraud_hsd = hsd[labels == 1]
    normal_hsd = hsd[labels == 0]
    ax.hist(normal_hsd, bins=50, alpha=0.6, color=NORMAL_COLOR, label="Normal", density=True)
    ax.hist(fraud_hsd, bins=50, alpha=0.6, color=FRAUD_COLOR, label="Fraud", density=True)
    ax.set_xlabel("HSD Score")
    ax.set_ylabel("Density")
    ax.set_title("HSD Histogram")
    ax.legend()

    # KDE
    ax2 = axes[1]
    if len(normal_hsd) > 1:
        sns.kdeplot(normal_hsd, ax=ax2, color=NORMAL_COLOR, label="Normal", fill=True, alpha=0.3)
    if len(fraud_hsd) > 1:
        sns.kdeplot(fraud_hsd, ax=ax2, color=FRAUD_COLOR, label="Fraud", fill=True, alpha=0.3)
    ax2.set_xlabel("HSD Score")
    ax2.set_ylabel("Density")
    ax2.set_title("HSD KDE")
    ax2.legend()

    # Add statistics text
    stats_text = (
        f"Fraud: mean={fraud_hsd.mean():.4f}, std={fraud_hsd.std():.4f}\n"
        f"Normal: mean={normal_hsd.mean():.4f}, std={normal_hsd.std():.4f}\n"
        f"Ratio: {fraud_hsd.mean() / max(normal_hsd.mean(), 1e-8):.2f}x"
    )
    ax2.text(
        0.98,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(title)
    plt.tight_layout()
    _save_fig(fig, out_path)


# ------------------------------------------------------------------
#  3. Edge Type Distribution
# ------------------------------------------------------------------


def plot_edge_distribution(
    hetero_data: HeteroData,
    out_path: str | Path,
    title: str = "Edge Type Distribution",
) -> None:
    """Bar chart of edge counts per relation type."""
    rel_names = []
    rel_counts = []
    for rel, label in [("rur", "R-U-R"), ("rtr", "R-T-R"), ("rsr", "R-S-R")]:
        key = ("review", rel, "review")
        if key in hetero_data.edge_types:
            count = hetero_data[key].edge_index.size(1)
            rel_names.append(label)
            rel_counts.append(count)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#f39c12", "#9b59b6"]
    bars = ax.bar(rel_names, rel_counts, color=colors[: len(rel_names)])
    for bar, count in zip(bars, rel_counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{count:,}", ha="center", va="bottom", fontsize=11
        )
    ax.set_ylabel("Number of Edges")
    ax.set_title(title)
    _save_fig(fig, out_path)


# ------------------------------------------------------------------
#  Internal
# ------------------------------------------------------------------


def _save_fig(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
