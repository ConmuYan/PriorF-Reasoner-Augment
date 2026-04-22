"""ASDA Layer: Adaptive Structural Discrepancy Attention.

Based on the empirical findings from E1:
- On YelpChi: HSD (δ) is effective but weak (1-2% effect size)
- On Amazon: |x_u-x_v| fails but HSD succeeds with strong effect (6-14%)

Design: HSD is the PRIMARY driver, |x_u-x_v| is auxiliary.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import softmax


def _scatter_add_dim0(src: torch.Tensor, index: torch.Tensor, *, dim_size: int) -> torch.Tensor:
    """Local dim-0 scatter-add fallback for inference/export environments.

    The original PriorF-GNN training environment depends on ``torch_scatter``.
    PriorF-Reasoner's teacher-export bridge must also run on workstations where
    that optional extension is unavailable, so this narrow fallback uses
    ``Tensor.scatter_add_`` for the exact ``dim=0`` shape needed by ASDA.
    """

    if index.dim() != 1:
        raise ValueError("ASDA scatter index must be 1-D")
    if src.size(0) != index.numel():
        raise ValueError("ASDA scatter src rows must match index length")
    output = src.new_zeros((dim_size, *src.shape[1:]))
    expanded_index = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    output.scatter_add_(0, expanded_index, src)
    return output


class ASDALayer(nn.Module):
    """Adaptive Structural Discrepancy Attention (ASDA) Layer.

    Computes edge-level discrepancy coefficients e_uv from HSD (δ) and feature
    differences, then performs dual-channel adaptive fusion (low-pass + high-pass).

    Formula:
        e_uv  = σ(MLP([δ_u, δ_v, |x_u - x_v|]))
        α_uv  = softmax_over_neighborhood(e_uv / τ)
        ᾱ_v   = sigmoid(MLP_2(δ_v))               # node-level frequency switch
        h_new = (1 - ᾱ_v) · low_pass + ᾱ_v · high_pass

    where:
        low_pass  = Σ_u Ã_uv · x_u  (symmetrically normalized mean)
        high_pass = Σ_u α_uv · (x_v - x_u)  (attention-weighted residual)

    Args:
        in_dim: Input feature dimension.
        tau: Temperature for softmax over neighborhood.
        edge_hidden: Hidden dim for edge MLP.
        node_hidden: Hidden dim for node switch MLP.
    """

    def __init__(
        self,
        in_dim: int,
        tau: float = 0.1,
        edge_hidden: int = 32,
        node_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.tau = tau

        # Edge-level discrepancy coefficient MLP
        # Input: [δ_u(1), δ_v(1), |x_u-x_v|(in_dim)] = in_dim + 2
        edge_in_dim = in_dim + 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, 1),
        )

        # Node-level frequency switch MLP: δ_v → ᾱ_v ∈ [0,1]
        self.node_switch = nn.Sequential(
            nn.Linear(1, node_hidden),
            nn.ReLU(),
            nn.Linear(node_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        hsd: torch.Tensor,
    ) -> torch.Tensor:
        """Run ASDA forward pass.

        Args:
            x: Node features (N, D)
            edge_index: Edge indices (2, E)
            hsd: Node-level HSD scores (N,)

        Returns:
            Transformed node features (N, D)
        """
        N = x.size(0)
        row, col = edge_index

        # Normalize HSD to avoid NaN/Inf on large graphs like Amazon
        hsd_norm = (hsd - hsd.mean()) / (hsd.std() + 1e-8)

        # ── 1. Edge-level discrepancy coefficient ────────────────────────────────
        # Collect features for each edge
        hsd_src = hsd_norm[row].unsqueeze(1)  # (E, 1)
        hsd_dst = hsd_norm[col].unsqueeze(1)  # (E, 1)
        diff_abs = torch.abs(x[row] - x[col])  # (E, D)

        edge_in = torch.cat([hsd_src, hsd_dst, diff_abs], dim=1)  # (E, D+2)
        e_uv = self.edge_mlp(edge_in)  # (E, 1)

        # ── 2. Normalized attention (softmax over destination neighborhood) ────
        alpha_uv = softmax(e_uv.squeeze(-1) / self.tau, col, dim=0, num_nodes=N).unsqueeze(-1)  # (E, 1)

        # ── 3. Node-level frequency switch ─────────────────────────────────────
        alpha_bar = self.node_switch(hsd_norm.unsqueeze(1))  # (N, 1)

        # ── 4. Dual-channel fusion ─────────────────────────────────────────────
        # Channel A: Low-pass (symmetrically normalized neighbor mean, with self-loop)
        # Ã_uv = 1 / sqrt(d_u * d_v), using scatter then normalize
        src_deg = torch.zeros(N, device=x.device)
        dst_deg = torch.zeros(N, device=x.device)
        src_deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        dst_deg.scatter_add_(0, col, torch.ones(col.size(0), device=x.device))
        src_deg = src_deg.clamp(min=1).sqrt()
        dst_deg = dst_deg.clamp(min=1).sqrt()

        # Edge weights: 1/sqrt(d_u * d_v)
        edge_w = 1.0 / (src_deg[row] * dst_deg[col])
        low_pass_msg = edge_w.unsqueeze(1) * x[row]  # (E, D)
        low_pass = _scatter_add_dim0(low_pass_msg, col, dim_size=N)  # (N, D)

        # Channel B: High-pass (attention-weighted residual)
        high_pass_msg = alpha_uv * (x[col] - x[row])  # (E, D)
        high_pass = _scatter_add_dim0(high_pass_msg, col, dim_size=N)  # (N, D)

        # Fuse: ᾱ_v controls high-pass proportion
        h_new = (1 - alpha_bar) * low_pass + alpha_bar * high_pass  # (N, D)

        return h_new
