"""LG-HGCL model -- pure PyTorch implementation (no PyGOD dependency).

Architecture (per paper):
    MLP branch:  X -> MLP -> z_mlp
    GNN branch:  SCRE(X) -> 2-layer RGCN -> z_gnn
    Prediction:  [z_mlp || z_gnn] -> Linear -> sigmoid -> score

Ablation flags allow selectively disabling SCRE, MLP, or GNN branches.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import RGCNConv
from torch_scatter import scatter_mean

from lghgcl.logging_utils import get_logger

logger = get_logger(__name__)


class SCRE(nn.Module):
    """Structure-Contrastive Residual Encoding (relation-aware).

    Computes  H_sc = X - A_R(X)  where A_R(X) is the *relation-aware*
    neighbourhood mean:

        context_v = (1/R_active) * sum_r  mean_{u in N_r(v)}  x_u

    This is a high-pass filter that suppresses smooth low-frequency
    components and retains discrepancy information.
    """

    def __init__(self, num_relations: int = 3):
        super().__init__()
        self.num_relations = num_relations

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        n = x.size(0)
        row, col = edge_index

        context_sum = x.new_zeros(n, x.size(1))
        relation_count = x.new_zeros(n, 1)

        for r in range(self.num_relations):
            mask = edge_type == r
            if not mask.any():
                continue
            r_row = row[mask]
            r_col = col[mask]
            r_ctx = scatter_mean(x[r_col], r_row, dim=0, dim_size=n)
            has_neigh = scatter_mean(
                torch.ones(r_row.size(0), 1, device=x.device),
                r_row, dim=0, dim_size=n,
            ).clamp(max=1.0)
            context_sum = context_sum + r_ctx
            relation_count = relation_count + has_neigh

        relation_count = relation_count.clamp_min(1.0)
        context = context_sum / relation_count
        return x - context


class LGHGCLNet(nn.Module):
    """Joint MLP + SCRE-RGCN network for LG-HGCL.

    Supports ablation flags for selectively disabling components.

    Args:
        in_dim: Input feature dimension (D_sem + D_beh + 1 for HSD).
        mlp_hidden: Hidden dim in MLP branch.
        gnn_hidden: Hidden dim in RGCN layers.
        out_hidden: Output embedding dim per branch.
        num_relations: Number of edge relation types.
        dropout: Dropout probability.
        use_scre: If False, feed raw X to RGCN (standard GNN aggregation).
        use_mlp_branch: If False, only GNN branch is used.
        use_gnn_branch: If False, only MLP branch is used.
    """

    def __init__(
        self,
        in_dim: int,
        mlp_hidden: int = 256,
        gnn_hidden: int = 256,
        out_hidden: int = 256,
        num_relations: int = 3,
        dropout: float = 0.1,
        use_scre: bool = True,
        use_mlp_branch: bool = True,
        use_gnn_branch: bool = True,
        proj_dim: int = 64,
    ) -> None:
        super().__init__()
        self.use_mlp = use_mlp_branch
        self.use_gnn = use_gnn_branch
        self.dropout = dropout

        if not use_mlp_branch and not use_gnn_branch:
            raise ValueError("At least one of use_mlp_branch or use_gnn_branch must be True")

        # MLP Branch
        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden),
                nn.BatchNorm1d(mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, out_hidden),
                nn.BatchNorm1d(out_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # GNN Branch
        if self.use_gnn:
            self.scre = SCRE(num_relations=num_relations) if use_scre else None
            self.rgcn1 = RGCNConv(in_dim, gnn_hidden, num_relations=num_relations)
            self.bn1 = nn.BatchNorm1d(gnn_hidden)
            self.rgcn2 = RGCNConv(gnn_hidden, out_hidden, num_relations=num_relations)
            self.bn2 = nn.BatchNorm1d(out_hidden)

        # Prediction head
        head_dim = 0
        if self.use_mlp:
            head_dim += out_hidden
        if self.use_gnn:
            head_dim += out_hidden
        self.out = nn.Linear(head_dim, 1)

        # Projection head for contrastive loss (SDCL)
        self.proj_head = nn.Sequential(
            nn.Linear(head_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        self.emb: tuple[torch.Tensor | None, torch.Tensor | None] | None = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits of shape ``(N,)``."""
        z_mlp = None
        z_gnn = None

        # MLP branch
        if self.use_mlp:
            z_mlp = self.mlp(x)

        # GNN branch
        if self.use_gnn:
            h_sc = self.scre(x, edge_index, edge_type) if self.scre is not None else x
            z = self.rgcn1(h_sc, edge_index, edge_type)
            z = self.bn1(z)
            z = F.relu(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
            z = self.rgcn2(z, edge_index, edge_type)
            z_gnn = self.bn2(z)
            z_gnn = F.relu(z_gnn)
            z_gnn = F.dropout(z_gnn, p=self.dropout, training=self.training)

        # Store detached embeddings for visualisation
        self.emb = (
            z_mlp.detach() if z_mlp is not None else None,
            z_gnn.detach() if z_gnn is not None else None,
        )

        # Fusion
        parts = [p for p in (z_mlp, z_gnn) if p is not None]
        z = torch.cat(parts, dim=-1)
        self.z_fused = z              # non-detached for SDCL contrastive loss
        self.z_proj = self.proj_head(z)  # projected embedding for SDCL
        logits = self.out(z).squeeze(-1)
        return logits
