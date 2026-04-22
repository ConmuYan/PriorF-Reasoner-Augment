"""LG-HGCL v2: ASDA-enhanced model.

Architecture:
    MLP branch:  X -> MLP -> z_mlp
    GNN branch:  ASDA(X, hsd) -> 2-layer RGCN -> z_gnn
    Prediction:  [z_mlp || z_gnn] -> Linear -> sigmoid -> score

ASDA (Adaptive Structural Discrepancy Attention) replaces SCRE as the first
processing step. It uses HSD (δ) as the primary driver for edge routing,
making it effective on both YelpChi (camouflage fraud) and Amazon (group fraud).

Ablation flags allow selectively disabling ASDA, MLP, or GNN branches.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import RGCNConv

from lghgcl.logging_utils import get_logger
from lghgcl.models.asda_layer import ASDALayer

logger = get_logger(__name__)


class LGHGCLNetV2(nn.Module):
    """ASDA-enhanced LG-HGCL model.

    Args:
        in_dim: Input feature dimension (D_sem + D_beh + 1 for HSD).
        mlp_hidden: Hidden dim in MLP branch.
        gnn_hidden: Hidden dim in RGCN layers.
        out_hidden: Output embedding dim per branch.
        num_relations: Number of edge relation types.
        dropout: Dropout probability.
        use_asda: If True, use ASDA layer; if False, feed raw X to RGCN.
        use_mlp_branch: If False, only GNN branch is used.
        use_gnn_branch: If False, only MLP branch is used.
        asda_tau: Temperature for ASDA edge attention softmax.
        proj_dim: Projection head dim for SDCL.
    """

    def __init__(
        self,
        in_dim: int,
        mlp_hidden: int = 256,
        gnn_hidden: int = 256,
        out_hidden: int = 256,
        num_relations: int = 3,
        dropout: float = 0.1,
        use_asda: bool = True,
        use_mlp_branch: bool = True,
        use_gnn_branch: bool = True,
        asda_tau: float = 0.1,
        proj_dim: int = 64,
    ) -> None:
        super().__init__()
        self.use_mlp = use_mlp_branch
        self.use_gnn = use_gnn_branch
        self.use_asda = use_asda
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
            self.asda = ASDALayer(in_dim=in_dim, tau=asda_tau) if use_asda else None
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

        # Projection head for SDCL
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
        hsd: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits of shape ``(N,)``.

        Args:
            x: Node features (N, D)
            edge_index: Edge indices (2, E)
            edge_type: Edge relation types (E,), values in {0, 1, 2}
            hsd: Node-level HSD scores (N,)
        """
        z_mlp = None
        z_gnn = None

        # MLP branch
        if self.use_mlp:
            z_mlp = self.mlp(x)

        # GNN branch
        if self.use_gnn:
            # ASDA pre-processing (or raw X if use_asda=False)
            h_input = self.asda(x, edge_index, hsd) if self.asda is not None else x
            z = self.rgcn1(h_input, edge_index, edge_type)
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
        self.z_fused = z
        z_proj = self.proj_head(z)
        self.z_proj = z_proj
        logits = self.out(z).squeeze(-1)
        return logits
