"""GNN + ESM-2 fusion model for protein-ligand affinity prediction.

Combines learned molecular representations (GIN) with pre-computed
protein embeddings (ESM-2) through concatenation fusion.

Architecture:
    Ligand branch:  SMILES → GIN (3 layers) → global pool → 256-dim
    Protein branch: ESM-2 (1280) → Linear → ReLU → 256-dim
    Fusion:         concat → MLP head → pActivity

This is the "full model" that should benefit from both:
    - Learned molecular structure (vs. fixed fingerprints)
    - Protein target identity (vs. target-agnostic baselines)

Hypothesis: should outperform all baselines on ALL splits, with the
largest improvement on the TARGET split where both structural and
target information matter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool

from target_affinity_ml.features.molecular_graphs import get_atom_feature_dim
from target_affinity_ml.models.deep_base import DeepModelBase


class FusionModel(DeepModelBase):
    """GNN + ESM-2 fusion model for protein-ligand affinity prediction."""

    def __init__(
        self,
        gnn_layers: int = 3,
        gnn_hidden_dim: int = 128,
        protein_input_dim: int = 1280,
        protein_projection_dim: int = 256,
        fusion_hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()

        if fusion_hidden_dims is None:
            fusion_hidden_dims = [256, 128]

        atom_dim = get_atom_feature_dim()

        self.config = {
            "gnn_layers": gnn_layers,
            "gnn_hidden_dim": gnn_hidden_dim,
            "protein_input_dim": protein_input_dim,
            "protein_projection_dim": protein_projection_dim,
            "fusion_hidden_dims": fusion_hidden_dims,
            "dropout": dropout,
        }

        # --- Ligand branch: GIN ---
        self.gnn_convs = nn.ModuleList()
        self.gnn_bns = nn.ModuleList()

        for i in range(gnn_layers):
            in_dim = atom_dim if i == 0 else gnn_hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, gnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            )
            self.gnn_convs.append(GINConv(mlp))
            self.gnn_bns.append(nn.BatchNorm1d(gnn_hidden_dim))

        # Ligand embedding: mean+max pool → 2 * gnn_hidden_dim → project to protein_projection_dim
        ligand_pool_dim = 2 * gnn_hidden_dim
        self.ligand_proj = nn.Sequential(
            nn.Linear(ligand_pool_dim, protein_projection_dim),
            nn.ReLU(),
        )

        # --- Protein branch: linear projection ---
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_input_dim, protein_projection_dim),
            nn.ReLU(),
        )

        # --- Fusion head ---
        fusion_input_dim = 2 * protein_projection_dim  # ligand + protein
        layers = []
        prev_dim = fusion_input_dim

        for i, dim in enumerate(fusion_hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            drop_rate = dropout * 0.67 if i == len(fusion_hidden_dims) - 1 else dropout
            layers.append(nn.Dropout(drop_rate))
            if i < len(fusion_hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(dim))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))
        self.fusion_head = nn.Sequential(*layers)

    def forward(self, batch) -> torch.Tensor:
        """Forward pass on a (graph_batch, protein_embeddings) tuple.

        Parameters
        ----------
        batch : tuple
            (graph_batch, protein_emb) where:
            - graph_batch: PyG Batch with x, edge_index, batch
            - protein_emb: tensor of shape (batch_size, protein_input_dim)
        """
        graph_batch, protein_emb = batch

        # --- Ligand branch ---
        x, edge_index, batch_idx = graph_batch.x, graph_batch.edge_index, graph_batch.batch

        for conv, bn in zip(self.gnn_convs, self.gnn_bns, strict=False):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        x_mean = global_mean_pool(x, batch_idx)
        x_max = global_max_pool(x, batch_idx)
        ligand_emb = self.ligand_proj(torch.cat([x_mean, x_max], dim=1))

        # --- Protein branch ---
        protein_proj = self.protein_proj(protein_emb)

        # --- Fusion ---
        combined = torch.cat([ligand_emb, protein_proj], dim=1)
        return self.fusion_head(combined)
