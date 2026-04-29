"""Graph Isomorphism Network (GIN) for molecular property prediction.

Converts molecules from SMILES to graphs and learns representations
through message passing. Uses atom/bond features as node/edge attributes.

GIN is provably the most expressive message-passing GNN (as powerful as
the Weisfeiler-Leman graph isomorphism test). Each layer updates a node's
representation by combining its own features with a learned aggregation
of its neighbors' features through a 2-layer MLP.

Architecture:
    Atom features → GINConv × 3 → global mean+max pool → MLP head → pActivity

Hypothesis: should capture structural features that Morgan fingerprints
miss (e.g., long-range connectivity, ring systems, 3D-relevant topology),
particularly helping on SCAFFOLD splits with novel chemical matter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool

from target_affinity_ml.features.molecular_graphs import get_atom_feature_dim
from target_affinity_ml.models.deep_base import DeepModelBase


class GNNModel(DeepModelBase):
    """GIN-based molecular property prediction model."""

    def __init__(
        self,
        num_layers: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()

        atom_dim = get_atom_feature_dim()

        self.config = {
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "atom_dim": atom_dim,
        }

        # GIN convolution layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = atom_dim if i == 0 else hidden_dim
            # Each GINConv uses a 2-layer MLP as the update function
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Readout: mean + max pooling concatenated → 2 * hidden_dim
        pool_dim = 2 * hidden_dim

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch) -> torch.Tensor:
        """Forward pass on a PyG Batch object.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batched molecular graphs with x, edge_index, and batch attributes.
        """
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        # Message passing
        for conv, bn in zip(self.convs, self.bns, strict=False):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # Global pooling (mean + max concatenated)
        x_mean = global_mean_pool(x, batch_idx)
        x_max = global_max_pool(x, batch_idx)
        x = torch.cat([x_mean, x_max], dim=1)

        # Prediction
        return self.head(x)
