"""ESM-2 protein embedding + Morgan fingerprint MLP model.

Concatenates pre-computed ESM-2 target embeddings (1280-dim) with
Morgan fingerprints (2048-dim) to create a 3328-dim input vector,
then predicts pActivity through an MLP.

This model tests the simplest form of protein-awareness: adding target
identity through a learned embedding rather than a one-hot encoding.

Hypothesis: should improve predictions on the TARGET split where
baselines fail hardest, because it encodes what makes each kinase's
binding site unique.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from target_affinity_ml.models.deep_base import DeepModelBase


class ESMFPMLPModel(DeepModelBase):
    """MLP that takes concatenated [Morgan FP | ESM-2 embedding] as input."""

    def __init__(
        self,
        input_dim: int = 3328,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.config = {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
        }

        layers = []
        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            # Lower dropout on last hidden layer
            drop_rate = dropout * 0.67 if i == len(hidden_dims) - 1 else dropout
            layers.append(nn.Dropout(drop_rate))
            if i < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(dim))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, batch) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        batch : torch.Tensor or tuple
            If tensor: combined [FP | ESM-2] features, shape (batch_size, input_dim).
            If tuple: (features_tensor, ...) where first element is the features.
        """
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        return self.network(x)
