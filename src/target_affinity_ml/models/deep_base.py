"""Base class for PyTorch deep learning models.

Provides shared functionality for all Phase 7 neural network models:
    - predict(): iterate over DataLoader, collect predictions
    - predict_with_uncertainty(): MC-Dropout for uncertainty estimation
    - save()/load(): torch state_dict + config persistence

Each subclass only needs to implement __init__() and forward().
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DeepModelBase(nn.Module):
    """Base class for PyTorch regression models with MC-Dropout uncertainty."""

    def predict(
        self,
        dataloader,
        device: str | None = None,
    ) -> np.ndarray:
        """Generate point predictions for all samples in the dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            PyTorch or PyG DataLoader.
        device : str, optional
            Device to use. Defaults to model's current device.

        Returns
        -------
        np.ndarray
            Predicted values, shape (n_samples,).
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                batch = _move_batch(batch, device)
                preds = self.forward(batch).squeeze(-1)
                all_preds.append(preds.cpu().numpy())

        return np.concatenate(all_preds)

    def predict_with_uncertainty(
        self,
        dataloader,
        n_samples: int = 20,
        device: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict with MC-Dropout uncertainty estimation.

        Enables dropout at test time and runs multiple forward passes.
        The mean across passes is the prediction; the std is the uncertainty.

        Parameters
        ----------
        dataloader : DataLoader
            PyTorch or PyG DataLoader.
        n_samples : int
            Number of stochastic forward passes.
        device : str, optional
            Device to use.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (mean_predictions, std_predictions), each shape (n_samples,).
        """
        if device is None:
            device = next(self.parameters()).device

        # Enable dropout for MC sampling
        _enable_dropout(self)

        all_passes = []
        for _ in range(n_samples):
            preds = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = _move_batch(batch, device)
                    pred = self.forward(batch).squeeze(-1)
                    preds.append(pred.cpu().numpy())
            all_passes.append(np.concatenate(preds))

        self.eval()  # Restore eval mode

        stacked = np.stack(all_passes, axis=0)  # (n_samples, n_data)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def save(self, path: str | Path) -> None:
        """Save model state dict and config.

        Parameters
        ----------
        path : str or Path
            Directory to save into.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), path / "model.pt")

        if hasattr(self, "config"):
            with open(path / "config.json", "w") as f:
                json.dump(self.config, f, indent=2)

        logger.info("Saved model to %s", path)

    def load(self, path: str | Path, device: str = "cpu") -> None:
        """Load model state dict.

        Parameters
        ----------
        path : str or Path
            Directory containing model.pt.
        device : str
            Device to load onto.
        """
        path = Path(path)
        state_dict = torch.load(path / "model.pt", map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        self.to(device)
        logger.info("Loaded model from %s", path)


def _enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers while keeping everything else in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def _move_batch(batch, device):
    """Move a batch to the specified device.

    Handles both standard tensors and PyG Batch objects.
    """
    if isinstance(batch, (list, tuple)):
        return [_move_batch(b, device) for b in batch]
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch
