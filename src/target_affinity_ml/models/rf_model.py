"""Random Forest model for binding affinity prediction.

Random Forest is a strong baseline for molecular property prediction.
It handles high-dimensional sparse fingerprints well and provides
natural uncertainty estimates from the variance across trees.

Uncertainty: Each tree in the forest makes an independent prediction.
The standard deviation across tree predictions estimates the model's
uncertainty — higher variance means the trees disagree and the
prediction is less reliable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


class RandomForestModel:
    """Random Forest regression model with uncertainty estimation."""

    def __init__(self, **kwargs):
        """Initialize with scikit-learn RandomForestRegressor parameters."""
        self.params = kwargs
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Random Forest on training data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).
        """
        logger.info(
            "Training RandomForest (n_samples=%d, n_features=%d)...",
            X.shape[0], X.shape[1],
        )

        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X, y)

        logger.info(
            "  Trained %d trees (max_depth=%s, max_features=%s)",
            self.model.n_estimators,
            self.model.max_depth,
            self.model.max_features,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate point predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,).
        """
        return self.model.predict(X)

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate predictions with uncertainty estimates.

        Collects predictions from each individual tree and computes
        mean and standard deviation across the ensemble.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (mean_prediction, std_prediction) where std is computed
            from individual tree predictions.
        """
        # Collect predictions from each tree
        tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        # tree_preds shape: (n_trees, n_samples)
        mean = tree_preds.mean(axis=0)
        std = tree_preds.std(axis=0)
        return mean, std

    def save(self, path: Path) -> None:
        """Save trained model to disk.

        Parameters
        ----------
        path : Path
            Directory to save model files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "params": self.params},
            path / "model.joblib",
        )
        logger.info("Saved RandomForest model to %s", path)

    @classmethod
    def load(cls, path: Path) -> RandomForestModel:
        """Load a trained model from disk.

        Parameters
        ----------
        path : Path
            Directory containing saved model files.

        Returns
        -------
        RandomForestModel
            Loaded model instance.
        """
        path = Path(path)
        data = joblib.load(path / "model.joblib")
        instance = cls(**data["params"])
        instance.model = data["model"]
        logger.info("Loaded RandomForest model from %s", path)
        return instance
