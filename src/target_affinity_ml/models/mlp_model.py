"""Simple MLP (Multi-Layer Perceptron) for binding affinity prediction.

A shallow neural network baseline using scikit-learn's MLPRegressor.
Serves as a bridge between classical ML baselines and deep learning
approaches.

Architecture: Input → Dense(hidden) → ReLU → Dense(hidden) → ReLU → Output

Uncertainty: Ensemble of independently trained MLPs — variance of
ensemble predictions estimates epistemic uncertainty. Each ensemble
member is initialized with a different random seed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MLPModel:
    """MLP regression model with ensemble uncertainty."""

    def __init__(self, n_ensemble: int = 5, **kwargs):
        """Initialize MLP ensemble.

        Parameters
        ----------
        n_ensemble : int
            Number of independently trained MLPs for uncertainty.
        **kwargs
            Passed to sklearn.neural_network.MLPRegressor.
        """
        self.n_ensemble = n_ensemble
        self.params = kwargs
        self.models: list[MLPRegressor] = []
        self.scaler = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train MLP ensemble on training data.

        Fits a StandardScaler on the training data, then trains
        n_ensemble MLPRegressors with different random seeds.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).
        """
        logger.info(
            "Training MLP ensemble (n_ensemble=%d, n_samples=%d, n_features=%d)...",
            self.n_ensemble, X.shape[0], X.shape[1],
        )

        # Fit scaler on training data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train ensemble members with different seeds
        base_seed = self.params.get("random_state", 42)
        self.models = []

        for i in range(self.n_ensemble):
            logger.info("  Training MLP ensemble member %d/%d...", i + 1, self.n_ensemble)

            # Override random_state for each ensemble member
            member_params = {k: v for k, v in self.params.items()}
            member_params["random_state"] = base_seed + i

            mlp = MLPRegressor(**member_params)
            mlp.fit(X_scaled, y)

            logger.info(
                "    Converged in %d iterations (loss=%.4f)",
                mlp.n_iter_, mlp.loss_,
            )
            self.models.append(mlp)

        logger.info("  MLP ensemble training complete (%d models)", self.n_ensemble)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate point predictions (ensemble mean).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,).
        """
        X_scaled = self.scaler.transform(X)
        preds = np.array([m.predict(X_scaled) for m in self.models])
        return preds.mean(axis=0)

    def predict_with_uncertainty(
        self, X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate predictions with ensemble uncertainty.

        Returns the mean and standard deviation across ensemble
        members' predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (mean_prediction, std_prediction).
        """
        X_scaled = self.scaler.transform(X)
        preds = np.array([m.predict(X_scaled) for m in self.models])
        # preds shape: (n_ensemble, n_samples)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std

    def save(self, path: Path) -> None:
        """Save trained models to disk.

        Parameters
        ----------
        path : Path
            Directory to save model files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "models": self.models,
                "scaler": self.scaler,
                "params": self.params,
                "n_ensemble": self.n_ensemble,
            },
            path / "model.joblib",
        )
        logger.info("Saved MLP ensemble (%d models) to %s", self.n_ensemble, path)

    @classmethod
    def load(cls, path: Path) -> MLPModel:
        """Load trained models from disk.

        Parameters
        ----------
        path : Path
            Directory containing saved model files.

        Returns
        -------
        MLPModel
            Loaded model instance.
        """
        path = Path(path)
        data = joblib.load(path / "model.joblib")

        instance = cls(n_ensemble=data["n_ensemble"], **data["params"])
        instance.models = data["models"]
        instance.scaler = data["scaler"]

        logger.info("Loaded MLP ensemble (%d models) from %s", instance.n_ensemble, path)
        return instance
