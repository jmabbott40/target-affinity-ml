"""ElasticNet model for binding affinity prediction.

ElasticNet combines L1 (Lasso) and L2 (Ridge) regularization,
making it suitable for high-dimensional descriptor spaces where
many features may be irrelevant. The l1_ratio parameter controls
the balance between feature selection (L1) and coefficient
shrinkage (L2).

Used with RDKit 2D descriptors rather than fingerprints, as the
continuous features benefit from linear model interpretation.

Uncertainty: Bootstrap resampling — train the model on N bootstrap
samples of the training data and compute prediction variance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ElasticNetModel:
    """ElasticNet regression model with bootstrap uncertainty."""

    def __init__(self, n_bootstrap: int = 100, **kwargs):
        """Initialize with ElasticNet parameters.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap resamples for uncertainty estimation.
        **kwargs
            Passed to sklearn.linear_model.ElasticNet.
        """
        self.n_bootstrap = n_bootstrap
        self.params = kwargs
        self.model = None
        self.bootstrap_models: list[ElasticNet] = []
        self.scaler = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train ElasticNet (primary model + bootstrap ensemble).

        Fits a StandardScaler on the training data, then trains:
        1. A primary ElasticNet model on all training data.
        2. n_bootstrap ElasticNet models on bootstrap resamples
           for uncertainty estimation.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).
        """
        logger.info(
            "Training ElasticNet (n_samples=%d, n_features=%d)...",
            X.shape[0], X.shape[1],
        )

        # Fit scaler on training data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Primary model on all training data
        self.model = ElasticNet(**self.params)
        self.model.fit(X_scaled, y)
        logger.info(
            "  Primary model trained (alpha=%.4f, l1_ratio=%.2f, n_features=%d)",
            self.model.alpha, self.model.l1_ratio,
            np.sum(self.model.coef_ != 0),
        )

        # Bootstrap ensemble for uncertainty
        rng = np.random.RandomState(self.params.get("random_state", 42))
        self.bootstrap_models = []
        n_samples = X_scaled.shape[0]

        for i in range(self.n_bootstrap):
            if (i + 1) % 20 == 0 or i == 0:
                logger.info(
                    "  Bootstrap model %d/%d...", i + 1, self.n_bootstrap,
                )
            # Resample with replacement
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_scaled[idx]
            y_boot = y[idx]

            boot_model = ElasticNet(**self.params)
            boot_model.fit(X_boot, y_boot)
            self.bootstrap_models.append(boot_model)

        logger.info(
            "  ElasticNet training complete (1 primary + %d bootstrap models)",
            self.n_bootstrap,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate point predictions from primary model.

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
        return self.model.predict(X_scaled)

    def predict_with_uncertainty(
        self, X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate predictions with bootstrap confidence intervals.

        Computes predictions from all bootstrap models and returns
        mean and standard deviation across the ensemble.

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

        # Collect predictions from all bootstrap models
        boot_preds = np.array([
            m.predict(X_scaled) for m in self.bootstrap_models
        ])
        # boot_preds shape: (n_bootstrap, n_samples)
        mean = boot_preds.mean(axis=0)
        std = boot_preds.std(axis=0)
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
            {
                "model": self.model,
                "bootstrap_models": self.bootstrap_models,
                "scaler": self.scaler,
                "params": self.params,
                "n_bootstrap": self.n_bootstrap,
            },
            path / "model.joblib",
        )
        logger.info("Saved ElasticNet model to %s", path)

    @classmethod
    def load(cls, path: Path) -> ElasticNetModel:
        """Load a trained model from disk.

        Parameters
        ----------
        path : Path
            Directory containing saved model files.

        Returns
        -------
        ElasticNetModel
            Loaded model instance.
        """
        path = Path(path)
        data = joblib.load(path / "model.joblib")

        instance = cls(n_bootstrap=data["n_bootstrap"], **data["params"])
        instance.model = data["model"]
        instance.bootstrap_models = data["bootstrap_models"]
        instance.scaler = data["scaler"]

        logger.info("Loaded ElasticNet model from %s", path)
        return instance
