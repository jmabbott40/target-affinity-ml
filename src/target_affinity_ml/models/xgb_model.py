"""XGBoost model for binding affinity prediction.

XGBoost (eXtreme Gradient Boosting) builds an ensemble of decision trees
sequentially, where each tree corrects errors made by previous trees.
It typically outperforms Random Forest on tabular data due to its
gradient-based optimization and built-in regularization.

Uncertainty: XGBoost supports quantile regression, which directly
predicts conditional quantiles (e.g., 5th and 95th percentile)
to construct prediction intervals.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost regression model with uncertainty estimation."""

    def __init__(self, **kwargs):
        """Initialize with XGBoost parameters."""
        self.params = kwargs
        self.model = None
        self.quantile_models: dict[float, xgb.XGBRegressor] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train XGBoost on training data.

        Trains three models:
        1. Primary model with squared error loss (point predictions)
        2. Lower quantile model (5th percentile)
        3. Upper quantile model (95th percentile)

        The quantile models define a 90% prediction interval.
        """
        logger.info(
            "Training XGBoost (n_samples=%d, n_features=%d)...",
            X.shape[0], X.shape[1],
        )

        # Primary model — standard squared error
        primary_params = {k: v for k, v in self.params.items()}
        primary_params["objective"] = "reg:squarederror"
        self.model = xgb.XGBRegressor(**primary_params)
        self.model.fit(X, y)
        logger.info("  Primary model trained (n_estimators=%d)", self.model.n_estimators)

        # Quantile models for uncertainty estimation
        for alpha in [0.05, 0.95]:
            logger.info("  Training quantile model (alpha=%.2f)...", alpha)
            q_params = {k: v for k, v in self.params.items()}
            q_params["objective"] = "reg:quantileerror"
            q_params["quantile_alpha"] = alpha
            q_model = xgb.XGBRegressor(**q_params)
            q_model.fit(X, y)
            self.quantile_models[alpha] = q_model

        logger.info("  XGBoost training complete (1 primary + 2 quantile models)")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate point predictions."""
        return self.model.predict(X)

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate predictions with uncertainty via quantile regression.

        Uses the 90% prediction interval (5th–95th percentile) to estimate
        standard deviation: std ≈ (q95 - q05) / (2 × 1.645).
        """
        mean = self.model.predict(X)
        q05 = self.quantile_models[0.05].predict(X)
        q95 = self.quantile_models[0.95].predict(X)

        # Convert 90% prediction interval to std
        std = np.maximum((q95 - q05) / (2 * 1.645), 0.0)
        return mean, std

    def save(self, path: Path) -> None:
        """Save trained model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(path / "model.json"))
        for alpha, q_model in self.quantile_models.items():
            q_model.save_model(str(path / f"quantile_{alpha:.2f}.json"))

        with open(path / "params.json", "w") as f:
            json.dump(self.params, f, indent=2, default=str)

        logger.info("Saved XGBoost model to %s", path)

    @classmethod
    def load(cls, path: Path) -> XGBoostModel:
        """Load a trained model from disk."""
        path = Path(path)

        with open(path / "params.json") as f:
            params = json.load(f)

        instance = cls(**params)
        instance.model = xgb.XGBRegressor()
        instance.model.load_model(str(path / "model.json"))

        for alpha in [0.05, 0.95]:
            q_model = xgb.XGBRegressor()
            q_model.load_model(str(path / f"quantile_{alpha:.2f}.json"))
            instance.quantile_models[alpha] = q_model

        logger.info("Loaded XGBoost model from %s", path)
        return instance
