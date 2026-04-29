"""Tests for all baseline model implementations.

Each model is tested with small synthetic data for:
- fit/predict correctness (output shape, reasonable values)
- uncertainty estimation (non-negative std, correct shape)
- save/load round-trip (predictions match after reload)
"""

import tempfile
from pathlib import Path

import numpy as np


def _make_synthetic_data(n_samples=200, n_features=100, seed=42):
    """Create simple synthetic regression data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    # Simple linear relationship + noise
    w = rng.randn(n_features)
    y = X @ w + rng.randn(n_samples) * 0.5
    return X, y


class TestRandomForestModel:
    """Test Random Forest model."""

    def test_fit_predict(self):
        """Fit and predict should return correct shapes."""
        from target_affinity_ml.models.rf_model import RandomForestModel

        X, y = _make_synthetic_data()
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_uncertainty(self):
        """Uncertainty std should be non-negative and correct shape."""
        from target_affinity_ml.models.rf_model import RandomForestModel

        X, y = _make_synthetic_data()
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        mean, std = model.predict_with_uncertainty(X)
        assert mean.shape == (X.shape[0],)
        assert std.shape == (X.shape[0],)
        assert np.all(std >= 0)

    def test_save_load(self):
        """Predictions should match after save/load round-trip."""
        from target_affinity_ml.models.rf_model import RandomForestModel

        X, y = _make_synthetic_data()
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir))
            loaded = RandomForestModel.load(Path(tmpdir))
            preds_after = loaded.predict(X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)


class TestXGBoostModel:
    """Test XGBoost model."""

    def test_fit_predict(self):
        """Fit and predict should return correct shapes."""
        from target_affinity_ml.models.xgb_model import XGBoostModel

        X, y = _make_synthetic_data()
        model = XGBoostModel(
            n_estimators=20, max_depth=3, random_state=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_uncertainty(self):
        """Uncertainty std should be non-negative and correct shape."""
        from target_affinity_ml.models.xgb_model import XGBoostModel

        X, y = _make_synthetic_data()
        model = XGBoostModel(
            n_estimators=20, max_depth=3, random_state=42,
        )
        model.fit(X, y)
        mean, std = model.predict_with_uncertainty(X)
        assert mean.shape == (X.shape[0],)
        assert std.shape == (X.shape[0],)
        assert np.all(std >= 0)

    def test_save_load(self):
        """Predictions should match after save/load round-trip."""
        from target_affinity_ml.models.xgb_model import XGBoostModel

        X, y = _make_synthetic_data()
        model = XGBoostModel(
            n_estimators=20, max_depth=3, random_state=42,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir))
            loaded = XGBoostModel.load(Path(tmpdir))
            preds_after = loaded.predict(X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)


class TestElasticNetModel:
    """Test ElasticNet model."""

    def test_fit_predict(self):
        """Fit and predict should return correct shapes."""
        from target_affinity_ml.models.elasticnet_model import ElasticNetModel

        X, y = _make_synthetic_data()
        model = ElasticNetModel(
            n_bootstrap=5, alpha=0.1, l1_ratio=0.5, random_state=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_uncertainty(self):
        """Bootstrap std should be non-negative and correct shape."""
        from target_affinity_ml.models.elasticnet_model import ElasticNetModel

        X, y = _make_synthetic_data()
        model = ElasticNetModel(
            n_bootstrap=5, alpha=0.1, l1_ratio=0.5, random_state=42,
        )
        model.fit(X, y)
        mean, std = model.predict_with_uncertainty(X)
        assert mean.shape == (X.shape[0],)
        assert std.shape == (X.shape[0],)
        assert np.all(std >= 0)

    def test_save_load(self):
        """Predictions should match after save/load round-trip."""
        from target_affinity_ml.models.elasticnet_model import ElasticNetModel

        X, y = _make_synthetic_data()
        model = ElasticNetModel(
            n_bootstrap=5, alpha=0.1, l1_ratio=0.5, random_state=42,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir))
            loaded = ElasticNetModel.load(Path(tmpdir))
            preds_after = loaded.predict(X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_scaler_applied(self):
        """Predictions should use internal scaler (not fail on unscaled data)."""
        from target_affinity_ml.models.elasticnet_model import ElasticNetModel

        X, y = _make_synthetic_data()
        model = ElasticNetModel(
            n_bootstrap=3, alpha=0.1, l1_ratio=0.5, random_state=42,
        )
        model.fit(X, y)
        assert model.scaler is not None
        # Should not raise
        preds = model.predict(X * 100)
        assert preds.shape == (X.shape[0],)


class TestMLPModel:
    """Test MLP model."""

    def test_fit_predict(self):
        """Fit and predict should return correct shapes."""
        from target_affinity_ml.models.mlp_model import MLPModel

        X, y = _make_synthetic_data()
        model = MLPModel(
            n_ensemble=2,
            hidden_layer_sizes=[32, 16],
            max_iter=50,
            random_state=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_uncertainty(self):
        """Ensemble std should be non-negative and correct shape."""
        from target_affinity_ml.models.mlp_model import MLPModel

        X, y = _make_synthetic_data()
        model = MLPModel(
            n_ensemble=3,
            hidden_layer_sizes=[32, 16],
            max_iter=50,
            random_state=42,
        )
        model.fit(X, y)
        mean, std = model.predict_with_uncertainty(X)
        assert mean.shape == (X.shape[0],)
        assert std.shape == (X.shape[0],)
        assert np.all(std >= 0)

    def test_ensemble_diversity(self):
        """Ensemble members should give different predictions (different seeds)."""
        from target_affinity_ml.models.mlp_model import MLPModel

        X, y = _make_synthetic_data()
        model = MLPModel(
            n_ensemble=3,
            hidden_layer_sizes=[32, 16],
            max_iter=50,
            random_state=42,
        )
        model.fit(X, y)
        # Check that not all ensemble members give identical predictions
        X_scaled = model.scaler.transform(X)
        preds = [m.predict(X_scaled) for m in model.models]
        # At least some should differ
        assert not np.allclose(preds[0], preds[1])

    def test_save_load(self):
        """Predictions should match after save/load round-trip."""
        from target_affinity_ml.models.mlp_model import MLPModel

        X, y = _make_synthetic_data()
        model = MLPModel(
            n_ensemble=2,
            hidden_layer_sizes=[32, 16],
            max_iter=50,
            random_state=42,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir))
            loaded = MLPModel.load(Path(tmpdir))
            preds_after = loaded.predict(X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)
