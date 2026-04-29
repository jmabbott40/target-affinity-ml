"""Tests for evaluation metrics."""

import numpy as np
import pytest

from target_affinity_ml.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    enrichment_factor,
    precision_at_k,
)


class TestRegressionMetrics:
    """Test regression metric computation."""

    def test_perfect_prediction(self):
        """Perfect predictions should give RMSE=0, R²=1, Pearson=1."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = compute_regression_metrics(y, y)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["pearson_r"] == pytest.approx(1.0)
        assert metrics["spearman_rho"] == pytest.approx(1.0)

    def test_known_values(self):
        """Test against manually computed values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.8])
        metrics = compute_regression_metrics(y_true, y_pred)
        # RMSE = sqrt(mean([0.01, 0.04, 0.04])) = sqrt(0.03) ≈ 0.1732
        assert metrics["rmse"] == pytest.approx(0.1732, abs=0.001)
        # MAE = mean([0.1, 0.2, 0.2]) = 0.1667
        assert metrics["mae"] == pytest.approx(0.1667, abs=0.001)
        # R² should be close to 1 for good predictions
        assert metrics["r2"] > 0.9
        assert metrics["pearson_r"] > 0.98

    def test_returns_all_keys(self):
        """Should return all expected metric keys."""
        y = np.random.randn(100)
        metrics = compute_regression_metrics(y, y + np.random.randn(100) * 0.1)
        expected_keys = {"rmse", "mae", "r2", "pearson_r", "spearman_rho"}
        assert set(metrics.keys()) == expected_keys


class TestClassificationMetrics:
    """Test classification metric computation."""

    def test_perfect_separation(self):
        """Perfectly separable predictions should give AUROC≈1."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
        # Higher scores for positives
        y_scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        metrics = compute_classification_metrics(y_true, y_scores)
        assert metrics["auroc"] == pytest.approx(1.0)
        assert metrics["auprc"] == pytest.approx(1.0)

    def test_single_class_returns_nan(self):
        """When only one class is present, AUROC/AUPRC should be NaN."""
        y_true = np.ones(10)
        y_scores = np.random.randn(10)
        metrics = compute_classification_metrics(y_true, y_scores)
        assert np.isnan(metrics["auroc"])
        assert np.isnan(metrics["auprc"])

    def test_returns_enrichment(self):
        """Should include enrichment factor metrics."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 100, dtype=float)
        y_scores = np.random.randn(1000)
        metrics = compute_classification_metrics(y_true, y_scores)
        assert "ef_1pct" in metrics
        assert "ef_5pct" in metrics


class TestPrecisionAtK:
    """Test precision@k computation."""

    def test_perfect_ranking(self):
        """Top-k all positive should give precision=1."""
        y_true = np.array([1, 1, 1, 0, 0, 0], dtype=float)
        y_scores = np.array([6, 5, 4, 3, 2, 1], dtype=float)
        assert precision_at_k(y_true, y_scores, k=3) == pytest.approx(1.0)

    def test_worst_ranking(self):
        """Top-k all negative should give precision=0."""
        y_true = np.array([1, 1, 1, 0, 0, 0], dtype=float)
        y_scores = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        assert precision_at_k(y_true, y_scores, k=3) == pytest.approx(0.0)

    def test_partial_ranking(self):
        """Mixed top-k should give expected precision."""
        y_true = np.array([1, 0, 1, 0, 1, 0], dtype=float)
        y_scores = np.array([6, 5, 4, 3, 2, 1], dtype=float)
        # Top-3 scores: indices 0, 1, 2 → labels [1, 0, 1]
        assert precision_at_k(y_true, y_scores, k=3) == pytest.approx(2.0 / 3)


class TestEnrichmentFactor:
    """Test enrichment factor computation."""

    def test_perfect_enrichment(self):
        """Perfect model should achieve maximum enrichment."""
        n = 1000
        y_true = np.zeros(n)
        y_true[:50] = 1  # 5% active
        y_scores = np.zeros(n)
        y_scores[:50] = 10.0  # High scores for actives
        # At 5%, all top 50 are actives → EF = 1.0/0.05 = 20
        ef = enrichment_factor(y_true, y_scores, fraction=0.05)
        assert ef == pytest.approx(20.0)

    def test_random_enrichment(self):
        """Random ranking should give EF ≈ 1.0."""
        rng = np.random.RandomState(42)
        n = 10000
        y_true = (rng.rand(n) < 0.1).astype(float)  # 10% active
        y_scores = rng.randn(n)
        ef = enrichment_factor(y_true, y_scores, fraction=0.1)
        # Should be close to 1.0 (random)
        assert ef == pytest.approx(1.0, abs=0.3)

    def test_no_actives(self):
        """No actives should return 0."""
        y_true = np.zeros(100)
        y_scores = np.random.randn(100)
        ef = enrichment_factor(y_true, y_scores, fraction=0.01)
        assert ef == 0.0
