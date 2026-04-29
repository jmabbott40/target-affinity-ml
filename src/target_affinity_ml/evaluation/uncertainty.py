"""Uncertainty quantification and calibration analysis.

A well-calibrated model's confidence should correlate with its actual
accuracy. This module provides tools to assess and visualize calibration:

    - Calibration curves: do 90% prediction intervals contain 90% of true values?
    - Miscalibration area: numerical measure of calibration quality
    - Error-uncertainty correlation: do uncertain predictions have higher errors?
    - Selective prediction: how much does accuracy improve if we abstain on
      high-uncertainty predictions?

These analyses are key to understanding when to trust model predictions,
which is critical in drug discovery where false positives waste
experimental resources.

Usage:
    from target_affinity_ml.evaluation.uncertainty import calibration_curve
    expected, observed = calibration_curve(y_true, y_pred, y_std)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr


def calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute regression calibration curve.

    For each confidence level (e.g., 50%, 60%, ..., 95%), compute the
    fraction of true values that fall within the corresponding prediction
    interval. A perfectly calibrated model should have expected = observed.

    The approach: for confidence level p, the prediction interval is
    [y_pred - z*y_std, y_pred + z*y_std] where z = norm.ppf((1+p)/2).
    The observed coverage is the fraction of y_true falling inside.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted means.
    y_std : np.ndarray
        Predicted standard deviations.
    n_bins : int
        Number of confidence levels to evaluate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (expected_coverage, observed_coverage) arrays of shape (n_bins,).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_std = np.asarray(y_std, dtype=np.float64)

    # Confidence levels from ~10% to ~95%
    expected = np.linspace(1 / (n_bins + 1), n_bins / (n_bins + 1), n_bins)

    # Standardized residuals (z-scores)
    # Clamp std to avoid division by zero
    y_std_safe = np.maximum(y_std, 1e-10)
    abs_z = np.abs((y_true - y_pred) / y_std_safe)

    observed = np.zeros(n_bins)
    for i, p in enumerate(expected):
        # Critical z-value for confidence level p
        z_crit = norm.ppf((1 + p) / 2)
        # Fraction of residuals within this interval
        observed[i] = np.mean(abs_z <= z_crit)

    return expected, observed


def miscalibration_area(
    expected: np.ndarray,
    observed: np.ndarray,
) -> float:
    """Compute the miscalibration area (area between calibration curve and diagonal).

    Lower is better. Zero means perfect calibration. Maximum is 0.5.

    Uses the trapezoidal rule to integrate |expected - observed|.

    Parameters
    ----------
    expected : np.ndarray
        Expected coverage fractions.
    observed : np.ndarray
        Observed coverage fractions.

    Returns
    -------
    float
        Miscalibration area.
    """
    expected = np.asarray(expected, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    return float(np.trapezoid(np.abs(expected - observed), expected))


def error_uncertainty_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> dict[str, float]:
    """Assess correlation between prediction error and uncertainty.

    A good uncertainty estimate should have high correlation with
    absolute prediction error — uncertain predictions should have
    larger errors on average.

    Parameters
    ----------
    y_true, y_pred, y_std : np.ndarray
        True values, predictions, and uncertainty estimates.

    Returns
    -------
    dict[str, float]
        Pearson and Spearman correlation between |error| and uncertainty,
        plus p-values.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_std = np.asarray(y_std, dtype=np.float64)

    abs_error = np.abs(y_true - y_pred)

    # Handle constant arrays (e.g., ElasticNet with zero-std predictions)
    if np.std(y_std) < 1e-10 or np.std(abs_error) < 1e-10:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
        }

    pr, pp = pearsonr(abs_error, y_std)
    sr, sp = spearmanr(abs_error, y_std)

    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
    }


def selective_prediction_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_points: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute RMSE as a function of retention fraction.

    If we only keep the most confident predictions (lowest uncertainty),
    how much does RMSE improve? This shows the value of uncertainty
    estimation for practical decision-making.

    The curve starts at retention=1.0 (all predictions, baseline RMSE)
    and moves toward retention→0 (only the most confident predictions).

    Parameters
    ----------
    y_true, y_pred, y_std : np.ndarray
        True values, predictions, and uncertainty estimates.
    n_points : int
        Number of retention fractions to evaluate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (retention_fraction, rmse_at_fraction) arrays.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_std = np.asarray(y_std, dtype=np.float64)

    n = len(y_true)

    # Sort by uncertainty (ascending = most confident first)
    order = np.argsort(y_std)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]

    # Retention fractions from small to full
    fractions = np.linspace(1 / n_points, 1.0, n_points)
    rmses = np.zeros(n_points)

    for i, frac in enumerate(fractions):
        k = max(1, int(n * frac))
        residuals = y_true_sorted[:k] - y_pred_sorted[:k]
        rmses[i] = np.sqrt(np.mean(residuals ** 2))

    return fractions, rmses
