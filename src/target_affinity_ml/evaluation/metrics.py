"""Regression and classification evaluation metrics.

Metrics selected for their relevance to drug discovery:

Regression (predicting pActivity values):
    - RMSE: Root mean squared error (penalizes large errors)
    - MAE: Mean absolute error (robust to outliers)
    - R²: Coefficient of determination (explained variance)
    - Pearson R: Linear correlation between predicted and actual
    - Spearman ρ: Rank correlation (important for compound ranking)

Classification (active/inactive at pActivity threshold):
    - AUROC: Area under ROC curve (overall discrimination)
    - AUPRC: Area under precision-recall curve (better for imbalanced data)
    - Precision@k: Precision in top-k predictions (drug discovery use case:
      "if I test my top 100 predictions, how many are truly active?")
    - Enrichment factor: fold improvement over random selection

Usage:
    from target_affinity_ml.evaluation.metrics import compute_regression_metrics
    metrics = compute_regression_metrics(y_true, y_pred)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute all regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True pActivity values.
    y_pred : np.ndarray
        Predicted pActivity values.

    Returns
    -------
    dict[str, float]
        Dictionary of metric_name → value.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearsonr(y_true, y_pred)[0]),
        "spearman_rho": float(spearmanr(y_true, y_pred).correlation),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics.

    In this project, y_pred_proba is typically the raw pActivity prediction
    (higher = more likely active), and y_true is the binary is_active label.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1).
    y_pred_proba : np.ndarray
        Predicted scores (higher = more likely positive/active).
    threshold : float
        Decision threshold for binary classification.

    Returns
    -------
    dict[str, float]
        Dictionary of metric_name → value.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)

    metrics = {}

    # AUROC — requires both classes to be present
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos > 0 and n_neg > 0:
        metrics["auroc"] = float(roc_auc_score(y_true, y_pred_proba))
        metrics["auprc"] = float(average_precision_score(y_true, y_pred_proba))
    else:
        logger.warning("Only one class present — AUROC/AUPRC undefined")
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    # Precision@k at various k values
    for k in [100, 500]:
        if k <= len(y_true):
            metrics[f"precision_at_{k}"] = precision_at_k(y_true, y_pred_proba, k)
        else:
            metrics[f"precision_at_{k}"] = float("nan")

    # Enrichment factor at 1% and 5%
    metrics["ef_1pct"] = enrichment_factor(y_true, y_pred_proba, fraction=0.01)
    metrics["ef_5pct"] = enrichment_factor(y_true, y_pred_proba, fraction=0.05)

    return metrics


def precision_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k: int,
) -> float:
    """Compute precision in the top-k predictions.

    Answers: "If I select the k compounds with the highest predicted
    activity, what fraction are truly active?"

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray
        Predicted scores (higher = more likely active).
    k : int
        Number of top predictions to evaluate.

    Returns
    -------
    float
        Precision in top-k.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    # Get indices of top-k scores (descending)
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    return float(np.mean(y_true[top_k_idx]))


def enrichment_factor(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    fraction: float = 0.01,
) -> float:
    """Compute enrichment factor at a given fraction.

    EF = (hits_in_top_fraction / n_top) / (total_hits / n_total)

    An EF of 10 at 1% means the model finds 10x more actives in
    its top 1% than random selection would.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray
        Predicted scores.
    fraction : float
        Top fraction to evaluate (e.g., 0.01 for top 1%).

    Returns
    -------
    float
        Enrichment factor.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    n = len(y_true)
    n_top = max(1, int(n * fraction))
    total_hits = y_true.sum()

    if total_hits == 0:
        return 0.0

    top_idx = np.argsort(y_scores)[::-1][:n_top]
    hits_in_top = y_true[top_idx].sum()

    return float((hits_in_top / n_top) / (total_hits / n))
