"""Error analysis and failure mode identification.

Goes beyond aggregate metrics to understand *why* models fail:

    - Worst predictions: compounds with largest absolute errors
    - Per-target breakdown: which kinases are easier/harder to predict?
    - Noise impact: are compounds flagged as "noisy" during curation
      systematically harder to predict?

This analysis drives scientific insight and guides model improvement.

Usage:
    from target_affinity_ml.evaluation.analysis import find_worst_predictions
    worst = find_worst_predictions(y_true, y_pred, df, top_n=50)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from target_affinity_ml.evaluation.metrics import compute_regression_metrics


def find_worst_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    """Identify the top-N worst predicted compounds.

    Returns a DataFrame sorted by absolute error (descending),
    including both the prediction details and compound metadata
    for investigating failure modes.

    Parameters
    ----------
    y_true : np.ndarray
        True pActivity values.
    y_pred : np.ndarray
        Predicted pActivity values.
    df : pd.DataFrame
        Original dataset rows corresponding to the predictions.
        Must have the same length as y_true/y_pred.
    top_n : int
        Number of worst predictions to return.

    Returns
    -------
    pd.DataFrame
        Worst predicted compounds with error, true/predicted values,
        and metadata for analysis.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    abs_error = np.abs(y_true - y_pred)
    signed_error = y_pred - y_true  # positive = overprediction

    result = df.copy().reset_index(drop=True)
    result["y_true"] = y_true
    result["y_pred"] = y_pred
    result["abs_error"] = abs_error
    result["signed_error"] = signed_error

    result = result.sort_values("abs_error", ascending=False).head(top_n)
    return result.reset_index(drop=True)


def per_target_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_ids: np.ndarray,
    min_samples: int = 10,
) -> pd.DataFrame:
    """Compute metrics broken down by target.

    Only targets with at least `min_samples` test compounds are
    included, since metrics are unreliable on very small sets.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted pActivity values.
    target_ids : np.ndarray
        Target identifiers for each measurement.
    min_samples : int
        Minimum number of test compounds per target.

    Returns
    -------
    pd.DataFrame
        Metrics (RMSE, MAE, R², Pearson, Spearman, count) per target,
        sorted by RMSE ascending.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    target_ids = np.asarray(target_ids)

    records = []
    for target in np.unique(target_ids):
        mask = target_ids == target
        n = mask.sum()
        if n < min_samples:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        # Skip targets with no variance (can't compute R²/correlation)
        if np.std(yt) < 1e-10:
            continue

        metrics = compute_regression_metrics(yt, yp)
        metrics["target_id"] = target
        metrics["n_compounds"] = n
        records.append(metrics)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    col_order = ["target_id", "n_compounds", "rmse", "mae", "r2",
                 "pearson_r", "spearman_rho"]
    result = result[[c for c in col_order if c in result.columns]]
    return result.sort_values("rmse", ascending=True).reset_index(drop=True)


def noise_impact_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_noisy: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compare model performance on noisy vs clean compounds.

    Noisy compounds have high measurement variance across replicate
    experiments (std > 1.0 pActivity units). If models perform worse
    on these, it suggests measurement noise — not model failure — is
    the bottleneck.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted values.
    is_noisy : np.ndarray
        Boolean array indicating noisy compounds (high measurement variance).

    Returns
    -------
    dict
        {'clean': {metrics}, 'noisy': {metrics}, 'delta': {metrics}}
        where delta = noisy - clean (positive means noisy is worse for RMSE).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    is_noisy = np.asarray(is_noisy, dtype=bool)

    clean_mask = ~is_noisy
    noisy_mask = is_noisy

    result = {
        "n_clean": int(clean_mask.sum()),
        "n_noisy": int(noisy_mask.sum()),
    }

    if clean_mask.sum() >= 2:
        result["clean"] = compute_regression_metrics(
            y_true[clean_mask], y_pred[clean_mask],
        )
    else:
        result["clean"] = {}

    if noisy_mask.sum() >= 2:
        result["noisy"] = compute_regression_metrics(
            y_true[noisy_mask], y_pred[noisy_mask],
        )
    else:
        result["noisy"] = {}

    # Compute deltas where both exist
    if result["clean"] and result["noisy"]:
        result["delta"] = {
            k: result["noisy"][k] - result["clean"][k]
            for k in result["clean"]
        }

    return result
