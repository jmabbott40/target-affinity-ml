"""Bootstrap confidence intervals and paired significance tests for test metrics.

Resamples existing predictions (no retraining required) to compute:
    1. Per-model CIs on RMSE, MAE, R², Pearson R, Spearman ρ, AUROC
    2. Paired bootstrap tests between model pairs (is the difference significant?)
    3. Win-rate matrices (fraction of bootstrap samples where model A beats B)

Usage:
    # Single model CI
    result = bootstrap_metrics(y_true, y_pred, n_bootstrap=10000)
    # result["rmse"] = {"mean": 0.818, "ci_lo": 0.810, "ci_hi": 0.826}

    # Paired test
    result = paired_bootstrap_test(y_true, y_pred_a, y_pred_b, metric="rmse")
    # result = {"delta": -0.043, "ci_lo": -0.052, "ci_hi": -0.034, "p_value": 0.001}

    # Full pipeline across all models/splits
    python -m target_affinity_ml.evaluation.bootstrap
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, roc_auc_score, root_mean_squared_error

logger = logging.getLogger(__name__)


def _compute_metric(y_true, y_pred, metric: str, y_active=None) -> float:
    """Compute a single metric value."""
    if metric == "rmse":
        return root_mean_squared_error(y_true, y_pred)
    elif metric == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    elif metric == "r2":
        return r2_score(y_true, y_pred)
    elif metric == "pearson_r":
        if np.std(y_pred) < 1e-10 or np.std(y_true) < 1e-10:
            return float("nan")
        return pearsonr(y_true, y_pred)[0]
    elif metric == "spearman_rho":
        if np.std(y_pred) < 1e-10 or np.std(y_true) < 1e-10:
            return float("nan")
        return spearmanr(y_true, y_pred).correlation
    elif metric == "auroc":
        if y_active is None:
            raise ValueError("y_active required for AUROC")
        n_pos, n_neg = y_active.sum(), len(y_active) - y_active.sum()
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        return roc_auc_score(y_active, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")


METRICS_HIGHER_BETTER = {"r2", "pearson_r", "spearman_rho", "auroc"}
METRICS_LOWER_BETTER = {"rmse", "mae"}
ALL_METRICS = ["rmse", "mae", "r2", "pearson_r", "spearman_rho", "auroc"]


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_active: np.ndarray | None = None,
    metrics: list[str] | None = None,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute bootstrap confidence intervals for multiple metrics.

    Parameters
    ----------
    y_true : (n,) array
        True pActivity values.
    y_pred : (n,) array
        Predicted pActivity values.
    y_active : (n,) array, optional
        Binary active/inactive labels (needed for AUROC).
    metrics : list of str
        Metrics to compute. Defaults to all regression metrics + AUROC.
    n_bootstrap : int
        Number of bootstrap samples.
    ci : float
        Confidence level (e.g. 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, dict]
        {metric_name: {"point": float, "mean": float, "ci_lo": float, "ci_hi": float, "std": float}}
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_active is not None:
        y_active = np.asarray(y_active, dtype=np.float64)

    if metrics is None:
        metrics = ["rmse", "mae", "r2", "pearson_r", "spearman_rho"]
        if y_active is not None:
            metrics.append("auroc")

    n = len(y_true)
    alpha = (1 - ci) / 2
    rng = np.random.default_rng(seed)

    # Pre-generate all bootstrap index arrays at once
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))

    # Pre-compute residuals for fast RMSE/MAE vectorization
    residuals = y_true - y_pred

    results = {}
    for metric in metrics:
        # Point estimate on full data
        point = _compute_metric(y_true, y_pred, metric,
                                y_active if metric == "auroc" else None)

        # Fast path: vectorize RMSE and MAE without Python loop
        if metric == "rmse":
            boot_residuals = residuals[boot_indices]  # (n_bootstrap, n)
            boot_values = np.sqrt(np.mean(boot_residuals ** 2, axis=1))
        elif metric == "mae":
            boot_residuals = residuals[boot_indices]
            boot_values = np.mean(np.abs(boot_residuals), axis=1)
        else:
            # General path for correlation-based metrics
            boot_values = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                idx = boot_indices[b]
                yt, yp = y_true[idx], y_pred[idx]
                ya = y_active[idx] if y_active is not None else None
                boot_values[b] = _compute_metric(yt, yp, metric, ya)

        # Filter NaN (can happen with AUROC or constant predictions)
        valid = ~np.isnan(boot_values)
        boot_valid = boot_values[valid]

        if len(boot_valid) == 0:
            logger.warning("  %s: all %d bootstrap samples had NaN — skipping",
                           metric, n_bootstrap)
            results[metric] = {
                "point": float(point),
                "mean": float("nan"),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "std": float("nan"),
            }
            continue

        if len(boot_valid) < n_bootstrap * 0.9:
            logger.warning("  %s: %d/%d bootstrap samples had NaN",
                           metric, n_bootstrap - len(boot_valid), n_bootstrap)

        results[metric] = {
            "point": float(point),
            "mean": float(np.mean(boot_valid)),
            "ci_lo": float(np.percentile(boot_valid, alpha * 100)),
            "ci_hi": float(np.percentile(boot_valid, (1 - alpha) * 100)),
            "std": float(np.std(boot_valid)),
        }

    return results


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric: str = "rmse",
    y_active: np.ndarray | None = None,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap test: is model A significantly different from model B?

    Resamples the same indices for both models, computes the difference in
    metric values, and checks whether the difference CI excludes zero.

    Parameters
    ----------
    y_true : (n,) array
    y_pred_a, y_pred_b : (n,) arrays
        Predictions from models A and B on the same test set.
    metric : str
        Metric to compare.
    y_active : (n,) array, optional
        Binary labels (for AUROC).
    n_bootstrap : int
    ci : float
    seed : int

    Returns
    -------
    dict with keys:
        delta_point : float
            Point estimate of metric_A - metric_B.
        delta_mean : float
            Mean bootstrap difference.
        ci_lo, ci_hi : float
            CI bounds on the difference.
        p_value : float
            Two-sided p-value (fraction of samples where sign flips).
        significant : bool
            True if CI excludes zero.
        a_better_frac : float
            Fraction of bootstrap samples where A is better than B.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_a = np.asarray(y_pred_a, dtype=np.float64)
    y_pred_b = np.asarray(y_pred_b, dtype=np.float64)
    if y_active is not None:
        y_active = np.asarray(y_active, dtype=np.float64)

    n = len(y_true)
    alpha = (1 - ci) / 2
    rng = np.random.default_rng(seed)

    ya_arg = y_active if metric == "auroc" else None

    point_a = _compute_metric(y_true, y_pred_a, metric, ya_arg)
    point_b = _compute_metric(y_true, y_pred_b, metric, ya_arg)
    delta_point = point_a - point_b

    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))

    # Fast path for RMSE/MAE
    if metric in ("rmse", "mae"):
        res_a = y_true - y_pred_a
        res_b = y_true - y_pred_b
        boot_res_a = res_a[boot_indices]
        boot_res_b = res_b[boot_indices]
        if metric == "rmse":
            vals_a = np.sqrt(np.mean(boot_res_a ** 2, axis=1))
            vals_b = np.sqrt(np.mean(boot_res_b ** 2, axis=1))
        else:
            vals_a = np.mean(np.abs(boot_res_a), axis=1)
            vals_b = np.mean(np.abs(boot_res_b), axis=1)
        deltas = vals_a - vals_b
    else:
        deltas = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = boot_indices[b]
            yt = y_true[idx]
            ya = y_active[idx] if y_active is not None else None
            val_a = _compute_metric(yt, y_pred_a[idx], metric, ya)
            val_b = _compute_metric(yt, y_pred_b[idx], metric, ya)
            deltas[b] = val_a - val_b

    valid = ~np.isnan(deltas)
    deltas_valid = deltas[valid]

    if len(deltas_valid) == 0:
        return {
            "metric": metric,
            "point_a": float(point_a),
            "point_b": float(point_b),
            "delta_point": float(delta_point),
            "delta_mean": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "a_better_frac": float("nan"),
        }

    ci_lo = float(np.percentile(deltas_valid, alpha * 100))
    ci_hi = float(np.percentile(deltas_valid, (1 - alpha) * 100))

    # Two-sided p-value: fraction of samples on the opposite side of zero
    if delta_point >= 0:
        p_value = float(np.mean(deltas_valid < 0)) * 2
    else:
        p_value = float(np.mean(deltas_valid > 0)) * 2
    p_value = min(p_value, 1.0)

    # For lower-is-better metrics, "A better" means delta < 0
    if metric in METRICS_LOWER_BETTER:
        a_better_frac = float(np.mean(deltas_valid < 0))
    else:
        a_better_frac = float(np.mean(deltas_valid > 0))

    significant = (ci_lo > 0) or (ci_hi < 0)  # CI excludes zero

    return {
        "metric": metric,
        "point_a": float(point_a),
        "point_b": float(point_b),
        "delta_point": float(delta_point),
        "delta_mean": float(np.mean(deltas_valid)),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
        "significant": significant,
        "a_better_frac": a_better_frac,
    }


def compute_win_rate_matrix(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    metric: str = "rmse",
    y_active: np.ndarray | None = None,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise win-rate matrix across all models.

    Returns
    -------
    win_rates : (n_models, n_models) array
        win_rates[i, j] = fraction of bootstrap samples where model i beats model j.
    model_names : list of str
    """
    model_names = list(predictions.keys())
    n_models = len(model_names)

    y_true = np.asarray(y_true, dtype=np.float64)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    # Pre-generate bootstrap indices for consistency
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))

    # Pre-compute all bootstrap metric values
    boot_metrics = {}
    for name, y_pred in predictions.items():
        y_pred = np.asarray(y_pred, dtype=np.float64)
        values = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = boot_indices[b]
            ya = y_active[idx] if y_active is not None and metric == "auroc" else None
            values[b] = _compute_metric(y_true[idx], y_pred[idx], metric, ya)
        boot_metrics[name] = values

    # Compute win rates
    win_rates = np.zeros((n_models, n_models))
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            if i == j:
                win_rates[i, j] = 0.5
                continue
            if metric in METRICS_LOWER_BETTER:
                win_rates[i, j] = np.mean(
                    boot_metrics[name_i] < boot_metrics[name_j]
                )
            else:
                win_rates[i, j] = np.mean(
                    boot_metrics[name_i] > boot_metrics[name_j]
                )

    return win_rates, model_names


# ═══════════════════════════════════════════════════════════════════════
# CLI: Run bootstrap analysis across all experiments
# ═══════════════════════════════════════════════════════════════════════

def run_full_bootstrap_analysis(
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
) -> None:
    """Run bootstrap CI and paired tests for all model × split combinations."""
    import json
    from pathlib import Path
    import pandas as pd

    PRED_DIR = Path("results/predictions")
    TABLES_DIR = Path("results/tables")
    FIGURES_DIR = Path("results/figures")

    ALL_MODELS = [
        "random_forest", "xgboost", "elasticnet", "mlp",
        "esm_fp_mlp", "gnn", "fusion",
    ]
    ALL_SPLITS = ["random", "scaffold", "target"]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Per-model bootstrap CIs ──
    logger.info("=" * 60)
    logger.info("STEP 1: Bootstrap confidence intervals (n=%d)", n_bootstrap)
    logger.info("=" * 60)

    ci_rows = []
    for split in ALL_SPLITS:
        for model in ALL_MODELS:
            path = PRED_DIR / f"{model}_{split}.npz"
            if not path.exists():
                continue

            logger.info("  %s × %s ...", model, split)
            data = np.load(path)
            y_true = data["y_test_true"]
            y_pred = data["y_test_pred"]
            y_active = data["y_test_active"] if "y_test_active" in data else None

            result = bootstrap_metrics(
                y_true, y_pred, y_active,
                n_bootstrap=n_bootstrap, ci=ci,
            )

            for metric, vals in result.items():
                ci_rows.append({
                    "model": model,
                    "split": split,
                    "metric": metric,
                    **vals,
                })
                logger.info("    %s: %.4f [%.4f, %.4f]",
                            metric, vals["point"], vals["ci_lo"], vals["ci_hi"])

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(TABLES_DIR / "bootstrap_confidence_intervals.csv", index=False)
    logger.info("Saved: bootstrap_confidence_intervals.csv (%d rows)", len(ci_df))

    # ── Step 2: Paired bootstrap tests ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Paired bootstrap tests")
    logger.info("=" * 60)

    # Key comparisons: best baseline vs best deep, and within-tier comparisons
    pair_rows = []
    for split in ALL_SPLITS:
        # Load all predictions for this split
        preds = {}
        actives = {}
        y_true_split = None
        for model in ALL_MODELS:
            path = PRED_DIR / f"{model}_{split}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            preds[model] = data["y_test_pred"]
            if "y_test_active" in data:
                actives[model] = data["y_test_active"]
            if y_true_split is None:
                y_true_split = data["y_test_true"]
                y_active_split = data.get("y_test_active")

        if y_true_split is None:
            continue

        # All unique pairs
        model_list = [m for m in ALL_MODELS if m in preds]
        for i, model_a in enumerate(model_list):
            for model_b in model_list[i + 1:]:
                for metric in ["rmse", "r2", "pearson_r"]:
                    ya = y_active_split if metric == "auroc" else None
                    result = paired_bootstrap_test(
                        y_true_split, preds[model_a], preds[model_b],
                        metric=metric, y_active=ya,
                        n_bootstrap=n_bootstrap, ci=ci,
                    )
                    result["model_a"] = model_a
                    result["model_b"] = model_b
                    result["split"] = split
                    pair_rows.append(result)

                    sig = "***" if result["significant"] else "   "
                    logger.info("  %s %s vs %s (%s): Δ=%.4f [%.4f, %.4f] %s",
                                split, model_a, model_b, metric,
                                result["delta_point"],
                                result["ci_lo"], result["ci_hi"], sig)

    pairs_df = pd.DataFrame(pair_rows)
    pairs_df.to_csv(TABLES_DIR / "bootstrap_paired_tests.csv", index=False)
    logger.info("Saved: bootstrap_paired_tests.csv (%d rows)", len(pairs_df))

    # ── Step 3: Win-rate matrices ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Win-rate matrices")
    logger.info("=" * 60)

    from target_affinity_ml.visualization.plots import MODEL_DISPLAY_NAMES

    for split in ALL_SPLITS:
        preds = {}
        y_true_split = None
        for model in ALL_MODELS:
            path = PRED_DIR / f"{model}_{split}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            preds[model] = data["y_test_pred"]
            if y_true_split is None:
                y_true_split = data["y_test_true"]

        if y_true_split is None or len(preds) < 2:
            continue

        win_rates, model_names = compute_win_rate_matrix(
            y_true_split, preds, metric="rmse",
            n_bootstrap=n_bootstrap,
        )

        _plot_win_rate_matrix(win_rates, model_names, split,
                             FIGURES_DIR / f"bootstrap_winrate_{split}.png")

    # ── Step 4: CI comparison plot ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: CI comparison plots")
    logger.info("=" * 60)

    _plot_ci_comparison(ci_df, FIGURES_DIR)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Bootstrap analysis complete!")
    logger.info("=" * 60)


def _plot_win_rate_matrix(win_rates, model_names, split, save_path):
    """Heatmap of pairwise win rates."""
    from target_affinity_ml.visualization.plots import MODEL_DISPLAY_NAMES

    n = len(model_names)
    display = [MODEL_DISPLAY_NAMES.get(m, m) for m in model_names]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(win_rates, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(display, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(display, fontsize=9)

    # Separator between baselines and deep models
    n_baselines = sum(1 for m in model_names if m in
                      {"random_forest", "xgboost", "elasticnet", "mlp"})
    if 0 < n_baselines < n:
        ax.axhline(n_baselines - 0.5, color="white", linewidth=2)
        ax.axvline(n_baselines - 0.5, color="white", linewidth=2)

    for i in range(n):
        for j in range(n):
            val = win_rates[i, j]
            color = "white" if val > 0.7 or val < 0.3 else "black"
            weight = "bold" if abs(val - 0.5) > 0.4 else "normal"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                    fontsize=10, color=color, fontweight=weight)

    fig.colorbar(im, ax=ax, label="Win Rate (row beats column, RMSE)",
                 shrink=0.8)
    ax.set_title(f"Bootstrap Win Rate Matrix — {split.capitalize()} Split\n"
                 f"(10K resamples, lower RMSE = win)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", save_path.name)


def _plot_ci_comparison(ci_df, figures_dir):
    """Forest plot: point estimates with CI error bars for key metrics."""
    import matplotlib.pyplot as plt
    from target_affinity_ml.visualization.plots import (
        MODEL_COLORS, MODEL_DISPLAY_NAMES, MODEL_ORDER,
    )

    for metric in ["rmse", "r2", "pearson_r"]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

        for ax, split in zip(axes, ["random", "scaffold", "target"]):
            sub = ci_df[(ci_df["split"] == split) & (ci_df["metric"] == metric)]
            if len(sub) == 0:
                continue

            # Order models
            ordered = [m for m in MODEL_ORDER if m in sub["model"].values]
            sub = sub.set_index("model").loc[ordered].reset_index()

            y_pos = np.arange(len(sub))
            colors = [MODEL_COLORS.get(m, "#888") for m in sub["model"]]
            labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in sub["model"]]

            points = sub["point"].values
            lo = points - sub["ci_lo"].values
            hi = sub["ci_hi"].values - points

            ax.barh(y_pos, points, xerr=[lo, hi], color=colors,
                    alpha=0.8, edgecolor="white", capsize=4, height=0.6)

            # Separator
            n_bl = sum(1 for m in sub["model"] if m in
                       {"random_forest", "xgboost", "elasticnet", "mlp"})
            if 0 < n_bl < len(sub):
                ax.axhline(n_bl - 0.5, color="gray", linestyle="--", alpha=0.4)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels if split == "random" else [], fontsize=9)
            ax.set_title(f"{split.capitalize()}", fontsize=11)
            ax.invert_yaxis()

            # Annotate values
            for i, (p, lo_v, hi_v) in enumerate(
                zip(points, sub["ci_lo"].values, sub["ci_hi"].values)
            ):
                ax.text(p, i, f" {p:.3f}", va="center", fontsize=8, alpha=0.8)

        metric_display = {
            "rmse": "RMSE (↓ better)",
            "r2": "R² (↑ better)",
            "pearson_r": "Pearson R (↑ better)",
        }
        fig.suptitle(f"Bootstrap 95% CI — {metric_display.get(metric, metric)}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        path = figures_dir / f"bootstrap_ci_{metric}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved: %s", path.name)


import matplotlib.pyplot as plt


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for test metrics",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000,
                        help="Number of bootstrap samples (default: 10000)")
    parser.add_argument("--ci", type=float, default=0.95,
                        help="Confidence level (default: 0.95)")
    args = parser.parse_args()

    run_full_bootstrap_analysis(n_bootstrap=args.n_bootstrap, ci=args.ci)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
