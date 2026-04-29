"""Aggregate results across multiple training seeds and split partitions.

Computes mean +/- SD of test metrics across seeds, enabling robust
uncertainty estimates that capture training stochasticity — not just
test-set sampling variability (which is what bootstrap CIs capture).

Usage:
    python -m target_affinity_ml.evaluation.multi_seed_analysis \
        --seeds 42 123 456 789 1024

    python -m target_affinity_ml.evaluation.multi_seed_analysis \
        --seeds 42 123 456 --models random_forest mlp esm_fp_mlp
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from target_affinity_ml.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")

ALL_MODELS = [
    "random_forest", "xgboost", "elasticnet", "mlp",
    "esm_fp_mlp", "gnn", "fusion",
]
ALL_SPLITS = ["random", "scaffold", "target"]
METRICS = ["rmse", "mae", "r2", "pearson_r", "spearman_rho", "auroc"]


def load_seed_predictions(
    model: str,
    split: str,
    seeds: list[int],
    pred_dir_pattern: str = "predictions_seed{seed}",
) -> list[dict]:
    """Load predictions from multiple seed runs.

    Parameters
    ----------
    model : str
        Model name.
    split : str
        Split strategy.
    seeds : list[int]
        Training seeds used.
    pred_dir_pattern : str
        Pattern for prediction directory names with {seed} placeholder.

    Returns
    -------
    list[dict]
        List of dicts with keys: seed, y_true, y_pred, y_active.
        Only seeds with existing prediction files are included.
    """
    results = []
    for seed in seeds:
        pred_dir = RESULTS_DIR / pred_dir_pattern.format(seed=seed)
        pred_path = pred_dir / f"{model}_{split}.npz"

        if not pred_path.exists():
            logger.warning("Missing predictions: %s", pred_path)
            continue

        data = np.load(pred_path)
        results.append({
            "seed": seed,
            "y_true": data["y_test_true"],
            "y_pred": data["y_test_pred"],
            "y_active": data["y_test_active"],
        })

    return results


def compute_seed_metrics(
    seed_predictions: list[dict],
) -> pd.DataFrame:
    """Compute metrics for each seed and aggregate.

    Parameters
    ----------
    seed_predictions : list[dict]
        Output of load_seed_predictions.

    Returns
    -------
    pd.DataFrame
        One row per seed with all metrics, plus a summary row.
    """
    rows = []
    for sp in seed_predictions:
        reg = compute_regression_metrics(sp["y_true"], sp["y_pred"])
        cls = compute_classification_metrics(sp["y_active"], sp["y_pred"])
        rows.append({
            "seed": sp["seed"],
            **reg,
            "auroc": cls.get("auroc", float("nan")),
        })

    return pd.DataFrame(rows)


def aggregate_across_seeds(
    models: list[str],
    splits: list[str],
    seeds: list[int],
    pred_dir_pattern: str = "predictions_seed{seed}",
) -> pd.DataFrame:
    """Compute mean +/- SD across seeds for all model-split combinations.

    Parameters
    ----------
    models : list[str]
        Model names to analyze.
    splits : list[str]
        Split strategies.
    seeds : list[int]
        Training seeds.
    pred_dir_pattern : str
        Pattern for prediction directories.

    Returns
    -------
    pd.DataFrame
        Aggregated results with columns:
        model, split, metric, mean, std, n_seeds, values
    """
    rows = []

    for model in models:
        for split in splits:
            seed_preds = load_seed_predictions(
                model, split, seeds, pred_dir_pattern,
            )

            if len(seed_preds) < 2:
                logger.warning(
                    "Skipping %s/%s: only %d seeds found (need >= 2)",
                    model, split, len(seed_preds),
                )
                continue

            seed_df = compute_seed_metrics(seed_preds)

            for metric in METRICS:
                if metric not in seed_df.columns:
                    continue
                values = seed_df[metric].dropna().values
                if len(values) == 0:
                    continue

                rows.append({
                    "model": model,
                    "split": split,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_seeds": len(values),
                    "values": values.tolist(),
                })

    return pd.DataFrame(rows)


def compute_pairwise_seed_significance(
    model_a: str,
    model_b: str,
    split: str,
    seeds: list[int],
    metric: str = "rmse",
    pred_dir_pattern: str = "predictions_seed{seed}",
) -> dict:
    """Paired t-test across seeds for two models.

    For each seed, compute the metric for both models on the same test set,
    then test whether the mean difference is significantly different from zero.

    Parameters
    ----------
    model_a, model_b : str
        Model names.
    split : str
        Split strategy.
    seeds : list[int]
        Training seeds.
    metric : str
        Metric to compare.
    pred_dir_pattern : str
        Pattern for prediction directories.

    Returns
    -------
    dict
        Paired comparison results.
    """
    from scipy import stats

    preds_a = load_seed_predictions(model_a, split, seeds, pred_dir_pattern)
    preds_b = load_seed_predictions(model_b, split, seeds, pred_dir_pattern)

    # Match by seed
    seeds_a = {sp["seed"]: sp for sp in preds_a}
    seeds_b = {sp["seed"]: sp for sp in preds_b}
    common_seeds = sorted(set(seeds_a.keys()) & set(seeds_b.keys()))

    if len(common_seeds) < 3:
        return {
            "model_a": model_a, "model_b": model_b,
            "split": split, "metric": metric,
            "n_seeds": len(common_seeds),
            "error": "insufficient seeds for paired test (need >= 3)",
        }

    values_a = []
    values_b = []
    for seed in common_seeds:
        reg_a = compute_regression_metrics(
            seeds_a[seed]["y_true"], seeds_a[seed]["y_pred"],
        )
        reg_b = compute_regression_metrics(
            seeds_b[seed]["y_true"], seeds_b[seed]["y_pred"],
        )
        values_a.append(reg_a.get(metric, float("nan")))
        values_b.append(reg_b.get(metric, float("nan")))

    values_a = np.array(values_a)
    values_b = np.array(values_b)
    deltas = values_a - values_b

    t_stat, p_value = stats.ttest_rel(values_a, values_b)

    return {
        "model_a": model_a,
        "model_b": model_b,
        "split": split,
        "metric": metric,
        "n_seeds": len(common_seeds),
        "mean_a": float(np.mean(values_a)),
        "mean_b": float(np.mean(values_b)),
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas, ddof=1)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_005": p_value < 0.05,
    }


def run_full_multi_seed_analysis(
    seeds: list[int],
    models: list[str] | None = None,
    splits: list[str] | None = None,
    pred_dir_pattern: str = "predictions_seed{seed}",
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run complete multi-seed analysis.

    Parameters
    ----------
    seeds : list[int]
        Training seeds.
    models : list[str], optional
        Models to analyze. Defaults to ALL_MODELS.
    splits : list[str], optional
        Splits to analyze. Defaults to ALL_SPLITS.
    pred_dir_pattern : str
        Pattern for prediction directories.
    output_dir : Path, optional
        Output directory. Defaults to results/tables.

    Returns
    -------
    dict
        "aggregated": DataFrame of mean +/- SD per model/split/metric
        "pairwise": DataFrame of paired t-tests for key comparisons
    """
    if models is None:
        models = ALL_MODELS
    if splits is None:
        splits = ALL_SPLITS
    if output_dir is None:
        output_dir = RESULTS_DIR / "tables"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Aggregate across seeds
    logger.info("Step 1: Aggregating metrics across %d seeds...", len(seeds))
    agg_df = aggregate_across_seeds(models, splits, seeds, pred_dir_pattern)

    if agg_df.empty:
        logger.warning("No multi-seed results found!")
        return {"aggregated": agg_df, "pairwise": pd.DataFrame()}

    # Drop the list column for CSV
    agg_save = agg_df.drop(columns=["values"], errors="ignore")
    agg_path = output_dir / "multi_seed_aggregated.csv"
    agg_save.to_csv(agg_path, index=False)
    logger.info("Saved aggregated results to %s", agg_path)

    # Print summary
    logger.info("\n--- Multi-seed Summary (RMSE, mean +/- SD) ---")
    rmse_df = agg_df[agg_df["metric"] == "rmse"][
        ["model", "split", "mean", "std", "n_seeds"]
    ].sort_values(["split", "mean"])
    logger.info("\n%s", rmse_df.to_string(index=False))

    # Step 2: Pairwise significance for key comparisons
    logger.info("\nStep 2: Pairwise significance tests...")
    key_pairs = [
        ("random_forest", "mlp"),
        ("random_forest", "esm_fp_mlp"),
        ("mlp", "esm_fp_mlp"),
        ("mlp", "gnn"),
        ("esm_fp_mlp", "fusion"),
        ("random_forest", "gnn"),
    ]

    pairwise_rows = []
    for model_a, model_b in key_pairs:
        if model_a not in models or model_b not in models:
            continue
        for split in splits:
            for metric in ["rmse", "r2", "pearson_r"]:
                result = compute_pairwise_seed_significance(
                    model_a, model_b, split, seeds, metric, pred_dir_pattern,
                )
                pairwise_rows.append(result)

    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        pairwise_path = output_dir / "multi_seed_pairwise.csv"
        pairwise_df.to_csv(pairwise_path, index=False)
        logger.info("Saved pairwise tests to %s", pairwise_path)

        sig = pairwise_df[pairwise_df.get("significant_005", False)]
        if not sig.empty:
            logger.info("\nSignificant differences (p < 0.05):")
            for _, row in sig.iterrows():
                logger.info(
                    "  %s vs %s (%s, %s): delta=%.4f, p=%.4f",
                    row["model_a"], row["model_b"],
                    row["split"], row["metric"],
                    row.get("mean_delta", float("nan")),
                    row.get("p_value", float("nan")),
                )

    return {"aggregated": agg_df, "pairwise": pairwise_df}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze results across multiple training seeds",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", required=True,
        help="Training seeds to aggregate (e.g., 42 123 456 789 1024)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Models to analyze (default: all)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=None,
        choices=ALL_SPLITS,
        help="Splits to analyze (default: all)",
    )
    parser.add_argument(
        "--pred-dir-pattern", default="predictions_seed{seed}",
        help="Pattern for prediction directories",
    )
    args = parser.parse_args()

    run_full_multi_seed_analysis(
        seeds=args.seeds,
        models=args.models,
        splits=args.splits,
        pred_dir_pattern=args.pred_dir_pattern,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
