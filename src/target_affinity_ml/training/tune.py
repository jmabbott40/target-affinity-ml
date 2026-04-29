"""Hyperparameter tuning via validation set search.

Uses the existing train/val splits to find optimal hyperparameters.
For each candidate in the search grid, trains on the train set and
evaluates on the validation set. The best configuration is then
re-evaluated on the test set.

Focuses on the two models that need tuning:
    - ElasticNet: alpha is too aggressive at 1.0 (all coefficients collapse)
    - XGBoost: max_depth may be limiting on 2048-dim binary data

Usage:
    # Tune ElasticNet on random split
    python -m target_affinity_ml.training.tune --model elasticnet --split random

    # Tune all models that need it
    python -m target_affinity_ml.training.tune --all
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from target_affinity_ml.evaluation.metrics import compute_regression_metrics
from target_affinity_ml.features import load_morgan_fingerprints, load_rdkit_descriptors
from target_affinity_ml.training.trainer import (
    build_feature_matrix,
    get_model_class,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

# Models + splits to tune
TUNE_CONFIGS = {
    "elasticnet": {
        "config_path": Path("configs/elasticnet_baseline.yaml"),
        "search_grid": {
            "alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        "fixed_params": {"max_iter": 10000, "random_state": 42},
        "extra_kwargs": {"n_bootstrap": 10},  # Fewer for speed during tuning
    },
    "xgboost": {
        "config_path": Path("configs/xgb_baseline.yaml"),
        "search_grid": {
            "max_depth": [4, 6, 8, 10, 12],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [300, 500],
        },
        "fixed_params": {
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        },
        "extra_kwargs": {},
    },
}


def tune_model(
    model_name: str,
    split_strategy: str = "random",
    dataset_version: str = "v1",
) -> dict:
    """Run hyperparameter search for a model on a specific split.

    Parameters
    ----------
    model_name : str
        One of 'elasticnet', 'xgboost'.
    split_strategy : str
        Split to tune on.
    dataset_version : str
        Dataset version.

    Returns
    -------
    dict
        Best hyperparameters and metrics.
    """
    tune_config = TUNE_CONFIGS[model_name]
    search_grid = tune_config["search_grid"]
    fixed_params = tune_config["fixed_params"]
    extra_kwargs = tune_config["extra_kwargs"]

    # Load config to get feature type
    with open(tune_config["config_path"]) as f:
        config = yaml.safe_load(f)
    feature_type = config["features"]["type"]

    logger.info("=" * 60)
    logger.info("TUNING: %s on %s split", model_name, split_strategy)
    logger.info("  Search grid: %s", {k: len(v) for k, v in search_grid.items()})
    logger.info("=" * 60)

    # Load data
    data_dir = DATA_DIR / dataset_version
    df = pd.read_parquet(data_dir / "curated_activities.parquet")

    with open(data_dir / "splits" / f"{split_strategy}_split.json") as f:
        split_indices = json.load(f)

    if feature_type == "morgan_fingerprint":
        feature_matrix, smiles_list = load_morgan_fingerprints(dataset_version)
    else:
        feature_matrix, _, smiles_list = load_rdkit_descriptors(dataset_version)

    smiles_to_row = {s: i for i, s in enumerate(smiles_list)}

    X_train, y_train, _ = build_feature_matrix(
        df, split_indices["train"], smiles_to_row, feature_matrix,
    )
    X_val, y_val, _ = build_feature_matrix(
        df, split_indices["val"], smiles_to_row, feature_matrix,
    )

    # Generate all combinations
    param_names = list(search_grid.keys())
    param_values = list(search_grid.values())
    combinations = list(itertools.product(*param_values))
    total = len(combinations)

    logger.info("  Total combinations: %d", total)

    best_rmse = float("inf")
    best_params = None
    best_metrics = None
    results = []

    model_cls = get_model_class(model_name)

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo, strict=False))
        params.update(fixed_params)

        logger.info("  [%d/%d] %s", i + 1, total, params)

        try:
            t0 = time.time()
            model = model_cls(**extra_kwargs, **params)
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            elapsed = time.time() - t0

            metrics = compute_regression_metrics(y_val, y_val_pred)
            metrics["params"] = params
            metrics["train_time"] = round(elapsed, 1)
            results.append(metrics)

            logger.info("    RMSE=%.4f, R²=%.4f (%.1fs)",
                         metrics["rmse"], metrics["r2"], elapsed)

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_params = params
                best_metrics = metrics
        except Exception:
            logger.exception("    FAILED")

    logger.info("=" * 60)
    logger.info("BEST: RMSE=%.4f, R²=%.4f", best_metrics["rmse"], best_metrics["r2"])
    logger.info("  Params: %s", best_params)
    logger.info("=" * 60)

    # Save tuning results
    tune_dir = RESULTS_DIR / "tuning"
    tune_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        tune_dir / f"tuning_{model_name}_{split_strategy}.csv",
        index=False,
    )

    # Save best params
    best_result = {
        "model": model_name,
        "split": split_strategy,
        "best_params": best_params,
        "best_val_rmse": best_rmse,
        "best_val_r2": best_metrics["r2"],
        "n_combinations_tested": total,
    }
    with open(tune_dir / f"best_{model_name}_{split_strategy}.json", "w") as f:
        json.dump(best_result, f, indent=2)

    return best_result


def tune_all(dataset_version: str = "v1") -> list[dict]:
    """Tune all models that need it, on the random split."""
    results = []
    for model_name in TUNE_CONFIGS:
        result = tune_model(model_name, "random", dataset_version)
        results.append(result)
    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("--model", choices=list(TUNE_CONFIGS.keys()))
    parser.add_argument("--split", choices=["random", "scaffold", "target"],
                        default="random")
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        tune_all(args.dataset_version)
    elif args.model:
        tune_model(args.model, args.split, args.dataset_version)
    else:
        parser.error("Provide --model MODEL or --all")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
