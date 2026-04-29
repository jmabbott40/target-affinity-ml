"""Unified training loop for all baseline models.

Config-driven training pipeline:
    1. Load dataset and split indices
    2. Load cached features
    3. Build feature matrices aligned with split indices
    4. Instantiate model from config
    5. Train on training set
    6. Evaluate on validation and test sets
    7. Save model, predictions, and metrics

The key alignment challenge: features are indexed by unique SMILES
(~206K rows), but activities have ~353K rows (same compound can appear
against multiple targets). Split indices index the activity DataFrame.
We use a smiles_to_row lookup dict to map activity records to feature
matrix rows.

Usage:
    # Single model + split
    python -m target_affinity_ml.training.trainer \\
        --config configs/rf_baseline.yaml --split scaffold

    # All 12 experiments (4 models × 3 splits)
    python -m target_affinity_ml.training.trainer --all
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from target_affinity_ml.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from target_affinity_ml.features import load_morgan_fingerprints, load_rdkit_descriptors

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "random_forest": "target_affinity_ml.models.rf_model.RandomForestModel",
    "xgboost": "target_affinity_ml.models.xgb_model.XGBoostModel",
    "elasticnet": "target_affinity_ml.models.elasticnet_model.ElasticNetModel",
    "mlp": "target_affinity_ml.models.mlp_model.MLPModel",
}

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

ALL_CONFIGS = [
    Path("configs/rf_baseline.yaml"),
    Path("configs/xgb_baseline.yaml"),
    Path("configs/elasticnet_baseline.yaml"),
    Path("configs/mlp_baseline.yaml"),
]

ALL_SPLITS = ["random", "scaffold", "target"]


def load_model_config(config_path: Path) -> dict:
    """Load model configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model_class(model_name: str):
    """Resolve model class from registry via dynamic import.

    Parameters
    ----------
    model_name : str
        Model name (key in MODEL_REGISTRY).

    Returns
    -------
    type
        Model class.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY)}"
        )

    module_path = MODEL_REGISTRY[model_name]
    # Split "target_affinity_ml.models.rf_model.RandomForestModel"
    # into module="target_affinity_ml.models.rf_model", class="RandomForestModel"
    parts = module_path.rsplit(".", 1)
    module_name, class_name = parts[0], parts[1]

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_feature_matrix(
    df: pd.DataFrame,
    indices: list[int],
    smiles_to_row: dict[str, int],
    feature_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map activity split indices to feature matrix rows.

    Features are indexed by unique SMILES (~206K rows), while
    activities have ~353K rows. This function performs the alignment.

    Parameters
    ----------
    df : pd.DataFrame
        Curated activity DataFrame.
    indices : list[int]
        Row indices into df for this split.
    smiles_to_row : dict[str, int]
        Maps SMILES string → row index in feature_matrix.
    feature_matrix : np.ndarray
        Full feature matrix (n_compounds, n_features).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (X, y_pactivity, y_is_active) aligned arrays.
    """
    subset = df.iloc[indices]
    smiles = subset["std_smiles"].values

    # Map each activity record to its feature row
    feature_rows = np.array([smiles_to_row[s] for s in smiles])

    X = feature_matrix[feature_rows]
    y = subset["pactivity"].values.astype(np.float64)
    y_active = subset["is_active"].values.astype(np.float64)

    return X, y, y_active


def train_and_evaluate(
    config_path: Path,
    split_strategy: str = "random",
    dataset_version: str = "v1",
    training_seed: int | None = None,
    data_dir_override: Path | None = None,
    output_suffix: str = "",
) -> dict:
    """Run the full train/evaluate pipeline for one model + split combination.

    Parameters
    ----------
    config_path : Path
        Path to model config YAML.
    split_strategy : str
        One of 'random', 'scaffold', 'target'.
    dataset_version : str
        Dataset version (subdirectory of data/processed/).
    training_seed : int, optional
        If provided, overrides the random_state in model config.
        Used for multi-seed robustness experiments.
    data_dir_override : Path, optional
        Override the data directory (for subset experiments).
    output_suffix : str
        Suffix for output directories (e.g., "_seed42", "_esm92").

    Returns
    -------
    dict
        Combined evaluation metrics on the test set.
    """
    config = load_model_config(config_path)
    model_name = config["model"]["name"]
    feature_type = config["features"]["type"]

    # Override random_state if training_seed is provided
    if training_seed is not None:
        config.setdefault("hyperparameters", {})["random_state"] = training_seed

    logger.info("=" * 70)
    logger.info(
        "EXPERIMENT: model=%s, split=%s, features=%s, seed=%s",
        model_name, split_strategy, feature_type, training_seed,
    )
    logger.info("=" * 70)

    # --- 1. Load curated dataset ---
    if data_dir_override is not None:
        data_dir = data_dir_override
    else:
        data_dir = DATA_DIR / dataset_version
    curated_path = data_dir / "curated_activities.parquet"
    logger.info("Loading curated dataset from %s", curated_path)
    df = pd.read_parquet(curated_path)
    logger.info("  Dataset: %d records", len(df))

    # --- 2. Load split indices ---
    split_path = data_dir / "splits" / f"{split_strategy}_split.json"
    if not split_path.exists():
        # Fall back to main dataset splits
        split_path = DATA_DIR / dataset_version / "splits" / f"{split_strategy}_split.json"
    logger.info("Loading %s split from %s", split_strategy, split_path)
    with open(split_path) as f:
        split_indices = json.load(f)

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]
    logger.info(
        "  Split sizes: train=%d, val=%d, test=%d",
        len(train_idx), len(val_idx), len(test_idx),
    )

    # --- 3. Load feature matrix ---
    if feature_type == "morgan_fingerprint":
        logger.info("Loading Morgan fingerprints...")
        feature_matrix, smiles_list = load_morgan_fingerprints(dataset_version)
    elif feature_type == "rdkit_descriptors":
        logger.info("Loading RDKit descriptors...")
        feature_matrix, _desc_names, smiles_list = load_rdkit_descriptors(
            dataset_version,
        )
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    logger.info("  Feature matrix shape: %s", feature_matrix.shape)

    # --- 4. Build SMILES-to-row lookup ---
    smiles_to_row = {smi: i for i, smi in enumerate(smiles_list)}
    logger.info("  SMILES lookup: %d unique compounds", len(smiles_to_row))

    # --- 5. Build aligned feature matrices ---
    logger.info("Building train/val/test feature matrices...")
    X_train, y_train, y_train_active = build_feature_matrix(
        df, train_idx, smiles_to_row, feature_matrix,
    )
    X_val, y_val, y_val_active = build_feature_matrix(
        df, val_idx, smiles_to_row, feature_matrix,
    )
    X_test, y_test, y_test_active = build_feature_matrix(
        df, test_idx, smiles_to_row, feature_matrix,
    )
    logger.info(
        "  Train: X=%s, Val: X=%s, Test: X=%s",
        X_train.shape, X_val.shape, X_test.shape,
    )

    # --- 6. Instantiate model ---
    model_cls = get_model_class(model_name)
    hyperparams = dict(config.get("hyperparameters", {}))

    # Handle uncertainty-specific params
    uncertainty_config = config.get("uncertainty", {})

    # Extract constructor-level kwargs that aren't sklearn hyperparams
    extra_kwargs = {}
    if model_name == "elasticnet":
        n_bootstrap = uncertainty_config.get("n_bootstrap", 100)
        extra_kwargs["n_bootstrap"] = n_bootstrap
    if model_name == "mlp":
        n_ensemble = uncertainty_config.get("n_ensemble", 5)
        extra_kwargs["n_ensemble"] = n_ensemble

    model = model_cls(**extra_kwargs, **hyperparams)
    logger.info("  Model: %s", model_cls.__name__)

    # --- 7. Train ---
    logger.info("Training model...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    logger.info("  Training completed in %.1f seconds", train_time)

    # --- 8. Predict on val + test ---
    logger.info("Generating predictions...")
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # --- 9. Compute metrics ---
    logger.info("Computing metrics...")
    val_reg_metrics = compute_regression_metrics(y_val, y_val_pred)
    test_reg_metrics = compute_regression_metrics(y_test, y_test_pred)

    val_cls_metrics = compute_classification_metrics(y_val_active, y_val_pred)
    test_cls_metrics = compute_classification_metrics(y_test_active, y_test_pred)

    logger.info("  Validation regression:  %s", _format_metrics(val_reg_metrics))
    logger.info("  Test regression:        %s", _format_metrics(test_reg_metrics))
    logger.info("  Test classification:    %s", _format_metrics(test_cls_metrics))

    # --- 10. Predict with uncertainty on test ---
    logger.info("Generating uncertainty estimates...")
    y_test_mean, y_test_std = model.predict_with_uncertainty(X_test)

    # --- 11. Save everything ---
    # Save model
    model_subdir = f"{model_name}{output_suffix}"
    model_dir = RESULTS_DIR / "models" / model_subdir / split_strategy
    model.save(model_dir)

    # Save predictions
    pred_dir = RESULTS_DIR / "predictions"
    if output_suffix:
        pred_dir = RESULTS_DIR / f"predictions{output_suffix}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        pred_dir / f"{model_name}_{split_strategy}.npz",
        y_test_true=y_test,
        y_test_pred=y_test_pred,
        y_test_active=y_test_active,
        y_test_mean=y_test_mean,
        y_test_std=y_test_std,
        y_val_true=y_val,
        y_val_pred=y_val_pred,
    )
    logger.info("  Saved predictions to %s", pred_dir)

    # Combine all metrics
    all_metrics = {
        "model": model_name,
        "split": split_strategy,
        "features": feature_type,
        "training_seed": training_seed,
        "output_suffix": output_suffix,
        "train_time_seconds": round(train_time, 1),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        **{f"val_{k}": v for k, v in val_reg_metrics.items()},
        **{f"test_{k}": v for k, v in test_reg_metrics.items()},
        **{f"test_{k}": v for k, v in test_cls_metrics.items()},
    }

    # Save metrics JSON
    metrics_dir = RESULTS_DIR / "tables"
    if output_suffix:
        metrics_dir = RESULTS_DIR / f"tables{output_suffix}"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{model_name}_{split_strategy}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=_json_default)
    logger.info("  Saved metrics to %s", metrics_path)

    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPLETE: %s / %s", model_name, split_strategy)
    logger.info(
        "  Test RMSE=%.3f, R²=%.3f, Pearson=%.3f",
        test_reg_metrics["rmse"], test_reg_metrics["r2"],
        test_reg_metrics["pearson_r"],
    )
    logger.info("=" * 70)

    return all_metrics


def run_all_experiments(
    dataset_version: str = "v1",
    training_seed: int | None = None,
    output_suffix: str = "",
) -> pd.DataFrame:
    """Run all model × split combinations and save summary.

    Parameters
    ----------
    dataset_version : str
        Dataset version.
    training_seed : int, optional
        Training seed for reproducibility.
    output_suffix : str
        Suffix for output directories.

    Returns
    -------
    pd.DataFrame
        Summary table of all experiment results.
    """
    results = []
    total = len(ALL_CONFIGS) * len(ALL_SPLITS)
    i = 0

    for config_path in ALL_CONFIGS:
        if not config_path.exists():
            logger.warning("Config not found, skipping: %s", config_path)
            continue
        for split in ALL_SPLITS:
            i += 1
            logger.info(
                "\n%s\n  RUNNING EXPERIMENT %d/%d: %s × %s (seed=%s)\n%s",
                "#" * 70, i, total, config_path.stem, split,
                training_seed, "#" * 70,
            )
            try:
                metrics = train_and_evaluate(
                    config_path, split, dataset_version,
                    training_seed=training_seed,
                    output_suffix=output_suffix,
                )
                results.append(metrics)
            except Exception:
                logger.exception(
                    "FAILED: %s × %s", config_path.stem, split,
                )

    # Save summary CSV
    if results:
        summary_df = pd.DataFrame(results)
        summary_dir = RESULTS_DIR / "tables"
        if output_suffix:
            summary_dir = RESULTS_DIR / f"tables{output_suffix}"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "phase4_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved summary to %s", summary_path)

        # Print summary table
        logger.info("\n%s", "=" * 70)
        logger.info("PHASE 4 SUMMARY — ALL EXPERIMENTS")
        logger.info("=" * 70)
        display_cols = [
            "model", "split", "train_time_seconds",
            "test_rmse", "test_r2", "test_pearson_r",
            "test_spearman_rho", "test_auroc",
        ]
        available_cols = [c for c in display_cols if c in summary_df.columns]
        logger.info("\n%s", summary_df[available_cols].to_string(index=False))

        return summary_df

    logger.warning("No experiments completed successfully!")
    return pd.DataFrame()


def _format_metrics(metrics: dict) -> str:
    """Format metrics dict as a concise string."""
    return ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())


def _json_default(obj):
    """JSON serializer for types not handled by default."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main() -> None:
    """CLI entry point for model training."""
    parser = argparse.ArgumentParser(
        description="Train baseline models for kinase affinity prediction",
    )
    parser.add_argument(
        "--config", type=Path,
        help="Model config YAML (for single experiment)",
    )
    parser.add_argument(
        "--split",
        choices=["random", "scaffold", "target"],
        default="random",
        help="Split strategy (default: random)",
    )
    parser.add_argument(
        "--dataset-version", default="v1",
        help="Dataset version (default: v1)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all 12 experiments (4 models × 3 splits)",
    )
    parser.add_argument(
        "--training-seed", type=int, default=None,
        help="Override random_state in model config for reproducibility",
    )
    parser.add_argument(
        "--output-suffix", default="",
        help="Suffix for output directories (e.g., '_seed42', '_esm92')",
    )
    parser.add_argument(
        "--data-dir-override", type=Path, default=None,
        help="Override data directory (e.g., data/processed/v1/subsets/esm92)",
    )
    args = parser.parse_args()

    if args.all:
        run_all_experiments(
            args.dataset_version,
            training_seed=args.training_seed,
            output_suffix=args.output_suffix,
        )
    elif args.config:
        train_and_evaluate(
            args.config, args.split, args.dataset_version,
            training_seed=args.training_seed,
            data_dir_override=args.data_dir_override,
            output_suffix=args.output_suffix,
        )
    else:
        parser.error("Provide --config CONFIG or --all")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
