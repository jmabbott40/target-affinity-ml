"""Phase 5 analysis runner: uncertainty calibration, error analysis, and plots.

Loads prediction .npz files from Phase 4 and runs:
    1. Uncertainty calibration curves + miscalibration area
    2. Error-uncertainty correlation analysis
    3. Selective prediction curves
    4. Per-target metric breakdown
    5. Noise impact analysis
    6. Worst prediction identification
    7. Visualization (scatter, calibration, selective prediction plots)

Usage:
    # Run all analyses for all experiments
    python -m target_affinity_ml.evaluation.run_phase5

    # Run for a specific model + split
    python -m target_affinity_ml.evaluation.run_phase5 --model random_forest --split random
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from target_affinity_ml.evaluation.analysis import (
    find_worst_predictions,
    noise_impact_analysis,
    per_target_metrics,
)
from target_affinity_ml.evaluation.uncertainty import (
    calibration_curve,
    error_uncertainty_correlation,
    miscalibration_area,
    selective_prediction_curve,
)
from target_affinity_ml.visualization.plots import (
    plot_calibration_diagram,
    plot_multi_model_calibration,
    plot_multi_model_selective,
    plot_per_target_histogram,
    plot_performance_degradation,
    plot_predicted_vs_actual,
    plot_selective_prediction,
    plot_split_comparison,
    plot_uncertainty_correlation,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
PRED_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

BASELINE_MODELS = ["random_forest", "xgboost", "elasticnet", "mlp"]
DEEP_MODELS = ["esm_fp_mlp", "gnn", "fusion"]
ALL_MODELS = BASELINE_MODELS + DEEP_MODELS
ALL_SPLITS = ["random", "scaffold", "target"]


def load_predictions(model: str, split: str) -> dict[str, np.ndarray]:
    """Load saved prediction arrays from Phase 4."""
    path = PRED_DIR / f"{model}_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_test_metadata(
    split: str,
    dataset_version: str = "v1",
) -> pd.DataFrame:
    """Load the test set metadata from curated data + split indices."""
    data_dir = DATA_DIR / dataset_version
    df = pd.read_parquet(data_dir / "curated_activities.parquet")

    split_path = data_dir / "splits" / f"{split}_split.json"
    with open(split_path) as f:
        split_indices = json.load(f)

    return df.iloc[split_indices["test"]].reset_index(drop=True)


def analyze_single_experiment(
    model: str,
    split: str,
    dataset_version: str = "v1",
) -> dict:
    """Run all Phase 5 analyses for one model × split combination."""
    logger.info("=" * 60)
    logger.info("PHASE 5 ANALYSIS: %s × %s", model, split)
    logger.info("=" * 60)

    # Load predictions
    preds = load_predictions(model, split)
    y_true = preds["y_test_true"]
    y_pred = preds["y_test_pred"]
    y_active = preds["y_test_active"]
    y_mean = preds["y_test_mean"]
    y_std = preds["y_test_std"]

    # Load test set metadata
    test_df = load_test_metadata(split, dataset_version)
    results = {"model": model, "split": split}

    # --- 1. Uncertainty calibration ---
    logger.info("  Computing calibration curve...")
    has_uncertainty = np.std(y_std) > 1e-10
    if has_uncertainty:
        expected, observed = calibration_curve(y_true, y_mean, y_std)
        miscal = miscalibration_area(expected, observed)
        results["miscalibration_area"] = miscal
        results["calibration_expected"] = expected.tolist()
        results["calibration_observed"] = observed.tolist()
        logger.info("    Miscalibration area: %.4f", miscal)

        # Plot
        fig = plot_calibration_diagram(
            expected, observed,
            title=f"Calibration: {model} / {split}",
            miscal_area=miscal,
            save_path=FIGURES_DIR / f"calibration_{model}_{split}.png",
        )
        plt.close(fig)
    else:
        logger.info("    Skipping calibration — constant uncertainty (likely ElasticNet collapse)")
        results["miscalibration_area"] = float("nan")

    # --- 2. Error-uncertainty correlation ---
    logger.info("  Computing error-uncertainty correlation...")
    if has_uncertainty:
        corr = error_uncertainty_correlation(y_true, y_pred, y_std)
        results.update({f"uq_{k}": v for k, v in corr.items()})
        logger.info("    Pearson r = %.3f, Spearman rho = %.3f",
                     corr["pearson_r"], corr["spearman_rho"])

        # Plot
        abs_error = np.abs(y_true - y_pred)
        fig = plot_uncertainty_correlation(
            abs_error, y_std,
            title=f"Error vs Uncertainty: {model} / {split}",
            save_path=FIGURES_DIR / f"error_vs_uncertainty_{model}_{split}.png",
        )
        plt.close(fig)
    else:
        results["uq_pearson_r"] = float("nan")
        results["uq_spearman_rho"] = float("nan")

    # --- 3. Selective prediction ---
    logger.info("  Computing selective prediction curve...")
    if has_uncertainty:
        retention, rmse_curve = selective_prediction_curve(y_true, y_pred, y_std)
        results["selective_rmse_at_50pct"] = float(rmse_curve[9])  # ~50% retention
        results["selective_rmse_at_100pct"] = float(rmse_curve[-1])
        results["selective_improvement_50pct"] = float(
            1.0 - rmse_curve[9] / rmse_curve[-1]
        )
        logger.info("    RMSE at 100%%: %.3f, at 50%%: %.3f (%.1f%% improvement)",
                     rmse_curve[-1], rmse_curve[9],
                     100 * (1.0 - rmse_curve[9] / rmse_curve[-1]))

        fig = plot_selective_prediction(
            retention, rmse_curve,
            title=f"Selective Prediction: {model} / {split}",
            save_path=FIGURES_DIR / f"selective_{model}_{split}.png",
        )
        plt.close(fig)
    else:
        results["selective_rmse_at_50pct"] = float("nan")
        results["selective_improvement_50pct"] = float("nan")

    # --- 4. Predicted vs actual scatter ---
    logger.info("  Generating scatter plot...")
    from target_affinity_ml.evaluation.metrics import compute_regression_metrics
    reg_metrics = compute_regression_metrics(y_true, y_pred)
    fig = plot_predicted_vs_actual(
        y_true, y_pred,
        title=f"Predicted vs Actual: {model} / {split}",
        rmse=reg_metrics["rmse"],
        r2=reg_metrics["r2"],
        save_path=FIGURES_DIR / f"scatter_{model}_{split}.png",
    )
    plt.close(fig)

    # --- 5. Per-target metrics ---
    logger.info("  Computing per-target metrics...")
    if "target_chembl_id" in test_df.columns:
        target_ids = test_df["target_chembl_id"].values
        target_df = per_target_metrics(y_true, y_pred, target_ids, min_samples=10)
        if len(target_df) > 0:
            results["n_targets_evaluated"] = len(target_df)
            results["per_target_rmse_median"] = float(target_df["rmse"].median())
            results["per_target_rmse_mean"] = float(target_df["rmse"].mean())
            results["per_target_rmse_std"] = float(target_df["rmse"].std())
            results["per_target_r2_median"] = float(target_df["r2"].median())
            logger.info("    %d targets evaluated (min 10 compounds each)", len(target_df))
            logger.info("    Per-target RMSE: median=%.3f, mean=%.3f, std=%.3f",
                         target_df["rmse"].median(), target_df["rmse"].mean(),
                         target_df["rmse"].std())

            # Save per-target breakdown
            target_path = TABLES_DIR / f"per_target_{model}_{split}.csv"
            target_df.to_csv(target_path, index=False)

            # Plot histogram
            fig = plot_per_target_histogram(
                target_df, metric="rmse",
                title=f"Per-Target RMSE: {model} / {split}",
                save_path=FIGURES_DIR / f"per_target_rmse_{model}_{split}.png",
            )
            plt.close(fig)

    # --- 6. Noise impact ---
    logger.info("  Computing noise impact...")
    if "is_noisy" in test_df.columns:
        is_noisy = test_df["is_noisy"].values
        noise_result = noise_impact_analysis(y_true, y_pred, is_noisy)
        results["noise_n_clean"] = noise_result["n_clean"]
        results["noise_n_noisy"] = noise_result["n_noisy"]
        if "clean" in noise_result and "noisy" in noise_result and noise_result["noisy"]:
            results["noise_clean_rmse"] = noise_result["clean"]["rmse"]
            results["noise_noisy_rmse"] = noise_result["noisy"]["rmse"]
            results["noise_rmse_delta"] = noise_result["delta"]["rmse"]
            logger.info("    Clean RMSE: %.3f (n=%d), Noisy RMSE: %.3f (n=%d), Delta: %.3f",
                         noise_result["clean"]["rmse"], noise_result["n_clean"],
                         noise_result["noisy"]["rmse"], noise_result["n_noisy"],
                         noise_result["delta"]["rmse"])

    # --- 7. Worst predictions ---
    logger.info("  Identifying worst predictions...")
    worst = find_worst_predictions(y_true, y_pred, test_df, top_n=100)
    worst_path = TABLES_DIR / f"worst_predictions_{model}_{split}.csv"
    # Save subset of columns for readability
    save_cols = [c for c in ["std_smiles", "target_chembl_id", "pref_name",
                              "gene_symbol", "kinase_group", "standard_type",
                              "y_true", "y_pred", "abs_error", "signed_error",
                              "is_noisy"] if c in worst.columns]
    worst[save_cols].to_csv(worst_path, index=False)
    logger.info("    Worst prediction error: %.3f pActivity units", worst["abs_error"].iloc[0])

    # Save full results JSON
    results_path = TABLES_DIR / f"phase5_{model}_{split}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    logger.info("  Saved Phase 5 results to %s", results_path)

    return results


def run_all_analyses(dataset_version: str = "v1") -> pd.DataFrame:
    """Run Phase 5 analysis for all available experiments."""
    all_results = []

    for model in ALL_MODELS:
        for split in ALL_SPLITS:
            pred_path = PRED_DIR / f"{model}_{split}.npz"
            if not pred_path.exists():
                logger.warning("Skipping %s/%s — predictions not found", model, split)
                continue
            try:
                result = analyze_single_experiment(model, split, dataset_version)
                all_results.append(result)
            except Exception:
                logger.exception("FAILED: %s × %s", model, split)

    if all_results:
        summary = pd.DataFrame(all_results)
        summary_path = TABLES_DIR / "phase5_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("Saved Phase 5 summary to %s", summary_path)

        # Generate cross-model comparison plots
        _generate_comparison_plots(dataset_version)

        # Generate multi-model overlay plots (calibration, selective, degradation)
        _generate_multi_model_plots(summary)

        return summary

    logger.warning("No experiments analyzed!")
    return pd.DataFrame()


def _generate_comparison_plots(dataset_version: str = "v1") -> None:
    """Generate plots comparing all models across splits.

    Merges Phase 4 and Phase 7 summaries for unified comparison heatmaps.
    """
    dfs = []
    for path_name in ["phase4_summary.csv", "phase7_summary.csv"]:
        path = TABLES_DIR / path_name
        if path.exists():
            dfs.append(pd.read_csv(path))

    if not dfs:
        return

    summary_df = pd.concat(dfs, ignore_index=True)
    # Drop duplicates in case of re-runs (keep latest)
    summary_df = summary_df.drop_duplicates(
        subset=["model", "split"], keep="last"
    )

    for metric in ["test_rmse", "test_r2", "test_auroc", "test_pearson_r",
                    "test_spearman_rho", "test_auprc"]:
        if metric in summary_df.columns:
            fig = plot_split_comparison(
                summary_df, metric=metric,
                save_path=FIGURES_DIR / f"comparison_{metric}.png",
            )
            plt.close(fig)
            logger.info("Saved comparison heatmap: %s", metric)


def _generate_multi_model_plots(phase5_summary: pd.DataFrame) -> None:
    """Generate multi-model overlay plots for calibration and selective prediction."""
    from target_affinity_ml.evaluation.uncertainty import (
        calibration_curve as calc_calibration,
        selective_prediction_curve as calc_selective,
    )

    for split in ALL_SPLITS:
        calibrations = {}
        selective_curves = {}

        for model in ALL_MODELS:
            pred_path = PRED_DIR / f"{model}_{split}.npz"
            if not pred_path.exists():
                continue

            preds = np.load(pred_path)
            y_true = preds["y_test_true"]
            y_pred = preds["y_test_pred"]
            y_mean = preds["y_test_mean"]
            y_std = preds["y_test_std"]

            has_uncertainty = np.std(y_std) > 1e-10
            if not has_uncertainty:
                continue

            # Calibration
            expected, observed = calc_calibration(y_true, y_mean, y_std)
            miscal = miscalibration_area(expected, observed)
            calibrations[model] = (expected, observed, miscal)

            # Selective prediction
            retention, rmse_curve = calc_selective(y_true, y_pred, y_std)
            selective_curves[model] = (retention, rmse_curve)

        if calibrations:
            fig = plot_multi_model_calibration(
                calibrations,
                title=f"Calibration Comparison: {split.capitalize()} Split",
                save_path=FIGURES_DIR / f"calibration_comparison_{split}.png",
            )
            plt.close(fig)
            logger.info("Saved multi-model calibration: %s", split)

        if selective_curves:
            fig = plot_multi_model_selective(
                selective_curves,
                title=f"Selective Prediction Comparison: {split.capitalize()} Split",
                save_path=FIGURES_DIR / f"selective_comparison_{split}.png",
            )
            plt.close(fig)
            logger.info("Saved multi-model selective prediction: %s", split)

    # Performance degradation plots
    dfs = []
    for path_name in ["phase4_summary.csv", "phase7_summary.csv"]:
        path = TABLES_DIR / path_name
        if path.exists():
            dfs.append(pd.read_csv(path))
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["model", "split"], keep="last")
        for metric in ["test_rmse", "test_r2", "test_auroc"]:
            if metric in combined.columns:
                fig = plot_performance_degradation(
                    combined, metric=metric,
                    save_path=FIGURES_DIR / f"degradation_{metric}.png",
                )
                plt.close(fig)
                logger.info("Saved degradation plot: %s", metric)


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Phase 5 evaluation and uncertainty analysis",
    )
    parser.add_argument("--model", choices=ALL_MODELS,
                        help="Specific model (baseline or deep)")
    parser.add_argument("--split", choices=ALL_SPLITS, help="Specific split")
    parser.add_argument("--all-models", action="store_true",
                        help="Run analysis for all models including deep models")
    parser.add_argument("--dataset-version", default="v1")
    args = parser.parse_args()

    if args.model and args.split:
        analyze_single_experiment(args.model, args.split, args.dataset_version)
    else:
        run_all_analyses(args.dataset_version)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
