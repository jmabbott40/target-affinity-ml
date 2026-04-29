"""Dataset curation: activity normalization, duplicate handling, quality filters.

This module takes raw ChEMBL data, applies molecule standardization, and produces
a clean, analysis-ready dataset. All thresholds are config-driven.

Key concepts explained
----------------------

**pActivity (pIC50, pKi, pKd)**:
    Raw activity values are reported in nanomolar (nM), but we convert to
    pActivity = -log10(value_in_molar). This transformation is standard because:

    1. Activity values span many orders of magnitude (0.01 nM to 100,000 nM).
       pActivity compresses this to a manageable range (~4 to ~11).
    2. Higher pActivity = more potent compound (easier to interpret).
    3. Errors in pActivity space are approximately normally distributed,
       which is better for regression models.

    Conversion: pActivity = -log10(nM * 1e-9) = 9 - log10(nM)
    Examples: 1 nM -> pActivity 9.0, 100 nM -> 7.0, 1 uM -> 6.0, 10 uM -> 5.0

**Duplicate handling**:
    The same compound is often tested multiple times against the same target,
    in different labs, assays, or conditions. These replicates can vary
    substantially (inter-lab variability is a known issue in bioactivity data).

    Strategy: group by (standardized_smiles, target, activity_type) and take
    the median pActivity. Also compute std to flag noisy measurements.

**Noise flagging**:
    If a compound has >= 3 measurements and their pActivity std > 1.0 log unit
    (i.e., a 10-fold disagreement), we flag it as "noisy". These compounds
    are kept in the dataset but tracked separately for error analysis — they
    represent genuine measurement uncertainty that no model can resolve.

Usage:
    python -m target_affinity_ml.data.curate
    python -m target_affinity_ml.data.curate --config configs/dataset_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from target_affinity_ml.data.splits import create_splits, save_splits
from target_affinity_ml.data.standardize import standardize_dataframe

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def convert_to_pactivity(df: pd.DataFrame) -> pd.DataFrame:
    """Convert standard_value (nM) to pActivity (-log10 M).

    pActivity = -log10(standard_value * 1e-9) = 9 - log10(standard_value)

    Removes rows where standard_value is missing, zero, or negative
    (log is undefined for non-positive values).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'standard_value' column in nM units.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'pactivity' column. Rows with invalid
        standard_value are removed.
    """
    df = df.copy()

    # ChEMBL API returns standard_value as string — coerce to numeric
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")

    # Remove invalid values (can't take log of zero or negative)
    mask = df["standard_value"].notna() & (df["standard_value"] > 0)
    n_removed = (~mask).sum()
    if n_removed > 0:
        logger.info("Removed %d rows with invalid standard_value (<= 0 or NaN)", n_removed)
    df = df[mask].copy()

    df["pactivity"] = 9.0 - np.log10(df["standard_value"].astype(float))
    return df


def handle_duplicates(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    aggregation: str = "median",
    noise_std_threshold: float = 1.0,
    min_measurements: int = 3,
) -> pd.DataFrame:
    """Collapse duplicate measurements for the same compound-target pair.

    For each group of duplicate measurements, computes:
    - Aggregated pActivity (median by default)
    - Number of original measurements
    - Standard deviation of measurements
    - Noise flag (high variance = unreliable measurement)

    Parameters
    ----------
    df : pd.DataFrame
        Activity data with potential duplicates.
    group_cols : list[str], optional
        Columns defining a unique measurement. Default:
        ['std_smiles', 'target_chembl_id', 'standard_type'].
    aggregation : str
        Aggregation method ('median' or 'mean').
    noise_std_threshold : float
        If std > threshold and n >= min_measurements, flag as noisy.
    min_measurements : int
        Minimum measurements to assess noise.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame with added columns:
        'n_measurements', 'pactivity_std', 'is_noisy'.
    """
    if group_cols is None:
        group_cols = ["std_smiles", "target_chembl_id", "standard_type"]

    n_before = len(df)
    agg_func = aggregation if aggregation in ("median", "mean") else "median"

    # Group and aggregate
    grouped = df.groupby(group_cols, as_index=False).agg(
        pactivity=(
            "pactivity",
            agg_func,
        ),
        n_measurements=("pactivity", "size"),
        pactivity_std=("pactivity", "std"),
        # Keep first occurrence of metadata columns
        **{
            col: (col, "first")
            for col in df.columns
            if col not in group_cols + ["pactivity", "activity_id"]
        },
    )

    # Fill NaN std (single measurements have undefined std)
    grouped["pactivity_std"] = grouped["pactivity_std"].fillna(0.0)

    # Flag noisy measurements
    grouped["is_noisy"] = (
        (grouped["n_measurements"] >= min_measurements)
        & (grouped["pactivity_std"] > noise_std_threshold)
    )

    n_after = len(grouped)
    n_noisy = grouped["is_noisy"].sum()
    logger.info(
        "Duplicate handling: %d -> %d unique (compound, target, type) groups",
        n_before, n_after,
    )
    logger.info(
        "  %d groups flagged as noisy (std > %.1f, n >= %d)",
        n_noisy, noise_std_threshold, min_measurements,
    )

    return grouped


def apply_quality_filters(
    df: pd.DataFrame,
    pactivity_min: float = 3.0,
    pactivity_max: float = 12.0,
) -> pd.DataFrame:
    """Remove measurements outside plausible pActivity range.

    Why these bounds?
    - pActivity < 3.0 means IC50 > 1 mM — essentially inactive,
      likely not a real binder.
    - pActivity > 12.0 means IC50 < 1 pM — implausibly potent,
      likely a data entry error or unit conversion mistake.

    Parameters
    ----------
    df : pd.DataFrame
        Curated activity data.
    pactivity_min : float
        Minimum pActivity.
    pactivity_max : float
        Maximum pActivity.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    n_before = len(df)
    mask = (df["pactivity"] >= pactivity_min) & (df["pactivity"] <= pactivity_max)
    df_filtered = df[mask].copy()
    n_removed = n_before - len(df_filtered)

    if n_removed > 0:
        logger.info(
            "Quality filter: removed %d records outside pActivity [%.1f, %.1f]",
            n_removed, pactivity_min, pactivity_max,
        )

    return df_filtered


def add_classification_labels(
    df: pd.DataFrame,
    threshold: float = 6.0,
) -> pd.DataFrame:
    """Add binary active/inactive labels based on pActivity threshold.

    The default threshold of 6.0 corresponds to:
        - IC50 <= 1 uM = "active"
        - IC50 > 1 uM = "inactive"

    This is a common convention in drug discovery. The 1 uM cutoff represents
    a reasonable potency for a hit compound, though medicinal chemists typically
    aim for sub-100 nM (pActivity >= 7) for drug candidates.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'pactivity' column.
    threshold : float
        pActivity threshold. >= threshold is 'active'.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'is_active' boolean column.
    """
    df = df.copy()
    df["is_active"] = df["pactivity"] >= threshold

    n_active = df["is_active"].sum()
    n_total = len(df)
    logger.info(
        "Classification labels: %d active (%.1f%%), %d inactive (%.1f%%)",
        n_active, n_active / n_total * 100,
        n_total - n_active, (n_total - n_active) / n_total * 100,
    )

    return df


def main() -> None:
    """Run the full curation pipeline: standardize -> curate -> split."""
    parser = argparse.ArgumentParser(description="Curate kinase bioactivity dataset")
    parser.add_argument(
        "--config", type=Path, default="configs/dataset_v1.yaml",
        help="Dataset config YAML",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    version = config["version"]
    output_dir = PROCESSED_DIR / version
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load raw data ---
    logger.info("=== Step 1: Loading raw data ===")
    raw_path = RAW_DATA_DIR / "chembl_kinase_activities.parquet"
    df = pd.read_parquet(raw_path)
    logger.info("Loaded %d raw activity records", len(df))

    # Also load target metadata for enriching the dataset
    targets_path = RAW_DATA_DIR / "chembl_kinase_targets.parquet"
    if targets_path.exists():
        targets_df = pd.read_parquet(targets_path)
        # Merge kinase group and gene symbol into activities
        target_info = targets_df[
            ["target_chembl_id", "pref_name", "kinase_group", "gene_symbol"]
        ].drop_duplicates(subset=["target_chembl_id"])
        df = df.merge(target_info, on="target_chembl_id", how="left")
        logger.info("Merged kinase metadata from targets file")

    # --- Step 2: Standardize molecules ---
    logger.info("=== Step 2: Standardizing molecules ===")
    df, std_stats = standardize_dataframe(df, config=config)

    # --- Step 3: Convert to pActivity ---
    logger.info("=== Step 3: Converting to pActivity ===")
    df = convert_to_pactivity(df)

    # --- Step 4: Handle duplicates ---
    logger.info("=== Step 4: Handling duplicates ===")
    dup_config = config["duplicates"]
    df = handle_duplicates(
        df,
        aggregation=dup_config["aggregation"],
        noise_std_threshold=dup_config["noise_std_threshold"],
        min_measurements=dup_config["min_measurements_for_noise_flag"],
    )

    # --- Step 5: Quality filters ---
    logger.info("=== Step 5: Applying quality filters ===")
    qual_config = config["quality"]
    df = apply_quality_filters(
        df,
        pactivity_min=qual_config["pactivity_min"],
        pactivity_max=qual_config["pactivity_max"],
    )

    # --- Step 6: Classification labels ---
    logger.info("=== Step 6: Adding classification labels ===")
    cls_config = config["classification"]
    df = add_classification_labels(df, threshold=cls_config["active_pactivity_threshold"])

    # --- Step 7: Save curated dataset ---
    logger.info("=== Step 7: Saving curated dataset ===")
    # Reset index for clean output
    df = df.reset_index(drop=True)
    curated_path = output_dir / "curated_activities.parquet"
    df.to_parquet(curated_path, index=False)
    logger.info("Saved %d curated records to %s", len(df), curated_path)

    # --- Step 8: Create splits ---
    logger.info("=== Step 8: Creating train/val/test splits ===")
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    for strategy in ["random", "scaffold", "target"]:
        logger.info("Creating %s split...", strategy)
        try:
            splits = create_splits(df, strategy=strategy, config=config)
            save_splits(splits, splits_dir / f"{strategy}_split.json")
            logger.info(
                "  %s split: train=%d, val=%d, test=%d",
                strategy, len(splits["train"]), len(splits["val"]), len(splits["test"]),
            )
        except Exception as e:
            logger.error("Failed to create %s split: %s", strategy, e)

    # --- Step 9: Save curation statistics ---
    stats = {
        "standardization": std_stats,
        "n_curated_records": len(df),
        "n_unique_compounds": int(df["std_smiles"].nunique()),
        "n_unique_targets": int(df["target_chembl_id"].nunique()),
        "n_noisy_records": int(df["is_noisy"].sum()),
        "n_active": int(df["is_active"].sum()),
        "n_inactive": int((~df["is_active"]).sum()),
        "activity_types": df["standard_type"].value_counts().to_dict(),
    }
    stats_path = output_dir / "curation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved curation statistics to %s", stats_path)

    # --- Summary ---
    logger.info("=== Curation complete (dataset %s) ===", version)
    logger.info("  Records: %d", stats["n_curated_records"])
    logger.info("  Compounds: %d", stats["n_unique_compounds"])
    logger.info("  Targets: %d", stats["n_unique_targets"])
    logger.info("  Noisy: %d", stats["n_noisy_records"])
    logger.info("  Active: %d (%.1f%%)", stats["n_active"],
                stats["n_active"] / stats["n_curated_records"] * 100)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
