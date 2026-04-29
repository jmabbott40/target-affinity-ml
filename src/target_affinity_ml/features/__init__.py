"""Molecular feature generation: fingerprints and descriptors.

This package provides two complementary molecular representations:

1. **Morgan fingerprints** (binary, 2048-bit by default):
   Circular fingerprints encoding local chemical environments.
   Used by Random Forest, XGBoost, and MLP models.

2. **RDKit 2D descriptors** (continuous, ~150-200 features):
   Physicochemical and topological molecular properties.
   Used by ElasticNet and optionally MLP.

Features are computed once from the curated dataset and cached as
compressed numpy arrays for fast loading during model training.

Usage (CLI):
    python -m target_affinity_ml.features
    python -m target_affinity_ml.features --force  # Recompute even if cached
    python -m target_affinity_ml.features --config configs/dataset_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from target_affinity_ml.features.descriptors import compute_descriptors
from target_affinity_ml.features.fingerprints import compute_fingerprints

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


def compute_and_cache_features(
    config_path: Path = Path("configs/dataset_v1.yaml"),
    force: bool = False,
) -> dict[str, Path]:
    """Compute molecular features and cache to disk.

    Loads the curated dataset, extracts unique SMILES, computes both
    Morgan fingerprints and RDKit descriptors, and saves them as
    compressed numpy archives (.npz).

    Features are keyed by SMILES order — the saved `smiles_index.json`
    maps row indices to SMILES strings for alignment with activity data.

    Parameters
    ----------
    config_path : Path
        Path to dataset config YAML.
    force : bool
        If True, recompute even if cached files exist.

    Returns
    -------
    dict[str, Path]
        Paths to saved feature files.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    version = config["version"]
    output_dir = PROCESSED_DIR / version / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    fp_path = output_dir / "morgan_fp.npz"
    desc_path = output_dir / "rdkit_descriptors.npz"
    smiles_index_path = output_dir / "smiles_index.json"

    # --- Load curated dataset ---
    curated_path = PROCESSED_DIR / version / "curated_activities.parquet"
    if not curated_path.exists():
        raise FileNotFoundError(
            f"Curated dataset not found at {curated_path}. "
            "Run the curation pipeline first: python -m target_affinity_ml.data.curate"
        )

    logger.info("Loading curated dataset from %s", curated_path)
    df = pd.read_parquet(curated_path)

    # Get unique SMILES (features are per-compound, not per-measurement)
    unique_smiles = df["std_smiles"].unique().tolist()
    n_compounds = len(unique_smiles)
    logger.info("Found %d unique compounds for featurization", n_compounds)

    # Save SMILES index for alignment
    if not smiles_index_path.exists() or force:
        with open(smiles_index_path, "w") as f:
            json.dump(unique_smiles, f)
        logger.info("Saved SMILES index to %s", smiles_index_path)

    saved_paths = {"smiles_index": smiles_index_path}

    # --- Compute Morgan fingerprints ---
    if fp_path.exists() and not force:
        logger.info("Morgan fingerprints already cached at %s (use --force to recompute)", fp_path)
    else:
        fp_config = config.get("features", {}).get("morgan", {})
        radius = fp_config.get("radius", 2)
        n_bits = fp_config.get("n_bits", 2048)

        logger.info(
            "Computing Morgan fingerprints (radius=%d, n_bits=%d) for %d compounds...",
            radius, n_bits, n_compounds,
        )
        fp_matrix = compute_fingerprints(unique_smiles, radius=radius, n_bits=n_bits)
        np.savez_compressed(fp_path, fingerprints=fp_matrix)
        logger.info(
            "Saved Morgan fingerprints %s to %s (%.1f MB)",
            fp_matrix.shape, fp_path, fp_path.stat().st_size / 1e6,
        )

    saved_paths["morgan_fp"] = fp_path

    # --- Compute RDKit descriptors ---
    if desc_path.exists() and not force:
        logger.info("RDKit descriptors already cached at %s (use --force to recompute)", desc_path)
    else:
        desc_config = config.get("features", {}).get("descriptors", {})
        drop_threshold = desc_config.get("drop_missing_threshold", 0.05)

        logger.info(
            "Computing RDKit 2D descriptors for %d compounds...", n_compounds,
        )
        desc_matrix, desc_names = compute_descriptors(
            unique_smiles, drop_missing_threshold=drop_threshold,
        )
        np.savez_compressed(
            desc_path,
            descriptors=desc_matrix,
            descriptor_names=np.array(desc_names),
        )
        logger.info(
            "Saved RDKit descriptors %s to %s (%.1f MB)",
            desc_matrix.shape, desc_path, desc_path.stat().st_size / 1e6,
        )

    saved_paths["rdkit_descriptors"] = desc_path

    return saved_paths


def load_morgan_fingerprints(
    version: str = "v1",
) -> tuple[np.ndarray, list[str]]:
    """Load cached Morgan fingerprints.

    Parameters
    ----------
    version : str
        Dataset version.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        (fingerprint_matrix, smiles_list).
    """
    features_dir = PROCESSED_DIR / version / "features"
    fp_data = np.load(features_dir / "morgan_fp.npz")
    with open(features_dir / "smiles_index.json") as f:
        smiles_list = json.load(f)
    return fp_data["fingerprints"], smiles_list


def load_rdkit_descriptors(
    version: str = "v1",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load cached RDKit descriptors.

    Parameters
    ----------
    version : str
        Dataset version.

    Returns
    -------
    tuple[np.ndarray, list[str], list[str]]
        (descriptor_matrix, descriptor_names, smiles_list).
    """
    features_dir = PROCESSED_DIR / version / "features"
    desc_data = np.load(features_dir / "rdkit_descriptors.npz", allow_pickle=True)
    with open(features_dir / "smiles_index.json") as f:
        smiles_list = json.load(f)
    return (
        desc_data["descriptors"],
        desc_data["descriptor_names"].tolist(),
        smiles_list,
    )


def load_esm2_embeddings(
    version: str = "v1",
) -> tuple[np.ndarray, dict[str, int]]:
    """Load cached ESM-2 protein embeddings.

    Parameters
    ----------
    version : str
        Dataset version.

    Returns
    -------
    tuple[np.ndarray, dict[str, int]]
        (embedding_matrix, target_to_row) where target_to_row maps
        target_chembl_id to row index in the embedding matrix.
    """
    features_dir = PROCESSED_DIR / version / "features"
    emb_data = np.load(features_dir / "esm2_embeddings.npz")
    with open(features_dir / "target_index.json") as f:
        target_to_row = json.load(f)
    return emb_data["embeddings"], target_to_row


def main() -> None:
    """CLI entry point for feature computation."""
    parser = argparse.ArgumentParser(
        description="Compute and cache molecular features",
    )
    parser.add_argument(
        "--config", type=Path, default="configs/dataset_v1.yaml",
        help="Dataset config YAML",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if cached files exist",
    )
    args = parser.parse_args()

    compute_and_cache_features(config_path=args.config, force=args.force)
    logger.info("Feature computation complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
