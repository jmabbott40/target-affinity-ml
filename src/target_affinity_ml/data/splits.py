"""Train/validation/test splitting strategies.

Why splitting strategy matters
------------------------------
The choice of how to split data into train/val/test has a dramatic impact on
reported model performance. In drug discovery ML, this is arguably the most
important methodological decision.

**Random split** (baseline):
    Randomly assigns data points to splits. This is the simplest approach but
    often gives overly optimistic performance because structurally similar
    compounds (analogs from the same medicinal chemistry series) can end up
    in both train and test sets. The model may simply memorize local SAR
    patterns rather than learning generalizable features.

**Scaffold split** (realistic):
    Groups molecules by their Murcko scaffold — the core ring system of the
    molecule with all side chains removed. Entire scaffold groups are assigned
    to one split. This prevents the model from seeing close analogs of test
    compounds during training, simulating the real scenario where you want to
    predict activity for a new chemical series.

    Example: If imatinib (a kinase inhibitor) and its analogs all share the
    same scaffold, they all go to the same split. The model must predict
    activity for structurally novel scaffolds.

**Target split** (hardest):
    Holds out entire kinase targets. Tests whether the model can predict
    binding affinity for a kinase it has never seen. This is the hardest
    evaluation and tests whether the model has learned general protein-ligand
    interaction principles rather than target-specific patterns.

Typical performance degradation:
    Random > Scaffold > Target (random gives highest apparent performance)

Usage:
    from target_affinity_ml.data.splits import create_splits, save_splits
    splits = create_splits(df, strategy="scaffold", config=config)
    save_splits(splits, Path("data/processed/v1/splits/scaffold_split.json"))
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


def random_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    stratify_col: str | None = None,
) -> dict[str, np.ndarray]:
    """Random train/val/test split with optional stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to split.
    train_frac, val_frac, test_frac : float
        Split proportions (must sum to 1.0).
    seed : int
        Random seed for reproducibility.
    stratify_col : str, optional
        Column to stratify by (e.g., 'target_chembl_id').

    Returns
    -------
    dict[str, np.ndarray]
        {'train': indices, 'val': indices, 'test': indices}
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    splits = {
        "train": np.sort(indices[:n_train]),
        "val": np.sort(indices[n_train : n_train + n_val]),
        "test": np.sort(indices[n_train + n_val :]),
    }

    logger.info(
        "Random split: train=%d, val=%d, test=%d",
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )
    return splits


def _get_murcko_scaffold(smiles: str) -> str:
    """Extract the generic Murcko scaffold from a SMILES string.

    Murcko decomposition extracts the core ring system:
        1. Remove all side chains (keep only ring atoms and linkers)
        2. Make generic (replace all atoms with carbon, all bonds with single)

    This gives the "skeleton" of the molecule. Two molecules with the same
    scaffold share the same ring topology, even if they have different
    substituents.

    Example:
        Imatinib's scaffold: the pyridine-pyrimidine core with connecting bonds
        All imatinib analogs share this scaffold.

    Parameters
    ----------
    smiles : str
        Canonical SMILES.

    Returns
    -------
    str
        Generic Murcko scaffold SMILES, or "NO_SCAFFOLD" for acyclic molecules.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"

    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        # Make generic: replace all atoms with C, all bonds with single
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        scaffold_smiles = Chem.MolToSmiles(generic)
        # Acyclic molecules have empty scaffolds
        return scaffold_smiles if scaffold_smiles else "NO_SCAFFOLD"
    except Exception:
        return "NO_SCAFFOLD"


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "std_smiles",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Scaffold-based split using generic Murcko scaffolds.

    Algorithm:
        1. Compute Murcko scaffold for each molecule
        2. Group molecules by scaffold
        3. Sort scaffolds by group size (largest first)
        4. Greedily assign scaffolds to splits:
           - Add to train until train is full
           - Then add to val until val is full
           - Remainder goes to test

    The greedy approach ensures the largest scaffolds (most data) go to
    training, while test gets the smaller/rarer scaffolds — which is a
    harder evaluation setting.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a SMILES column.
    smiles_col : str
        Column containing standardized SMILES.
    train_frac, val_frac, test_frac : float
        Target split proportions.
    seed : int
        Random seed for shuffling scaffolds of equal size.

    Returns
    -------
    dict[str, np.ndarray]
        {'train': indices, 'val': indices, 'test': indices}
    """
    logger.info("Computing Murcko scaffolds...")
    scaffolds = df[smiles_col].apply(_get_murcko_scaffold)

    # Group indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, scaffold in enumerate(scaffolds):
        scaffold_to_indices[scaffold].append(idx)

    logger.info("Found %d unique scaffolds", len(scaffold_to_indices))

    # Sort scaffolds: largest groups first, then shuffle within same size
    rng = np.random.default_rng(seed)
    scaffold_groups = list(scaffold_to_indices.values())
    # Add small random tiebreaker to sort key for deterministic shuffling
    scaffold_groups.sort(key=lambda x: (-len(x), rng.random()))

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_indices, val_indices, test_indices = [], [], []
    train_count, val_count = 0, 0

    for group in scaffold_groups:
        if train_count < n_train:
            train_indices.extend(group)
            train_count += len(group)
        elif val_count < n_val:
            val_indices.extend(group)
            val_count += len(group)
        else:
            test_indices.extend(group)

    splits = {
        "train": np.sort(np.array(train_indices)),
        "val": np.sort(np.array(val_indices)),
        "test": np.sort(np.array(test_indices)),
    }

    logger.info(
        "Scaffold split: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(splits["train"]), len(splits["train"]) / n * 100,
        len(splits["val"]), len(splits["val"]) / n * 100,
        len(splits["test"]), len(splits["test"]) / n * 100,
    )
    return splits


def target_split(
    df: pd.DataFrame,
    target_col: str = "target_chembl_id",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Target-based split: hold out entire kinase targets.

    Randomly assigns targets (not individual compounds) to train/val/test.
    All measurements for a held-out target go to the same split.

    This tests whether the model can predict activity for kinases it has
    never seen — the hardest generalization test. Performance will typically
    be much lower than random or scaffold splits.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain target identifier column.
    target_col : str
        Column with target identifiers.
    train_frac, val_frac, test_frac : float
        Target split proportions (applied to targets, not samples).
    seed : int
        Random seed for selecting holdout targets.

    Returns
    -------
    dict[str, np.ndarray]
        {'train': indices, 'val': indices, 'test': indices}
    """
    rng = np.random.default_rng(seed)

    unique_targets = list(df[target_col].unique())
    n_targets = len(unique_targets)
    rng.shuffle(unique_targets)

    n_train_targets = int(n_targets * train_frac)
    n_val_targets = int(n_targets * val_frac)

    train_targets = set(unique_targets[:n_train_targets])
    val_targets = set(unique_targets[n_train_targets : n_train_targets + n_val_targets])
    test_targets = set(unique_targets[n_train_targets + n_val_targets :])

    train_idx = df.index[df[target_col].isin(train_targets)].to_numpy()
    val_idx = df.index[df[target_col].isin(val_targets)].to_numpy()
    test_idx = df.index[df[target_col].isin(test_targets)].to_numpy()

    splits = {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }

    len(df)
    logger.info(
        "Target split: %d train targets (%d samples), "
        "%d val targets (%d samples), %d test targets (%d samples)",
        len(train_targets), len(splits["train"]),
        len(val_targets), len(splits["val"]),
        len(test_targets), len(splits["test"]),
    )
    return splits


def create_splits(
    df: pd.DataFrame,
    strategy: str,
    config: dict,
) -> dict[str, np.ndarray]:
    """Create train/val/test split using the specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Full curated dataset.
    strategy : str
        One of 'random', 'scaffold', 'target'.
    config : dict
        Full dataset configuration (reads from 'splits' key).

    Returns
    -------
    dict[str, np.ndarray]
        Split indices.
    """
    split_config = config["splits"][strategy]
    seed = split_config.get("seed", 42)

    if strategy == "random":
        return random_split(
            df,
            train_frac=split_config.get("train", 0.8),
            val_frac=split_config.get("val", 0.1),
            test_frac=split_config.get("test", 0.1),
            seed=seed,
        )
    elif strategy == "scaffold":
        return scaffold_split(
            df,
            train_frac=split_config.get("train", 0.8),
            val_frac=split_config.get("val", 0.1),
            test_frac=split_config.get("test", 0.1),
            seed=seed,
        )
    elif strategy == "target":
        return target_split(
            df,
            train_frac=split_config.get("train", 0.8),
            val_frac=split_config.get("val", 0.1),
            test_frac=split_config.get("test", 0.1),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")


def save_splits(splits: dict[str, np.ndarray], output_path: Path) -> None:
    """Save split indices to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v.tolist() for k, v in splits.items()}
    with open(output_path, "w") as f:
        json.dump(serializable, f)
    logger.info("Saved splits to %s", output_path)


def load_splits(path: Path) -> dict[str, np.ndarray]:
    """Load split indices from JSON."""
    with open(path) as f:
        data = json.load(f)
    return {k: np.array(v) for k, v in data.items()}
