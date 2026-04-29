"""RDKit 2D molecular descriptor calculation.

RDKit provides ~200 computed molecular descriptors covering:
    - Constitutional: MW, heavy atom count, ring count
    - Topological: Wiener index, Balaban J, etc.
    - Electronic: partial charges, TPSA
    - Physicochemical: LogP, HBA, HBD, rotatable bonds
    - Complexity: BertzCT, fragment counts

Unlike fingerprints, descriptors are continuous real-valued features
that benefit from feature scaling (StandardScaler). They provide
complementary information to fingerprint-based models.

Usage:
    from target_affinity_ml.features.descriptors import compute_descriptors
    desc_matrix, desc_names = compute_descriptors(smiles_list)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

logger = logging.getLogger(__name__)

# Pre-compute the list of all available 2D descriptor names.
# This avoids re-discovering them on every call.
_ALL_DESCRIPTOR_NAMES: list[str] = [name for name, _ in Descriptors.descList]


def smiles_to_descriptors(smiles: str) -> dict[str, float] | None:
    """Compute all RDKit 2D descriptors for a single molecule.

    Uses the full set of ~210 RDKit descriptors, including molecular weight,
    LogP, TPSA, H-bond donors/acceptors, ring counts, and many more.

    Parameters
    ----------
    smiles : str
        Canonical SMILES string.

    Returns
    -------
    dict[str, float] or None
        Dictionary of descriptor_name → value, or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # CalcMolDescriptors returns a dict of {name: value} for all descriptors
    return Descriptors.CalcMolDescriptors(mol)


def compute_descriptors(
    smiles_list: list[str],
    drop_missing_threshold: float = 0.05,
) -> tuple[np.ndarray, list[str]]:
    """Compute RDKit descriptors for a list of SMILES.

    Computes all ~210 RDKit 2D descriptors for each molecule, then applies
    quality filters:
    1. Drops descriptors with >threshold fraction of NaN/Inf values
    2. Replaces remaining NaN/Inf with column median (imputation)

    Parameters
    ----------
    smiles_list : list[str]
        List of canonical SMILES strings.
    drop_missing_threshold : float
        Drop descriptors with more than this fraction of NaN/Inf values.
        Default 0.05 = drop if >5% of molecules have invalid values.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        (descriptor_matrix, descriptor_names).
        Matrix shape: (n_molecules, n_descriptors), dtype float64.
        descriptor_names: list of retained descriptor names.
    """
    n_mols = len(smiles_list)
    all_desc_dicts: list[dict[str, float] | None] = []
    n_failed = 0

    for i, smi in enumerate(smiles_list):
        if (i + 1) % 50_000 == 0 or i == 0:
            logger.info(
                "  Descriptors: %d/%d molecules (%.0f%%)",
                i + 1, n_mols, (i + 1) / n_mols * 100,
            )

        desc = smiles_to_descriptors(smi)
        if desc is not None:
            all_desc_dicts.append(desc)
        else:
            # Use NaN dict for failed molecules so DataFrame stays aligned
            all_desc_dicts.append({name: np.nan for name in _ALL_DESCRIPTOR_NAMES})
            n_failed += 1

    if n_failed > 0:
        logger.warning(
            "Failed to compute descriptors for %d/%d molecules (%.2f%%)",
            n_failed, n_mols, n_failed / n_mols * 100,
        )

    # Build DataFrame from list of dicts
    df = pd.DataFrame(all_desc_dicts)
    n_total_desc = len(df.columns)
    logger.info("Computed %d descriptors for %d molecules", n_total_desc, n_mols)

    # Replace Inf with NaN for uniform missing-value handling
    df = df.replace([np.inf, -np.inf], np.nan)

    # Step 1: Drop descriptors with too many missing values
    missing_frac = df.isna().mean()
    cols_to_drop = missing_frac[missing_frac > drop_missing_threshold].index.tolist()
    if cols_to_drop:
        logger.info(
            "Dropping %d descriptors with >%.0f%% missing values: %s",
            len(cols_to_drop),
            drop_missing_threshold * 100,
            cols_to_drop[:10],  # Show first 10 for brevity
        )
        df = df.drop(columns=cols_to_drop)

    # Step 2: Impute remaining NaN with column median
    n_remaining_nan = df.isna().sum().sum()
    if n_remaining_nan > 0:
        logger.info(
            "Imputing %d remaining NaN values with column medians", n_remaining_nan,
        )
        df = df.fillna(df.median())

    descriptor_names = df.columns.tolist()
    matrix = df.to_numpy(dtype=np.float64)

    logger.info(
        "Final descriptor matrix: (%d, %d) — %d descriptors retained out of %d",
        matrix.shape[0], matrix.shape[1], len(descriptor_names), n_total_desc,
    )

    return matrix, descriptor_names
