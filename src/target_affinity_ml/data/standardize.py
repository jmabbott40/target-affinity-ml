"""Molecule standardization using RDKit.

Why standardize molecules?
--------------------------
The same molecule can be represented by many different SMILES strings:
    - With or without salts: "CC(=O)O.[Na]" vs "CC(=O)O"
    - Different charge states: "[NH3+]CC" vs "NCC"
    - Different atom ordering: "c1ccccc1" vs "C1=CC=CC=C1"
    - With stereochemistry or without: "C/C=C/C" vs "CC=CC"

If we don't standardize, the same compound may appear as multiple entries
in our dataset, inflating our training data and leaking information between
splits. Worse, fingerprints computed from different SMILES representations
of the same molecule will differ, confusing our models.

Standardization pipeline
------------------------
Each molecule goes through these steps:

1. **Parse SMILES** -> RDKit Mol object
   - Validates chemical structure (rejects invalid SMILES)
   - Detects aromaticity, sets implicit H counts

2. **Remove salts** (keep largest fragment)
   - Many compounds are stored as salts: "drug.[Na]", "drug.Cl"
   - The salt doesn't affect binding to the target protein
   - We keep the largest fragment (the drug molecule)

3. **Neutralize charges**
   - Convert "[NH3+]CC" -> "NCC", "[O-]c1ccccc1" -> "Oc1ccccc1"
   - Charged forms are pH-dependent; neutral form is canonical

4. **Generate canonical SMILES**
   - RDKit's canonical SMILES is a unique string for each molecule
   - Ensures "c1ccccc1" and "C1=CC=CC=C1" both become "c1ccccc1"

5. **Molecular weight / size filter**
   - Remove very small molecules (MW < 100): likely fragments, salts, or solvents
   - Remove very large molecules (MW > 900): likely peptides or non-drug-like
   - Remove molecules with > 100 heavy atoms: computational tractability

Usage:
    from target_affinity_ml.data.standardize import standardize_smiles

    canonical, is_valid = standardize_smiles("CC(=O)Oc1ccccc1C(=O)O.[Na]")
    # Returns: ("CC(=O)Oc1ccccc1C(=O)O", True) -- aspirin without sodium salt
"""

from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)

# Suppress RDKit's own warnings (we handle errors ourselves)
RDLogger.logger().setLevel(RDLogger.ERROR)

# RDKit's built-in salt remover knows common counterions
_SALT_REMOVER = SaltRemover.SaltRemover()

# Uncharger neutralizes molecules by removing unnecessary charges
_UNCHARGER = rdMolStandardize.Uncharger()


def standardize_smiles(
    smiles: str,
    mw_min: float = 100.0,
    mw_max: float = 900.0,
    max_heavy_atoms: int = 100,
) -> tuple[str | None, bool]:
    """Standardize a SMILES string through the full pipeline.

    Parameters
    ----------
    smiles : str
        Input SMILES string (may contain salts, charges, etc.).
    mw_min : float
        Minimum molecular weight in Da.
    mw_max : float
        Maximum molecular weight in Da.
    max_heavy_atoms : int
        Maximum number of heavy (non-hydrogen) atoms.

    Returns
    -------
    tuple[str | None, bool]
        (canonical_smiles, is_valid). If any step fails, returns (None, False).
    """
    if not smiles or not isinstance(smiles, str):
        return None, False

    # Step 1: Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, False

    # Step 2: Remove salts (keep largest fragment)
    try:
        mol = _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)
    except Exception:
        # If salt removal fails, try keeping the largest fragment manually
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if not frags:
            return None, False
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

    # Step 3: Neutralize charges
    try:
        mol = _UNCHARGER.uncharge(mol)
    except Exception:
        pass  # Keep the molecule even if neutralization fails

    # Step 4: Sanitize and generate canonical SMILES
    try:
        Chem.SanitizeMol(mol)
        canonical = Chem.MolToSmiles(mol)
    except Exception:
        return None, False

    # Step 5: Molecular weight filter
    mw = Descriptors.ExactMolWt(mol)
    if mw < mw_min or mw > mw_max:
        return None, False

    # Step 6: Heavy atom count filter
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy > max_heavy_atoms:
        return None, False

    return canonical, True


def standardize_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    config: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply standardization to a DataFrame of molecules.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a SMILES column.
    smiles_col : str
        Name of the column containing SMILES strings.
    config : dict, optional
        Standardization parameters from dataset config.
        If None, uses defaults from function signature.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (cleaned_df, stats) where:
        - cleaned_df has a new 'std_smiles' column and invalid rows removed
        - stats contains counts of molecules removed at each step
    """
    # Extract config parameters
    if config is not None:
        std_config = config.get("standardization", {})
        mw_min = std_config.get("mw_min", 100.0)
        mw_max = std_config.get("mw_max", 900.0)
        max_heavy = std_config.get("max_heavy_atoms", 100)
    else:
        mw_min, mw_max, max_heavy = 100.0, 900.0, 100

    n_initial = len(df)
    logger.info("Standardizing %d molecules...", n_initial)

    # Apply standardization to each SMILES
    results = df[smiles_col].apply(
        lambda s: standardize_smiles(s, mw_min, mw_max, max_heavy)
    )

    df = df.copy()
    df["std_smiles"] = results.apply(lambda x: x[0])
    df["is_valid_mol"] = results.apply(lambda x: x[1])

    # Count removals
    n_invalid_smiles = df[smiles_col].isna().sum()
    n_failed_standardization = (~df["is_valid_mol"]).sum() - n_invalid_smiles

    # Remove invalid molecules
    df_clean = df[df["is_valid_mol"]].drop(columns=["is_valid_mol"]).copy()
    n_after = len(df_clean)

    stats = {
        "n_initial": n_initial,
        "n_invalid_smiles": int(n_invalid_smiles),
        "n_failed_standardization": int(n_failed_standardization),
        "n_removed_total": n_initial - n_after,
        "n_remaining": n_after,
        "pct_retained": round(n_after / n_initial * 100, 1) if n_initial > 0 else 0,
    }

    logger.info(
        "Standardization complete: %d -> %d molecules (%.1f%% retained)",
        n_initial, n_after, stats["pct_retained"],
    )
    logger.info("  Invalid SMILES: %d", stats["n_invalid_smiles"])
    logger.info("  Failed standardization (MW/size filter): %d", stats["n_failed_standardization"])

    return df_clean, stats
