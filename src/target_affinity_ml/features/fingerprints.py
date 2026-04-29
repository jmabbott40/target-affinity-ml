"""Morgan fingerprint generation using RDKit.

Morgan (circular) fingerprints encode the local chemical environment
around each atom up to a given radius. They are the standard baseline
representation for molecular property prediction.

Key parameters:
    - radius: number of bonds from each atom to consider
      (radius=2 ≈ ECFP4, radius=3 ≈ ECFP6)
    - n_bits: fingerprint length (2048 is standard)

The fingerprint is a bit vector where each bit indicates the presence
or absence of a particular substructural feature.

Usage:
    from target_affinity_ml.features.fingerprints import smiles_to_morgan_fp
    fp = smiles_to_morgan_fp("c1ccccc1", radius=2, n_bits=2048)
"""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

logger = logging.getLogger(__name__)


def smiles_to_morgan_fp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray | None:
    """Convert a SMILES string to a Morgan fingerprint bit vector.

    Uses the modern rdFingerprintGenerator API (replaces deprecated
    GetMorganFingerprintAsBitVect).

    Parameters
    ----------
    smiles : str
        Canonical SMILES string.
    radius : int
        Fingerprint radius (2 = ECFP4, 3 = ECFP6).
    n_bits : int
        Length of the bit vector.

    Returns
    -------
    np.ndarray or None
        Binary fingerprint vector of shape (n_bits,), or None if
        the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Use the modern MorganGenerator API
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprintAsNumPy(mol)
    return fp.astype(np.uint8)


def compute_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute Morgan fingerprints for a list of SMILES.

    Creates a single MorganGenerator instance and reuses it for all
    molecules (more efficient than creating one per molecule).

    Parameters
    ----------
    smiles_list : list[str]
        List of canonical SMILES strings.
    radius : int
        Fingerprint radius.
    n_bits : int
        Fingerprint length.

    Returns
    -------
    np.ndarray
        Fingerprint matrix of shape (n_molecules, n_bits), dtype uint8.
        Rows for invalid SMILES are all zeros (should be rare for
        standardized input).
    """
    n_mols = len(smiles_list)
    fp_matrix = np.zeros((n_mols, n_bits), dtype=np.uint8)
    n_failed = 0

    # Create generator once and reuse (avoids per-molecule overhead)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    for i, smi in enumerate(smiles_list):
        if (i + 1) % 50_000 == 0 or i == 0:
            logger.info(
                "  Fingerprints: %d/%d molecules (%.0f%%)",
                i + 1, n_mols, (i + 1) / n_mols * 100,
            )

        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp_matrix[i] = gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
        else:
            n_failed += 1

    if n_failed > 0:
        logger.warning(
            "Failed to compute fingerprints for %d/%d molecules (%.2f%%)",
            n_failed, n_mols, n_failed / n_mols * 100,
        )
    else:
        logger.info(
            "Successfully computed fingerprints for all %d molecules", n_mols,
        )

    return fp_matrix
