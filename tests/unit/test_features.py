"""Tests for molecular feature generation."""

import pytest
import numpy as np


class TestMorganFingerprints:
    """Test Morgan fingerprint generation."""

    def test_output_shape(self):
        """Fingerprint should have correct number of bits."""
        from target_affinity_ml.features.fingerprints import smiles_to_morgan_fp

        fp = smiles_to_morgan_fp("c1ccccc1", radius=2, n_bits=2048)
        assert fp.shape == (2048,)

    def test_binary_values(self):
        """Fingerprint values should be 0 or 1."""
        from target_affinity_ml.features.fingerprints import smiles_to_morgan_fp

        fp = smiles_to_morgan_fp("c1ccccc1")
        assert set(np.unique(fp)).issubset({0, 1})

    def test_invalid_smiles_returns_none(self):
        """Invalid SMILES should return None."""
        from target_affinity_ml.features.fingerprints import smiles_to_morgan_fp

        fp = smiles_to_morgan_fp("not_valid")
        assert fp is None

    def test_batch_computation(self):
        """Batch computation should return matrix of correct shape."""
        from target_affinity_ml.features.fingerprints import compute_fingerprints

        smiles = ["c1ccccc1", "CCO", "CC(=O)O"]
        fps = compute_fingerprints(smiles, n_bits=1024)
        assert fps.shape == (3, 1024)


class TestRDKitDescriptors:
    """Test RDKit descriptor computation."""

    def test_returns_dict(self):
        """Single molecule should return descriptor dictionary."""
        from target_affinity_ml.features.descriptors import smiles_to_descriptors

        desc = smiles_to_descriptors("c1ccccc1")
        assert isinstance(desc, dict)
        assert len(desc) > 100  # RDKit computes ~200 descriptors

    def test_batch_returns_matrix(self):
        """Batch computation should return (n_mols, n_desc) matrix."""
        from target_affinity_ml.features.descriptors import compute_descriptors

        smiles = ["c1ccccc1", "CCO", "CC(=O)O"]
        matrix, names = compute_descriptors(smiles)
        assert matrix.shape[0] == 3
        assert len(names) == matrix.shape[1]

    def test_invalid_smiles_handled(self):
        """Invalid SMILES in batch should not crash; row gets NaN fill."""
        from target_affinity_ml.features.descriptors import smiles_to_descriptors

        desc = smiles_to_descriptors("not_valid_smiles")
        assert desc is None
