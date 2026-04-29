"""Tests for dataset splitting strategies."""

import numpy as np
import pandas as pd
import pytest


def _make_dummy_df(n: int = 100) -> pd.DataFrame:
    """Create a minimal DataFrame for split testing."""
    return pd.DataFrame({
        "std_smiles": [f"C{'C' * i}" for i in range(n)],
        "target_chembl_id": [f"CHEMBL{i % 10}" for i in range(n)],
        "pactivity": np.random.default_rng(42).normal(6.0, 1.5, n),
    })


class TestRandomSplit:
    """Test random splitting."""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_split_sizes(self):
        """Split sizes should approximate target fractions."""
        from target_affinity_ml.data.splits import random_split

        df = _make_dummy_df(1000)
        splits = random_split(df, train_frac=0.8, val_frac=0.1, test_frac=0.1)
        assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 1000

    @pytest.mark.skip(reason="Not yet implemented")
    def test_no_overlap(self):
        """Splits should not contain overlapping indices."""
        from target_affinity_ml.data.splits import random_split

        df = _make_dummy_df(100)
        splits = random_split(df)
        all_indices = np.concatenate([splits["train"], splits["val"], splits["test"]])
        assert len(all_indices) == len(set(all_indices))

    @pytest.mark.skip(reason="Not yet implemented")
    def test_reproducibility(self):
        """Same seed should produce same splits."""
        from target_affinity_ml.data.splits import random_split

        df = _make_dummy_df(100)
        s1 = random_split(df, seed=42)
        s2 = random_split(df, seed=42)
        np.testing.assert_array_equal(s1["train"], s2["train"])


class TestScaffoldSplit:
    """Test scaffold-based splitting."""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_no_scaffold_leakage(self):
        """No scaffold should appear in more than one split."""
        # Requires RDKit for Murcko scaffold extraction
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_all_indices_covered(self):
        """All dataset indices should be assigned to exactly one split."""
        pass
