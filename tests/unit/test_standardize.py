"""Tests for molecule standardization pipeline."""

import pytest


class TestStandardizeSmiles:
    """Test SMILES standardization."""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_valid_smiles(self):
        """Valid SMILES should return canonical form."""
        from target_affinity_ml.data.standardize import standardize_smiles

        canonical, is_valid = standardize_smiles("c1ccccc1")
        assert is_valid
        assert canonical == "c1ccccc1"  # benzene canonical

    @pytest.mark.skip(reason="Not yet implemented")
    def test_salt_removal(self):
        """Salts should be removed, keeping the largest fragment."""
        from target_affinity_ml.data.standardize import standardize_smiles

        canonical, is_valid = standardize_smiles("CC(=O)O.[Na]")
        assert is_valid
        assert "." not in canonical  # No salt separator

    @pytest.mark.skip(reason="Not yet implemented")
    def test_invalid_smiles(self):
        """Invalid SMILES should return (None, False)."""
        from target_affinity_ml.data.standardize import standardize_smiles

        canonical, is_valid = standardize_smiles("not_a_smiles")
        assert not is_valid
        assert canonical is None

    @pytest.mark.skip(reason="Not yet implemented")
    def test_mw_filter_too_small(self):
        """Molecules below MW minimum should be filtered."""
        from target_affinity_ml.data.standardize import standardize_smiles

        canonical, is_valid = standardize_smiles("C", mw_min=100.0)
        assert not is_valid

    @pytest.mark.skip(reason="Not yet implemented")
    def test_mw_filter_too_large(self):
        """Molecules above MW maximum should be filtered."""

        # Very large molecule — this would need a real large SMILES
        pass
