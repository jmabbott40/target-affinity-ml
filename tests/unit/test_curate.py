"""Tests for curate_activities: the class-agnostic curation entry point.

Covers:
- GO-based (kinase-style) path: target metadata merged from targets file,
  kinase_group renamed to subfamily.
- Explicit-target-list path: subfamily populated from config.subfamily_map,
  no targets file required.
- Missing targets file: graceful handling (no merge, no crash).
- Returns a DataFrame with required columns and correct dtypes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from target_affinity_ml.data.target_class_config import TargetClassConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Minimal valid dataset_config dict matching what curate.main() reads from YAML
MINIMAL_DATASET_CONFIG = {
    "version": "vtest",
    "standardization": {
        "mw_min": 100.0,
        "mw_max": 900.0,
        "max_heavy_atoms": 100,
    },
    "duplicates": {
        "aggregation": "median",
        "noise_std_threshold": 1.0,
        "min_measurements_for_noise_flag": 3,
    },
    "quality": {
        "pactivity_min": 3.0,
        "pactivity_max": 12.0,
    },
    "classification": {
        "active_pactivity_threshold": 6.0,
    },
    "splits": {
        "random": {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42},
        "scaffold": {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42},
        "target": {"seed": 42},
    },
}

# Aspirin (valid, MW ~180, passes all filters)
_ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
# Ibuprofen (valid, MW ~206)
_IBUPROFEN_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
# Caffeine (valid, MW ~194)
_CAFFEINE_SMILES = "Cn1c(=O)c2c(ncn2C)n(c1=O)C"


def _make_raw_activities(target_ids: list[str]) -> pd.DataFrame:
    """Create a minimal raw activities DataFrame that passes standardization."""
    smiles_list = [_ASPIRIN_SMILES, _IBUPROFEN_SMILES, _CAFFEINE_SMILES]
    records = []
    for i, target_id in enumerate(target_ids):
        records.append({
            "activity_id": f"ACT{i}",
            "molecule_chembl_id": f"CHEMBL{i}",
            "canonical_smiles": smiles_list[i % len(smiles_list)],
            "target_chembl_id": target_id,
            "standard_type": "IC50",
            "standard_value": str(100.0 * (i + 1)),  # nM — ChEMBL returns strings
            "standard_units": "nM",
            "standard_relation": "=",
            "pchembl_value": str(7.0 - i * 0.5),
            "assay_chembl_id": f"ASSAY{i}",
            "assay_type": "B",
            "data_validity_comment": None,
        })
    return pd.DataFrame(records)


def _make_kinase_targets(target_ids: list[str]) -> pd.DataFrame:
    """Create a targets DataFrame with kinase_group column (GO-based class)."""
    records = []
    for i, tid in enumerate(target_ids):
        records.append({
            "target_chembl_id": tid,
            "pref_name": f"Target {i}",
            "gene_symbol": f"GENE{i}",
            "kinase_group": "Tyrosine kinase" if i % 2 == 0 else "Serine/Threonine kinase",
            "organism": "Homo sapiens",
        })
    return pd.DataFrame(records)


@pytest.fixture()
def kinase_config():
    """GO-based TargetClassConfig (kinase-style)."""
    return TargetClassConfig(
        class_name="kinase_test",
        raw_filename_stem="test_kinase",
        go_terms={"GO:0016301"},
    )


@pytest.fixture()
def gpcr_config():
    """Explicit-target-list TargetClassConfig (GPCR aminergic-style)."""
    return TargetClassConfig(
        class_name="gpcr_aminergic_test",
        raw_filename_stem="test_gpcr",
        explicit_target_ids=["CHEMBL1", "CHEMBL2", "CHEMBL3"],
        subfamily_map={
            "CHEMBL1": "dopamine",
            "CHEMBL2": "serotonin",
            "CHEMBL3": "adrenergic",
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCurateActivitiesImport:
    """curate_activities must be importable from the module."""

    def test_import(self):
        from target_affinity_ml.data.curate import curate_activities  # noqa: F401


class TestCurateActivitiesGoBased:
    """GO-based class (kinase): targets file present, kinase_group -> subfamily."""

    def test_returns_dataframe(self, tmp_path, kinase_config):
        from target_affinity_ml.data.curate import curate_activities

        target_ids = ["CHEMBL1", "CHEMBL2", "CHEMBL3"]
        raw_acts = _make_raw_activities(target_ids)
        raw_tgts = _make_kinase_targets(target_ids)

        raw_acts.to_parquet(tmp_path / kinase_config.raw_activities_filename)
        raw_tgts.to_parquet(tmp_path / kinase_config.raw_targets_filename)

        result = curate_activities(kinase_config, MINIMAL_DATASET_CONFIG, raw_dir=tmp_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_has_subfamily_column_not_kinase_group(self, tmp_path, kinase_config):
        """Output must have 'subfamily', NOT 'kinase_group'."""
        from target_affinity_ml.data.curate import curate_activities

        target_ids = ["CHEMBL1", "CHEMBL2", "CHEMBL3"]
        raw_acts = _make_raw_activities(target_ids)
        raw_tgts = _make_kinase_targets(target_ids)

        raw_acts.to_parquet(tmp_path / kinase_config.raw_activities_filename)
        raw_tgts.to_parquet(tmp_path / kinase_config.raw_targets_filename)

        result = curate_activities(kinase_config, MINIMAL_DATASET_CONFIG, raw_dir=tmp_path)

        assert "subfamily" in result.columns, "Expected 'subfamily' column in output"
        assert "kinase_group" not in result.columns, (
            "'kinase_group' must be renamed to 'subfamily' in curate_activities output"
        )

    def test_subfamily_values_from_targets_file(self, tmp_path, kinase_config):
        """subfamily values must come from kinase_group in the targets file."""
        from target_affinity_ml.data.curate import curate_activities

        target_ids = ["CHEMBL1", "CHEMBL2", "CHEMBL3"]
        raw_acts = _make_raw_activities(target_ids)
        raw_tgts = _make_kinase_targets(target_ids)

        raw_acts.to_parquet(tmp_path / kinase_config.raw_activities_filename)
        raw_tgts.to_parquet(tmp_path / kinase_config.raw_targets_filename)

        result = curate_activities(kinase_config, MINIMAL_DATASET_CONFIG, raw_dir=tmp_path)

        # All values should be one of the expected kinase groups
        expected_groups = {"Tyrosine kinase", "Serine/Threonine kinase"}
        actual = set(result["subfamily"].dropna().unique())
        assert actual.issubset(expected_groups), (
            f"Unexpected subfamily values: {actual - expected_groups}"
        )

    def test_required_output_columns(self, tmp_path, kinase_config):
        """Core curation columns must all be present."""
        from target_affinity_ml.data.curate import curate_activities

        target_ids = ["CHEMBL1", "CHEMBL2", "CHEMBL3"]
        raw_acts = _make_raw_activities(target_ids)
        raw_tgts = _make_kinase_targets(target_ids)

        raw_acts.to_parquet(tmp_path / kinase_config.raw_activities_filename)
        raw_tgts.to_parquet(tmp_path / kinase_config.raw_targets_filename)

        result = curate_activities(kinase_config, MINIMAL_DATASET_CONFIG, raw_dir=tmp_path)

        required = ["std_smiles", "target_chembl_id", "pactivity", "is_active", "is_noisy",
                    "n_measurements", "pactivity_std"]
        for col in required:
            assert col in result.columns, f"Missing required column: {col}"

    def test_missing_targets_file_graceful(self, tmp_path, kinase_config):
        """When targets file is absent, curate_activities should not crash."""
        from target_affinity_ml.data.curate import curate_activities

        target_ids = ["CHEMBL1", "CHEMBL2", "CHEMBL3"]
        raw_acts = _make_raw_activities(target_ids)
        raw_acts.to_parquet(tmp_path / kinase_config.raw_activities_filename)
        # No targets file written

        result = curate_activities(kinase_config, MINIMAL_DATASET_CONFIG, raw_dir=tmp_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # No crash is the key assertion; subfamily may be absent when file is missing


class TestCurateActivitiesExplicitTargetList:
    """Explicit-target-list class (GPCR): subfamily from config.subfamily_map."""

    def test_returns_dataframe(self, tmp_path, gpcr_config):
        from target_affinity_ml.data.curate import curate_activities

        target_ids = ["CHEMBL1", "CHEMBL2", "CHEMBL3"]
        raw_acts = _make_raw_activities(target_ids)
        raw_acts.to_parquet(tmp_path / gpcr_config.raw_activities_filename)

        result = curate_activities(gpcr_config, MINIMAL_DATASET_CONFIG, raw_dir=tmp_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_has_subfamily_from_map(self, tmp_path, gpcr_config):
        """subfamily must be populated from config.subfamily_map."""
        from target_affinity_ml.data.curate import curate_activities

        target_ids = ["CHEMBL1", "CHEMBL2", "CHEMBL3"]
        raw_acts = _make_raw_activities(target_ids)
        raw_acts.to_parquet(tmp_path / gpcr_config.raw_activities_filename)

        result = curate_activities(gpcr_config, MINIMAL_DATASET_CONFIG, raw_dir=tmp_path)

        assert "subfamily" in result.columns, "Expected 'subfamily' column"
        # All rows should have a subfamily value from the map
        expected = set(gpcr_config.subfamily_map.values())
        actual = set(result["subfamily"].dropna().unique())
        assert actual.issubset(expected), (
            f"Unexpected subfamily values: {actual - expected}"
        )

    def test_no_targets_file_needed(self, tmp_path, gpcr_config):
        """Explicit-list classes need no targets parquet file."""
        from target_affinity_ml.data.curate import curate_activities

        target_ids = ["CHEMBL1", "CHEMBL2", "CHEMBL3"]
        raw_acts = _make_raw_activities(target_ids)
        raw_acts.to_parquet(tmp_path / gpcr_config.raw_activities_filename)
        # Deliberately do NOT create a targets file

        result = curate_activities(gpcr_config, MINIMAL_DATASET_CONFIG, raw_dir=tmp_path)
        assert isinstance(result, pd.DataFrame)
        assert "subfamily" in result.columns

    def test_does_not_have_kinase_group_column(self, tmp_path, gpcr_config):
        """The output for explicit-list class should never contain kinase_group."""
        from target_affinity_ml.data.curate import curate_activities

        target_ids = ["CHEMBL1", "CHEMBL2", "CHEMBL3"]
        raw_acts = _make_raw_activities(target_ids)
        raw_acts.to_parquet(tmp_path / gpcr_config.raw_activities_filename)

        result = curate_activities(gpcr_config, MINIMAL_DATASET_CONFIG, raw_dir=tmp_path)
        assert "kinase_group" not in result.columns


class TestCurateActivitiesDefaultRawDir:
    """curate_activities raw_dir parameter defaults to Path('data/raw')."""

    def test_default_raw_dir_is_data_raw(self):
        """Verify the function signature has raw_dir=Path('data/raw')."""
        import inspect
        from target_affinity_ml.data.curate import curate_activities

        sig = inspect.signature(curate_activities)
        assert "raw_dir" in sig.parameters
        default = sig.parameters["raw_dir"].default
        assert default == Path("data/raw"), f"Expected Path('data/raw'), got {default!r}"
