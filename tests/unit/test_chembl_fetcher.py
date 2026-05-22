"""Mocked tests for chembl_fetcher.py.

The ChEMBL client is imported inside each function body:
    from chembl_webresource_client.new_client import new_client

We patch at the source — ``chembl_webresource_client.new_client.new_client`` —
so that both ``fetch_kinase_targets`` and ``fetch_bioactivities`` (and by
extension ``fetch_target_class``) use our mock objects instead of hitting the
live API.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from target_affinity_ml.data.chembl_fetcher import (
    KINASE_CONFIG,
    fetch_bioactivities,
    fetch_kinase_targets,
    fetch_target_class,
)
from target_affinity_ml.data.target_class_config import TargetClassConfig

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SAMPLE_ACTIVITY = {
    "activity_id": "1",
    "molecule_chembl_id": "CHEMBL123",
    "canonical_smiles": "c1ccccc1",
    "target_chembl_id": "CHEMBL217",
    "standard_type": "IC50",
    "standard_value": "100",
    "standard_units": "nM",
    "standard_relation": "=",
    "pchembl_value": "7.0",
    "assay_chembl_id": "CHEMBL_ASSAY_1",
    "assay_type": "B",
    "data_validity_comment": None,
}

_SAMPLE_TARGET = {
    "target_chembl_id": "CHEMBL217",
    "pref_name": "Dopamine D2 receptor",
    "target_type": "SINGLE PROTEIN",
    "organism": "Homo sapiens",
    "target_components": [
        {
            "target_component_xrefs": [
                {"xref_src_db": "GoFunction", "xref_id": "GO:0016301"},
            ],
            "target_component_synonyms": [
                {"syn_type": "GENE_SYMBOL", "component_synonym": "DRD2"},
            ],
        }
    ],
}


def _make_activity_api_mock(activity_records: list[dict]) -> MagicMock:
    """Return a mock activity API whose .filter() returns activity_records."""
    activity_api = MagicMock()
    activity_api.filter.return_value = activity_records
    return activity_api


def _make_target_api_mock(target_records: list[dict]) -> MagicMock:
    """Return a mock target API whose .search().filter() and .filter() return target_records.

    fetch_kinase_targets uses a two-pass strategy:
      1. Fast pass: target_api.search("kinase").filter(...)
      2. Fallback (if < MIN_EXPECTED_KINASES=100 found): target_api.filter(...)

    Both paths must return records so the fallback isn't triggered when we have
    a small synthetic fixture that legitimately has fewer than 100 targets.
    """
    target_api = MagicMock()
    target_api.search.return_value.filter.return_value = target_records
    # Fallback path — also returns the same records so the function does not
    # re-enter the full-download branch with an unconfigured mock.
    target_api.filter.return_value = target_records
    return target_api


# ---------------------------------------------------------------------------
# Tests for fetch_target_class — explicit-ID path
# ---------------------------------------------------------------------------


class TestFetchTargetClassExplicit:
    """fetch_target_class with explicit_target_ids skips GO discovery."""

    def test_queries_only_explicit_ids(self):
        """Activities are requested for exactly the explicit IDs; no target search."""
        cfg = TargetClassConfig(
            class_name="gpcr_aminergic",
            explicit_target_ids=["CHEMBL217", "CHEMBL224"],
            raw_filename_stem="chembl_gpcr_aminergic",
        )

        activity_mock = _make_activity_api_mock([_SAMPLE_ACTIVITY])
        client_mock = MagicMock()
        client_mock.activity = activity_mock

        with patch(
            "chembl_webresource_client.new_client.new_client", client_mock
        ):
            activities_df, targets_df = fetch_target_class(cfg)

        # Activity API should have been called once per target ID
        assert activity_mock.filter.call_count == 2

        # Each call should use one of the explicit target IDs
        called_target_ids = {
            call.kwargs.get("target_chembl_id") or call.args[0]
            for call in activity_mock.filter.call_args_list
        }
        # filter is called with keyword arg target_chembl_id=
        filter_calls = activity_mock.filter.call_args_list
        queried_ids = {c.kwargs["target_chembl_id"] for c in filter_calls}
        assert queried_ids == {"CHEMBL217", "CHEMBL224"}

        # targets_df should only contain the explicit IDs
        assert set(targets_df["target_chembl_id"]) == {"CHEMBL217", "CHEMBL224"}

        # Target search should NOT have been called (no GO discovery)
        assert client_mock.target.search.call_count == 0

    def test_returns_dataframes(self):
        """Return types are (pd.DataFrame, pd.DataFrame)."""
        cfg = TargetClassConfig(
            class_name="gpcr_aminergic",
            explicit_target_ids=["CHEMBL217"],
            raw_filename_stem="chembl_gpcr_aminergic",
        )

        activity_mock = _make_activity_api_mock([_SAMPLE_ACTIVITY])
        client_mock = MagicMock()
        client_mock.activity = activity_mock

        with patch(
            "chembl_webresource_client.new_client.new_client", client_mock
        ):
            activities_df, targets_df = fetch_target_class(cfg)

        assert isinstance(activities_df, pd.DataFrame)
        assert isinstance(targets_df, pd.DataFrame)

    def test_subfamily_map_populated_in_targets(self):
        """targets_df.subfamily reflects config.subfamily_map."""
        cfg = TargetClassConfig(
            class_name="gpcr_aminergic",
            explicit_target_ids=["CHEMBL217", "CHEMBL224"],
            raw_filename_stem="chembl_gpcr_aminergic",
            subfamily_map={"CHEMBL217": "dopamine", "CHEMBL224": "serotonin"},
        )

        activity_mock = _make_activity_api_mock([])
        client_mock = MagicMock()
        client_mock.activity = activity_mock

        with patch(
            "chembl_webresource_client.new_client.new_client", client_mock
        ):
            _, targets_df = fetch_target_class(cfg)

        assert targets_df.set_index("target_chembl_id").loc["CHEMBL217", "subfamily"] == "dopamine"
        assert targets_df.set_index("target_chembl_id").loc["CHEMBL224", "subfamily"] == "serotonin"


# ---------------------------------------------------------------------------
# Tests for fetch_target_class — GO-term discovery path
# ---------------------------------------------------------------------------


class TestFetchTargetClassGO:
    """fetch_target_class with go_terms runs GO discovery via fetch_kinase_targets."""

    def test_go_path_calls_target_search(self):
        """GO-based config triggers a search call on the target API."""
        cfg = TargetClassConfig(
            class_name="kinase",
            go_terms={"GO:0016301"},
            name_keywords=["kinase"],
            raw_filename_stem="chembl_kinase",
        )

        target_mock = _make_target_api_mock([_SAMPLE_TARGET])
        activity_mock = _make_activity_api_mock([_SAMPLE_ACTIVITY])
        client_mock = MagicMock()
        client_mock.target = target_mock
        client_mock.activity = activity_mock

        with patch(
            "chembl_webresource_client.new_client.new_client", client_mock
        ):
            activities_df, targets_df = fetch_target_class(cfg)

        # Target search should have been called
        target_mock.search.assert_called_once_with("kinase")

        assert isinstance(targets_df, pd.DataFrame)
        assert isinstance(activities_df, pd.DataFrame)

    def test_go_path_returns_discovered_targets(self):
        """targets_df contains the targets discovered via GO filtering."""
        cfg = KINASE_CONFIG  # the module-level constant

        target_mock = _make_target_api_mock([_SAMPLE_TARGET])
        activity_mock = _make_activity_api_mock([_SAMPLE_ACTIVITY])
        client_mock = MagicMock()
        client_mock.target = target_mock
        client_mock.activity = activity_mock

        with patch(
            "chembl_webresource_client.new_client.new_client", client_mock
        ):
            _, targets_df = fetch_target_class(cfg)

        assert "CHEMBL217" in targets_df["target_chembl_id"].values


# ---------------------------------------------------------------------------
# Regression guard: fetch_kinase_targets and fetch_bioactivities unchanged
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Existing public functions keep their original signatures and behavior."""

    def test_fetch_kinase_targets_returns_dataframe(self):
        """fetch_kinase_targets(organism) -> pd.DataFrame (unchanged signature)."""
        target_mock = _make_target_api_mock([_SAMPLE_TARGET])
        client_mock = MagicMock()
        client_mock.target = target_mock

        with patch(
            "chembl_webresource_client.new_client.new_client", client_mock
        ):
            df = fetch_kinase_targets(organism="Homo sapiens")

        assert isinstance(df, pd.DataFrame)
        # Should have discovered the sample target (GO:0016301 in KINASE_GO_TERMS)
        assert "CHEMBL217" in df["target_chembl_id"].values

    def test_fetch_kinase_targets_has_kinase_group_column(self):
        """kinase_group column is still present (downstream code relies on it)."""
        target_mock = _make_target_api_mock([_SAMPLE_TARGET])
        client_mock = MagicMock()
        client_mock.target = target_mock

        with patch(
            "chembl_webresource_client.new_client.new_client", client_mock
        ):
            df = fetch_kinase_targets()

        assert "kinase_group" in df.columns

    def test_fetch_bioactivities_returns_dataframe(self):
        """fetch_bioactivities(target_chembl_ids, ...) -> pd.DataFrame (unchanged)."""
        activity_mock = _make_activity_api_mock([_SAMPLE_ACTIVITY])
        client_mock = MagicMock()
        client_mock.activity = activity_mock

        with patch(
            "chembl_webresource_client.new_client.new_client", client_mock
        ):
            df = fetch_bioactivities(
                target_chembl_ids=["CHEMBL217"],
                activity_types=["IC50"],
            )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["molecule_chembl_id"] == "CHEMBL123"

    def test_fetch_bioactivities_max_targets_respected(self):
        """max_targets truncates the target list (unchanged behavior)."""
        activity_mock = _make_activity_api_mock([_SAMPLE_ACTIVITY])
        client_mock = MagicMock()
        client_mock.activity = activity_mock

        with patch(
            "chembl_webresource_client.new_client.new_client", client_mock
        ):
            fetch_bioactivities(
                target_chembl_ids=["CHEMBL217", "CHEMBL224", "CHEMBL999"],
                max_targets=1,
            )

        # Only 1 target queried despite 3 provided
        assert activity_mock.filter.call_count == 1

    def test_kinase_config_constant_exported(self):
        """KINASE_CONFIG is available at module level with correct class_name."""
        assert KINASE_CONFIG.class_name == "kinase"
        assert KINASE_CONFIG.raw_filename_stem == "chembl_kinase"
        assert not KINASE_CONFIG.uses_explicit_target_list
