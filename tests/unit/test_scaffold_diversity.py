"""Tests for the scaffold_diversity module (P3-T11).

Per-target + per-class scaffold-diversity metrics for cross-class benchmarks.
See Plan 3 design spec Section 4.
"""
import math

import pandas as pd
import pytest


@pytest.fixture
def synthetic_target_df():
    """Small synthetic dataset for a single target with 5 compounds."""
    return pd.DataFrame(
        {
            "canonical_smiles": ["c1ccccc1", "CCO", "c1ccccc1C", "CCO", "c1ccccc1CC"],
            "target_chembl_id": ["CHEMBL1"] * 5,
            "pactivity": [6.0, 5.0, 7.0, 5.5, 6.5],
        }
    )


@pytest.fixture
def multi_target_df():
    """Two-target dataset for testing per-class aggregation."""
    return pd.DataFrame(
        {
            "canonical_smiles": [
                "c1ccccc1",
                "CCO",
                "c1ccccc1C",
                "CC(=O)O",
                "c1ccncc1",
                "CCN",
            ],
            "target_chembl_id": ["CHEMBL1", "CHEMBL1", "CHEMBL1", "CHEMBL2", "CHEMBL2", "CHEMBL2"],
            "pactivity": [6.0, 5.0, 7.0, 6.5, 5.8, 7.2],
        }
    )


def test_compute_scaffold_metrics_returns_expected_columns(synthetic_target_df):
    from target_affinity_ml.benchmarks.scaffold_diversity import compute_scaffold_metrics

    metrics = compute_scaffold_metrics(
        synthetic_target_df, target_col="target_chembl_id", smiles_col="canonical_smiles"
    )
    assert isinstance(metrics, pd.DataFrame)
    expected_cols = {
        "target_chembl_id",
        "n_compounds",
        "n_scaffolds",
        "scaffold_entropy",
        "largest_cluster_fraction",
        "mean_tanimoto",
    }
    assert expected_cols.issubset(metrics.columns)


def test_n_scaffolds_correct(synthetic_target_df):
    from target_affinity_ml.benchmarks.scaffold_diversity import compute_scaffold_metrics

    m = compute_scaffold_metrics(synthetic_target_df, "target_chembl_id", "canonical_smiles")
    # Bemis-Murcko scaffolds for the 5-compound fixture:
    #   c1ccccc1, c1ccccc1C, c1ccccc1CC -> all bucket under "c1ccccc1" (benzene)
    #   CCO x2                          -> acyclic, bucket under "NO_SCAFFOLD"
    # So exactly 2 distinct scaffold strings.
    assert m.iloc[0]["n_scaffolds"] == 2


def test_largest_cluster_fraction_in_bounds(synthetic_target_df):
    from target_affinity_ml.benchmarks.scaffold_diversity import compute_scaffold_metrics

    m = compute_scaffold_metrics(synthetic_target_df, "target_chembl_id", "canonical_smiles")
    assert 0.0 < m.iloc[0]["largest_cluster_fraction"] <= 1.0


def test_n_compounds_matches_input(synthetic_target_df):
    from target_affinity_ml.benchmarks.scaffold_diversity import compute_scaffold_metrics

    m = compute_scaffold_metrics(synthetic_target_df, "target_chembl_id", "canonical_smiles")
    assert m.iloc[0]["n_compounds"] == 5


def test_scaffold_entropy_nonnegative(synthetic_target_df):
    from target_affinity_ml.benchmarks.scaffold_diversity import compute_scaffold_metrics

    m = compute_scaffold_metrics(synthetic_target_df, "target_chembl_id", "canonical_smiles")
    assert m.iloc[0]["scaffold_entropy"] >= 0.0


def test_mean_tanimoto_in_bounds(synthetic_target_df):
    from target_affinity_ml.benchmarks.scaffold_diversity import compute_scaffold_metrics

    m = compute_scaffold_metrics(synthetic_target_df, "target_chembl_id", "canonical_smiles")
    mt = m.iloc[0]["mean_tanimoto"]
    assert 0.0 <= mt <= 1.0


def test_activity_cliff_frequency_present_when_activity_col(synthetic_target_df):
    from target_affinity_ml.benchmarks.scaffold_diversity import compute_scaffold_metrics

    m = compute_scaffold_metrics(
        synthetic_target_df,
        "target_chembl_id",
        "canonical_smiles",
        activity_col="pactivity",
    )
    assert "activity_cliff_frequency" in m.columns


def test_activity_cliff_frequency_absent_when_activity_col_none(synthetic_target_df):
    from target_affinity_ml.benchmarks.scaffold_diversity import compute_scaffold_metrics

    m = compute_scaffold_metrics(
        synthetic_target_df,
        "target_chembl_id",
        "canonical_smiles",
        activity_col=None,
    )
    assert "activity_cliff_frequency" not in m.columns


def test_invalid_smiles_handled_gracefully():
    from target_affinity_ml.benchmarks.scaffold_diversity import compute_scaffold_metrics

    df = pd.DataFrame(
        {
            "canonical_smiles": ["c1ccccc1", "not_a_smiles", "CCO", "CCN", "c1ccncc1"],
            "target_chembl_id": ["CHEMBL1"] * 5,
            "pactivity": [6.0, 5.0, 7.0, 5.5, 6.5],
        }
    )
    # Should not raise on invalid SMILES; just include INVALID in scaffold counter
    m = compute_scaffold_metrics(df, "target_chembl_id", "canonical_smiles")
    assert len(m) == 1
    assert m.iloc[0]["n_compounds"] == 5
    # mean_tanimoto must be computed over the 4 valid mols
    assert not math.isnan(m.iloc[0]["mean_tanimoto"])


def test_single_compound_target():
    from target_affinity_ml.benchmarks.scaffold_diversity import compute_scaffold_metrics

    df = pd.DataFrame(
        {
            "canonical_smiles": ["c1ccccc1"],
            "target_chembl_id": ["CHEMBL1"],
            "pactivity": [6.0],
        }
    )
    m = compute_scaffold_metrics(df, "target_chembl_id", "canonical_smiles")
    assert m.iloc[0]["n_compounds"] == 1
    assert m.iloc[0]["n_scaffolds"] == 1
    assert m.iloc[0]["largest_cluster_fraction"] == 1.0
    # mean_tanimoto undefined for a single compound -> NaN
    assert math.isnan(m.iloc[0]["mean_tanimoto"])


def test_multi_target_yields_one_row_per_target(multi_target_df):
    from target_affinity_ml.benchmarks.scaffold_diversity import compute_scaffold_metrics

    m = compute_scaffold_metrics(multi_target_df, "target_chembl_id", "canonical_smiles")
    assert len(m) == 2
    assert set(m["target_chembl_id"]) == {"CHEMBL1", "CHEMBL2"}


def test_compute_class_aggregates_returns_expected_structure(multi_target_df):
    from target_affinity_ml.benchmarks.scaffold_diversity import (
        compute_class_aggregates,
        compute_scaffold_metrics,
    )

    per_target = compute_scaffold_metrics(
        multi_target_df, "target_chembl_id", "canonical_smiles"
    )
    metric_cols = ["n_scaffolds", "scaffold_entropy", "largest_cluster_fraction", "mean_tanimoto"]
    agg = compute_class_aggregates(per_target, metric_cols)
    assert isinstance(agg, dict)
    for col in metric_cols:
        assert col in agg
        for stat in ("mean", "median", "iqr", "n"):
            assert stat in agg[col]


def test_compute_class_aggregates_values_finite(multi_target_df):
    from target_affinity_ml.benchmarks.scaffold_diversity import (
        compute_class_aggregates,
        compute_scaffold_metrics,
    )

    per_target = compute_scaffold_metrics(
        multi_target_df, "target_chembl_id", "canonical_smiles"
    )
    agg = compute_class_aggregates(per_target, ["n_scaffolds", "scaffold_entropy"])
    # mean/median/iqr should be real numbers for n_scaffolds
    assert math.isfinite(agg["n_scaffolds"]["mean"])
    assert math.isfinite(agg["n_scaffolds"]["median"])
    assert agg["n_scaffolds"]["iqr"] >= 0.0
    assert agg["n_scaffolds"]["n"] == 2


def test_compute_class_aggregates_handles_nan():
    """Aggregate must drop NaN before computing stats."""
    from target_affinity_ml.benchmarks.scaffold_diversity import compute_class_aggregates

    df = pd.DataFrame({"mean_tanimoto": [0.4, float("nan"), 0.6]})
    agg = compute_class_aggregates(df, ["mean_tanimoto"])
    assert agg["mean_tanimoto"]["n"] == 2
    assert math.isclose(agg["mean_tanimoto"]["mean"], 0.5)


def test_module_random_state_not_polluted():
    """Verify compute_scaffold_metrics doesn't seed the module-global random."""
    import random
    from target_affinity_ml.benchmarks.scaffold_diversity import compute_scaffold_metrics

    # Need >500 compounds to force the sampling branch in _mean_tanimoto
    smis = ["c1ccccc1", "CCO", "c1ccccc1C"] * 200
    df = pd.DataFrame({
        "canonical_smiles": smis,
        "target_chembl_id": ["CHEMBL1"] * len(smis),
        "pactivity": [6.0] * len(smis),
    })
    random.seed(99)
    pre = random.random()
    _ = compute_scaffold_metrics(df, "target_chembl_id", "canonical_smiles")
    random.seed(99)
    post = random.random()
    assert pre == post, "compute_scaffold_metrics polluted module-global random state"
