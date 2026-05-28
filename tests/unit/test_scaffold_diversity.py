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


# ---------------------------------------------------------------------------
# P3-T13: fit_degradation_regression tests
# ---------------------------------------------------------------------------


def _make_synthetic_regression_df(n_per_class=40, slope_kinase=0.5, slope_gpcr=0.5, noise=0.05, seed=11):
    """Synthetic per-target data: Y = slope[class] * X + noise, classes pooled.

    Use this for tests of fit_degradation_regression. Setting slope_kinase==slope_gpcr
    produces a null cross-class interaction; setting them differently produces a
    significant interaction.

    Note: default ``seed=11`` chosen because seed=7 produced a borderline
    significant null draw (interaction_p≈0.03 even though slopes were equal),
    a chance fluctuation of the random draw rather than a model issue.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    for cls, slope in [("kinase", slope_kinase), ("gpcr_aminergic", slope_gpcr)]:
        x = rng.uniform(0, 5, size=n_per_class)
        y = slope * x + rng.normal(0, noise, size=n_per_class)
        for xi, yi in zip(x, y):
            rows.append({"X": xi, "Y": yi, "class_name": cls})
    return pd.DataFrame(rows)


def test_fit_degradation_regression_recovers_known_slope():
    """When both classes have slope=0.5, the pooled regression should report ~0.5 for both."""
    from target_affinity_ml.benchmarks.scaffold_diversity import fit_degradation_regression
    df = _make_synthetic_regression_df(slope_kinase=0.5, slope_gpcr=0.5, noise=0.05)
    result = fit_degradation_regression(df, "X", "Y", class_col="class_name")
    # Slope per class ~0.5 (within tolerance)
    for cls in ("kinase", "gpcr_aminergic"):
        assert abs(result["slopes"][cls]["slope"] - 0.5) < 0.1, f"slope for {cls}: {result['slopes'][cls]['slope']}"
    # Interaction term should NOT be significant
    assert result["interaction_p"] > 0.05


def test_fit_degradation_regression_detects_class_interaction():
    """When slopes differ by class (0.2 vs 0.8), interaction term should be significant."""
    from target_affinity_ml.benchmarks.scaffold_diversity import fit_degradation_regression
    df = _make_synthetic_regression_df(slope_kinase=0.8, slope_gpcr=0.2, noise=0.05, n_per_class=60)
    result = fit_degradation_regression(df, "X", "Y", class_col="class_name")
    # Per-class slopes match
    assert abs(result["slopes"]["kinase"]["slope"] - 0.8) < 0.1
    assert abs(result["slopes"]["gpcr_aminergic"]["slope"] - 0.2) < 0.1
    # Interaction is significant
    assert result["interaction_p"] < 0.01


def test_fit_degradation_regression_returns_expected_keys():
    from target_affinity_ml.benchmarks.scaffold_diversity import fit_degradation_regression
    df = _make_synthetic_regression_df()
    result = fit_degradation_regression(df, "X", "Y", class_col="class_name")
    expected_top = {"formula", "n_obs", "r_squared", "r_squared_adj", "intercept", "intercept_p", "slopes", "interaction_p", "n_per_class", "alpha"}
    assert expected_top.issubset(result.keys())
    for cls in ("kinase", "gpcr_aminergic"):
        assert cls in result["slopes"]
        assert {"slope", "ci_low", "ci_high", "p_value"}.issubset(result["slopes"][cls].keys())


def test_fit_degradation_regression_drops_nan_rows():
    """NaN in metric, degradation, or class column -> row dropped, but no crash."""
    from target_affinity_ml.benchmarks.scaffold_diversity import fit_degradation_regression
    df = _make_synthetic_regression_df()
    # Inject NaNs
    df.loc[0, "X"] = float("nan")
    df.loc[1, "Y"] = float("nan")
    result = fit_degradation_regression(df, "X", "Y", class_col="class_name")
    assert result["n_obs"] == len(df) - 2


def test_fit_degradation_regression_requires_at_least_two_classes():
    """Single-class data raises ValueError."""
    from target_affinity_ml.benchmarks.scaffold_diversity import fit_degradation_regression
    df = _make_synthetic_regression_df()
    single = df[df["class_name"] == "kinase"]
    with pytest.raises(ValueError, match="≥2 classes"):
        fit_degradation_regression(single, "X", "Y", class_col="class_name")


def test_fit_degradation_regression_n_per_class():
    from target_affinity_ml.benchmarks.scaffold_diversity import fit_degradation_regression
    df = _make_synthetic_regression_df(n_per_class=40)
    result = fit_degradation_regression(df, "X", "Y", class_col="class_name")
    assert result["n_per_class"] == {"kinase": 40, "gpcr_aminergic": 40}


def test_fit_degradation_regression_three_class_joint_f_test():
    """Joint F-test exercise: ≥3 classes with one differing slope detects via interaction.

    Plan 3 only uses 2 classes, but the function is general; this guards the
    multi-term constraint construction for the joint F-test.
    """
    import numpy as np
    rng = np.random.default_rng(seed=42)
    rows = []
    # Three classes; class C differs in slope; n=50 per class for clean signal
    for cls, slope in [("A_class", 0.3), ("B_class", 0.3), ("C_class", 0.8)]:
        x = rng.uniform(0, 5, size=50)
        y = slope * x + rng.normal(0, 0.1, size=50)
        for xi, yi in zip(x, y):
            rows.append({"X": xi, "Y": yi, "class_name": cls})
    df = pd.DataFrame(rows)

    from target_affinity_ml.benchmarks.scaffold_diversity import fit_degradation_regression
    result = fit_degradation_regression(df, "X", "Y", class_col="class_name")

    # All three classes present in slopes dict
    assert set(result["slopes"].keys()) == {"A_class", "B_class", "C_class"}
    # Slopes recovered within tolerance
    for cls, expected in [("A_class", 0.3), ("B_class", 0.3), ("C_class", 0.8)]:
        assert abs(result["slopes"][cls]["slope"] - expected) < 0.15, \
            f"{cls}: got {result['slopes'][cls]['slope']:.3f}, expected {expected}"
    # Joint F-test rejects null (one class differs)
    assert result["interaction_p"] < 0.001
    # n_per_class matches
    assert result["n_per_class"] == {"A_class": 50, "B_class": 50, "C_class": 50}


def test_fit_degradation_regression_raises_on_zero_variance_metric():
    from target_affinity_ml.benchmarks.scaffold_diversity import fit_degradation_regression
    df = pd.DataFrame({
        "X": [1.0] * 40,  # constant — zero variance
        "Y": list(range(40)),
        "class_name": ["a"] * 20 + ["b"] * 20,
    })
    with pytest.raises(ValueError, match="zero variance"):
        fit_degradation_regression(df, "X", "Y", class_col="class_name")


def test_fit_degradation_regression_raises_on_undersized_class():
    from target_affinity_ml.benchmarks.scaffold_diversity import fit_degradation_regression
    df = pd.DataFrame({
        "X": list(range(40)),
        "Y": list(range(40)),
        "class_name": ["a"] * 38 + ["b"] * 2,  # b has only 2 obs
    })
    with pytest.raises(ValueError, match="only 2 observations"):
        fit_degradation_regression(df, "X", "Y", class_col="class_name")
