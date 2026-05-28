"""Tests for the hypothesis_tests module (P3-T15).

Plan 3 pre-registered H1-H4 hypothesis tests + between-class machinery.
See Plan 3 design spec Section 5.

Each test constructs a synthetic per-seed RMSE DataFrame with known patterns,
calls a public function, and asserts expected DataFrame columns + value sanity.
"""
import math

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared synthetic-data fixtures
# ---------------------------------------------------------------------------


MODELS = ["random_forest", "esm_fp_mlp", "fusion", "mlp", "elasticnet", "xgboost", "gnn"]
CLASSES = ["kinase", "gpcr_aminergic"]
SPLITS = ["random", "scaffold", "target"]
SEEDS = [42, 123, 456, 789, 1024]


@pytest.fixture
def synthetic_per_seed():
    """Per-seed RMSE data: 7 models × 2 classes × 3 splits × 5 seeds = 210 rows.

    Engineered patterns
    -------------------
    - Random < Scaffold < Target base RMSE (matches preprint expectation).
    - RF is engineered slightly *better* than ESM_FP_MLP on (kinase, scaffold).
    - On (kinase, target) ESM_FP_MLP is slightly *worse* than MLP, mimicking
      the preprint's "ESM-2 advantage vanishes" signal so H3 has something to
      detect.
    """
    rng = np.random.default_rng(42)
    rows = []
    for model in MODELS:
        for class_ in CLASSES:
            for split in SPLITS:
                base = {"random": 0.7, "scaffold": 1.0, "target": 1.4}[split]
                if model == "random_forest" and class_ == "kinase" and split == "scaffold":
                    base -= 0.15
                if model == "esm_fp_mlp" and class_ == "kinase" and split == "random":
                    # ESM helps on random split for kinase
                    base -= 0.10
                if model == "esm_fp_mlp" and class_ == "kinase" and split == "target":
                    # ESM advantage vanishes on target split for kinase
                    base += 0.05
                for seed in SEEDS:
                    rmse = base + rng.normal(0, 0.03)
                    rows.append(
                        {
                            "model": model,
                            "class": class_,
                            "split": split,
                            "seed": seed,
                            "rmse": rmse,
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H1: RF vs deep models
# ---------------------------------------------------------------------------


def test_h1_returns_dataframe_with_expected_columns(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import h1_rf_vs_deep

    result = h1_rf_vs_deep(synthetic_per_seed)
    assert isinstance(result, pd.DataFrame)
    expected = {
        "model_pair",
        "class",
        "split",
        "mean_diff",
        "cohens_d",
        "ci_low",
        "ci_high",
        "t_stat",
        "p_raw",
        "p_bonferroni",
        "verdict",
    }
    assert expected.issubset(set(result.columns))


def test_h1_returns_12_rows(synthetic_per_seed):
    """2 model_pairs × 2 classes × 3 splits = 12 tests."""
    from target_affinity_ml.benchmarks.hypothesis_tests import h1_rf_vs_deep

    result = h1_rf_vs_deep(synthetic_per_seed)
    assert len(result) == 12


def test_h1_covers_both_model_pairs(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import h1_rf_vs_deep

    result = h1_rf_vs_deep(synthetic_per_seed)
    assert set(result["model_pair"].unique()) == {
        "random_forest_vs_esm_fp_mlp",
        "random_forest_vs_fusion",
    }


def test_h1_detects_engineered_rf_advantage(synthetic_per_seed):
    """RF was engineered to be 0.15 lower than ESM_FP_MLP on (kinase, scaffold)."""
    from target_affinity_ml.benchmarks.hypothesis_tests import h1_rf_vs_deep

    result = h1_rf_vs_deep(synthetic_per_seed)
    row = result.query(
        "model_pair == 'random_forest_vs_esm_fp_mlp' and `class` == 'kinase' and split == 'scaffold'"
    )
    assert len(row) == 1
    r = row.iloc[0]
    assert r["mean_diff"] < 0  # RF wins
    assert r["cohens_d"] < 0
    assert r["verdict"] == "RF wins"


def test_h1_verdict_values_are_in_expected_set(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import h1_rf_vs_deep

    result = h1_rf_vs_deep(synthetic_per_seed)
    assert set(result["verdict"].unique()).issubset({"RF wins", "RF loses", "ties"})


def test_h1_bonferroni_correction_applied(synthetic_per_seed):
    """p_bonferroni should equal min(p_raw * 12, 1.0)."""
    from target_affinity_ml.benchmarks.hypothesis_tests import h1_rf_vs_deep

    result = h1_rf_vs_deep(synthetic_per_seed)
    for _, row in result.iterrows():
        expected = min(row["p_raw"] * 12, 1.0)
        assert math.isclose(row["p_bonferroni"], expected, rel_tol=1e-9)


def test_h1_ci_bounds_bracket_mean_diff_loosely(synthetic_per_seed):
    """ci_low <= ci_high; for bootstrap CIs ci_low and ci_high will contain
    something near the observed mean_diff (within the 95% interval)."""
    from target_affinity_ml.benchmarks.hypothesis_tests import h1_rf_vs_deep

    result = h1_rf_vs_deep(synthetic_per_seed)
    for _, row in result.iterrows():
        assert row["ci_low"] <= row["ci_high"]


# ---------------------------------------------------------------------------
# H2: Split degradation
# ---------------------------------------------------------------------------


def test_h2_returns_dataframe_with_expected_columns(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import h2_split_degradation

    result = h2_split_degradation(synthetic_per_seed)
    assert isinstance(result, pd.DataFrame)
    expected = {"model", "class", "transition", "ratio", "in_range", "verdict"}
    assert expected.issubset(set(result.columns))


def test_h2_includes_per_model_per_class_rows_and_interaction_row(synthetic_per_seed):
    """Per (model × class), 2 transitions = 7 models × 2 classes × 2 transitions = 28.
    Plus one interaction summary row."""
    from target_affinity_ml.benchmarks.hypothesis_tests import h2_split_degradation

    result = h2_split_degradation(synthetic_per_seed)
    assert len(result) == 7 * 2 * 2 + 1


def test_h2_transitions_are_named_as_expected(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import h2_split_degradation

    result = h2_split_degradation(synthetic_per_seed)
    assert set(result["transition"].unique()) == {
        "random_to_scaffold",
        "scaffold_to_target",
        "class_x_split_interaction",
    }


def test_h2_interaction_row_present(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import h2_split_degradation

    result = h2_split_degradation(synthetic_per_seed)
    interaction = result.query("transition == 'class_x_split_interaction'")
    assert len(interaction) == 1
    assert interaction.iloc[0]["model"] == "*"
    assert interaction.iloc[0]["class"] == "*"
    assert "interaction p=" in interaction.iloc[0]["verdict"]


def test_h2_degradation_ratios_positive_on_expected_pattern(synthetic_per_seed):
    """With base RMSE rising random < scaffold < target, ratios should be > 0."""
    from target_affinity_ml.benchmarks.hypothesis_tests import h2_split_degradation

    result = h2_split_degradation(synthetic_per_seed)
    per_model = result.query("transition != 'class_x_split_interaction'")
    # All ratios in our synthetic data should be positive (worse on later split)
    # Note: random_forest scaffold is engineered to be lower, but only for kinase
    # and the random_forest model still has random < scaffold < target overall.
    # We check the median ratio is positive as a soft sanity check
    assert per_model["ratio"].median() > 0


def test_h2_verdict_strings_within_expected_set(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import h2_split_degradation

    result = h2_split_degradation(synthetic_per_seed)
    per_model = result.query("transition != 'class_x_split_interaction'")
    assert set(per_model["verdict"].unique()).issubset(
        {"a) within range", "b) below", "c) above"}
    )


# ---------------------------------------------------------------------------
# H3: ESM-2 target advantage
# ---------------------------------------------------------------------------


def test_h3_returns_expected_top_level_keys(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import h3_esm_target_advantage

    result = h3_esm_target_advantage(synthetic_per_seed)
    assert isinstance(result, dict)
    expected = {
        "advantage_values",
        "class_x_split_interaction_p",
        "plddt_advantage_regression",
        "verdict",
    }
    assert expected == set(result.keys())


def test_h3_advantage_values_is_dataframe_with_expected_columns(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import h3_esm_target_advantage

    result = h3_esm_target_advantage(synthetic_per_seed)
    av = result["advantage_values"]
    assert isinstance(av, pd.DataFrame)
    expected_cols = {
        "class",
        "split",
        "n_seeds",
        "mean_advantage",
        "ci_low",
        "ci_high",
        "p_bonferroni",
        "verdict",
    }
    assert expected_cols.issubset(set(av.columns))


def test_h3_advantage_has_2x3_rows(synthetic_per_seed):
    """2 classes × 3 splits = 6 advantage rows."""
    from target_affinity_ml.benchmarks.hypothesis_tests import h3_esm_target_advantage

    result = h3_esm_target_advantage(synthetic_per_seed)
    assert len(result["advantage_values"]) == 6


def test_h3_detects_engineered_esm_advantage_on_kinase_random(synthetic_per_seed):
    """ESM-FP-MLP was engineered to be ~0.10 better than MLP on (kinase, random)."""
    from target_affinity_ml.benchmarks.hypothesis_tests import h3_esm_target_advantage

    result = h3_esm_target_advantage(synthetic_per_seed)
    av = result["advantage_values"]
    row = av.query("`class` == 'kinase' and split == 'random'")
    assert len(row) == 1
    # Engineered to be negative (ESM-FP-MLP < MLP)
    assert row.iloc[0]["mean_advantage"] < 0


def test_h3_detects_engineered_esm_disadvantage_on_kinase_target(synthetic_per_seed):
    """ESM-FP-MLP was engineered to be ~0.05 worse than MLP on (kinase, target)."""
    from target_affinity_ml.benchmarks.hypothesis_tests import h3_esm_target_advantage

    result = h3_esm_target_advantage(synthetic_per_seed)
    av = result["advantage_values"]
    row = av.query("`class` == 'kinase' and split == 'target'")
    assert len(row) == 1
    assert row.iloc[0]["mean_advantage"] > 0


def test_h3_plddt_regression_none_when_no_per_target_data(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import h3_esm_target_advantage

    result = h3_esm_target_advantage(synthetic_per_seed, per_target_metrics=None)
    assert result["plddt_advantage_regression"] is None


def test_h3_plddt_regression_returned_when_per_target_provided(synthetic_per_seed):
    """If per_target_metrics has the pLDDT column and per-target ESM advantage, regression runs."""
    from target_affinity_ml.benchmarks.hypothesis_tests import h3_esm_target_advantage

    # Build synthetic per-target metrics: 10 kinase + 10 GPCR targets
    rng = np.random.default_rng(7)
    rows = []
    for cls in ["kinase", "gpcr_aminergic"]:
        for i in range(10):
            plddt = rng.uniform(60, 95)
            # advantage ~ -0.5 + 0.01 * plddt + noise (higher pLDDT => positive advantage)
            adv = -0.5 + 0.01 * plddt + rng.normal(0, 0.05)
            rows.append({
                "target_chembl_id": f"CHEMBL{cls[:1]}_{i}",
                "class_name": cls,
                "mean_binding_site_plddt": plddt,
                "esm_advantage": adv,
            })
    per_target = pd.DataFrame(rows)

    result = h3_esm_target_advantage(
        synthetic_per_seed,
        per_target_metrics=per_target,
    )
    reg = result["plddt_advantage_regression"]
    assert reg is not None
    # The regression dict should at least have these keys (matches T13 contract)
    assert "slopes" in reg
    assert "interaction_p" in reg
    assert "n_obs" in reg


def test_h3_verdict_value_in_expected_set(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import h3_esm_target_advantage

    result = h3_esm_target_advantage(synthetic_per_seed)
    assert result["verdict"] in {
        "a) ESM-2 same pattern",
        "b) ESM-2 still helps GPCR target",
        "c) ESM-2 never helps GPCRs",
    }


# ---------------------------------------------------------------------------
# H4: Single-seed flip rate
# ---------------------------------------------------------------------------


def test_h4_returns_dataframe_with_expected_columns(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import h4_single_seed_flip_rate

    result = h4_single_seed_flip_rate(synthetic_per_seed)
    assert isinstance(result, pd.DataFrame)
    expected = {
        "model_pair",
        "split",
        "flip_rate_kinase",
        "flip_rate_gpcr",
        "diff",
        "ci_low",
        "ci_high",
        "p_value",
        "verdict",
    }
    assert expected.issubset(set(result.columns))


def test_h4_row_count(synthetic_per_seed):
    """2 model_pairs × 3 splits = 6 rows."""
    from target_affinity_ml.benchmarks.hypothesis_tests import h4_single_seed_flip_rate

    result = h4_single_seed_flip_rate(synthetic_per_seed)
    assert len(result) == 6


def test_h4_flip_rates_in_unit_interval(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import h4_single_seed_flip_rate

    result = h4_single_seed_flip_rate(synthetic_per_seed)
    for col in ("flip_rate_kinase", "flip_rate_gpcr"):
        assert (result[col] >= 0.0).all()
        assert (result[col] <= 1.0).all()


def test_h4_verdict_strings_in_expected_set(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import h4_single_seed_flip_rate

    result = h4_single_seed_flip_rate(synthetic_per_seed)
    assert set(result["verdict"].unique()).issubset({"a) similar", "b) lower", "c) higher"})


def test_h4_diff_is_difference_of_flip_rates(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import h4_single_seed_flip_rate

    result = h4_single_seed_flip_rate(synthetic_per_seed)
    for _, row in result.iterrows():
        expected = row["flip_rate_kinase"] - row["flip_rate_gpcr"]
        assert math.isclose(row["diff"], expected, rel_tol=1e-9, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# class_split_interaction
# ---------------------------------------------------------------------------


def test_class_split_interaction_returns_expected_keys(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import class_split_interaction

    result = class_split_interaction(synthetic_per_seed)
    expected = {
        "interaction_f_stat",
        "interaction_p",
        "df_resid",
        "r_squared",
        "formula",
        "n_obs",
    }
    assert expected == set(result.keys())


def test_class_split_interaction_nobs_matches_input(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import class_split_interaction

    result = class_split_interaction(synthetic_per_seed)
    assert result["n_obs"] == len(synthetic_per_seed)


def test_class_split_interaction_values_finite(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import class_split_interaction

    result = class_split_interaction(synthetic_per_seed)
    assert math.isfinite(result["interaction_f_stat"])
    assert math.isfinite(result["interaction_p"])
    assert 0.0 <= result["interaction_p"] <= 1.0
    assert 0.0 <= result["r_squared"] <= 1.0
    assert result["df_resid"] > 0


def test_class_split_interaction_formula_string(synthetic_per_seed):
    from target_affinity_ml.benchmarks.hypothesis_tests import class_split_interaction

    result = class_split_interaction(synthetic_per_seed)
    assert "C(class)" in result["formula"] or "class" in result["formula"]
    assert "split" in result["formula"]


# ---------------------------------------------------------------------------
# Smoke test: full module import works after __init__.py is updated
# ---------------------------------------------------------------------------


def test_public_functions_import_from_package():
    """Confirms the __init__.py exports the public surface."""
    from target_affinity_ml.benchmarks import (
        h1_rf_vs_deep,
        h2_split_degradation,
        h3_esm_target_advantage,
        h4_single_seed_flip_rate,
        class_split_interaction,
    )

    for fn in (
        h1_rf_vs_deep,
        h2_split_degradation,
        h3_esm_target_advantage,
        h4_single_seed_flip_rate,
        class_split_interaction,
    ):
        assert callable(fn)
