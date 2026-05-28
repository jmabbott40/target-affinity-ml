"""Plan 3 pre-registered hypothesis tests H1-H4 + cross-class machinery.

See Plan 3 design spec Section 5 for the test design. Each public function
operates on a per-seed RMSE DataFrame with columns
``{model, class, split, seed, rmse}`` and emits a structured DataFrame
or dict for downstream reporting (notebook 07_cross_class_comparison.ipynb).

Pre-registered hypotheses
-------------------------
- **H1**: RF competitive with deep models on scaffold/target splits
  (paired t-test, Bonferroni across the 12-test family).
- **H2**: Random→scaffold (+12-52%) + scaffold→target (+25-60%) degradation
  per the kinase preprint; class × split interaction F-test.
- **H3**: ESM-2 advantage vanishes on the target split. Per-(class × split)
  advantage values + optional per-target pLDDT regression (T13 machinery).
- **H4**: Single-seed scaffold tests flip with multi-seed; per-class flip
  rates + bootstrap CI on the between-class difference.

Plus :func:`class_split_interaction` — the standalone ``rmse ~ C(class) *
C(split) + C(model)`` F-test, exposed for the notebook.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


__all__ = [
    "h1_rf_vs_deep",
    "h2_split_degradation",
    "h3_esm_target_advantage",
    "h4_single_seed_flip_rate",
    "class_split_interaction",
]


# Pre-registered degradation ranges (kinase preprint replication).
_H2_RANDOM_TO_SCAFFOLD_RANGE = (0.12, 0.52)
_H2_SCAFFOLD_TO_TARGET_RANGE = (0.25, 0.60)

# Pre-registered hypothesis-test family sizes. Bonferroni divides by these
# constants — never by the surviving row count — so a skipped cell can never
# weaken the multiple-comparison correction.
N_TESTS_H1 = 12  # 2 model_pairs × 2 classes × 3 splits
N_TESTS_H3 = 6  # 2 classes × 3 splits

# If |multi_mean_diff| falls below this tolerance, the cell is effectively at
# zero and the flip-rate sign comparison is undefined (sign(0) ambiguity).
_FLIP_RATE_MULTI_MEAN_TOL = 1e-10


# ---------------------------------------------------------------------------
# H1
# ---------------------------------------------------------------------------


def h1_rf_vs_deep(
    per_seed: pd.DataFrame,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """H1: paired t-test ``random_forest`` vs deep models, per (class × split).

    Computes 12 tests = 2 model_pairs × 2 classes × 3 splits. Cohen's d for
    effect size; percentile bootstrap CI for the mean of paired RMSE
    differences across seeds; Bonferroni correction across the full 12-test
    family.

    Parameters
    ----------
    per_seed : pd.DataFrame
        Required columns: ``model``, ``class``, ``split``, ``seed``, ``rmse``.
        Must include ``model`` values ``"random_forest"`` and the two deep
        comparators (``"esm_fp_mlp"`` and ``"fusion"``) for each (class, split)
        with seeds aligned 1:1.
    n_bootstrap : int
        Number of bootstrap resamples for the CI (default 10,000).
    seed : int
        RNG seed for the bootstrap.

    Returns
    -------
    pd.DataFrame
        One row per test with columns
        ``model_pair, class, split, mean_diff, cohens_d, ci_low, ci_high,
        t_stat, p_raw, p_bonferroni, verdict``.

        - ``mean_diff = mean(rmse_rf) - mean(rmse_deep)``; negative means
          RF has lower RMSE (RF wins).
        - ``verdict`` is ``"RF wins"`` (mean_diff < 0 and p_bonferroni <
          0.05), ``"RF loses"`` (mean_diff > 0 and p_bonferroni < 0.05), or
          ``"ties"`` otherwise.

    Notes
    -----
    The 5-seed paired bootstrap CIs reported here have degraded nominal
    coverage at n_seeds < 10; they're reported for transparency rather than
    as strict inference. Use them as ballpark bounds, not formal hypothesis
    tests.
    """
    _validate_per_seed(per_seed)

    deep_models = ["esm_fp_mlp", "fusion"]
    classes = sorted(per_seed["class"].unique())
    splits = sorted(per_seed["split"].unique())

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for deep in deep_models:
        for class_ in classes:
            for split in splits:
                sub = per_seed.query(
                    "(model == @rf or model == @deep) and "
                    "`class` == @class_ and split == @split",
                    local_dict={
                        "rf": "random_forest",
                        "deep": deep,
                        "class_": class_,
                        "split": split,
                    },
                ).copy()
                if sub.empty:
                    continue
                # Align RF & deep on seed
                rf_seeds = sub.loc[sub["model"] == "random_forest"].set_index("seed")["rmse"]
                deep_seeds = sub.loc[sub["model"] == deep].set_index("seed")["rmse"]
                common = sorted(set(rf_seeds.index) & set(deep_seeds.index))
                if len(common) < 2:
                    logger.warning(
                        "h1: <2 common seeds for %s vs %s on (%s, %s); skipping",
                        "random_forest", deep, class_, split,
                    )
                    continue
                rf_vals = rf_seeds.loc[common].to_numpy(dtype=float)
                deep_vals = deep_seeds.loc[common].to_numpy(dtype=float)
                diffs = rf_vals - deep_vals
                mean_diff = float(np.mean(diffs))
                cohens_d = _cohens_d_paired(rf_vals, deep_vals)
                t_stat, p_raw = stats.ttest_rel(rf_vals, deep_vals)
                t_stat = float(t_stat) if not np.isnan(t_stat) else float("nan")
                p_raw = float(p_raw) if not np.isnan(p_raw) else 1.0
                ci_low, ci_high = _bootstrap_ci_mean(diffs, n_bootstrap=n_bootstrap, rng=rng)
                rows.append(
                    {
                        "model_pair": f"random_forest_vs_{deep}",
                        "class": class_,
                        "split": split,
                        "mean_diff": mean_diff,
                        "cohens_d": cohens_d,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "t_stat": t_stat,
                        "p_raw": p_raw,
                    }
                )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df["p_bonferroni"] = (df["p_raw"] * N_TESTS_H1).clip(upper=1.0)
    if len(df) < N_TESTS_H1:
        logger.warning(
            "h1_rf_vs_deep: only %d/%d planned tests had complete data; "
            "Bonferroni still divides by %d per the pre-registered plan",
            len(df), N_TESTS_H1, N_TESTS_H1,
        )
    df["verdict"] = df.apply(_h1_verdict, axis=1)
    return df


def _h1_verdict(row: pd.Series) -> str:
    if row["p_bonferroni"] < 0.05:
        if row["mean_diff"] < 0:
            return "RF wins"
        if row["mean_diff"] > 0:
            return "RF loses"
    return "ties"


# ---------------------------------------------------------------------------
# H2
# ---------------------------------------------------------------------------


def h2_split_degradation(per_seed: pd.DataFrame) -> pd.DataFrame:
    """H2: per (model × class) random→scaffold + scaffold→target degradation.

    For each (model, class), computes:

    - ``random_to_scaffold = (mean_rmse_scaffold - mean_rmse_random) / mean_rmse_random``
    - ``scaffold_to_target = (mean_rmse_target - mean_rmse_scaffold) / mean_rmse_scaffold``

    Means are taken across seeds. A final row carries the class × split
    interaction F-test using ``rmse ~ C(class) * C(split) + C(model)``.

    Parameters
    ----------
    per_seed : pd.DataFrame
        Required columns: ``model, class, split, seed, rmse``. Must contain
        all three splits ``"random"``, ``"scaffold"``, ``"target"`` for each
        (model, class) cell to compute both ratios.

    Returns
    -------
    pd.DataFrame
        Columns: ``model, class, transition, ratio, in_range, verdict``.

        - Per-model rows have ``transition`` in ``{random_to_scaffold,
          scaffold_to_target}``, ``ratio`` the relative degradation, and
          ``in_range`` whether the ratio falls within the pre-registered
          kinase-preprint range.
        - The interaction row has ``model="*"``, ``class="*"``,
          ``transition="class_x_split_interaction"``, ``ratio=NaN``,
          ``in_range=NaN``, and ``verdict`` formatted as
          ``"interaction p=<value>"``.
    """
    _validate_per_seed(per_seed)

    # Mean RMSE per (model, class, split)
    mean_rmse = (
        per_seed.groupby(["model", "class", "split"], as_index=False)["rmse"]
        .mean()
        .pivot_table(
            index=["model", "class"], columns="split", values="rmse"
        )
    )

    rows: list[dict] = []
    for (model, class_), row in mean_rmse.iterrows():
        for transition, lo_split, hi_split, target_range in (
            ("random_to_scaffold", "random", "scaffold", _H2_RANDOM_TO_SCAFFOLD_RANGE),
            ("scaffold_to_target", "scaffold", "target", _H2_SCAFFOLD_TO_TARGET_RANGE),
        ):
            if lo_split not in row.index or hi_split not in row.index:
                continue
            lo = row[lo_split]
            hi = row[hi_split]
            if pd.isna(lo) or pd.isna(hi) or lo == 0:
                ratio = float("nan")
                in_range = False
                verdict = "b) below"
            else:
                ratio = float((hi - lo) / lo)
                lo_bound, hi_bound = target_range
                in_range = bool(lo_bound <= ratio <= hi_bound)
                if in_range:
                    verdict = "a) within range"
                elif ratio < lo_bound:
                    verdict = "b) below"
                else:
                    verdict = "c) above"
            rows.append(
                {
                    "model": model,
                    "class": class_,
                    "transition": transition,
                    "ratio": ratio,
                    "in_range": in_range,
                    "verdict": verdict,
                }
            )

    # Class × split interaction row. ``in_range`` is meaningless for this
    # summary row; we use ``pd.NA`` + nullable BooleanDtype so the column
    # stays bool-typed (the float-NaN alternative coerced the whole column
    # to object dtype).
    inter = class_split_interaction(per_seed)
    rows.append(
        {
            "model": "*",
            "class": "*",
            "transition": "class_x_split_interaction",
            "ratio": float("nan"),
            "in_range": pd.NA,
            "verdict": f"interaction p={inter['interaction_p']:.4g}",
        }
    )
    df = pd.DataFrame(rows)
    df["in_range"] = df["in_range"].astype("boolean")
    return df


# ---------------------------------------------------------------------------
# H3
# ---------------------------------------------------------------------------


def h3_esm_target_advantage(
    per_seed: pd.DataFrame,
    per_target_metrics: pd.DataFrame | None = None,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    """H3: ESM-2 advantage = ``mean(RMSE_ESM_FP_MLP) − mean(RMSE_MLP)``.

    Two-part analysis:

    **Part A** — per (class, split): the seed-averaged ESM-2 advantage with
    a bootstrap CI over per-seed paired differences and Bonferroni
    correction across the six tests (2 classes × 3 splits). A class × split
    interaction F-test summarises whether the pattern differs by class.

    **Part B** (optional) — per-target pLDDT regression of the embedding
    advantage on ``mean_binding_site_plddt`` (or any metric column named on
    the ``per_target_metrics`` frame) via the T13 ``fit_degradation_regression``.

    Parameters
    ----------
    per_seed : pd.DataFrame
        Required columns: ``model, class, split, seed, rmse``. Must include
        ``"esm_fp_mlp"`` and ``"mlp"`` for each (class, split) with seed-aligned
        rows.
    per_target_metrics : pd.DataFrame, optional
        Per-target frame with at least columns ``target_chembl_id``,
        ``class_name``, the metric column (e.g.,
        ``mean_binding_site_plddt``), and ``esm_advantage`` (per-target
        embedding advantage). If ``None``, Part B is skipped.
    n_bootstrap : int
        Bootstrap resamples for the per-(class, split) CI.
    seed : int
        RNG seed.

    Returns
    -------
    dict with keys:

    - ``advantage_values`` : pd.DataFrame
        Columns ``class, split, n_seeds, mean_advantage, ci_low, ci_high,
        p_bonferroni, verdict``. Negative ``mean_advantage`` = ESM-2 helps.
    - ``class_x_split_interaction_p`` : float
        F-test interaction p-value on the per-seed RMSE table restricted to
        ``{esm_fp_mlp, mlp}``.
    - ``plddt_advantage_regression`` : dict or None
        Output of :func:`fit_degradation_regression` if
        ``per_target_metrics`` is provided; else None.
    - ``verdict`` : str
        One of ``"a) ESM-2 same pattern"`` (interaction n.s. and ESM-2 wins
        on target for both classes), ``"b) ESM-2 still helps GPCR target"``
        (interaction n.s. with ESM advantage GPCR-favoured on target), or
        ``"c) ESM-2 never helps GPCRs"`` (otherwise).

    Notes
    -----
    The 5-seed paired bootstrap CIs reported here have degraded nominal
    coverage at n_seeds < 10; they're reported for transparency rather than
    as strict inference. Use them as ballpark bounds, not formal hypothesis
    tests.
    """
    _validate_per_seed(per_seed)

    classes = sorted(per_seed["class"].unique())
    splits = sorted(per_seed["split"].unique())

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for class_ in classes:
        for split in splits:
            sub = per_seed.query(
                "(model == 'esm_fp_mlp' or model == 'mlp') and "
                "`class` == @class_ and split == @split",
                local_dict={"class_": class_, "split": split},
            )
            if sub.empty:
                continue
            esm_seeds = sub.loc[sub["model"] == "esm_fp_mlp"].set_index("seed")["rmse"]
            mlp_seeds = sub.loc[sub["model"] == "mlp"].set_index("seed")["rmse"]
            common = sorted(set(esm_seeds.index) & set(mlp_seeds.index))
            if len(common) < 2:
                logger.warning(
                    "h3: <2 common seeds for esm_fp_mlp vs mlp on (%s, %s); skipping",
                    class_, split,
                )
                continue
            esm_vals = esm_seeds.loc[common].to_numpy(dtype=float)
            mlp_vals = mlp_seeds.loc[common].to_numpy(dtype=float)
            diffs = esm_vals - mlp_vals
            mean_adv = float(np.mean(diffs))
            t_stat, p_raw = stats.ttest_rel(esm_vals, mlp_vals)
            p_raw = float(p_raw) if not np.isnan(p_raw) else 1.0
            ci_low, ci_high = _bootstrap_ci_mean(diffs, n_bootstrap=n_bootstrap, rng=rng)
            rows.append(
                {
                    "class": class_,
                    "split": split,
                    "n_seeds": len(common),
                    "mean_advantage": mean_adv,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_raw": p_raw,
                }
            )

    av = pd.DataFrame(rows)
    if len(av) > 0:
        av["p_bonferroni"] = (av["p_raw"] * N_TESTS_H3).clip(upper=1.0)
        if len(av) < N_TESTS_H3:
            logger.warning(
                "h3_esm_target_advantage: only %d/%d planned tests had complete "
                "data; Bonferroni still divides by %d per the pre-registered plan",
                len(av), N_TESTS_H3, N_TESTS_H3,
            )
        av["verdict"] = av.apply(_h3_advantage_verdict, axis=1)
        av = av.drop(columns=["p_raw"])

    # Class × split interaction restricted to {esm_fp_mlp, mlp}
    sub_interaction = per_seed[per_seed["model"].isin(["esm_fp_mlp", "mlp"])]
    if not sub_interaction.empty:
        inter = class_split_interaction(sub_interaction)
        interaction_p = float(inter["interaction_p"])
    else:
        interaction_p = float("nan")

    # Optional Part B: pLDDT regression
    plddt_reg = None
    if per_target_metrics is not None and len(per_target_metrics) > 0:
        plddt_reg = _h3_plddt_regression(per_target_metrics)

    verdict = _h3_overall_verdict(av, interaction_p)

    return {
        "advantage_values": av,
        "class_x_split_interaction_p": interaction_p,
        "plddt_advantage_regression": plddt_reg,
        "verdict": verdict,
    }


def _h3_advantage_verdict(row: pd.Series) -> str:
    """Per-row verdict for the advantage_values frame."""
    if row["p_bonferroni"] >= 0.05:
        return "ties"
    return "ESM-2 helps" if row["mean_advantage"] < 0 else "ESM-2 hurts"


def _h3_overall_verdict(av: pd.DataFrame, interaction_p: float) -> str:
    """Spec verdict: (a) same pattern, (b) GPCR target still helps, (c) never helps GPCRs."""
    if av.empty:
        return "c) ESM-2 never helps GPCRs"
    gpcr_target = av.query("`class` == 'gpcr_aminergic' and split == 'target'")
    kinase_target = av.query("`class` == 'kinase' and split == 'target'")
    gpcr_helps_target = (
        not gpcr_target.empty and gpcr_target.iloc[0]["mean_advantage"] < 0
    )
    kinase_helps_target = (
        not kinase_target.empty and kinase_target.iloc[0]["mean_advantage"] < 0
    )

    if not np.isnan(interaction_p) and interaction_p >= 0.05:
        return "a) ESM-2 same pattern"
    if gpcr_helps_target and not kinase_helps_target:
        return "b) ESM-2 still helps GPCR target"
    if not gpcr_helps_target:
        return "c) ESM-2 never helps GPCRs"
    return "a) ESM-2 same pattern"


def _h3_plddt_regression(per_target_metrics: pd.DataFrame) -> dict:
    """Run the per-target pLDDT vs ESM-advantage regression via T13."""
    from target_affinity_ml.benchmarks.scaffold_diversity import fit_degradation_regression

    # Auto-detect the pLDDT metric column (prefer the canonical name).
    metric_candidates = [
        "mean_binding_site_plddt",
        "plddt",
        "mean_plddt",
    ]
    metric_col = None
    for cand in metric_candidates:
        if cand in per_target_metrics.columns:
            metric_col = cand
            break
    if metric_col is None:
        raise ValueError(
            "per_target_metrics must contain a pLDDT column "
            f"(one of {metric_candidates}); got {list(per_target_metrics.columns)}"
        )
    if "esm_advantage" not in per_target_metrics.columns:
        raise ValueError(
            "per_target_metrics must contain an 'esm_advantage' column "
            "(per-target ESM-FP-MLP RMSE minus MLP RMSE)"
        )
    return fit_degradation_regression(
        per_target_metrics,
        metric_col=metric_col,
        degradation_col="esm_advantage",
        class_col="class_name",
    )


# ---------------------------------------------------------------------------
# H4
# ---------------------------------------------------------------------------


def h4_single_seed_flip_rate(
    per_seed: pd.DataFrame,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    flip_diff_tol: float = 0.05,
) -> pd.DataFrame:
    """H4: rate at which single-seed verdicts flip from the multi-seed mean.

    For each (model_pair × split), counts the fraction of (class, seed)
    cells where ``sign(rmse_a − rmse_b)_seed ≠ sign(mean_diff)``. Per-class
    rates + their between-class difference + bootstrap CI on the difference.
    The verdict is derived from the size of ``diff`` relative to
    ``flip_diff_tol``; the bootstrap CI on ``diff`` is the primary inference
    on whether the kinase and GPCR flip rates differ.

    Parameters
    ----------
    per_seed : pd.DataFrame
        Required columns: ``model, class, split, seed, rmse``.
    n_bootstrap : int
        Bootstrap resamples for the CI of the between-class difference.
    seed : int
        RNG seed.
    flip_diff_tol : float
        Threshold for the ``b) lower`` / ``c) higher`` verdict on the
        between-class difference (default 0.05 absolute difference).

    Returns
    -------
    pd.DataFrame
        Columns: ``model_pair, split, flip_rate_kinase, flip_rate_gpcr,
        diff, ci_low, ci_high, verdict``.

    Notes
    -----
    The 5-seed paired bootstrap CIs reported here have degraded nominal
    coverage at n_seeds < 10; they're reported for transparency rather than
    as strict inference. Use them as ballpark bounds, not formal hypothesis
    tests.
    """
    _validate_per_seed(per_seed)

    deep_models = ["esm_fp_mlp", "fusion"]
    classes = sorted(per_seed["class"].unique())
    splits = sorted(per_seed["split"].unique())

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for deep in deep_models:
        for split in splits:
            class_rates: dict[str, float] = {}
            class_flip_arrays: dict[str, np.ndarray] = {}
            for class_ in classes:
                sub = per_seed.query(
                    "(model == @rf or model == @deep) and `class` == @class_ and split == @split",
                    local_dict={
                        "rf": "random_forest",
                        "deep": deep,
                        "class_": class_,
                        "split": split,
                    },
                )
                if sub.empty:
                    class_rates[class_] = float("nan")
                    class_flip_arrays[class_] = np.array([], dtype=float)
                    continue
                rf_seeds = sub.loc[sub["model"] == "random_forest"].set_index("seed")["rmse"]
                deep_seeds = sub.loc[sub["model"] == deep].set_index("seed")["rmse"]
                common = sorted(set(rf_seeds.index) & set(deep_seeds.index))
                if not common:
                    class_rates[class_] = float("nan")
                    class_flip_arrays[class_] = np.array([], dtype=float)
                    continue
                rf_vals = rf_seeds.loc[common].to_numpy(dtype=float)
                deep_vals = deep_seeds.loc[common].to_numpy(dtype=float)
                diffs = rf_vals - deep_vals
                multi_mean_diff = float(np.mean(diffs))
                if abs(multi_mean_diff) < _FLIP_RATE_MULTI_MEAN_TOL:
                    # Cell is too close to zero; flip rate is undefined
                    # (every nonzero per-seed sign would register as a flip).
                    class_rates[class_] = float("nan")
                    class_flip_arrays[class_] = np.array([], dtype=float)
                    continue
                multi_sign = np.sign(multi_mean_diff)
                per_seed_signs = np.sign(diffs)
                flips = (per_seed_signs != multi_sign).astype(float)
                class_flip_arrays[class_] = flips
                class_rates[class_] = float(flips.mean()) if len(flips) else float("nan")

            kinase_rate = class_rates.get("kinase", float("nan"))
            gpcr_rate = class_rates.get("gpcr_aminergic", float("nan"))
            diff = kinase_rate - gpcr_rate

            # Bootstrap CI on the difference of mean flip-rates. This CI
            # is the primary inference on whether the two class flip rates
            # differ; we deliberately do not report a binomial p_value here
            # (the GPCR rate is itself a noisy 5-seed estimate, not a fixed
            # null probability).
            kinase_flips = class_flip_arrays.get("kinase", np.array([], dtype=float))
            gpcr_flips = class_flip_arrays.get("gpcr_aminergic", np.array([], dtype=float))
            ci_low, ci_high = _bootstrap_ci_diff_of_means(
                kinase_flips, gpcr_flips, n_bootstrap=n_bootstrap, rng=rng
            )

            if math.isnan(diff):
                verdict = "a) similar"
            elif diff > flip_diff_tol:
                verdict = "c) higher"
            elif diff < -flip_diff_tol:
                verdict = "b) lower"
            else:
                verdict = "a) similar"

            rows.append(
                {
                    "model_pair": f"random_forest_vs_{deep}",
                    "split": split,
                    "flip_rate_kinase": kinase_rate,
                    "flip_rate_gpcr": gpcr_rate,
                    "diff": diff,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "verdict": verdict,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# class_split_interaction (public)
# ---------------------------------------------------------------------------


def class_split_interaction(per_seed: pd.DataFrame) -> dict:
    """``rmse ~ C(class) * C(split) + C(model)`` interaction F-test.

    Parameters
    ----------
    per_seed : pd.DataFrame
        Required columns: ``model, class, split, rmse``. (``seed`` is allowed
        but ignored.)

    Returns
    -------
    dict
        Keys: ``interaction_f_stat, interaction_p, df_resid, r_squared,
        formula, n_obs``.
    """
    _validate_per_seed(per_seed, require_seed=False)
    import statsmodels.formula.api as smf

    df = per_seed.rename(columns={"class": "class_"})  # avoid Python keyword in formula
    formula = "rmse ~ C(class_) * C(split) + C(model)"
    model = smf.ols(formula, data=df).fit()

    interaction_terms = [
        k for k in model.params.index
        if k.startswith("C(class_)[") and ":C(split)[" in k
    ]
    if interaction_terms:
        constraint = ", ".join(f"{t} = 0" for t in interaction_terms)
        ft = model.f_test(constraint)
        f_stat = float(np.atleast_1d(ft.fvalue).ravel()[0])
        p_value = float(np.atleast_1d(ft.pvalue).ravel()[0])
    else:
        f_stat = float("nan")
        p_value = float("nan")

    return {
        "interaction_f_stat": f_stat,
        "interaction_p": p_value,
        "df_resid": int(model.df_resid),
        "r_squared": float(model.rsquared),
        "formula": formula.replace("class_", "class"),
        "n_obs": int(model.nobs),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_per_seed(per_seed: pd.DataFrame, require_seed: bool = True) -> None:
    """Defensive check on the per-seed input frame."""
    if not isinstance(per_seed, pd.DataFrame):
        raise TypeError(f"per_seed must be a DataFrame; got {type(per_seed).__name__}")
    required = {"model", "class", "split", "rmse"}
    if require_seed:
        required = required | {"seed"}
    missing = required - set(per_seed.columns)
    if missing:
        raise ValueError(
            f"per_seed missing required columns: {sorted(missing)}; "
            f"got {list(per_seed.columns)}"
        )


def _bootstrap_ci_mean(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of ``values``.

    Returns ``(NaN, NaN)`` if ``values`` is empty.

    Notes
    -----
    The 5-seed paired bootstrap CIs computed by the H1/H3/H4 callers have
    degraded nominal coverage at n_seeds < 10; they're reported for
    transparency rather than as strict inference. Use them as ballpark
    bounds, not formal hypothesis tests.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    n = len(values)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def _bootstrap_ci_diff_of_means(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap CI on ``mean(a) - mean(b)`` with independent resampling."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    idx_a = rng.integers(0, len(a), size=(n_bootstrap, len(a)))
    idx_b = rng.integers(0, len(b), size=(n_bootstrap, len(b)))
    diffs = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return lo, hi


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples = mean(diff) / std(diff, ddof=1).

    Returns 0.0 if the diff has zero variance.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(diff) < 2:
        return 0.0
    s = float(diff.std(ddof=1))
    if s == 0.0 or math.isnan(s):
        return 0.0
    return float(diff.mean() / s)
