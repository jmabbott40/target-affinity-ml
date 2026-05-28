"""Per-target + per-class scaffold-diversity metrics for cross-class benchmarks.

See Plan 3 design spec Section 4 for the methodology. Per-target metrics include
Bemis-Murcko scaffold counts, Shannon entropy of the scaffold distribution, the
largest-cluster fraction, mean pairwise Morgan-FP Tanimoto similarity, and an
activity-cliff frequency.

CRITICAL RDKIT IDIOM: scaffold extraction uses ``MurckoScaffold.GetScaffoldForMol(mol)``
followed by ``Chem.MolToSmiles(scaff)`` (the corrected Plan 2 Task 8 idiom).
Do NOT use ``MurckoScaffold.MolToSmiles`` directly — it returns empty strings.
"""
from __future__ import annotations

import math
import random
from collections import Counter

import pandas as pd


def compute_scaffold_metrics(
    df: pd.DataFrame,
    target_col: str = "target_chembl_id",
    smiles_col: str = "canonical_smiles",
    pairwise_sample_size: int = 500,
    activity_col: str | None = "pactivity",
) -> pd.DataFrame:
    """Per-target scaffold-diversity metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input table with one row per compound-target measurement.
    target_col : str
        Column identifying the target (groupby key).
    smiles_col : str
        Column with canonical SMILES strings.
    pairwise_sample_size : int
        Maximum number of compound pairs to sample for mean-Tanimoto and
        activity-cliff calculations. If the target has fewer pairs than this,
        all pairs are used.
    activity_col : str or None
        Column with pActivity values used to compute activity-cliff frequency.
        If ``None`` or absent, no activity-cliff column is added.

    Returns
    -------
    pd.DataFrame
        One row per target with columns: ``target_chembl_id``, ``n_compounds``,
        ``n_scaffolds``, ``scaffold_entropy``, ``largest_cluster_fraction``,
        ``mean_tanimoto``, and (optionally) ``activity_cliff_frequency``.
    """
    rows = []
    for tid, sub in df.groupby(target_col):
        scaffolds = _bemis_murcko_scaffolds(sub[smiles_col])
        counts = Counter(scaffolds)
        n_comp = len(sub)
        n_scaff = len(counts)
        entropy = _shannon_entropy(list(counts.values()))
        lcf = max(counts.values()) / n_comp if n_comp else float("nan")
        mt = _mean_tanimoto(sub[smiles_col].tolist(), sample_size=pairwise_sample_size)
        row = {
            "target_chembl_id": tid,
            "n_compounds": n_comp,
            "n_scaffolds": n_scaff,
            "scaffold_entropy": entropy,
            "largest_cluster_fraction": lcf,
            "mean_tanimoto": mt,
        }
        if activity_col is not None and activity_col in sub.columns:
            row["activity_cliff_frequency"] = _activity_cliff_frequency(
                sub, smiles_col, activity_col
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_class_aggregates(
    per_target: pd.DataFrame, metric_cols: list[str]
) -> dict[str, dict[str, float]]:
    """Per-class aggregates (mean, median, IQR, n) across targets.

    Parameters
    ----------
    per_target : pd.DataFrame
        Per-target metrics dataframe (typically the output of
        :func:`compute_scaffold_metrics`).
    metric_cols : list[str]
        Columns to aggregate.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{metric_name: {"mean": ..., "median": ..., "iqr": ..., "n": ...}}``.
        NaN values are dropped before aggregation.
    """
    out: dict[str, dict[str, float]] = {}
    for col in metric_cols:
        s = per_target[col].dropna()
        out[col] = {
            "mean": float(s.mean()) if len(s) else float("nan"),
            "median": float(s.median()) if len(s) else float("nan"),
            "iqr": float(s.quantile(0.75) - s.quantile(0.25)) if len(s) else float("nan"),
            "n": int(len(s)),
        }
    return out


def fit_degradation_regression(
    per_target: pd.DataFrame,
    metric_col: str,
    degradation_col: str,
    class_col: str = "class_name",
    alpha: float = 0.05,
) -> dict:
    """Fit per-target degradation ~ metric * class_col OLS regression.

    Pools both classes; estimates class-specific slopes via the class × metric
    interaction term. Returns a structured dict with the per-class slopes,
    confidence intervals, R², and the interaction-term p-value (the cross-class
    test of whether the metric-degradation relationship differs by class).

    Parameters
    ----------
    per_target : pd.DataFrame
        One row per target with at least columns ``metric_col``,
        ``degradation_col``, ``class_col``.
    metric_col : str
        Per-target scaffold-diversity metric (e.g., "scaffold_entropy").
    degradation_col : str
        Per-target degradation (e.g., "delta_rmse_random_to_scaffold").
    class_col : str
        Column with class label (e.g., "class_name"). Must have ≥2 distinct levels.
    alpha : float
        Significance level for confidence intervals (default 0.05 → 95% CI).

    Returns
    -------
    dict with keys:
      "formula"             : str — the OLS formula used
      "n_obs"               : int — observations after NaN-drop
      "r_squared"           : float
      "r_squared_adj"       : float
      "intercept"           : float
      "intercept_p"         : float
      "slopes"              : dict[class_value, dict] — per-class slope + CI + p-value
                              keys per class: "slope", "ci_low", "ci_high", "p_value"
      "interaction_p"       : float — p-value of the (metric × class) interaction term
                              (the cross-class slope-difference test)
      "n_per_class"         : dict[class_value, int]
      "alpha"               : float (passed through)
    """
    import statsmodels.formula.api as smf

    df = per_target.dropna(subset=[metric_col, degradation_col, class_col]).copy()
    classes = sorted(df[class_col].unique())
    if len(classes) < 2:
        raise ValueError(
            f"fit_degradation_regression requires ≥2 classes in {class_col!r}; "
            f"got {len(classes)}: {classes}"
        )

    # Rename columns to safe identifiers for the statsmodels formula parser.
    df = df.rename(columns={metric_col: "_X", degradation_col: "_Y", class_col: "_C"})
    formula = "_Y ~ _X * C(_C)"
    model = smf.ols(formula, data=df).fit()

    # Per-class slopes: with Treatment-coded categorical, the reference class
    # slope is the coefficient on "_X". Each non-reference class's slope is
    # "_X" + "_X:C(_C)[T.<class>]" interaction. statsmodels picks the first
    # sorted level as the reference by default.
    ref_class = classes[0]
    main_slope = float(model.params["_X"])
    conf = model.conf_int(alpha=alpha)
    main_slope_ci_low = float(conf.loc["_X", 0])
    main_slope_ci_high = float(conf.loc["_X", 1])
    main_slope_p = float(model.pvalues["_X"])

    slopes: dict = {
        ref_class: {
            "slope": main_slope,
            "ci_low": main_slope_ci_low,
            "ci_high": main_slope_ci_high,
            "p_value": main_slope_p,
        }
    }

    # Build per-class slopes for non-reference classes via the linear-combination
    # of "_X" + interaction term using model.t_test (gives proper CI + p-value).
    for cls in classes[1:]:
        interaction_key = f"_X:C(_C)[T.{cls}]"
        contrast = f"_X + {interaction_key} = 0"
        tt = model.t_test(contrast)
        slope_cls = float(tt.effect[0])
        ci_low, ci_high = (float(x) for x in tt.conf_int(alpha=alpha)[0])
        slopes[cls] = {
            "slope": slope_cls,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": float(tt.pvalue),
        }

    # Interaction p-value: joint F-test across all interaction terms (slope
    # differences from the reference class).
    interaction_terms = [k for k in model.params.index if k.startswith("_X:C(_C)")]
    if interaction_terms:
        constraint = ", ".join(f"{t} = 0" for t in interaction_terms)
        ft = model.f_test(constraint)
        interaction_p = float(ft.pvalue)
    else:
        interaction_p = float("nan")

    return {
        "formula": formula,
        "n_obs": int(model.nobs),
        "r_squared": float(model.rsquared),
        "r_squared_adj": float(model.rsquared_adj),
        "intercept": float(model.params["Intercept"]),
        "intercept_p": float(model.pvalues["Intercept"]),
        "slopes": slopes,
        "interaction_p": interaction_p,
        "n_per_class": {cls: int((df["_C"] == cls).sum()) for cls in classes},
        "alpha": alpha,
    }


def _bemis_murcko_scaffolds(smiles: pd.Series) -> list[str]:
    """Compute Bemis-Murcko scaffolds via the corrected Plan 2 Task 8 idiom.

    Uses ``MurckoScaffold.GetScaffoldForMol(mol)`` followed by
    ``Chem.MolToSmiles(scaff)``. For acyclic molecules,
    ``MurckoScaffold.GetScaffoldForMol`` returns a non-None empty ``Mol`` whose
    canonical SMILES is the empty string ``""`` — we tag these ``"NO_SCAFFOLD"``
    so acyclics share a single bucket (aligns with ``data/splits.py:139``).
    Invalid SMILES are tagged ``"INVALID"`` so they still occupy a counter bin
    without crashing aggregation downstream.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    out: list[str] = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            out.append("INVALID")
            continue
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(scaff) if scaff is not None else ""
        out.append(smi if smi else "NO_SCAFFOLD")
    return out


def _shannon_entropy(counts: list[int]) -> float:
    """Shannon entropy (nats) of a list of nonnegative counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts if c > 0)


def _mean_tanimoto(smiles: list[str], sample_size: int = 500) -> float:
    """Mean pairwise Morgan-FP Tanimoto similarity over sampled pairs.

    All-pairs is used if the total number of pairs is at most ``sample_size``;
    otherwise pairs are drawn uniformly at random (seeded for reproducibility).
    Returns NaN if fewer than 2 valid molecules are present.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    if len(fps) < 2:
        return float("nan")
    pairs: list[float] = []
    n_total_pairs = len(fps) * (len(fps) - 1) // 2
    if n_total_pairs <= sample_size:
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                pairs.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    else:
        rng = random.Random(42)
        seen: set[tuple[int, int]] = set()
        while len(pairs) < sample_size:
            i, j = sorted(rng.sample(range(len(fps)), 2))
            if (i, j) in seen:
                continue
            seen.add((i, j))
            pairs.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return sum(pairs) / len(pairs) if pairs else float("nan")


def _activity_cliff_frequency(
    sub: pd.DataFrame,
    smiles_col: str,
    activity_col: str,
    tanimoto_threshold: float = 0.7,
    activity_delta_threshold: float = 1.5,
    sample_size: int = 1000,
) -> float:
    """Fraction of sampled pairs that are activity cliffs.

    A pair is an activity cliff when ``Tanimoto >= tanimoto_threshold`` and
    ``|Δactivity| >= activity_delta_threshold`` (defaults: 0.7 / 1.5 pActivity).
    Returns NaN if fewer than 2 valid molecules with non-NaN activities remain.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    smis = sub[smiles_col].tolist()
    acts = sub[activity_col].tolist()
    fps = []
    keep_acts: list[float] = []
    for s, a in zip(smis, acts):
        mol = Chem.MolFromSmiles(s)
        if mol is None or pd.isna(a):
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        keep_acts.append(a)
    if len(fps) < 2:
        return float("nan")
    n_total = len(fps) * (len(fps) - 1) // 2
    n_cliff = 0
    n_sampled = 0
    if n_total <= sample_size:
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                tani = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                if (
                    tani >= tanimoto_threshold
                    and abs(keep_acts[i] - keep_acts[j]) >= activity_delta_threshold
                ):
                    n_cliff += 1
                n_sampled += 1
    else:
        rng = random.Random(42)
        seen: set[tuple[int, int]] = set()
        while n_sampled < sample_size:
            i, j = sorted(rng.sample(range(len(fps)), 2))
            if (i, j) in seen:
                continue
            seen.add((i, j))
            tani = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            if (
                tani >= tanimoto_threshold
                and abs(keep_acts[i] - keep_acts[j]) >= activity_delta_threshold
            ):
                n_cliff += 1
            n_sampled += 1
    return n_cliff / n_sampled if n_sampled else float("nan")
