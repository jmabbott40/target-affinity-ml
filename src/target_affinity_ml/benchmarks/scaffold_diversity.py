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


def _bemis_murcko_scaffolds(smiles: pd.Series) -> list[str]:
    """Compute Bemis-Murcko scaffolds via the corrected Plan 2 Task 8 idiom.

    Uses ``MurckoScaffold.GetScaffoldForMol(mol)`` followed by
    ``Chem.MolToSmiles(scaff)``. Invalid SMILES are tagged ``INVALID`` so they
    still occupy a counter bin without crashing aggregation downstream.
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
        out.append(Chem.MolToSmiles(scaff) if scaff is not None else "NO_SCAFFOLD")
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
        random.seed(42)
        seen: set[tuple[int, int]] = set()
        while len(pairs) < sample_size:
            i, j = sorted(random.sample(range(len(fps)), 2))
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
    random.seed(42)
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
        seen: set[tuple[int, int]] = set()
        while n_sampled < sample_size:
            i, j = sorted(random.sample(range(len(fps)), 2))
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
