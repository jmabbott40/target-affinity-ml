"""TargetClassConfig: declares how to identify and curate a protein target class.

This abstraction replaces hardcoded kinase logic so the data pipeline works for
any target class (kinases, GPCRs, proteases, etc.). A class is identified either
by GO molecular-function terms (the kinase approach) or by an explicit list of
ChEMBL target IDs (the GPCR aminergic approach, where the 30 targets are
hand-curated).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TargetClassConfig:
    """Declarative configuration for a protein target class.

    Parameters
    ----------
    class_name : str
        Short identifier, e.g. "kinase" or "gpcr_aminergic".
    raw_filename_stem : str
        Stem for raw data files; e.g. "chembl_kinase" yields
        "chembl_kinase_activities.parquet" and "chembl_kinase_targets.parquet".
    go_terms : set[str]
        GO molecular-function terms identifying the class. Used when
        explicit_target_ids is not provided.
    name_keywords : list[str]
        Keywords that (case-insensitive) appear in target names of this class.
        Used as a secondary filter alongside GO terms.
    explicit_target_ids : list[str] | None
        If provided, the class is defined by exactly these ChEMBL target IDs.
        Takes precedence over go_terms.
    subfamily_map : dict[str, str]
        Optional mapping of target_chembl_id -> subfamily name, used for the
        target-held-out split. For kinases this is the kinase group; for
        aminergic GPCRs it is the receptor family (dopamine, serotonin, etc.).
    """

    class_name: str
    raw_filename_stem: str
    go_terms: set[str] = field(default_factory=set)
    name_keywords: list[str] = field(default_factory=list)
    explicit_target_ids: list[str] | None = None
    subfamily_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.go_terms and not self.explicit_target_ids:
            raise ValueError(
                "TargetClassConfig requires either go_terms or "
                "explicit_target_ids to identify the class."
            )

    @property
    def uses_explicit_target_list(self) -> bool:
        """True if the class is defined by an explicit ChEMBL ID list."""
        return self.explicit_target_ids is not None

    @property
    def raw_activities_filename(self) -> str:
        return f"{self.raw_filename_stem}_activities.parquet"

    @property
    def raw_targets_filename(self) -> str:
        return f"{self.raw_filename_stem}_targets.parquet"
