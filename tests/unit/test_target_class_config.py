"""Tests for the TargetClassConfig abstraction."""
import pytest

from target_affinity_ml.data.target_class_config import TargetClassConfig


def test_minimal_config_construction():
    cfg = TargetClassConfig(
        class_name="kinase",
        go_terms={"GO:0016301"},
        name_keywords=["kinase"],
        raw_filename_stem="chembl_kinase",
    )
    assert cfg.class_name == "kinase"
    assert "GO:0016301" in cfg.go_terms
    assert cfg.raw_activities_filename == "chembl_kinase_activities.parquet"
    assert cfg.raw_targets_filename == "chembl_kinase_targets.parquet"


def test_config_with_explicit_target_ids():
    """A class can be defined by an explicit ChEMBL ID list (GPCR aminergic case)."""
    cfg = TargetClassConfig(
        class_name="gpcr_aminergic",
        explicit_target_ids=["CHEMBL217", "CHEMBL224"],
        raw_filename_stem="chembl_gpcr_aminergic",
    )
    assert cfg.explicit_target_ids == ["CHEMBL217", "CHEMBL224"]
    assert cfg.uses_explicit_target_list is True


def test_config_go_based_is_not_explicit():
    cfg = TargetClassConfig(
        class_name="kinase",
        go_terms={"GO:0016301"},
        raw_filename_stem="chembl_kinase",
    )
    assert cfg.uses_explicit_target_list is False


def test_config_requires_identification_method():
    """Must provide either go_terms or explicit_target_ids."""
    with pytest.raises(ValueError, match="go_terms.*or.*explicit_target_ids"):
        TargetClassConfig(class_name="empty", raw_filename_stem="x")
