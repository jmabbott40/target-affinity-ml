"""Tests for the RNS scoring module.

This file is populated incrementally by Plan 3 Tasks 2-6 (fetch_structure,
fetch_binding_site, compute_msa, compute_per_residue_rns, validation_gate).
The single test below confirms the bundled reference data loads correctly and
has the expected structure required by Task 6's validation_gate().
"""
import json
from pathlib import Path

import pytest


@pytest.fixture
def reference_data():
    path = (
        Path(__file__).parent.parent.parent
        / "src/target_affinity_ml/benchmarks/_rns_reference_data.json"
    )
    with open(path) as fh:
        return json.load(fh)


def test_reference_data_loads(reference_data):
    """The bundled RNS reference data has 5+ entries with required fields."""
    assert "reference_proteins" in reference_data
    assert len(reference_data["reference_proteins"]) >= 5
    for p in reference_data["reference_proteins"]:
        assert "uniprot" in p
        assert "binding_site_residues" in p
        assert isinstance(p["binding_site_residues"], list)
        assert len(p["binding_site_residues"]) > 0
        assert "published_target_rns" in p
        assert isinstance(p["published_target_rns"], (int, float))
