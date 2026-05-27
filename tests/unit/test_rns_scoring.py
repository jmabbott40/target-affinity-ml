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


def test_fetch_structure_returns_structure_and_provenance(tmp_path):
    """Fetches EGFR (P00533) AlphaFold structure and returns provenance dict."""
    from target_affinity_ml.benchmarks.rns_scoring import fetch_structure

    structure, provenance = fetch_structure("P00533", cache_dir=tmp_path, prefer="alphafold")
    assert structure is not None
    assert provenance["source"] == "AlphaFold"
    assert provenance["uniprot_id"] == "P00533"
    assert provenance["pdb_id"] is None
    assert provenance["pdb_resolution"] is None
    assert provenance["binding_site_pLDDT_mean"] is None
    assert provenance["conformational_state"] == "unknown"
    # Cached file exists on disk
    assert (tmp_path / "alphafold" / "P00533.pdb").exists()


def test_fetch_structure_caches_correctly(tmp_path):
    """Re-fetching the same accession returns cached file without re-download."""
    import time as _time

    from target_affinity_ml.benchmarks.rns_scoring import fetch_structure

    _ = fetch_structure("P00533", cache_dir=tmp_path, prefer="alphafold")
    mtime_first = (tmp_path / "alphafold" / "P00533.pdb").stat().st_mtime
    _time.sleep(0.01)  # ensure mtime resolution
    _ = fetch_structure("P00533", cache_dir=tmp_path, prefer="alphafold")
    mtime_second = (tmp_path / "alphafold" / "P00533.pdb").stat().st_mtime
    assert mtime_first == mtime_second
