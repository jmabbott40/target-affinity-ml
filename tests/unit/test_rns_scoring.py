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


def test_fetch_binding_site_kinase(tmp_path):
    """Kinase binding-site routes to KLIFS and returns ~85 residues."""
    from target_affinity_ml.benchmarks.rns_scoring import fetch_binding_site
    # EGFR is CHEMBL203 — well-covered by KLIFS
    residues = fetch_binding_site("CHEMBL203", class_name="kinase", cache_dir=tmp_path)
    assert isinstance(residues, list)
    assert all(isinstance(r, int) for r in residues)
    assert 60 <= len(residues) <= 100  # KLIFS canonical pocket is 85; allow drift for non-standard kinases


def test_fetch_binding_site_gpcr(tmp_path):
    """GPCR binding-site routes to GPCRdb and returns ~25-40 residues."""
    from target_affinity_ml.benchmarks.rns_scoring import fetch_binding_site
    # DRD2 is CHEMBL217 — well-covered by GPCRdb
    residues = fetch_binding_site(
        "CHEMBL217",
        class_name="gpcr_aminergic",
        cache_dir=tmp_path,
        uniprot_id="P14416",  # provide UniProt as fallback
    )
    assert isinstance(residues, list)
    assert all(isinstance(r, int) for r in residues)
    # GPCRdb orthosteric pocket sizes vary; accept a broad range
    assert 10 <= len(residues) <= 60


def test_fetch_binding_site_missing_target_returns_empty(tmp_path):
    """A target not in KLIFS returns empty list (and logs WARNING) — does NOT raise."""
    from target_affinity_ml.benchmarks.rns_scoring import fetch_binding_site
    residues = fetch_binding_site("CHEMBL_INVALID_999", class_name="kinase", cache_dir=tmp_path)
    assert residues == []


def test_fetch_binding_site_caches_correctly(tmp_path):
    """Re-querying the same target reads cached JSON, not the API."""
    from target_affinity_ml.benchmarks.rns_scoring import fetch_binding_site
    _ = fetch_binding_site("CHEMBL203", class_name="kinase", cache_dir=tmp_path)
    cache_file = tmp_path / "binding_sites" / "kinase_CHEMBL203.json"
    assert cache_file.exists()
    # Mtime check (Plan 2 pattern)
    mtime_first = cache_file.stat().st_mtime
    import time as _t; _t.sleep(0.01)
    _ = fetch_binding_site("CHEMBL203", class_name="kinase", cache_dir=tmp_path)
    assert cache_file.stat().st_mtime == mtime_first


# ---------------------------------------------------------------------------
# compute_msa tests (P3-T4)
# ---------------------------------------------------------------------------


def test_compute_msa_invokes_jackhmmer_with_expected_args(tmp_path, monkeypatch):
    """compute_msa shells out to jackhmmer with the documented argument set
    and writes a Stockholm-format MSA at the cached path."""
    from target_affinity_ml.benchmarks.rns_scoring import compute_msa
    from pathlib import Path
    captured_args = []

    def fake_run(args, **kw):
        captured_args.append(args)
        # jackhmmer's -A flag is followed by the MSA output path; write a
        # minimal valid Stockholm file so the cache check passes on re-entry
        idx = args.index("-A")
        out_path = Path(args[idx + 1])
        out_path.write_text("# STOCKHOLM 1.0\nseq1 MGKLA\n//\n")
        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "target_affinity_ml.benchmarks.rns_scoring._fetch_uniprot_fasta",
        lambda uid: f">sp|{uid}|TEST\nMGKLAEDIENPNN\n"
    )

    out = compute_msa(
        "P00533",
        db_path=Path("/fake/uniref50.fasta"),
        out_dir=tmp_path,
        n_iter=1,
        n_cpu=2,
    )
    assert out.exists()
    assert out == tmp_path / "P00533.sto"
    # Confirm the jackhmmer invocation used the right flags
    assert len(captured_args) == 1
    args = captured_args[0]
    assert args[0] == "jackhmmer"
    assert "-N" in args and args[args.index("-N") + 1] == "1"
    assert "--cpu" in args and args[args.index("--cpu") + 1] == "2"
    assert "-A" in args and args[args.index("-A") + 1] == str(tmp_path / "P00533.sto")


def test_compute_msa_is_idempotent_on_existing_cache(tmp_path, monkeypatch):
    """If the .sto file already exists, compute_msa returns immediately without
    fetching the query sequence or invoking jackhmmer."""
    from target_affinity_ml.benchmarks.rns_scoring import compute_msa
    # Pre-create a non-empty cache file
    (tmp_path / "P00533.sto").write_text("# STOCKHOLM 1.0\nseq1 MGKLA\n//\n")
    fetch_calls = []
    run_calls = []
    monkeypatch.setattr(
        "target_affinity_ml.benchmarks.rns_scoring._fetch_uniprot_fasta",
        lambda uid: (fetch_calls.append(uid), "")[1]
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: (run_calls.append(a), None)[1]
    )
    out = compute_msa("P00533", db_path=Path("/fake"), out_dir=tmp_path)
    assert out.exists()
    assert fetch_calls == []  # network was NOT called
    assert run_calls == []     # jackhmmer was NOT invoked


def test_compute_msa_raises_on_jackhmmer_failure(tmp_path, monkeypatch):
    """A nonzero jackhmmer returncode propagates as RuntimeError with stderr."""
    from target_affinity_ml.benchmarks.rns_scoring import compute_msa

    def fake_run(args, **kw):
        class R:
            returncode = 1
            stderr = "jackhmmer: database not found"
        return R()
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "target_affinity_ml.benchmarks.rns_scoring._fetch_uniprot_fasta",
        lambda uid: ">sp|TEST|TEST\nMGKLA\n"
    )

    with pytest.raises(RuntimeError, match="jackhmmer failed"):
        compute_msa("P00533", db_path=Path("/fake"), out_dir=tmp_path)


@pytest.mark.slow
def test_compute_msa_real_jackhmmer(tmp_path):
    """Integration: run real jackhmmer against UniRef50 (~10-30 min). Skips locally."""
    from target_affinity_ml.benchmarks.rns_scoring import compute_msa
    db = Path("~/databases/uniref50.fasta").expanduser()
    if not db.exists():
        pytest.skip("UniRef50 not available in this env (this test runs on AWS)")
    msa = compute_msa("P00533", db_path=db, out_dir=tmp_path, n_iter=1, n_cpu=4)
    assert msa.exists()
    assert msa.stat().st_size > 1000  # nontrivial alignment


# ---------------------------------------------------------------------------
# Helpers for P3-T5 unit tests
# ---------------------------------------------------------------------------


def _build_tiny_structure_with_known_geometry():
    """Return a 5-residue Bio.PDB Structure with CA atoms on a straight line.

    Residues 1-5 are placed with CA coordinates at:
      1: (0.0,   0.0, 0.0)
      2: (3.8,   0.0, 0.0)
      3: (7.6,   0.0, 0.0)
      4: (11.4,  0.0, 0.0)
      5: (15.2,  0.0, 0.0)

    This means:
      - Residues 1, 2, 3 are all mutually within 8 Å of each other.
      - Residue 4 is 7.6 Å from residue 2 (within the boundary) and 3.8 Å
        from residue 3, so it counts as a neighbor of both residue 2 and 3.
      - Residue 5 (15.2 Å from residue 1) is far from 1-3 and outside the
        8 Å radius for residues 1-3.

    The one-character amino-acid codes cycle through A, G, K, L, A.
    """
    from Bio.PDB import Structure, Model, Chain, Residue, Atom
    import numpy as np

    structure = Structure.Structure("test")
    model = Model.Model(0)
    chain = Chain.Chain("A")

    # Amino-acid codes (index 0 → residue 1, etc.)
    aa_codes = ["ALA", "GLY", "LYS", "LEU", "ALA"]
    coords = [
        (0.0,  0.0, 0.0),
        (3.8,  0.0, 0.0),
        (7.6,  0.0, 0.0),
        (11.4, 0.0, 0.0),
        (15.2, 0.0, 0.0),
    ]

    for i, (resname, coord) in enumerate(zip(aa_codes, coords), start=1):
        res = Residue.Residue((" ", i, " "), resname, " ")
        ca = Atom.Atom(
            name="CA",
            coord=np.array(coord, dtype="f"),
            bfactor=0.0,
            occupancy=1.0,
            altloc=" ",
            fullname=" CA ",
            serial_number=i,
            element="C",
        )
        res.add(ca)
        chain.add(res)

    model.add(chain)
    structure.add(model)
    return structure


def _build_synthetic_msa(
    n_seqs: int = 5,
    length: int = 10,
    conserved_positions: list[int] | None = None,
) -> str:
    """Return a minimal Stockholm-format MSA string.

    The query sequence (first entry, id="query") contains 'A' at conserved
    positions and a cycling alphabet ('A','C','D','E','F','G','H','I','K','L')
    at variable positions.  All homologs share the conserved positions ('A')
    but independently randomize the variable positions.

    Position indices in *conserved_positions* are 1-based (matching residue
    numbering convention used throughout the codebase).

    Returns a Stockholm-format string ready to write to a .sto file.
    """
    import random

    rng = random.Random(42)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    conserved = set(conserved_positions or [])

    # Build the query sequence: 'A' at conserved positions, others vary
    query_bases = []
    for pos in range(1, length + 1):
        if pos in conserved:
            query_bases.append("A")
        else:
            query_bases.append(rng.choice(amino_acids))
    query_seq = "".join(query_bases)

    lines = ["# STOCKHOLM 1.0", f"query {query_seq}"]

    # Build homolog sequences
    for idx in range(1, n_seqs):
        seq_chars = []
        for pos in range(1, length + 1):
            if pos in conserved:
                seq_chars.append("A")  # conserved: all homologs agree
            else:
                seq_chars.append(rng.choice(amino_acids))
        lines.append(f"seq{idx} {''.join(seq_chars)}")

    lines.append("//")
    return "\n".join(lines) + "\n"


def _build_synthetic_msa_with_varying_conservation() -> str:
    """Return a Stockholm MSA where positions 1-2 are highly conserved (all 'A')
    and positions 3-5 are maximally variable (each sequence has a different AA).

    The MSA has 10 sequences, each of length 5.  Column index 1 is position 1,
    column index 2 is position 2, etc. (1-based, no gaps in the query).

    Conservation pattern:
      Col 1: all 'A' (conserved)
      Col 2: all 'A' (conserved)
      Col 3: K/C/E/G/I/P/R/V/Y/H (variable, 10 distinct AAs)
      Col 4: L/F/V/W/M/Q/S/T/N/H (variable, 10 distinct AAs)
      Col 5: D/E/F/G/H/I/K/L/M/N (variable, 10 distinct AAs)

    With sequence_window=1 and the 5-residue test structure:
      residue 2's neighborhood covers cols 1-4 (2 conserved + 2 variable)
      residue 3's neighborhood covers cols 1-5 (2 conserved + 3 variable)
    So residue 2 gets a strictly higher RNS score than residue 3.
    """
    seqs = {
        "query": "AAKLD",
        "seq1":  "AACFE",
        "seq2":  "AAEVF",
        "seq3":  "AAGWG",
        "seq4":  "AAIMH",
        "seq5":  "AAPQI",
        "seq6":  "AARSK",
        "seq7":  "AAVTL",
        "seq8":  "AAYNM",
        "seq9":  "AAHHN",
    }
    lines = ["# STOCKHOLM 1.0"]
    for name, seq in seqs.items():
        lines.append(f"{name} {seq}")
    lines.append("//")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# compute_per_residue_rns tests (P3-T5)
# ---------------------------------------------------------------------------


def test_compute_per_residue_rns_returns_normalized_scores(tmp_path):
    """RNS scores are floats in [0, 1] for each binding-site residue with valid neighborhoods."""
    from target_affinity_ml.benchmarks.rns_scoring import compute_per_residue_rns

    structure = _build_tiny_structure_with_known_geometry()
    msa_path = tmp_path / "test.sto"
    msa_path.write_text(_build_synthetic_msa(n_seqs=5, length=10, conserved_positions=[2, 3]))
    scores = compute_per_residue_rns(structure, binding_site=[2, 3], msa_path=msa_path)
    # Only binding-site residues should appear as keys
    assert set(scores.keys()).issubset({2, 3})
    # All scores must be in [0, 1]
    assert all(0.0 <= v <= 1.0 for v in scores.values()), f"Out-of-range scores: {scores}"


def test_compute_per_residue_rns_higher_score_for_more_conserved(tmp_path):
    """A residue with a more-conserved neighborhood gets a higher RNS than one with a more-variable neighborhood.

    Uses sequence_window=1 so that residue 2's sequence neighbourhood (cols 1-3)
    and residue 3's sequence neighbourhood (cols 2-4) don't fully overlap on
    the 5-residue test structure.  Combined with the spatial neighbourhood
    (residue 2 sees cols 1-4; residue 3 sees cols 1-5), residue 2 ends up with
    2 conserved columns out of 4 while residue 3 has 2 conserved columns out of 5.
    The assertion is strict (>) because the conservation difference is real, not vacuous.
    """
    from target_affinity_ml.benchmarks.rns_scoring import compute_per_residue_rns

    structure = _build_tiny_structure_with_known_geometry()
    msa_path = tmp_path / "test.sto"
    # Positions 1-2 are conserved (all 'A'); positions 3-5 are maximally variable.
    msa_path.write_text(_build_synthetic_msa_with_varying_conservation())
    scores = compute_per_residue_rns(
        structure, binding_site=[2, 3], msa_path=msa_path, sequence_window=1
    )
    assert 2 in scores and 3 in scores, (
        f"Both binding-site residues must appear in scores; got keys: {set(scores)}"
    )
    assert scores[2] > scores[3], (
        f"Expected conserved residue (pos 2, score {scores[2]:.4f}) > "
        f"variable residue (pos 3, score {scores[3]:.4f})"
    )


def test_compute_per_residue_rns_skips_missing_residue(tmp_path):
    """A binding-site residue not present in the structure is excluded from output (not an exception)."""
    from target_affinity_ml.benchmarks.rns_scoring import compute_per_residue_rns

    structure = _build_tiny_structure_with_known_geometry()  # has residues 1-5
    msa_path = tmp_path / "test.sto"
    msa_path.write_text(_build_synthetic_msa(n_seqs=5, length=10))
    # residue 999 is not in the structure
    scores = compute_per_residue_rns(structure, binding_site=[2, 999], msa_path=msa_path)
    assert 999 not in scores
    # key assertion: no exception was raised (we reach this line successfully)


def test_compute_conservation_entropy_lower_for_more_conserved(tmp_path):
    """Conservation entropy is lower for highly-conserved binding-site columns."""
    from target_affinity_ml.benchmarks.rns_scoring import compute_conservation_entropy

    msa_path = tmp_path / "test.sto"
    # conserved_positions=[2, 3] → all seqs agree on those columns → near-zero entropy
    msa_path.write_text(_build_synthetic_msa(n_seqs=10, length=10, conserved_positions=[2, 3]))
    h_conserved = compute_conservation_entropy([2, 3], msa_path)
    # positions 5 and 7 are variable → higher entropy
    h_variable = compute_conservation_entropy([5, 7], msa_path)
    assert h_conserved < h_variable, (
        f"Expected conserved entropy ({h_conserved:.4f}) < variable entropy ({h_variable:.4f})"
    )


# ---------------------------------------------------------------------------
# Helper for P3-T6 unit tests
# ---------------------------------------------------------------------------


def _build_tiny_structure_with_plddt(plddt_map: dict[int, float]):
    """Return a Bio.PDB Structure with CA atoms, using plddt_map values as B-factors.

    *plddt_map* maps residue_idx (1-based int) → pLDDT value (float).
    Each residue is placed at an arbitrary coordinate; only the B-factor
    (used by _get_residue_plddt to read pLDDT) matters for the tests.
    """
    from Bio.PDB import Structure, Model, Chain, Residue, Atom
    import numpy as np

    structure = Structure.Structure("plddt_test")
    model = Model.Model(0)
    chain = Chain.Chain("A")

    for i, (res_idx, plddt_val) in enumerate(sorted(plddt_map.items())):
        res = Residue.Residue((" ", res_idx, " "), "ALA", " ")
        ca = Atom.Atom(
            name="CA",
            coord=np.array([float(i) * 3.8, 0.0, 0.0], dtype="f"),
            bfactor=float(plddt_val),
            occupancy=1.0,
            altloc=" ",
            fullname=" CA ",
            serial_number=i + 1,
            element="C",
        )
        res.add(ca)
        chain.add(res)

    model.add(chain)
    structure.add(model)
    return structure


# ---------------------------------------------------------------------------
# aggregate_target_rns + validation_gate tests (P3-T6)
# ---------------------------------------------------------------------------


def test_aggregate_target_rns_pdb_uniform_weights(tmp_path):
    """PDB structures use uniform-weight mean over per-residue RNS."""
    from target_affinity_ml.benchmarks.rns_scoring import aggregate_target_rns
    per_residue = {10: 0.6, 20: 0.8, 30: 0.4}
    prov = {"source": "PDB"}
    result = aggregate_target_rns(per_residue, prov, structure=None, use_plddt_weighting=True)
    assert abs(result - 0.6) < 1e-6  # (0.6 + 0.8 + 0.4) / 3


def test_aggregate_target_rns_alphafold_plddt_weighted(tmp_path):
    """AlphaFold structures apply pLDDT weighting."""
    from target_affinity_ml.benchmarks.rns_scoring import aggregate_target_rns
    # Build a tiny structure with controlled pLDDT values in bfactors
    structure = _build_tiny_structure_with_plddt({10: 90.0, 20: 70.0, 30: 40.0})
    per_residue = {10: 0.6, 20: 0.8, 30: 0.4}
    prov = {"source": "AlphaFold"}
    result = aggregate_target_rns(per_residue, prov, structure=structure)
    # weights: residue 10 = max(0, (90-50)/50) = 0.8
    #          residue 20 = max(0, (70-50)/50) = 0.4
    #          residue 30 = max(0, (40-50)/50) = 0.0
    # weighted mean = (0.6*0.8 + 0.8*0.4 + 0.4*0.0) / (0.8 + 0.4 + 0.0) = 0.8 / 1.2 = 0.6667
    assert abs(result - 0.6667) < 0.001


def test_aggregate_target_rns_empty_returns_nan(tmp_path):
    """Empty per-residue dict returns nan."""
    import math
    from target_affinity_ml.benchmarks.rns_scoring import aggregate_target_rns
    result = aggregate_target_rns({}, {"source": "PDB"}, structure=None)
    assert math.isnan(result)


def test_validation_gate_pass_path(tmp_path, monkeypatch):
    """validation_gate returns passed=True when our values rank-correlate with reference."""
    from target_affinity_ml.benchmarks import rns_scoring
    # Mock the heavy pipeline calls
    def fake_fetch(uid, cache_dir, **kw): return ("fake_structure", {"source": "PDB"})
    def fake_msa(uid, db, out): out.mkdir(parents=True, exist_ok=True); return out / f"{uid}.sto"
    def fake_per_res(s, bs, m): return {r: 0.1 * i for i, r in enumerate(bs, 1)}
    # Return values that correlate with the reference: rank order matches ConSurf
    # 8 reference proteins in order: EGFR(0.88), ABL1(0.85), CDK2(0.82), p38(0.79),
    # HSP90(0.75), ADRB2(0.71), p53(0.68), HIV-pr(0.62)
    _pass_seq = iter([0.88, 0.85, 0.82, 0.79, 0.75, 0.71, 0.68, 0.62])
    def fake_agg(pr, p, s, **kw): return next(_pass_seq)
    monkeypatch.setattr(rns_scoring, "fetch_structure", fake_fetch)
    monkeypatch.setattr(rns_scoring, "compute_msa", fake_msa)
    monkeypatch.setattr(rns_scoring, "compute_per_residue_rns", fake_per_res)
    monkeypatch.setattr(rns_scoring, "aggregate_target_rns", fake_agg)

    passed, dev, csv = rns_scoring.validation_gate(cache_dir=tmp_path, db_path=Path("/fake"))
    assert passed is True
    assert dev["spearman_rho"] > 0.9  # perfect rank order


def test_validation_gate_fail_path(tmp_path, monkeypatch):
    """validation_gate returns passed=False when our values are uncorrelated with reference."""
    from target_affinity_ml.benchmarks import rns_scoring
    def fake_fetch(uid, cache_dir, **kw): return ("fake_structure", {"source": "PDB"})
    def fake_msa(uid, db, out): out.mkdir(parents=True, exist_ok=True); return out / f"{uid}.sto"
    def fake_per_res(s, bs, m): return {r: 0.5 for r in bs}
    # Return values UNCORRELATED with the ConSurf reference (random-ish)
    bad_vals = iter([0.20, 0.95, 0.45, 0.60, 0.15, 0.85, 0.30, 0.55])
    def fake_agg(pr, p, s, **kw): return next(bad_vals)
    monkeypatch.setattr(rns_scoring, "fetch_structure", fake_fetch)
    monkeypatch.setattr(rns_scoring, "compute_msa", fake_msa)
    monkeypatch.setattr(rns_scoring, "compute_per_residue_rns", fake_per_res)
    monkeypatch.setattr(rns_scoring, "aggregate_target_rns", fake_agg)

    passed, dev, csv = rns_scoring.validation_gate(cache_dir=tmp_path, db_path=Path("/fake"))
    # MAD = mean(|our - ref|) — with chosen values, will exceed 0.10 threshold
    # Spearman correlation is low/negative because rank order is jumbled
    assert passed is False  # both criteria miss
