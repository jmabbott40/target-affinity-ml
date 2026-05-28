"""Prabakaran-Bromberg RNS scoring with conservation-entropy fallback.

Functions
---------
fetch_structure          : Fetch a protein structure from PDB or AlphaFold DB
fetch_binding_site       : Identify binding-site residues (KLIFS / GPCRdb)
compute_msa              : Build a multiple sequence alignment with jackhmmer
compute_per_residue_rns  : Compute per-residue RNS from an MSA
aggregate_target_rns     : Aggregate per-residue RNS over a binding site
compute_conservation_entropy : Conservation-entropy fallback for gapped sites
validation_gate          : GO/NO-GO gate comparing computed RNS to reference values
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import requests
from Bio.PDB import PDBParser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Binding-site constants
# ---------------------------------------------------------------------------

# Canonical Ballesteros-Weinstein (BW) generic-number stems that define the
# Class A aminergic GPCR orthosteric pocket.  These positions were assembled
# from: (1) published mutagenesis data for monoamine receptors, (2) the set
# of contact residues in co-crystal structures of DRD2/DRD3/DRD4, ADRB1/2,
# CHRM1-5, and 5-HT-receptor subtypes with orthosteric ligands.
#
# Format: "X.YY" where X is TM helix number (or "45" for ECL2) and YY is
# the residue position within that helix.  The full GPCRdb generic number
# includes an "xNN" suffix; we match only on the stem (before "x").
#
# Reference: Gloriam et al., Trends Pharmacol. Sci. 2018; GPCRdb.org 2024.
_GPCR_AMINERGIC_ORTHOSTERIC_BW: frozenset[str] = frozenset({
    "2.53", "2.57", "2.60", "2.61", "2.64",
    "3.28", "3.29", "3.32", "3.33", "3.36", "3.37",
    "4.56", "4.57", "4.60",
    "5.38", "5.39", "5.42", "5.43", "5.46",
    "6.48", "6.51", "6.52", "6.55",
    "7.35", "7.36", "7.39", "7.40", "7.42", "7.43",
    "45.52", "45.53",  # ECL2 loop residues
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download_with_backoff(url: str, path: Path, max_retries: int) -> None:
    """Download *url* to *path*, retrying on 5xx / 429 with exponential backoff.

    The function is idempotent in intent — the caller is responsible for
    checking whether the file already exists before calling.

    Parameters
    ----------
    url:
        Full HTTPS URL to fetch.
    path:
        Destination file path.  Parent directories are created automatically.
    max_retries:
        Maximum number of attempts (including the first).

    Raises
    ------
    RuntimeError
        After all retries are exhausted without a successful download.
    requests.HTTPError
        Immediately on any 4xx status other than 429 (client errors that
        are unlikely to resolve on retry).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                path.write_bytes(r.content)
                return
            if 500 <= r.status_code < 600 or r.status_code == 429:
                time.sleep(2**attempt)
                continue
            r.raise_for_status()  # 4xx other than 429 — give up immediately
        except requests.exceptions.RequestException:
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {url} after {max_retries} retries")


def _resolve_alphafold_pdb_url(uniprot_id: str) -> str:
    """Return the current PDB download URL for *uniprot_id* from the AlphaFold DB API.

    The AlphaFold DB increments its model version periodically (v3 → v4 → v6
    etc.).  Hardcoding a version number would break silently.  This helper
    queries the lightweight JSON API endpoint to obtain the canonical URL,
    then falls back to a v4-style pattern only if the API itself is
    unreachable.

    Parameters
    ----------
    uniprot_id:
        UniProt accession string.

    Returns
    -------
    str
        HTTPS URL pointing to a ``.pdb`` file on the AlphaFold static CDN.

    Raises
    ------
    RuntimeError
        If the API returns a non-200 status or the response contains no
        ``pdbUrl`` field.
    """
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        r = requests.get(api_url, timeout=30)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"AlphaFold API unreachable for {uniprot_id}: {exc}"
        ) from exc

    if r.status_code != 200:
        raise RuntimeError(
            f"AlphaFold API returned HTTP {r.status_code} for {uniprot_id}"
        )

    entries = r.json()
    if not entries or "pdbUrl" not in entries[0]:
        raise RuntimeError(
            f"AlphaFold API response for {uniprot_id} missing pdbUrl field"
        )

    return entries[0]["pdbUrl"]


def _extract_resolution(structure: Any) -> float | None:
    """Return the crystallographic resolution (Å) from *structure.header*, or None.

    AlphaFold DB structures carry no resolution metadata, so this returns
    None for those files.  PDB structures report resolution in the REMARK 2
    record, which Biopython surfaces as ``structure.header['resolution']``.
    A resolution of 0.0 is treated as absent (some old PDB entries use 0 as
    a sentinel).

    Parameters
    ----------
    structure:
        A Biopython ``Structure`` object (from ``PDBParser``).

    Returns
    -------
    float or None
    """
    value = structure.header.get("resolution")
    if value is None or value == 0:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_json_with_backoff(url: str, max_retries: int = 4) -> Any:
    """Fetch *url* and parse the response as JSON, retrying on 5xx / 429.

    Unlike :func:`_download_with_backoff`, which writes to disk, this helper
    returns the parsed JSON object directly.  A 404 response is treated as a
    "target not in database" signal — it raises :class:`requests.HTTPError`
    immediately (no retry) so that callers can handle it as an empty-result
    sentinel.

    Parameters
    ----------
    url:
        Full HTTPS URL to fetch.
    max_retries:
        Maximum number of attempts (including the first).

    Returns
    -------
    Any
        Parsed JSON (list or dict).

    Raises
    ------
    requests.HTTPError
        On 404 or any non-retried 4xx.
    RuntimeError
        After all retries are exhausted without a successful response.
    """
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                r.raise_for_status()  # raises HTTPError immediately — no retry
            if 500 <= r.status_code < 600 or r.status_code == 429:
                time.sleep(2**attempt)
                continue
            r.raise_for_status()  # other 4xx — give up immediately
        except requests.exceptions.JSONDecodeError:
            time.sleep(2**attempt)
        except requests.exceptions.RequestException:
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch JSON from {url} after {max_retries} retries")


def _resolve_uniprot_from_chembl(chembl_id: str) -> str | None:
    """Return the primary UniProt accession for a ChEMBL target ID, or None.

    Uses the ChEMBL REST API (https://www.ebi.ac.uk/chembl/api/data/target/).
    Returns None (rather than raising) when the target is not found (404) or
    has no target_components.  Callers should treat None as "target unknown."

    Parameters
    ----------
    chembl_id:
        ChEMBL target identifier, e.g. ``"CHEMBL203"``.

    Returns
    -------
    str or None
        UniProt accession of the first (canonical) target component, or None.
    """
    url = f"https://www.ebi.ac.uk/chembl/api/data/target/{chembl_id}.json"
    try:
        data = _fetch_json_with_backoff(url)
    except requests.HTTPError:
        return None
    except RuntimeError:
        return None

    components = data.get("target_components", [])
    if not components:
        return None
    return components[0].get("accession")


def _klifs_binding_site(chembl_id: str, uniprot_id: str | None) -> list[int]:
    """Return the KLIFS 85-residue ATP-pocket positions for *chembl_id*.

    Endpoint discovery (verified 2026-05-27 against https://klifs.net/api_v2/):
    - ``GET /kinase_information`` — returns all kinases; each entry has fields:
      ``kinase_ID``, ``name``, ``gene_name``, ``family``, ``group``,
      ``subfamily``, ``species``, ``full_name``, ``uniprot``, ``iuphar``,
      ``pocket`` (85-residue amino-acid string).  No ChEMBL ID field exists;
      UniProt is used as the bridge identifier.
    - ``GET /structures_list?kinase_ID={id}`` — returns all co-crystal
      structures for a kinase; each entry includes ``structure_ID``,
      ``quality_score``, ``resolution``, and ``pocket``.
    - ``GET /interactions_match_residues?structure_ID={id}`` — returns a
      list of 85 dicts with ``index``, ``Xray_position`` (1-indexed residue
      number in the UniProt sequence), and ``KLIFS_position``.  This gives
      the actual residue numbers, not just the pocket amino-acid string.

    Resolution strategy: pick the highest-quality_score structure (ties
    broken by lowest resolution) and read its per-residue Xray_position
    mapping.  Residues where Xray_position is blank or non-numeric (absent
    in the structure) are silently skipped.

    Parameters
    ----------
    chembl_id:
        ChEMBL target identifier, e.g. ``"CHEMBL203"``.
    uniprot_id:
        Pre-resolved UniProt accession (caller may supply to avoid a
        redundant ChEMBL lookup).

    Returns
    -------
    list[int]
        1-indexed residue positions, or empty list if the kinase is not in
        KLIFS.
    """
    # Resolve UniProt if not supplied
    if uniprot_id is None:
        uniprot_id = _resolve_uniprot_from_chembl(chembl_id)
    if uniprot_id is None:
        logger.warning(
            "KLIFS: could not resolve UniProt for %s; returning empty binding site",
            chembl_id,
        )
        return []

    # Fetch kinase_information and match by UniProt + Human species
    try:
        all_kinases = _fetch_json_with_backoff("https://klifs.net/api_v2/kinase_information")
    except (requests.HTTPError, RuntimeError) as exc:
        logger.warning("KLIFS kinase_information fetch failed for %s: %s", chembl_id, exc)
        return []

    matches = [
        k for k in all_kinases
        if k.get("uniprot") == uniprot_id and k.get("species") == "Human"
    ]
    if not matches:
        logger.warning(
            "KLIFS: UniProt %s (ChEMBL %s) not found in kinase_information; "
            "returning empty binding site",
            uniprot_id, chembl_id,
        )
        return []

    kinase_id = matches[0]["kinase_ID"]

    # Fetch structures for this kinase and pick the best one
    try:
        structures = _fetch_json_with_backoff(
            f"https://klifs.net/api_v2/structures_list?kinase_ID={kinase_id}"
        )
    except (requests.HTTPError, RuntimeError) as exc:
        logger.warning("KLIFS structures_list fetch failed for kinase_ID %d: %s", kinase_id, exc)
        return []

    if not structures:
        logger.warning("KLIFS: no structures found for kinase_ID %d", kinase_id)
        return []

    # Sort: highest quality_score first, then lowest resolution
    structures_sorted = sorted(
        structures,
        key=lambda s: (-float(s.get("quality_score") or 0), float(s.get("resolution") or 99)),
    )
    best_structure_id = structures_sorted[0]["structure_ID"]

    # Fetch per-residue Xray_position mapping for the chosen structure
    try:
        residue_records = _fetch_json_with_backoff(
            f"https://klifs.net/api_v2/interactions_match_residues?structure_ID={best_structure_id}"
        )
    except (requests.HTTPError, RuntimeError) as exc:
        logger.warning(
            "KLIFS interactions_match_residues failed for structure_ID %d: %s",
            best_structure_id, exc,
        )
        return []

    positions: list[int] = []
    for rec in residue_records:
        raw = str(rec.get("Xray_position", "")).strip()
        if raw and raw.lstrip("-").isdigit():
            positions.append(int(raw))

    return positions


def _gpcrdb_binding_site(chembl_id: str, uniprot_id: str | None) -> list[int]:
    """Return orthosteric-pocket residue positions for an aminergic GPCR.

    Endpoint discovery (verified 2026-05-27 against https://gpcrdb.org/services/):
    - ``GET /services/receptorlist/`` — returns all receptors; each entry has
      fields: ``entry_name``, ``name``, ``accession`` (UniProt), ``receptor_class``,
      ``receptor_family``, ``ligand_type``, ``subfamily``, ``endogenous_ligands``,
      ``species``, ``sequence``.
    - ``GET /services/residues/{entry_name}/`` — returns per-residue data; each
      dict has ``sequence_number``, ``amino_acid``, ``protein_segment``, and
      ``display_generic_number`` (Ballesteros-Weinstein notation, e.g.
      ``"3.32x32"``, or null for non-TM residues).

    There is no single "binding site" endpoint in GPCRdb.  Instead, we use the
    canonical set of 30 Ballesteros-Weinstein generic-number stems that define
    the aminergic GPCR orthosteric pocket (``_GPCR_AMINERGIC_ORTHOSTERIC_BW``),
    filter the residue list to those positions, and return their sequence_number
    values.

    Resolution strategy: look up the UniProt accession in receptorlist to get the
    GPCRdb ``entry_name`` (e.g. ``"drd2_human"``), then fetch residues for that
    entry.  Falls back to the supplied ``uniprot_id`` if ChEMBL lookup fails.

    Parameters
    ----------
    chembl_id:
        ChEMBL target identifier, e.g. ``"CHEMBL217"``.
    uniprot_id:
        Pre-resolved UniProt accession (caller may supply; reduces one HTTP call).

    Returns
    -------
    list[int]
        1-indexed residue positions in the canonical UniProt sequence, or an
        empty list if the receptor is not in GPCRdb.
    """
    # Resolve UniProt if not supplied
    if uniprot_id is None:
        uniprot_id = _resolve_uniprot_from_chembl(chembl_id)
    if uniprot_id is None:
        logger.warning(
            "GPCRdb: could not resolve UniProt for %s; returning empty binding site",
            chembl_id,
        )
        return []

    # Fetch full receptor list and match by UniProt accession
    try:
        receptor_list = _fetch_json_with_backoff("https://gpcrdb.org/services/receptorlist/")
    except (requests.HTTPError, RuntimeError) as exc:
        logger.warning("GPCRdb receptorlist fetch failed for %s: %s", chembl_id, exc)
        return []

    matches = [rx for rx in receptor_list if rx.get("accession") == uniprot_id]
    if not matches:
        logger.warning(
            "GPCRdb: UniProt %s (ChEMBL %s) not found in receptorlist; "
            "returning empty binding site",
            uniprot_id, chembl_id,
        )
        return []

    entry_name = matches[0]["entry_name"]

    # Fetch per-residue annotations for this receptor
    try:
        residue_data = _fetch_json_with_backoff(
            f"https://gpcrdb.org/services/residues/{entry_name}/"
        )
    except (requests.HTTPError, RuntimeError) as exc:
        logger.warning("GPCRdb residues fetch failed for %s: %s", entry_name, exc)
        return []

    # Filter to canonical orthosteric-pocket positions by BW generic-number stem
    positions: list[int] = []
    for res in residue_data:
        gn = res.get("display_generic_number")
        if gn:
            stem = gn.split("x")[0]  # e.g. "3.32x32" → "3.32"
            if stem in _GPCR_AMINERGIC_ORTHOSTERIC_BW:
                positions.append(res["sequence_number"])

    return positions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_binding_site(
    target_id: str,
    class_name: str,
    cache_dir: Path,
    uniprot_id: str | None = None,
) -> list[int]:
    """Return binding-site residue indices (1-indexed) for *target_id*.

    Routes to KLIFS for kinases, GPCRdb for aminergic GPCRs.  Caches the
    JSON response to ``cache_dir/binding_sites/{class_name}_{target_id}.json``
    so that subsequent calls with the same target are served from disk.

    Parameters
    ----------
    target_id:
        ChEMBL target identifier, e.g. ``"CHEMBL203"`` (EGFR) or
        ``"CHEMBL217"`` (DRD2).
    class_name:
        One of ``"kinase"`` or ``"gpcr_aminergic"``.
    cache_dir:
        Root cache directory.  Binding-site JSON files land in
        ``cache_dir/binding_sites/``.
    uniprot_id:
        Optional UniProt accession.  When supplied it is passed directly to
        the backend, skipping the ChEMBL → UniProt lookup step.

    Returns
    -------
    list[int]
        1-indexed residue positions in the canonical UniProt sequence.
        Returns an empty list (and logs a WARNING) if the target is absent
        from the database.  Never raises on a missing-target condition so
        that downstream code can handle the graceful-empty case.

    Raises
    ------
    ValueError
        If *class_name* is not ``"kinase"`` or ``"gpcr_aminergic"``.
    """
    if class_name not in {"kinase", "gpcr_aminergic"}:
        raise ValueError(
            f"class_name must be 'kinase' or 'gpcr_aminergic', got {class_name!r}"
        )

    cache_dir = Path(cache_dir)
    cache_file = cache_dir / "binding_sites" / f"{class_name}_{target_id}.json"

    # Serve from cache if available
    if cache_file.exists():
        with cache_file.open() as fh:
            return json.load(fh)

    # Fetch from the appropriate backend
    if class_name == "kinase":
        residues = _klifs_binding_site(target_id, uniprot_id)
    else:
        residues = _gpcrdb_binding_site(target_id, uniprot_id)

    # Persist to cache (even for empty results — avoids repeated failed lookups)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w") as fh:
        json.dump(residues, fh)

    return residues


def fetch_structure(
    uniprot_id: str,
    cache_dir: Path,
    prefer: Literal["pdb", "alphafold"] = "pdb",
    pdb_id: str | None = None,
    max_retries: int = 4,
) -> tuple[Any, dict]:
    """Fetch a protein structure (PDB experimental preferred, AlphaFold fallback).

    The function first checks an on-disk cache keyed by accession so that
    repeated calls with the same inputs are idempotent — no extra network
    traffic is generated.

    Parameters
    ----------
    uniprot_id:
        UniProt accession string (e.g. ``"P00533"`` for EGFR).
    cache_dir:
        Root directory for the on-disk cache.  PDB files land in
        ``cache_dir/pdb/``, AlphaFold files in ``cache_dir/alphafold/``.
    prefer:
        ``"pdb"`` — attempt an experimental PDB structure first (requires
        *pdb_id* to be supplied); fall back to AlphaFold DB if unavailable.
        ``"alphafold"`` — go directly to AlphaFold DB.
    pdb_id:
        Four-character PDB entry code (case-insensitive).  Only consulted
        when ``prefer="pdb"``.
    max_retries:
        Maximum download attempts per URL (passed to
        :func:`_download_with_backoff`).

    Returns
    -------
    (structure, provenance_dict)
        structure : Biopython ``Structure`` object
        provenance : dict with keys:

            * ``source`` — ``"PDB"`` or ``"AlphaFold"``
            * ``uniprot_id`` — the accession that was requested
            * ``pdb_id`` — str or ``None``
            * ``pdb_resolution`` — float or ``None``
            * ``binding_site_pLDDT_mean`` — ``None`` (populated downstream by Task 5)
            * ``binding_site_pLDDT_min`` — ``None`` (populated downstream by Task 5)
            * ``conformational_state`` — ``"active" | "inactive" | "unknown"``
              (default ``"unknown"``)
    """
    cache_dir = Path(cache_dir)
    parser = PDBParser(QUIET=True)

    # ------------------------------------------------------------------
    # Attempt PDB path
    # ------------------------------------------------------------------
    if prefer == "pdb" and pdb_id is not None:
        pdb_id_lower = pdb_id.lower()
        pdb_cache_path = cache_dir / "pdb" / f"{pdb_id_lower}.pdb"

        if not pdb_cache_path.exists():
            pdb_url = f"https://files.rcsb.org/download/{pdb_id_lower}.pdb"
            try:
                _download_with_backoff(pdb_url, pdb_cache_path, max_retries)
            except (RuntimeError, requests.HTTPError):
                # Fall through to AlphaFold
                pdb_cache_path = None  # type: ignore[assignment]

        if pdb_cache_path is not None and pdb_cache_path.exists():
            structure = parser.get_structure(pdb_id_lower, str(pdb_cache_path))
            provenance = {
                "source": "PDB",
                "uniprot_id": uniprot_id,
                "pdb_id": pdb_id_lower,
                "pdb_resolution": _extract_resolution(structure),
                "binding_site_pLDDT_mean": None,
                "binding_site_pLDDT_min": None,
                "conformational_state": "unknown",
            }
            return structure, provenance

    # ------------------------------------------------------------------
    # AlphaFold DB fallback (or primary when prefer="alphafold")
    # ------------------------------------------------------------------
    af_cache_path = cache_dir / "alphafold" / f"{uniprot_id}.pdb"

    if not af_cache_path.exists():
        af_url = _resolve_alphafold_pdb_url(uniprot_id)
        _download_with_backoff(af_url, af_cache_path, max_retries)

    structure = parser.get_structure(uniprot_id, str(af_cache_path))
    provenance = {
        "source": "AlphaFold",
        "uniprot_id": uniprot_id,
        "pdb_id": None,
        "pdb_resolution": None,  # AlphaFold structures have no crystallographic resolution
        "binding_site_pLDDT_mean": None,
        "binding_site_pLDDT_min": None,
        "conformational_state": "unknown",
    }
    return structure, provenance


def _fetch_uniprot_fasta(uniprot_id: str) -> str:
    """Fetch UniProt FASTA via REST. Returns the FASTA text."""
    r = requests.get(
        f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta", timeout=30
    )
    r.raise_for_status()
    return r.text


def compute_msa(
    uniprot_id: str,
    db_path: Path,
    out_dir: Path,
    n_iter: int = 3,
    n_cpu: int = 4,
) -> Path:
    """Run jackhmmer (uniprot_id query sequence against db_path) and cache MSA.

    Parameters
    ----------
    uniprot_id:
        UniProt accession string (e.g. ``"P00533"`` for EGFR).
    db_path:
        Path to the sequence database to search (e.g. UniRef50 FASTA).
    out_dir:
        Directory where the query FASTA and Stockholm MSA are written.
    n_iter:
        Number of jackhmmer iterations (``-N`` flag).  Defaults to 3.
    n_cpu:
        Number of CPU threads for jackhmmer (``--cpu`` flag).  Defaults to 4.

    Returns
    -------
    Path to the Stockholm-format MSA file at out_dir / "{uniprot_id}.sto".
    Idempotent — if the file already exists and is non-empty, returns immediately
    without re-running jackhmmer.

    Raises
    ------
    RuntimeError
        If jackhmmer exits with a nonzero return code.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    msa_path = out_dir / f"{uniprot_id}.sto"

    # Cache check: return immediately if a non-empty MSA already exists
    if msa_path.exists() and msa_path.stat().st_size > 0:
        logger.debug("compute_msa: cache hit for %s at %s", uniprot_id, msa_path)
        return msa_path

    # Fetch query sequence from UniProt and write to disk
    fasta_text = _fetch_uniprot_fasta(uniprot_id)
    query_path = out_dir / f"{uniprot_id}_query.fasta"
    query_path.write_text(fasta_text)

    # Run jackhmmer
    args = [
        "jackhmmer",
        "-N", str(n_iter),
        "--cpu", str(n_cpu),
        "-A", str(msa_path),
        str(query_path),
        str(db_path),
    ]
    logger.info(
        "compute_msa: running jackhmmer for %s (n_iter=%d, n_cpu=%d)",
        uniprot_id, n_iter, n_cpu,
    )
    result = subprocess.run(args, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"jackhmmer failed for {uniprot_id}: {result.stderr}"
        )

    return msa_path


def compute_per_residue_rns(
    structure: Any,
    binding_site: list[int],
    msa_path: Path,
    neighbor_radius_angstrom: float = 8.0,
    sequence_window: int = 5,
    blosum_name: str = "BLOSUM62",
) -> dict[int, float]:
    """Per-residue RNS scores for binding-site residues.

    For each binding-site residue:
      1. Find spatial neighbors (Cα distance <= neighbor_radius_angstrom) via
         Bio.PDB.NeighborSearch.
      2. Add sequence neighbors (residue ± sequence_window positions).
      3. Compute Shannon entropy of the local sequence environment across the MSA
         for each neighbor column.
      4. Weight by BLOSUM62 evolutionary conservation (diagonal self-score of the
         most-frequent amino acid in the column, normalized to [0, 1] by dividing
         by the maximum BLOSUM62 diagonal score, 11 for Trp).
      5. Combine: RNS = (1 − mean_normalized_entropy) * mean_blosum_weight,
         clipped to [0, 1].  Higher means more functionally significant (more
         conserved, stronger BLOSUM support).

    Algorithm notes vs. the original template
    ------------------------------------------
    * Gap characters ('-' and '.') are excluded from the column frequency count
      when computing BLOSUM weights (we only ask "which real AA dominates this
      column?"), but they *are* included as their own symbol when computing the
      Shannon entropy so that gapped, poorly-covered regions are penalised with
      higher entropy (lower RNS).  This is a deliberate deviation: excluding
      gaps from entropy would artificially inflate conservation scores for
      columns where many homologs simply lack sequence coverage.
    * The BLOSUM62 normalization ceiling is 11.0 (Trp self-score), matching the
      template.  Scores below 0 are floored at 0.0 before averaging so that
      negative diagonal entries (e.g. Cys–Cys = 9 is fine, but the very few
      historically mis-assembled BLOSUM variants that have negative diagonals)
      do not drag the weight below zero.
    * Residues whose neighborhood produces zero valid MSA columns (all neighbors
      missing from the MSA mapping) are silently excluded from the returned dict.

    Parameters
    ----------
    structure:
        Bio.PDB.Structure object from :func:`fetch_structure`.
    binding_site:
        1-indexed residue numbers from :func:`fetch_binding_site`.
    msa_path:
        Stockholm-format MSA from :func:`compute_msa`.  The first sequence in
        the file is treated as the query (jackhmmer convention).
    neighbor_radius_angstrom:
        Cα–Cα distance threshold for spatial neighbourhood (Å).
    sequence_window:
        Half-width of the sequence neighbourhood window (residues either side).
    blosum_name:
        Name of the substitution matrix to load via
        ``Bio.Align.substitution_matrices.load``.

    Returns
    -------
    dict[int, float]
        ``{residue_index: rns_score}`` for each binding-site residue whose
        neighbourhood yields valid entropy / BLOSUM estimates.
    """
    from Bio.PDB import NeighborSearch
    from Bio.AlignIO import read as alignio_read
    from Bio.Align import substitution_matrices
    import numpy as np
    from scipy.stats import entropy as shannon_entropy

    # 1. Load BLOSUM matrix
    blosum = substitution_matrices.load(blosum_name)

    # 2. Load MSA and map query sequence positions to MSA column indices
    msa_path = Path(msa_path)
    msa = alignio_read(str(msa_path), "stockholm")
    if len(msa) == 0:
        return {}

    query_seq = str(msa[0].seq)
    # Build a 1-based position → 0-based column-index mapping for the query.
    # Columns where the query has '-' or '.' are insertions in other sequences
    # relative to the query; they don't correspond to any query residue.
    seq_pos_to_col: dict[int, int] = {}
    seq_pos = 0
    for col_idx, aa in enumerate(query_seq):
        if aa not in ("-", "."):
            seq_pos += 1
            seq_pos_to_col[seq_pos] = col_idx

    # 3. Collect all Cα atoms from ATOM records (exclude HETATM via id[0] == " ")
    ca_atoms = [
        atom
        for atom in structure.get_atoms()
        if atom.get_name() == "CA" and atom.get_parent().id[0] == " "
    ]
    ns = NeighborSearch(ca_atoms)

    # Map residue number → Bio.PDB.Residue (first model, all chains)
    res_by_num: dict[int, Any] = {}
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] == " ":  # standard (ATOM) residue
                    res_by_num[res.id[1]] = res

    # 4. Compute RNS for each binding-site residue
    rns_scores: dict[int, float] = {}
    # Pre-compute max entropy for normalisation: log2(21) symbols (20 AAs + gap)
    max_entropy = float(np.log2(21))
    amino_acids_and_gap = "ACDEFGHIKLMNPQRSTVWY-"

    for residue_idx in binding_site:
        target_res = res_by_num.get(residue_idx)
        if target_res is None:
            continue  # residue absent from structure → skip silently

        try:
            target_ca = target_res["CA"]
        except KeyError:
            continue  # Cα absent (e.g. Gly with missing backbone) → skip

        # Spatial neighbours
        spatial_neighbors = ns.search(
            target_ca.coord, neighbor_radius_angstrom, level="R"
        )
        spatial_idxs = {r.id[1] for r in spatial_neighbors if r.id[0] == " "}

        # Sequence neighbours (±window positions)
        seq_idxs = {
            residue_idx + d
            for d in range(-sequence_window, sequence_window + 1)
        }

        # Union of spatial and sequence neighbourhoods (including the residue itself)
        neighborhood = spatial_idxs | seq_idxs

        # 5. For each neighbour, compute MSA-column entropy + BLOSUM weight
        col_entropies: list[float] = []
        col_blosum_weights: list[float] = []

        for nbr_idx in neighborhood:
            col_idx = seq_pos_to_col.get(nbr_idx)
            if col_idx is None:
                continue  # neighbour not in MSA mapping

            # Extract the column (upper-case, preserving gaps)
            column = [str(rec.seq)[col_idx].upper() for rec in msa]

            # Shannon entropy: count all symbols including gaps ('-') to
            # penalise poorly-covered columns.  Dots ('.') are folded into '-'.
            column_norm = [c if c != "." else "-" for c in column]
            freqs: dict[str, int] = {}
            for aa in column_norm:
                freqs[aa] = freqs.get(aa, 0) + 1

            total = sum(freqs.values())
            if total == 0:
                continue

            # Build probability vector over the 21-symbol alphabet
            counts = np.array(
                [freqs.get(c, 0) for c in amino_acids_and_gap], dtype=float
            )
            nonzero = counts[counts > 0]
            probs = nonzero / nonzero.sum()
            ent = float(shannon_entropy(probs, base=2))
            col_entropies.append(ent)

            # BLOSUM weight from most-frequent real amino acid in the column.
            # Exclude gap when picking the dominant AA for BLOSUM lookup.
            real_freqs = {k: v for k, v in freqs.items() if k in "ACDEFGHIKLMNPQRSTVWY"}
            if real_freqs:
                most_freq_aa = max(real_freqs, key=lambda k: real_freqs[k])
                try:
                    blosum_self = float(blosum[(most_freq_aa, most_freq_aa)])
                    # Normalise to [0, 1]; BLOSUM62 diagonal ≈ 4–11
                    col_blosum_weights.append(max(0.0, blosum_self / 11.0))
                except (KeyError, ValueError):
                    col_blosum_weights.append(0.5)  # unknown AA: neutral weight
            # If the column is all gaps, contribute no BLOSUM weight entry
            # (and the entropy already captures the poor coverage)

        if not col_entropies:
            continue  # no valid columns in neighbourhood → exclude residue

        # 6. Aggregate: lower entropy → higher conservation → higher RNS
        normalized_entropy = float(np.mean(col_entropies)) / max_entropy
        mean_blosum_weight = (
            float(np.mean(col_blosum_weights)) if col_blosum_weights else 0.5
        )
        rns = max(0.0, min(1.0, (1.0 - normalized_entropy) * mean_blosum_weight))
        rns_scores[residue_idx] = rns

    return rns_scores


def _get_residue_plddt(structure: Any, residue_idx: int) -> float:
    """Extract pLDDT from CA atom's bfactor field. Returns 50.0 if residue/CA missing."""
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] == " " and res.id[1] == residue_idx:
                    if "CA" in res:
                        return float(res["CA"].get_bfactor())
                    return 50.0
    return 50.0


def aggregate_target_rns(
    per_residue_rns: dict[int, float],
    provenance: dict,
    structure: Any,
    use_plddt_weighting: bool = True,
) -> float:
    """Aggregate per-residue RNS to a single per-target value.

    For AlphaFold structures, applies pLDDT weighting (spec 5.4 Tier 2):
    residues below pLDDT 50 contribute nothing, residues at 90+ contribute fully.
    PDB structures use uniform weights.

    Formula (AlphaFold + use_plddt_weighting=True):
        target_RNS = sum(rns × w) / sum(w)
        where w = max(0.0, (pLDDT - 50) / 50)

    Formula (PDB or use_plddt_weighting=False):
        target_RNS = mean(rns)  (uniform weighting)

    Returns nan if per_residue_rns is empty or if the AlphaFold case has zero
    total weight (all binding-site residues have pLDDT < 50).
    """
    if not per_residue_rns:
        return float("nan")

    use_af_weighting = (
        use_plddt_weighting
        and provenance.get("source") == "AlphaFold"
        and structure is not None
    )

    if use_af_weighting:
        weighted_sum = 0.0
        total_weight = 0.0
        for residue_idx, rns in per_residue_rns.items():
            plddt = _get_residue_plddt(structure, residue_idx)
            w = max(0.0, (plddt - 50.0) / 50.0)
            weighted_sum += rns * w
            total_weight += w
        if total_weight == 0.0:
            return float("nan")
        return weighted_sum / total_weight
    else:
        # Uniform weighting (PDB or use_plddt_weighting=False)
        values = list(per_residue_rns.values())
        return sum(values) / len(values)


def validation_gate(
    cache_dir: Path,
    db_path: Path,
    reference_set: str = "prabakaran_bromberg",
    spearman_threshold: float = 0.7,
    mad_threshold: float = 0.10,
) -> tuple[bool, dict, Path]:
    """Run the full RNS pipeline on bundled reference proteins; compare to published.

    Returns
    -------
    (passed, deviations_dict, summary_csv_path)

    passed = True iff Spearman rho >= spearman_threshold OR MAD <= mad_threshold.
    summary_csv has one row per reference protein with: name, uniprot, our_rns,
    published_rns, abs_dev (or error field if pipeline failed for that protein).
    """
    import math
    import pandas as pd
    from scipy.stats import spearmanr

    if reference_set != "prabakaran_bromberg":
        raise ValueError(
            f"reference_set must be 'prabakaran_bromberg', got {reference_set!r}. "
            "Alternative reference sets are not yet supported."
        )

    # Load bundled reference data
    ref_data_path = Path(__file__).parent / "_rns_reference_data.json"
    with ref_data_path.open() as fh:
        ref_data = json.load(fh)

    reference_proteins = ref_data["reference_proteins"]
    cache_dir = Path(cache_dir)
    msa_dir = cache_dir / "msas"

    rows: list[dict] = []
    for protein in reference_proteins:
        name = protein["name"]
        uniprot = protein["uniprot"]
        published_rns = protein["published_target_rns"]
        binding_site = protein["binding_site_residues"]

        try:
            structure, prov = fetch_structure(
                uniprot, cache_dir, pdb_id=protein.get("pdb_id")
            )
            msa_path = compute_msa(uniprot, db_path, msa_dir)
            per_residue = compute_per_residue_rns(structure, binding_site, msa_path)
            our_rns = aggregate_target_rns(per_residue, prov, structure)
            rows.append({
                "name": name,
                "uniprot": uniprot,
                "our_rns": our_rns,
                "published_rns": published_rns,
                "abs_dev": abs(our_rns - published_rns) if not math.isnan(our_rns) else float("nan"),
                "error": None,
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("validation_gate: pipeline failed for %s (%s): %s", name, uniprot, exc)
            rows.append({
                "name": name,
                "uniprot": uniprot,
                "our_rns": float("nan"),
                "published_rns": published_rns,
                "abs_dev": float("nan"),
                "error": str(exc),
            })

    df = pd.DataFrame(rows)
    summary_csv_path = cache_dir / "rns_validation_summary.csv"
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv_path, index=False)

    # Filter to rows where pipeline succeeded (our_rns is not nan)
    succeeded = df[df["our_rns"].notna() & ~df["our_rns"].apply(math.isnan)]
    n_succeeded = len(succeeded)

    min_required = 5
    if n_succeeded < min_required:
        deviations = {
            "spearman_rho": float("nan"),
            "mad": float("nan"),
            "n_succeeded": n_succeeded,
            "diagnostic": (
                f"Only {n_succeeded} of {len(reference_proteins)} reference proteins "
                f"succeeded — minimum {min_required} required for a reliable gate."
            ),
        }
        return False, deviations, summary_csv_path

    rho, _ = spearmanr(succeeded["our_rns"].values, succeeded["published_rns"].values)
    mad = float(succeeded["abs_dev"].mean())

    passed = bool(rho >= spearman_threshold or mad <= mad_threshold)

    deviations = {
        "spearman_rho": float(rho),
        "mad": mad,
        "n_succeeded": n_succeeded,
    }

    return passed, deviations, summary_csv_path


def compute_conservation_entropy(
    binding_site: list[int],
    msa_path: Path,
) -> float:
    """Simpler fallback metric: mean Shannon entropy of binding-site MSA columns.

    Loads the Stockholm MSA, identifies the columns corresponding to binding-site
    residues (via the query-sequence gap mapping from the first MSA record), then
    computes Shannon entropy per column over the 20 amino acid types plus gap ('-').
    Returns the mean across binding-site columns.

    Used in parallel with :func:`compute_per_residue_rns` for the sensitivity
    analysis AND as the pivot metric if the Task 6 validation gate fails.

    A lower value means more conserved (more informative).  Typical range is
    ``[0, log2(21)] ≈ [0, 4.39]``.  Callers may normalise to ``[0, 1]`` by
    dividing by ``math.log2(21)``.

    Algorithm note: gap characters ('.' in Stockholm format) are folded into '-'
    before counting so that insertion-column conventions do not create spurious
    extra symbols.

    Parameters
    ----------
    binding_site:
        1-indexed residue numbers.
    msa_path:
        Stockholm-format MSA file.

    Returns
    -------
    float
        Mean column entropy (bits) over binding-site columns, or ``float('nan')``
        if no binding-site column could be mapped.
    """
    from Bio.AlignIO import read as alignio_read
    import numpy as np
    from scipy.stats import entropy as shannon_entropy

    msa_path = Path(msa_path)
    msa = alignio_read(str(msa_path), "stockholm")
    if len(msa) == 0:
        return float("nan")

    # Map 1-based query sequence positions to 0-based MSA column indices
    query_seq = str(msa[0].seq)
    seq_pos_to_col: dict[int, int] = {}
    seq_pos = 0
    for col_idx, aa in enumerate(query_seq):
        if aa not in ("-", "."):
            seq_pos += 1
            seq_pos_to_col[seq_pos] = col_idx

    col_entropies: list[float] = []
    for residue_idx in binding_site:
        col_idx = seq_pos_to_col.get(residue_idx)
        if col_idx is None:
            continue

        column = [str(rec.seq)[col_idx].upper() for rec in msa]
        # Fold '.' (insertion convention) into '-'
        column = [c if c != "." else "-" for c in column]

        freqs: dict[str, int] = {}
        for aa in column:
            freqs[aa] = freqs.get(aa, 0) + 1

        counts = np.array(list(freqs.values()), dtype=float)
        nonzero = counts[counts > 0]
        if nonzero.sum() == 0:
            continue
        probs = nonzero / nonzero.sum()
        col_entropies.append(float(shannon_entropy(probs, base=2)))

    if not col_entropies:
        return float("nan")
    return float(np.mean(col_entropies))
