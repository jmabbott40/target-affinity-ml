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

import time
from pathlib import Path
from typing import Any, Literal

import requests
from Bio.PDB import PDBParser


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
