"""Fetch protein sequences for kinase targets from ChEMBL/UniProt.

Maps target_chembl_id → UniProt accession → amino acid sequence.
Sequences are needed for ESM-2 protein language model embeddings.

Pipeline:
    1. Load unique kinase targets from curated data
    2. Query ChEMBL API for UniProt cross-references
    3. Fetch sequences from UniProt REST API
    4. Save as JSON cache

Usage:
    python -m target_affinity_ml.data.protein_sequences
    python -m target_affinity_ml.data.protein_sequences --max-targets 10  # test
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

UNIPROT_API = "https://rest.uniprot.org/uniprotkb"


def fetch_uniprot_accessions(
    target_chembl_ids: list[str],
) -> dict[str, str]:
    """Query ChEMBL for UniProt accessions for each target.

    Extracts UniProt cross-references from target_component_xrefs.

    Parameters
    ----------
    target_chembl_ids : list[str]
        ChEMBL target identifiers.

    Returns
    -------
    dict[str, str]
        Mapping of target_chembl_id → UniProt accession.
    """
    from chembl_webresource_client.new_client import new_client

    target_api = new_client.target
    target_to_uniprot = {}
    failed = []

    for i, tid in enumerate(target_chembl_ids):
        if (i + 1) % 50 == 0:
            logger.info("  Fetching UniProt for target %d/%d...", i + 1, len(target_chembl_ids))

        try:
            target = target_api.get(tid)
            if target is None:
                failed.append(tid)
                continue

            # Extract UniProt accessions from target components
            uniprot_ids = []
            for comp in target.get("target_components", []):
                for xref in comp.get("target_component_xrefs", []):
                    if xref.get("xref_src_db") == "UniProt":
                        uniprot_ids.append(xref["xref_id"])

            if uniprot_ids:
                # Prefer Swiss-Prot (reviewed) entries — typically 6 chars
                # TrEMBL (unreviewed) entries are typically longer
                reviewed = [u for u in uniprot_ids if len(u) == 6]
                target_to_uniprot[tid] = reviewed[0] if reviewed else uniprot_ids[0]
            else:
                failed.append(tid)

            # Rate limiting
            if (i + 1) % 20 == 0:
                time.sleep(0.5)

        except Exception as e:
            logger.warning("Failed to fetch target %s: %s", tid, e)
            failed.append(tid)

    logger.info("UniProt accessions found: %d/%d (failed: %d)",
                len(target_to_uniprot), len(target_chembl_ids), len(failed))
    if failed:
        logger.info("  Failed targets: %s", failed[:10])

    return target_to_uniprot


def fetch_sequences_from_uniprot(
    uniprot_ids: list[str],
    batch_size: int = 50,
) -> dict[str, str]:
    """Fetch protein sequences from UniProt REST API.

    Parameters
    ----------
    uniprot_ids : list[str]
        UniProt accession numbers.
    batch_size : int
        Number of sequences to fetch per request.

    Returns
    -------
    dict[str, str]
        Mapping of UniProt accession → amino acid sequence.
    """
    sequences = {}
    unique_ids = list(set(uniprot_ids))

    for i in range(0, len(unique_ids), batch_size):
        batch = unique_ids[i:i + batch_size]
        query = " OR ".join(f"accession:{uid}" for uid in batch)

        try:
            resp = requests.get(
                f"{UNIPROT_API}/search",
                params={
                    "query": query,
                    "format": "json",
                    "fields": "accession,sequence",
                    "size": batch_size,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            for result in data.get("results", []):
                acc = result["primaryAccession"]
                seq = result.get("sequence", {}).get("value", "")
                if seq:
                    sequences[acc] = seq

            logger.info("  Fetched sequences %d-%d/%d",
                        i + 1, min(i + batch_size, len(unique_ids)), len(unique_ids))
            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            logger.warning("Failed to fetch batch starting at %d: %s", i, e)
            # Fall back to individual fetches
            for uid in batch:
                try:
                    resp = requests.get(
                        f"{UNIPROT_API}/{uid}.fasta",
                        timeout=15,
                    )
                    if resp.ok:
                        lines = resp.text.strip().split("\n")
                        seq = "".join(lines[1:])  # Skip header
                        sequences[uid] = seq
                    time.sleep(0.3)
                except Exception:
                    logger.warning("Failed individual fetch for %s", uid)

    logger.info("Sequences retrieved: %d/%d", len(sequences), len(unique_ids))
    return sequences


def build_protein_sequence_cache(
    dataset_version: str = "v1",
    max_targets: int | None = None,
) -> dict:
    """Build and save protein sequence cache for all kinase targets.

    Parameters
    ----------
    dataset_version : str
        Dataset version.
    max_targets : int, optional
        Limit number of targets (for testing).

    Returns
    -------
    dict
        Full sequence cache: {target_chembl_id: {uniprot_id, gene_symbol, sequence, length}}.
    """
    data_dir = DATA_DIR / dataset_version

    # Load unique targets from curated data
    df = pd.read_parquet(data_dir / "curated_activities.parquet")
    targets = df[["target_chembl_id", "gene_symbol", "pref_name"]].drop_duplicates(
        subset="target_chembl_id"
    )
    logger.info("Unique kinase targets: %d", len(targets))

    target_ids = targets["target_chembl_id"].tolist()
    if max_targets:
        target_ids = target_ids[:max_targets]
        logger.info("Limiting to %d targets for testing", max_targets)

    # Step 1: Get UniProt accessions from ChEMBL
    logger.info("Step 1: Fetching UniProt accessions from ChEMBL...")
    target_to_uniprot = fetch_uniprot_accessions(target_ids)

    # Step 2: Fetch sequences from UniProt
    logger.info("Step 2: Fetching sequences from UniProt...")
    uniprot_ids = list(target_to_uniprot.values())
    uniprot_to_seq = fetch_sequences_from_uniprot(uniprot_ids)

    # Step 3: Build final cache
    gene_map = dict(zip(targets["target_chembl_id"], targets["gene_symbol"], strict=False))
    name_map = dict(zip(targets["target_chembl_id"], targets["pref_name"], strict=False))

    cache = {}
    for tid in target_ids:
        if tid not in target_to_uniprot:
            continue
        uid = target_to_uniprot[tid]
        if uid not in uniprot_to_seq:
            continue

        seq = uniprot_to_seq[uid]
        cache[tid] = {
            "uniprot_id": uid,
            "gene_symbol": gene_map.get(tid, "Unknown"),
            "pref_name": name_map.get(tid, "Unknown"),
            "sequence": seq,
            "length": len(seq),
        }

    logger.info("Final cache: %d targets with sequences (out of %d)",
                len(cache), len(target_ids))

    # Save
    cache_path = data_dir / "protein_sequences.json"
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    logger.info("Saved to %s", cache_path)

    # Summary stats
    lengths = [v["length"] for v in cache.values()]
    if lengths:
        logger.info("Sequence lengths: min=%d, max=%d, mean=%.0f, median=%.0f",
                     min(lengths), max(lengths),
                     sum(lengths) / len(lengths),
                     sorted(lengths)[len(lengths) // 2])

    return cache


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch protein sequences for kinase targets",
    )
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--max-targets", type=int, default=None,
                        help="Limit targets for testing")
    args = parser.parse_args()

    build_protein_sequence_cache(args.dataset_version, args.max_targets)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
