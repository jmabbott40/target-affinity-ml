"""Fetch kinase bioactivity data from ChEMBL.

This module queries the ChEMBL database for protein kinase targets and their
associated bioactivity measurements (IC50, Ki, Kd). Raw data is saved as
parquet files for downstream processing.

How the ChEMBL API works
------------------------
ChEMBL is the largest open-access database of bioactivity data, containing
~2.4M compounds and ~20M activity records. We use the `chembl_webresource_client`
Python package, which provides a Django ORM-like interface to the ChEMBL REST API.

Key concepts:
    - **Target**: A biological entity (usually a protein) that a compound binds to.
      We filter for SINGLE PROTEIN targets (confidence that the measured activity
      corresponds to a specific protein, not a cell or complex).

    - **Activity**: A single measurement of a compound's effect on a target.
      Includes the measurement type (IC50/Ki/Kd), value, units, and the assay
      it came from.

    - **pChEMBL value**: A standardized potency measure = -log10(molar IC50/Ki/Kd).
      ChEMBL only assigns pChEMBL when the measurement has exact relation ('='),
      nanomolar units, and a positive value. This serves as an implicit quality
      filter — if pChEMBL is present, the measurement meets basic quality criteria.

    - **GO annotations**: Each target's protein components carry Gene Ontology (GO)
      molecular function annotations. Kinases are identified by GO terms like
      GO:0016301 (kinase activity) and GO:0004672 (protein kinase activity).

Data flow:
    1. Fetch all human SINGLE PROTEIN targets from ChEMBL
    2. Filter to kinases using GO molecular function annotations
    3. For each kinase target, fetch IC50/Ki/Kd activities with pChEMBL values
    4. Save targets and activities as parquet files in data/raw/

Usage:
    python -m target_affinity_ml.data.chembl_fetcher
    python -m target_affinity_ml.data.chembl_fetcher --max-targets 10  # Quick test
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
DEFAULT_CONFIG = Path("configs/dataset_v1.yaml")

# Columns to extract from ChEMBL activity records
ACTIVITY_COLUMNS = [
    "activity_id",
    "molecule_chembl_id",
    "canonical_smiles",
    "target_chembl_id",
    "standard_type",
    "standard_value",
    "standard_units",
    "standard_relation",
    "pchembl_value",
    "assay_chembl_id",
    "assay_type",
    "data_validity_comment",
]


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    """Load dataset configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# GO molecular function terms that identify kinase proteins.
# GO:0016301 (kinase activity) is the parent term. We include children
# to catch targets annotated only with specific sub-terms.
KINASE_GO_TERMS = {
    "GO:0016301",  # kinase activity
    "GO:0004672",  # protein kinase activity
    "GO:0004674",  # protein serine/threonine kinase activity
    "GO:0004713",  # protein tyrosine kinase activity
    "GO:0004714",  # transmembrane receptor protein tyrosine kinase activity
    "GO:0004712",  # protein serine/threonine/tyrosine kinase activity
    "GO:0016773",  # phosphotransferase activity, alcohol group as acceptor
    "GO:0004715",  # non-membrane spanning protein tyrosine kinase activity
}


def _classify_kinase(go_ids: set[str]) -> str:
    """Assign a broad kinase group from GO term annotations."""
    if go_ids & {"GO:0004713", "GO:0004714", "GO:0004715"}:
        return "Tyrosine kinase"
    if go_ids & {"GO:0004674"}:
        return "Serine/Threonine kinase"
    if go_ids & {"GO:0004712"}:
        return "Dual-specificity kinase"
    return "Other kinase"


def _is_kinase_by_name(target: dict) -> bool:
    """Check if target name/description indicates a kinase.

    Falls back to name-based matching for targets lacking GO annotations.
    Uses negative patterns to exclude false positives (phosphatases,
    phosphodiesterases, etc. that mention "kinase" in their description).
    """
    name = target.get("pref_name", "").lower()

    # Must have kinase in the name
    if "kinase" not in name:
        return False

    # Exclude common false positives
    exclude_patterns = [
        "phosphatase", "phosphodiesterase", "kinase-associated",
        "kinase interacting", "kinase interaction", "kinase anchor",
        "kinase regulatory", "kinase substrate", "kinase binding",
    ]
    return not any(pat in name for pat in exclude_patterns)


def _extract_kinase_records(targets: list[dict]) -> list[dict]:
    """Extract kinase records from target dicts using GO annotations + names.

    Identifies kinases by two complementary strategies:
    1. GO annotations: target component has kinase GO molecular function terms
    2. Name matching: target name contains "kinase" (catches targets with
       incomplete GO annotations)
    """
    kinase_records = []
    seen_ids = set()

    for t in targets:
        if t["target_chembl_id"] in seen_ids:
            continue

        is_kinase = False
        go_ids = set()

        for comp in t.get("target_components", []):
            xrefs = comp.get("target_component_xrefs", [])
            comp_go_ids = {
                x["xref_id"]
                for x in xrefs
                if x.get("xref_src_db") == "GoFunction"
            }
            go_ids |= comp_go_ids

            if comp_go_ids & KINASE_GO_TERMS:
                is_kinase = True

        # Fallback: name-based matching
        if not is_kinase:
            is_kinase = _is_kinase_by_name(t)

        if not is_kinase:
            continue

        seen_ids.add(t["target_chembl_id"])

        # Extract gene symbol from first component
        gene_symbol = "Unknown"
        for comp in t.get("target_components", []):
            for syn in comp.get("target_component_synonyms", []):
                if syn.get("syn_type") == "GENE_SYMBOL":
                    gene_symbol = syn["component_synonym"]
                    break
            if gene_symbol != "Unknown":
                break

        kinase_records.append(
            {
                "target_chembl_id": t["target_chembl_id"],
                "pref_name": t["pref_name"],
                "target_type": t["target_type"],
                "organism": t["organism"],
                "kinase_group": _classify_kinase(go_ids),
                "gene_symbol": gene_symbol,
            }
        )
    return kinase_records


def fetch_kinase_targets(organism: str = "Homo sapiens") -> pd.DataFrame:
    """Fetch all single-protein kinase targets from ChEMBL.

    Strategy (optimized two-pass approach):
        1. **Fast pass**: Search ChEMBL for targets mentioning "kinase" in any
           text field (~600 results vs ~5800 for all human targets). Filter
           by GO molecular function annotations to keep only true kinases.
        2. **Completeness check**: If the fast pass returns suspiciously few
           targets (<200), falls back to downloading all human single-protein
           targets and filtering in memory.

    The GO annotation filter is the ground truth — the search step is purely
    an optimization to reduce download size.

    Parameters
    ----------
    organism : str
        Species filter. Default is human targets only.

    Returns
    -------
    pd.DataFrame
        Kinase targets with columns: target_chembl_id, pref_name,
        target_type, organism, kinase_group, gene_symbol.
    """
    from chembl_webresource_client.new_client import new_client

    target_api = new_client.target

    # --- Fast pass: search-based approach ---
    logger.info("Searching ChEMBL for kinase targets (%s)...", organism)
    try:
        candidates = list(
            target_api.search("kinase").filter(
                target_type="SINGLE PROTEIN", organism=organism,
            )
        )
        logger.info("Search returned %d candidate targets", len(candidates))
        kinase_records = _extract_kinase_records(candidates)
    except Exception as e:
        logger.warning("Search-based fetch failed (%s), using full download", e)
        kinase_records = []

    # --- Fallback: full download if search found too few ---
    MIN_EXPECTED_KINASES = 100
    if len(kinase_records) < MIN_EXPECTED_KINASES:
        logger.info(
            "Fast search found only %d kinases (expected >=%d). "
            "Falling back to full target download (this may take a few minutes)...",
            len(kinase_records), MIN_EXPECTED_KINASES,
        )
        all_targets = list(
            target_api.filter(target_type="SINGLE PROTEIN", organism=organism)
        )
        logger.info("Downloaded %d total single-protein targets", len(all_targets))
        kinase_records = _extract_kinase_records(all_targets)

    if not kinase_records:
        logger.error("No kinase targets found! Check ChEMBL API connectivity.")
        return pd.DataFrame()

    df = pd.DataFrame(kinase_records).drop_duplicates(subset=["target_chembl_id"])
    logger.info(
        "Found %d kinase targets (%s)",
        len(df),
        df["kinase_group"].value_counts().to_dict(),
    )
    return df


def fetch_bioactivities(
    target_chembl_ids: list[str],
    activity_types: list[str] | None = None,
    max_targets: int | None = None,
) -> pd.DataFrame:
    """Fetch bioactivity measurements for given targets from ChEMBL.

    Queries the ChEMBL activity API for each target individually. This is
    slower than a bulk query but more reliable (avoids timeouts on large
    result sets) and allows progress tracking.

    Filters applied at the API level:
        - standard_type in activity_types (IC50, Ki, Kd)
        - standard_relation = '=' (exact measurements only, not '>' or '<')
        - pchembl_value is not null (implicit quality filter)

    Parameters
    ----------
    target_chembl_ids : list[str]
        ChEMBL target IDs to query.
    activity_types : list[str], optional
        Activity types to include. Default: ["IC50", "Ki", "Kd"].
    max_targets : int, optional
        Limit number of targets to query (for testing).

    Returns
    -------
    pd.DataFrame
        Raw bioactivity records with columns defined in ACTIVITY_COLUMNS.
    """
    from chembl_webresource_client.new_client import new_client

    if activity_types is None:
        activity_types = ["IC50", "Ki", "Kd"]

    activity_api = new_client.activity

    if max_targets is not None:
        target_chembl_ids = target_chembl_ids[:max_targets]

    n_targets = len(target_chembl_ids)
    logger.info(
        "Fetching activities for %d targets (types: %s)...",
        n_targets,
        activity_types,
    )

    all_activities = []
    failed_targets = []

    for i, target_id in enumerate(target_chembl_ids):
        if (i + 1) % 25 == 0 or i == 0:
            logger.info(
                "  Progress: %d/%d targets (%.0f%%) — %d activities so far",
                i + 1,
                n_targets,
                (i + 1) / n_targets * 100,
                len(all_activities),
            )

        try:
            results = activity_api.filter(
                target_chembl_id=target_id,
                standard_type__in=activity_types,
                standard_relation="=",
                pchembl_value__isnull=False,
            )

            # Convert queryset to list of dicts, keeping only needed columns
            for record in results:
                row = {col: record.get(col) for col in ACTIVITY_COLUMNS}
                all_activities.append(row)

        except Exception as e:
            logger.warning("Failed to fetch target %s: %s", target_id, e)
            failed_targets.append(target_id)
            time.sleep(2)  # Back off on errors

    if failed_targets:
        logger.warning(
            "%d targets failed: %s", len(failed_targets), failed_targets[:10]
        )

    df = pd.DataFrame(all_activities)
    logger.info(
        "Fetched %d total activity records for %d targets",
        len(df),
        n_targets - len(failed_targets),
    )

    # Log summary statistics
    if len(df) > 0:
        logger.info("  Unique compounds: %d", df["molecule_chembl_id"].nunique())
        logger.info("  Unique targets: %d", df["target_chembl_id"].nunique())
        logger.info(
            "  Activity types: %s",
            df["standard_type"].value_counts().to_dict(),
        )

    return df


def main() -> None:
    """Run the full data fetching pipeline."""
    parser = argparse.ArgumentParser(description="Fetch kinase data from ChEMBL")
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Limit number of targets (for testing). Default: all.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to dataset config YAML.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if cached files exist.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    targets_path = RAW_DATA_DIR / "chembl_kinase_targets.parquet"
    activities_path = RAW_DATA_DIR / "chembl_kinase_activities.parquet"

    # Step 1: Fetch kinase targets (or load from cache)
    if targets_path.exists() and not args.force:
        logger.info("=== Step 1: Loading cached targets from %s ===", targets_path)
        targets = pd.read_parquet(targets_path)
        logger.info("Loaded %d cached kinase targets", len(targets))
    else:
        logger.info("=== Step 1: Fetching kinase targets from ChEMBL ===")
        targets = fetch_kinase_targets(organism=config["source"]["organism"])
        targets.to_parquet(targets_path, index=False)
        logger.info("Saved %d targets to %s", len(targets), targets_path)

    # Step 2: Fetch bioactivity data (or load from cache)
    if activities_path.exists() and not args.force:
        logger.info("=== Step 2: Loading cached activities from %s ===", activities_path)
        activities = pd.read_parquet(activities_path)
        logger.info("Loaded %d cached activity records", len(activities))
    else:
        logger.info("=== Step 2: Fetching bioactivity data from ChEMBL ===")
        activities = fetch_bioactivities(
            target_chembl_ids=targets["target_chembl_id"].tolist(),
            activity_types=config["activity"]["types"],
            max_targets=args.max_targets,
        )
        activities.to_parquet(activities_path, index=False)
        logger.info("Saved %d activities to %s", len(activities), activities_path)

    # Step 3: Summary
    logger.info("=== Fetch complete ===")
    logger.info("  Targets: %d", len(targets))
    logger.info("  Activities: %d", len(activities))
    if len(activities) > 0:
        logger.info(
            "  Unique compounds: %d", activities["molecule_chembl_id"].nunique()
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
