#!/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python
"""CLI driver for the RNS validation gate.

Usage
-----
    # Local (no real jackhmmer; will fail on the MSA step — meant for unit-test pass)
    python scripts/run_rns_validation_gate.py --cache-dir /tmp/rns_validation

    # AWS (with real UniRef50 and libstdc++ fix)
    LD_LIBRARY_PATH=~/miniforge3/envs/kinase-affinity/lib:$LD_LIBRARY_PATH \\
        ~/miniforge3/envs/kinase-affinity/bin/python scripts/run_rns_validation_gate.py \\
        --cache-dir ~/rns_validation \\
        --db ~/databases/uniref50.fasta

Exit code: 0 if gate passes, 1 if gate fails.
"""
import argparse
import sys
from pathlib import Path

from target_affinity_ml.benchmarks.rns_scoring import validation_gate


def main():
    parser = argparse.ArgumentParser(description="RNS validation gate driver")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("/home/ubuntu/databases/uniref50.fasta"))
    parser.add_argument("--spearman-threshold", type=float, default=0.7)
    parser.add_argument("--mad-threshold", type=float, default=0.10)
    args = parser.parse_args()

    passed, deviations, csv_path = validation_gate(
        cache_dir=args.cache_dir,
        db_path=args.db,
        spearman_threshold=args.spearman_threshold,
        mad_threshold=args.mad_threshold,
    )
    print("=" * 60)
    print(f"VALIDATION GATE: {'PASS' if passed else 'FAIL'}")
    print(f"  spearman_rho        = {deviations.get('spearman_rho', float('nan')):.4f}")
    print(f"  mean abs deviation  = {deviations.get('mad', float('nan')):.4f}")
    print(f"  n succeeded         = {deviations.get('n_succeeded', 0)}")
    print(f"  summary csv         = {csv_path}")
    print("=" * 60)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
