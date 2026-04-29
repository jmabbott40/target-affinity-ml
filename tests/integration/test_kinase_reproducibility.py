"""Integration test: RF on kinase random split with library v1.0
matches preprint v1 numerical output within tolerance.

This is the validation gate for the library refactor. Failure here
indicates the refactor changed numerical behavior — investigate and
fix before proceeding to Plan 2 (GPCR work).

Reference values come from recomputing metrics on saved preprint v1
prediction NPZ files (see scripts/extract_reference_metrics.py).
"""
import json
from pathlib import Path

import pytest

try:
    from target_affinity_ml.training import train_and_evaluate
except ImportError as e:
    train_and_evaluate = None
    _IMPORT_ERROR = str(e)

REFERENCE_PATH = Path(__file__).parent / "kinase_v1_reference.json"
KINASE_REPO = Path("/Users/joshuaabbott/mlproject")


@pytest.fixture(scope="session")
def reference():
    if not REFERENCE_PATH.exists():
        pytest.skip(
            f"Reference file missing: {REFERENCE_PATH}. "
            "Run scripts/extract_reference_metrics.py first."
        )
    with open(REFERENCE_PATH) as f:
        return json.load(f)


@pytest.mark.slow
@pytest.mark.skipif(
    train_and_evaluate is None,
    reason="train_and_evaluate not importable from library",
)
def test_rf_random_matches_preprint_v1(reference):
    """RF on random split reproduces preprint v1 metrics within tolerance."""
    config_path = KINASE_REPO / "configs" / "rf_baseline.yaml"
    dataset_dir = KINASE_REPO / "data" / "processed" / "v1"

    if not config_path.exists() or not dataset_dir.exists():
        pytest.skip(
            f"Kinase repo data not available at {dataset_dir}. "
            "Integration test requires kinase repo cloned + data downloaded."
        )

    result = train_and_evaluate(
        config_path=str(config_path),
        split_strategy=reference["split"],
        dataset_version="v1",
        dataset_dir=str(dataset_dir),
    )

    actual = result["test_metrics"]
    expected = reference["metrics"]
    tolerance = reference["tolerance"]

    failures = []
    for metric in ["rmse", "r2", "pearson_r"]:
        if metric not in actual:
            failures.append(f"  {metric}: not in actual metrics dict")
            continue
        diff = abs(actual[metric] - expected[metric])
        if diff > tolerance[metric]:
            failures.append(
                f"  {metric}: got {actual[metric]:.6f}, "
                f"expected {expected[metric]:.6f}, diff={diff:.6f} "
                f"> tolerance {tolerance[metric]}"
            )

    assert not failures, (
        f"\nRF random reference mismatch:\n" + "\n".join(failures)
    )
