"""Integration smoke test: ESM-FP MLP deep model dispatcher (L4).

Verifies that ``deep_train_and_evaluate`` actually dispatches to and runs
the ESM-FP MLP code path end-to-end, including model instantiation, the
training loop, MC-Dropout uncertainty collection, and metric computation.

WHY THIS TEST EXISTS
--------------------
Plan 1 integration coverage was limited to Random Forest.  A bug in the
deep-model dispatcher could slip through undetected.  This test exercises
the real training loop on a tiny synthetic dataset — no mocks of the
trainer itself — so dispatch bugs, shape mismatches, and forward-pass
errors all surface here.

DESIGN
------
The test creates a minimal but *genuine* training run:

* 20 synthetic compounds × 2 targets, random pActivity values.
* fp_dim=4, esm_dim=4 → input_dim=8 (avoids 3328-dim parameter count).
* hidden_dims=[8], 1 epoch, batch_size=8.
* ``load_morgan_fingerprints`` and ``load_esm2_embeddings`` are patched in
  the deep_trainer module namespace to return in-memory arrays — these are
  pure I/O helpers that load pre-computed caches; patching them keeps the
  test hermetic without hiding any training logic.
* RESULTS_DIR is redirected to a temp directory so no stray files are
  written to the source tree.

The skip guard (``pytest.importorskip``) keeps CI on torch-free hosts
clean; locally torch IS installed so the test runs and passes.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Synthetic dataset constants
# ---------------------------------------------------------------------------

N_COMPOUNDS = 20       # rows in the synthetic activities table
N_TARGETS = 2
FP_DIM = 4             # tiny fingerprint dimension
ESM_DIM = 4            # tiny ESM embedding dimension
INPUT_DIM = FP_DIM + ESM_DIM  # 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_smiles(n: int) -> list[str]:
    """Return n distinct placeholder SMILES strings."""
    return [f"C{'C' * i}" for i in range(n)]


def _make_activities_df(smiles: list[str], n_targets: int) -> pd.DataFrame:
    """Build a minimal curated_activities DataFrame."""
    rng = np.random.default_rng(0)
    target_ids = [f"CHEMBL000{i}" for i in range(n_targets)]

    rows = []
    for i, smi in enumerate(smiles):
        tid = target_ids[i % n_targets]
        pact = float(rng.uniform(4.0, 8.0))
        rows.append({
            "std_smiles": smi,
            "target_chembl_id": tid,
            "pactivity": pact,
            "is_active": int(pact >= 6.0),
        })
    return pd.DataFrame(rows)


def _make_split_indices(n: int) -> dict[str, list[int]]:
    """Random 60/20/20 split over n rows."""
    idx = list(range(n))
    train = idx[:12]
    val = idx[12:16]
    test = idx[16:]
    return {"train": train, "val": val, "test": test}


def _make_config_yaml(tmp_dir: Path) -> Path:
    """Write a tiny esm_fp_mlp config YAML and return its path."""
    cfg = {
        "model": {"name": "esm_fp_mlp", "type": "regression"},
        "features": {"ligand": "morgan_fingerprint", "protein": "esm2_embedding"},
        "hyperparameters": {
            "input_dim": INPUT_DIM,
            "hidden_dims": [8],
            "dropout": 0.1,
            "batch_size": 8,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "max_epochs": 1,
            "patience": 10,
            "lr_scheduler": "cosine",
        },
        "uncertainty": {
            "method": "mc_dropout",
            "mc_dropout_samples": 2,
        },
    }
    path = tmp_dir / "esm_fp_mlp_smoke.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_esm_fp_mlp_smoke():
    """ESM-FP MLP dispatcher runs end-to-end and returns test_rmse."""
    from target_affinity_ml.training.deep_trainer import deep_train_and_evaluate

    smiles = _make_synthetic_smiles(N_COMPOUNDS)
    df = _make_activities_df(smiles, N_TARGETS)
    split_indices = _make_split_indices(N_COMPOUNDS)

    # Synthetic feature arrays
    rng = np.random.default_rng(1)
    fp_matrix = rng.random((N_COMPOUNDS, FP_DIM), dtype=float).astype(np.float32)
    esm_matrix = rng.random((N_TARGETS, ESM_DIM), dtype=float).astype(np.float32)
    target_to_row = {f"CHEMBL000{i}": i for i in range(N_TARGETS)}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Write curated_activities.parquet
        parquet_path = tmp_dir / "curated_activities.parquet"
        df.to_parquet(parquet_path, index=False)

        # Write split JSON
        splits_dir = tmp_dir / "splits"
        splits_dir.mkdir()
        with open(splits_dir / "random_split.json", "w") as f:
            json.dump(split_indices, f)

        # Write tiny config
        config_path = _make_config_yaml(tmp_dir)

        # Redirect RESULTS_DIR so model checkpoints go to tmp
        with (
            patch(
                "target_affinity_ml.training.deep_trainer.load_morgan_fingerprints",
                return_value=(fp_matrix, smiles),
            ),
            patch(
                "target_affinity_ml.training.deep_trainer.load_esm2_embeddings",
                return_value=(esm_matrix, target_to_row),
            ),
            patch(
                "target_affinity_ml.training.deep_trainer.RESULTS_DIR",
                tmp_dir / "results",
            ),
        ):
            metrics = deep_train_and_evaluate(
                config_path=config_path,
                split_strategy="random",
                dataset_version="smoke",
                training_seed=42,
                data_dir_override=tmp_dir,
            )

    # Primary assertion: the dispatcher ran, returned a dict, and computed RMSE.
    assert isinstance(metrics, dict), "deep_train_and_evaluate must return a dict"
    assert "test_rmse" in metrics, (
        f"Expected 'test_rmse' in metrics; got keys: {list(metrics.keys())}"
    )
    assert isinstance(metrics["test_rmse"], float), "test_rmse must be a float"
    assert metrics["test_rmse"] >= 0.0, "RMSE must be non-negative"

    # Confirm the dispatcher used the ESM-FP MLP path (not GNN/fusion).
    assert metrics.get("model") == "esm_fp_mlp", (
        f"Expected model='esm_fp_mlp', got: {metrics.get('model')}"
    )
