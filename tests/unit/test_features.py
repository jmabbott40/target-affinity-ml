"""Tests for molecular feature generation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


class TestMorganFingerprints:
    """Test Morgan fingerprint generation."""

    def test_output_shape(self):
        """Fingerprint should have correct number of bits."""
        from target_affinity_ml.features.fingerprints import smiles_to_morgan_fp

        fp = smiles_to_morgan_fp("c1ccccc1", radius=2, n_bits=2048)
        assert fp.shape == (2048,)

    def test_binary_values(self):
        """Fingerprint values should be 0 or 1."""
        from target_affinity_ml.features.fingerprints import smiles_to_morgan_fp

        fp = smiles_to_morgan_fp("c1ccccc1")
        assert set(np.unique(fp)).issubset({0, 1})

    def test_invalid_smiles_returns_none(self):
        """Invalid SMILES should return None."""
        from target_affinity_ml.features.fingerprints import smiles_to_morgan_fp

        fp = smiles_to_morgan_fp("not_valid")
        assert fp is None

    def test_batch_computation(self):
        """Batch computation should return matrix of correct shape."""
        from target_affinity_ml.features.fingerprints import compute_fingerprints

        smiles = ["c1ccccc1", "CCO", "CC(=O)O"]
        fps = compute_fingerprints(smiles, n_bits=1024)
        assert fps.shape == (3, 1024)


class TestRDKitDescriptors:
    """Test RDKit descriptor computation."""

    def test_returns_dict(self):
        """Single molecule should return descriptor dictionary."""
        from target_affinity_ml.features.descriptors import smiles_to_descriptors

        desc = smiles_to_descriptors("c1ccccc1")
        assert isinstance(desc, dict)
        assert len(desc) > 100  # RDKit computes ~200 descriptors

    def test_batch_returns_matrix(self):
        """Batch computation should return (n_mols, n_desc) matrix."""
        from target_affinity_ml.features.descriptors import compute_descriptors

        smiles = ["c1ccccc1", "CCO", "CC(=O)O"]
        matrix, names = compute_descriptors(smiles)
        assert matrix.shape[0] == 3
        assert len(names) == matrix.shape[1]

    def test_invalid_smiles_handled(self):
        """Invalid SMILES in batch should not crash; row gets NaN fill."""
        from target_affinity_ml.features.descriptors import smiles_to_descriptors

        desc = smiles_to_descriptors("not_valid_smiles")
        assert desc is None


class TestDataDirParameter:
    """Test that feature loaders honour the data_dir parameter."""

    def test_load_morgan_fingerprints_reads_from_data_dir(self, tmp_path):
        """load_morgan_fingerprints(data_dir=...) reads from the supplied directory."""
        from target_affinity_ml.features import load_morgan_fingerprints

        # Build synthetic fixture under tmp_path/v1/features/
        features_dir = tmp_path / "v1" / "features"
        features_dir.mkdir(parents=True)

        smiles = ["c1ccccc1", "CCO"]
        fp_matrix = np.zeros((2, 2048), dtype=np.uint8)
        np.savez_compressed(features_dir / "morgan_fp.npz", fingerprints=fp_matrix)
        with open(features_dir / "smiles_index.json", "w") as f:
            json.dump(smiles, f)

        fps, smiles_out = load_morgan_fingerprints(version="v1", data_dir=tmp_path)

        assert fps.shape == (2, 2048)
        assert smiles_out == smiles

    def test_compute_and_cache_features_writes_under_data_dir(self, tmp_path):
        """compute_and_cache_features(data_dir=...) writes outputs under the supplied directory."""
        from target_affinity_ml.features import compute_and_cache_features

        # Minimal config YAML
        config = {
            "version": "vtest",
            "features": {
                "morgan": {"radius": 2, "n_bits": 128},
                "descriptors": {"drop_missing_threshold": 0.05},
            },
        }
        config_path = tmp_path / "dataset_vtest.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Minimal curated-activities parquet that the loader expects
        curated_dir = tmp_path / "vtest"
        curated_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "std_smiles": ["c1ccccc1", "CCO", "CC(=O)O"],
            "pactivity": [6.0, 5.5, 7.0],
        })
        df.to_parquet(curated_dir / "curated_activities.parquet", index=False)

        saved = compute_and_cache_features(
            config_path=config_path,
            data_dir=tmp_path,
        )

        # All output paths must live under tmp_path, not under cwd
        for key, path in saved.items():
            assert Path(path).is_relative_to(tmp_path), (
                f"Output '{key}' landed outside data_dir: {path}"
            )
