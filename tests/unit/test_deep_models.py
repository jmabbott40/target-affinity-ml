"""Tests for Phase 7 deep learning models and molecular graph conversion.

All tests use small synthetic data and run on CPU.
PyTorch and torch-geometric must be installed.
"""

import tempfile

import numpy as np
import pytest

# Skip all tests if PyTorch or PyG not installed
torch = pytest.importorskip("torch")
torch_geometric = pytest.importorskip("torch_geometric")

# E402 suppressed: torch.utils.data must be imported AFTER importorskip,
# otherwise the module-level import would crash before pytest can skip the file.
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

# ---- Molecular graph conversion tests ----

class TestMolecularGraphConversion:
    def test_ethanol(self):
        from target_affinity_ml.features.molecular_graphs import smiles_to_graph
        data = smiles_to_graph("CCO")
        assert data is not None
        assert data.x.shape[0] == 3  # 3 heavy atoms
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == 4  # 2 bonds × 2 directions

    def test_benzene(self):
        from target_affinity_ml.features.molecular_graphs import smiles_to_graph
        data = smiles_to_graph("c1ccccc1")
        assert data is not None
        assert data.x.shape[0] == 6  # 6 carbons
        assert data.edge_index.shape[1] == 12  # 6 bonds × 2

    def test_feature_dimensions(self):
        from target_affinity_ml.features.molecular_graphs import (
            get_atom_feature_dim,
            get_bond_feature_dim,
            smiles_to_graph,
        )
        data = smiles_to_graph("CCO")
        assert data.x.shape[1] == get_atom_feature_dim()
        assert data.edge_attr.shape[1] == get_bond_feature_dim()

    def test_invalid_smiles(self):
        from target_affinity_ml.features.molecular_graphs import smiles_to_graph
        result = smiles_to_graph("not_a_smiles")
        assert result is None

    def test_complex_molecule(self):
        from target_affinity_ml.features.molecular_graphs import smiles_to_graph
        # Aspirin
        data = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")
        assert data is not None
        assert data.x.shape[0] == 13  # 13 heavy atoms
        assert data.edge_attr.shape[0] > 0


# ---- ESM-FP MLP model tests ----

class TestESMFPMLPModel:
    def test_fit_predict(self):
        from target_affinity_ml.models.esm_fp_mlp_model import ESMFPMLPModel

        model = ESMFPMLPModel(input_dim=64, hidden_dims=[32, 16])
        X = torch.randn(50, 64)
        y = torch.randn(50)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)

        # Forward pass
        model.eval()
        with torch.no_grad():
            pred = model(X)
        assert pred.shape == (50, 1)

        # Predict via base class
        preds = model.predict(loader)
        assert preds.shape == (50,)

    def test_uncertainty(self):
        from target_affinity_ml.models.esm_fp_mlp_model import ESMFPMLPModel

        model = ESMFPMLPModel(input_dim=64, hidden_dims=[32, 16], dropout=0.3)
        X = torch.randn(30, 64)
        y = torch.randn(30)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)

        mean, std = model.predict_with_uncertainty(loader, n_samples=5)
        assert mean.shape == (30,)
        assert std.shape == (30,)
        assert np.all(std >= 0)

    def test_save_load(self):
        from target_affinity_ml.models.esm_fp_mlp_model import ESMFPMLPModel

        model = ESMFPMLPModel(input_dim=32, hidden_dims=[16])
        X = torch.randn(20, 32)
        y = torch.randn(20)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10)

        preds_before = model.predict(loader)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)

            model2 = ESMFPMLPModel(input_dim=32, hidden_dims=[16])
            model2.load(tmpdir)
            preds_after = model2.predict(loader)

        np.testing.assert_array_almost_equal(preds_before, preds_after)


# ---- GNN model tests ----

class TestGNNModel:
    @pytest.fixture
    def sample_graphs(self):
        from target_affinity_ml.features.molecular_graphs import smiles_to_graph
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "c1ccc(O)cc1"]
        graphs = []
        for smi in smiles_list:
            g = smiles_to_graph(smi)
            g.y = torch.tensor([np.random.randn()], dtype=torch.float32)
            graphs.append(g)
        return graphs

    def test_fit_predict(self, sample_graphs):
        from torch_geometric.loader import DataLoader as PyGDataLoader

        from target_affinity_ml.models.gnn_model import GNNModel

        model = GNNModel(num_layers=2, hidden_dim=32)
        loader = PyGDataLoader(sample_graphs, batch_size=3)

        preds = model.predict(loader)
        assert preds.shape == (5,)

    def test_uncertainty(self, sample_graphs):
        from torch_geometric.loader import DataLoader as PyGDataLoader

        from target_affinity_ml.models.gnn_model import GNNModel

        model = GNNModel(num_layers=2, hidden_dim=32, dropout=0.3)
        loader = PyGDataLoader(sample_graphs, batch_size=3)

        mean, std = model.predict_with_uncertainty(loader, n_samples=5)
        assert mean.shape == (5,)
        assert std.shape == (5,)
        assert np.all(std >= 0)

    def test_save_load(self, sample_graphs):
        from torch_geometric.loader import DataLoader as PyGDataLoader

        from target_affinity_ml.models.gnn_model import GNNModel

        model = GNNModel(num_layers=2, hidden_dim=32)
        loader = PyGDataLoader(sample_graphs, batch_size=3)

        preds_before = model.predict(loader)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)

            model2 = GNNModel(num_layers=2, hidden_dim=32)
            model2.load(tmpdir)
            preds_after = model2.predict(loader)

        np.testing.assert_array_almost_equal(preds_before, preds_after)


# ---- Fusion model tests ----

class TestFusionModel:
    @pytest.fixture
    def sample_fusion_data(self):
        from target_affinity_ml.features.molecular_graphs import smiles_to_graph
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC"]
        graphs = []
        for smi in smiles_list:
            g = smiles_to_graph(smi)
            g.y = torch.tensor([np.random.randn()], dtype=torch.float32)
            g.protein_emb = torch.randn(1, 64)  # Smaller for testing
            graphs.append(g)
        return graphs

    def test_fit_predict(self, sample_fusion_data):
        from torch_geometric.loader import DataLoader as PyGDataLoader

        from target_affinity_ml.models.fusion_model import FusionModel

        model = FusionModel(
            gnn_layers=2, gnn_hidden_dim=32,
            protein_input_dim=64, protein_projection_dim=32,
            fusion_hidden_dims=[32],
        )
        loader = PyGDataLoader(sample_fusion_data, batch_size=2)

        # Manual forward pass
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                protein_emb = batch.protein_emb
                pred = model((batch, protein_emb))
                all_preds.append(pred)

        preds = torch.cat(all_preds).squeeze(-1).numpy()
        assert preds.shape == (4,)

    def test_uncertainty(self, sample_fusion_data):
        from torch_geometric.loader import DataLoader as PyGDataLoader

        from target_affinity_ml.models.deep_base import _enable_dropout
        from target_affinity_ml.models.fusion_model import FusionModel

        model = FusionModel(
            gnn_layers=2, gnn_hidden_dim=32,
            protein_input_dim=64, protein_projection_dim=32,
            fusion_hidden_dims=[32], dropout=0.3,
        )
        loader = PyGDataLoader(sample_fusion_data, batch_size=2)

        # MC-Dropout manually
        _enable_dropout(model)
        all_passes = []
        for _ in range(5):
            preds = []
            with torch.no_grad():
                for batch in loader:
                    protein_emb = batch.protein_emb
                    pred = model((batch, protein_emb)).squeeze(-1)
                    preds.append(pred)
            all_passes.append(torch.cat(preds).numpy())

        stacked = np.stack(all_passes)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        assert mean.shape == (4,)
        assert std.shape == (4,)
        assert np.all(std >= 0)
