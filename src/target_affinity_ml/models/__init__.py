"""Seven model implementations with unified interface.

Baselines (RF/XGB/ElasticNet/MLP) import unconditionally. Deep models
(ESM-FP MLP/GNN/Fusion) require the [deep] extras (torch, torch-geometric,
fair-esm) and the target_affinity_ml.features submodule; if either is
unavailable they are skipped here but remain importable directly via
their submodule once dependencies are present.
"""

from target_affinity_ml.models.elasticnet_model import ElasticNetModel
from target_affinity_ml.models.mlp_model import MLPModel
from target_affinity_ml.models.rf_model import RandomForestModel
from target_affinity_ml.models.xgb_model import XGBoostModel

try:
    from target_affinity_ml.models.esm_fp_mlp_model import ESMFPMLPModel
    from target_affinity_ml.models.fusion_model import FusionModel
    from target_affinity_ml.models.gnn_model import GNNModel
    _DEEP_AVAILABLE = True
except ImportError:
    ESMFPMLPModel = None  # type: ignore[assignment]
    GNNModel = None  # type: ignore[assignment]
    FusionModel = None  # type: ignore[assignment]
    _DEEP_AVAILABLE = False

MODEL_REGISTRY = {
    "random_forest": "target_affinity_ml.models.rf_model.RandomForestModel",
    "xgboost": "target_affinity_ml.models.xgb_model.XGBoostModel",
    "elasticnet": "target_affinity_ml.models.elasticnet_model.ElasticNetModel",
    "mlp": "target_affinity_ml.models.mlp_model.MLPModel",
    "esm_fp_mlp": "target_affinity_ml.models.esm_fp_mlp_model.ESMFPMLPModel",
    "gnn": "target_affinity_ml.models.gnn_model.GNNModel",
    "fusion": "target_affinity_ml.models.fusion_model.FusionModel",
}

__all__ = [
    "RandomForestModel", "XGBoostModel", "ElasticNetModel", "MLPModel",
    "ESMFPMLPModel", "GNNModel", "FusionModel",
    "MODEL_REGISTRY",
]
