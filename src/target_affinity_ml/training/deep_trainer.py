"""Training loop for PyTorch deep learning models (Phase 7).

Separate from trainer.py to keep the baseline trainer clean.
Shares evaluation code (metrics) but owns its own data loading
and training loop for heterogeneous inputs (graphs, embeddings).

Models:
    - esm_fp_mlp: Morgan FP + ESM-2 embedding → MLP
    - gnn: Molecular graphs → GIN
    - fusion: Molecular graphs + ESM-2 → GIN + MLP fusion

Usage:
    python -m target_affinity_ml.training.deep_trainer \\
        --config configs/gnn.yaml --split scaffold

    python -m target_affinity_ml.training.deep_trainer --all
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from target_affinity_ml.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from target_affinity_ml.features import load_esm2_embeddings, load_morgan_fingerprints

logger = logging.getLogger(__name__)

DEEP_MODEL_REGISTRY = {
    "esm_fp_mlp": "target_affinity_ml.models.esm_fp_mlp_model.ESMFPMLPModel",
    "gnn": "target_affinity_ml.models.gnn_model.GNNModel",
    "fusion": "target_affinity_ml.models.fusion_model.FusionModel",
}

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

ALL_CONFIGS = [
    Path("configs/esm_fp_mlp.yaml"),
    Path("configs/gnn.yaml"),
    Path("configs/fusion.yaml"),
]
ALL_SPLITS = ["random", "scaffold", "target"]

# Fallback strategies for targets without ESM-2 embeddings.
# "row0": original behavior — use embedding of whichever target sorts first
# "zero": neutral zero vector — no protein information for unknown targets
# "mean": average embedding across all known targets — generic kinase prior
FALLBACK_STRATEGIES = ("row0", "zero", "mean")


def _get_fallback_vector(
    strategy: str,
    esm_matrix: np.ndarray,
) -> np.ndarray:
    """Return the fallback embedding vector for targets without real embeddings.

    Parameters
    ----------
    strategy : str
        One of "row0", "zero", or "mean".
    esm_matrix : np.ndarray
        Full embedding matrix (n_targets_with_embeddings, embed_dim).

    Returns
    -------
    np.ndarray
        Fallback vector of shape (embed_dim,).
    """
    if strategy == "row0":
        return esm_matrix[0]
    elif strategy == "zero":
        return np.zeros(esm_matrix.shape[1], dtype=esm_matrix.dtype)
    elif strategy == "mean":
        return esm_matrix.mean(axis=0)
    else:
        raise ValueError(
            f"Unknown fallback strategy: {strategy!r}. "
            f"Choose from {FALLBACK_STRATEGIES}"
        )


def _resolve_esm_embeddings(
    target_ids: np.ndarray,
    esm_matrix: np.ndarray,
    target_to_row: dict[str, int],
    fallback_strategy: str = "row0",
) -> np.ndarray:
    """Resolve ESM-2 embeddings for a list of target IDs.

    Targets present in target_to_row get their real embedding.
    Missing targets receive the fallback vector determined by strategy.

    Parameters
    ----------
    target_ids : np.ndarray
        Array of target_chembl_id strings.
    esm_matrix : np.ndarray
        Embedding matrix (n_targets, embed_dim).
    target_to_row : dict[str, int]
        Mapping from target_chembl_id to row in esm_matrix.
    fallback_strategy : str
        How to handle missing targets.

    Returns
    -------
    np.ndarray
        Resolved embeddings, shape (len(target_ids), embed_dim).
    """
    fallback_vec = _get_fallback_vector(fallback_strategy, esm_matrix)
    embed_dim = esm_matrix.shape[1]
    result = np.empty((len(target_ids), embed_dim), dtype=np.float32)

    n_fallback = 0
    for i, tid in enumerate(target_ids):
        row_idx = target_to_row.get(tid, -1)
        if row_idx >= 0:
            result[i] = esm_matrix[row_idx]
        else:
            result[i] = fallback_vec
            n_fallback += 1

    if n_fallback > 0:
        logger.debug(
            "  ESM fallback (%s): %d/%d targets used fallback embedding",
            fallback_strategy, n_fallback, len(target_ids),
        )

    return result


def _set_training_seed(seed: int | None) -> None:
    """Set random seeds for reproducible training.

    Parameters
    ----------
    seed : int or None
        If None, no seeds are set (non-deterministic training).
    """
    if seed is None:
        return
    import random as python_random
    python_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For full determinism (slower):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    logger.info("  Training seed set to %d", seed)


def get_deep_model_class(model_name: str):
    """Resolve deep model class from registry."""
    if model_name not in DEEP_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown deep model: {model_name}. Available: {list(DEEP_MODEL_REGISTRY)}"
        )
    module_path = DEEP_MODEL_REGISTRY[model_name]
    module_name, class_name = module_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _build_esm_fp_loaders(
    df: pd.DataFrame,
    split_indices: dict,
    dataset_version: str,
    batch_size: int,
    fallback_strategy: str = "row0",
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Build DataLoaders for ESM-FP MLP model.

    Concatenates Morgan FP (2048) with ESM-2 embedding (1280) per record.

    Parameters
    ----------
    fallback_strategy : str
        How to handle targets without real ESM-2 embeddings.
        "row0" (default): use first target's embedding (original behavior).
        "zero": use zero vector (neutral / no protein info).
        "mean": use mean embedding across all known targets.
    """
    fp_matrix, smiles_list = load_morgan_fingerprints(dataset_version)
    esm_matrix, target_to_row = load_esm2_embeddings(dataset_version)

    smiles_to_row = {s: i for i, s in enumerate(smiles_list)}

    loaders = {}
    test_active = None
    test_y = None

    for split_name in ["train", "val", "test"]:
        indices = split_indices[split_name]
        subset = df.iloc[indices]

        # Get ligand features
        fp_rows = np.array([smiles_to_row[s] for s in subset["std_smiles"].values])
        X_fp = fp_matrix[fp_rows].astype(np.float32)

        # Get protein embeddings with configurable fallback
        target_ids = subset["target_chembl_id"].values
        X_esm = _resolve_esm_embeddings(
            target_ids, esm_matrix, target_to_row, fallback_strategy,
        )

        # Concatenate
        X = np.concatenate([X_fp, X_esm], axis=1)
        y = subset["pactivity"].values.astype(np.float32)

        dataset = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
        )
        shuffle = split_name == "train"
        loaders[split_name] = DataLoader(dataset, batch_size=batch_size,
                                          shuffle=shuffle, num_workers=0)

        if split_name == "test":
            test_active = subset["is_active"].values.astype(np.float64)
            test_y = y

    return loaders["train"], loaders["val"], loaders["test"], test_active, test_y


def _build_gnn_loaders(
    df: pd.DataFrame,
    split_indices: dict,
    batch_size: int,
) -> tuple:
    """Build PyG DataLoaders for GNN model."""
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from target_affinity_ml.features.molecular_graphs import smiles_to_graph

    loaders = {}
    test_active = None
    test_y = None

    for split_name in ["train", "val", "test"]:
        indices = split_indices[split_name]
        subset = df.iloc[indices]

        graphs = []
        for _, row in subset.iterrows():
            graph = smiles_to_graph(row["std_smiles"])
            if graph is not None:
                graph.y = torch.tensor([row["pactivity"]], dtype=torch.float32)
                graph.is_active = torch.tensor([row["is_active"]], dtype=torch.float32)
                graphs.append(graph)

        if split_name == "train":
            logger.info("  GNN graphs built: %d/%d valid", len(graphs), len(subset))

        shuffle = split_name == "train"
        loaders[split_name] = PyGDataLoader(graphs, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=0)

        if split_name == "test":
            test_active = np.array([g.is_active.item() for g in graphs])
            test_y = np.array([g.y.item() for g in graphs])

    return loaders["train"], loaders["val"], loaders["test"], test_active, test_y


def _build_fusion_loaders(
    df: pd.DataFrame,
    split_indices: dict,
    dataset_version: str,
    batch_size: int,
    fallback_strategy: str = "row0",
) -> tuple:
    """Build DataLoaders for Fusion model (graphs + ESM-2 embeddings).

    Parameters
    ----------
    fallback_strategy : str
        How to handle targets without real ESM-2 embeddings.
        See _resolve_esm_embeddings for options.
    """
    from torch_geometric.data import Batch as PyGBatch
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from target_affinity_ml.features.molecular_graphs import smiles_to_graph

    esm_matrix, target_to_row = load_esm2_embeddings(dataset_version)
    fallback_vec = _get_fallback_vector(fallback_strategy, esm_matrix)

    loaders = {}
    test_active = None
    test_y = None

    for split_name in ["train", "val", "test"]:
        indices = split_indices[split_name]
        subset = df.iloc[indices]

        graphs = []
        for _, row in subset.iterrows():
            graph = smiles_to_graph(row["std_smiles"])
            if graph is not None:
                graph.y = torch.tensor([row["pactivity"]], dtype=torch.float32)
                graph.is_active = torch.tensor([row["is_active"]], dtype=torch.float32)
                # Attach target embedding with fallback
                tidx = target_to_row.get(row["target_chembl_id"], -1)
                if tidx >= 0:
                    emb_vec = esm_matrix[tidx].astype(np.float32)
                else:
                    emb_vec = fallback_vec.astype(np.float32)
                graph.protein_emb = torch.from_numpy(
                    emb_vec
                ).unsqueeze(0)  # (1, 1280)
                graphs.append(graph)

        if split_name == "train":
            logger.info("  Fusion graphs built: %d/%d valid", len(graphs), len(subset))

        shuffle = split_name == "train"
        loaders[split_name] = PyGDataLoader(graphs, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=0)

        if split_name == "test":
            test_active = np.array([g.is_active.item() for g in graphs])
            test_y = np.array([g.y.item() for g in graphs])

    return loaders["train"], loaders["val"], loaders["test"], test_active, test_y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _extract_targets(batch, model_name: str) -> torch.Tensor:
    """Extract target values from a batch depending on model type."""
    if model_name == "esm_fp_mlp":
        # TensorDataset: batch = (X, y)
        return batch[1]
    else:
        # PyG batch: y is an attribute
        return batch.y


def _prepare_batch(batch, model_name: str, device: str):
    """Prepare batch for forward pass."""
    if model_name == "esm_fp_mlp":
        return batch[0].to(device)
    elif model_name == "gnn":
        return batch.to(device)
    elif model_name == "fusion":
        graph_batch = batch.to(device)
        protein_emb = graph_batch.protein_emb  # (batch_size, 1280)
        return (graph_batch, protein_emb)
    return batch


def train_epoch(model, loader, optimizer, model_name: str, device: str) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        y = _extract_targets(batch, model_name).to(device)
        prepared = _prepare_batch(batch, model_name, device)

        optimizer.zero_grad()
        pred = model(prepared).squeeze(-1)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(model, loader, model_name: str, device: str) -> float:
    """Evaluate on validation set. Returns RMSE."""
    model.eval()
    all_pred, all_true = [], []

    for batch in loader:
        y = _extract_targets(batch, model_name).to(device)
        prepared = _prepare_batch(batch, model_name, device)
        pred = model(prepared).squeeze(-1)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _collect_predictions(model, loader, model_name: str, device: str):
    """Collect all predictions from a DataLoader."""
    model.eval()
    all_pred = []

    with torch.no_grad():
        for batch in loader:
            prepared = _prepare_batch(batch, model_name, device)
            pred = model(prepared).squeeze(-1)
            all_pred.append(pred.cpu().numpy())

    return np.concatenate(all_pred)


def _collect_mc_dropout(model, loader, model_name: str, device: str, n_samples: int = 20):
    """MC-Dropout uncertainty estimation."""
    from target_affinity_ml.models.deep_base import _enable_dropout
    _enable_dropout(model)

    all_passes = []
    for _ in range(n_samples):
        preds = []
        with torch.no_grad():
            for batch in loader:
                prepared = _prepare_batch(batch, model_name, device)
                pred = model(prepared).squeeze(-1)
                preds.append(pred.cpu().numpy())
        all_passes.append(np.concatenate(preds))

    model.eval()
    stacked = np.stack(all_passes, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def deep_train_and_evaluate(
    config_path: Path,
    split_strategy: str = "random",
    dataset_version: str = "v1",
    training_seed: int | None = None,
    fallback_strategy: str | None = None,
    data_dir_override: Path | None = None,
    output_suffix: str = "",
) -> dict:
    """Train and evaluate a deep model on one split.

    Parameters
    ----------
    config_path : Path
        Path to model config YAML.
    split_strategy : str
        One of 'random', 'scaffold', 'target'.
    dataset_version : str
        Dataset version (subdirectory of data/processed/).
    training_seed : int, optional
        If provided, set all random seeds for reproducible training.
        If None, no seeds are set (original non-deterministic behavior).
    fallback_strategy : str, optional
        How to handle targets without ESM-2 embeddings.
        If None, reads from config or defaults to "row0".
        Options: "row0", "zero", "mean".
    data_dir_override : Path, optional
        Override the data directory (for subset experiments).
        If None, uses DATA_DIR / dataset_version.
    output_suffix : str
        Suffix for output files (e.g., "_seed42", "_esm92").
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    hyperparams = config.get("hyperparameters", {})
    uncertainty_config = config.get("uncertainty", {})

    # Resolve fallback strategy: CLI arg > config > default
    if fallback_strategy is None:
        fallback_strategy = config.get("features", {}).get(
            "protein_fallback", "row0"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 70)
    logger.info("DEEP EXPERIMENT: model=%s, split=%s, device=%s, seed=%s, fallback=%s",
                model_name, split_strategy, device, training_seed, fallback_strategy)
    logger.info("=" * 70)

    # Set training seed if requested
    _set_training_seed(training_seed)

    # Load data — support subset directories
    if data_dir_override is not None:
        data_dir = data_dir_override
    else:
        data_dir = DATA_DIR / dataset_version

    df = pd.read_parquet(data_dir / "curated_activities.parquet")

    # Look for splits in the data directory, then fall back to main splits
    split_path = data_dir / "splits" / f"{split_strategy}_split.json"
    if not split_path.exists():
        split_path = DATA_DIR / dataset_version / "splits" / f"{split_strategy}_split.json"

    with open(split_path) as f:
        split_indices = json.load(f)

    logger.info("  Dataset: %d records", len(df))
    logger.info("  Split: train=%d, val=%d, test=%d",
                len(split_indices["train"]), len(split_indices["val"]),
                len(split_indices["test"]))

    # Build DataLoaders
    batch_size = hyperparams.get("batch_size", 512)
    logger.info("Building DataLoaders (batch_size=%d, fallback=%s)...",
                batch_size, fallback_strategy)

    if model_name == "esm_fp_mlp":
        train_loader, val_loader, test_loader, test_active, test_y = \
            _build_esm_fp_loaders(df, split_indices, dataset_version, batch_size,
                                  fallback_strategy=fallback_strategy)
    elif model_name == "gnn":
        train_loader, val_loader, test_loader, test_active, test_y = \
            _build_gnn_loaders(df, split_indices, batch_size)
    elif model_name == "fusion":
        train_loader, val_loader, test_loader, test_active, test_y = \
            _build_fusion_loaders(df, split_indices, dataset_version, batch_size,
                                  fallback_strategy=fallback_strategy)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Instantiate model
    model_cls = get_deep_model_class(model_name)
    model_kwargs = {k: v for k, v in hyperparams.items()
                    if k not in ("batch_size", "learning_rate", "weight_decay",
                                 "max_epochs", "patience", "lr_scheduler")}
    model = model_cls(**model_kwargs).to(device)
    logger.info("  Model: %s (%d parameters)",
                model_cls.__name__,
                sum(p.numel() for p in model.parameters()))

    # Optimizer + scheduler
    lr = hyperparams.get("learning_rate", 0.001)
    wd = hyperparams.get("weight_decay", 0.0001)
    max_epochs = hyperparams.get("max_epochs", 100)
    patience = hyperparams.get("patience", 10)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Training loop
    logger.info("Training for up to %d epochs (patience=%d)...", max_epochs, patience)
    t0 = time.time()
    best_val_rmse = float("inf")
    best_epoch = 0
    patience_counter = 0

    model_subdir = f"{model_name}{output_suffix}"
    model_dir = RESULTS_DIR / "models" / model_subdir / split_strategy
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, model_name, device)
        val_rmse = evaluate_epoch(model, val_loader, model_name, device)
        scheduler.step()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            model.save(model_dir)
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            logger.info("  Epoch %3d: train_loss=%.4f, val_rmse=%.4f (best=%.4f @ %d)",
                         epoch, train_loss, val_rmse, best_val_rmse, best_epoch)

        if patience_counter >= patience:
            logger.info("  Early stopping at epoch %d (best=%d)", epoch, best_epoch)
            break

    train_time = time.time() - t0
    logger.info("  Training completed in %.1f seconds (best epoch %d)", train_time, best_epoch)

    # Load best model
    model.load(model_dir, device=device)

    # Predictions
    logger.info("Generating predictions...")
    y_test_pred = _collect_predictions(model, test_loader, model_name, device)

    # Uncertainty
    mc_samples = uncertainty_config.get("mc_dropout_samples", 20)
    logger.info("Computing MC-Dropout uncertainty (%d samples)...", mc_samples)
    y_test_mean, y_test_std = _collect_mc_dropout(
        model, test_loader, model_name, device, mc_samples)

    # Metrics
    logger.info("Computing metrics...")
    test_reg = compute_regression_metrics(test_y, y_test_pred)
    test_cls = compute_classification_metrics(test_active, y_test_pred)

    logger.info("  Test RMSE=%.3f, R²=%.3f, Pearson=%.3f, AUROC=%.3f",
                 test_reg["rmse"], test_reg["r2"],
                 test_reg["pearson_r"], test_cls.get("auroc", float("nan")))

    # Save predictions
    pred_dir = RESULTS_DIR / "predictions"
    if output_suffix:
        pred_dir = RESULTS_DIR / f"predictions{output_suffix}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        pred_dir / f"{model_name}_{split_strategy}.npz",
        y_test_true=test_y,
        y_test_pred=y_test_pred,
        y_test_active=test_active,
        y_test_mean=y_test_mean,
        y_test_std=y_test_std,
    )

    # Save metrics
    all_metrics = {
        "model": model_name,
        "split": split_strategy,
        "training_seed": training_seed,
        "fallback_strategy": fallback_strategy,
        "output_suffix": output_suffix,
        "train_time_seconds": round(train_time, 1),
        "best_epoch": best_epoch,
        "n_train": len(split_indices["train"]),
        "n_val": len(split_indices["val"]),
        "n_test": len(split_indices["test"]),
        **{f"test_{k}": v for k, v in test_reg.items()},
        **{f"test_{k}": v for k, v in test_cls.items()},
    }

    metrics_dir = RESULTS_DIR / "tables"
    if output_suffix:
        metrics_dir = RESULTS_DIR / f"tables{output_suffix}"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{model_name}_{split_strategy}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=_json_default)

    logger.info("=" * 70)
    logger.info("COMPLETE: %s / %s — RMSE=%.3f, R²=%.3f",
                model_name, split_strategy, test_reg["rmse"], test_reg["r2"])
    logger.info("=" * 70)

    return all_metrics


def run_all_deep_experiments(
    dataset_version: str = "v1",
    training_seed: int | None = None,
    fallback_strategy: str | None = None,
    output_suffix: str = "",
) -> pd.DataFrame:
    """Run all deep model × split combinations.

    Parameters
    ----------
    dataset_version : str
        Dataset version.
    training_seed : int, optional
        Training seed for reproducibility.
    fallback_strategy : str, optional
        ESM-2 fallback strategy for missing targets.
    output_suffix : str
        Suffix for output directories (e.g., "_seed42", "_zero").
    """
    results = []
    total = len(ALL_CONFIGS) * len(ALL_SPLITS)
    i = 0

    for config_path in ALL_CONFIGS:
        if not config_path.exists():
            logger.warning("Config not found: %s", config_path)
            continue
        for split in ALL_SPLITS:
            i += 1
            logger.info("\n%s\n  DEEP EXPERIMENT %d/%d: %s × %s (seed=%s, fallback=%s)\n%s",
                        "#" * 70, i, total, config_path.stem, split,
                        training_seed, fallback_strategy, "#" * 70)
            try:
                metrics = deep_train_and_evaluate(
                    config_path, split, dataset_version,
                    training_seed=training_seed,
                    fallback_strategy=fallback_strategy,
                    output_suffix=output_suffix,
                )
                results.append(metrics)
            except Exception:
                logger.exception("FAILED: %s × %s", config_path.stem, split)

    if results:
        summary = pd.DataFrame(results)
        summary_dir = RESULTS_DIR / "tables"
        if output_suffix:
            summary_dir = RESULTS_DIR / f"tables{output_suffix}"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "phase7_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("Saved Phase 7 summary to %s", summary_path)
        return summary

    logger.warning("No deep experiments completed!")
    return pd.DataFrame()


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train deep learning models for kinase affinity prediction",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", choices=ALL_SPLITS, default="random")
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--training-seed", type=int, default=None,
        help="Set random seed for reproducible training",
    )
    parser.add_argument(
        "--fallback-strategy", choices=FALLBACK_STRATEGIES, default=None,
        help="ESM-2 fallback for targets without embeddings (default: from config or row0)",
    )
    parser.add_argument(
        "--output-suffix", default="",
        help="Suffix for output directories (e.g., '_seed42', '_zero')",
    )
    parser.add_argument(
        "--data-dir-override", type=Path, default=None,
        help="Override data directory (e.g., data/processed/v1/subsets/esm92)",
    )
    args = parser.parse_args()

    if args.all:
        run_all_deep_experiments(
            args.dataset_version,
            training_seed=args.training_seed,
            fallback_strategy=args.fallback_strategy,
            output_suffix=args.output_suffix,
        )
    elif args.config:
        deep_train_and_evaluate(
            args.config, args.split, args.dataset_version,
            training_seed=args.training_seed,
            fallback_strategy=args.fallback_strategy,
            data_dir_override=args.data_dir_override,
            output_suffix=args.output_suffix,
        )
    else:
        parser.error("Provide --config CONFIG or --all")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
