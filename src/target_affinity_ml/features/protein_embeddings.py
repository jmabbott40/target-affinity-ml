"""Compute and cache ESM-2 protein language model embeddings.

Uses Meta's ESM-2 (esm2_t33_650M_UR50D) to produce a single 1280-dim
embedding per kinase target by mean-pooling over residue representations.

The model is used as a frozen feature extractor — no fine-tuning.
With only 507 targets, this is a one-time GPU computation (~10-30 min).

Usage:
    python -m target_affinity_ml.features.protein_embeddings
    python -m target_affinity_ml.features.protein_embeddings --model esm2_t6_8M_UR50D  # smaller
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")

# ESM-2 model variants
ESM2_MODELS = {
    "esm2_t33_650M_UR50D": {"dim": 1280, "layers": 33},
    "esm2_t6_8M_UR50D": {"dim": 320, "layers": 6},
}

# ESM-2 max sequence length (including BOS/EOS tokens)
MAX_SEQ_LEN = 1022


def compute_esm2_embeddings(
    dataset_version: str = "v1",
    model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 4,
    device: str | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute ESM-2 embeddings for all cached protein sequences.

    Parameters
    ----------
    dataset_version : str
        Dataset version.
    model_name : str
        ESM-2 model variant.
    batch_size : int
        Sequences per batch (reduce if OOM).
    device : str, optional
        Device ('cuda', 'cpu'). Auto-detected if None.

    Returns
    -------
    tuple[np.ndarray, dict[str, int]]
        (embedding_matrix, target_to_row) where embedding_matrix has shape
        (n_targets, embedding_dim) and target_to_row maps target_chembl_id
        to row index.
    """
    import esm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Load protein sequences
    data_dir = DATA_DIR / dataset_version
    seq_path = data_dir / "protein_sequences.json"
    with open(seq_path) as f:
        seq_cache = json.load(f)

    logger.info("Loaded %d protein sequences", len(seq_cache))

    # Load ESM-2 model
    model_info = ESM2_MODELS[model_name]
    embed_dim = model_info["dim"]
    logger.info("Loading ESM-2 model: %s (dim=%d)...", model_name, embed_dim)

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    # Prepare sequences (truncate if needed)
    target_ids = sorted(seq_cache.keys())
    target_to_row = {tid: i for i, tid in enumerate(target_ids)}
    sequences = []
    truncated = 0

    for tid in target_ids:
        seq = seq_cache[tid]["sequence"]
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[:MAX_SEQ_LEN]
            truncated += 1
        sequences.append((tid, seq))

    if truncated:
        logger.info("Truncated %d sequences to %d residues", truncated, MAX_SEQ_LEN)

    # Sort by length for efficient batching (less padding waste)
    sequences.sort(key=lambda x: len(x[1]))

    # Compute embeddings
    embeddings = np.zeros((len(target_ids), embed_dim), dtype=np.float32)
    n_batches = (len(sequences) + batch_size - 1) // batch_size

    logger.info("Computing embeddings: %d sequences in %d batches...", len(sequences), n_batches)

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_seqs)
            batch_tokens = batch_tokens.to(device)

            results = model(batch_tokens, repr_layers=[model_info["layers"]])
            token_repr = results["representations"][model_info["layers"]]

            # Mean pool over residue positions (excluding BOS at 0 and EOS/padding)
            for j, (tid, seq) in enumerate(batch_seqs):
                seq_len = len(seq)
                # Tokens: [BOS, aa1, aa2, ..., aaN, EOS, PAD...]
                # We want positions 1 to seq_len (inclusive)
                residue_repr = token_repr[j, 1:seq_len + 1, :]
                mean_repr = residue_repr.mean(dim=0).cpu().numpy()
                row_idx = target_to_row[tid]
                embeddings[row_idx] = mean_repr

            if (i // batch_size + 1) % 10 == 0:
                logger.info("  Batch %d/%d complete", i // batch_size + 1, n_batches)

    logger.info("Embeddings computed: shape=%s", embeddings.shape)

    # Save
    features_dir = data_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        features_dir / "esm2_embeddings.npz",
        embeddings=embeddings,
    )
    with open(features_dir / "target_index.json", "w") as f:
        json.dump(target_to_row, f, indent=2)

    logger.info("Saved embeddings to %s", features_dir / "esm2_embeddings.npz")
    logger.info("Saved target index to %s", features_dir / "target_index.json")

    return embeddings, target_to_row


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute ESM-2 protein embeddings",
    )
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--model", default="esm2_t33_650M_UR50D",
                        choices=list(ESM2_MODELS.keys()))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()

    compute_esm2_embeddings(args.dataset_version, args.model,
                            args.batch_size, args.device)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
