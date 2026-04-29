"""target-affinity-ml: Class-agnostic ML benchmarking for protein-ligand affinity.

Public API (populated as Tasks 3-8 migrate modules):
    from target_affinity_ml.training import train_and_evaluate
    from target_affinity_ml.evaluation.metrics import compute_regression_metrics
    from target_affinity_ml.data.splits import random_split, scaffold_split, target_split
"""

__version__ = "1.0.0"

from target_affinity_ml import features  # noqa: F401

__all__ = [
    "__version__",
]

from target_affinity_ml import data  # noqa: F401
