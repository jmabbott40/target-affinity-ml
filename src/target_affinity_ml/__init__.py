"""target-affinity-ml: Class-agnostic ML benchmarking for protein-ligand affinity.

Public API (populated as Tasks 3-8 migrate modules):
    from target_affinity_ml.training import train_and_evaluate
    from target_affinity_ml.evaluation.metrics import compute_regression_metrics
    from target_affinity_ml.data.splits import random_split, scaffold_split, target_split
"""

__version__ = "1.0.0"

# Subpackages (populated by Plan 1 Tasks 3-8 migrations)
from target_affinity_ml import (
    benchmarks,  # noqa: F401
    data,  # noqa: F401
    evaluation,  # noqa: F401
    features,  # noqa: F401
    models,  # noqa: F401
    training,  # noqa: F401
    visualization,  # noqa: F401
)

__all__ = [
    "__version__",
    "data", "features", "models", "training", "evaluation", "visualization",
    "benchmarks",
]
