"""Training and evaluation orchestration."""

# Best-effort re-exports — function names should match source modules.
# Sibling modules (evaluation, features, models) are migrated in other tasks;
# wrap in try/except so this package imports cleanly during incremental migration.
try:
    from target_affinity_ml.training.trainer import *  # noqa: F401, F403
except ImportError:
    pass

try:
    from target_affinity_ml.training.deep_trainer import *  # noqa: F401, F403
except ImportError:
    # Deep deps (torch, etc.) may not be installed — that's OK.
    pass

try:
    from target_affinity_ml.training.tune import *  # noqa: F401, F403
except ImportError:
    pass
