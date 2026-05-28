"""Cross-class methodology modules for the target-affinity-ml benchmark suite.

Modules
-------
scaffold_diversity : per-target + per-class scaffold metrics + correlation regressions
rns_scoring        : Prabakaran-Bromberg RNS scoring with conservation-entropy fallback
hypothesis_tests   : Pre-registered H1-H4 hypothesis tests + between-class machinery

These modules are populated incrementally by Plan 3 Tasks 1-15. Until those
tasks complete, individual function imports may fail; that's expected.
"""

# Imports will be uncommented as Tasks 2-15 populate the corresponding modules.
from target_affinity_ml.benchmarks.hypothesis_tests import (  # noqa: F401
    class_split_interaction,
    h1_rf_vs_deep,
    h2_split_degradation,
    h3_esm_target_advantage,
    h4_single_seed_flip_rate,
)
from target_affinity_ml.benchmarks.rns_scoring import (
    aggregate_target_rns,  # noqa: F401
    compute_binding_site_plddt,  # noqa: F401
    compute_conservation_entropy,  # noqa: F401
    compute_msa,  # noqa: F401
    compute_per_residue_rns,  # noqa: F401
    fetch_binding_site,  # noqa: F401
    fetch_structure,  # noqa: F401
    validation_gate,  # noqa: F401
)
from target_affinity_ml.benchmarks.scaffold_diversity import (  # noqa: F401
    compute_class_aggregates,
    compute_scaffold_metrics,
    fit_degradation_regression,
)
