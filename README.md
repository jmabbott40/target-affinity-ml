# target-affinity-ml

A class-agnostic Python library for benchmarking machine learning models
on protein-ligand binding affinity prediction tasks. Implements the
seven-model framework (Random Forest, XGBoost, ElasticNet, MLP, ESM-FP MLP,
GIN, GIN+ESM Fusion) with three splitting strategies (random, scaffold,
target-held-out), 5-seed multi-seed validation, and bootstrap CIs.

## Installation

```bash
pip install target-affinity-ml
```

For deep models (ESM-FP MLP, GIN, Fusion), install with the `deep` extra:

```bash
pip install target-affinity-ml[deep]
```

## Usage

```python
from target_affinity_ml.training import train_and_evaluate

results = train_and_evaluate(
    config_path="configs/rf_baseline.yaml",
    split_strategy="random",
    dataset_version="v1",
)
```

## Used by

This library is used by the following application repositories:

- [`kinase-affinity-baselines`](https://github.com/jmabbott40/kinase-affinity-baselines) — frozen at preprint v1.0
- [`gpcr-aminergic-benchmarks`](https://github.com/jmabbott40/gpcr-aminergic-benchmarks) — Phase 1 of multi-class expansion

## Versioning

This library uses semantic versioning. Major version bumps indicate any
change that produces different numerical results. Both
`kinase-affinity-baselines` and `gpcr-aminergic-benchmarks` pin to specific
versions for reproducibility.

## Citation

[Add citation when papers are published]

## License

MIT
