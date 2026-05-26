# Changelog

All notable changes to `target-affinity-ml` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-05-26

### Added

- **`TargetClassConfig` abstraction** (`src/target_affinity_ml/data/target_class_config.py`): a
  frozen dataclass declaring how to identify and curate a protein target class via GO terms or an
  explicit ChEMBL ID list, with an optional `subfamily_map`. Enables kinases, GPCRs, and future
  target classes to share the data pipeline without code duplication.
- **Class-agnostic `fetch_target_class` orchestrator** in `chembl_fetcher.py`, plus the
  `KINASE_CONFIG` constant that reproduces the prior kinase-only behaviour. Existing functions
  `fetch_kinase_targets` and `fetch_bioactivities` are preserved unchanged for full backward
  compatibility.
- **Class-agnostic `curate_activities(config, dataset_config, raw_dir, stats=None)`** in
  `curate.py`. Populates a generic `subfamily` column from either the targets file (GO-based
  classes) or `config.subfamily_map` (explicit-list classes). The optional `stats` out-param
  threads standardization metadata back to callers.
- **`data_dir` parameter on all four feature functions** (`compute_and_cache_features`,
  `load_morgan_fingerprints`, `load_rdkit_descriptors`, `load_esm2_embeddings`). Default `None`
  preserves existing relative-`PROCESSED_DIR` behaviour — the kinase application repo is
  unaffected.
- **Deep-model integration smoke test** (`tests/integration/test_deep_model_smoke.py`) exercising
  `deep_train_and_evaluate` (ESM-FP MLP dispatch) on synthetic data, marked `@pytest.mark.slow`.
  Addresses Plan 1 limitation L4.

### Changed

- Curated datasets now carry a generic `subfamily` column instead of the kinase-specific
  `kinase_group`. `run_phase5.py` worst-predictions CSV column list updated accordingly.

### Fixed

- Removed dead `len(df)` expression statement in `splits.py`.

### Backward compatibility

All public APIs used by `kinase-affinity-baselines` (v1.0.0 import paths, `fetch_kinase_targets`,
`fetch_bioactivities`, feature functions with no `data_dir` argument) continue to work without
modification.

## [1.0.0] - 2026-04-29

### Added
- Library extracted from `kinase-affinity-baselines`
- Class-agnostic data, features, models, training, evaluation, visualization modules
- 7 model implementations: RF, XGBoost, ElasticNet, MLP, ESM-FP MLP, GIN, GIN+ESM Fusion
- Three split strategies: random, scaffold (Bemis-Murcko), target-held-out
- Multi-seed validation framework + bootstrap CIs
- Empty `benchmarks/` placeholder for Plan 3 (scaffold diversity, RNS)
- CI workflow with unit tests and lint checks
- Kinase reproducibility integration test (validation gate for refactor)

### Migration notes from `kinase_affinity` v1
- Imports change: `kinase_affinity.X` → `target_affinity_ml.X`
- `fetch.py` renamed to `chembl_fetcher.py`
- All other module names preserved
- **Known limitation:** `chembl_fetcher.py` and `curate.py` still contain
  kinase-specific logic (KINASE_GO_TERMS, hardcoded file paths). Refactor
  for true class-agnosticism deferred to v1.1.0 (Plan 2 work).
