# Changelog

All notable changes to `target-affinity-ml` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
