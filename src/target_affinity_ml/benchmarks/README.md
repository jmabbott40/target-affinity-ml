# benchmarks/

Cross-class benchmarking methodology subpackage for the target-affinity-ml library.

**Status (v1.1.0):** Module scaffolded (Plan 3, Task 1). The three `.py` modules
are populated incrementally by Plan 3 Tasks 2-15. Until then, `__init__.py` lists
all eventual public functions as commented-out imports.

---

## Modules

### `scaffold_diversity.py` (Tasks 11-13)

Per-target and per-class scaffold concentration metrics plus correlation regressions.

Public functions:
- `compute_scaffold_metrics(df)` — Bemis-Murcko entropy, largest-cluster fraction,
  mean intra-cluster Tanimoto, activity-cliff frequency
- `compute_class_aggregates(metrics_df)` — roll up per-target metrics to class level
- `fit_degradation_regression(metrics_df, results_df)` — fit scaffold-diversity →
  model-performance degradation slope per split type

### `rns_scoring.py` (Tasks 2-6)

Prabakaran-Bromberg Random Neighbor Score pipeline with a conservation-entropy
fallback for targets that cannot be scored (no structure, failed MSA, etc.).

Public functions:
- `fetch_structure(uniprot_id, pdb_id=None)` — retrieve PDB or AlphaFold structure
- `fetch_binding_site(uniprot_id, target_class)` — retrieve binding-site residue
  list from KLIFS (kinases) or GPCRdb (GPCRs)
- `compute_msa(fasta_path)` — run jackhmmer to build a multiple sequence alignment
- `compute_per_residue_rns(structure, msa, binding_site)` — score each binding-site
  residue using the Prabakaran-Bromberg neighborhood-fraction metric
- `aggregate_target_rns(per_residue_scores)` — mean over binding-site residues
- `compute_conservation_entropy(msa, binding_site)` — fallback: Shannon entropy-
  based conservation for targets that fail the RNS pipeline
- `validation_gate(pipeline_scores, reference_data)` — GO/NO-GO gate: Spearman
  rho >= 0.7 OR mean absolute deviation <= 10% vs. the bundled reference set;
  fails over to conservation-entropy if gate does not pass

### `hypothesis_tests.py` (Task 15)

Pre-registered H1-H4 hypothesis tests and between-class interaction machinery.

Public functions:
- `h1_rf_vs_deep(results_df)` — H1: RF/XGB match deep models on random split
- `h2_split_degradation(results_df)` — H2: scaffold/target splits degrade > random
- `h3_esm_target_advantage(results_df)` — H3: ESM-2 advantage on target split
- `h4_single_seed_flip_rate(results_df)` — H4: single-seed conclusions flip >= 20%
- `class_split_interaction(results_df)` — between-class interaction test (GPCR vs
  kinase degradation magnitude)

---

## Bundled reference data

### `_rns_reference_data.json`

Eight well-characterized proteins with peer-reviewed binding-site residue lists and
ConSurf-derived mean conservation scores. Used exclusively by `validation_gate()` to
check rank-order concordance between pipeline output and literature expectations.

**Source note:** The Prabakaran-Bromberg RNS (Random Neighbor Score; Nature Methods
2026, doi:10.1038/s41592-026-03028-7) is a protein-language-model evaluation metric
and does not publish per-protein binding-site conservation reference values. The
reference values in this bundle are ConSurf binding-site mean conservation scores
assembled from peer-reviewed structural biology papers (sources per entry in the JSON
`notes` field). The validation gate uses a Spearman rho criterion, so rank-order
concordance — not absolute value identity — is what matters.

**Proteins bundled:**

| Name | UniProt | PDB | Reference RNS |
|------|---------|-----|--------------|
| EGFR | P00533 | 2ITY | 0.88 |
| ABL1 | P00519 | 2HYY | 0.85 |
| CDK2 | P24941 | 1HCL | 0.82 |
| p38-MAPK | Q16539 | 1A9U | 0.79 |
| HSP90-alpha | P07900 | 1YET | 0.75 |
| beta-2 adrenergic receptor | P07550 | 2RH1 | 0.71 |
| p53 | P04637 | 2OCJ | 0.68 |
| HIV-1 protease | P04585 | 1OHR | 0.62 |

---

## Implementation plan

See `docs/superpowers/plans/` for the full Plan 3 specification.
Tasks 2-15 fill in the module stubs in the following order:

- T2: `fetch_structure()`
- T3: `fetch_binding_site()`
- T4: `compute_msa()`
- T5: `compute_per_residue_rns()` + `aggregate_target_rns()` + `compute_conservation_entropy()`
- T6: `validation_gate()`
- T11: `scaffold_diversity.py` (all functions)
- T13: `fit_degradation_regression()`
- T15: `hypothesis_tests.py` (all functions)
