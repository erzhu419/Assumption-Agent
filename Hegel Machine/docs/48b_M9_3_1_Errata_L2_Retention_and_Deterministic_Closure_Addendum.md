# `M9.3.1` Errata, Online `L2` Retention Re-Analysis & Deterministic Audit Closure Addendum (`0` LLM Calls)

- **Addendum Scope**: Deterministic corrections, precision errata, and production runtime closures (`F01`–`F10`) on top of the immutable sealed `M9.3` bundle ([`m9_3_online_source_freeze_lifecycle_20261126`](../evidence/m9_3_online_source_freeze_lifecycle_20261126/m9_3_online_lifecycle_master_summary.json))
- **Sealed Evidence Directory**: [`docs/evidence/m9_3_1_audit_closure_and_regrade_20261126/`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/SHA256SUMS.txt)
- **Live LLM Calls**: **`0`** (`100%` deterministic codebase closure, AST/handoff regrade, and negative-control verification)

---

## 1. Scientific Errata & Corrections to `docs/48` and `docs/49`

1. **Positive Finding Previously Under-Reported (`Online Stage S L2 Execution & Exact AST Retention`)**:
   - In `M9.3`, `docs/48` tracked `L1` primitive calls (`3` calls on prior `v7` `L1`, `0` on newly compiled `L1`) and missed genuine online `Stage S` **`L2`** reuse!
   - Re-auditing `04c_full_skill_activations.jsonl` and AST-normalized submitted deliverables ([`m9_3_source_aware_l1_l2_adoption_funnel.json`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/m9_3_source_aware_l1_l2_adoption_funnel.json)) proves:
     - `E` (`executable_source_reuse`) executed newly compiled `Stage S` `L2` functions **`1,021` times** across `2/8` successor episodes (`rep_01/suc01_C`: **`1,018`** calls to `autogenous_hereditary_spectrum_identification`; `rep_02/suc02_T`: **`3`** calls to `autogenous_separable_sylvester_kkt_inversion`), plus **`3`** calls in `Stage R` (`rec02_R`).
     - In `E / rep_01 / suc01_C`, the host AST static packager bundled `align_reference_panel` and `hereditary_design` at Turn 4, and the final verified submission retained **exact AST-normalized bodies** (`2/2` functions reachable from `solve_viscoplastic_hereditary_revision`).
2. **Exact Baseline `B` Identity**:
   - `B` (`bare_no_source_reuse`) in `M9.3` is **`prior_frozen_v7_no_source_growth`** (it receives the pre-existing `v7` forest without any `Stage S` growth), NOT a zero-library bare GPT baseline.
3. **Provider-Attempt & Token-Proxy Accounting Precision (`F06`)**:
   - `M9.3` executed `330` logical calls (`324` episode decisions + `6` `L2`/`L1` compiler calls), corresponding to **`>= 333` actual provider attempts** in episode `01b_provider_attempts` (`3` episode calls had `2` attempts each after `~360s` CLI timeouts with `usage_unknown = true`).
   - Because `K` (`knowledge_only_source_reuse`) experienced `1` unknown-usage `360s` timeout (`rep_01/suc02_C` Turn 6), `E - K = +18,076` token-cost proxy is strictly **`known-only`**, whereas `E` vs `B` (`1,134,884` vs `1,259,035`, `−124,151` / `−9.86%`) had `0` unknown-usage attempts in either arm.
   - `src01/01b_provider_attempts/` contained `3` unmatched staging files (`attempts_010..012.json`, `88,471` proxy) from a shared `stage_subdir`; `00_episode_summary.json` (`385,093` proxy across `9` matched calls) was unaffected, and `M9.3.1` now stamps per-call UUIDs (`attempts_<seq>_<uuid8>.json`) and copies by strict whitelist only.
4. **`suc01_T` Verifier CPU Timing Driver**:
   - The `E` vs `B` mean verifier CPU speedup on `suc01_T` (`0.102s` vs `0.268s`) was driven by `rep_01` (`0.044s` vs `0.487s`, `0` skill calls), NOT by `rep_02`'s `3` `v7` `L1` calls (`0.160s` vs `0.049s`).
5. **`19` Sealed `M9.3` Nogoods Reclassified (`F03`)**:
   - In `M9.3`, `/holdout_task/validate_public_schema.py` (`{"schema_valid": true}`) promoted `19` scientific nogoods (`numerical_residual`, `forward_model_misspecification`, `physical_or_causal_mechanism`) to `resolved = True` with empty `verified_scope`. All `19` are formally reclassified as `scientific_resolution_not_established_by_recorded_receipt` in [`a5_scientific_probe_vs_schema_and_19_nogood_errata.json`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/a5_scientific_probe_vs_schema_and_19_nogood_errata.json).
6. **Exact `Track A` (`m8_persistent_variational_planner.py`) Mathematical Attribution**:
   - `m8_persistent_variational_planner.py` implements Empirical-Bayes scale-covariant Normal-Inverse-Gamma (`NIG`) marginal likelihood (`F(M) = -log p(y | X, M) - log p_0(M)`), adjusted-`R^2` / tail-concentration residual proposal ranking (`diagnose_residual_signals`), and prediction-disagreement variance per unit cost (`select_discriminating_observation_probe`), rather than full `H_leaf x Rao-score` or entropy-exact `BALD`.

---

## 2. Summary of `M9.3.1` Deterministic Closures (`F01`–`F10`)

| Closure ID | Production Fix | Verification Receipt |
| :--- | :--- | :--- |
| **`F01`** | Explicit `source_node_id`, `source_version`, `violated_component_id`, and `preserved_component_ids` in `consolidate_autogenous_episode_into_forest` (`node_auto_07` untouched; `forest.nodes[parent].claim_contract.preconditions` synced). | [`c1_c2_component_bound_bifurcation_and_guard_controls.json`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/c1_c2_component_bound_bifurcation_and_guard_controls.json) |
| **`F02`** | 3-valued (`TRUE / FALSE / UNKNOWN`) `ObservablePreconditionGuard` checking BOTH history & hold overstress + `residual_plastic_state`, and exact 2D additive residual `||V - mean_x - mean_y + mean||_F / max(||V||_F, 1)`. | [`c1_c2_component_bound_bifurcation_and_guard_controls.json`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/c1_c2_component_bound_bifurcation_and_guard_controls.json) |
| **`F03`** | Strict `PublicSchemaReceipt` vs `ScientificProbeReceipt` separation + runtime SHA-256 re-verification of `/holdout_task/validate_public_schema.py` + `execute_host_trusted_scientific_probe`. | [`a5_scientific_probe_vs_schema_and_19_nogood_errata.json`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/a5_scientific_probe_vs_schema_and_19_nogood_errata.json) |
| **`F04`** | Tightened output/handoff verifier rejecting `np.zeros(7)` (`src01`), `np.array([np.nan])` (`rec01_R`), inconsistent secant/crossing (`suc01_T`), and mismatched `weights` vs `kkt_certificate['weights']` (`Chain 2`); **`28/28`** sealed `M9.3` deliverables pass! | [`m9_3_28of28_tightened_handoff_regrade.json`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/m9_3_28of28_tightened_handoff_regrade.json) |
| **`F05`** | Multi-label source-and-tier-aware funnel (`classify_source_aware_adoption_funnel`) separating `new_source_L2`, `new_source_L1`, and `prior_v7_L1`. | [`m9_3_source_aware_l1_l2_adoption_funnel.json`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/m9_3_source_aware_l1_l2_adoption_funnel.json) |
| **`F06`–`F10`** | Per-call UUID attempt ledgers + `_LLM_ERROR` accounting + whitelist `01b_provider_attempts` copy + compiler attempt receipts (`F06`); optional KKT witness helper (`F07`); 2-process determinism & input-immutability gate + container `.Image` inspection + cold-start `ensure_runtime_qualification_context` (`F08`); disjoint dev seeds `88512..88621` (`F09`); `/holdout_task/public_asset_index.json` + AST domain-guard separation (`F10`). | [`f06_f07_f08_f09_ast_and_hipporag_controls.json`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/f06_f07_f08_f09_ast_and_hipporag_controls.json) |
