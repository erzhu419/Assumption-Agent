# M9.1.1 Audit Errata, `hd01` Contract-Sensitivity Regrade, and Deterministic Closure Addendum

- **Date**: 2026-11-26
- **Branch**: `exp/m6-causal-wormhole-sandbox`
- **Sealed Evidence Bundle**: `docs/evidence/m9_1_1_audit_closure_and_regrade_20261126/` (`SHA256SUMS.txt` verified; zero paid LLM calls)
- **Immutable Upstream Bundle Preserved**: `docs/evidence/m9_1_zero_tax_demand_retrieval_20261126/` (`d3ffa0018`, untouched)

---

## 1. Terminology & Turn-1 Token Accounting Errata (`docs/46_*`)

1. **Zero Turn-1 Skill Manifest vs. Full Turn-1 Request Tokens (`+127` Input Tokens)**:
   - In `M9.1`, `demand_triggered_zero_tax_library` carried an empty Turn-1 `staged_wormhole_manifest` (`[]`, `2` JSON chars — identical `init_payload` to `bare_no_library`), eliminating `M8.9`/`M9.0a`'s upfront per-skill card manifest (`450–672` chars).
   - However, `demand_triggered_zero_tax_library` included a fixed `+425`-character system-prompt instruction block (`5. DORMANT ON-DEMAND PRIMITIVE LIBRARY`), resulting in an exact **`+127` Turn-1 input-token delta** across all `8/8` (`Demand` vs `Bare`) task-replicate pairs (`m9_1_turn1_full_request_accounting_errata.json`).
   - All code and reporting in `M9.1.1+` strictly distinguish `turn1_staged_manifest_json_chars` (`2`) from `turn1_request_accounting` (`turn1_system_prompt_chars`, `turn1_init_payload_json_chars`, `turn1_full_request_chars`, `turn1_prompt_tokens`).
2. **Subscription CLI Token Proxy Naming**:
   - Because runs execute through the official OpenAI ChatGPT Subscription Codex CLI (`auth_mode == "chatgpt"`), `effective_billed_tokens` was a weighted analytical proxy rather than an API invoice amount. `M9.1.1+` reports raw `(input_tokens, cached_input_tokens, output_tokens)` triples alongside `token_cost_proxy` (`cache_weight=0.25, output_weight=1.0`).

---

## 2. Retrospective `hd01` Contract-Sensitivity Regrade (`18/18` Instances)

- **Scope Notice**: This regrade (`hd01_contract_sensitivity_regrade_18of18.json`) is a **retrospective contract-representation sensitivity check** on the 6 immutable `M9.1` `hd01` deliverables, **not** a new blind model evaluation. Original sealed `M9.1` verdicts (`6/8, 7/8, 7/8`) remain untouched in `docs/evidence/m9_1_zero_tax_demand_retrieval_20261126/`.
- **Root Cause**: `hd01` v1 specified `fixed_zero` / `fixed_active` leaf masks in prose without enforcing canonical `bool[p]` typing in a public schema validator; 4 of 6 submissions emitted valid disjoint coordinate index lists (`list[int]`), which tripped `len(fz) != p` (`bad_masks`).
- **Normalized Results (`normalize_and_validate_box_l0_public_schema(..., allow_index_list_compat=True)`)**:
  - **`18 / 18` candidate-instance pairs pass** all three stages (`aligned_mae = 0.0569..0.0752 <= 0.12`, `potency_rel_l2 = 0.0268..0.0574 <= 0.08`, certified duality gap `3.71e-8..4.05e-8 <= 1e-6`), bringing all three `M9.1` arms to **`8 / 8`** under canonicalized mask representation:

| Arm | Original Sealed `M9.1` Strict Passes | Schema-Normalized Passes (`hd01` `bool[p]`) |
| --- | :---: | :---: |
| `bare_no_library` | `6 / 8` | **`8 / 8`** |
| `progressive_upfront_library` | `7 / 8` | **`8 / 8`** |
| `demand_triggered_zero_tax_library` | `7 / 8` | **`8 / 8`** |

---

## 3. Deterministic Engineering & Protocol Closures (`C1..C6`)

1. **`C1` (`normalize_and_validate_box_l0_public_schema`, `hegel_box_l0_cert_schema_v2`)**:
   - Canonical `bool[p]` public schema & binary-tree partition validator (`0` private ground-truth access), rejecting duplicate indices, out-of-bounds coordinates, overlapping `fixed_zero`/`fixed_active`, and non-finite bounds.
2. **`C2` (4-Tuple `L1` Qualification Identity & Order-Invariant Multi-Version Pool)**:
   - Upgraded `L1` qualification (`hegel_l1_qual_4tuple_v2`) to bind `(verified_code_sha256, synthetic_unit_test_sha256, verifier_protocol_version, dependency_env_digest)`. Editing `synthetic_unit_test_code` immediately invalidates qualification and triggers clean-container reverification; `reverify_in_clean_container=False` never silently stamps missing `verified_code_sha256`.
   - Container-reverified all `6/6` `L1` primitives into `pretrained_autogenous_forest_v6_m911.json` (`v6_4tuple_qualification_receipt.json`, `sha256 = 7868bad10fbfaabe512fee2df2d75fc378a7407eab1c7e8d831932dfe02020c3`).
   - Made duplicate-stem `L1` version selection order-invariant with explicit `superseded_l1_versions` logging.
3. **`C3` (Synonym/Function Search, Subproblem-Scoped Preconditions, Event-Clean Trigger, & Repair Preconditions)**:
   - Added mathematical synonym expansion and function-level scoring to `search_skills` and `describe_ref.py` (`"robust regression"` retrieves `cauchy_linear_fit` / `robust_residual_weights`; `"certified lower bound"` retrieves `box_ridge_cube_lower_bound`; `"complex Helmholtz preconditioner"` reports precondition rejection).
   - Upgraded `evaluate_l1_skill_staging_net_value` to soft-unwrap wrapped bullets and evaluate preconditions per subproblem clause with negation awareness (`vr04` retains `0` recommended `L1` modules while multi-stage mixed `complex128 + real float64 IRLS` tasks retain real primitives for real stages).
   - Hardened the on-demand hint state machine (`echo` mentions, failed `--describe` typos, benign `grep` exit-1, `L2` fallback, candidate-SHA verification binding) and wired `staged_skill_preconditions_by_stem` into `LocalContainerRepair`.
4. **`C4` (Quantitative Nogood Witness, Treatment-Invariant Router `v5`, Cross-Python AST Hash, & Multi-Attempt Accounting)**:
   - Rejected reflection-only (`callable`/`hasattr`) and keyword-only `nogood` spoofing while preserving genuine quantitative multi-gauge/anchor witnesses.
   - Made `classify_public_container_verification_state` and `build_pre_action_feature_vector` treatment-invariant by default (`treatment_invariant_mode=True`, removing `skill_activation_count` / `nogood_count` from verification-state promotion); replayed `m7_4` (`Frontier-20`) 20-fold LOTO transparently (`v5_treatment_invariant`: `13/20` passes, `11` spawns vs `v4_legacy`: `14/20` passes, `12` spawns; `canonical_feature_builder_sha256 = 2ef62967469251caee2af1c0fec5d67e26fd95df041383b4c5f692690516c351`).
   - Accumulated `FrontierCodexClient.call_structured` token/wall usage across all retries and `TimeoutExpired` attempts (`attempts_<seq>.json`).
5. **`C6` (Retention Alias/Shadowing Precision & End-to-End Temporal `ObservationContext`)**:
   - `inspect_final_deliverable_skill_retention` revokes top-level rebound/deleted skill aliases and excludes locally shadowed bindings from `entrypoint_reachable_l1_functions`.
   - `_evaluate_task_model_on_probe_input` and `AcquireObservation` preserve `past_driver_history`, `history_dt`, `timestamps`, `trajectory_ids`, and `initial_memory_states` across candidate evaluation, active probe acquisition, and checkpoint reload (`test_m9_1_1_audit_closure_suite.py`, `5/5` passed).
