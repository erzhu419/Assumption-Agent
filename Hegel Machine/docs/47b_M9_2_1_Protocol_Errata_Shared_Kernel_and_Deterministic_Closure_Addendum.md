# M9.2.1 Protocol Errata, Shared Certificate Kernel Regrade, and Deterministic Closure Addendum

- **Date**: 2026-11-26
- **Branch**: `exp/m6-causal-wormhole-sandbox`
- **Sealed M9.2.1 Evidence Bundle**: `docs/evidence/m9_2_1_audit_closure_and_regrade_20261126/` (`SHA256SUMS.txt` + `junit_m9_2_1_and_regressions.xml` verified; `0` paid LLM calls)
- **Immutable Upstream M9.2 Bundle Preserved**: `docs/evidence/m9_2_equal_information_on_demand_reuse_20261126/` (`315` files untouched)

---

## 1. `M9.2` Protocol Deviation & Telemetry Errata (`docs/47_*`)

1. **Container-Boot Public Fixture Deviation (`m9_2_protocol_deviation_and_funnel_errata.json`)**:
   - In `M9.2` (`1883431e5`), `m6_frontier_real_benchmarks.py` omitted `"Hegel-M92-EqualInfo-4"` from the container-boot `public_setup_py` branch, while `run_m9_2_equal_information_on_demand_experiment.py` hardcoded `shared_public_schema_validator_all_arms = True`.
   - Actual `M9.2` protocol state across all `24/24` episodes: `protocol_deviation_public_setup_missing = True`, `shared_public_schema_validator_all_arms_actual = False`. Because `K` and `E` could inspect library implementations while `B` constructed tests from scratch, `M9.2` represents policy performance without pre-staged public fixtures rather than a fully realized shared-fixture run.
2. **Equal-Information Reader Divergence in `M9.2`**:
   - Although `K` and `E` shared identical mathematical catalogs and reference sources (`4/4`), separate reader implementations in `M9.2` produced `16/24` rank differences and `9/24` member-set differences on offline replay. `M9.2.1` (`A3`) unifies both arms onto a single `_SHARED_CATALOG_READER_CORE_PY` (`0/24` rank or member disagreements in `m9_2_unified_reader_24_query_replay.json`).
3. **Funnel, Turn-1, Verifier Timing, & Negative-Control Precision**:
   - `K` `inspected_knowledge_ref_in_bash` was `True` in **`7 / 8`** sealed `M9.2` episodes (`run_m9_2` had queried missing key `inspected_knowledge_reference_txt`).
   - Episode `00_episode_summary.json` `turn1_prompt_tokens` now matches both `D_TURN_01` and `D2_TURN_01`.
   - `student_elapsed_sec` in `M9.2` measured monotonic verifier-window wall-clock (including GT generation in `eq03`/`eq04`), not isolated candidate process CPU; `M9.2.1` (`A8`) instruments child-worker `perf_counter()` and `process_time()` (`candidate_timing_breakdown`).
   - On `eq04`, `E` inspected `solve_and_certify_l0` twice, whereas `K` inspected `fit_ridge_nuisance` + `box_ridge_cube_lower_bound` in `rep_01` and issued `--search` only in `rep_02` (`0` `L1` calls or retentions in all episodes).

---

## 2. Deterministic Closures (`A1..A9`) & Fixed-Artifact Regrade

- **`A1` (`eq02_shared_kernel_fixed_artifact_regrade_12of12.json`)**: Deleted duplicated certificate math in `eq02` (`private_verifier_py` and `validate_public_schema.py`), delegating to centralized `m8_9_1_box_l0_certificate_kernel` (`verify_eq02_style_box_l0_certificate` with explicit `dual_nu_sign=-1.0` + Stage 2->3 public handoff check). Both negative-correlation Fenchel dual counterexamples and overlapping/incomplete schema counterexamples are rejected; all `6` frozen `M9.2` `eq02` deliverables pass **`12 / 12`** instances (`handoff_max_abs_diff <= 3.55e-15`).
- **`A2`**: Capability-driven `holdout_public_setup_py` bootstrap with pre-Turn-1 `required_public_assets` SHA-256 verification and dynamic `public_setup_receipt` aggregation.
- **`A3`**: Single shared `_shared_search_catalog` / `_shared_describe_catalog` across `K` and `E` (`0/24` disagreements).
- **`A4`**: Strict equality check on `(code_sha256, test_sha256, verifier_protocol_version, dependency_env_digest)`, immutable Docker `ImageId` + package manifest hashing (`hegel_env_v2:...`), independent host double-execution repeatability check, online `L1` compiler v2 issuance, and container-verified `pretrained_autogenous_forest_v7_m921.json`.
- **`A5`**: Host-controlled `HostControlledProbeReceipt` vs `agent_claimed_unverified` separation (closing `assert 1 == 1; print('gauge scale residual=0.0')`).
- **`A6`**: Pre-oracle `history_dt` inheritance & schema validation, per-trajectory `initial_memory_states` dict merge, rejection of unaligned new per-row temporal columns, and permanent paid-observation/decision cost logging on post-oracle validation failure.
- **`A7`–`A9`**: Per-attempt isolated response files (`resp_<seq>_att<i>.json`), `01b_provider_attempts/` sealing, `usage_unknown_attempt_count` / `cost_accounting_status` propagation, derivative-consistent nullspace twins + future-independent `critical_strain` in `eq03`, and unified 4-tuple manifest selection in `LocalContainerRepair`.
