# Hegel M8.4 Audit Reconciliation (`M8.2` + `M8.3`), Boundary Isolation & Frozen Live Ladder Report

- **Branch**: `exp/m6-causal-wormhole-sandbox`
- **Audited Commits Reconciled**: `002fcc34c4fea186b0209ab4c1d16828bf7fe9b6` (`M8.2`) and `b3fdc3134005ae2b39e8c101f95e51c1daf721c0` (`M8.3`)
- **Frozen Code Commit Executed in Live Litmus**: `724aaa1033ff4f9b75012eaf1f594e632097e7f6` (`code_tree_sha256: da34c1f7e99cdadddac830504ed376bf7f8e88f781f5f6ddd324c64acca9c955`, identical pre-run and post-run)
- **Model Backend**: Official ChatGPT CLI (`codex`, `auth_mode == "chatgpt"`, `gpt-6.2-sol`)

---

## 1. Chronological Reconciliation & Sovereign Triage (`Hegel_M8_2_Audit` vs `Hegel_M8_3_Audit`)

### 1.1 Superseded Items (Raised in `Hegel_M8_2_Audit`, Already Closed in `b3fdc3134`, Confirmed by `Hegel_M8_3_Audit`)
1. **Main-Engine Mounting**: `run_m6_9_a_two_branch_episode` directly mounts public invariant probing and Tier-1 `LocalContainerRepair`.
2. **Real Challenger Execution**: Main-engine `SpawnDialecticalChallenger` executes `_exec_thesis()` and blind public synthesis rather than returning a standalone intent string.
3. **Live Guard Prompt Injection**: Persisted `container_invariance_guard` entries enter live main-engine repair prompts across checkpoints.

### 1.2 Genuine Technical Findings Adopted & Closed in `M8.4`
1. **Supervisor-vs-Candidate Process Isolation & Zero Host `exec`** (`m7_holdout_ood_blind_benchmark.py`, `m8_container_mechanism_adapter.py`, `m6_8_b_prime_capability_bridge.py`, `m7_autogenous_forest_pretraining.py`):
   - Both `run_holdout_private_verifier` and `_GENERAL_PUBLIC_INVARIANCE_PROBE_PY` now execute candidate Python modules inside an isolated `python3 -I` child RPC worker (`stdout` redirected to `stderr`, pure JSON+ndarray wire codec, zero `pickle`, zero access to `sentinel` or ground-truth variables), blocking `import __main__; _hegel_orig_json_dumps(...)` and `atexit` stdout spoofing.
   - Removed host `exec(compile(...))` from `lift_python_scientific_expressions_to_programs` (replaced with a pure side-effect-free recursive AST scalar evaluator) and blocked I/O calls (`savetxt`, `save`, `savez`, `tofile`, `memmap`, etc.) in `validate_and_decorate_autogenous_python_code`.
   - Upgraded `_extract_code_ast_operator_profile` to bind canonical `ast.dump` of public functions into `ast_sha12` (and wormhole metadata), invalidating stale `sub_opdag` subtrees under operator sign/expression edits (`sin(x)+x*x -> sin(x)-x*x`) on default `load_autogenous_forest_bundle`.
2. **Persistent Mechanism Graph Lifecycle, Temporal `u_prev`, & Pre-Action Budget Reservation** (`m8_persistent_variational_planner.py`):
   - Persisted `task_datasets` and `paid_observations` inside `hegel_m8_authoritative_state_v3` checkpoints so cold restarts resume prior tasks without `KeyError`, retain `observed_y`, and evaluate true shared-ancestor collateral deltas; added `TaskModel.bound_shared_weights` + `unverified_collateral_task_ids` (`COLLATERAL_UNVERIFIED_WITHHELD`) when datasets are explicitly evicted.
   - Restored full 3-element `[t_last, z_last, u_last]` in `TaskModel.from_dict` (`0.0` cold-restart validation drift) and prevented backward leakage of terminal state when re-evaluating training prefixes (`dt0 <= 0 -> z[0] = 0.0`, `0.0` RMSE discrepancy vs `fit_and_score_task_model`).
   - Enforced strict pre-execution `_try_spend` across all candidates and `AcquireObservation` (`actual_cost <= budget_remaining` with `0.0` overrun).
   - Added role-mapped sample-wise `scope_guard` & `support_bounds` checks (blocking out-of-scope `a > 10` reuse on `max(u) <= 2.7`) and autonomous `temporal_memory` recall across renamed columns.
3. **Public Probe Soundness, Soft Identifiability Tiering, & Collateral/Nogood-Aware Routing** (`m8_container_mechanism_adapter.py`, `m8_offline_container_planner.py`):
   - Note on Evidence Scope: Public probes use **public calibration bundles + private-label-free metamorphic transformations** (never private verifier seeds), and do **not** prove universal correctness over arbitrary unseen distributions.
   - Enforced `schema -> finite -> physical bounds -> accuracy -> equivariance` (`allow_nan=False`), rejecting `NaN` outputs in `ho06` and `ho10` (`REFUTED`), rejecting data-blind constant ring graphs in `ho05` via non-cyclic general permutations on both `adjacency` and `sparse_precision`, and verifying cohort prevalence + `[3.2, 7.5]` depth ratio + wildtype codon match + positive plasmid-tag renaming equivariance in `ho12`.
   - Converted `ho08` `65%`-prefix `G_inf` shift from a hard refutation into `identifiability_warning = True` (`SUPPORTED_IN_PARTIAL_SCOPE`), preserving acceptance of both historical M7.5 passing `ho08` solvers while preventing blind `hard_domain_verified_commit` when either `identifiability_warning` or an unresolved `RUNG_3` target-identifiability `nogood` is active.
   - Added conditional synthesis-damage feedback (`posterior_damage_rate` + two-sided NIG target) to `M8OfflineContainerPlanner`, preserving `14/20` raw Frontier LOTO and `12/12` Holdout-12 (`0.3333` spawn rate) while reacting immediately when synthesis harms passing primaries.
4. **Transactional Main-Engine Repair, Deferred Guard Publication, & Separate Decision Records** (`m8_container_mechanism_adapter.py`, `m6_9_a_dialectical_synthesis_engine.py`):
   - Wrapped staging + re-probing in `try/except/finally` so probe exceptions always restore `orig_code` and return a typed failure receipt preserving spent LLM tokens.
   - Deferred `SUPPORTED_IN_SCOPE` publication of content-addressed guards (`mech_guard_<slug>_<sha8>`, scoped by `symmetry_probe_family`) until secondary `verify_candidate_artifact_contract_in_container` also accepts.
   - Separated `repair_decision_record` (`0.35u` nominal units) from post-repair `branch_decision_record` (`CommitPrimary1x` vs `SpawnDialecticalChallenger`), archived immutable `04a_raw_antithesis_snapshot` / `04a_repaired_antithesis_snapshot` / `04a_failed_repair_attempt.py` snapshots, and merged worker `decision_ledger` + `action_return_history` into `PersistentMechanismGraph`.

### 1.3 Overly Conservative Suggestions Streamlined / Not Adopted
- Skipped expensive 12-hour full 20-task / 12-task benchmark re-runs in favor of deterministic reproduction of all audit counterexamples (`27/27` pytest modules passing) plus a code-frozen 2-OS-process live `gpt-6.2-sol` integration litmus (`288.7s`).
- Kept official ChatGPT CLI (`codex`, `gpt-6.2-sol`) rather than local `node007` models, and kept changes surgical inside existing M7/M8 modules rather than introducing new orchestration layers.

---

## 2. Verification Summary

1. **Deterministic & Container Regression Suite (`27/27` Passed)**:
   - `tests/test_m8_4_audit_reconciliation_and_controls.py` (`5/5` passed)
   - `tests/test_m8_3_general_symmetry_and_engine_ladder.py` (`2/2` passed)
   - `tests/test_m8_2_audit_closure_and_lifecycle.py` (`2/2` passed)
   - `tests/test_m8_offline_container_planner.py` (`2/2` passed)
   - `tests/test_m8_persistent_variational_planner.py` (`2/2` passed)
   - `tests/test_m7_audit_controls_and_fixes.py` (`9/9` passed)
   - `tests/test_m7_autogenous_forest_pretraining.py` (`3/3` passed)
   - `tests/test_m7_holdout_ood_blind_benchmark.py` (`2/2` passed)

2. **Code-Frozen 2-OS-Process Live `GPT CLI` (`gpt-6.2-sol`) Litmus** (`docs/evidence/m8_4_frozen_audit_reconciliation_litmus_20261125/00_m8_4_litmus_summary.json`):
   - **Code Freeze Provenance**: Executed on committed `HEAD = 724aaa1033ff4f9b75012eaf1f594e632097e7f6`; `pre_run` and `post_run` `code_tree_sha256` both equal `da34c1f7e99cdadddac830504ed376bf7f8e88f781f5f6ddd324c64acca9c955` (`code_tree_frozen_throughout_run = True`).
   - **OS Process 1 (`PID 530049`, Seeded Primary + Live `gpt-6.2-sol` Tier-1 Repair in Main Engine)**:
     - `ho06__plate_evaporation_deconfounded_ic50`: Raw confounded primary graded `False`; live 1-turn `LocalContainerRepair` (`24,265` prompt / `6,249` completion tokens, `1.35u` nominal units) passed process-isolated public calibration + renaming/permutation probes and secondary contract check, published `mech_guard_control_anchored_spa_1cff216a`, skipped `Thesis`, and passed air-gapped private verification (`strict_pass = True`).
     - `ho12__multicopy_plasmid_cohort_consensus`: Raw archived M8.1 failed primary graded `False`; live 1-turn `LocalContainerRepair` (`24,489` prompt / `3,315` completion tokens, `1.35u` nominal units) passed cohort prevalence + depth/codon witnesses + negative/positive plasmid-tag renaming equivariance and secondary contract check, published `mech_guard_opaque_labels_and_is_37d932ed`, skipped `Thesis`, and passed air-gapped private verification (`strict_pass = True`).
   - **OS Process 2 (`PID 536059`, Cold-Restart Checkpoint Reload & `ho08` Target-Risk Routing)**:
     - Restored `checkpoint_after_p1.json` (`sha256: 8ca3c7d4...`), recovering both guards, all `4` decision records, and both `action_return_history` action families; verified `scope_guard` domain filtering (`1` assay guard recalled on pharmacology vs `0` on rheology).
     - Re-evaluated M8.3 `ho08` candidate + `RUNG_3` equilibrium-modulus nogood: classified as `SUPPORTED_IN_PARTIAL_SCOPE` (`identifiability_warning = True`, `unresolved_target_identifiability_nogood = True`, zero false `REFUTED`) and routed via `learned_nig_posterior_utility` (`U = +0.28`) to `SpawnDialecticalChallenger`.
