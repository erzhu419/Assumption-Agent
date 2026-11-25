# Hegel M8.8 Unified Delivery Contract, Audit Closure (`M8.6` + `M8.7`), and 32-Worker Parallel Paired Generalization Report

- **Date**: `2026-11-25` (`UTC` run window `2026-11-25T15:25:43Z` – `2026-11-25T16:03:17Z`, `37.6 min` wall-clock)
- **Branch**: `exp/m6-causal-wormhole-sandbox`
- **Frozen Execution Code Commit**: `1372f490df4a154df3a59a1f59077bfbd993e5d0` (`code_worktree_clean = true`, `pre_run == post_run` `code_tree_sha256 = 720a33f3ae523e99596040f4400867e958e968cb595e7a786cc2cc652de67424`)
- **Model & Auth Backend**: `gpt-6.2-sol` via official ChatGPT Subscription CLI (`codex`, `auth_mode = "chatgpt"`, `0` `node007` calls)
- **Concurrency Architecture**: `max_episode_workers = 16`, `max_codex_concurrency = 32` (`BoundedSemaphore(32)` + thread-safe `_CODEX_SEQ_LOCK` + isolated per-episode worker `PersistentMechanismGraph`s merged deterministically post-run)
- **Sealed Evidence Bundle**: `docs/evidence/m8_8_parallel_paired_ablation_and_unseen_ood_20261125/` (`746 / 746` files verified via `sha256sum -c SHA256SUMS.txt`, `single_clean_invocation = true`, `resume_used = false`, `invocation_trial_id = m8_8_trial_20261125_88cf2874`)

---

## 1. Executive Summary & Triage of `Hegel_M8_6_Completion_Audit` and `Hegel_M8_7_Audit`

Following independent external audits of `M8.6` (`Hegel_M8_6_Completion_Audit`) and `M8.7` (`Hegel_M8_7_Audit`), `M8.8` reconciled overlapping recommendations, closed all deterministic engineering, verification-binding, static-packager, compiler-provenance, and negative-cache gaps across `src/`, expanded the regression suite to **`43 / 43` passing `M7 + M8` unit tests**, and executed a **single-invocation 16-episode (`8 tasks x 2 arms`) controlled paired experiment** running all `16` episodes concurrently across up to **`32` concurrent `gpt-6.2-sol` CLI processes**.

### Headline Empirical & Engineering Outcomes

1. **`100%` Pretraining-Only `L1` Primitive Qualification (`6 / 6` Source `L2` Modules -> `6 / 6` Verified `L1` Modules in `v4`)**:
   - Without lowering any numerical unit-test tolerances (`rtol=1e-13..1e-5`, `atol=1e-13..1e-4`), fixing the clean-container bare-module alias (`sys.modules[mod_stem] = _hegel_wh_mod`) and enforcing `require_all_public_fns_called = True` + substantive AST assertion checks recovered both previously uncompiled pretraining modules on Attempt 1:
     - `pt05` (`autogenous_control_panel_deconfounding.py`) $\rightarrow$ `autogenous_prim_reference_panel_balancing.py` (`align_reference_panel`, `balanced_cell_weights`, `weighted_column_fill`).
     - `pt02` (`autogenous_creep_corrected_retraction_calibration.py`) $\rightarrow$ `autogenous_prim_robust_trend_decomposition.py` (`cauchy_linear_fit`, `fit_anchored_relaxation`, `fit_local_level_step`).
   - Content-hash deduplication (`skip_already_compiled_sources = True`) retained the 4 existing `v3` `L1` modules at `0` re-billing cost while enriching all `6` `L1` modules with structured `mathematical_preconditions` and hardening `box_ridge_cube_lower_bound` with an explicit positive-semidefinite minorant check (`(X^T X + \lambda I) - \operatorname{diag}(d) \succeq 0`).
2. **Clean-Room Numerical Equivalence (`4 / 4` Core `L1` Modules, `11` Exported Functions) + Complete `v4` (`12 / 12`) & `M8.5` Repair (`2 / 2`) Clean-Room Load Proofs**:
   - Every exported function of the 4 core `L1` modules was executed on concrete NumPy/SciPy numerical inputs **before** static packaging (importing from `/tmp/hegel_wormhole_skills`) and **after** `auto_package_wormhole_imports_in_container` + `rm -rf /tmp/hegel_wormhole_skills` under `PYTHONPATH="" python3 -I`, producing **identical JSON numerical digests (`4 / 4`)**.
   - All `12 / 12` `v4` modules (`6` `L2` + `6` `L1`) AND all `2 / 2` `M8.5` compiled executable repair modules (`autogenous_spatial_dose_response_calibration.py`, `autogenous_cohort_consensus.py` from `docs/evidence/m8_5_unseeded_slice_and_executable_repair_20261125/part2_executable_repair_skill_transfer`) passed clean-room packaging and contract verification (`0` uninlined imports, `0` silent `0/0` vacuous checks).
3. **`16 / 16` (`100%`) `pass@1` AND `16 / 16` (`100%`) Whole-Policy `strict_pass` Across Both Exposed Regression and Brand-New Post-Freeze Unseen OOD Benchmarks**:
   - **Cohort A (`development_regression_exposed`, `4` `Holdout-12` numerical tasks: `ho06`, `ho01`, `ho04`, `ho11`)**: `4 / 4` `pass@1` and `4 / 4` `strict_pass` in **both** `l2_only` and `l1_l2`.
   - **Cohort B (`post_freeze_unseen_ood`, `4` brand-new unseen tasks authored post-freeze in `m8_8_unseen_postfreeze_benchmark.py`: `uf01..uf04`)**: `4 / 4` `pass@1` and `4 / 4` `strict_pass` in **both** `l2_only` and `l1_l2`.
4. **Decisive Causal Disentanglement of `M8.7` vs `M8.8` Adoption & Token Economics**:
   - Holding code commit, static packager, `api_quick_reference`, and a **strictly neutral permissive System Prompt Rule 5** (`"you may import and call it directly ... after checking its mathematical preconditions"`) identical across `l2_only` and `l1_l2` isolates what drove `M8.7`'s voluntary `L1` code adoption:
     - **Prompt Framing Sensitivity**: Switching from `M8.7`'s directive `"Whenever a mounted L1 primitive matches..., import and call it directly"` (`2 / 2` numerical tasks calling & retaining `L1` primitives) to `M8.8`'s permissive precondition-gated wording dropped voluntary code imports to `0 / 8` in `l1_l2`, while **conceptual mechanism transfer remained at `100%` (`8 / 8` `pass@1` in both arms)**.
     - **Multi-Turn Context Overhead**: Mounting `4` additional `L1` modules with verbose multi-paragraph `mathematical_preconditions` in `l1_l2` (`5–8` staged modules vs `1–4` in `l2_only`) without active code reuse increased prompt context across turns (`3,210,741 -> 4,231,748` cache-adjusted token equivalents, `+31.80%`) and pushed two `18`-turn trajectories (`ho01`, `uf04`) onto the selective router's `near_budget_exhaustion_ambiguity` boundary (`18 / 18` turns, despite `pass@1 = True`), spawning `2` redundant `Thesis` branches (`8.0u -> 10.0u`).

---

## 2. Audit Disposition Matrix (`M8.6 Completion` + `M8.7 Audit`)

| Audit Item | Source Audit | Triage Decision | Exact Code & Test Closure in `M8.8` |
|---|---|---|---|
| **`M88-POSTPKG-VERIFY`** (Post-package execution identity) | `M8.6` (`P0`) | **Adopted (`100%`)** | `m6_frontier_real_benchmarks.py`: When `auto_package_wormhole_imports_in_container` mutates target bytes (`post_package_sha256 != pre_package_sha256`), pre-package `ex_res` is **not** inherited into `verification_receipts_by_sha[post_package_sha256]`; `artifacts_modified_since_last_verify` remains `True` with an explicit notice so `submit_completed_task` binds strictly to post-package clean execution, while `verification_receipts_history` preserves all historical receipts. |
| **`M88-REPAIR-SYNTH-PKG`** (Packager coverage in `LocalContainerRepair` & `CodeSublation`) | `M8.6` (`P1`) & `M8.7` | **Adopted (`100%`)** | Wired `staged_skill_sources_by_stem` + `skill_packaging_receipt` into `execute_in_container_local_repair`, `choose_and_execute_container_mechanism_action`, and `reconcile_thesis_and_antithesis_publicly` (before `rm -rf /tmp/hegel_wormhole_skills` and clean-room contract/public benchmark checks). Verified in `test_m8_8_packager_reflection_private_attr_atomic_multifile_rollback_and_failclosed_scanner`. |
| **`M88-PKG-TXN` & `M88-SCAN-FAILCLOSED`** (Atomic multi-file rollback, reflection/private-attr ban, fail-closed scanner) | `M8.6` (`P1`) | **Adopted (`100%`)** | `m6_4_executable_wormhole_library.py`: Rejects `globals()`/`locals()`/`vars()`/`eval()`/`exec()` inside skill modules and private `m._helper` / dynamic `getattr(m, ...)` on whole-module aliases; `scan_wormhole_imports_in_container` fails closed (`["<SCAN_ERROR:...>"]`) on container/JSON errors; multi-file packaging stages all files first and rolls back every already-copied target on any failure. |
| **`M88-CACHE-VIEW`** (`PersistentMechanismGraph` negative cache view/incumbent binding) | `M8.6` (`P0`) & `M8.7` | **Adopted (`100%`)** | `m8_persistent_variational_planner.py`: `neg_scope_key` now hashes rounded float digests of `ds_tr` + `ds_va` (`view_sha10`), incumbent features/lifts/memory/role_maps (`inc_sig`), and `scope_sig6`. Reproduces and passes GPT-6's `81`-row validation-view swap counterexample (`-26.86 -> +2.83`) in `test_m8_8_negative_cache_binds_validation_view_and_incumbent_state`. |
| **`M88-WAVE1-MULTIFILE-DIGEST`** (Multi-file & non-empty hash check in `consolidate_wave1_via_aufhebung_compiler`) | `M8.6` (`P1`) & `M8.7` | **Adopted (`100%`)** | `m7_autogenous_forest_pretraining.py`: `_bundle_matches_expected_sha` requires non-empty `expected_rep_sha` across both single-file and multi-file `04a_repaired_antithesis_snapshot` directories, failing closed (`skipped_missing_or_mismatched_repaired_snapshot`) on empty or mismatched digests. |
| **`M88-NOGOOD-SCHEMA-RESOLVE`** (`risk_kind` & `resolves_nogood_ids` in strict OpenAI JSON schemas) | `M8.6` & `M8.7` | **Adopted (`100%`)** | `m6_frontier_real_benchmarks.py`: Exposed `risk_kind` and `resolves_nogood_ids` in `properties` AND `required` of both `HEGEL_M6_FRONTIER_DECISION_SCHEMA` and `HEGEL_M6_2_FRONTIER_DECISION_SCHEMA` (validated recursively for OpenAI Structured Outputs strict-mode compliance) and wired prior-nogood resolution (`resolved = True`, `superseded_by`). |
| **`M88-TEST-ALIAS` & `M88-COMPILER-PAYLOADS` & `M88-PRIMITIVE-CONTRACTS`** | `M8.7` (`C3`, `C4`, `C5`) | **Adopted (`100%`)** | `m7_autogenous_forest_pretraining.py`: Bound `sys.modules[mod_stem] = _hegel_wh_mod`, enforced `require_all_public_fns_called = True` + substantive AST assertion checks + `HEGEL_SKILL_CALLER_ROLE=compiler_self_test`, archived full pass/fail `attempts` payloads, enforced non-empty `mathematical_preconditions`, and added the PSD-minorant entry guard on `box_ridge_cube_lower_bound`. |
| **`M88-PRIMITIVE-RETRIEVAL` & `M88-RETENTION-AST`** | `M8.7` (`C2`, `C6`) | **Adopted (`100%`)** | `m6_4_executable_wormhole_library.py`: Added capability-indexed `L1` candidate scoring across `full_forest_archive` before `recalled_notes[:3]` truncation (`allow_l1_primitives` toggle) and `inspect_final_deliverable_skill_retention` checking canonical AST match + public entrypoint call-reachability on final locked snapshots. |
| **Sample Size & Serial Execution Conservatism** (`GPT-6` suggested small serial pilot) | `M8.7` | **Overridden per User Directive** | Scaled up to **`16` concurrent paired episodes (`8 tasks x 2 arms`)** across **`32` concurrent `gpt-6.2-sol` CLI workers**, completing both `development_regression_exposed` (`4` tasks) and brand-new `post_freeze_unseen_ood` (`4` tasks) in **`37.6` minutes** wall-clock time. |

---

## 3. Explicit Errata & Provenance Reconciliation for Sealed `docs/42` (`M8.7`)

Per strict immutability rules, previously sealed `docs/42_M8_7_Compositional_Primitives_and_OOD_Adoption_Report.md` and `docs/evidence/m8_7_compositional_primitives_and_ood_adoption_20261125/` are left untouched. The three narrative/reporting discrepancies noted in `Hegel_M8_7_Audit` are formally reconciled below against the raw JSON receipts:

1. **Exact `pt01..pt06` Pretraining Provenance Mapping (`10` Forest Nodes Across `4` `tree_id`s, `6` Unique `L2` Modules)**:
   - `pt01__dispersive_plate_flaw_tomography_v2` $\rightarrow$ `autogenous_dispersive_wave_tomography.py` (`node_auto_09`) $\rightarrow$ `autogenous_prim_ridge_profiled_matching.py` (**qualified in `v3` & `v4`**).
   - `pt02__viscoelastic_retraction_and_slip_calibration` $\rightarrow$ `autogenous_creep_corrected_retraction_calibration.py` (`node_auto_04`) $\rightarrow$ failed numerical assert in `v3`; **qualified in `v4` as `autogenous_prim_robust_trend_decomposition.py`**.
   - `pt03__high_contrast_darcy_multigrid` $\rightarrow$ `autogenous_conservative_elliptic_multigrid.py` (`node_auto_05`) $\rightarrow$ `autogenous_prim_conservative_spd_solve.py` (**qualified in `v3` & `v4`**).
   - `pt04__network_varproj_and_valve_assimilation` $\rightarrow$ `autogenous_robust_network_assimilation.py` (`node_auto_06`, `node_auto_08`) $\rightarrow$ `autogenous_prim_robust_affine_projection.py` (**qualified in `v3` & `v4`**).
   - `pt05__confounded_multicenter_deconfounding` $\rightarrow$ `autogenous_control_panel_deconfounding.py` (`node_auto_07`) $\rightarrow$ failed bare-module import in `v3`; **qualified in `v4` as `autogenous_prim_reference_panel_balancing.py`**.
   - `pt06__exact_l0_l2_compacted_certificate` $\rightarrow$ `autogenous_sparse_ridge_certification.py` (`node_auto_06`, `node_auto_08`) $\rightarrow$ `autogenous_prim_box_ridge_bounds.py` (**qualified in `v3` & `v4`**).
   - Accordingly, `M8.7` `ho06` composed `pt01` (`ridge_profiled_matching`) + `pt04` (`robust_affine_projection`) across two distinct physical domains and two `tree_id`s (`root_A01` & `root_A03`).
2. **`ho01` `M8.6 -> M8.7` Root Episode Wall-Time Comparison**:
   - `M8.6` `ho01` root episode wall time was `1,164.0s` (`839.2s` was the inner Antithesis branch wall time before `LocalContainerRepair`), so `M8.6 -> M8.7` root wall time changed `1,164.0s -> 882.5s (-24.18%)`.
3. **`ho01` `M8.7` Final Selected File Digest vs Turn-3 Packaging Event Digest**:
   - In `M8.7` `ho01`, `fd857f4a...` was the Turn-3 packaging event digest, whereas the final submitted and verified `04_submitted_artifacts_snapshot/tomography_solver.py` file SHA-256 was `766bb0e8aa1e1b16da3d6ec2278e4c21c2c78eb58d9d9aaa9900b55ba5190b07` (candidate bundle digest `95ce9c6d...`, verified AST-matching & entrypoint-reachable via `inspect_final_deliverable_skill_retention`).

---

## 4. `v4` Autogenous Library Qualification, Numerical Equivalence, & Compilation Ledger

- **Bundle**: `docs/evidence/m8_8_parallel_paired_ablation_and_unseen_ood_20261125/pretrained_autogenous_forest_v4_m88.json` (`SHA-256 = 5c0e7a6c5cbb119975c686f2c8a30b20060801ef425eef4260aa1e003dca9f87`)
- **Qualification Scope**: `require_pretraining_only = true` (`source_qualification_status = "pretraining_only_l1_primitive_sublation"`, `0` Holdout or Unseen OOD episodes in forest!).
- **Compilation & Amortization Cost**:
  - Historical one-time `M8.7` `v3` build (`4 / 6` qualified): `396,727` raw tokens (`275,383` cache-adjusted token equivalents).
  - Incremental `M8.8` `v4` pass (`4` retained via SHA-256 deduplication at `0` tokens + `2` newly compiled in parallel in `138.1s`): **`51,911` raw tokens** (`43,018` prompt, `14,208` cached, `8,893` completion; **`41,255` cache-adjusted token equivalents**).
- **Clean-Room Proofs (`v4_numerical_equivalence_and_repair_cleanroom_proof.json`)**:
  - Pre-package vs post-package isolated `python3 -I` numerical output equivalence: **`4 / 4` core `L1` modules (`11` functions) byte-identical JSON digests**.
  - `v4` clean-room static packaging + contract check: **`12 / 12` (`100%`)**.
  - `M8.5` executable repair skill clean-room packaging (`part2_executable_repair_skill_transfer`): **`2 / 2` expected, `2 / 2` discovered, `2 / 2` verified, `0` failed**.

---

## 5. Controlled 16-Episode Paired Experiment (`l2_only` vs `l1_l2`)

All `16` episodes (`4` `development_regression_exposed` tasks + `4` `post_freeze_unseen_ood` tasks $\times$ `2` arms) ran in a single clean invocation on frozen commit `1372f490df4a154df3a59a1f59077bfbd993e5d0` with deterministic randomized launch order (`seed = 20261125`), identical `M8OfflineContainerPlanner` calibrated strictly on `Frontier-20` (`m7_4`), identical static packager, identical `api_quick_reference` format, and identical permissive System Prompt Rule 5.

### 5.1 Cohort & Overall Aggregate Summary

| Cohort | Arm | Episodes | `pass@1` (Primary `1x`) | Whole-Policy `strict_pass` | Nominal Units | Raw Tokens (`prompt + out`) | Cache-Adjusted Token Equiv. | Agent Dynamic `L1` Calls | Final Retained `L1` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Cohort A: `development_regression_exposed`** (`ho06`, `ho01`, `ho04`, `ho11`) | `l2_only` | `4` | **`4 / 4` (`100%`)** | **`4 / 4` (`100%`)** | `4.0` | `1,787,316` | `1,546,452` | `0` | `0` |
| **Cohort A: `development_regression_exposed`** (`ho06`, `ho01`, `ho04`, `ho11`) | `l1_l2` | `4` | **`4 / 4` (`100%`)** | **`4 / 4` (`100%`)** | `5.0` | `2,304,493` | `2,008,717` | `0` | `0` |
| **Cohort B: `post_freeze_unseen_ood`** (`uf01`, `uf02`, `uf03`, `uf04`) | `l2_only` | `4` | **`4 / 4` (`100%`)** | **`4 / 4` (`100%`)** | `4.0` | `1,901,505` | `1,664,289` | `0` | `0` |
| **Cohort B: `post_freeze_unseen_ood`** (`uf01`, `uf02`, `uf03`, `uf04`) | `l1_l2` | `4` | **`4 / 4` (`100%`)** | **`4 / 4` (`100%`)** | `5.0` | `2,567,767` | `2,223,031` | `0` | `0` |
| **Combined (`8` Tasks)** | `l2_only` | `8` | **`8 / 8` (`100%`)** | **`8 / 8` (`100%`)** | **`8.0`** | **`3,688,821`** | **`3,210,741`** | `0` | `0` |
| **Combined (`8` Tasks, excl. `L1` compile)** | `l1_l2` | `8` | **`8 / 8` (`100%`)** | **`8 / 8` (`100%`)** | **`10.0`** | **`4,872,260`** | **`4,231,748`** | `0` | `0` |
| **Combined (`8` Tasks, incl. `M8.8` inc. compile)** | `l1_l2` | `8` | **`8 / 8` (`100%`)** | **`8 / 8` (`100%`)** | `10.0` | `4,924,171` | `4,273,003` | `0` | `0` |

### 5.2 Per-Task Paired Comparison (`l2_only` vs `l1_l2`)

| Task ID | Cohort | `l2_only` (`pass@1` / `pass` / `u` / `turns`) | `l2_only` Raw / Cache-Adj. Tokens (Wall `s`) | `l1_l2` (`pass@1` / `pass` / `u` / `turns`) | `l1_l2` Raw / Cache-Adj. Tokens (Wall `s`) | Staged Mods (`l2` vs `l1_l2`) | $\Delta$ Cache-Adj. Tokens (`l1_l2 - l2_only`) |
|---|---|---|---|---|---|---|---|
| `ho06__plate_evaporation_deconfounded_ic50` | `dev_exposed` | `T / T / 1.0u / 12t` | `372,496` / `327,088` (`510.9s`) | `T / T / 1.0u / 12t` | `430,083` / `375,939` (`430.9s`) | `4` vs `8` (`4 L1`) | `+48,851` (`-15.7%` wall) |
| `ho01__crosswell_anisotropic_clock_tomography` | `dev_exposed` | `T / T / 1.0u / 17t` | `576,708` / `485,892` (`850.2s`) | `T / T / 2.0u / 23t` (`18+5`) | `888,631` / `789,751` (`1352.8s`) | `2` vs `6` (`4 L1`) | `+303,859` (`near_budget` spawn) |
| `ho04__magnetometer_hard_soft_iron_calibration` | `dev_exposed` | `T / T / 1.0u / 13t` | `350,365` / `316,573` (`448.7s`) | `T / T / 1.0u / 14t` | `471,765` / `411,093` (`519.6s`) | `1` vs `5` (`4 L1`) | `+94,520` |
| `ho11__certified_capacitated_facility_bnb` | `dev_exposed` | `T / T / 1.0u / 16t` | `487,747` / `416,899` (`675.7s`) | `T / T / 1.0u / 14t` | `514,014` / `431,934` (`633.7s`) | `3` vs `7` (`4 L1`) | `+15,035` (`-2t`, `-6.2%` wall) |
| `uf01__electrical_resistivity_electrode_contact_tomography` | **`unseen_ood`** | `T / T / 1.0u / 16t` | `558,019` / `491,203` (`858.2s`) | `T / T / 1.0u / 14t` | `507,205` / `448,069` (`672.7s`) | `3` vs `7` (`4 L1`) | **`-43,134`** (`-2t`, `-21.6%` wall) |
| `uf02__batch_drifted_substrate_inhibition_kinetics` | **`unseen_ood`** | `T / T / 1.0u / 13t` | `404,139` / `346,731` (`642.4s`) | `T / T / 1.0u / 12t` | `423,708` / `376,476` (`453.5s`) | `3` vs `7` (`4 L1`) | `+29,745` (`-1t`, `-29.4%` wall) |
| `uf03__orthotropic_robin_heat_sink_inversion` | **`unseen_ood`** | `T / T / 1.0u / 13t` | `405,399` / `363,255` (`531.7s`) | `T / T / 1.0u / 13t` | `506,499` / `430,947` (`1046.6s`) | `4` vs `8` (`4 L1`) | `+67,692` |
| `uf04__certified_box_l0_sensor_selection` | **`unseen_ood`** | `T / T / 1.0u / 15t` | `533,948` / `463,100` (`938.6s`) | `T / T / 2.0u / 32t` (`18+14`) | `1,130,355` / `967,539` (`2083.5s`) | `3` vs `7` (`4 L1`) | `+504,439` (`near_budget` spawn) |

---

## 6. Scientific Interpretation & Architectural Design Laws for `M8.9+`

1. **Unseen Post-Freeze Generalization Validated (`8 / 8` Unseen Episodes Passed at `pass@1`)**:
   - All four post-freeze blind OOD benchmarks (`uf01` cross-borehole ERT with Helmholtz-basis electrode impedance nuisance + Cauchy IRLS, `uf02` 96-well Haldane substrate-inhibition kinetics under quadratic spatial gain drift, `uf03` 2D harmonic-flux orthotropic Robin FVM heat-sink inversion, and `uf04` certified box-constrained `L0 + L2` branch-and-bound sensor selection) were solved on the primary `1x` Antithesis branch in **both** arms (`8 / 8` `pass@1`), demonstrating strong out-of-distribution transfer of the `PhilosophicalForest` conceptual invariants and `M8` container verification discipline without any task-specific tuning.
2. **Why `M8.7` Adopted `L1` Code (`2 / 2`) While `M8.8` Neutral Framing Did Not (`0 / 8`)**:
   - In `M8.7`, System Prompt Rule 5 stated: *"Whenever a mounted `L1` primitive matches an inner subproblem..., import and call it directly"*. Under that explicit recommendation (with zero numeric call quotas), `gpt-6.2-sol` imported, executed, and retained `L1` primitives in `ho06` and `ho01`.
   - In `M8.8`, to keep `l2_only` and `l1_l2` 100% prompt-identical while incorporating `Hegel_M8_7_Audit`'s precondition emphasis, Rule 5 was softened to *"you may import and call it directly ... after checking its mathematical preconditions"*, alongside multi-paragraph `mathematical_preconditions` arrays in `mounted_executable_wormhole_skills`. Faced with strict multi-clause preconditions (e.g., shape/rank/minorant raising `ValueError`) vs writing 15 lines of tailored `numpy`/`scipy` directly, `gpt-6.2-sol` rationally chose self-contained inline synthesis while still applying the recalled mathematical decompositions (hence `uf01` and `uf02` actually finished in **fewer turns and `-21.6%` / `-29.4%` less wall time** in `l1_l2` than `l2_only`!).
3. **Two Concrete Design Laws Established by the Paired Ledger**:
   - **Law 1 — Progressive Disclosure of Skill Contracts (`Index Card -> On-Demand Inspect`)**: Mounting `8` full `api_quick_reference` + multi-paragraph `mathematical_preconditions` blocks into Turn 1's conversation history adds `~3k–6k` prompt tokens on *every subsequent turn* (`+48k` to `+94k` tokens per episode) and nudged `ho01` and `uf04` to `18 / 18` turns (`+808k` cache-adjusted tokens from 2 redundant challenger branches). Turn-1 manifests should expose **compact `<80`-token 1-line capability signatures**, leaving full preconditions inside module docstrings read via `python3 -c "help(...)"` only when matched.
   - **Law 2 — Complexity-Thresholded `L1` Staging**: Short algebraic helpers (`<25` LOC, such as 1-step ridge or Cauchy weights) have near-zero authoring cost for frontier models, whereas multi-invariant algorithmic kernels (`>60` LOC, such as Galerkin AMG preconditioners or certified perspective B&B lower bounds) carry real bug/turn risk when re-implemented from scratch. `L1` sublation and staging should target high-complexity algorithmic kernels rather than micro-wrappers.
