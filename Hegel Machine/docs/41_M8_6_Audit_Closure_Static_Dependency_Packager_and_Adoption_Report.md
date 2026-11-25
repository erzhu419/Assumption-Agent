# Hegel M8.6 (`M8.5.1 + M8.6` Unified): `Hegel_M8_5_Audit` Closure, Static Wormhole Dependency Packager & Benign Numerical Adoption Report

- **Code Execution Commit (Frozen Pre-Run)**: `135bb20089147b5b33a834343cc4289fba2b3223` (`branch: exp/m6-causal-wormhole-sandbox`)
- **Pre/Post Run `code_tree_sha256`**: `48b5077b221197f76fa2304858913fdbc6c6faab072cc5f6605b726de99ea639` (`2,637` `.py` files across `src/`, `scripts/`, `tests/`; `0` mutations during execution)
- **Pre/Post Run `src_and_tests_sha256`**: `6b25c8c7bdea8ae485ea8dd2cdcc08019d34c9307dbf903d2c36fb5591a5ad44`
- **Model & Auth Backend**: Official ChatGPT Subscription CLI (`codex`, `auth_mode == "chatgpt"`, `gpt-6.2-sol`; zero `node007` endpoints)
- **Evidence Bundle**: `docs/evidence/m8_6_static_packager_and_benign_adoption_20261125/` (`184/184` files verified via `SHA256SUMS.txt`)

---

## 1. Sovereign Triage of `Hegel_M8_5_Audit`

Following engineering review of `Hegel_M8_5_Audit` (which confirmed `145/145` M8.5 SHA-256 digests, `10` DAGs / `179` records, `30/30` JUnit tests, and closure of all 12 M8.4 counterexamples), we adopted **all deterministic state/provenance/mathematical defects (`M8.5.1`)** and **the pre-verification static AST dependency closure packager + benign numerical adoption slice (`M8.6`)**, while rejecting multi-hour full-benchmark re-sweeps and any attempt to work around safety refusals on `ho12`.

### 1.1 Closed `M8.5.1` Deterministic Counterexamples

| Audit ID | Defect Reproduced in `Hegel_M8_5_Audit` | Deterministic Fix Implemented & Verified |
|---|---|---|
| **`M85-CACHE`** | `partial_role_search_control.py`: Under `budget=0.05`, evaluating only `s -> a` (`gain = -0.641`) wrote mechanism-level `partial_role_budget@v1:n81` into `inapplicable_task_ids`, blocking untried `s -> b` (`gain = +0.398`) on retry. | Key negative cache entries per evaluated role map (`...rm={"s":"a"}`); add mechanism-level `neg_scope_key` **only** when zero eligible role maps were skipped due to budget (`m8_persistent_variational_planner.py`). |
| **`M85-SCOPE`** | `planner_controls.py:scoped_math_domain`: `np.log(x['a'])` under `scope_guard: a > 0` failed `estimate_conduit_conditioning_and_domain` across negative rows (`0` recalled candidates). | Evaluate `scope_guard` mask **before** conditioning, out-of-support inflation, and `build_task_design_matrix` evaluation; evaluate `fn(x_sub)` strictly on active guarded rows (`0` `FloatingPointError` even under `np.errstate(invalid='raise', divide='raise')`). |
| **`M85-TIME`** | `planner_controls.py:interior_temporal_replay`: Evaluating committed `TaskModel` on interior slice `ts[10:25]` (`110..124`) without explicit boundary seed jumped from `t=99` (`RMSE = 1.9105` vs `8.2559e-7`). | Commit per-step trajectory states `train_step_memory_states` on `TaskModel`; `_resolve_eval_initial_memory_states` reconstructs the exact boundary `t=109` (`RMSE = 8.2559e-7`) and raises `ValueError` if an interior slice omits step history without an explicit boundary seed. |
| **`M85-TXN`** | `transaction_and_event_controls.py`: When `save_checkpoint` raised `OSError` inside `commit_verified_container_guard`, outer acceptance cleared `persisted_guard_id` but left `gid` inside `graph.mechanisms` and persisted it in `finally: save_checkpoint`. | Snapshot mutable `graph` & `forest` state before guard registration in both `commit_verified_container_guard` and outer acceptance; restore exact pre-guard state on any checkpoint/acceptance exception (`guard_present_after_cold_reload == False`, pre-existing mechanisms preserved, `100%` of LLM tokens retained). |
| **`M85-EVENT`** | `transaction_and_event_controls.py`: Two independent runs (`independent_run_one` vs `independent_run_two`) on `numeric_same_task` emitted identical `m8_rep_dec_001_numeric_same_task`, collapsing to `1` record on `merge_worker_graph_into_shared`. | Bind deterministic `run_id` slug+hash into `repair_dec_id` and `decide_second_branch(..., run_id=run_id)` (`2` distinct merged decisions & return events across independent runs; idempotent on repeated merge of the same run). |
| **`M85-SOURCE` / `M85-BIND` / `M85-TESTSRC`** | Raw snapshot fallback could inherit `repaired_public_verified_episode` if `04a_repaired` was missing; `compile_verified_repair_to_executable_skill` did not check SHA-256 or `probe_after`; `synthetic_unit_test_code` was not retained in forest metadata. | Enforce SHA-256 + `probe_after` verification in `compile_verified_repair_to_executable_skill`; fail closed (`skipped_missing_or_mismatched_repaired_snapshot`) when `repaired_verified=True` lacks matching repaired snapshot; persist `synthetic_unit_test_code`, `synthetic_unit_test_sha256`, and `source_candidate_sha256` in `mod_entry`. |
| **`M85-ROUTER` / `M85-NOGOOD`** | `utility_unit` label omitted UCB bonus; `from_json_dict` accepted legacy mixed-target dicts without `fail_nig_fit`; producer nogoods lacked `candidate_sha256` / `risk_kind`. | Set `utility_unit = "ucb_adjusted_net_accuracy_gain_estimate"`; guard `from_json_dict` against missing `fail_nig_fit`; tag legacy untyped `action_return_history` entries; stamp `nogood_id`, `candidate_sha256`, `risk_kind`, `resolved`, and `superseded_by` on producer nogoods. |

---

## 2. `M8.6` Static Dependency Closure Packager (`M86-CLOSURE`)

Implemented `package_self_contained_wormhole_dependencies` and `auto_package_wormhole_imports_in_container` in `src/hegel_machine/cognition/m6_4_executable_wormhole_library.py`, wired **before** candidate SHA-256 calculation and `verify_candidate_artifact_contract_in_container` in `m6_frontier_real_benchmarks.py` and `m8_container_mechanism_adapter.py`:

1. **Pure Host-Side AST Transitive Closure (`0` Host `exec`, `0` Truncation)**:
   - Resolves `from hegel_wormhole_skills.<mod> import ...`, `import hegel_wormhole_skills.<mod> [as alias]`, and `from hegel_wormhole_skills import <mod> [as alias]`.
   - Walks the full module AST to include all required scientific imports, module constants, default-argument expressions, private `_helper` functions, classes, and unexecuted conditional-branch helpers without line or function-count truncation.
   - Strips `@track_wormhole_skill` telemetry decorators while failing closed on `import *`, dynamic `import_module` / `__import__`, unknown symbols, or unsupported semantic decorators.
   - Encapsulates each packaged module inside a collision-free `__hegel_load_skill_<mod>()` namespace factory (preserving PEP 563 stringified annotations when `from __future__ import annotations` is present), preventing any name collision with candidate globals.
2. **Clean-Room Verification Proof (`mechanical_packager_cleanroom_proof.json` + `junit_m7_m8_35tests.xml`)**:
   - All **35/35** M7+M8 pytest tests passed (`pytest_receipts/junit_m7_m8_35tests.xml`).
   - All **6/6** autogenous modules in `pretrained_autogenous_forest_v2.json` (`autogenous_conservative_elliptic_multigrid`, `autogenous_control_panel_deconfounding`, `autogenous_creep_corrected_retraction_calibration`, `autogenous_dispersive_wave_tomography`, `autogenous_robust_network_assimilation`, `autogenous_sparse_ridge_certification`) passed container auto-packaging, physical `rm -rf /tmp/hegel_wormhole_skills` deletion, `verify_candidate_artifact_contract_in_container` (`computed_candidate_sha256 == post_package_sha256`), and isolated `PYTHONPATH="" python3 -I` execution.

---

## 3. Single-Invocation Live `gpt-6.2-sol` Benign Numerical Adoption Slice

Executed `scripts/run_m8_6_static_packager_and_benign_adoption.py` in a **single clean invocation** (`resume_used = False`, `invocation_wall_ms = 1,190,833.08 ms` = `19.85 min`) mounting the unseeded OOD `pretrained_autogenous_forest_v2.json` (`0` Holdout-12 exposure) and the Frontier-20-calibrated `M8OfflineContainerPlanner` (`0` Holdout-12 exposure):

| Task ID | Public State & Router Driver | Nominal Units | Turns | Billed Tokens | 5-Layer Skill Adoption Funnel (`Recalled -> Staged -> Inspected -> Dynamic Calls -> Packaged`) | Official Verifier |
|---|---|---:|---:|---:|---|---|
| `ho06__plate_evaporation_deconfounded_ic50` | `VERIFIED` (`U = -0.05695`) -> `CommitPrimary1x` | `1.00u` | `13` | `398,125` | `5 -> 4 -> True -> 0 -> 0` | **`PASS (1.0)`** |
| `ho01__crosswell_anisotropic_clock_tomography` | `SUPPORTED_IN_PARTIAL_SCOPE` (`U = +0.11341`) -> `SpawnDialecticalChallenger` | `2.00u` | `25` (`18A+7T`) | `781,713` | `5 -> 2 -> True -> 0 -> 0` | **`PASS (1.0)`** |
| **Slice Aggregate (`2` tasks)** | `1x` commit on `VERIFIED`; `2x` challenger on `PARTIAL_SCOPE` | **`3.00u`** | **`38`** | **`1,179,838`** | **`2/2` Recalled, `2/2` Staged, `2/2` Inspected in Bash, `0` Direct Function Calls** | **`2 / 2` (`100%`)** |

### Calibrated Scientific Diagnosis: Engineering Closure vs. Behavioral OOD Adoption

1. **Packaging Barrier Closed (Engineering)**: Direct `hegel_wormhole_skills` imports in target `.py` deliverables are now mechanically transformed into self-contained modules prior to `VerificationReceipt` hashing and separate-verifier grading.
2. **Why `inspected_in_bash == True (2/2)` while `dynamic_skill_activation_count == 0` on Benign OOD Tasks**:
   - With biological safety refusals (`ho12`) and manual-inlining friction (`M8.6`) both removed, the primary turn-by-turn logs (`02_human_readable_trajectory.md`, Turn 1–2 on both `ho06` and `ho01`) reveal the exact behavioral boundary:
     - On **`ho06`**, `gpt-6.2-sol` read `/tmp/hegel_wormhole_skills/autogenous_control_panel_deconfounding.py` (`sed -n '1,135p'`) and explicitly recorded under `Alternatives Weighed`: *"Apply the mounted control-panel aligner directly — rejected: The documented operator estimates a shared center-level transform, while this assay requires a position-dependent gain; inspect its control-separation logic for reuse."*
     - On **`ho01`**, `gpt-6.2-sol` inspected `autogenous_dispersive_wave_tomography.py` and `autogenous_creep_corrected_retraction_calibration.py` (`_robust_linear`) and recorded: *"Use waveform packet localization directly — rejected: The stated observables are scalar arrival picks, not waveforms; inspect the mounted operator for transferable nuisance-handling ideas."*
   - **Strict Scope Claim**: This 2-task slice (`2/2` `strict_pass`) does **not** claim full 12-task or 20-task benchmark superiority, nor does it claim spontaneous runtime function-call reuse on OOD schemas. Rather, it isolates with clean telemetry that across OOD scientific schemas (`pt01..pt06 -> Holdout-12`), pretrained Python operators currently act as **inspected algorithmic reference code (`2/2` inspected)** whose fixed function signatures (`X_train, center_train` or `waveforms`) are rationally adapted into task-specific equations rather than called verbatim. Achieving high runtime function-call adoption across OOD tasks will require **lower-level, schema-agnostic numerical primitives** (e.g., standalone 2D IRLS surface detrenders or Huber/Cauchy variable-projection kernels) rather than episode-level pipeline functions.
