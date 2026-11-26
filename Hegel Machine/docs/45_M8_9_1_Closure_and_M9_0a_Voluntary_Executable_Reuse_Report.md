# Hegel M8.9.1 Deterministic Audit Closure & M9.0a Controlled Voluntary Executable Reuse Report (`docs/45`)

- **Branch**: `exp/m6-causal-wormhole-sandbox`
- **Three-Stage Clean Commit Separation**:
  - **Commit 1 (`be2a3b29fb978f19d94b9b9b8a19956ba96d3857`)**: `M8.9.1` deterministic correctness closure (`R1..R7`), standalone sound Box-$\ell_0$ certificate kernel (`m8_9_1_box_l0_certificate_kernel.py`), non-destructive sealed `M8.9` `hs01` independent regrade receipt (`18/18` pass; fake-LB negative control rejected), true 20-fold `LOTO` router receipt (`14/20` vs `13/20`), `57/57` pytest suite pass, and `docs/44b` formal errata addendum.
  - **Commit 2 (`99c8d82fec81bc4607879faf45a37ffef62a1331`)**: `M9.0a` zero-naming-hint stratified benchmark (`Hegel-M90a-VoluntaryReuse-4`: `vr01..vr04`), reference + negative-control container verifier suite (`8/8` pytest pass), and parallel interleaved experiment runner (`run_m9_0a_voluntary_reuse_and_knowledge_isolation.py`) frozen cleanly before live evaluation (`git_dirty_at_start == false`).
  - **Commit 3 (`HEAD`)**: Sealed `24`-episode (`3 Arms × 4 Tasks × 2 Independent Replicates`) `gpt-6.2-sol` live evidence (`docs/evidence/m9_0a_voluntary_reuse_and_knowledge_isolation_20261126/`, `778` files in `SHA256SUMS.txt`) and bilingual `docs/45` reports.

---

## 1. Executive Summary & Sovereign Audit Disposition (`Hegel_M8_9_Audit`)

### 1.1 Adopted Deterministic Engineering Closures (`M8.9.1`, `R1..R7`)

All reproducible counterexamples from `markdown/Hegel_M8_9_Audit` were closed and unit-tested (`tests/test_m8_9_1_audit_closure_and_certificate_kernel.py`, `57/57` total tests passing in `Commit 1`):

1. **`R1` (`Sound Box-L0 Subcube Certificate Kernel & Sealed HS01 Regrade`)**:
   - Implemented `src/hegel_machine/cognition/m8_9_1_box_l0_certificate_kernel.py` (`verify_box_l0_subcube_certificate`), accepting `1 <= len(leaves) <= max_leaves`, checking exact all-$2^p$-support continuous convex-QP leaf lower bounds ($p \le 16$) or proof-carrying Fenchel perspective dual (`dual_nu`) / PSD diagonal minorant (`minorant_diagonal`, `minorant_center`) witnesses (arbitrary $p$), and separating `solution_correct`, `partition_valid`, `leaf_bounds_valid`, `gap_satisfied`, and `certificate_valid`.
   - Upgraded `hs01` and `PT06` verifiers accordingly.
   - Non-destructively regraded all `6` unique SHA-pinned `M8.9` `hs01` deliverables across original private seeds `(99101, 99102, 99103)` ($p = 14, 14, 15$; `65,536` total convex-QP support solves): **all `18/18` artifact-case pairs pass (`max_gap <= 4.25e-7`)**, raising regraded `M8.9` pass rates from `7/8` to **`8/8` (`100%`) across all 3 arms (`Primary 1x` and `Full Policy`)** while preserving original `M8.9` labels untouched and rejecting GPT-6's fake-LB negative control (`leaf_bounds_valid == False`).
2. **`R2` (`Content-Addressed Qualification & Compiler Lineage`)**:
   - Keyed `enrich_and_harden_l1_primitive_contracts` reverification by `(stem, actual_code_sha256, synthetic_unit_test_sha256)` and propagated receipts strictly to byte-identical copies.
   - Keyed `sublate_autogenous_forest_into_compositional_primitives` L2 source collection by `(filename, recomputed_source_sha256)` and stamped `verified_code_sha256` on newly verified L1 receipts.
   - Added fail-closed `verify_l1_module_qualification_record` rejecting missing, empty, `False`, or SHA-mismatched receipts.
   - Updated `M8` container repair to inject real preconditions + prefetched top-k skill contracts and explicitly label `"repaired_digest_kind": "raw_single_file_sha256"`.
3. **`R3` (`Two-Phase Atomic Packaging & Bidirectional Reachable Retention`)**:
   - Upgraded `auto_package_wormhole_imports_in_container` to stage and SHA-verify all rewritten and rollback files on host before any `docker cp` commit.
   - Upgraded `inspect_final_deliverable_skill_retention` to enforce ordered helper-closure set equality, `SimpleNamespace` return export-map verification, and scope-aware static call-graph reachability (excluding uncalled nested `def`/`class` bodies), plus explicit `declared_callable_entrypoints` across all tasks.
4. **`R4` (`Substantive Verification & Proposition-Bound Nogood Witness`)**:
   - `is_substantive_verification_command` now rejects import-only `python3 -c "import sol"` smoke checks; `nogood` resolution witness promotion enforces `risk_kind` (`target_identifiability` requires an explicit anchor/gauge/identifiability witness) and `scope` alignment.
5. **`R5..R7` (`Router Schema Hash Binding, True LOTO, Planner Incumbent Purity, & Symmetric Progressive Catalog`)**:
   - Fixed `CONTAINER_STATE_FEATURE_NAMES` order (`unknown_code_or_ident_risk` at col 2, `unknown_biomystery` at col 3), enforced `(feature_schema_version, feature_names, feature_builder_sha256)` validation in `from_json_dict`, archived true 20-fold `LOTO` (`14/20` current vs `13/20` relaxed), stripped `dependent_task_ids`/`inapplicable_task_ids` from `inc_sig`, added temporal `ObservationContext` probe discrimination, replaced single-word `"iterative"` flipping with numeric scale detection (`>= 4000 unknowns`), unified the `6`-module progressive catalog, and added `progressive_knowledge_only` (`/tmp/hegel_knowledge_ref`).

### 1.2 Sovereign Strategic Decision Beyond Conservative Pause (`M9.0a`)

Rather than stopping at deterministic bug-fixes alone, we immediately froze `Commit 2` (`99c8d82fe`) and executed **`M9.0a`**: a clean, unresumed **`3-Arm × 4-Task-Family × 2-Independent-Replicate = 24`-Episode** live parallel experiment (`gpt-6.2-sol`, `auth_mode == "chatgpt"`, `max_turns = 28`, `0` skill naming hints in prompts, `0` online skill compilation) on frozen `v5` (`6` qualified `L1` modules, `17` functions) separating:
1. **`bare_no_library` (`Arm 1`)**: Empty forest/archive (`0` reference cards, `0` executable skills),
2. **`knowledge_only_reference` (`Arm 2`)**: Read-only `progressive_knowledge_only` reference cards + `/tmp/hegel_knowledge_ref/describe_ref.py` + `.reference.txt` notes (`0` importable `hegel_wormhole_skills` Python modules),
3. **`progressive_executable_library` (`Arm 3`)**: Importable `/tmp/hegel_wormhole_skills` Python library (`disclosure_mode="progressive"`, `staging_policy="net_value_gated"`) + two-phase atomic static dependency-closure packager.

---

## 2. `M9.0a` Pre-Run Task Hardness & Provenance Audit

Recorded in `docs/evidence/m9_0a_voluntary_reuse_and_knowledge_isolation_20261126/m9_0a_pre_run_hardness_and_applicability_audit.json` and `m9_0a_voluntary_reuse_master_summary.json`:

- **Single-Invocation Provenance**:
  - `code_commit_at_start == code_commit_at_end == "99c8d82fec81bc4607879faf45a37ffef62a1331"`
  - `source_tree_py_sha256_at_start == source_tree_py_sha256_at_end == "ddd58dd7199024647b1d30af999a5a70e031c9b590073d0c72198bae1922b966"`
  - `git_dirty_at_start = false`, `git_dirty_at_end = false`, `resumed_episode_count = 0`, `fresh_executed_episode_count = 24`, `single_clean_unresumed_invocation = true`
  - `frozen_v5_bundle_sha256 = "0fcb0ca4b8b664515f29c184cd531700c817710a8e7e8a2281ef212641a6888c"`, `online_skill_compilation_enabled = false`, `zero_skill_naming_hints = true`
  - Concurrency: `24` interleaved episode workers, `peak_concurrent_codex_calls = 24`, `total_codex_calls = 296`, `total_wall_clock_sec = 857.22s` (`14.29` min).
- **Stratified Task Regimes (`Hegel-M90a-VoluntaryReuse-4`)**:
  1. **`vr01__proof_carrying_box_l0_minorant_bnb` (`Exact Subproblem Fit + Non-Trivial Search`)**: Symmetric-box `[-M, M]^p` $\ell_0$-ridge at private $p = (20, 21, 22)$ (`1.05M`, `2.10M`, `4.19M` supports), measured unbranched root minorant gaps `(0.410, 0.526, 0.529)` and root Fenchel dual gaps `(1.238, 1.366, 1.390)`, polynomial-time proof-carrying leaf verifier.
  2. **`vr02__large_robin_spd_preconditioned_cg_multi_rhs` (`Partial Subroutine Fit`)**: `72 x 72 = 5,184` unknowns, `4,000x` conductivity contrast, series Robin BCs (where `conservative_grid_operator` zero-ghost Dirichlet assembly fails, while `galerkin_preconditioner` + `solve_residual_checked_cg` + `fit_ridge_nuisance` + `profile_candidate_amplitudes` match inner subproblems).
  3. **`vr03__prehistory_viscoelastic_control_panel_calibration` (`Temporal Pre-Window + Grouped Control Panel Composition`)**: `6` centers × `3` strata cohort with pre-window ZOH Prony loading + `11.1%` Cauchy spikes + `5`-column reference control panel (`12%` `NaN`) + Simpson's paradox.
  4. **`vr04__indefinite_helmholtz_wrapped_phase_inversion` (`Precondition-Violated Negative Control`)**: `32 x 32 = 1,024` complex symmetric indefinite Helmholtz (`18` strictly negative Hermitian-part eigenvalues, `min_eig = -0.1933`) + wrapped modulo-$2\pi$ receiver phase + impulse spikes (`autogenous_prim_conservative_spd_solve` vetoed by `indefinite_operator_violates_spd_precondition`).

---

## 3. `M9.0a` Main Quantitative Results (`Primary 1x` Strictly Separated from `Full Policy`)

### 3.1 Aggregate Performance Across All `8` Episodes per Arm (`4 Tasks × 2 Replicates`)

| Metric | Arm 1: `bare_no_library` | Arm 2: `knowledge_only_reference` | Arm 3: `progressive_executable_library` |
| :--- | :---: | :---: | :---: |
| **`Primary 1x` Strict Pass (`N=8`, Replicate SD)** | **`8/8` (`100.0%`, SD `0.0`)** | **`8/8` (`100.0%`, SD `0.0`)** | **`8/8` (`100.0%`, SD `0.0`)** |
| **`Full Policy` Strict Pass (`N=8`, Replicate SD)** | **`8/8` (`100.0%`, SD `0.0`)** | **`8/8` (`100.0%`, SD `0.0`)** | **`8/8` (`100.0%`, SD `0.0`)** |
| **Mean `Primary 1x` Turns (`max_turns = 28`)** | `9.00` | `9.62` | `10.62` |
| **`Primary 1x` Total Tokens (`8` eps sum)** | `1,626,388` | `2,344,308` (`+44.14%` vs Arm 1) | `2,563,061` (`+9.33%` vs Arm 2; `+57.59%` vs Arm 1) |
| **`Primary 1x` Effective Billed Tokens (`8` eps)** | `1,329,268` | `2,011,860` | `2,180,309` |
| **Mean `Primary 1x` Wall-Clock (s)** | `323.75s` | `342.25s` | `378.10s` |
| **Mean `Full Policy` Nominal Units (`1.0u..2.0u`)** | `1.750u` (`6/8` 2nd-branch*) | `1.375u` (`3/8` 2nd-branch*) | `1.250u` (`2/8` 2nd-branch*) |
| **`Full Policy` Total Tokens (`8` eps sum)** | `2,212,947` | `2,630,802` (`+18.88%` vs Arm 1) | `2,723,347` (`+3.52%` vs Arm 2; `+23.06%` vs Arm 1) |
| **Mean `Full Policy` Wall-Clock (s)** | `526.30s` | `440.86s` | `427.98s` |

> **Important Methodological Self-Audit Note on `Full Policy` Second-Branch Spawns (`*`)**: Inspection of `00_episode_summary.json` reveals that on non-`TB-Science` tasks without a task-specific host public probe, `M8OfflineContainerPlanner` assigns `public_verification_state = "UNKNOWN"` when `staged_module_count == 0` (`Arm 1`), triggering `SpawnDialecticalChallenger` (`6/8`), whereas `staged_module_count > 0` / `skill_activation_count > 0` (`Arms 2 & 3`) transitions state to `SUPPORTED_IN_PARTIAL_SCOPE` (`3/8` and `2/8` spawns). Therefore, **`Primary 1x` (`1.00u` single branch across all 24 episodes) is the strictly unconfounded primary comparison**, and the `Full Policy` nominal-unit reduction (`1.750u -> 1.250u`) MUST NOT be claimed as an intrinsic accuracy rescue effect!

### 3.2 Six-Stage Voluntary Skill Adoption & Precondition Abstention Funnel

| Adoption Stage | Arm 1: `bare_no_library` | Arm 2: `knowledge_only_reference` | Arm 3: `progressive_executable_library` |
| :--- | :---: | :---: | :---: |
| **Turn-1 Recommended `L1` Modules (mean)** | `0.00` | `2.50` | `2.50` |
| **Mounted Executable `L1` Modules on Disk** | `0.00` | **`0.00`** (`6.00` `.reference.txt` notes) | **`6.00`** (`/tmp/hegel_wormhole_skills`) |
| **Queryable `--describe` Catalog `L1` Modules** | `0.00` | `6.00` | `6.00` |
| **Positive-Fit (`vr01..vr03`, `N=6`) `--describe` CLI Used** | `0/6` (`0` queries) | `3/6` (`10` queries; `4/6` read `.reference.txt`) | **`6/6` (`100%`, `17` queries)** |
| **Positive-Fit (`vr01..vr03`, `N=6`) Dynamic `L1` Calls** | `0/6` (`0` calls) | `0/6` (`0` calls) | **`5/6` (`83.3%`, `58` calls)** |
| **Positive-Fit (`vr01..vr03`, `N=6`) Bidirectional Retained `L1`** | `0/6` | `0/6` | **`4/6` (`66.7%`, `100%` AST-verified)** |
| **Positive-Fit (`vr01..vr03`, `N=6`) Entrypoint-Reachable `L1`** | `0/6` | `0/6` | **`4/6` (`66.7%`)** |
| **Negative-Control (`vr04`, `N=2`) Invalid `L1` Calls / Retention** | `0/2` / `0/2` | `0/2` / `0/2` | **`0/2` (`0` calls) / `0/2` (`0` retained)** |

### 3.3 Per-Task Breakdown of Voluntary Adoption in `Arm 3` (`Zero` Prompt Naming Hints)

| Task Family (`2` Replicates per Arm) | `Arm 1` `Primary 1x` Pass / Mean Turns / Sum Tokens | `Arm 2` `Primary 1x` Pass / Mean Turns / Sum Tokens | `Arm 3` `Primary 1x` Pass / Mean Turns / Sum Tokens | `Arm 3` Voluntary `L1` Inspection, Dynamic Execution, & Final Reachable Retention |
| :--- | :---: | :---: | :---: | :--- |
| **`vr01` (`Proof-Carrying Box-L0`, $p=20..22$)** | `2/2` / `11.0t` / `525,149` | `2/2` / `13.5t` / `913,742` | `2/2` / `14.5t` / `924,916` | `2/2` `--describe` (`4` queries); **`2/2` dynamic `L1`** (`6` calls: `gram_diagonal_minorant`, `row_incidence_blocks`); **`1/2` (`rep_02`) entrypoint-reachable retained** (`gram_diagonal_minorant`). |
| **`vr02` (`5,184-DOF Robin SPD + Profiling`)** | `2/2` / `7.0t` / `284,663` | `2/2` / `7.0t` / `378,151` | `2/2` / `9.5t` / `540,649` (`rep_02`: `7t`, `186,256`!) | `2/2` `--describe` (`8` queries); **skipped `conservative_grid_operator` (Dirichlet mismatch!)** while **`2/2` dynamically called (`44` calls) & `2/2` entrypoint-reachably retained** `galerkin_preconditioner` + `solve_residual_checked_cg`! |
| **`vr03` (`Pre-History Viscoelastic + Panel`)** | `2/2` / `10.0t` / `494,907` | `2/2` / `9.5t` / `587,075` | `2/2` / `10.5t` / `669,335` | `2/2` `--describe` (`5` queries); **`1/2` (`rep_02`) dynamically called (`8` calls) & entrypoint-reachably retained** `align_reference_panel`, `balanced_cell_weights`, `weighted_column_fill`. |
| **`vr04` (`Indefinite Complex Helmholtz Neg-Ctrl`)** | `2/2` / `8.0t` / `321,669` | `2/2` / `8.5t` / `465,340` | `2/2` / `8.0t` / `428,161` | **`0/2` `--describe`, `0/2` dynamic `L1` calls, `0/2` retained `L1` — 100% clean precondition abstention!** |

---

## 4. Calibrated Scientific Conclusions & Implications for `M9.1`

1. **What `M8.9.1` + `M9.0a` Decisively Establish (Verified Mechanism Claims)**:
   - **Soundness & Audit Hygiene**: All `R1..R7` counterexamples (`hs01` certificate soundness, content-addressed qualification/compiler identity, two-phase atomic packaging, bidirectional entrypoint-reachable retention, proposition-bound `nogood` witness promotion, hash-bound router feature schema, and true `LOTO`) are deterministically closed and verified; sealed `M8.9` `hs01` artifacts pass `18/18` under independent continuous-QP regrading.
   - **Genuine Voluntary Reuse & Partial-Subroutine Selectivity Without Prompt Hints**: Even with zero skill names in task instructions, `gpt-6.2-sol` voluntarily inspects `--describe` (`6/6` positive-fit episodes), dynamically executes `L1` primitives (`5/6` episodes, `58` calls), selectively imports `galerkin_preconditioner` + `solve_residual_checked_cg` while avoiding the boundary-mismatched `conservative_grid_operator` (`2/2` on `vr02`), retains exact AST-verified `L1` closures reachable from declared entrypoints (`4/6`), and abstains `100%` (`0/2` calls) on the precondition-violated indefinite Helmholtz negative control (`vr04`).
2. **What `M9.0a` Equally Clearly Falsifies / Bounds (Calibrated Economic Net-Value Diagnosis)**:
   - **Ceiling Saturation & Turn-1 Prompt Tax**: On `28`-turn self-contained single-file tasks where `Bare` (`Arm 1`) already achieves **`8/8` (`100%`) `Primary 1x` pass rate in `9.0` mean turns**, neither `Knowledge-Only` (`+44.14%` `Primary 1x` tokens) nor `Progressive Executable` (`+57.59%` vs `Bare`, `+9.33%` vs `Knowledge-Only`) yields a pass-rate gain (`100% -> 100%`) or token reduction.
   - Crucially, the 3-arm decomposition proves that **`77%` of the primary token overhead (`+44.14%` of `+57.59%`) already arises in `Knowledge-Only` (`Arm 2 - Arm 1`)** from unconditional Turn-1 card injection + contract reading, while executable import + post-package re-verification adds only `+9.33%` (`Arm 3 - Arm 2`, or `+3.52%` whole-policy).
3. **Clear Architectural Direction for `M9.1` (`Demand-Triggered Zero-Turn-1-Tax Retrieval on Sub-100% Tasks`)**:
   - **Eliminate unconditional Turn-1 skill manifest injection**: Start every episode identically to `Bare` (`0` Turn-1 skill tokens!), exposing the library strictly on demand when an empirical `nogood` / residual failure occurs or when the agent explicitly queries a `1`-line capability search tool — guaranteeing `0%` token overhead on routine solvable tasks while preserving executable reuse on genuinely hard multi-component bottlenecks where `Bare 1x < 100%`.
