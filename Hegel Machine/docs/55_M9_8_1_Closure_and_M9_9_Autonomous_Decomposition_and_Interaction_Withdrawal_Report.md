# Hegel M9.8.1 Audit Closure & M9.9 Equal-Tool Autonomous Decomposition and Counterexample-Scoped Interaction Withdrawal Report

- **Date**: `2026-11-28`
- **Pre-Confirmation Frozen Git Commit (`100%` of Paid LLM Calls)**: `7624c43bf72d283e82442504eff35ff6ee02b40d` (`paid_llm_working_tree_dirty_at_start = False`, `paid_llm_source_tree_unchanged_during_run = True`; post-run `0`-LLM ablation finalizer handled `alpha_m99_c03` empty-withdrawal branch cleanly)
- **M9.8.1 Closure Commit**: `b959102fa9c284efebb8921ce39229c4b23f45b1`
- **Shared Verification Kernel**: `hegel_shared_math_handoff_kernel_v3_m961_three_layer` | **Cost Policy**: `hegel_cost_policy_v2_m961_call_uuid_conserved`
- **Air-Gapped Sandbox Digest**: `sha256:bb111245520085c97361fff57b2e141911dd86a0e63c5c49761bc1ab692d910a`
- **Live Model & Concurrency (`R13`)**: `gpt-6.2-sol` (`reasoning_effort = medium`, `configured_max_codex_concurrency = 32`, observed `peak_active_codex_concurrency = 24`)
- **Usage Accounting Completeness**: `partial_unknown_usage` (`1,750/1,751` calls `100%` usage-complete; `0` unknown attempts on `Stage S`, `Hegel`, and `Monolithic`; `1` unknown attempt `x >= 0` on `Flat_PlainModuleRegistry_Online` `alpha_m99_c04/E1_down`, making `Hegel - Plain <= -2,535,683` a strict upper bound on `Hegel` cost)

---

## 1. Executive Summary & Honest Two-Baseline Attribution (`R01–R16` Closed)

This milestone completes two deliverables closing `100%` of the findings (`R01–R16`) in `Hegel_M9_8_Audit`:

1. **`M9.8.1` Full Technical, Evidence-Semantics, and Two-Table Attribution Closure (`0` Paid LLM Calls)**:
   - Hardened `SolverEvaluationWitness` (`v2` schema: `execution_status`, `evidence_origin`, nullable `metric_value` with zero `999.0` sentinels, `probe_id`, `setup_source_sha256`, Python/NumPy-invariant `canonical_numeric_sha256`, `dependency_bundle_sha256`, and `verification_trust_state`), `EmpiricalCoverageReceipt.verify_integrity()` (`R06`), and `ImmutableBranchLineageController._hydrate_from_graph()` (`R07`).
   - Executed `M9.8` two-plain-tables replay (`01_m98_executed_two_plain_tables_paid_prefix_replay.json`: `56/56` host-container pickle SHA-256 parity under `Python 3.10.12 / NumPy 2.2.6`, `32/56` public gate accepted $\to$ `32/32 = 100.0%` zero-LLM private `strict_pass`), proving honestly that on **purely separable** `U × D` streams with pre-given `u0/d0`, two factorized plain tables (`upstream_table(up_obs)` + `downstream_table(dn_obs)`) match the 2-layer graph's zero-LLM routing (`32/56`).
   - Executed all `16/16` `C1_down`/`C2_down` upstream `u0` holdout probes directly in Docker (`max RMSE = 0.00019837 <= 0.03`, `9734.25 ms`) and sealed the 64-run `(R0, U_iso)` twin ablation (`16/16` history-aware vs `0/16` prefix-locked masked vs `8/16` fixed `D0`/`D1`).

2. **`M9.9` Pre-Registered `n = 8` Unexposed Confirmation Campaign (`176` Live Containerized Episodes, `100%` Equal Tool Flags)**:
   - **Zero Host-Seeded `u0/d0` & Equal Permissions (`R02, R03` Closed)**: `Stage S` synthesizes an unfactored monolithic `solve_batch` (`8/8` pass on attempt 1). On `E1_down`, BOTH modular arms invoke `propose_and_verify_m99_autonomous_two_stage_factorization` (`100%` metered in `Scope-1`, `8/8` `Hegel`, `7/8` `PlainModuleRegistry` passing 4-probe Docker verification including non-triviality check). All 3 arms enforce `enable_auto_skill_packaging = False`, `allow_l1_primitives = False`, `max_pre_llm_probes = 1`, and identical large-margin Kleene guard compilation (`min_clause_margin = 0.035`), with byte-identical modular repair prompts across `PlainModuleRegistry` and `Hegel`.
   - **Result A — `Hegel_AutonomousInteractionGraph_Online` Decisively Outperforms `Flat_PlainModuleRegistry_Online` (`R01 + R02` Core Modular Attribution)**:
     - **Zero-LLM Passed Closures Doubled**: **`18/56` (`Hegel`) vs `9/56` (`PlainModuleRegistry`)** (`5` cluster wins, `3` ties, **`0` losses**, exact two-sided sign-flip `p = 0.0625`), driven by `7/8` content-addressed `SeparabilityWithdrawalReceipt`s preventing `D2_sep` / `D2_coupled` thrashing on `(E6, E7)` (`12/16` vs `3/16` zero-LLM passes across `(E6, E7)`).
     - **Lower Compute Cost**: Scope-1 `proxy_v2 = 11,050,627` vs `13,586,310 + x` (**`-18.66%`** or greater for any `x >= 0`, equal-cluster mean `-17.30%`, Scope-Total **`-16.96%`**, `mean Δv2 = -316,960.38 - x/8`, `95%` bootstrap CI **`[-576158.88, -72127.96]`** excluding `0`, two-sided sign-flip `p = 0.0625`, winning `6/8` clusters overall and `4/4` Beta clusters `p = 0.125`), while maintaining `strict_pass` parity (`50/56` vs `49/56`, `p = 1.0`).
     - **Executed `0`-LLM Factorized-Ceiling Control (`48` Verifier Runs Across All `8` Unfiltered Clusters at `step_05`)**: `hegel_selective_cross_factor_graph = 12/16` (`75.0%`) vs `factorized_fixed_D2_separable = 7/16` (`43.75%`) and `factorized_fixed_D2_coupled = 6/16` (`37.5%`) — strictly surpassing the `<= 50%` (`<= 8/16`) theoretical ceiling of any factorized two-table selector (`16/20 = 80.0%` vs `45.0% / 40.0%` combined across `dev + confirmation`).
   - **Result B — Honest Short-Horizon Amortization Boundary Against `Flat_MonolithicConditionalTable_Online` (`R02`)**:
     - Over this intentionally minimal `7`-step horizon (`1` unseen separable product cell `E4_valid_comb`, where modular arms pay upfront live decomposition cost `367,190` `v2`), **Beta (`n = 4` PDE/KKT clusters)** achieves `28/28` `strict_pass`, `12/28` vs `8/28` zero-LLM closures, and **`4/4` cluster cost wins** (`mean Δv2 = -201,744.75`, `beta_sign_flip_p = 0.125`, replicating `2/2` dev calibration wins).
     - On **Alpha (`n = 4` nonlinear viscoelastic Prony + cubic bridge clusters)**, `Hegel` wins `alpha_m99_c01` (`-243,947` `v2`, `5/7` pass matching `Monolithic` `5/7`), whereas on `alpha_m99_c02..c04`, autonomous module boundary drift under shifted `v3_gains` nonlinear cubic extrapolation (`E7`), deferred `E2_down` decomposition timing (`c02`), or single-episode `E2_down` modular synthesis miss (`c03`) incurs extra repair turns, yielding **overall 8-cluster cost parity** with `Monolithic` (`11,050,627` vs `11,268,739` = **`-1.94%`** Scope-1 / **`-1.73%`** Scope-Total, `mean Δv2 = -27,264.00`, `95%` CI `[-191565.48, 158164.51]`, `p = 0.765625`; `zero_turn_passed_closures = 18/56` vs `15/56`, `p = 0.453125`; `strict_pass = 50/56` vs `54/56`, `p = 0.5`), while cutting non-LLM empirical coverage container CPU by **`3.32×`** (`207.6s` vs `689.3s`).

---

## 2. Primary Three-Arm Confirmation Ledger (`n = 8` Unexposed Clusters, `176` Episodes)

| Arm (`100%` Equal Tool Flags) | `strict_pass` | `0`-LLM Passed Closures | `Scope-1` Calls (`solve` / `decomp` / `guard`) | `Scope-1 proxy_v1` | `Scope-1 proxy_v2` | `Scope-Total proxy_v2` (`Stage S + Scope-1`) | `Hegel` vs Row (`Scope-1` / `Total`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicConditionalTable_Online` | `54/56` | `15/56` | `538` (`499` / `0` / `39`) | `11,045,607` | `11,268,739` | `12,637,624` | `-1.94%` / `-1.73%` |
| `Flat_PlainModuleRegistry_Online` | `49/56` | `9/56` | `632` (`593` / `19` / `20`) | `13,044,562 + x_1` | `13,586,310 + x` | `14,955,195 + x` | **`-18.66%` / `-16.96%`** (`x >= 0`) |
| **`Hegel_AutonomousInteractionGraph_Online`** | **`50/56`** | **`18/56`** | **`505`** (`458` / `14` / `33`) | **`10,510,878`** | **`11,050,627`** | **`12,419,512`** | reference |

- **Shared `Stage S` Baseline Source Ledger (`8/8` Pass, `0` Unknown Attempts)**: `76` calls, `proxy_v1 = 1,319,886`, `proxy_v2 = 1,368,885` (if allocated `1/3` per arm: `+456,295` `v2` per arm, giving `Hegel` vs `Plain` `-18.08%` and `Hegel` vs `Monolithic` `-1.86%`).
- **Separate Paired Exact Sign-Flip Tests (`n = 8` Independent Clusters, `R10`)**:
  - `Hegel - Flat_PlainModuleRegistry_Online`: `mean Δv2 = -316,960.38 - x/8` (`95%` bootstrap CI `[-576158.88, -72127.96]`), **cost `p = 0.0625`** (Alpha `p = 0.75`, Beta `p = 0.125`), **zero-LLM closure `p = 0.0625`** (`diffs = [0, 0, 2, 0, 2, 1, 2, 2]`), **quality `strict_pass` `p = 1.0`** (`diffs = [-1, -2, 1, 2, 0, 1, 0, 0]`).
  - `Hegel - Flat_MonolithicConditionalTable_Online`: `mean Δv2 = -27,264.00` (`95%` bootstrap CI `[-191565.48, 158164.51]`), **cost `p = 0.765625`** (Alpha `p = 0.5`, Beta `p = 0.125`), **zero-LLM closure `p = 0.453125`** (`diffs = [1, -1, 0, -1, 1, 1, 1, 1]`), **quality `strict_pass` `p = 0.5`** (`diffs = [0, -2, -2, 0, 0, 0, 0, 0]`).

---

## 3. Cluster-by-Cluster Paired Ledger (`n = 8` Unexposed Confirmation Worlds)

| Cluster (`Family`) | `Monolithic` Pass (`0`-LLM) | `Monolithic` `v2` | `PlainModuleRegistry` Pass (`0`-LLM) | `PlainModuleRegistry` `v2` | `Hegel` Pass (`0`-LLM) | `Hegel` `v2` | `Hegel - Plain` `Δv2` | `Hegel - Mono` `Δv2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `alpha_m99_c01` (`alpha`) | `5/7` (`1` 0-LLM) | `1,817,884` | `6/7` (`2` 0-LLM) | `1,538,139` | **`5/7` (`2` 0-LLM)** | **`1,573,937`** | `+35,798` (`+2.33%`) | `-243,947` (`-13.42%`) |
| `alpha_m99_c02` (`alpha`) | `7/7` (`2` 0-LLM) | `1,384,580` | `7/7` (`1` 0-LLM) | `1,694,689` | **`5/7` (`1` 0-LLM)** | **`1,625,970`** | `-68,719` (`-4.05%`) | `+241,390` (`+17.43%`) |
| `alpha_m99_c03` (`alpha`) | `7/7` (`2` 0-LLM) | `1,459,773` | `4/7` (`0` 0-LLM) | `2,572,510` | **`5/7` (`2` 0-LLM)** | **`1,633,626`** | `-938,884` (`-36.50%`) | `+173,853` (`+11.91%`) |
| `alpha_m99_c04` (`alpha`) | `7/7` (`2` 0-LLM) | `1,486,913` | `5/7` (`1` 0-LLM) | `1,687,401 + x` | **`7/7` (`1` 0-LLM)** | **`1,904,484`** | `+217,083 - x` (`+12.86%`) | `+417,571` (`+28.08%`) |
| `beta_m99_c01` (`beta`) | `7/7` (`2` 0-LLM) | `1,307,632` | `7/7` (`1` 0-LLM) | `1,724,869` | **`7/7` (`3` 0-LLM)** | **`1,083,332`** | `-641,537` (`-37.19%`) | `-224,300` (`-17.15%`) |
| `beta_m99_c02` (`beta`) | `7/7` (`2` 0-LLM) | `1,265,233` | `6/7` (`2` 0-LLM) | `1,256,286` | **`7/7` (`3` 0-LLM)** | **`1,056,855`** | `-199,431` (`-15.87%`) | `-208,378` (`-16.47%`) |
| `beta_m99_c03` (`beta`) | `7/7` (`2` 0-LLM) | `1,340,092` | `7/7` (`1` 0-LLM) | `1,611,170` | **`7/7` (`3` 0-LLM)** | **`1,031,739`** | `-579,431` (`-35.96%`) | `-308,353` (`-23.01%`) |
| `beta_m99_c04` (`beta`) | `7/7` (`2` 0-LLM) | `1,206,632` | `7/7` (`1` 0-LLM) | `1,501,246` | **`7/7` (`3` 0-LLM)** | **`1,140,684`** | `-360,562` (`-24.02%`) | `-65,948` (`-5.47%`) |

---

## 4. Event-by-Event Causal Trajectory (`7` Continual Events × `8` Clusters)

| Stream Event | `Flat_MonolithicConditionalTable_Online` | `Flat_PlainModuleRegistry_Online` | `Hegel_AutonomousInteractionGraph_Online` |
|---|---|---|---|
| `E1_down` | `8/8` pass, `0/8` 0-LLM (`2,276,080`) | `8/8` pass, `0/8` 0-LLM (`2,699,681+x`) | **`8/8` pass, `0/8` 0-LLM (`2,958,613`)** |
| `E2_down` | `8/8` pass, `0/8` 0-LLM (`2,204,585`) | `7/8` pass, `0/8` 0-LLM (`2,618,713`) | **`6/8` pass, `0/8` 0-LLM (`2,618,803`)** |
| `E3_up` | `8/8` pass, `0/8` 0-LLM (`2,314,814`) | `8/8` pass, `0/8` 0-LLM (`1,825,362`) | **`8/8` pass, `0/8` 0-LLM (`1,935,624`)** |
| `E4_valid_comb` | `8/8` pass, `0/8` 0-LLM (`1,905,926`) | `6/8` pass, `6/8` 0-LLM (`527,575`) | **`7/8` pass, `6/8` 0-LLM (`462,190`)** |
| `E5_coupled_break` | `7/8` pass, `0/8` 0-LLM (`2,304,938`) | `8/8` pass, `0/8` 0-LLM (`2,525,615`) | **`8/8` pass, `0/8` 0-LLM (`2,424,695`)** |
| `E6_separable_revisit` | `8/8` pass, `8/8` 0-LLM (`0`) | `4/8` pass, `0/8` 0-LLM (`2,069,789`) | **`6/8` pass, `6/8` 0-LLM (`443,205`)** |
| `E7_coupled_revisit` | `7/8` pass, `7/8` 0-LLM (`262,396`) | `8/8` pass, `3/8` 0-LLM (`1,319,575`) | **`7/8` pass, `6/8` 0-LLM (`207,497`)** |

---

## 5. Non-LLM Container Probe Wall-Clock Ledger (`R11`)

| Arm | Pre-LLM Gate (`ms`) | Post-LLM Gate (`ms`) | Parent Localization (`ms`) | Decomp Verification (`ms`) | Empirical Coverage (`ms`) | **Total Non-LLM Agent Probes (`ms`)** | Official Private Verifier (`ms`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicConditionalTable_Online` | `343,727.5` | `238,770.6` | `0.0` | `0.0` | `689,334.9` | **`1,271,833.0`** | `281,695.4` |
| `Flat_PlainModuleRegistry_Online` | `343,242.0` | `269,890.7` | `23,124.6` | `99,936.1` | `397,631.6` | **`1,133,825.1`** | `278,044.5` |
| `Hegel_AutonomousInteractionGraph_Online` | `342,415.0` | `215,443.8` | `19,457.8` | `110,218.1` | `207,640.0` | **`895,174.8`** | `279,372.0` |

---

## 6. Grand Campaign Ledger Across Dev Calibration & Confirmation (`R09 / A02`)

1. **Pre-Freeze Diagnostic Dev Run (Cancelled at Step 3 on `dev` seeds `850100/860100` ONLY — `0` Confirmation Seeds Touched)**: Diagnosed `n = 1` singleton-witness 1-clause ambient-noise shortcut in Kleene guard compilation; resolved symmetrically across all 3 arms via `verify_m99_large_margin_witness_partition(min_clause_margin=0.035)` prior to freezing `M9.9`.
2. **Sealed `n = 2` Pre-Freeze Dev Calibration (`docs/evidence/m9_9_dev_live_calibration_20261128`, `1,398` SHA-256 verified files)**: `392` logical calls (`0` unknown usage attempts, `peak_concurrency = 6`), `Hegel` `14/14` `strict_pass` (`6/14` 0-LLM) vs `PlainModuleRegistry` `12/14` (`4/14` 0-LLM, `mean Δv2 = -330,801.0`) vs `Monolithic` `14/14` (`4/14` 0-LLM, `mean Δv2 = -202,667.0`); `0`-LLM factorized ceiling control `4/4 (100%)` vs `2/4 (50%)` / `2/4 (50%)`.
3. **Sealed `n = 8` Unexposed Confirmation Campaign (`docs/evidence/m9_9_autonomous_decomposition_and_interaction_withdrawal_20261128`, `6,226` SHA-256 verified files)**: `1751` logical calls (`peak_concurrency = 24` under `cap = 32`), `1` unknown-usage attempt on `PlainModuleRegistry` (`x >= 0`). **Combined Live Campaign Total (`dev + confirmation`)**: `2,143` mirrored `gpt-6.2-sol` calls.

---

## 7. Complete `Hegel_M9_8_Audit` (`R01–R16`) Closure Matrix & Remaining Honest Boundaries

| Finding | Resolution & Verification Evidence | Status |
|---|---|---|
| `R01` (Two-plain-tables baseline) | Executed `0`-LLM two-table replay on `M9.8` (`32/32` 0-LLM pass) + live equal-tool `Flat_PlainModuleRegistry_Online` arm in `M9.9` (`Hegel` doubles 0-LLM closures `18/56` vs `9/56` and cuts `v2` by `-18.66%`, `95%` CI `[-576,159, -72,128]`). | **CLOSED** |
| `R02` (Host-seeded `u0/d0` & modular repair prompts) | `0` host-seeded `u0/d0`; live `propose_and_verify_m99_autonomous_two_stage_factorization` (`100%` metered in `Scope-1`) + byte-identical modular repair prompts across `PlainModuleRegistry` and `Hegel`; honest short-horizon comparison vs `Monolithic` (`4/4` Beta wins, `-1.94%` 8-cluster parity). | **CLOSED** |
| `R03` (`enable_auto_skill_packaging` / `allow_l1_primitives` flag parity) | Both flags forced `False` across `Stage S` and all 3 continual arms (`test_m99_mocked_three_arm...` asserts on every call). | **CLOSED** |
| `R04` (`C1_down`/`C2_down` unexecuted `0.0` RMSE) | `diagnose_m98_factored_conflict_locus` emits `execution_status="not_executed", chosen_up_holdout_rmse=None`; all `16/16` `M9.8` probes directly executed in Docker (`max RMSE = 0.00019837`). | **CLOSED** |
| `R05` (`TimeoutExpired` / `999.0` sentinel) | `TimeoutExpired` and Docker `rc=125` fail closed to `locus="unknown_infra", chosen_up_holdout_rmse=None`, blocking false negative witness writes. | **CLOSED** |
| `R06` (`EmpiricalCoverageReceipt.verify_integrity()` `ecov_invalid` bypass) | Unconditionally enforces 29-char `ecov_` SHA-256 equality and witness/count invariants (`test_m9_8_1_audit_closure_suite.py`). | **CLOSED** |
| `R07` (`_hydrate_from_graph` missing-hash & legacy trust bypass) | Rejects missing `state_sha256`/`witness_content_sha256` on current schemas, enforces binding/code SHA foreign keys, and isolates legacy schemas to `unverified_legacy` (`0` minting authority). | **CLOSED** |
| `R08` (`HistoryMasked` fixed-`D0` vs prefix-locked twin ablation) | Executed `64`-run 4-policy Docker matrix (`16/16` history-aware vs `0/16` prefix-locked masked vs `8/16` fixed `D0`/`D1`). | **CLOSED** |
| `R09` (`Scope-Total` & ratio-of-sums vs cluster-mean reporting) | Reported side-by-side in `docs/54*`, `docs/54b*`, and §2 above. | **CLOSED** |
| `R10` (Separate cost vs quality vs 0-LLM `p`-values) | Reported separately in `docs/54*`, `docs/54b*`, and §2 above. | **CLOSED** |
| `R11` (Localization & decomposition container probe ledger) | Full `probe_id` and `wall_ms` ledger metered across all probe classes (§5). | **CLOSED** |
| `R12` (Witness `public_batch.pkl` vs `0444` dependency & Python/NumPy pickle drift) | Added `compute_canonical_numeric_item_sha256`, `setup_source_sha256`, and `dependency_bundle_sha256`; verified `56/56` host-container pickle SHA match. | **CLOSED** |
| `R13` (`cap=32` vs observed `peak` concurrency) | Reported side-by-side (`configured = 32`, `peak = 24`). | **CLOSED** |
| `R14` (`M9.7` Beta `p = 1.0` & KKT `L1` open gap in `docs/53b*`) | Reconciled in `docs/53b*` (`p = 1.0`, `R1` retired, `A14` KKT `L1` open gap). | **CLOSED** |
| `R15` (5-status `A01–A17` errata table in `docs/54*`) | Reconciled in `docs/54*` and `docs/54b*`. | **CLOSED** |
| `R16` (`Table` twin `15/16` wording in `docs/54*`) | Reconciled in `docs/54*` and `docs/54b*` (`15/16` `R0 + U_iso` 0-LLM). | **CLOSED** |

### Remaining Open Scientific Boundaries
- **Proven in `M9.9`**: Under equal tools and zero host-seeded decomposition, `Hegel_AutonomousInteractionGraph_Online` forms verified 2-stage decompositions online (`8/8` `Hegel`, `7/8` `Plain`), reuses unseen separable combinations (`7/8` pass, `6/8` zero-LLM closures on `E4_valid_comb` vs `0/8` zero-LLM `Monolithic`), and withdraws separability on falsified `(u, d)` pairs (`7/8` `SeparabilityWithdrawalReceipt`s) to outperform two-plain-tables `Flat_PlainModuleRegistry_Online` (`18/56` vs `9/56` 0-LLM closures, `-18.66%` `v2` cost, `75.0%` vs `43.75%/37.50%` frozen `step_05` ceiling control).
- **Honest Boundary vs Monolithic Conditional Tables & Next Questions**: Over a minimal `7`-step stream (`1` unseen separable cell `E4`), unseeded decomposition overhead beats `Monolithic` consistently when modular interfaces hold across gain shifts (`4/4` Beta confirmation + `2/2` dev), but breaks even (`-1.94%` overall, `p = 0.7656`) on `3/4` Alpha confirmation worlds (`strict_pass` `50/56` vs `54/56`). Section 8 below provides the complete episode-level forensic attribution, combinatorial amortization formula, and explicit decision matrix (`D1–D4`) handed off to the next independent audit (`GPT-6`).

---

## 8. Deep-Dive Diagnosis of Open Question 2 & Decision Matrix for Next Independent Audit (`GPT-6`)

To support an unvarnished architectural adjudication in the next independent `GPT-6` audit, we document the exact empirical forensics behind the Alpha-vs-Beta divergence against `Flat_MonolithicConditionalTable_Online`, derive the exact combinatorial break-even equation under zero-seeded online decomposition, and formulate four concrete design decisions (`D1–D4`).

### 8.1 Forensic Attribution Across All `8` Confirmation Worlds (`4` Beta Wins vs `4` Alpha Trajectories)

1. **Why Beta (`beta_m99_c01..c04`, 2D Elliptic PDE + Box-Simplex KKT) Achieved `28/28` Pass & `4/4` Cost Wins vs BOTH Baselines (`mean Δv2 = -201,744.75` vs `Monolithic`, `-445,240.25` vs `Plain`)**:
   - Every Beta campaign supplies structured `sensor_calibration_sweeps` spanning the full transducer range, allowing the autonomously decomposed upstream operator `u1` (`calibrate_upstream`) to re-fit and invert the coupled cubic rosette map dynamically per campaign with `< 5e-5` holdout RMSE across both `v2_gains` (`E4`) and `v3_gains` (`E5, E7`). Consequently, `Hegel` closed `E4_valid_comb`, `E6_separable_revisit`, and `E7_coupled_revisit` (`12/12` episodes) at `0` LLM turns with `100%` private `strict_pass`.
2. **Episode-Level Forensics of the `4` Alpha Confirmation Worlds (`Viscoelastic Prony + 4-Channel Cubic Load-Cell Bridge`)**:
   - **`alpha_m99_c01` (`5/7` `strict_pass` matching `Monolithic` `5/7`, `2/7` vs `1/7` 0-LLM closures, `-243,947` `v2` / `-13.42%` win vs `Monolithic`)**: Autonomous decomposition succeeded on `E1_down` (`attempt 2`, `upstream_holdout_rmse = 0.00011588`), `E4_valid_comb` and `E7_coupled_revisit` both closed at `0` LLM turns (`strict_pass = True`), and `SeparabilityWithdrawalReceipt` `swdr_9dbe611c8cc839270fd88efd` cleanly separated `D2_coupled`.
   - **`alpha_m99_c02` (`5/7` vs `Monolithic` `7/7`, `+241,390` `v2`) — Mode A1: Deferred `E2_down` Decomposition Timing + `0.001267` Public-to-Private Cubic Extrapolation Margin Drift on `E7`**:
     - *(i) Unmodularized `E1_down` solver*: `E1_down` passed monolithically (`14` turns), and two-stage factorization triggered on `E2_down` (`attempt 2`, `upstream_holdout_rmse = 0.00014221`) over `(Stage S, E2_down)`. Because the earlier `E1_down` (`D1`) monolithic solution was not retroactively projected into the modular graph, `E4_valid_comb` (`U1 × D1`) could not close at `0` LLM turns and spent `13` synthesis turns (`strict_pass = False`).
     - *(ii) Extreme-load cubic inversion drift on `E7_coupled_revisit`*: After `swdr_e9f24c87c360b342738b13ae` (`E5`) and a `0`-LLM `strict_pass` on `E6_separable_revisit`, `Hegel` selected the exact target pair `(bind_alpha_m99_c02_up_b01, bind_alpha_m99_c02_dn_b02)` at `0` LLM turns on `E7_coupled_revisit` (`v3_gains`) and **passed the public 3-layer gate** (`public strain_mae = 0.028451`, `rate_mae = 0.017701`, `tau_rel_err = 0.013759`). On private verification, it passed `seed = 952712` (`strain_mae = 0.027367 <= 0.070`) and recovered exact constitutive parameters on `seed = 952711` (`tau_rel = 0.001655`, `theta_err = 0.002274`), but missed the `0.070` strain threshold by **`0.001267`** (`strain_mae = 0.071267`) due to cubic Newton-inversion / outlier-trimming sensitivity on high-amplitude test loads under `v3_gains`.
   - **`alpha_m99_c03` (`5/7` vs `Monolithic` `7/7`, `-938,884` `v2` vs `Plain`, `+173,853` `v2` vs `Monolithic`) — Mode A2: Single-Episode Synthesis Miss on `E2_down` Cascading into `E6_separable_revisit`**:
     - `E1_down` decomposition (`rmse = 0.00015501`), `E3_up`, `E4_valid_comb` (`0`-LLM `strict_pass`), `E5_coupled_break`, and `E7_coupled_revisit` (`0`-LLM `strict_pass`) all succeeded (`5/5`). However, `E2_down` (`U0 × D2_sep`) exhausted `13` turns without passing its public gate, leaving no verified `d2_sep` module in the graph (`separability_withdrawal_receipts = []` at `E5`) and forcing `E6_separable_revisit` (which has no pilot hints) into a cold `13`-turn synthesis (`strict_pass = False`).
   - **`alpha_m99_c04` (`7/7` `100%` `strict_pass` matching `Monolithic` `7/7`, `+417,571` `v2` vs `Monolithic`) — Mode A3: Chirality/Gain-Specialized `u1` Re-Synthesis on `E4` & `E7` + Unsigned `up_stat_02` Guard Collision**:
     - `Hegel` achieved `7/7` private `strict_pass` (`E7 max_strain_mae = 0.006808`) and issued `swdr_f2a244850cd1febd6fb10f3b`, closing `E6_separable_revisit` at `0` LLM turns. However, the `u1` synthesized on `E3_up` over-specialized to the `E3` cyclic interaction chirality (`next-channel` vs `previous-channel` quadratic cross-talk) and `cal_frobenius_rms`, failing the direct `56/24` upstream probe under `v2_gains` on `E4_valid_comb` (`10` repair turns) and under `v3_gains` on `E7_coupled_revisit` (`10` repair turns). Moreover, upstream guard induction on `step 7` (`accepted = False`) could not separate `up_b01` and `up_b03` because `up_stat_02_off_diagonal_corr_max` takes an **unsigned** `max |corr|` that is invariant to off-diagonal sign/chirality flips.

### 8.2 Exact Combinatorial Amortization Law (`|U| · |D|` vs `|U| + |D| - 1 + m_c`) & Why `M9.9` Is a `1`-Episode Razor's-Edge Stress Test

Let $|U|$ denote the number of upstream transducer mechanisms, $|D|$ the number of downstream constitutive mechanisms, and $m_c$ the number of non-separable coupled exception cells in $U \times D$:

$$N_{\text{mono\_in\_stream}} = |U| \cdot |D| - 1, \qquad N_{\text{hegel\_in\_stream}} = (|U| - 1) + (|D| - 1) + m_c, \qquad N_{\text{sep\_unseen}} = (|U| - 1)(|D| - 1) - m_c.$$

| Product Grid $(|U| \times |D|,\; m_c)$ | In-Stream Monolithic Syntheses | In-Stream Modular Syntheses (`+ 1` Decomp) | Zero-LLM Unseen Separable Cells $N_{\text{sep\_unseen}}$ | Expected Net `Scope-1` Amortization Margin (`@ ~238k v2/ep`, `-45.9k decomp`) |
|---|---:|---:|---:|---:|
| **`M9.9` Minimal Withdrawal Grid `(2 × 3, m_c = 1)`** | `5` (`E1..E5`) | `4` (`E1, E2, E3, E5`) + `1` decomp | **`1` cell (`E4_valid_comb`)** | **`~ -192k v2/cluster`** (`1`-episode margin; realized `-201.7k` on Beta, flipped by `>= 1` extra Alpha repair) |
| **Medium Grid `(3 × 3, m_c = 1)`** | `8` | `5` + `1` decomp | **`3` cells** | **`~ -668k v2/cluster`** (`3`-episode margin; absorbs `1–2` boundary repairs with net savings intact) |
| **Larger Grid `(4 × 4, m_c = 2)`** | `15` | `7` + `1` decomp | **`7` cells** | **`~ -1,620k v2/cluster`** (`7`-episode quadratic scaling + `O(|U|+|D|)` vs `O(|U|·|D|)` coverage CPU) |

In `M9.9`, `Monolithic` spent `1,905,926` `v2` across `8` clusters (`238,240.8` `v2`/cluster) solving `E4_valid_comb`, while `Hegel` paid `367,190` `v2` (`45,898.8` `v2`/cluster) upfront for live autonomous two-stage decomposition. Because $N_{\text{sep\_unseen}} = 1$, `M9.9` intentionally tested `SeparabilityWithdrawalReceipt` at the **minimum possible horizon**, leaving a net theoretical buffer of `< 1` synthesis episode (`~192k` `v2`/cluster).

### 8.3 Structured Decision Matrix Handed Off to `GPT-6` Audit (`D1–D4`)

| Decision ID | Core Architectural / Experimental Question | Option A | Option B | Key Trade-Off & Equal-Tool Constraint |
|---|---|---|---|---|
| **`D1` (`Alpha` Upstream Extrapolation & Chirality Drift)** | How should autonomous upstream modules prevent `v2/v3` gain-extrapolation drift (`c02/E7` `0.0713` vs `0.070`) and unsigned `up_stat_02` chirality collisions (`c04/E4,E7`) without domain hardcoding? | **Norm-Stratified Public Extrapolation Gate (`0` Private Peek)**: Fit upstream probe on inner `70%` of calibration row `L_inf` norms and qualify on outer `30%` extreme-amplitude rows across both modular arms. | **Signed/Directional Generic Interface Spectrum**: Augment generic `up_stat` with signed off-diagonal / skew-symmetric correlation moments (`min_offdiag_corr`, `max_offdiag_corr`, log-condition ratio) identically in `Plain` and `Hegel`. | `D1-A` hardens autonomous module synthesis itself (`u1` must dynamically estimate chirality/scale per item); `D1-B` lets Kleene guards split sub-variants when `u1` is sign-locked (`D1-A + D1-B` are compatible). |
| **`D2` (`M10.0` Primary Horizon & Topology Scope)** | Should the next milestone first establish multi-cell `Scope-Total` net wins against `Monolithic` on `>= 3 × 3` grids, or jump directly to `>= 3`-stage variable-depth DAG discovery? | **`3 × 3` / `4 × 4` Equal-Tool Break-Even Curve ($K^*$) First**: Measure empirical break-even step $K^*$ under `2`-stage unseeded decomposition + selective withdrawal, requiring simultaneous wins over `Plain` AND `Monolithic`. | **Unseeded Variable-Depth (`k in {1,2,3}`) Open DAG Discovery (`U -> M -> D`)**: Remove fixed 2-slot `(upstream, downstream)` compiler signature on a `2 × 2 × 2` (`8`-cell) world combining depth discovery with multi-cell amortization. | `D2-A` isolates combinatorial amortization cleanly; `D2-B` advances structural autonomy (`R02` long-term north star) while naturally providing `N_sep_unseen >= 3`. |
| **`D3` (Retroactive Projection of Pre-Decomp Episodes)** | When decomposition triggers on `E2_down` (`c02`) instead of `E1_down`, what happens to the unfactored `E1_down` (`D1`) monolithic solver? | **Symmetric Retroactive Residual Projection**: Immediately test/extract `d1` against the newly verified `u0` on `E1_down`'s public witness (`100%` metered in `Scope-1` across both `Plain` and `Hegel`). | **Strict Forward-Only Modularization**: Leave pre-decomposition episodes monolithic unless revisited later in the stream. | `D3-A` eliminates the `c02` `E4_valid_comb` cold-synthesis penalty symmetrically across both modular arms without host privilege. |
| **`D4` (Reverse-Order Withdrawal & Missed-Branch Recovery)** | How should the stream test `SeparabilityWithdrawalReceipt` when a separable branch (`D2_sep`) misses on `E2_down` (`c03`) or arrives *after* `D2_coupled`? | **Bidirectional Order Randomization (`E2=coupled -> E5=sep` in half of clusters)**: Explicitly test whether `SeparabilityWithdrawalReceipt` fires regardless of whether `D2_sep` or `D2_coupled` is encountered first. | **Keep Fixed Canonical Event Order (`sep -> coupled`) with Bounded Re-Qualification**: Preserve current stream ordering and evaluate single-budget pass rates strictly. | `D4-A` prevents order-dependent blind spots and tests symmetric falsification (`(U0, D2_coupled)` failing downstream gate after `(U1, D2_coupled)` succeeded). |
