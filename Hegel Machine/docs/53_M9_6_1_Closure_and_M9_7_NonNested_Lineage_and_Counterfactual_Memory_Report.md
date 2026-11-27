# Hegel Machine M9.6.1 Closure & M9.7 Non-Nested Multi-Conflict Lineage and Counterfactual Memory Report

- **Confirmation Frozen Commit**: `b9df86b10153ea57d0e2215acffb61b256cc03db` (`source_tree_py_sha256`: `5098f9cef2374acdedc7b03198259d10f96e039f19e41f45e5eab68b984f085f`, start == end, `tracked_tree_dirty_at_end = False`)
- **Sealed Evidence Directories**:
  - `M9.6.1` Closure & Errata (`0` paid LLM calls): `docs/evidence/m9_6_1_semantic_closure_and_paid_pool_attribution_20261127/` (`docs/52b_M9_6_1_Audit_Errata_Semantic_Closure_and_Paid_Pool_Attribution.md`)
  - `M9.7` Official Confirmation Run (`run3`, `90` live episodes): `docs/evidence/m9_7_non_nested_counterfactual_lineage_run3_20261127/` (`1,996` entries verified in `SHA256SUMS.txt`)
  - `M9.7` Aborted Pre-Confirmation Runs (100% forensically preserved & counted in `Scope 3+`):
    - `docs/evidence/m9_7_aborted_run1_openblas_setup_timeout_20261127/` (`510` calls, `proxy_v2 = 12,241,271`)
    - `docs/evidence/m9_7_aborted_run2_beta_self_guard_witness_kind_20261127/` (`521` calls, `proxy_v2 = 12,103,475`)
- **Live Provider & Concurrency**: Official ChatGPT subscription Codex CLI (`gpt-6.2-sol`, `reasoning_effort=medium`), configured semaphore cap = `32`, observed peak concurrent active CLI calls = `12` across `12` independent parallel continual streams (`wall_clock_sec = 3228.55`).

---

## 1. Summary of M9.6.1 Semantic & Accounting Closure (`N01–N11`, `0` Paid LLM Calls)

1. **`N01` Call-UUID Hierarchical Cost Conservation**: Eliminated legacy `token_cost_proxy` (`v1`) leakage into per-episode `v2` tables (`COST_POLICY_VERSION = "hegel_cost_policy_v2_m961_call_uuid_conserved"`). Reconciled `M9.6` `beta_m96_w01` delta = `-7,308` (`7/8` worlds higher, $n=8$ $p = 0.015625$; $n=4$ disjoint source pairs $p = 0.125$; L2 compilation share of excess = `89.55%`).
2. **`N02` & `N03` Unified Three-Layer Scientific Contract (`hegel_shared_math_handoff_kernel_v3_m961_three_layer`)**: Separated Layer-1 structure, Layer-2 `CalibrationWitness` estimated-coordinate self-consistency (`<= 2.01e-14` on `alpha_m96_w03 / M`, correcting the `initial_memory_states` misdiagnosis while keeping official `M9.6` scores at `63/64` vs `63/64`), and Layer-3 truth accuracy; re-verified all `34` `M9.4` episodes (`64/68` private instances, `100%` of `beta_ep03` dual-channel episodes passing after fixing `terminal_field_B` dispatch).
3. **`N04, N07, N09–N11`**: Enforced exact `executable_kind` SHA matching + content-recomputed `QualificationReceipt` + full `parent_parameter_versions` vector (`N04`), `ImmutableBranchLineageController` + `SolverEvaluationWitness` + `EmpiricalCoverageReceipt` (`N07`), reconstructed `M9.6` `Full` `T` skill inspection funnel (`7/8` episodes described new source skills, `0/8` executed/retained, `N09/N10`), and completed sequential paid-pool component attribution (`N11`).

---

## 2. M9.7 Experimental Protocol (`n = 6` Independent Clusters, `90` Live Episodes)

- **Primary Inferential Unit (`n = 6` Completely Independent Source-World Clusters)**: `3` Alpha rheology clusters (`alpha_m97_c01..c03`) + `3` Beta 2D reaction-transport clusters (`beta_m97_c01..c03`), each with its own independent `Stage S` (`C0`) solve (`0` cross-cluster shared compilation).
- **7-Step Continual Stream per Cluster (`42` continual episodes/arm = `84` stream + `6` Stage S = `90` live episodes)**:
  `C1 (first conflict) -> R0 (return to C0) -> C2 (second non-nested conflict) -> R1 (post-C2 return to C1, sparse spot calibration) -> U_iso (strict present-matched counterfactual twin of R0) -> G_perm (selective permuted parent p0 -> v3 + p1 EMI spikes + wired aux_telemetry p2 jump) -> M_couple (v3 + sparse spot + C1 counterfactual memory)`.
- **Strict Present-Matched Counterfactual Pair `(R0, U_iso)` (`N06`)**: Generated from identical `present_seed` so all present inputs, thresholds, hold vectors, calibration sweeps, and terminal `t = 0-` snapshots match to `0.0`, differing solely in `past_stress_cmd[:, :-1]` / `past_vortex_circulation_trace[:-1]`.
- **Unhandicapped Baseline (`Flat_EqualInfo_Multiversion`) vs Treatment (`Hegel_DemandGated_Graph`)**: Both receive identical `0444` prefix calibration history, identical `/tmp/hegel_continual_archive/` verified solvers (with full freedom to write unified/piecewise dispatchers), identical `max_pre_llm_probes = 1`, and validated `2/3–1/3` calibration updates. `Hegel_DemandGated_Graph` adds `ImmutableBranchLineageController` (`b00, b01, b02`), `EmpiricalCoverageReceipt` checks before guard induction, neutral `obs_01..obs_08` Kleene `all_of` guards, containerized `cal_node_p0..p3` perturbation sensitivity, and **M8 demand-gated consolidation (`N08`)**.

---

## 3. Primary Results & Four-Scope Conserved Cost Ledger (`hegel_cost_policy_v2_m961_call_uuid_conserved`)

| Metric / Scope | `Flat_EqualInfo_Multiversion` | `Hegel_DemandGated_Graph` | Delta (`Hegel - Flat`) | Statistical Inference (`n = 6` Clusters) |
| :--- | :---: | :---: | :---: | :---: |
| **Continual Strict Passes (`42` eps)** | **`41/42` (`97.62%`)** | **`41/42` (`97.62%`)** | `+0` | two-sided sign-flip $p = 1.0$ |
| **Zero-LLM Episodes / Passed Zero-Turn Closures** | `18/42` / **`18/42`** | `30/42` / **`29/42`** | **`+11`** | two-sided sign-flip $p = 0.25$ |
| **Total Continual LLM Turns (`mean/ep`)** | `307` (`7.31`) | **`148` (`3.52`)** | **`-159` (`-51.79%`)** | mean `-26.50`/cluster, $p = 0.15625$ |
| **Pre-LLM Container Probe Wall Time (ms)** | `311,517.44` | `319,824.16` | `+8,306.72` | explicit non-LLM Docker/CPU cost |
| **Scope 1a: In-Stream Solve `proxy_v2`** | `7,139,945` (`307` calls) | `3,300,381` (`148` calls) | `-3,839,564` | — |
| **Scope 1b: In-Stream L2 Compile `proxy_v2` (`N08`)** | `0` (`0` calls) | **`0` (`0` calls)** | `0` | demand gate deferred `100%` negative-EV compiles |
| **Scope 1c: Guard Induction `proxy_v2`** | `0` (`0` calls) | `98,267` (`6` calls) | `+98,267` | `2.89%` of `Hegel` stream spend |
| **Scope 1 Total In-Stream `proxy_v1`** | `6,646,651` | **`3,168,064`** | **`-3,478,587` (`-52.34%`)** | `4/6` clusters lower, $p = 0.15625$ |
| **Scope 1 Total In-Stream `proxy_v2`** | `7,139,945` | **`3,398,648`** | **`-3,741,297` (`-52.40%`)** | 95% CI `[-1105388.5, -138044.17]`, $p = 0.15625$ |
| **Scope 2 Standalone (`+ Stage S` = `1,556,485`)** | `8,696,430` | **`4,955,133`** | **`-3,741,297` (`-43.02%`)** | includes all `6` shared `Stage S` solves |

- **Scope 3 (`run3` Complete Physical Spend)**: `533` logical calls (`534` provider attempts, `533/533` mirrored raw files, `1` unknown-usage retry attempt -> `cost_accounting_status = "partial_unknown_usage"`), `proxy_v1 = 11,236,849`, `proxy_v2 = 12,095,078`.
- **Scope 3+ (Grand Campaign Physical Spend Across `aborted_run1 + aborted_run2 + run3`)**: `1,564` logical calls, `total_tokens = 42,135,882`, `proxy_v1 = 33,743,850`, `proxy_v2 = 36,439,824`.

---

## 4. Per-Event & Per-Cluster Breakdown: Why Alpha & Beta Bifurcate Cleanly

### 4.1 Per-Event Continual Stream Table (`6` Independent Clusters per Event)

| Event | `Flat` Pass | `Hegel` Pass | `Flat` `0`-Turn Pass | `Hegel` `0`-Turn Pass | `Flat` Mean Turns | `Hegel` Mean Turns | `Flat` `proxy_v2` | `Hegel` `proxy_v2` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `C1` | `5/6` | `6/6` | `0/6` | `0/6` | `13.17` | `13.17` | `1,769,704` | `1,904,622` |
| `R0` | `6/6` | `6/6` | `3/6` | `6/6` | `6.67` | `0.00` | `956,450` | `0` |
| `C2` | `6/6` | `6/6` | `0/6` | `0/6` | `11.83` | `11.50` | `1,526,569` | `1,494,026` |
| `R1` | `6/6` | `5/6` | `3/6` | `5/6` | `6.17` | `0.00` | `933,229` | `0` |
| `U_iso` | `6/6` | `6/6` | `6/6` | `6/6` | `0.00` | `0.00` | `0` | `0` |
| `G_perm` | `6/6` | `6/6` | `3/6` | `6/6` | `6.83` | `0.00` | `996,547` | `0` |
| `M_couple` | `6/6` | `6/6` | `3/6` | `6/6` | `6.50` | `0.00` | `957,446` | `0` |

### 4.2 Per-Independent-Cluster Table (`n = 6`) & Family Stratification

| Independent Cluster | `Flat` Pass | `Hegel` Pass | `Flat` `0`-Turn Pass | `Hegel` `0`-Turn Pass | `Flat` Turns | `Hegel` Turns (`Delta`) | `Flat` `proxy_v2` | `Hegel` `proxy_v2` (`Delta`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `alpha_m97_c01` (`cluster_01_alpha`) | `7/7` | `7/7` | `1/7` | `5/7` | `82` | `26` (`-56`) | `2,011,969` | `651,217` (`-1,360,752`) |
| `alpha_m97_c02` (`cluster_02_alpha`) | `6/7` | `6/7` | `1/7` | `4/7` | `80` | `25` (`-55`) | `1,960,534` | `630,048` (`-1,330,486`) |
| `alpha_m97_c03` (`cluster_03_alpha`) | `7/7` | `7/7` | `1/7` | `5/7` | `77` | `27` (`-50`) | `1,787,101` | `675,933` (`-1,111,168`) |
| `beta_m97_c01` (`cluster_04_beta`) | `7/7` | `7/7` | `5/7` | `5/7` | `21` | `24` (`+3`) | `419,502` | `497,639` (`+78,137`) |
| `beta_m97_c02` (`cluster_05_beta`) | `7/7` | `7/7` | `5/7` | `5/7` | `26` | `22` (`-4`) | `567,229` | `439,304` (`-127,925`) |
| `beta_m97_c03` (`cluster_06_beta`) | `7/7` | `7/7` | `5/7` | `5/7` | `21` | `24` (`+3`) | `393,610` | `504,507` (`+110,897`) |
| **Alpha Subtotal (`3` Statistical Inverse Clusters)** | **`20/21`** (`['alpha_m97_c02:C1']`) | **`20/21`** (`['alpha_m97_c02:R1']`) | `3/21` | **`14/21`** | `239` | **`78` (`-161`)** | `5,759,604` | **`1,957,198` (`-66.02%`)** |
| **Beta Subtotal (`3` Forward PDE + KKT Clusters)** | **`21/21`** (`[]`) | **`21/21`** (`[]`) | `15/21` | **`15/21`** | `68` | **`70` (`+2`)** | `1,380,341` | **`1,441,450` (`+4.43%`)** |

### 4.3 Mechanistic Explanation of the Alpha vs Beta Bifurcation (`§5` Empirical Coverage in Action)

1. **Family Alpha (Nonlinear Statistical Inverse Estimation Under Cauchy Glitches)**:
   - Fitting `C0` (`phi_0(t)=t`, 7-parameter IRLS), `C1` (`phi_1(t)=log(1+t)` + history work-damage `D_cryst`, 9-parameter IRLS), and `C2` (`phi_2(t)=t/(1+0.85t)` + slip, 9-parameter IRLS) requires incompatible statistical estimators. `gpt-6.2-sol` synthesized specialized single-regime solvers on both arms.
   - `EmpiricalCoverageReceipt` refuted subsumption **`9/9` times** on Alpha (`3` at `C1`, `6` at `C2`), triggering **`6/6` accepted multi-branch guard & permuted-parent inductions** (`100%` selected exact physical parents `["cal_node_p0", "cal_node_p1"]` inside isolated containers while rejecting wired `aux_chamber_telemetry` nuisance `cal_node_p2` and `cal_node_p3`, and induced clean Kleene `all_of` partitions over `obs_01` and `obs_03`).
   - Result on Alpha: `Flat` suffered continual thrashing (`18/21` episodes required LLM synthesis, `239` turns, `5,759,604 v2`), whereas `Hegel_DemandGated_Graph` retained `b00, b01, b02` immutably, closed `14/15` post-conflict episodes in `0` turns (`78` turns total), and reduced `Scope 1` `v2` cost by **`-66.02%`** (`1,957,198` vs `5,759,604`).
2. **Family Beta (Known-Coefficient Forward PDE + Box-Simplex KKT)**:
   - Because Beta's forward 1D/2D operators have analytically specified boundary laws and `/tmp/hegel_continual_archive/` exposes prior verified solvers equally to both arms, an unhandicapped `gpt-6.2-sol` spontaneously synthesized unified piecewise solvers (`C0 u C1` at `Step 1`, `C0 u C1 u C2` at `Step 3`) on **both** arms.
   - Crucially, `Hegel_DemandGated_Graph`'s **`EmpiricalCoverageReceipt` (`0` LLM calls) verified `supported_on_witnesses` `9/9` times** on Beta, automatically bypassing all `6` guard-induction LLM calls (`0` guard calls on Beta!) and bypassing all post-episode L2 compilations (`N08`), resulting in an exact statistical tie with `Flat` (`15/15` vs `15/15` zero-turn closures; `+4.43%` `v2` difference).

---

## 5. Explicit Scope Caveats & Limitations

1. **Pooled `n = 6` Sign-Flip Significance Caveat**: While `Hegel_DemandGated_Graph` reduces pooled `Scope 1` `v2` cost by **`-52.40%`** (`95%` cluster bootstrap CI `[-1105388.5, -138044.17]` excluding `0`) and wins `3/3` Alpha clusters by `> 1.1M v2` each, the exact two-sided sign-flip test across all `6` mixed-family clusters yields **`p = 0.15625` (`4/6` clusters lower, NOT `< 0.05`)** because the `3` Beta clusters tie near zero (`+78k, -128k, +111k`). Future confirmation targeting statistical-estimation regimes specifically requires `n >= 6` statistical-inverse clusters for standalone non-parametric sign-flip significance at $\alpha = 0.05$.
2. **Finite-Witness Coverage vs Symbolic Equivalence**: `EmpiricalCoverageReceipt` (`status = 'supported_on_witnesses'`) proves execution compatibility on frozen positive witness tasks under isolated container probes (`0` LLM calls, explicit CPU wall-ms recorded), **not** universal mathematical equivalence over all possible inputs; older branches (`b00, b01`) are therefore always retained immutably rather than deleted.
3. **Zero-LLM != Zero-Compute & Proxy Accounting**: `v1` and `v2` are deterministic token-cost proxies (`CostPolicy`), not provider invoice dollars; `1` retry attempt out of `534` in `run3` had unknown token usage (`partial_unknown_usage`); and zero-LLM episodes still incur `~6.0s` isolated Docker container verification probes (`total_pre_llm_probe_wall_ms` reported in Section 3).
4. **Cross-Task L1/L2 Skill Funnel Scope**: With synchronous post-episode L2 compilation demand-gated off (`0` compiles, saving `~1.45M v2` vs `M9.6`), `M9.7` isolates `MechanismExecutableBinding` + `ImmutableBranchLineageController` + `M8` multi-parent calibration rather than `hegel_wormhole_skills` L1 AST splicing (`0` retained L1 modules in final deliverables).
