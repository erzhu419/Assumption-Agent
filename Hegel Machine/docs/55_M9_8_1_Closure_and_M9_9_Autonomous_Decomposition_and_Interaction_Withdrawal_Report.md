# Hegel M9.8.1 Audit Closure & M9.9 Equal-Tool Online Module Formation and Counterexample-Scoped Interaction Withdrawal Report (`Calibrated in M9.9.1`)

> [!IMPORTANT]
> **M9.9.1 Independent Audit Calibration & Errata Cross-Reference (`Q01–Q14`)**: All original sealed `M9.9` scores, costs, and receipts remain `100%` immutable. This report has been deterministically calibrated by `scripts/generate_m9_9_report_from_sealed_json.py` to incorporate every provenance, statistical, wording, and forensic root-cause correction from `Hegel_M9_9_Audit` (`Q01–Q14` + original `R01–R16` ID crosswalk). See companion errata `docs/55b_M9_9_1_Audit_Closure_Forensic_Reverification_and_Errata.md` for the `30`-probe Docker cross-diagonal back-verification (`0/30` passed, `30/30` scientific/domain rejected, `0/30` schema rejected), `c02/c04` forensic container reproductions, and provenance ledgers.

- **Date**: `2026-11-28`
- **Declared Pre-Confirmation Freeze Commit (`7624c43bf`, `Q05`)**: `7624c43bf72d283e82442504eff35ff6ee02b40d` (`2,700` `.py` files, reconstructed commit tree SHA-256 `915f469dc3f50a05c84e5065f1b03e99cdd80cf661bf4826dab16ff8e98af2c0`; raw mirrored call envelope `3,718.765s` from `2026-11-27T19:30:25.746185+00:00` to `2026-11-27T20:32:24.511243+00:00` under execution run ID `m99_confirmation_20261127T193020Z`). *Provenance note (`Q05`)*: Original paid execution did not persist a separate start-time manifest; `m9_9_continual_summary.json` recorded post-freeze finalizer run `m99_confirmation_20261127T203824Z` (`219.26s`, post-fix tree SHA `be7190fbfab8df60c5da023c3133f59fa0f71c6dde95ebc967367de3fd9d40bc`). `git diff` confirms `0` modifications to the paid online loop, world generators, or official verifier (`execution_manifest.json` separation added in `M9.9.1`).
- **M9.8.1 Closure Commit**: `b959102fa9c284efebb8921ce39229c4b23f45b1`
- **Shared Verification Kernel**: `hegel_shared_math_handoff_kernel_v3_m961_three_layer` | **Cost Policy**: `hegel_cost_policy_v2_m961_call_uuid_conserved`
- **Air-Gapped Sandbox Digest**: `sha256:bb111245520085c97361fff57b2e141911dd86a0e63c5c49761bc1ab692d910a`
- **Live Model & Concurrency (`R16`)**: `gpt-6.2-sol` (`reasoning_effort = medium`, `configured_max_codex_concurrency = 32`, observed `peak_active_codex_concurrency = 24`)
- **Usage Accounting & Unknown `x` Sensitivity (`Q06`)**: `partial_unknown_usage` (`1,750/1,751` calls usage-complete; `0` unknown attempts on `Stage S`, `Hegel`, and `Monolithic`; `1` unknown timeout attempt `x >= 0` on `Flat_PlainModuleRegistry_Online` `alpha_m99_c04/E1_down` `call_uuid=0391d702b0c547a488608847b5dff224`: `Hegel - Plain = -2,535,683 - x`, with exact two-sided sign-flip `p = 0.0625` for `0 <= x <= 17,652`, `p = 0.0546875` for `17,652 < x <= 53,450`, and **`p < 0.05` iff `x > 53,450`**).

---

## 1. Executive Summary & Scoped Attribution Boundaries (`Q01–Q14` & Original `R01–R16`)

1. **`M9.8.1` Evidence-Contract Hardening & Executed Attribution Controls (`0` Paid LLM Calls)**:
   - Hardened `SolverEvaluationWitness` (`v2` schema: `execution_status`, `evidence_origin`, nullable `metric_value` with zero `999.0` sentinels, `probe_id`, `setup_source_sha256`, `canonical_numeric_sha256`, `dependency_bundle_sha256`, `verification_trust_state`), `EmpiricalCoverageReceipt.verify_integrity()` (`R06`), and `_hydrate_from_graph()` (`R07`, with idempotent legacy `save -> reload` round-trip finalized in `M9.9.1` `Q08`).
   - Executed `M9.8` two-plain-tables replay (`56/56` setup, pickle, and canonical numeric SHA parity across both container `Python 3.10.12 / NumPy 2.2.6` and host `Python 3.12 / NumPy 2.3.5`; `32/56` public gate accepted $\to$ `32/32 = 100.0%` zero-LLM private `strict_pass`), `16/16` direct `u0` upstream Docker probes (`max RMSE = 0.00019837 <= 0.03`), and `64` `(R0, U_iso)` prefix-locked / fixed-policy controls (`16/16` history-aware vs `0/16` masked vs `8/16` fixed `D0`/`D1`).

2. **`M9.9` Pre-Registered `n = 8` Unexposed Confirmation Campaign (`176` Live Episodes, `100%` Equal Tool Flags)**:
   - **Zero Host-Seeded Module Source Code Under a Prescribed Two-Stage Interface (`Q09`)**: `Stage S` synthesizes an unfactored monolithic `solve_batch` (`8/8` pass). Within the host-prescribed two-stage `(calibrate_upstream, solve_downstream)` interface and parent parameter topology (open unknown-depth DAG discovery is NOT claimed here), modular arms invoke `propose_and_verify_m99_autonomous_two_stage_factorization` (`100%` metered in `Scope-1`), succeeding on `E1_down` in **`7/8` `Hegel` and `6/8` `Plain`** clusters (**`8/8` and `7/8` cumulative** across the full stream after `E2_down`). *Audit note (`Q01`)*: Original `M9.9` decomposition executed the `2` diagonal public-gate probes while passing `30` unexecuted off-diagonal constraints to `d0d1_gind` and writing `8` unexecuted `d0@E` projection witnesses; `M9.9.1` upgrades the verifier to a full `2x2` container check and executes all `30` historical cross-diagonal probes in Docker (`docs/55b_*`).
   - **Result A — `Hegel_AutonomousInteractionGraph_Online` vs Fixed-Separable `Flat_PlainModuleRegistry_Online` (`Q06, Q10`)**:
     - **Zero-LLM Passed Closures**: **`18/56` (`Hegel`) vs `9/56` (`PlainModuleRegistry`)** (`5` cluster wins, `3` ties, `0` losses, exact two-sided sign-flip `p = 0.0625`), driven by `7/8` traceable `SeparabilityWithdrawalReceipt`s (`12/16` vs `3/16` zero-LLM passes on `(E6, E7)`).
     - **Compute Cost & Statistical Boundary**: Known Scope-1 `proxy_v2 = 11,050,627` vs `13,586,310 + x` (**`-18.66%`** known ratio-of-sums, equal-cluster mean `-17.30%`, Scope-Total `-16.96%`, `mean Δv2 = -316,960.38 - x/8`, `95%` bootstrap CI `[-576158.88, -72127.96]`, **exact two-sided sign-flip `p = 0.0625` at `x = 0`** — directional `6/8` cluster cost reduction that is marginal at `alpha = 0.05` unless `x > 53,450`; `strict_pass` `50/56` vs `49/56`, `p = 1.0`).
     - **Scoped Frozen `step_05` Routing Ablation (`48` Runs, `Q10`)**: Given the frozen `step_05` module prefix, `hegel_selective_cross_factor_graph = 12/16` (`75.0%`) vs `dn_obs`-only selectors `factorized_fixed_D2_separable = 7/16` (`43.75%`) and `factorized_fixed_D2_coupled = 6/16` (`37.5%`). This isolates the **selective condition-expansion algorithm** (`dn_obs -> up_obs + dn_obs` upon counterexample) relative to fixed `dn_obs`-only routing; a plain Python dict/list given the same expanded `up_obs + dn_obs` features reproduces `16/16` selected branches and candidate SHAs (`03_m99_provenance_x_sensitivity_and_plain_dict_replay.json`).
   - **Result B — Short-Horizon Comparison Against `Flat_MonolithicConditionalTable_Online` (`Q06, Q13`)**:
     - On **Beta (`n = 4` PDE/KKT clusters)**, `Hegel` achieves `28/28` `strict_pass`, `12/28` vs `8/28` zero-LLM closures, and `4/4` cluster cost reductions (`mean Δv2 = -201,744.75`, `beta_sign_flip_p = 0.125`).
     - Across all `8` clusters (`1` unseen separable cell `E4_valid_comb`), total cost is indistinguishable from `Monolithic` (`11,050,627` vs `11,268,739` = `-1.94%` Scope-1 / `-1.73%` Scope-Total, `p = 0.765625`), while `Hegel` had **`4` fewer strict passes on Alpha (`50/56 = 89.29%` vs `54/56 = 96.43%`, `p = 0.5` — `p > 0.05` does NOT establish quality non-inferiority)**. Summed non-LLM agent container probe **elapsed wall-clock** is `1.421×` lower overall (`1,271,833.0 ms` vs `895,174.8 ms`), with the empirical coverage sub-item elapsed wall-ms `3.320×` lower (`689,259.6 ms` vs `207,604.6 ms`).

---

## 2. Primary Three-Arm Confirmation Ledger (`n = 8` Unexposed Clusters, `176` Episodes)

| Arm (`100%` Equal Tool Flags) | `strict_pass` | `0`-LLM Passed Closures | `Scope-1` Calls (`solve` / `decomp` / `guard`) | `Scope-1 proxy_v1` | `Scope-1 proxy_v2` | `Scope-Total proxy_v2` (`Stage S + Scope-1`) | `Hegel` vs Row (`Scope-1` / `Total`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicConditionalTable_Online` | `54/56` | `15/56` | `538` (`499` / `0` / `39`) | `11,045,607` | `11,268,739` | `12,637,624` | `-1.94%` / `-1.73%` |
| `Flat_PlainModuleRegistry_Online` *(fixed-separable `dn_obs`)* | `49/56` | `9/56` | `632` (`593` / `19` / `20`) | `13,044,562 + x_1` | `13,586,310 + x` | `14,955,195 + x` | `-18.66%` / `-16.96%` (`x >= 0`) |
| **`Hegel_AutonomousInteractionGraph_Online`** | **`50/56`** | **`18/56`** | **`505`** (`458` / `14` / `33`) | **`10,510,878`** | **`11,050,627`** | **`12,419,512`** | reference |

- **Shared `Stage S` Source Ledger (`8/8` Pass, `0` Unknown Attempts)**: `76` calls, `proxy_v1 = 1,319,886`, `proxy_v2 = 1,368,885` (`1/3`-allocated: `Hegel` vs `Plain` `-18.08%`, `Hegel` vs `Monolithic` `-1.86%`).
- **Separate Paired Exact Sign-Flip Tests (`n = 8` Independent Clusters; Minimum Two-Sided Resolution `2/256 = 0.0078125`, Distinct from Statistical Power)**:
  - `Hegel - Flat_PlainModuleRegistry_Online`: `mean Δv2 = -316,960.38 - x/8` (`95%` bootstrap CI `[-576158.88, -72127.96]`), **cost `p = 0.0625` at `x = 0`** (`p < 0.05` iff `x > 53,450`), **zero-LLM closure `p = 0.0625`**, **quality `strict_pass` `p = 1.0`**.
  - `Hegel - Flat_MonolithicConditionalTable_Online`: `mean Δv2 = -27,264.00` (`95%` bootstrap CI `[-191565.48, 158164.51]`), **cost `p = 0.765625`**, **zero-LLM closure `p = 0.453125`**, **quality `strict_pass` `p = 0.5`** (`50/56` vs `54/56`).

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

## 5. Non-LLM Container Probe Elapsed Wall-Clock Ledger (`R11 / Q13` — Summed Container Elapsed `ms`, Not CPU Time)

| Arm | Pre-LLM Gate (`ms`) | Post-LLM Gate (`ms`) | Parent Localization (`ms`) | Decomp Verification (`ms`) | Empirical Coverage (`ms`) | **Total Non-LLM Agent Probes (`ms`)** | Official Private Verifier (`ms`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicConditionalTable_Online` | `343,727.5` | `238,770.6` | `0.0` | `0.0` | `689,334.9` | **`1,271,833.0`** | `281,695.4` |
| `Flat_PlainModuleRegistry_Online` | `343,242.0` | `269,890.7` | `23,124.6` | `99,936.1` | `397,631.6` | **`1,133,825.1`** | `278,044.5` |
| `Hegel_AutonomousInteractionGraph_Online` | `342,415.0` | `215,443.8` | `19,457.8` | `110,218.1` | `207,640.0` | **`895,174.8`** | `279,372.0` |

---

## 6. Full Campaign R&D Ledger Including Unarchived Cancelled Dev Diagnostic Run (`Q12`)

1. **Unarchived Pre-Freeze Cancelled Dev Diagnostic Run (`dev` seeds `850100/860100` ONLY — `0` Confirmation Seeds Touched)**: Stopped at step 3 after diagnosing singleton-witness 1-clause ambient-noise guard splits (fixed symmetrically via `min_clause_margin = 0.035`). Raw logs were not archived before cancellation; its cost is recorded explicitly as **`unverifiable_cancelled_run = C_cancelled > 0`** (never filled as `0`).
2. **Sealed `n = 2` Pre-Freeze Dev Calibration (`docs/evidence/m9_9_dev_live_calibration_20261128`, `1,398` files, raw call envelope `2,259.938s`)**: `392` calls, `8,409,059` physical `v2` (`0` unknown attempts, `peak_concurrency = 6`).
3. **Sealed `n = 8` Unexposed Confirmation Campaign (`docs/evidence/m9_9_autonomous_decomposition_and_interaction_withdrawal_20261128`, `6,226` files, raw call envelope `3,718.765s`)**: `1751` calls, `37,274,561 + x` physical `v2` (`peak_concurrency = 24` under `cap = 32`).
4. **Total Campaign R&D Accounting**: Sealed two-package total = **`2,143` calls / `45,683,620 + x` physical `v2`**; full R&D total including the unarchived cancelled dev diagnostic run = **`45,683,620 + x + C_cancelled` physical `v2`**.

---

## 7. Original `Hegel_M9_8_Audit` (`R01–R16`) Exact-ID Crosswalk (`Q14`)

| Original `M9.8` ID & Topic | Verified `M9.8.1` / `M9.9` / `M9.9.1` Evidence & Scope | Status |
|---|---|---|
| `R01` (Graph vs two plain tables attribution) | `56/56` `M9.8` two-table replay (`32/32` 0-LLM pass) + `M9.9` `Flat_PlainModuleRegistry_Online` + `16/16` plain-dict `up+dn` replay (`03_m99_provenance...json`): isolates selective `dn -> up+dn` expansion policy from substrate. | **`backverified_and_claim_narrowed`** |
| `R02` (Host-seeded `u0/d0`, parent sets & repair prompts) | `u0/d0/d1` source code synthesized & metered online (`2x2` verified in `M9.9.1`); two-stage interface & parent topology remain host-prescribed (open DAG deferred). | **`partially_changed_claim_narrowed`** |
| `R03` (`L1` / auto-packaging flag & repair task parity) | `enable_auto_skill_packaging=False`, `allow_l1_primitives=False` across all arms; identical modular repair templates across `Plain` and `Hegel`. | **`technically_fixed`** |
| `R04` (`C1/C2` unexecuted `r2>=0.96` logged as `0.0` RMSE) | Fixed in `M9.8.1` + `16/16` direct Docker probes (`max RMSE = 0.00019837`); `M9.9.1` also closes `Q01` (`2x2` decomp + `30` Docker probes) and `Q08` (`M9.8` `compatibility_inference` isolation). | **`technically_fixed_plus_m991_q01_q08`** |
| `R05` (Infra/timeout misclassified as scientific negative) | Main & localization timeouts/rc125 fail closed to `unknown_infra` (`0` scientific witnesses); `M9.9.1` logs decomposition non-triviality timeout `probe_id`/`wall_ms`. | **`technically_fixed`** |
| `R06` (`EmpiricalCoverageReceipt` `ecov_invalid` bypass) | Enforces 29-char `ecov_` SHA-256 and count/list invariants. | **`technically_fixed`** |
| `R07` (`_hydrate_from_graph` missing digest & legacy trust) | Rejects missing hashes on current schemas; `M9.9.1` (`Q08`) makes legacy downgrade `load -> migrate -> save -> reload` idempotent (`0` authority). | **`technically_fixed`** |
| `R08` (`HistoryMasked` future step-4 & fixed-`D2` flaw) | Sealed `64`-run prefix-locked (`0/16`) & fixed `D0`/`D1` (`8/16`, `8/16`) vs history-aware (`16/16`) post-hoc control. | **`backverified_executed`** |
| `R09` (Cost `p`-value conflated with quality `p`-value) | Cost, zero-LLM closure, and `strict_pass` `p`-values reported in separate rows/columns across `docs/54*` and `docs/55*`. | **`claim_narrowed_and_reformatted`** |
| `R10` (Last-entry observable precursor overstated as long latent memory) | Narrowed explicitly in `docs/54*`: `archive_apparatus_log[-1]` is an observable precursor state, not long-horizon latent POMDP inference. | **`claim_narrowed`** |
| `R11` (`8` `M9.8` localization probes untimed; elapsed vs CPU) | Reference rerun (`4,967.12 ms`) distinguished from unrecovered original time; `M9.9` 6-category ledger explicitly labeled **elapsed container wall-ms** (`1.421x` total, `3.320x` coverage sub-item). | **`technically_fixed_and_reworded`** |
| `R12` (Input hash vs source hash & cross-env reproducibility) | `56/56` setup, pickle, and `canonical_numeric_sha256` verified across container (`3.10/2.2.6`) and independent `3.12/2.3.5` host. | **`technically_fixed`** |
| `R13` (`M9.7` Beta `p = 0.50` typo & `16/16` vs `15/16` twin text) | Corrected in `docs/53b*` (`Beta p = 1.0`) and `docs/54*` (`15/16` `Table` twin 0-LLM). | **`technically_fixed`** |
| `R14` (`R1` error-code diagnosis overstated as numerical fix) | Explicitly retired (`diagnosis_and_error_code_closed_numerical_candidate_retired`; original `41/42` score retained). | **`retired_by_design`** |
| `R15` (Legacy KKT `L1` certificate interface / live adoption) | Explicitly retained as an open boundary (`retained_open_gap` in `docs/53b*` and `docs/54*`; orthogonal to `M9.9/M10.0`). | **`retained_open_gap`** |
| `R16` (`cap=32` vs `peak`, sign-flip resolution vs power, push wording) | `cap` and `peak` reported side-by-side; `2/256` resolution separated from power; broken `54b` hub link fixed in `M9.9.1`. | **`technically_fixed`** |

---

## 8. Calibrated Forensic Attribution of `M9.9` Alpha Trajectories & Conditional Amortization Equation (`Q02–Q04, Q11, Q13`)

### 8.1 Corrected Episode-Level Forensics Across the `8` Confirmation Worlds (`E4 = v3_gains`, `E5 = v2_gains`, `E7 = v3_gains`)

1. **Beta (`beta_m99_c01..c04`, `28/28` Pass, `4/4` Cost Reductions vs Both Baselines)**: `u1` dynamically re-fits the coupled cubic sensor rosette per campaign (`< 5e-5` holdout RMSE across `v3_gains` on `E4, E7` and `v2_gains` on `E5`), closing `12/12` `(E4, E6, E7)` episodes at `0` LLM turns.
2. **Alpha (`alpha_m99_c01..c04`) — Audit-Verified Root Causes (`Hegel_M9_9_Audit` `Q02–Q04`)**:
   - **`alpha_m99_c01` (`5/7` Pass Matching `Monolithic` `5/7`, `-243,947` `v2`)**: Decomposed on `E1_down`, closed `E4` and `E7` at `0` LLM turns, and issued `swdr_9dbe611c8cc839270fd88efd` on `E5`.
   - **`alpha_m99_c02` (`5/7` vs `Monolithic` `7/7`, `+241,390` `v2`) — Corrected Forensics (`Q02` & `Q04`)**:
     - *(i) Missing executable fallback / retroactive projection after `E2_down` decomposition (`Q04`)*: `E1_down` (`D1`) solved monolithically (`14` turns), and decomposition triggered on `E2_down` (`step 2`), leaving `E1_down` unprojected and non-callable on `E4_valid_comb` (`v3_gains`, `13` turns, failed).
     - *(ii) Downstream test-only reference standardization violating the public `train+test` pooled contract on `E7_coupled_revisit` (`Q02` — Correcting Prior Upstream-Newton Attribution)*: On `E7` (`v3_gains`), `bind_alpha_m99_c02_up_b01` inverted test loads to **`~1.4e-5` RMSE** (substituting exact oracle upstream loads still yields `strain_mae = 0.071270 > 0.070`!). The true root cause was `bind_alpha_m99_c02_dn_b02`'s `_chamber_features(test)` normalizing `test` on `test['ref_coupons']` alone with unweighted `NaN` mean instead of the documented pooled `vstack([train, test])` per-lab standardization and `(lab_id, stratum)` inverse-count weighted imputation (`BLIND_ALPHA_INSTRUCTION_TEXT`). Fixing **only** that public preprocessing contract drops `seed = 952711` `strain_mae` from **`0.071267` to `0.004857`** (`<= 0.070`!), while `tau`/kernel approximation (`0.004857 -> 0.001425`) was secondary (`public strain_mae = 0.028451`).
   - **`alpha_m99_c03` (`5/7` vs `Monolithic` `7/7`, `+173,853` `v2`) — Single-Episode Miss on `E2_down` (`Q04`)**: `E2_down` (`U0 × D2_sep`) failed its `13`-turn synthesis budget, leaving `d2_sep` missing (`separability_withdrawal_receipts = []` at `E5`) and propagating into pilot-free `E6_separable_revisit`.
   - **`alpha_m99_c04` (`7/7` Pass Matching `Monolithic` `7/7`, `+417,571` `v2`) — Corrected Forensics (`Q03`)**:
     - On **`E4_valid_comb` (`v3_gains`)**, `up_b01` (learned at `E3_up` under `v2_gains`) had hardcoded `E3`'s fitted polynomial coefficients instead of re-fitting `calibration_load_cells` per item, genuinely requiring `10` synthesis turns to learn **`bind_alpha_m99_c04_up_b02`**.
     - On **`E7_coupled_revisit` (`v3_gains`)**, however, **`bind_alpha_m99_c04_up_b02` was ALREADY in the `step_06` library** and passes the deterministic `56/24` (`0..55` / `>=56` outlier-filtered) upstream probe (`RMSE = 0.00007655 <= 0.03`), the 3-layer public gate (`accepted = True`), AND the official verifier (`strict_pass = True`) when composed with `dn_b03` (`02_m99_q03_c04_e7_existing_candidate_container_verification.json`). It was never tried on `E7` only because its 2-sample guard required `up_stat_03_cal_frobenius_rms > 1.289054` (`E7` had `1.274262`), routing back to `up_b01` (`RMSE = 0.5763`) and spending `10` avoidable repair turns!

### 8.2 Conditional Unseen-Cell Count (`|(U × D) \ (R ∪ X)|`) & Why Break-Even $K^*$ Must Be Measured Empirically (`Q11`)

Let $U$ and $D$ denote upstream and downstream operator sets covered by single-factor exposure episodes $R$, and let $X \subseteq U \times D$ denote coupled interaction exceptions. Under **ideal 1-shot module synthesis and zero false guard exclusions**, the number of unseen separable cells is:

$$N_{\text{sep,unseen}} = |(U \times D) \setminus (R \cup X)|$$

(note: $(|U|-1)(|D|-1) - |X|$ holds only when $X \cap R = \emptyset$). In real online continual streams, actual cumulative cost difference $\Delta C(K) = C_{\text{Hegel}}(K) - C_{\text{baseline}}(K)$ depends on decomposition cost $C_{\text{decomp}}$, guard/fallback probe costs, and non-zero repair rates when guards or contracts drift (as seen on `alpha_m99_c02..c04`). Therefore, `(3 × 3)` and `(4 × 4)` rows are **unverified conditional projections** in `M9.9`, and empirical quality-constrained break-even $K^*$ is evaluated prospectively in **`M10.0`** (`docs/56_*`).
