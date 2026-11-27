# Hegel Machine M9.7.1 Audit Closure & M9.8 Blind Mismatch Discovery + Modular Factored System Report

- **Report Schema**: `hegel_m9_7_1_and_m9_8_sealed_report_v2_m981_reconciled`
- **M9.8 Run ID**: `m98_run_20261127T142012Z`
- **Frozen Pre-Confirmation Code Commit**: `2963e5d65f68d91510746e21c60c0985ec333358` (`working_tree_dirty_at_start = False`, `source_tree_sha256 = b1574bd7dac87b13a8f069d3bf361b4c1bcaeeaa3a43423004816d2913b66547`)
- **Sandbox Image Digest**: `sha256:bb111245520085c97361fff57b2e141911dd86a0e63c5c49761bc1ab692d910a`
- **Model & Concurrency**: `gpt-6.2-sol` (`reasoning_effort = medium`, `configured_cap = 32`, `observed_peak_active = 24`)
- **Evidence Provenance**: `executed` (`usage_completeness = complete`, `unknown_usage_attempts = 0`)
- **Shared Verification Kernel & Cost Policy**: `hegel_shared_math_handoff_kernel_v3_m961_three_layer` / `hegel_cost_policy_v2_m961_call_uuid_conserved`
- **Post-Audit Representation & Contract Companion (`M9.8.1`)**: See `docs/54b_M9_8_1_Audit_Errata_Evidence_Contract_and_TwoTable_Attribution.md`

---

## 1. Executive Summary & Honest Attribution Scope (`R01–R16` Reconciled)

1. **Phase 1 (`M9.7.1`, `0` Paid LLM Calls — Sealed in `docs/evidence/m9_7_1_audit_closure_and_paid_pool_attribution_20261127/`)**:
   - **Real Containerized Paid-Pool Replays (`A01`, `A02`)**:
     - `01_n11_executed_m95_paid_pool_replay.json`: Executed all `48` `M9.5` reuse episodes across all 6 strategies in air-gapped `hegel-clean-scipy:m7v1` (`1_Unvalidated = 28/48`, `2_ValidatedWingOLS = 40/48`, **`3_OldestFirst_Probes1 = 32/48`**, `4_Multiversion_Probes2 = 48/48`, **`5_SharedConditionalTable_Probes1 = 48/48`**, `6_Hegel_MechanismGraph_Probes1 = 48/48`).
     - `02_m97_executed_flat_conditional_table_attribution.json`: Executed prefix-locked `Flat_ConditionalTable` (`0` `PersistentMechanismGraph`) across all `42` `M9.7` continual episodes (`42/42` route match, `30/42` public gate accepted, `29/42` zero-LLM private `strict_pass`), proving that on 1-stage uniform-parent `M9.7` streams, conditional multi-version retention + Kleene guards matched `Hegel_DemandGated_Graph`.
   - **Errata & Contract Hardening (`docs/53b_*` + `M9.8.1` Closure)**:
     - Enforced `PARENT_VERSION_SET_MISMATCH` (`A07`), unidirectional coverage + canonical `ecov_` receipt digest (`A08 + R06`), 4-tier failure classification + localization infra isolation (`A09 + R05`), fail-closed lineage checkpoint digest & foreign-key hydration (`A10 + R07`), unified calibration (`A11`), retired defective `M9.7` `R1` candidate after diagnostic error-code split (`A06 + R14`), and marked legacy `M9.4/M9.6` `L1` KKT dual-helper adoption as `retained_open_gap` (`A14 + R15`).

2. **Phase 2 (`M9.8`, Pre-Registered Unexposed `n = 8` Confirmation Clusters — `176` Live Containerized Episodes)**:
   - **What `M9.8` Proves vs What Requires `M9.9` (`R01–R03, R10` Scope Boundary)**:
     - `M9.8` tests a **host-supplied two-stage decomposition (`U -> D`) + localized upstream synthesis harness + cross-factor composition system** (`Hegel_FactoredMechanismGraph`) against a **monolithic online conditional table** (`Flat_ConditionalTable_Online`) and `Flat_LatestFirst` (note: in `M9.8`, `Hegel` also had `allow_l1_primitives=True` and specialized `upstream_transducer.py` repair prompts on `C_up`; paid-prefix two-table replay in `docs/54b` confirms `56/56` routing/code equivalence with two plain tables once that `U/D` factorization and prefix programs exist, motivating `M9.9`'s strict equal-tool autonomous decomposition & non-separable interaction benchmark).
   - **Head-to-Head Online Outcomes (`56` Continual Episodes / Arm)**:
     - **`Flat_LatestFirst`**: `44/56` private `strict_pass`, `4/56` zero-LLM closures, Scope-1 `v2 = 13,078,368` (`618` calls).
     - **`Flat_ConditionalTable_Online`**: `50/56` private `strict_pass`, **`15/56` zero-LLM closures** (**`15/16`** zero-LLM passes on `(R0, U_iso)`, while achieving **`0/16` zero-LLM reuse** on unseen cross-factor `M_comb1 + M_comb2` and falling back to full LLM re-synthesis for **`16/16` final `strict_pass`** at `204` calls / `4,260,223 v2`, accounting for `85.87%` of net `v2` savings!), Scope-1 `v2 = 11,139,318` (`531` calls: `504` solve + `27` guard).
     - **`Hegel_FactoredMechanismGraph`**: `56/56` private `strict_pass`, **`32/56` zero-LLM closures** (`16/16` on `R0 + U_iso` and **`16/16` zero-LLM composition** on `M_comb1 + M_comb2`!), Scope-1 `v2 = 6,177,898` (`310` calls: `286` solve + `24` guard).
   - **Cluster-Level Cost vs Quality Inference (`n = 8` Clusters; Minimum Two-Sided Sign-Flip Resolution = `2 / 2^8 = 0.0078125`, `R09, R16`)**:
     - **Scope-1 Continual `v2` Cost (`Hegel` vs `Flat_ConditionalTable_Online`)**: Ratio-of-sums **`-44.54%`** (equal-cluster mean relative change `-43.36%`; Scope-Total including full standalone `Stage S` per system: `7,371,303` vs `12,332,723` = **`-40.23%`**), mean cluster delta `-620,178`, 95% bootstrap CI `[-775347.3, -480665.62]`, exact two-sided sign-flip **`p = 0.0078125`** (`Alpha n=4: -48.74%, p = 0.125`; `Beta n=4: -38.32%, p = 0.125`).
     - **Scope-1 Continual `v2` Cost (`Hegel` vs `Flat_LatestFirst`)**: Ratio-of-sums **`-52.76%`** (Scope-Total with `Stage S`: **`-48.35%`**), exact two-sided sign-flip **`p = 0.0078125`**.
     - **Cluster-Level Final Quality (`strict_pass`, Reported Separately from Cost per `R09`)**: `Hegel - Flat_ConditionalTable_Online` cluster pass diffs `[1, 2, 1, 2, 0, 0, 0, 0]` (`56/56` vs `50/56`, `+10.71` pp, all 6 gains in Alpha while Beta tied `28/28`), exact two-sided sign-flip **`p = 0.125`** (observational quality advantage, NOT `< 0.05` at `n = 8`; vs `Flat_LatestFirst` `[2, 2, 2, 2, 1, 1, 1, 1]`, `p = 0.0078125`).
   - **Byte-Identical `(R0, U_iso)` Twins & Post-Hoc vs Prefix-Locked Mask Controls (`A03, R08`)**:
     - Base `public_batch.pkl` bytes and base `instruction_text` match `100%` on `(R0, U_iso)` across all `8` clusters (with baseline domain equations present in shared base instructions, while shifted `D1/D2` equations/labels are unrevealed; live prompts differ by archive history and `Hegel`'s `C_up` repair harness).
     - The `0/16` table in §5 below is a **post-hoc step-04 library `D2`-fallback log-deletion sensitivity check**; strict causal prefix-locked (`step-1` / `step-4`) and fixed `D0`/`D1` current-only controls (`8/16 = 50.0%` matching the disjoint-acceptance-ball ceiling) are executed and sealed in `docs/54b`.

---

## 2. Scope-1 & Scope-Total Cost, Call-Role & Non-LLM Container Wall-Clock Ledger (`A05, R09, R11`)

> **Shared `Stage S` Initialization Cost**: `73` calls, `v1 = 1,115,434`, `v2 = 1,193,405`, LLM `wall_ms = 1,946,861`.
> **Full `M9.8` Grand Campaign Cost Ledger (`4` Sealed Runs, `100%` `call_uuid` Mirrored)**:
> - `m9_8_aborted_run1_stage_s_contract_underspec_20261127` (aborted in `Stage S` on retired seeds `911000..924000`): `201` calls, `v2 = 5,318,353`.
> - `m9_8_dev_live_calibration_pass1_20261127` + `m9_8_dev_live_calibration_20261127` (`DEV_M98_WORLD_SPECS` `810100, 820100`): `643` calls, `v2 = 14,095,339`.
> - `m9_8_blind_mismatch_factored_graph_20261127` (frozen confirmation run on unexposed `931000..944000`): `1532` calls, `v2 = 31,588,989`.
> - **Grand Total Across All `M9.8` Runs**: **`2376` calls, `v2 = 51,002,681`**.

| Metric (`56` Continual Episodes / Arm across `n = 8` Clusters) | `Flat_LatestFirst` | `Flat_ConditionalTable_Online` | `Hegel_FactoredMechanismGraph` | `Hegel` vs `Flat_Table` | `Hegel` vs `Flat_Latest` |
|---|---:|---:|---:|---:|---:|
| Private `strict_pass` (Cluster Sign-Flip `p`) | `44/56` | `50/56` | `56/56` | `p = 0.125` | `p = 0.0078125` |
| Zero-LLM (`0`-turn) Passed Closures | `4/56` | `15/56` | `32/56` | `+`17 episodes | `+`28 episodes |
| Solve LLM Calls (`stream_solve`) | `618` | `504` | `286` | `-43.25%` | `-53.72%` |
| Guard Induction LLM Calls (`autonomous_guard_induction`) | `0` | `27` | `24` | `-11.11%` | — |
| **Total Scope-1 Continual LLM Calls (`solve + guard`)** | **`618`** | **`531`** | **`310`** | **`-41.62%`** | **`-49.84%`** |
| Scope-1 `proxy_v1_u_025c_1o` | `12,452,860` | `10,534,868` | `5,802,048` | `-44.93%` | `-53.41%` |
| **Scope-1 `proxy_v2_u_010c_4o` (In-Stream Primary)** | **`13,078,368`** | **`11,139,318`** | **`6,177,898`** | **`-44.54%`** | **`-52.76%`** |
| **Scope-Total `proxy_v2` (Standalone incl. `Stage S`)** | **`14,271,773`** | **`12,332,723`** | **`7,371,303`** | **`-40.23%`** | **`-48.35%`** |
| — Family Alpha (`n = 4` clusters) Scope-1 `proxy_v2` | `7,091,417` | `6,646,428` | `3,406,721` | `-48.74%` | `-51.96%` |
| — Family Beta (`n = 4` clusters) Scope-1 `proxy_v2` | `5,986,951` | `4,492,890` | `2,771,177` | `-38.32%` | `-53.71%` |
| Provider LLM `wall_ms` (`solve + guard`) | `16,766,155` | `13,638,091` | `7,713,468` | `-43.44%` | `-53.99%` |
| Non-LLM Pre-LLM Gate Probe `wall_ms` (`56` probes) | `366,919` | `374,914` | `377,177` | — | — |
| Non-LLM Empirical Coverage Probe `wall_ms` | `0` | `365,519` | `147,523` | `-59.64%` | — |
| Non-LLM `C_up` Localization Container Probes (`R11`) | `0` | `0` | `8` (`unmetered_in_m98`; metered in `docs/54b`) | — | — |

---

## 3. Event-Type Breakdown (`8` Clusters per Event — Isolating Cross-Factor Reuse on `M_comb1` & `M_comb2`)

| Event (`true_regime`) | `Flat_LatestFirst` (`pass`, `0-LLM`, `v2`) | `Flat_ConditionalTable_Online` (`pass`, `0-LLM`, `v2`) | `Hegel_FactoredMechanismGraph` (`pass`, `0-LLM`, `v2`) |
|---|---|---|---|
| `C1_down` | `7/8` (`0` 0-LLM, `2,370,988`) | `8/8` (`0` 0-LLM, `2,151,790`) | `8/8` (`0` 0-LLM, `1,974,374`) |
| `R0` | `8/8` (`4` 0-LLM, `626,727`) | `8/8` (`8` 0-LLM, `0`) | `8/8` (`8` 0-LLM, `0`) |
| `C2_down` | `7/8` (`0` 0-LLM, `2,180,956`) | `7/8` (`0` 0-LLM, `2,250,209`) | `8/8` (`0` 0-LLM, `2,221,100`) |
| `C_up` | `6/8` (`0` 0-LLM, `2,388,555`) | `4/8` (`0` 0-LLM, `2,275,667`) | `8/8` (`0` 0-LLM, `1,982,424`) |
| `U_iso` | `0/8` (`0` 0-LLM, `1,454,261`) | `7/8` (`7` 0-LLM, `201,429`) | `8/8` (`8` 0-LLM, `0`) |
| `M_comb1` | `8/8` (`0` 0-LLM, `2,246,849`) | `8/8` (`0` 0-LLM, `2,260,045`) | `8/8` (`8` 0-LLM, `0`) |
| `M_comb2` | `8/8` (`0` 0-LLM, `1,810,032`) | `8/8` (`0` 0-LLM, `2,000,178`) | `8/8` (`8` 0-LLM, `0`) |

---

## 4. Per-Cluster Results Across All `n = 8` Independent Confirmation Clusters

| Cluster (`Family`) | `Flat_LatestFirst` | `Flat_ConditionalTable_Online` | `Hegel_FactoredMechanismGraph` | `Latest` `v2` | `Flat_Table` `v2` | `Hegel` `v2` | `Hegel` vs `Flat_Table` |
|---|---|---|---|---:|---:|---:|---:|
| `alpha_m98_v2_c01` (`alpha`) | `5/7` (`1/7` 0-LLM) | `6/7` (`2/7` 0-LLM) | `7/7` (`4/7` 0-LLM) | `1,528,392` | `1,547,464` | `728,960` | `-52.89%` |
| `alpha_m98_v2_c02` (`alpha`) | `5/7` (`1/7` 0-LLM) | `5/7` (`1/7` 0-LLM) | `7/7` (`4/7` 0-LLM) | `1,822,959` | `1,760,714` | `896,092` | `-49.11%` |
| `alpha_m98_v2_c03` (`alpha`) | `5/7` (`1/7` 0-LLM) | `6/7` (`2/7` 0-LLM) | `7/7` (`4/7` 0-LLM) | `1,898,976` | `1,478,392` | `906,103` | `-38.71%` |
| `alpha_m98_v2_c04` (`alpha`) | `5/7` (`1/7` 0-LLM) | `5/7` (`2/7` 0-LLM) | `7/7` (`4/7` 0-LLM) | `1,841,090` | `1,859,858` | `875,566` | `-52.92%` |
| `beta_m98_v2_c01` (`beta`) | `6/7` (`0/7` 0-LLM) | `7/7` (`2/7` 0-LLM) | `7/7` (`4/7` 0-LLM) | `1,504,510` | `1,125,968` | `731,678` | `-35.02%` |
| `beta_m98_v2_c02` (`beta`) | `6/7` (`0/7` 0-LLM) | `7/7` (`2/7` 0-LLM) | `7/7` (`4/7` 0-LLM) | `1,451,913` | `1,164,968` | `703,532` | `-39.61%` |
| `beta_m98_v2_c03` (`beta`) | `6/7` (`0/7` 0-LLM) | `7/7` (`2/7` 0-LLM) | `7/7` (`4/7` 0-LLM) | `1,555,025` | `1,103,210` | `652,758` | `-40.83%` |
| `beta_m98_v2_c04` (`beta`) | `6/7` (`0/7` 0-LLM) | `7/7` (`2/7` 0-LLM) | `7/7` (`4/7` 0-LLM) | `1,475,503` | `1,098,744` | `683,209` | `-37.82%` |

---

## 5. Post-Hoc Step-04 Library `D2`-Fallback Log-Deletion Sensitivity (`m9_8_history_masked_ablation.json`; Prefix-Locked & `D0/D1` `8/16` Controls in `docs/54b`)

| Cluster | Event | `SHA256(/holdout_task/public_batch.pkl)` | Post-Hoc Step-04 Fallback Branch | Masked Private `strict_pass` |
|---|---|---|---|---|
| `alpha_m98_v2_c01` | `R0` | `ae4a31e07207547a...` | `bind_alpha_m98_v2_c01_dn_b02` | `False` |
| `alpha_m98_v2_c01` | `U_iso` | `ae4a31e07207547a...` | `bind_alpha_m98_v2_c01_dn_b02` | `False` |
| `alpha_m98_v2_c02` | `R0` | `a442fb3bf9640e4e...` | `bind_alpha_m98_v2_c02_dn_b02` | `False` |
| `alpha_m98_v2_c02` | `U_iso` | `a442fb3bf9640e4e...` | `bind_alpha_m98_v2_c02_dn_b02` | `False` |
| `alpha_m98_v2_c03` | `R0` | `5934e05b785216ee...` | `bind_alpha_m98_v2_c03_dn_b02` | `False` |
| `alpha_m98_v2_c03` | `U_iso` | `5934e05b785216ee...` | `bind_alpha_m98_v2_c03_dn_b02` | `False` |
| `alpha_m98_v2_c04` | `R0` | `d46c5bf54b8418b3...` | `bind_alpha_m98_v2_c04_dn_b02` | `False` |
| `alpha_m98_v2_c04` | `U_iso` | `d46c5bf54b8418b3...` | `bind_alpha_m98_v2_c04_dn_b02` | `False` |
| `beta_m98_v2_c01` | `R0` | `6b21aad5b6fbf76c...` | `bind_beta_m98_v2_c01_dn_b02` | `False` |
| `beta_m98_v2_c01` | `U_iso` | `6b21aad5b6fbf76c...` | `bind_beta_m98_v2_c01_dn_b02` | `False` |
| `beta_m98_v2_c02` | `R0` | `c49246d5f780d292...` | `bind_beta_m98_v2_c02_dn_b02` | `False` |
| `beta_m98_v2_c02` | `U_iso` | `c49246d5f780d292...` | `bind_beta_m98_v2_c02_dn_b02` | `False` |
| `beta_m98_v2_c03` | `R0` | `bf91a8bf8337c91a...` | `bind_beta_m98_v2_c03_dn_b02` | `False` |
| `beta_m98_v2_c03` | `U_iso` | `bf91a8bf8337c91a...` | `bind_beta_m98_v2_c03_dn_b02` | `False` |
| `beta_m98_v2_c04` | `R0` | `6c2059b36a6e8b80...` | `bind_beta_m98_v2_c04_dn_b02` | `False` |
| `beta_m98_v2_c04` | `U_iso` | `6c2059b36a6e8b80...` | `bind_beta_m98_v2_c04_dn_b02` | `False` |

---

## 6. Granular `A01–A17` Audit Status Matrix (`5`-Status Classification per `Hegel_M9_8_Audit`)

| Finding | Granular Status | Evidence & Scope Boundary |
|---|---|---|
| `A01`, `A17` | `verified_closed` | `48`-episode `N11` RPC execution (`01_n11_executed_m95_paid_pool_replay.json`) + `docs/52b` `64/68` heading correction. |
| `A02` | `scope_narrowed` | `M9.7` conditional table matched `30/42` pub, `29/42` priv; `M9.8` factored system beat monolithic `Flat_ConditionalTable_Online` (`-44.54%`), while `M9.8.1` two-table paid-prefix replay (`56/56`) narrows `M9.8` to modular `U/D` system gain rather than graph-vs-two-table storage gain. |
| `A03`, `A13` | `scope_narrowed` | Base `public_batch.pkl` & base `instruction_text` match on `(R0, U_iso)` (`<= 8/16` fixed current-only ceiling); baseline physics equations appear in shared instructions while shifted `D1/D2` equations do not; prefix log acts as observable precursor state; full prefix-locked & `D0/D1` controls sealed in `docs/54b`. |
| `A04`, `A05`, `A15` | `verified_closed` | `M9.7` reclassified as exposed-seed replication (`Beta p = 1.0` fixed in `docs/53b`); `M9.8` `n = 8` confirmation frozen at `2963e5d65f` with `0` unknown usage, cost `p = 0.0078125`, quality `p = 0.125`, and `8` localization probes documented (`docs/54b`). |
| `A06` | `diagnosed_and_retired` | `alpha_m97_c02 / R1` (`2b098388239b...`) diagnosed (`PARAMETER_ORDER`, `STRAIN_ERROR`, `RATE_ERROR`), historical `41/42` preserved, candidate retired (`docs/53b`). |
| `A07`, `A11`, `A12`, `A16` | `verified_closed` | `PARENT_VERSION_SET_MISMATCH` (`A07`), shared `2/3–1/3` affine estimator (`A11`), named `M9.7` consolidation rule (`A12`), and updated `historical_capabilities.json` (`A16`). |
| `A08`, `A09`, `A10` | `superseded_in_m9_8_1` | `M9.8.1` closes malformed `ecov_invalid` bypass (`R06`), missing-digest cold hydration & foreign-key checks (`R07`), `r2 >= 0.96` unexecuted witness labeling (`R04`), and localization `rc=125` / `TimeoutExpired` handling (`R05`) (`tests/test_m9_8_1_audit_closure_suite.py` + `docs/54b`). |
| `A14` | `retained_open_gap` | Standalone whole-program KKT solvers pass exact primal-dual checks, while legacy `M9.4/M9.6` `L1` compiled helper dual-dictionary adoption remains `retained_open_gap`. |
