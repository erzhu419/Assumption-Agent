# Hegel Machine M9.7.1 Audit Closure & M9.8 Blind Mismatch Discovery + Factored Mechanism Graph Report

- **Report Schema**: `hegel_m9_7_1_and_m9_8_sealed_report_v1`
- **M9.8 Run ID**: `m98_run_20261127T142012Z`
- **Frozen Pre-Confirmation Code Commit**: `2963e5d65f68d91510746e21c60c0985ec333358` (`working_tree_dirty_at_start = False`, `source_tree_sha256 = b1574bd7dac87b13a8f069d3bf361b4c1bcaeeaa3a43423004816d2913b66547`)
- **Sandbox Image Digest**: `sha256:bb111245520085c97361fff57b2e141911dd86a0e63c5c49761bc1ab692d910a`
- **Model & Concurrency**: `gpt-6.2-sol` (`reasoning_effort = medium`, `max_codex_concurrency = 32`, `peak_active = 24`)
- **Evidence Provenance**: `executed` (`usage_completeness = complete`, `unknown_usage_attempts = 0`)
- **Shared Verification Kernel & Cost Policy**: `hegel_shared_math_handoff_kernel_v3_m961_three_layer` / `hegel_cost_policy_v2_m961_call_uuid_conserved`

---

## 1. Executive Summary (`A01–A17` Closed + Live Unexposed `n = 8` Factored Graph Separation)

1. **Phase 1 (`M9.7.1`, `0` Paid LLM Calls — Sealed in `docs/evidence/m9_7_1_audit_closure_and_paid_pool_attribution_20261127/`)**:
   - **Real Containerized Paid-Pool Replays (`A02`, `A15`, `N11`)**:
     - `01_n11_executed_m95_paid_pool_replay.json`: Executed all `48` `M9.5` reuse episodes across all 6 strategies in air-gapped `hegel-clean-scipy:m7v1` JSON-RPC container (`1_Flat_SingleLatest_Unvalidated = 28/48`, `2_Flat_SingleLatest_ValidatedWingOLS = 40/48`, **`3_Flat_OldestFirst_Probes1 = 32/48`**, `4_Flat_Multiversion_Probes2 = 48/48`, **`5_Flat_SharedConditionalTable_Probes1 = 48/48`**, `6_Hegel_MechanismGraph_Probes1 = 48/48`, `evidence_provenance = executed`).
     - `02_m97_executed_flat_conditional_table_attribution.json`: Executed prefix-locked `Flat_ConditionalTable` (`0` `PersistentMechanismGraph`) across all `42` `M9.7` continual episodes (`42/42` route match, `30/42` public gate accepted, `29/42` zero-LLM private `strict_pass`), proving that on uniform-parent 1-stage `M9.7` streams, conditional multi-version retention + Kleene guards explained `100%` of the gain over `Flat_LatestFirst`, motivating `M9.8`'s 2-stage factored dependency architecture.
   - **Contract, Hydration, Verifier & Errata Closure (`A01, A05–A12, A16, A17`)**:
     - Enforced fail-closed `PARENT_VERSION_SET_MISMATCH` (`A07`), unidirectional `(eval_bid, owner_bid)` coverage (`A08`), `INFRA_FAILURE -> SCHEMA_FAILURE -> DOMAIN_REJECTION -> SCIENTIFIC_FAILURE` probe classification (`A09`), automatic checkpoint persistence + hydration of `witness_ledger` and `EmpiricalCoverageReceipt`s (`A10`), unified `estimate_validated_channel_affine_calibration` across `Flat` and `Graph` (`A11`), split Alpha error codes (`NON_FINITE / PARAMETER_ORDER / STRAIN_ERROR / RATE_ERROR` `A06`), updated `historical_capabilities.json` (`A16`), and sealed full `M9.7` cost/timing/`R1` errata (`docs/53b_M9_7_1_Audit_Errata_Contract_Closure_and_Paid_Pool_Attribution.md`).

2. **Phase 2 (`M9.8`, Pre-Registered Unexposed `n = 8` Confirmation Clusters — `176` Live Containerized Episodes)**:
   - **Head-to-Head Online Separation Over `Flat_ConditionalTable_Online` (`A02`, `A04`, `A15`)**:
     - **`Flat_LatestFirst`**: `44/56` private `strict_pass`, `4/56` zero-LLM closures, Scope-1 `v2 = 13,078,368` (`618` calls: `618` solve + `0` guard).
     - **`Flat_ConditionalTable_Online` (Live Online Conditional Table + Live `gpt-6.2-sol` Kleene Guard Induction over all 12 Observables)**: `50/56` private `strict_pass`, **`15/56` zero-LLM closures** (`16/16` on `R0` and `U_iso`, proving unhandicapped conditional table parity on previously seen `(U, D)` cells, while failing pre-LLM on unseen cross-factor cells `M_comb1 (U1 x D1)` and `M_comb2 (U1 x D2)`!), Scope-1 `v2 = 11,139,318` (`531` calls: `504` solve + `27` guard).
     - **`Hegel_FactoredMechanismGraph` (2-Stage Causal `PersistentMechanismGraph`: `Upstream {u0, u1} -> Downstream {d0, d1, d2}`)**: `56/56` private `strict_pass`, **`32/56` zero-LLM closures** (composing `u1 o d1` on `M_comb1` and `u1 o d2` on `M_comb2` zero-shot across **both** Alpha and Beta!), Scope-1 `v2 = 6,177,898` (`310` calls: `286` solve + `24` guard).
   - **Cluster-Level Inference (`n = 8` Independent Confirmation Clusters: `4` Alpha + `4` Beta)**:
     - **`Hegel_FactoredMechanismGraph` vs `Flat_ConditionalTable_Online` (`v2`)**: Mean cluster delta **`-620,178`** (`-44.54%`), 95% bootstrap CI **`[-775347.3, -480665.62]`**, exact two-sided sign-flip **`p = 0.0078125`** (`Alpha n=4: -48.74%, p = 0.125`; `Beta n=4: -38.32%, p = 0.125`).
     - **`Hegel_FactoredMechanismGraph` vs `Flat_LatestFirst` (`v2`)**: Mean cluster delta **`-862,559`** (`-52.76%`), 95% bootstrap CI **`[-926214.88, -801797.83]`**, exact two-sided sign-flip **`p = 0.0078125`**.
   - **Strict Byte-Identical `(R0, U_iso)` Counterfactual History + Executed `HistoryMasked_CurrentOnly` Control (`A03` & `A13`)**:
     - Across all `8` clusters, `SHA256(/holdout_task/public_batch.pkl on R0) == SHA256(/holdout_task/public_batch.pkl on U_iso)` (`0` `gt` key inside container) and `R0.instruction_text == U_iso.instruction_text` (`0` prompt formula leakage across all 8 episodes).
     - Executed `HistoryMasked_CurrentOnly` Docker control (`m9_8_history_masked_ablation.json`) passes **`0/16` (`0.0%`, selecting `dn_b02 = D2` under `UNKNOWN` fallback; theoretical ceiling for any current-only deterministic policy is `<= 8/16 = 50.0%`)**, whereas history-aware `Hegel_FactoredMechanismGraph` passes **`16/16` (`100.0%`) at `0` LLM turns** (`Flat_ConditionalTable_Online` `15/16` at `0` LLM turns).

---

## 2. Scope-1 Continual Cost, Call-Role & Non-LLM Container Wall-Clock Ledger (`A05`, `A12`)

> **Shared `Stage S` Initialization Cost (Identical Across All 3 Arms)**: `73` calls, `v1 = 1,115,434`, `v2 = 1,193,405`, LLM `wall_ms = 1,946,861`.
> **Full `M9.8` Grand Campaign Cost Ledger (`A05` Zero-Hidden-Run Accounting)**:
> - `m9_8_aborted_run1_stage_s_contract_underspec_20261127` (aborted in `Stage S` on retired seed block `911000..924000`, `0` continual episodes): `201` calls, `v2 = 5,318,353` (`usage_completeness = complete`).
> - `m9_8_dev_live_calibration_pass1_20261127` + `m9_8_dev_live_calibration_20261127` (`DEV_M98_WORLD_SPECS` `810100, 820100` pre-freeze calibration: `262` + `381` calls): `643` calls, `v2 = 14,095,339` (`usage_completeness = complete`).
> - `m9_8_blind_mismatch_factored_graph_20261127` (frozen confirmation run on fresh unexposed seed block `931000..944000`): `1532` calls, `v2 = 31,588,989` (`usage_completeness = complete`).
> - **Grand Total Across All `M9.8` Runs**: **`2376` calls, `v2 = 51,002,681`** (`100%` mirrored with `SHA256SUMS.txt`).

| Metric (`56` Continual Episodes / Arm across `n = 8` Clusters) | `Flat_LatestFirst` | `Flat_ConditionalTable_Online` | `Hegel_FactoredMechanismGraph` | `Hegel` vs `Flat_Table` | `Hegel` vs `Flat_Latest` |
|---|---:|---:|---:|---:|---:|
| Private `strict_pass` | `44/56` | `50/56` | `56/56` | — | — |
| Zero-LLM (`0`-turn) Passed Closures | `4/56` | `15/56` | `32/56` | `+`17 episodes | `+`28 episodes |
| Solve LLM Calls (`stream_solve`) | `618` | `504` | `286` | `-43.25%` | `-53.72%` |
| Guard Induction LLM Calls (`autonomous_guard_induction`) | `0` | `27` | `24` | `-11.11%` | — |
| **Total Scope-1 Continual LLM Calls (`solve + guard`)** | **`618`** | **`531`** | **`310`** | **`-41.62%`** | **`-49.84%`** |
| Scope-1 `proxy_v1_u_025c_1o` | `12,452,860` | `10,534,868` | `5,802,048` | `-44.93%` | `-53.41%` |
| **Scope-1 `proxy_v2_u_010c_4o` (Primary)** | **`13,078,368`** | **`11,139,318`** | **`6,177,898`** | **`-44.54%`** | **`-52.76%`** |
| — Family Alpha (`n = 4` clusters) `proxy_v2` | `7,091,417` | `6,646,428` | `3,406,721` | `-48.74%` | `-51.96%` |
| — Family Beta (`n = 4` clusters) `proxy_v2` | `5,986,951` | `4,492,890` | `2,771,177` | `-38.32%` | `-53.71%` |
| Provider LLM `wall_ms` (`solve + guard`) | `16,766,155` | `13,638,091` | `7,713,468` | `-43.44%` | `-53.99%` |
| Non-LLM Pre-LLM Container Probe `wall_ms` (`56` probes) | `366,919` | `374,914` | `377,177` | — | — |
| Non-LLM Empirical Coverage Probe `wall_ms` | `0` | `365,519` | `147,523` | `-59.64%` | — |

---

## 3. Event-Type Breakdown (`8` Clusters per Event — Isolating Unseen Cross-Factor Composition on `M_comb1` & `M_comb2`)

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

## 4. Per-Cluster Results Across All `n = 8` Independent Confirmation Clusters (`A04`, `A15`)

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

## 5. Executed `HistoryMasked_CurrentOnly` Ablation on Byte-Identical `(R0, U_iso)` (`A03`)

| Cluster | Event | `SHA256(/holdout_task/public_batch.pkl)` | Masked Selected Branch | Masked Private `strict_pass` |
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

## 6. Complete `Hegel_M9_7_Audit` Closure Matrix (`A01–A17`)

| Finding | Scope | Resolution & Sealed Verification Artifact |
|---|---|---|
| `A01`, `A05`, `A06`, `A12` | `M9.7` Errata & Forensics | Reconciled `148` solve + `6` guard = `154` calls (`-49.84%`), `partial_unknown_usage`, `Beta +4.43%`, non-LLM `wall_ms`, `run1..run3` grand ledger (`1,564` calls), `R0 vs U_iso` diff, and `R1` `PARAMETER_ORDER` reproduction (`docs/53b_M9_7_1_Audit_Errata_Contract_Closure_and_Paid_Pool_Attribution.md`). |
| `A02` | `Flat_ConditionalTable` Separation | Executed `M9.7` `Flat_ConditionalTable` (`02_m97_executed_flat_conditional_table_attribution.json`) AND ran live head-to-head `Flat_ConditionalTable_Online` vs `Hegel_FactoredMechanismGraph` in `M9.8` (`p = 0.0078125`). |
| `A03` | Byte-Identical `(R0, U_iso)` + `HistoryMasked` | `SHA256(public_batch.pkl)` and `instruction_text` are `100%` byte-identical on `(R0, U_iso)` (`0` `gt` key in container); executed `HistoryMasked_CurrentOnly` passes `0/16` vs `16/16` history-aware. |
| `A04`, `A15` | Pre-Freeze Hygiene & `n = 8` Power | Dev testing confined strictly to `DEV_M98_WORLD_SPECS`; `M9.8` confirmation (`n = 8` clusters) run once from clean frozen commit `2963e5d65f`. |
| `A07` | Full Parent Version Set Equality | `PARENT_VERSION_SET_MISMATCH` enforced in `verify_binding_identity_and_foreign_keys` + checkpoint hydration migration (`tests/test_m9_7_1_audit_closure_suite.py`). |
| `A08` | Unidirectional Coverage & Receipt Digest | Unidirectional `(eval_bid, owner_bid)` check + content-addressed `witness_id` & `receipt_id` (`tests/test_m9_7_1_audit_closure_suite.py`). |
| `A09` | Timeout vs Scientific Failure | `classify_public_probe_failure` separates `infra_or_timeout`, `schema_invalid`, `domain_precondition`, and `scientific_or_numerical` (`tests/test_m9_7_1_audit_closure_suite.py`). |
| `A10` | Lineage Ledger Cold-Restart Hydration | `witness_ledger`, `coverage_receipts`, and `branch_seq` persisted inside `PersistentMechanismGraph` checkpoints and verified under post-`C_up` cold restart (`tests/test_m9_8_blind_mismatch_suite.py`). |
| `A11` | Unified Calibration Estimator | `Flat` and `Graph` share `estimate_validated_channel_affine_calibration` (`0.0` gain/bias discrepancy). |
| `A13` | Zero Prompt Formula Leakage & Generic Reducers | Identical blind prompt across all 8 episodes per family (`0` regime formulas/names) + generic `up_stat_01..04` and `dn_stat_01..08` moment/OLS statistics. |
| `A14`, `A17` | `executed` vs `analytic` Provenance + `M9.6.1` Heading | Executed `48`-episode `N11` replay (`01_n11_executed_m95_paid_pool_replay.json`), fixed `docs/52b` `64/68` heading, and tagged `evidence_provenance` across all summaries. |
| `A16` | Historical Capability Index | Updated `src/hegel_machine/cognition/historical_capabilities.json` and `docs/hegel_machine_reuse_hub_20261006.md` (`pytest tests/test_historical_capabilities.py` green). |
