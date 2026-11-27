# M9.7.1 Audit Errata, Contract Hardening, Cold-Restart Closure & Executed Paid-Pool Attribution (`0` Paid LLM Calls)

- **Evidence Bundle**: `docs/evidence/m9_7_1_audit_closure_and_paid_pool_attribution_20261127/` (`SHA256SUMS.txt` verified; all historical `M9.4`–`M9.7` bundles 100% untouched)
- **Provenance Discipline (`A01`)**: Every reported table explicitly tags `evidence_provenance` (`executed`, `recomputed_from_raw`, `derived_expectation`). Zero unexecuted constants are presented as replay receipts.

## 1. M9.7 Statistical, Call-Count, Partial-Usage & Non-LLM Wall-Time Errata (`A04, A05, A12, A15, A17`)

1. **Study Classification & Grand Physical Ledger (`A04`)**:
   - Because `CONFIRMATION_M97_WORLD_SPECS` were exposed in aborted `run1` (OpenBLAS container thread contention) and aborted `run2` (Beta domain-exception witness classification) prior to commit `27b8cb97c`, `M9.7` `run3` is formally classified as a **post-repair frozen replicated mechanism experiment on exposed seeds** (`evidence_provenance = recomputed_from_raw`).
   - Grand physical ledger across `run1 + run2 + run3`: **`1,564` unique `call_uuid`s**, **`36,439,824` known `proxy_v2`** (`Alpha` savings replicated across all three runs: `-64.24%`, `-65.70%`, `-66.02%`).
2. **Call-Count & Partial-Unknown-Usage Qualification (`A04, A05`)**:
   - Continual `Hegel_DemandGated_Graph` LLM calls: **`148` solve calls + `6` guard-induction calls = `154` total continual calls** vs `Flat`'s **`307` calls** (`-51.79%` solve calls; **`-49.84%` total continual calls**).
   - Usage completeness: **`partial_unknown_usage`** (`1` timed-out attempt `1b028fe68e4745c692589f62c402f78b` on `alpha_m97_c01:C1`, `360,015.06 ms`). Exact Scope 1 relation: $C_{\text{Hegel}} - C_{\text{Flat}} = -3{,}741{,}297 + x$ ($x \ge 0$).
3. **Family-Stratified Results & Beta Clarification (`A05, A15`)**:
   - **Alpha (`n = 3`)**: `20/21` vs `20/21` `strict_pass`, `14/21` vs `3/21` zero-LLM passed closures, known `Scope 1 proxy_v2` `1,957,198` vs `5,759,604` (**`-66.02%`**, `3/3` negative, exact sign-flip `p = 0.25`; Alpha accounts for `80.67%` of `Flat` `v2`).
   - **Beta (`n = 3`)**: `21/21` vs `21/21` `strict_pass`, `15/21` vs `15/21` zero-LLM passed closures, `0` guard-induction calls, `0` in-stream L2 compile calls, known `Scope 1 proxy_v2` **`1,441,450` vs `1,380,341` (`+4.43%`, `+61,109`)** (exact two-sided sign-flip `p = 1.0`; replaces imprecise "exact tie / zero tax" phrasing).
   - **Pooled (`n = 6` clusters)**: mean cluster `v2` diff `-623,549.50`, mean per-cluster relative change **`-28.90%`**, exact sign-flip `p = 0.15625`.
4. **Complete Non-LLM Container Wall-Time Accounting (`A05`, `wall_ms` strictly separated from CPU-seconds)**:
   - `Flat_EqualInfo_Multiversion`: `42` pre-LLM container probes = **`311,517.44 ms`** (`0` coverage probes).
   - `Hegel_DemandGated_Graph`: `42` pre-LLM container probes (`319,824.16 ms`) + `18` `EmpiricalCoverageReceipt` container probes (`115,258.55 ms`) = **`435,082.71 ms`** (`+123,565.27 ms` extra container wall time, plus `24` parent-sensitivity container runs across `6` guard inductions).
5. **Scope Narrowing on `R0` vs `U_iso` (`A03, A13`), Consolidation Policy Naming (`A12`), & Legacy KKT `L1` Helper Status (`A14`)**:
   - In `M9.7`, `R0` and `U_iso` matched test covariates (`obs_02..08 == 0.0` diff), declared thresholds, calibration sweeps, and terminal snapshots, while `train` responses (`17.78` max diff), Beta `y_obs_raw` (`0.0325`), and prompt headers differed; `M9.7`'s consolidation gate acted effectively as `disable_eager_l2_when_canonical_binding_covers_contract`.
   - **`A14` (`retained_open_gap`)**: While standalone task solvers satisfy full KKT primal-dual checks, legacy `M9.4/M9.6` `L1` compiled helper dual-certificate adoption remains `retained_open_gap` (not claimed as invoked or closed here).
6. **Exact Numerical Forensics & Candidate Retirement on `alpha_m97_c02 / R1` (`A06`, Status = `diagnosis_and_error_code_closed_numerical_candidate_retired`)**:
   - Host reproduction of SHA-pinned `2b098388239b...` confirms all 5 output arrays were `100%` finite (`all_outputs_finite = True`) on both private seeds; seed `991672` failed **`PARAMETER_ORDER`** (`estimated_taus = [1.630723, 0.280998]` vs true `[0.66, 4.35]`), **`STRAIN_ERROR`** (`0.081975 > 0.070`), and **`RATE_ERROR`** (`0.162279 > 0.055`). Verifier error codes are split cleanly, the historical `41/42` score is preserved unchanged, and the defective `2b098388239b...` solver candidate is formally retired rather than claiming post-hoc numerical solver repair.

## 2. Executed Paid-Pool Attribution (`A01` on `M9.5` + `A02` on `M9.7`, `evidence_provenance = executed`)

| Paid-Pool Replay Experiment | Strategy Arm | Requires Graph? | Public Accepted | Zero-LLM Private `strict_pass` | Probes | Provenance |
|---|---|---:|---:|---:|---:|---|
| `M9.5` (`48` reuse steps, `A01`) | `1_SingleLatest_Unvalidated_Probes1` | No | `28/48` | `28/48` | `48` | `executed` |
| `M9.5` (`48` reuse steps, `A01`) | `2_SingleLatest_ValidatedWingOLS_Probes1` | No | `40/48` | `40/48` | `48` | `executed` |
| `M9.5` (`48` reuse steps, `A01`) | `3_Multiversion_OldestFirst_Probes1` | No | **`32/48`** | **`32/48`** | `48` | `executed` |
| `M9.5` (`48` reuse steps, `A01`) | `4_Multiversion_ValidatedWingOLS_Probes2` | No | `48/48` | `48/48` | `56` | `executed` |
| `M9.5` (`48` reuse steps, `A01`) | `5_Flat_SharedConditionalTable_Probes1` | No | **`48/48`** | **`48/48`** | `48` | `executed` |
| `M9.5` (`48` reuse steps, `A01`) | `6_Hegel_MechanismGraph_Guard_Probes1` | Yes | `48/48` | `48/48` | `48` | `executed` |
| `M9.7` (`42` continual steps, `A02`) | `Flat_LatestFirst_Probes1` (`run3`) | No | `18/42` | `18/42` | `42` | `recomputed_from_raw` |
| `M9.7` (`42` continual steps, `A02`) | **`Flat_ConditionalTable_Executed_Probes1`** | **No** | **`30/42`** | **`29/42`** | `42` | **`executed` (Docker)** |
| `M9.7` (`42` continual steps, `A02`) | `Hegel_DemandGated_Graph_Probes1` (`run3`) | Yes | `30/42` | `29/42` | `42` | `recomputed_from_raw` |

- **Key Attribution Finding (`A02`)**: Across all `42` `M9.7` continual steps, prefix-locked `Flat_ConditionalTable` (`0` `PersistentMechanismGraph`) matches `Hegel_DemandGated_Graph` **`42/42`** in branch routing, **`30/42`** in Docker public-gate acceptance, and **`29/42`** in zero-LLM Docker private verification (with already-paid formation cost `6` guard calls = `98,267 v2` + `115,258.55 ms` coverage probes). Because all `M9.7` branches shared the identical `{cal_node_p0, cal_node_p1}` parent set, isolating graph representation value over `Flat_ConditionalTable` requires **branch-specific disjoint upstream dependencies & selective invalidation** (`M9.8`).

## 3. Contract Hardening, Cold-Restart Hydration & Reuse Index Closure (`A07, A08, A09, A10, A11, A16`)

- **`A07`**: `verify_binding_identity_and_foreign_keys` strictly enforces `set(parent_parameter_versions.keys()) == set(expected_parents)` (`PARENT_VERSION_SET_MISMATCH`), migrating incomplete legacy records to `unknown_requires_reverification`.
- **`A08`**: `verify_m97_multibranch_witness_partition` enforces strictly unidirectional `(eval_bid, owner_bid)` coverage and content-addressed `EmpiricalCoverageReceipt` integrity.
- **`A09`**: `classify_public_probe_failure` enforces `INFRA_FAILURE -> SCHEMA_FAILURE -> DOMAIN_REJECTION -> SCIENTIFIC_FAILURE` so timeouts (`[TIMEOUT:`) never become scientific negative witnesses.
- **`A10`**: `ImmutableBranchLineageController` persists & digest-verifies `witness_ledger` and `coverage_receipts` inside `PersistentMechanismGraph` checkpoints; cold stop after `C2` + hydration reproduces identical `R1 / U_iso / G_perm / M_couple` routes (`0` extra LLM calls).
- **`A11`**: `estimate_validated_channel_affine_calibration` unifies `Flat` and `Graph` onto the exact same stratified `2/3–1/3` estimator (`0.0` gain/bias difference across all dev steps).
- **`A16`**: `historical_capabilities.json` (`181` versions preserved + `m6_m97_continual_mechanism_lineage_and_verification` route) and `docs/hegel_machine_reuse_hub_20261006.md` now index `M6–M9.7.1`.
