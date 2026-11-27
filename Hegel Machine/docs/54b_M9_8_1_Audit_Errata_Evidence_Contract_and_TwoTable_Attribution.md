# M9.8.1 Audit Closure: Evidence-Contract Hardening, Containerized Two-Table Attribution & Prefix-Locked Twin Controls (`0` Paid LLM Calls)

- **Sealed Evidence Bundle**: `docs/evidence/m9_8_1_audit_closure_and_twotable_attribution_20261128/` (`SHA256SUMS.txt` verified; all historical `M9.4`–`M9.8` evidence bundles `100%` untouched)
- **Paid Model Calls**: `0` (`evidence_provenance = executed` in air-gapped `hegel_clean_scipy_sandbox:v1`)
- **Scope**: Closes `R01–R16` from `Hegel_M9_8_Audit` prior to freezing `M9.9`.

## 1. P0 Evidence-Semantics, Timeout/Infra Isolation & Digest Hardening (`R04, R05, R06, R07`)

1. **`R04` (`r2_diag >= 0.96` Unexecuted Shortcut vs Direct Component Execution)**:
   - `diagnose_m98_factored_conflict_locus` now returns `component_probe_executed = False`, `execution_status = "not_executed"`, `evidence_origin = "compatibility_inference"`, and `chosen_up_holdout_rmse = None` (never `0.0`) when `r2_diag >= 0.96` is used without container execution, and `run_m98_equal_capability_pre_llm_trial` never records a `tested_passed` witness on unexecuted components.
   - **Direct Container Verification of the `16` `M9.8` `C1_down`/`C2_down` Instances (`02_m98_r04_16_upstream_direct_verification_and_r11_probe_ledger.json`)**: Executing all `16` instances in Docker confirms `16/16` `execution_status = "completed"` with max holdout `RMSE = 0.00019837` (`<= 0.03`, total `wall_ms = 9,734.25 ms`), while reclassifying the original `M9.8` records honestly as `compatibility_inference`.
2. **`R05` (Main Probe Timeout & Localization Container Infra / `rc=125` / `TimeoutExpired` Isolation)**:
   - Main pre-LLM `INFRA_FAILURE` skips scientific localization and leaves both component witnesses `None`.
   - Localization `subprocess.TimeoutExpired` and non-zero Docker infra codes (`rc=125`, daemon error, OOM, empty/corrupt JSON) fail closed to `locus = "unknown_infra"`, `chosen_up_holdout_rmse = None` (never `999.0`), and `0` scientific negative witnesses (`tests/test_m9_8_1_audit_closure_suite.py`).
3. **`R06` & `R07` (Canonical `ecov_` Receipt Digest, Missing-Digest Rejection & Foreign-Key Hydration)**:
   - `EmpiricalCoverageReceipt.verify_integrity()` unconditionally enforces `29`-char `ecov_` digest equality (`ecov_invalid` rejected), list-length equality, and `passed == evaluated >= 1` for `supported_on_witnesses`.
   - `ImmutableBranchLineageController` rejects current-schema checkpoints missing `state_sha256` or `witness_content_sha256`, verifies `binding_id` & `solver_code_sha256` foreign keys, and restricts legacy-schema migration strictly to `unverified_legacy` (`0` branch-minting or coverage authority).

## 2. Containerized Two-Plain-Tables Paid-Prefix Attribution & Pickle Reproducibility (`R01, R11, R12`)

| Metric (`01_m98_executed_two_plain_tables_paid_prefix_replay.json`) | Result |
|---|---:|
| Experiment Class | `paid_prefix_representation_control` (`0` Graph / `0` Registry / `0` Controller) |
| Host `SHA256(public_batch.pkl)` Match vs Sealed `M9.8` (`Python 3.10 / NumPy 2.2.6`) | **`56/56`** (+ `canonical_numeric_sha256` sealed) |
| Upstream / Downstream Feature, Binding, Mode & Composed Code SHA Match | **`56/56`** |
| Docker Public Gate **Accepted** Candidates (`R0, U_iso, M_comb1, M_comb2`) | **`32/56`** (`100%` match vs `Hegel_FactoredMechanismGraph`) |
| — Zero-LLM Private Verifier `strict_pass` on Public-Accepted Candidates | **`32/32` (`100.0%`)** |
| Docker Public Gate **Rejected** Candidates (`C1_down: 8, C2_down: 8, C_up: 8`) | **`24/56`** (triggered online LLM repair in `M9.8`) |
| Already-Paid `M9.8` Formation Cost Accounted Separately | `310` continual calls (`6,177,898 v2`) + `73` `Stage S` calls (`1,193,405 v2`) |
| Reference Timing for `8` `C_up` Localization Container Probes (`R11`) | `4,967.12 ms` (`mean = 620.89 ms`; `8/8` RMSE match vs sealed `M9.8`) |

- **`R12` Cross-Environment Pickle Note**: Executing in the original `M9.8` host runtime (`Python 3.10.12 / NumPy 2.2.6`) reproduces all `56/56` sealed `public_batch.pkl` hashes byte-for-byte, whereas cross-version `Python 3.12.14 / NumPy 2.3.5` alters CPython/NumPy pickle framing/dtype-metadata bytes while preserving `100%` identical IEEE-754 float64 array values (`56/56` feature & code SHA matches); `compute_canonical_numeric_item_sha256` now provides a Python/NumPy-version-invariant 64-hex digest.
- **`R01` Honest Attribution Conclusion**: Given the same `U/D` interface and already-paid prefix modules/guards, two plain tables reproduce `56/56` candidate programs and `32/32` zero-LLM private passes. Therefore `M9.8` demonstrates the value of **modular factored revision + cross-factor composition over a monolithic table**, motivating `M9.9` to test **equal-tool autonomous decomposition formation and non-separable `U–D` interaction withdrawal**.

## 3. Executed Prefix-Locked & Fixed Current-Only Controls on Byte-Identical `(R0, U_iso)` (`R08`)

| Policy (`03_m98_r08_prefix_locked_and_fixed_policy_twin_ablations.json`, `16` Episodes Each) | Checkpoint Used | `R0` Pass | `U_iso` Pass | Total `strict_pass` | Verifier `wall_ms` |
|---|---|---:|---:|---:|---:|
| `online_prefix_locked_history_aware` | `step_01` (`R0`), `step_04` (`U_iso`) | `8/8` | `8/8` | **`16/16` (`100.0%`)** | `86,397.51` |
| `online_prefix_locked_history_masked` (`log = []`) | `step_01` (`R0` -> `D1`), `step_04` (`U_iso` -> `D2`) | `0/8` | `0/8` | **`0/16` (`0.0%`)** | `86,272.58` |
| `fixed_current_only_D0_policy` (`u0 o d0`, `log = []`) | Prefix-calibrated `u0 o d0` | `8/8` | `0/8` | **`8/16` (`50.0%`)** | `86,075.54` |
| `fixed_current_only_D1_policy` (`u0 o d1`, `log = []`) | Prefix-calibrated `u0 o d1` | `0/8` | `8/8` | **`8/16` (`50.0%`)** | `86,359.05` |
