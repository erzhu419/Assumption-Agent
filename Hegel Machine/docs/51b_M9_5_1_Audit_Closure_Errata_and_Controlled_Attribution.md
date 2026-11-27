# Hegel Machine M9.5.1 — Audit Closure, Errata on `docs/51`, and Controlled 4-Factor Attribution (`0` LLM Calls)

> **Immutable Evidence Guarantee**: Sealed `M9.5` primary scores (`Archive = 43/64`, `Flat = 56/64`, `Full = 63/64` inside `docs/evidence/m9_5_equal_tool_continual_attribution_run2_20261127`) and `M9.4` primary scores (`docs/evidence/m9_4_continual_mechanism_binding_world_20261126`) remain **100% untouched**. All counterfactuals, regrades, and controlled replays below are machine-generated from `docs/evidence/m9_5_1_audit_closure_and_controlled_attribution_20261127/`.

## 1. Formal Errata & Statistical Clarifications on `docs/51` (`F04`, `F12`)

1. **Statistical Separation of Accuracy vs Cost (`F12`)**:
   - **Continual Stream Cost Reduction (`8/8` worlds)**: Two-sided exact sign-flip $p = 0.0078125$ (`Full vs Flat` and `Full vs Archive` both $8/8$ worlds lower under both cost proxies).
   - **Strict Pass Rate (`Full 63/64` vs `Flat 56/64`)**: Per-world pass differences `[0, 3, 0, 3, 0, 0, 0, 1]` (`3` improved, `5` tied) -> two-sided exact sign-flip $p = 0.25$ (**NOT** statistically significant at $\alpha = 0.05$!); `Full vs Archive` (`63/64` vs `43/64`) has $p = 0.015625$.
2. **Root Cause of `T -> G1 -> N` Failure on `alpha_w04` / `beta_w04` (`F01`)**:
   - In `M9.5`, `Flat_ParameterizedCache` failed `N` after `G1` because `prior_entries` stored the already-wrapped `G1` deliverable rather than the canonical unwrapped solver (`g_old * (g_new * x + b_new) + b_old`), **NOT** because nominal `N` inherently defeats flat OLS. Under `F01` 1-hop canonical export, `6/8` double-wrapped episodes (`12/12` private instances) immediately pass!
3. **Read-Only Historical Dependency Contract (`F02`)**:
   - `5` Beta episodes (`Archive beta_w02 C'/R/G2/M` and `Flat beta_w04 R`) read `/tmp/hegel_continual_archive/archive_calibration_history.json`, which was staged in the agent container but omitted in the private verifier container. Supplying the prefix-locked (`<= t - 1`, `0444`, SHA-256 verified) history manifest flips `5/5` episodes (`10/10` private instances) to `strict_pass=True`.
4. **Concurrency & Parameter-Epoch Labels (`F12`)**:
   - Configured CLI semaphore cap = `32` across `24` stream workers; observed peak concurrent active Codex CLI calls = `26`.
   - Parameter mechanism `version` increments (`v1 -> v2 -> v3`) denote `M8` graph commit epochs rather than fixed physical ground-truth labels (`A5/B4` in `docs/51` corrected).

## 2. Unified Three-Scope Cost Ledger (`CostPolicy`: `v1` & `v2`, `F04`)

| Scope / Arm | Calls | `proxy_v1` (`U + 0.25C + O`) | `proxy_v2` (`U + 0.10C + 4O`) |
|---|---:|---:|---:|
| **In-Stream (`solve + compile`)**: `Archive_AutoReplay` | `626` | `14,471,431` | `16,082,306` |
| **In-Stream (`solve + compile`)**: `Flat_ParameterizedCache` | `584` | `13,540,685` | `15,065,362` |
| **In-Stream (`solve + compile`)**: `Full_Mechanism_Graph` | `225` | `5,273,707` (`-61.05%` vs Flat) | `5,935,605` (`-60.6%` vs Flat) |
| **Shared `Stage S` (`8` worlds `solve + L2`)** | `109` | `2,414,012` | `2,805,352` |
| **Standalone Strategy + Full `Stage S`**: `Flat` vs `Full` | — | `15,954,697` vs `7,687,719` | `17,870,714` vs `8,740,957` |
| **Campaign Physical Spend (`1,544` main + `105` aborted)** | `1649` | `38,300,886` | `42,765,192` |

## 3. Controlled 4-Factor Zero-Turn Attribution Matrix on the `48` Non-Discovery Continual Steps (`N, G1, R, C', G2, M`) (`F03`)

| Controlled Replay Variant | Public Probes Used | Zero-Turn Public Gate Accepted (`/48`) | Zero-Turn Private `strict_pass` (`/48`) | Key Attribution Finding |
|---|---:|---:|---:|---|
| `Flat_F01_Unvalidated_SingleLatest_Probes1` | `48` | `28/48` | `28/48` | Fixing `F01` restores `N/G1`, but single-version cache misses regime switches & EMI `G2` |
| `Flat_Multiversion_Validated_Probes1` | `48` | `40/48` | `40/48` | Validated `2/3–1/3` wing OLS matches `M8` on `8/8` `G2` steps, but under `probes=1` without a guard it still misses `R / G2` after `C`! |
| `Flat_Multiversion_Validated_Probes2` | `56` | `48/48` | `48/48` | Brute-force `2`-probe multi-version cache catches up on 2-branch `M9.5` at higher probe cost |
| `Full_Mechanism_Graph_Guard_Probes1` | `48` | `48/48` | `48/48` | Routes `48/48` with `1` probe/step via 3-valued guard (motivating `M9.6` autonomous model-proposed guards!) |

## 4. `F05–F11` Closure Verification Summary

- **`F05` (`alpha_w03 T` 2-layer handoff sensitivity)**: `3/3` arms pass when algebraic `secant_compliance` handoff is checked on publicly-supported OLS stress (`<= 1e-3`) while physical accuracy checks `gt['te_secant']` (`<= 0.095`).
- **`F06` (Pure JSON Numerical RPC)**: Custom class `__reduce__` rejected (`True`), `object` dtype rejected (`True`), `0` untrusted `pickle.loads`.
- **`F07` (`QualificationReceipt` Foreign Keys)**: Unissued receipt rejected (`True`), dangling `forest_node_id` rejected (`True`).
- **`F08` (`M9.3` Locked-Float Plugin Control)**: `direct_locked_float_plugin_fails_all_successors = True` (`origin = locked_stage_S_candidate_output`).
- **`F09` (`domain_guard` Slot Paths)**: Prefixed paths `[[1, 1]]` verified against `Program(op='arg')`.
- **`F10` (Per-Split Guard Completeness)**: Missing `test` history returns `UNKNOWN`.
- **`F11` (`M9.4` Full `68`-Private-Instance Regrade)**: `60/68` private instances (`30/34` episodes) pass vNext shared kernel via pure-JSON IPC.
