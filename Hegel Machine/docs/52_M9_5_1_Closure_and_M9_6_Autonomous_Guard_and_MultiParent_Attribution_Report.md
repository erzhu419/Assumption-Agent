# M9.5.1 Audit Closure & M9.6 Autonomous Structure Continual Attribution Report

- **M9.5.1 Closure Commit (`Commit 1`, `0` LLM calls)**: `767d7b4cceb93d6f39c8d39883616f0f5d829aa7` (`docs/evidence/m9_5_1_audit_closure_and_controlled_attribution_20261127/`)
- **M9.6 Confirmation Freeze Commit (`Commit 2.1`)**: `6c39d51e5e90872b1ad8be61c0d7226f75777d6a` (`source_tree_py_sha256` start == end == `e7ac0ec61423e2319a7e1a9b4304a99cf6bbca1938e18ebc2efb559eaab9d35c`)
- **M9.6 Sealed Run Bundle (`run2`)**: `docs/evidence/m9_6_autonomous_guard_multiparent_continual_run2_20261127/` (`2,173` SHA-256 verified files, `136/136` valid provenance DAGs, `567/567` mirrored raw provider ledgers, `0` unknown-usage attempts)
- **Preserved Aborted Run (`run1`, pre-Turn-1 `hash_probe` readonly-asset walk bug)**: `docs/evidence/m9_6_aborted_run1_hash_probe_readonly_asset_walk_bug_20261127/` (`114` calls, `v2 = 3,244,728`, 100% included in Scope 3 physical campaign spend)
- **Live Execution Engine**: `32`-way parallel `gpt-6.2-sol` CLI (`configured_codex_concurrency_cap = 32`, `configured_stream_workers = 32`, observed peak active CLI calls = `16`)

---

## 1. Executive Scientific Verdict (Both Breakthroughs & Falsifications Stated Plainly)

1. **M9.5 Audit Closure (`M9.5.1`, `F01–F14`, `0` LLM calls)**:
   - **`F01` (Double-Wrapping Bug)**: Re-exporting canonical algorithms 1-hop flips `6/8` double-wrapped `Flat_ParameterizedCache` episodes (`12/12` private instances) to `strict_pass = True`; only the `2` genuine cross-regime return episodes (`beta_w02 R`, `beta_w04 R`) still fail without a regime guard.
   - **`F02` (Verifier History Manifest)**: Staging prefix-locked (`<= t - 1`, `0444`, SHA-256 verified) `/tmp/hegel_continual_archive/archive_calibration_history.json` in both public probe and private verifier flips `5/5` history-reading Beta episodes (`10/10` private instances) to `strict_pass = True`.
   - **`F03` (4-Factor Offline Replay Across `48` Non-Discovery Reuse Steps)**:
     - `Flat_F01_Unvalidated_SingleLatest_Probes1`: `28/48` (`48` probes)
     - `Flat_Multiversion_Validated_Probes1`: `40/48` (`48` probes — plain `2/3–1/3` holdout-MSE wing OLS matches `M8` on `8/8` `G2` steps!)
     - `Flat_Multiversion_Validated_Probes2`: `48/48` (`56` probes)
     - `Full_Mechanism_Graph_Guard_Probes1`: `48/48` (`48` probes)
   - **`F04–F14`**: Centralized `CostPolicy` (`v1`: `14,471,431 / 13,540,685 / 5,273,707`, `v2`: `16,082,306 / 15,065,362 / 5,935,605`), two-layer Alpha OLS stress handoff (`F05`), zero-`pickle` pure-JSON numerical RPC (`F06`), `QualificationReceipt` FK enforcement (`F07`), real `M9.3` locked sources (`F08`), AST slot prefix `[1, *path]` (`F09`), train+test history check (`F10`), and full `68`-instance `M9.4` vNext regrade (`60/68` instances = `30/34` episodes, `F11`).

2. **M9.6 Positive Breakthrough — Zero-Hardcoding Autonomous Guard & Multi-Parent Induction (`8/8` Worlds, `+2.90%` Solver Cost)**:
   - With **`0` author-hardcoded guards** in the runner and a 12-feature generic observable bank populated with myopic snapshot and Frobenius-norm distractors, `gpt-6.2-sol` triggered autonomous split induction on `8/8` worlds at Step `C` (before non-Markovian `U` ever occurred!) and succeeded on **`8/8` (`100%`) on attempt #1** (`0` rejected proposals, total guard-induction spend = `145,244` `v2` across `8` worlds = `18,156` `v2`/world).
   - Across all `4/4` Alpha worlds, the model autonomously selected `max_history_and_hold_plus_residual_state_excess`; across all `4/4` Beta worlds, it autonomously selected `combined_nonseparable_or_vortex_history_excess`; and across all `8/8` worlds, contrastive + perturbation-sensitivity checks locked onto the exact 2-parent diamond `(wing_a, wing_b)` while rejecting both `aux_bench_c_calibration` and `sentinel_shared_acoustic_root` (`0.0` drift across all `64` stream steps).
   - Downstream, those autonomously induced guards routed **`8/8` `R -> C0_base`** and **`8/8` non-Markovian `U -> C1_extension`** (`same present input as R, different past!`) in **`0` turns** (`31` downstream `0`-turn closures via `publicly_contrastively_validated` guards).

3. **M9.6 Critical Negative / Falsifying Result — Nested Physical Subsumption Eliminates `Full`'s Advantage Over Steelmanned `Flat`, Leaving `L2/L1` Compilation as Pure Overhead (`+32.44%` `v2`, `p = 0.0078125`)**:
   - Once `Flat_Multiversion_Validated` received 100% of `M9.5.1`'s equal-capability fixes (`F01` 1-hop canonical export + `F02` `0444` history manifest + `F03` validated `2/3–1/3` per-wing OLS + `0` compilation tax), **both arms tied identically on accuracy and turns**:
     - `strict_pass`: **`63/64` (`98.44%`) Flat vs `63/64` (`98.44%`) Full** (`p = 1.0`).
     - Zero-turn closures: **`47/64` Flat vs `47/64` Full** (`p = 1.0`); total turns: **`210` vs `210`** (`p = 1.0`).
   - **Why `Flat` also passed `R` and `G2` in `0` turns under `probes = 1`**: Because physically, `C1` (Perzyna viscoplasticity / 2D Sylvester + vortex circulation) is a **mathematical superset** (`C1 ⊃ C0`) of `C0`! When sub-yield (`|stress| <= yield_stress`) or separable (`V_2d = rx + ry`, `vortex <= 0.50`) inputs arrive at `R` and `G2`, `Flat`'s latest canonical solver `C1` naturally outputs `viscoplastic_weights = [0, 0]` (or zero non-separable/vortex correction) and passes `R` and `G2` on probe #1 without needing to switch back to `C0`!
   - Meanwhile, on cross-domain `T`, `Full` accessed the skill catalog `8/8` times (`8/8`) but executed/retained **`0/8` (`0/8` exact reachable)** online `L2/L1` wormhole skills (`T` mean turns `13.00` Full vs `13.12` Flat; solver-only `v2` `5,028,063` Full vs `5,003,696` Flat, `+0.49%`).
   - Consequently, `Full`'s online `L2/L1` forest compilation (`1,453,819` `v2` in-stream + `820,842` `v2` in Stage S) went unamortized, making `Full` **`+32.44%` more expensive in-stream (`v2`: `6,627,126` vs `5,003,696`, exact sign-flip `p = 0.0078125`)** and **`+32.57%` more expensive standalone (`Scope 2`)**.

---

## 2. Three-Scope CostPolicy Ledger (`v1` & `v2`, Zero Omitted Calls)

| Accounting Scope | Arm / Stage | Logical Calls | Prompt Tokens | Cached Tokens | Completion Tokens | `proxy_v1` (`U + 0.25C + O`) | `proxy_v2` (`U + 0.10C + 4O`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Stage S Shared Solve** | `Stage_S_Shared_Source` (`8` worlds) | `100` | `2,636,649` | `648,320` | `112,018` | `2,262,427` | `2,501,197` |
| **Stage S L2+L1 Compile** | Charged ONLY to `Full` (`0` to `Flat`!) | `19` | `465,700` | `149,120` | `122,339` | `476,199` | `820,842` |
| **Scope 1: In-Stream Solve Only** | `Flat_Multiversion_Validated` | `210` | `5,587,244` | `1,521,024` | `196,365` | `4,642,841` | `5,003,696` |
| **Scope 1: In-Stream Solve Only** | `Full_Mechanism_Graph_Autonomous` | `210` | `5,621,095` | `1,510,528` | `191,632` | `4,679,831` | `5,028,063` |
| **Scope 1: In-Stream Guard Induction** | `Full_Mechanism_Graph_Autonomous` | `8` | `141,428` | `15,488` | `4,439` | `134,251` | `145,244` |
| **Scope 1: In-Stream L2 Compile** | `Full_Mechanism_Graph_Autonomous` | `20` | `1,260,291` | `555,904` | `173,462` | `1,016,825` | `1,453,819` |
| **Scope 1: In-Stream Total** | `Flat_Multiversion_Validated` | `210` | `5,587,244` | `1,521,024` | `196,365` | `4,642,841` | `5,003,696` |
| **Scope 1: In-Stream Total** | `Full_Mechanism_Graph_Autonomous` | `238` | `7,022,814` | `2,081,920` | `369,533` | `5,830,907` | `6,627,126` |
| **Scope 2: Standalone + Stage S** | `Flat_Multiversion_Validated` | `310` | `8,223,893` | `2,169,344` | `308,383` | `6,905,268` | `7,504,893` |
| **Scope 2: Standalone + Stage S** | `Full_Mechanism_Graph_Autonomous` | `357` | `10,125,163` | `2,879,360` | `603,890` | `8,569,533` | `9,949,165` |
| **Scope 3: Sealed Run 2 Total** | All `run2` Stages Combined | `567` | `15,712,407` | `4,400,384` | `800,255` | `13,212,374` | `14,952,861` |
| **Scope 3: Aborted Run 1 Total** | Preserved `run1` (`hash_probe` bug) | `114` | `2,967,251` | `590,720` | `202,288` | `2,726,499` | `3,244,728` |
| **Scope 3: Grand Campaign Spend** | **`run1 + run2` (`0` unknown usage)** | **`681`** | — | — | — | **`15,938,873`** | **`18,197,589`** (`19,682,201` tok) |

---

## 3. Event-by-Event Continual Breakdown (`64` Episodes per Arm)

| Causal Event | `Flat_Multiversion_Validated` Pass (`0`-Turn) | `Full_Mechanism_Graph_Autonomous` Pass (`0`-Turn) | Mean Turns (`Flat` vs `Full`) | In-Stream `proxy_v2` (`Flat` vs `Full`) |
| --- | --- | --- | --- | --- |
| `N` | `8/8` (`8` 0-turn) | `8/8` (`8` 0-turn) | `0.00` vs `0.00` | `0` vs `0` |
| `G1` | `8/8` (`8` 0-turn) | `8/8` (`8` 0-turn) | `0.00` vs `0.00` | `0` vs `0` |
| `T` | `8/8` (`0` 0-turn) | `8/8` (`0` 0-turn) | `13.12` vs `13.00` | `2,336,391` vs `2,930,741` |
| `C` | `8/8` (`0` 0-turn) | `8/8` (`0` 0-turn) | `13.12` vs `13.25` | `2,306,450` vs `3,348,153` |
| `R` | `8/8` (`8` 0-turn) | `8/8` (`8` 0-turn) | `0.00` vs `0.00` | `0` vs `0` |
| `U` | `8/8` (`8` 0-turn) | `8/8` (`8` 0-turn) | `0.00` vs `0.00` | `0` vs `0` |
| `G2` | `8/8` (`8` 0-turn) | `8/8` (`8` 0-turn) | `0.00` vs `0.00` | `0` vs `0` |
| `M` | `7/8` (`7` 0-turn) | `7/8` (`7` 0-turn) | `0.00` vs `0.00` | `0` vs `0` |

> **Forensic Note on the Single Private Failure (`alpha_m96_w03` Step 8 `M`, `7/8` on BOTH Arms)**:
> On `alpha_m96_w03_batch_08` (`M`), both `Flat` and `Full` passed the public gate in `0` turns and achieved accurate physical predictions against private ground truth (`strain_mae = 0.00956`, `rate_mae = 0.00440`, `vp_err = 0.00044` on seed `973862`), but private seed `973861` (`preload_scale = 0.55`) triggered `weights_prediction_handoff_mismatch` because `_build_m96_alpha_task`'s private consistency check initialized `z_m = np.zeros((N, 2))` instead of `ds["test"]["initial_memory_states"]` when checking `rem_c - rem_r * t` (`> 0.08`). Because both arms hit the identical seed identically (`7/8` vs `7/8`), the paired difference is `0`.

---

## 4. Per-World Paired Deltas (`Full - Flat`) & Autonomous Guard/Parent Induction

| World (`Pair`) | `Δ Pass` | `Δ 0-Turn` | `Δ Turns` | `Δ In-Stream v1` | `Δ In-Stream v2` | Autonomously Induced Guard Feature (`Step C`) | Validated Multi-Parent Roots |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `alpha_m96_w01` (`pair_01`) | `+0` | `+0` | `-1` | `+85,037` | `+133,969` | `max_history_and_hold_plus_residual_state_excess` | `alpha_wing_a_calibration,alpha_wing_b_calibration` |
| `alpha_m96_w02` (`pair_02`) | `+0` | `+0` | `+3` | `+231,609` | `+280,955` | `max_history_and_hold_plus_residual_state_excess` | `alpha_wing_a_calibration,alpha_wing_b_calibration` |
| `alpha_m96_w03` (`pair_03`) | `+0` | `+0` | `-1` | `+184,783` | `+246,739` | `max_history_and_hold_plus_residual_state_excess` | `alpha_wing_a_calibration,alpha_wing_b_calibration` |
| `alpha_m96_w04` (`pair_04`) | `+0` | `+0` | `+2` | `+312,415` | `+390,498` | `max_history_and_hold_plus_residual_state_excess` | `alpha_wing_a_calibration,alpha_wing_b_calibration` |
| `beta_m96_w01` (`pair_01`) | `+0` | `+0` | `-4` | `-36,691` | `+10,452` | `combined_nonseparable_or_vortex_history_excess` | `beta_wing_a_calibration,beta_wing_b_calibration` |
| `beta_m96_w02` (`pair_02`) | `+0` | `+0` | `-1` | `+73,110` | `+115,549` | `combined_nonseparable_or_vortex_history_excess` | `beta_wing_a_calibration,beta_wing_b_calibration` |
| `beta_m96_w03` (`pair_03`) | `+0` | `+0` | `-1` | `+156,340` | `+228,066` | `combined_nonseparable_or_vortex_history_excess` | `beta_wing_a_calibration,beta_wing_b_calibration` |
| `beta_m96_w04` (`pair_04`) | `+0` | `+0` | `+3` | `+181,463` | `+229,825` | `combined_nonseparable_or_vortex_history_excess` | `beta_wing_a_calibration,beta_wing_b_calibration` |

- **World-Level (`n = 8`) Exact Sign-Flip Tests (`Full - Flat`)**:
  - `strict_pass` delta: `0` (`p = 1.0`)
  - `zero_turn_closures` delta: `0` (`p = 1.0`)
  - `in_stream_v1` delta: `+1,188,066` (`+25.59%`, `p = 0.015625`)
  - `in_stream_v2` delta: `+1,623,430` (`+32.44%`, 95% bootstrap CI `[126614.0, 277839.12]`, `p = 0.0078125`)
- **Disjoint Pair-Cluster (`n = 4`) Test**: `4/4` disjoint pairs had higher `Full` `in_stream_v2` (`pair_01..04` deltas: `[144421, 396504, 474805, 620323]`), exact two-sided sign-flip `p = 0.125` (the minimum attainable two-sided $p$-value at $n = 4$).

---

## 5. Architectural Synthesis & Recommended Minimal Architecture Going Forward

1. **Retain (High ROI, Verified Essential)**:
   - **1-Hop Canonical Solver Export + Holdout-Validated Per-Wing Parameter Adaptation (`F01 + F03`)**: Eliminates `47/64` (`73.4%`) continual stream LLM episodes at `0` token cost.
   - **Lazy On-Conflict Autonomous Contrastive Guard Induction (`M9.6` without `L2/L1` Compilation)**: Costs only `~18k` `v2` tokens per regime split (`+2.90%` solver overhead) with `8/8` (`100%`) accuracy, and should be triggered **only** when two retained canonical solvers are **mutually non-subsuming** (i.e., when the newer solver fails on a positive witness of an older solver!). When `C_new` already passes all historical positive witnesses (`C1 ⊃ C0`), `1`-probe subsumption suffices with `0` guard-induction calls!
2. **Retire / Disable in Continual Streams (Verified Negative ROI)**:
   - **Eager Online `L2/L1` Autogenous Forest & Cross-Domain Operator Wormhole Compilation (`compile_episode_dialectical_sublation` + `sublate_autogenous_forest_into_compositional_primitives`)**: Accounted for `91.4%` of `Full`'s extra in-stream cost (`1,453,819 / 1,623,430` `v2`) plus `820,842` `v2` in Stage S while delivering `0/8` cross-domain execution/retention on `T`.
