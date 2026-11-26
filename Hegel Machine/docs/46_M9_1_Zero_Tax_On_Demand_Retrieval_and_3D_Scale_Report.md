# Hegel M9.1 Technical Report: Zero-Turn-1-Tax Demand-Triggered Retrieval, Unconfounded Router Parity, and 3D Scale Evaluation

- **Milestone**: `M9.1` (`Demand-Triggered Zero-Turn-1-Tax Retrieval + Unconfounded Router Parity + Hard Multi-Stage/3D-Scale Live Evaluation`)
- **Branch / Worktree**: `exp/m6-causal-wormhole-sandbox`
- **Frozen Code Commit (`Commit 4`)**: `8f7cc449f006854518dcb9a3b2f64e566f994ed0` (`source_tree_py_sha256 = 9fe7c6f7b524d695d5b952652f6761c6221009bd87a36c3f2fd5d47d100dfb86`, `zero_source_mutation_during_run = true`, `single_clean_unresumed_invocation = true`)
- **Sealed Evidence Bundle**: `docs/evidence/m9_1_zero_tax_demand_retrieval_20261126/` (`532` files verified via `SHA256SUMS.txt`)
- **Frozen Pretrained Library**: `pretrained_autogenous_forest_v5_m89.json` (`SHA-256 = 0fcb0ca4b8b664515f29c184cd531700c817710a8e7e8a2281ef212641a6888c`)
- **Live Execution Backend**: Official ChatGPT Subscription Codex CLI (`gpt-6.2-sol`, `reasoning_effort = "medium"`, `auth_mode = "chatgpt"`, `24` concurrent episode workers, `peak_concurrent_codex_calls = 24`, `209` total CLI calls, `808.41s` wall-clock)

---

## 1. Motivation & Hypotheses

In `M9.0a` (`docs/45`), factor-isolated decomposition revealed two structural bottlenecks:
1. **Unconditional Turn-1 Prompt Tax**: `77%` (`+44.14%` of `+57.59%`) of the `Primary 1x` token overhead of `progressive_executable_library` over `bare_no_library` originated from unconditionally injecting recalled `L2` notes and `L1` index cards into Turn 1 of every episode — even when the agent did not need library assistance or when the task was a negative control (`vr04`).
2. **Post-Primary Router Confound**: On non-`TB-Science` benchmarks lacking hardcoded domain metric extractors, `M8OfflineContainerPlanner` classified `Bare` (`skill_activation_count == 0`, `nogood_count == 0`) as `UNKNOWN` (`spawning a 2nd branch on 6/8 episodes`) while classifying `Executable` (`skill_activation_count >= 1`) as `SUPPORTED_IN_PARTIAL_SCOPE` (`spawning on 2/8 episodes`).

`M9.1` tests two falsifiable engineering hypotheses:
- **H1 (`Zero-Turn-1-Tax Retrieval`)**: Suppressing all Turn-1 skill/methodology manifest bytes (`disclosure_mode = "demand_triggered_zero_tax"`, byte-identical initial user payload size to `Bare`) while keeping the `6` qualified `v5` `L1` modules mounted silently on disk (`/tmp/hegel_wormhole_skills` with pure-JSON `--search <keywords>` / `--describe <mod>.<fn>` CLI and at most one `< 450`-char event-driven bottleneck hint) eliminates the unconditional prompt tax without degrading solve rate relative to `progressive_upfront_library`.
- **H2 (`Unconfounded Router Parity`)**: Crediting `in_container_substantive_receipt_verified == True` symmetrically across `Bare` and `Skill` arms on non-`TB-Science`/non-`BioMystery` benchmarks eliminates asymmetric 2nd-branch spawning while preserving exact `14/20` true Leave-One-Task-Out (`LOTO`) accuracy on `m7_4` (`Frontier-20`).

---

## 2. Architectural Changes (`Commit 4`: `8f7cc449f`)

1. **`demand_triggered_zero_tax` Disclosure Mode & Pure-Data `--search` CLI (`m6_4_executable_wormhole_library.py`, `m6_frontier_real_benchmarks.py`)**:
   - Turn-1 `init_payload` sets `"recalled_wormhole_methodology_and_wave1_sublations": []` and `"mounted_executable_wormhole_skills": []` (`turn1_staged_manifest_json_chars == 2`, byte-identical initial user payload to `Bare`).
   - Adds `python3 -m hegel_wormhole_skills --search "<keywords>"` over `_skill_contracts.json` (`0` skill module import/execution, logged to `/tmp/.hegel_skill_inspections.jsonl` as `skill_search_executed`).
   - Emits at most **one** compact (`<= 3` top net-value-qualified cards) `on_demand_skill_hint` inside `observation` if and only if an empirical bottleneck occurs (`nogood` recorded, non-zero command failure, or `turn_idx >= 5` without a verified incumbent).
   - Strengthens complex/wrapped-phase Helmholtz precondition vetoes in `evaluate_l1_skill_staging_net_value` so `vr04` yields `demand_triggered_recommended_l1_count == 0` (`0` false-positive hints).
2. **Symmetric Unconfounded `M8` Router State (`m8_offline_container_planner.py`, `m6_9_a_dialectical_synthesis_engine.py`)**:
   - Wires `in_container_substantive_receipt_verified` into `public_metrics` for both `Thesis` and `Antithesis` branches on non-`TB-Science`/non-`BioMystery` tasks, mapping cleanly to `SUPPORTED_IN_PARTIAL_SCOPE` regardless of `skill_activation_count`, while leaving `TB-Science` and `BioMystery` (`m7_4` `14/20` `LOTO`) completely unchanged.
3. **New `Hegel-M91-HardDemand-2` Suite (`m9_1_hard_demand_benchmark.py`)**:
   - `hd01__multicenter_cauchy_panel_and_certified_box_l0_pipeline`: 3-stage coupled multi-center reference-control-panel alignment & cell-balanced imputation + Cauchy-glitch-robust potency regression + $p \in \{20, 21, 22\}$ (`1.05M..4.19M` supports) proof-carrying symmetric-box $\ell_0$-ridge certification (`<= 2048` subcube leaves).
   - `hd02__3d_high_contrast_robin_amg_multi_rhs_and_nuisance_tomography`: True 3D (`28 × 28 × 28 = 21,952` unknowns, `4,000x` conductivity contrast) 6-face Robin elliptic PDE across `M = 8` source modes × `2` held-out instances (`16` 3D solves, `< 14.0s` solver wall-clock budget, `rel_res <= 1e-7`) + SVD-whitened ridge nuisance amplitude profiling.

---

## 3. Experimental Results (`3 Arms × 4 Tasks × 2 Replicates = 24 Episodes`)

### 3.1 Unconfounded Router Parity Verification
Across all `24/24` episodes (`8/8` per arm), `M8OfflineContainerPlanner` assigned `public_verification_state = "SUPPORTED_IN_PARTIAL_SCOPE"`, yielding `second_branch_spawn_count = 0` and `mean_policy_nominal_units = 1.000` identically across all three arms. Consequently, `Primary 1x` and `Full Policy` metrics are 100% identical with zero router-induced token confound.

### 3.2 Overall 3-Arm Aggregate Summary (`n = 8` Episodes / Arm, `max_turns = 18`)

| Arm (`n = 8` episodes / arm) | Turn-1 Manifest JSON Chars | `Primary 1x` / `Policy` Pass (`Rep1, Rep2`, SD) | Mean Turns | Total Tokens (`Δ vs Bare`) | Effective Billed Tokens (`Δ vs Bare`) | Mean Primary Wall (s) | `--search` / `--describe` Eps | Dynamic `L1` Calls (`Retained Eps`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 1: `bare_no_library`** | **`2.0`** | `6/8` (`75.0%`; `0.75, 0.75`, SD `0.00`) | `8.62` | `1,559,277` (`0.00%`) | `1,195,725` (`0.00%`) | `320.73s` | `0/8` / `0/8` | `0` (`0/8`) |
| **Arm 2: `progressive_upfront_library`** | `6,721.0` | `7/8` (`87.5%`; `1.00, 0.75`, SD `0.18`) | `8.50` | `1,958,452` (**`+25.60%`**) | `1,630,612` (**`+36.37%`**) | `307.39s` | `0/8` / `5/8` (`18` q) | `50` (`2/8`) |
| **Arm 3: `demand_triggered_zero_tax_library`** | **`2.0`** | `7/8` (`87.5%`; `0.75, 1.00`, SD `0.18`) | `9.00` | `1,643,587` (**`+5.41%`**, **`-16.08%` vs Arm 2**) | `1,295,491` (**`+8.34%`**, **`-20.55%` vs Arm 2**) | `314.79s` | `1/8` / `2/8` (`5` q) | `0` (`0/8`) |

### 3.3 Per-Task Decomposition (`2` Replicates / Cell)

| Task (`2` reps / arm) | `Bare` Pass / Tokens / Turns | `Progressive Upfront` Pass / Tokens (`Δ vs Bare`) / Turns | `Demand Zero-Tax` Pass / Tokens (`Δ vs Bare`, `Δ vs Prog`) / Turns | Key Telemetry & Mechanism Observation |
| :--- | :---: | :---: | :---: | :--- |
| **`vr02`** (`5,184`-DOF Robin SPD, Longitudinal) | `2/2` / `314,085` / `7.5` | `2/2` / `368,369` (`+17.28%`) / `7.0` | `2/2` / **`273,045` (`-13.07%`, `-25.88%`)** / `7.0` | `Progressive` retained `galerkin_preconditioner` + `solve_residual_checked_cg` (`2/2`); `Demand Zero-Tax` solved cleanly in 7 turns with `0` hints and **lowest tokens & wall-clock (`195.88s`)**. |
| **`vr04`** (`Complex Helmholtz`, Negative Control) | `2/2` / `278,733` / `7.0` | `2/2` / `406,736` (`+45.92%`) / `8.0` | `2/2` / **`325,535` (`+16.79%`, `-19.96%`)** / `8.0` | `0/2` false-positive hints, `0/2` `--describe`, `0/2` invalid `L1` calls across both library arms; `Demand Zero-Tax` eliminated all `3,294` chars of `L2` Turn-1 manifest tax (`8.0` vs `7.0` turns accounts for residual token delta). |
| **`hd01`** (`3-Stage Panel + Cauchy + Box-L0`) | `0/2` / `661,333` / `12.5` | `1/2` / `812,855` (`+22.91%`) / `12.0` | `1/2` / **`743,091` (`+12.36%`, `-8.58%`)** / `13.5` | `Demand Zero-Tax` emitted 1-shot bottleneck hint at Turn 6 in `rep_01` (`2` `--describe`) and proactively ran `--search` + `3` `--describe` in `rep_02` (`PASS` in `302,870` tokens, `-11.08%` below `Bare` `rep_02`). |
| **`hd02`** (`3D 21,952-DOF Robin Elliptic + Profiling`) | `2/2` / `305,126` / `7.5` | `2/2` / `370,492` (`+21.42%`) / `7.0` | `2/2` / **`301,916` (`-1.05%`, `-18.51%`)** / `7.5` | All 3 arms solved `16` 3D (`21,952`-DOF) high-contrast Robin systems in `2.47s..5.32s < 14.0s` (`rel_res = 0.0`) by reusing a single sparse `splu` factorization across 8 RHS; `Demand Zero-Tax` beat `Bare` by `-1.05%` tokens (`-2.28%` billed). |

---

## 4. Calibrated Scientific Discussion & Limitations

1. **Prompt Tax Eliminated (`79%` Reduction in Net Overhead; `0` Tax on Longitudinal `M9.0a` Pair)**:
   - Removing unconditional Turn-1 cards reduced total token overhead vs `Bare` from **`+25.60%` (`+36.37%` billed)** in `Progressive Upfront` down to **`+5.41%` (`+8.34%` billed)** in `Demand Zero-Tax` across all 8 episodes (`-16.08%` total / `-20.55%` billed tokens saved directly against `Progressive Upfront` at identical `7/8` pass rate).
   - On the longitudinal `M9.0a` subset (`vr02 + vr04`), `Demand Zero-Tax` consumed `598,580` total tokens vs `Bare`'s `592,818` (**`+0.97%`**) and **`444,884` effective billed tokens vs `Bare`'s `454,386` (`-2.09%`)**, proving that dormant container-mounted skills impose **zero Turn-1 token tax** when unneeded.
2. **Fundamental Retrieval Trade-Off (`Upfront Runtime Reuse` vs `Zero-Tax On-Demand Contract Consultation`)**:
   - In `Progressive Upfront`, seeing compact 1-line cards on Turn 1 led `gpt-6.2-sol` to import and retain `L1` primitives on `vr02` (`2/2` retained `autogenous_prim_conservative_spd_solve`) and dynamically execute `autogenous_prim_reference_panel_balancing` on `hd01` (`1/2`, `6` calls), at a `+25.60%` token cost.
   - In `Demand Zero-Tax`, when the base model solved `vr02` and `hd02` smoothly on its first attempt (`7` turns, `0` nogoods), it never needed to query `/tmp/hegel_wormhole_skills` (`0` dynamic `L1` calls, `0` tax). When it encountered a hard multi-stage bottleneck (`hd01`), `2/2` replicates used `--search` / `--describe` as a **read-only mathematical & contract specification reference** (`0` runtime imports) rather than importing the Python functions.
3. **Transparent Disclosure of `hd01` Failure Mode (`99301:bad_masks` Schema Ambiguity) & Sample Size (`n = 2` Replicates)**:
   - Post-experiment verifier audit of all 6 `hd01` episodes shows that all 6 runs achieved strong numerical accuracy on Stage 1 (`max_aligned_mae = 0.063..0.075 <= 0.12`) and Stage 2 (`max_potency_weight_rel_l2 = 0.033..0.057 <= 0.08`) and implemented valid diagonal-minorant branch-and-bound on Stage 3.
   - All 4 failed `hd01` episodes (`2/2` `Bare`, `1/2` `Progressive`, `1/2` `Demand Zero-Tax`) failed specifically with `"99301:bad_masks"` because `hd01`'s prompt required each leaf to include `'fixed_zero'` and `'fixed_active'` without explicitly stating `(p,)` boolean masks vs integer index lists (`np.flatnonzero(...).tolist()`). In the 2 passing library episodes (`rep_01` `Progressive` and `rep_02` `Demand Zero-Tax`), inspecting `box_ridge_cube_lower_bound` via `--describe` revealed the `(p,)` boolean mask signature `(fixed_zero: np.ndarray | None, fixed_active: np.ndarray | None)`, disambiguating the leaf format (`gap = 4e-8`, `0.10s`).
   - Accordingly, we **do not** claim the `1/2 vs 0/2` (`+1` episode, replicate SD `0.71` on `hd01`) difference as an algorithmic capability breakthrough over `gpt-6.2-sol`: on `hd02`, `gpt-6.2-sol` already recognized from the explicit PDE specification that pre-factorizing `scipy.sparse.linalg.splu(A.tocsc())` once solves all 8 3D RHS in `~2.6s` without needing algebraic multigrid, and on `hd01` the `+1` pass reflects contract disambiguation via `--describe` at `n = 2` replicates.
