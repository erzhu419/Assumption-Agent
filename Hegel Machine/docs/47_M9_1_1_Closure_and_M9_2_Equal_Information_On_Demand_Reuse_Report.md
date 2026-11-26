# M9.1.1 Deterministic Audit Closure & M9.2 Equal-Information On-Demand Executable Reuse Report

- **Date**: 2026-11-26
- **Branch**: `exp/m6-causal-wormhole-sandbox`
- **Upstream Audited Snapshot**: `d3ffa001880a6c960c85a4b8bae539f2ffef06db` (`M9.1`, preserved 100% immutable)
- **M9.1.1 Closure Commit**: `3d0da506e` (`docs/evidence/m9_1_1_audit_closure_and_regrade_20261126/`, `SHA256SUMS.txt` verified)
- **M9.2 Frozen Code Commit**: `1883431e5d3792d4ddeb579d87c65550eec2b469` (`source_tree_py_sha256 = 58e7a9f9ffbd928a2d2c3ad3b1c88230c2c1b6267c9b451937648d7bb80d16fb`)
- **M9.2 Sealed Live Evidence**: `docs/evidence/m9_2_equal_information_on_demand_reuse_20261126/` (`24/24` fresh single-branch `gpt-6.2-sol` episodes, `315` hashed files verified `100% OK`)

---

## 1. Executive Adjudication of `Hegel_M9_1_Audit`

Following independent technical adjudication of `Hegel_M9_1_Audit`, we separated **reproducible correctness, accounting, and contract-fairness findings** (adopted in full in `M9.1.1`) from **overly conservative process proposals** (streamlined in `M9.2` so we could immediately run a clean, decisive 24-episode equal-information experiment):

| Audit Finding / Recommendation | Sovereign Decision | Concrete Action & Evidence |
| --- | :---: | --- |
| **`hd01` Leaf Mask Contract Asymmetry** (`list[int]` vs `bool[p]`) | **Adopted** | Built `normalize_and_validate_box_l0_public_schema` (`hegel_box_l0_cert_schema_v2`); ran deterministic `18/18` candidate-instance sensitivity regrade (`8/8` across all 3 `M9.1` arms) while keeping original sealed `6/8, 7/8, 7/8` verdicts untouched (`docs/46b_*`). |
| **Turn-1 `+127` Input-Token Accounting & Subscription Proxy Naming** | **Adopted** | Issued explicit errata (`m9_1_turn1_full_request_accounting_errata.json`); separated `turn1_staged_manifest_json_chars` (`2`) from full-request `turn1_prompt_tokens` (`+127` in `M9.1`); replaced `effective_billed_tokens` with raw `(input, cached, output)` triples + `token_cost_proxy_w025`. |
| **4-Tuple `L1` Qualification Identity (`code, test, protocol, env`)** | **Adopted** | Enforced `hegel_l1_qual_4tuple_v2` across enrichment, staging, and export; removed silent SHA stamping when `reverify_in_clean_container=False`; container-reverified all `6/6` `L1` modules into `pretrained_autogenous_forest_v6_m911.json` (`7868bad10fbf...`). |
| **Subproblem-Scoped Preconditions & Synonym/Function Search** | **Adopted** | Soft-unwrapped clause splitting + negation-aware subproblem precondition checks (`eq04`/`vr04` keep `0` recommended `L1` skills while mixed `complex128 + real float64` `eq02` retains `4` real `L1` modules); added mathematical synonym & function-level search (`"robust regression"` -> `cauchy_linear_fit`). |
| **Quantitative `nogood` Witness & Treatment-Invariant Router (`v5`)** | **Adopted** | Closed `callable`/`hasattr` keyword spoofing; removed `skill_activation_count`/`nogood_count` from `classify_public_container_verification_state` (`v5_treatment_invariant`: `13/20` passes, `11` spawns vs `v4_legacy`: `14/20` passes, `12` spawns on `m7_4` 20-fold LOTO); normalized cross-Python AST hash (`2ef629674692...`); summed tokens across retries/timeouts. |
| **Temporal `ObservationContext` Production Wiring & Alias Retention** | **Adopted** | Wired `past_driver_history`, `history_dt`, `timestamps`, `trajectory_ids`, and `initial_memory_states` end-to-end through `_evaluate_task_model_on_probe_input` and `AcquireObservation`; revoked rebound/deleted/shadowed skill aliases in `inspect_final_deliverable_skill_retention`. |
| **Equal-Information 3-Arm Design (`B` vs `K` vs `E`) + Zero Artificial Solver Bans** | **Adopted & Accelerated** | Instead of splitting deterministic fixes into 5 micro-commits or deferring temporal/coupled regimes, we consolidated `M9.1.1` into `Commit 1`, froze `Hegel-M92-EqualInfo-4` (`eq01..eq04`) in `Commit 2` with `splu`/`cg`/`lsq_linear`/block decomposition 100% legal, and executed all `24` concurrent episodes (`peak = 24` CLI workers, `850.16s` wall-clock). |

---

## 2. Summary of `M9.1.1` Closure & Retrospective Regrade (`3d0da506e`)

1. **`hd01` Retrospective Contract-Sensitivity Regrade (`18 / 18` Instances Pass)**:
   - Re-evaluating the 6 immutable `M9.1` `hd01` submissions across the 3 private seeds with `allow_index_list_compat=True` (converting unambiguous disjoint coordinate index lists to `bool[p]` without altering weights, objectives, leaf counts, bounds, or witnesses) yields **`18 / 18` passes** (`aligned_mae = 0.0569..0.0752 <= 0.12`, `potency_rel_l2 = 0.0268..0.0574 <= 0.08`, duality gap `3.71e-8..4.05e-8 <= 1e-6`), bringing all three `M9.1` arms (`bare`, `progressive_upfront`, `demand_triggered`) from `(6/8, 7/8, 7/8)` strict verdicts to **`(8/8, 8/8, 8/8)`** schema-normalized verdicts.
2. **Deterministic Closure Verification**:
   - `tests/test_m9_1_1_audit_closure_suite.py` (`5/5` passed) and cross-suite regressions (`21/21` passed across `M8.9`, `M8.9.1`, `M9.1`, `M9.1.1`) deterministically verify all `C1..C6` counterexamples.

---

## 3. `M9.2` Experimental Design: Equal-Information On-Demand Reuse (`Hegel-M92-EqualInfo-4`)

### 3.1 Three-Arm Protocol (`Single-Branch Locked`, `max_turns = 18`, `2` Replicates)

| Arm ID | Role | Recalled Forest Notes | Turn-1 Skill Manifest Chars | Mean Turn-1 Input Tokens | On-Demand Catalog & Execution Surface |
| --- | --- | :---: | :---: | :---: | --- |
| **`B` (`bare_no_library`)** | Practical same-scaffold no-library baseline | None (`forest=None`) | `2` (`[]`) | `14,387.8` | No library; full NumPy/SciPy freedom (`splu`, `cg`, `lsq_linear`, etc.). |
| **`K` (`on_demand_knowledge_only`)** | Equal-information read-only control | `5` Wave-1 nodes | `2` (`[]`) | `14,535.8` (`+148.0` vs `B`) | `/tmp/hegel_knowledge_ref/describe_ref.py` (`--search`, `--describe`) + `.reference.txt` notes; **no importable Python package**. |
| **`E` (`on_demand_executable_library`)** | Equal-information executable treatment | `5` Wave-1 nodes (identical to `K`) | `2` (`[]`) | `14,542.8` (`+7.0` vs `K`) | `python3 -m hegel_wormhole_skills` (`--search`, `--describe`) + importable `v6` `L1` package + static AST closure packager. |

- **Equal-Information Identification**: `K` and `E` share identical `init_payload` byte lengths (`4,432`, `4,692`, `4,457`, `3,967` chars across `eq01..eq04`), virtually identical Turn-1 system prompts (`5,311..5,319` vs `5,306..5,314` chars; `+7.0` Turn-1 input tokens), identical recalled `PhilosophicalForest` notes, identical search ranking, identical mathematical preconditions, and identical value-positive lookup policy. Any difference between `K` and `E` isolates the **executable package + static packaging interface** under matched mathematical information. `B` measures total net economic value relative to running the Hegel antithesis scaffold with no historical library assets at all.

### 3.2 Four Calibrated Successor Tasks (`0` Artificial Solver Bans)

1. **`eq01` (`eq01_3d_multi_operator_robin_continuation`)**: `28 x 28 x 28 = 21,952`-DOF 3D variable-contrast (`~3,500x`) Robin PDE continuation across `K = 9` parameter states and `M = 3` source modes (`27` 3D solves/study, `54` across 2 private studies, budget `<= 25.0s`), stacked with SVD/QR ridge-whitened nuisance amplitude profiling (`4` `L1` modules recommended). Sequential per-state 3D `splu` costs `~35s` on CPU, naturally rewarding reusable preconditioning (multigrid or reference-state `splu + CG`) without banning `splu`.
2. **`eq02` (`eq02_coupled_interferometry_cauchy_box_l0`)**: Causally coupled 3-stage pipeline where Stage-1 within-center control-panel affine alignment + balanced-stratum imputation and Stage-2a `complex128` interferometric phase demodulation feed Stage-2b Cauchy IRLS calibration (`12%` heavy glitches), whose deconfounded residual vector `r_s3 = y[s3] - H[s3] @ theta_hat` forms the exact response of Stage-3 **single connected component (`p = 18, 19`)** box-constrained `L0 + L2` certified selection (`bool[p]` leaf masks explicitly specified in the task prompt; `4` `L1` modules recommended despite `complex128` Stage 2a).
3. **`eq03` (`eq03_viscoelastic_hereditary_memory`)**: Multi-center panel alignment coupled to multi-timescale Prony viscoelastic hereditary memory forecasting with explicit **`same-present / different-past`** twin loading histories (`past_driver_history`, `history_dt`) and threshold crossing-time prediction (`4` `L1` modules recommended).
4. **`eq04` (`eq04_nonhermitian_complex_coupled_negctrl`)**: Non-Hermitian complex advection-reaction (`complex128`) coupled to exact discrete subset selection with **non-separable pairwise Ising interaction penalties** (`0` `L1` modules recommended — precondition-mismatch negative control).

---

## 4. `M9.2` Sealed 24-Episode Empirical Results (`1883431e5`)

### 4.1 Provenance & Concurrency Telemetry

- **Clean Frozen Execution**: `code_commit_at_start == code_commit_at_end == 1883431e5d3792d4ddeb579d87c65550eec2b469`, `source_tree_py_sha256 == 58e7a9f9ffbd...` before and after, `git_dirty_at_start == False`, `git_dirty_at_end == False`, `fresh_executed_episode_count = 24`, `resumed_episode_count = 0`.
- **Parallel Throughput**: `24` episode workers (`max_codex_concurrency = 32`, observed `peak_concurrent_codex_calls = 24`, `257` total Codex CLI calls), completing all `24` container episodes + air-gapped private verifications in **`850.16s` (`14.17 min`)**.

### 4.2 Primary Arm Aggregates (Quality, Raw Token Triples, Cost Proxy, & Wall-Clock)

| Slice & Arm | Strict Pass (`Rep1, Rep2`) | Mean Turns | Raw Token Triple Sum `(Input, Cached, Output)` | Total Tokens (`vs B` / `vs K`) | `token_cost_proxy_w025` (`vs B` / `vs K`) | Mean Episode Wall (s) | Mean Student CPU (s) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All 8 Episodes (`eq01..eq04`)** | | | | | | | |
| `B`: `bare_no_library` | **`8/8`** (`1.0, 1.0`) | `10.62` | `(1,921,742, 521,856, 95,152)` | `2,016,894` (base / `-10.00%`) | `1,625,502` (base / `-10.34%`) | `472.63s` | `0.575s` |
| `K`: `on_demand_knowledge_only` | **`8/8`** (`1.0, 1.0`) | `11.25` | `(2,148,704, 570,624, 92,150)` | `2,240,854` (`+11.10%` / base) | `1,812,886` (`+11.53%` / base) | `445.96s` | `1.833s` |
| **`E`: `on_demand_executable_library`** | **`8/8`** (`1.0, 1.0`) | **`10.25`** | **`(1,849,750, 496,512, 79,262)`** | **`1,929,012` (`-4.36%` / `-13.92%`)** | **`1,556,628` (`-4.24%` / `-14.14%`)** | **`405.27s`** | `1.820s` |
| **Reuse-Eligible 6 Episodes (`eq01..eq03`)** | | | | | | | |
| `B`: `bare_no_library` | **`6/6`** (`1.0, 1.0`) | `10.83` | `(1,498,618, 396,288, 72,210)` | `1,570,828` (base / `-3.96%`) | `1,273,612` (base / `-4.17%`) | `509.13s` | `0.665s` |
| `K`: `on_demand_knowledge_only` | **`6/6`** (`1.0, 1.0`) | `10.83` | `(1,571,436, 408,832, 64,243)` | `1,635,679` (`+4.13%` / base) | `1,329,055` (`+4.35%` / base) | `445.35s` | `2.337s` |
| **`E`: `on_demand_executable_library`** | **`6/6`** (`1.0, 1.0`) | **`9.83`** | **`(1,318,421, 330,368, 55,436)`** | **`1,373,857` (`-12.54%` / `-16.01%`)** | **`1,126,081` (`-11.58%` / `-15.27%`)** | **`409.19s`** | `2.309s` |
| **Negative Control 2 Episodes (`eq04`)** | | | | | | | |
| `B`: `bare_no_library` | **`2/2`** (`1.0, 1.0`) | `10.00` | `(423,124, 125,568, 22,942)` | `446,066` (base / `-26.29%`) | `351,890` (base / `-27.27%`) | `363.13s` | `0.304s` |
| `K`: `on_demand_knowledge_only` | **`2/2`** (`1.0, 1.0`) | `12.50` | `(577,268, 161,792, 27,907)` | `605,175` (`+35.67%` / base) | `483,831` (`+37.50%` / base) | `447.79s` | `0.320s` |
| **`E`: `on_demand_executable_library`** | **`2/2`** (`1.0, 1.0`) | `11.50` | `(531,329, 166,144, 23,826)` | `555,155` (`+24.46%` / `-8.27%`) | `430,547` (`+22.35%` / `-11.01%`) | `393.50s` | `0.355s` |

### 4.3 Per-Task Paired Breakdown (`rep_01` & `rep_02`)

| Task | Replicate | `B` (`bare_no_library`) Tokens / Turns / CPU | `K` (`knowledge_only`) Tokens / Turns / CPU | `E` (`executable_library`) Tokens / Turns / CPU | 2-Rep Task Sum (`E vs B` / `E vs K`) |
| --- | :---: | :---: | :---: | :---: | :---: |
| **`eq01`** (3D `21,952`-DOF Robin Continuation) | `rep_01` | `229,550` / `10t` / `1.86s` | `377,514` / `14t` / `6.07s` | `299,865` / `12t` / `7.52s` | `E=531,559`, `B=561,866`, `K=584,679`<br>(**`-5.39%` vs `B`**, **`-9.09%` vs `K`**) |
| **`eq01`** | `rep_02` | `332,316` / `13t` / `1.75s` | `207,165` / `9t` / `7.61s` | `231,694` / `10t` / `5.97s` | |
| **`eq02`** (Coupled Panel + `complex128` + Connected Box-L0) | `rep_01` | `265,302` / `11t` / `0.13s` | `253,139` / `10t` / `0.09s` | **`209,084` / `9t` / `0.11s`** | `E=447,196`, `K=610,105`, `B=648,762`<br>(**`-31.07%` vs `B`**, **`-26.70%` vs `K`**) |
| **`eq02`** | `rep_02` | `383,460` / `14t` / `0.09s` | `356,966` / `13t` / `0.10s` | **`238,112` / `10t` / `0.09s`** (`33` `L1` calls) | |
| **`eq03`** (Viscoelastic `Same-Present/Different-Past`) | `rep_01` | `160,665` / `8t` / `0.09s` | `230,895` / `10t` / `0.06s` | `224,683` / `10t` / `0.08s` | `B=360,200`, `E=395,102`, `K=440,895`<br>(`+9.69%` vs `B`, **`-10.39%` vs `K`**) |
| **`eq03`** | `rep_02` | `199,535` / `9t` / `0.08s` | `210,000` / `9t` / `0.10s` | **`170,419` / `8t` / `0.08s`** | |
| **`eq04`** (Non-Hermitian + Coupled Ising NegCtrl) | `rep_01` | `186,843` / `9t` / `0.29s` | `299,196` / `12t` / `0.21s` | `359,026` / `14t` / `0.37s` | `B=446,066`, `E=555,155`, `K=605,175`<br>(`+24.46%` vs `B`, **`-8.27%` vs `K`**) |
| **`eq04`** | `rep_02` | `259,223` / `11t` / `0.32s` | `305,979` / `13t` / `0.43s` | **`196,129` / `9t` / `0.34s`** | |

- **Pairwise Consistency**:
  - **`E` vs `K` (Equal-Information Execution vs Read-Only Knowledge)**: `E` uses fewer tokens than `K` in **`6 / 8` episode pairs** and in **`4 / 4` (`100%`) task-family aggregates** (`-9.09%`, `-26.70%`, `-10.39%`, `-8.27%`), saving `-311,842` tokens (`-13.92%`) and `-40.69s` mean wall-clock per episode at identical `8/8` pass rate.
  - **`E` vs `B` (Practical Net-Benefit Comparison)**: `E` uses fewer tokens than `B` in **`5 / 8` episode pairs** (`rep_01/eq02`, `rep_02/eq01`, `rep_02/eq02`, `rep_02/eq03`, `rep_02/eq04`), achieving **`-12.54%` (`-196,971` tokens) on reuse-eligible tasks (`eq01..eq03`)** and **`-4.36%` (`-87,882` tokens) across all 8 episodes** (including the negative control `eq04`), alongside **`-14.25%` (`-67.36s`) faster mean episode wall-clock**.

---

## 5. Mechanistic Funnel & Forensic Trajectory Analysis (No Overclaiming)

| Funnel Stage | `B` (`bare_no_library`) | `K` (`on_demand_knowledge_only`) | `E` (`on_demand_executable_library`) |
| --- | :---: | :---: | :---: |
| Turn-1 `staged_wormhole_manifest` JSON chars | `2` (`0` cards) | `2` (`0` cards) | `2` (`0` cards) |
| Episodes voluntarily invoking `--search` / `--describe` | `0 / 8` | **`8 / 8`** (`16` searches, `21` `--describe`s) | **`8 / 8`** (`8` searches, `27` `--describe`s) |
| Direct source/reference inspection (`sed`/`grep`) | `0 / 8` | Observed on `.reference.txt` (e.g., `eq01`, `eq02`) | Observed on `/tmp/hegel_wormhole_skills/*.py` (`eq01`, `eq02`) |
| Live `/tmp` runtime calls to `hegel_wormhole_skills` | `0` (`0/8`) | `0` (`0/8`) | **`33` calls (`1/8`: `rep_02/eq02`)** |
| Static `hegel_wormhole_skills` imports in final deliverable | `0 / 8` | `0 / 8` | **`0 / 8`** (all wrote self-contained `<235`-line NumPy/SciPy code) |

Forensic inspection of the 24 trajectories establishes **why** `E` achieved lower total tokens than both `K` and `B` despite `0/8` static import retention in final deliverables:

1. **Value-Positive Policy Restored `100%` Voluntary On-Demand Discovery**:
   - Replacing `M9.1`'s restrictive `"Leave untouched ... ONLY if bottleneck"` wording with `M9.2`'s symmetric value-positive rule lifted voluntary `--search` / `--describe` from `0/8` (`M9.1`) to **`8/8` in both `K` and `E`**, without paying `M8.9`/`M9.0a`'s upfront Turn-1 manifest tax (`+155` Turn-1 tokens vs `B` instead of `+6,491`).
2. **How `E` Outperformed `K` (`-26.70%` on `eq02`, `-13.92%` Overall)**:
   - On `eq02` (coupled panel + `complex128` phase + Cauchy IRLS + connected `p=18..19` Box-L0), subproblem-scoped gating correctly recommended all `4` real `L1` modules despite Stage 2a's `complex128` interferometry arrays.
   - In `E`, agents ran `1` targeted `--search` + `5–6` `--describe` calls, inspected `autogenous_prim_box_ridge_bounds.py` directly, wrote compact self-contained branch-and-bound solvers (`188–232` lines), and in `rep_02/eq02` Turn 6 **imported and executed `box_ridge_cube_lower_bound` `33` times inside `/tmp/eq02_end_to_end.py` as an executable mathematical oracle** to verify every leaf's `minorant_center` and `minorant_diagonal` bound before submitting in `9–10` turns (`447,196` tokens vs `610,105` in `K` and `648,762` in `B`).
   - By contrast, `K` agents issued `2x` as many `--search` queries (`16` vs `8`), spent extra shell pipelines locating/slicing `.reference.txt` files, and had to build custom synthetic bound-replay oracles from scratch because `/tmp/hegel_knowledge_ref` is non-importable (`11.25` mean turns vs `10.25` in `E`).
3. **Why `Bare` Had Faster Student CPU Runtime on `eq01` (`1.80s` vs `6.75s` in `E`)**:
   - Because `M9.2` enforced **zero artificial solver bans**, `Bare` wrote a custom 4-level Galerkin `_Multigrid` preconditioner (`1.80s` student runtime across `54` `21,952`-DOF 3D solves, costing `561,866` tokens and `710.36s` episode wall-clock), whereas `E` (`531,559` tokens, `626.67s` wall-clock) and `K` (`584,679` tokens) inspected `autogenous_prim_conservative_spd_solve`, noted `pyamg` was absent, and used a single geometric-mean reference-state `splu(A(sqrt(theta_min * theta_max)))` `LinearOperator` preconditioner inside `scipy.sparse.linalg.cg` (`6.75s` / `6.84s`, comfortably inside the `25.0s` budget).
4. **Honest Negative-Control Lookup Cost (`eq04`)**:
   - On `eq04` (`0` recommended `L1` modules), both `K` and `E` ran exploratory Turn-1/Turn-2 `--search` and `--describe autogenous_sparse_ridge_certification.solve_and_certify_l0`, saw the `p <= 16` real-only precondition mismatch, and rejected reuse (`0` calls). That exploratory lookup added `+24.46%` tokens on `eq04` for `E` (`+35.67%` for `K`), showing that zero-Turn-1-manifest on-demand retrieval reduces — but does not eliminate — voluntary lookup overhead on unsuited tasks. Crucially, `E`'s savings on `eq01 + eq02` (`-231,873` tokens vs `B`) exceeded `eq04 + eq03` overhead (`+143,991` tokens), yielding a net `-87,882` token (`-4.36%`) saving across the full 8-episode suite.

---

## 6. Self-Audit & Residual Plumbing Notes (Transparent Scope Boundaries)

1. **Statistical Scope (`24` Diagnostic Episodes)**:
   - `4` tasks × `2` replicates × `3` arms (`24` episodes) is a controlled mechanism-and-cost slice on a single frozen commit, **not** a universal multi-benchmark proof of asymptotic RSI.
2. **`B` vs `K`/`E` Forest-Note Scope**:
   - `K` vs `E` is the exact equal-information comparison (identical `5` recalled `PhilosophicalForest` notes + identical `--search`/`--describe` catalog, differing only in executable Python importability + static packaging). `B` (`forest=None`) is the practical no-library baseline.
3. **Two Minor Runner Plumbing Observations Disclosed Upfront (Zero Arm Asymmetry)**:
   - **Container-boot `public_setup_py` whitelist**: While `test_m9_2_equal_info_reuse_suite.py` executed `public_setup_py` (verifying `/holdout_task/validate_public_schema.py` and `public_eq0*_studies.pkl`), `m6_frontier_real_benchmarks.py` added `"Hegel-M92-EqualInfo-4"` to the end-of-episode air-gapped private verifier dispatch without adding `"Hegel-M92-EqualInfo-4"` to the container-boot `public_setup_py` whitelist. Consequently, all `24/24` episodes across `B`, `K`, and `E` **symmetrically** started with an empty container (`0` privileged format channels in any arm), constructed their own synthetic multi-regime stress/oracle suites in `/tmp`, and achieved **`24/24` (`100%`) strict passes** on the air-gapped private verifier because `eq01..eq04`'s prompt contracts already specified exact `bool[p]` types, keys, and shapes.
   - **`inspected_knowledge_reference_txt` counter substring**: The summary counter in `run_m9_2_equal_information_on_demand_experiment.py` checked `"hegel_skill_knowledge_refs"` rather than `"/tmp/hegel_knowledge_ref"` + `".reference.txt"`, reporting `0` even though `K` trajectories (e.g., `rep_01/eq02` Turn 3, `rep_02/eq01` Turn 2) explicitly read `.reference.txt` via `sed`. Both minor items should be tightened before the next online-growth successor chain (`M9.3`).
