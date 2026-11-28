# Hegel Machine `M9.9.1` Audit Closure and `M10.0` Reliable Modular Amortization & Bidirectional Separability Withdrawal Report

> **Deterministic Generation Notice (`Q05 / Q12`)**: Generated strictly from sealed SHA-256-verified JSON ledgers (`8` files in `m9_9_1_audit_closure_and_forensic_reverification_20261128`, `3946` files in `m10_0_dev_economic_calibration_20261128`, and `7880` files in `m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_20261128`) with zero hand-edited metrics.

---

## 1. Executive Summary & Calibrated Claim Boundary (`Q09 / Q10 / D1–D2`)

Following the independent GPT-6 audit of `M9.9` (`d4b1cd7f0`), this milestone delivers two strictly separated stages:
1. **`M9.9.1` Forensic Closure (`0` Paid LLM Calls)**: Closed all `Q01–Q14` and historical `R01–R16` audit items without touching any existing sealed archive (`m9_8*`, `m9_9*`). All `30/30` historical off-diagonal `M9.9` decomposition pairs (`u0∘d0@E1` and `u0∘d1@S0`) and `7/7` historical withdrawal counterexamples were re-executed in `hegel-clean-scipy:m7v1`, confirming genuine scientific/domain rejections; `bind_alpha_m99_c04_up_b02 ∘ dn_b03` was verified in Docker on `alpha_m99_c04/E7` (`strict_pass = True`, diagnostic only); `alpha_m99_c02/E7` (`0.071267 -> 0.004857`) was isolated to missing pooled `ref_coupons` standardization/imputation; and historical cancelled dev calls (`333` calls, `8,220,922 + y` `proxy_v2`) were recovered into `53,904,542 + x + y` (`docs/55b_*`, `docs/55c_*`).
2. **`M10.0` Prospective `3 × 3` Bidirectional Modular Amortization (`n = 4` Dev Calibration $\to$ Zero-Code-Change Freeze $\to$ Single-Shot `n = 8` Unexposed Confirmation)**:
   - **Scoped Mechanism Claim (`Q09 / Q10`)**: Under a **prescribed two-stage interface** (`calibrate_upstream`, `solve_downstream`), `Hegel_SelectiveConditionExpansion_Online` tests **counterexample-driven selective condition-set expansion (`dn_obs -> up_obs + dn_obs`) and bidirectional separability withdrawal** across `{U0, U1, U2} × {D0, D1, D2}` (`U1 ⊗ D2 -> D2_coupled`), balanced `50% / 50%` between `Order A (sep_first)` and `Order B (coupled_first)` and upgraded symmetrically across modular arms with `B1` (public Alpha pooled `ref_coupons` exchange invariance), `B2` (`1`-sibling upstream fallback), and `B3` (unfactored monolithic retention + metered lazy downstream projection).
   - **Plain-Dict Algorithmic Equivalence (`Q10`)**: On the frozen `step_08` prefix across all confirmation clusters, a plain Python dictionary evaluating the same learned modules, joint `15`-observable vectors, and Kleene `all_of` rules reproduces **`16/16` (`100.0%`)** of `Hegel`'s composed candidate SHA-256 hashes (`8/8 = 100.0%` on `dev`). Thus, the empirical gains below belong to the **selective condition-expansion & withdrawal algorithm and symmetric verification gates (`B1–B3`)**, NOT to the `PersistentMechanismGraph` container data structure or open-topology arbitrary-depth DAG discovery (which remains future work).

### Headline `M10.0` Single-Shot Confirmation Results (`n = 8` Independent Unexposed Clusters, `80` Stream Episodes/Arm, `x = 0` Unknown Attempts)

| Arm (`Equal Tools, Same Sandbox & Guard Compiler`) | Official `strict_pass` (`80` eps) | `0`-LLM Passed Closures | Solver Turns | `Scope-1` Calls | `Scope-1` `proxy_v2` | `Scope-Total` (`+Stage S`) `proxy_v2` | Container Probe + Verifier Elapsed (`sec`) | Verified Withdrawals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicConditionalTable_Online` | `72/80` (`90.00%`) | `15/80` | `762` | `823` | `17,006,487` | `18,454,099` | `2,932.35s` | `0` |
| `Flat_PlainModuleRegistry_Online` (`B1+B2+B3`) | `74/80` (`92.50%`) | `29/80` | `587` | `635` | `12,592,301` | `14,039,913` | `1,906.42s` | `0` |
| **`Hegel_SelectiveConditionExpansion_Online` (`B1+B2+B3`)** | **`77/80` (`96.25%`)** | **`37/80`** | **`503`** | **`560`** | **`11,340,259`** (`-33.32%` vs Mono / `-9.94%` vs Plain) | **`12,787,871`** (`-30.70%` / `-8.92%`) | **`1,736.30s`** | **`8`** |

Key takeaways on the `n = 8` unexposed confirmation split (`981000..988000`, frozen commit `042ab4299`, zero `.py` changes after `dev`):
- **Simultaneous Quality Superiority & Pre-Registered Non-Inferiority (`delta_NI = -0.05`)**: `Hegel` achieved **`77/80` (`96.25%`)**, outscoring both `Flat_PlainModuleRegistry_Online` (`74/80 = 92.50%`, diff **`+0.0375`**, 95% bootstrap CI `[-0.04, 0.1]`) and `Flat_MonolithicConditionalTable_Online` (`72/80 = 90.00%`, diff **`+0.0625`**, 95% bootstrap CI `[-0.01, 0.14]`). Both point estimates and 95% lower bootstrap bounds (`-0.01` and `-0.04`) lie strictly above `-0.05`.
- **Decisive Economic Amortization Against Monolithic Primary Benchmark**: `Hegel` reduced `Scope-1` `proxy_v2` cost on **`8/8` (`100%`)** independent clusters (`4/4` Alpha `p = 0.125`, `4/4` Beta `p = 0.125`, combined exact two-sided sign-flip **`p = 0.0078125`**), saving **`5,666,228` `proxy_v2` (`-33.32%`)** with empirical quality-constrained break-even at **`K* = 3`** (out of `10` steps), winning in both `sep_first` (`mean_diff = -663,071.00`) and `coupled_first` (`mean_diff = -753,486.00`).
- **Positive Net Savings & Break-Even Against Equal-Tool Plain Modular Registry**: `Hegel` saved **`1,252,042` `proxy_v2` (`-9.94%`)** relative to `Flat_PlainModuleRegistry_Online` (`6/8` clusters cheaper, exact two-sided `p = 0.21875`, Beta `4/4` `p = 0.125`), reaching empirical quality-constrained break-even at **`K* = 9`** when colliding `D2` revisits arrive, and expanding `0`-LLM passed closures from `29/80` to **`37/80`** (`7/8` clusters positive, exact two-sided `p = 0.0546875`; vs Monolithic `15/80`, `8/8` clusters positive, **`p = 0.0078125`**).
- **Step-08 Constrained Routing Ceiling**: On all `16` colliding `D2` revisit episodes (`S09, S10` across `8` clusters), `hegel_selective_condition_expansion` achieved **`16/16` (`100.0%`)** `0`-LLM private `strict_pass`, versus **`8/16` (`50.0%`)** for `dn_only_fixed_prior_d2` and **`8/16` (`50.0%`)** for `dn_only_fixed_latest_d2`.

---

## 2. Execution Provenance, Concurrency, & Full R&D Cost Ledger (`C1–C2 / Q05 / Q12`)

### 2.1 Immutable Start/Completion/Ablation Provenance (`0` Post-Dev Source Edits)

| Split | `execution_run_id` | Start Commit | Dirty at Start? | `.py` Source Tree SHA-256 (`src+scripts+tests`) | Tree Unchanged During Run? | Paid Execution Elapsed (`sec`) | Ablation Elapsed (`sec`) | Concurrency (`cap / peak`) | Unknown Attempts |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `dev` (`n=4`) | `m100_dev_20261128T021932Z_f5bac8c4` | `1697f4d04` | `False` | `8b13f56969f8219f...` | `True` | `3,877.17s` | `103.66s` | `32 / 12` | `0` |
| `confirmation` (`n=8`) | `m100_confirmation_20261128T032437Z_6dc6e01a` | `042ab4299` | `False` | `8b13f56969f8219f...` | `True` | `3,824.73s` | `245.27s` | `32 / 24` | `0` |

- **Zero Post-Dev Source Modification**: `source_tree_sha256_at_start` is byte-identical (`8b13f56969f8219f54d8bd21e6fe91f48d24aefe22aeac9932579bcdbdc4cfcb`) across `dev` and `confirmation`. Commit `042ab4299` added only the sealed `m10_0_dev_economic_calibration_20261128` directory before launching `confirmation`.
- **Complete `M10.0` Paid R&D Ledger (`0` Cancelled Runs)**:
  - `M10.0` `dev` (`n = 4`, `124` episodes): `1,028` calls, `26,451,981` total tokens, **`22,295,301` `proxy_v2`** (`0` unknown attempts).
  - `M10.0` `confirmation` (`n = 8`, `248` episodes): `2,096` calls, `52,977,017` total tokens, **`42,386,659` `proxy_v2`** (`0` unknown attempts).
  - `M10.0` Grand Total (`dev + confirmation`, `0` cancelled runs): **`3,124` calls / `64,681,960` exact `proxy_v2`**.

### 2.2 Hierarchical Role Conservation on `M10.0` Confirmation (`all_provider_call_receipts.json`)

| Role Partition | `Monolithic` Calls / `proxy_v2` | `Plain` Calls / `proxy_v2` | `Hegel` Calls / `proxy_v2` |
|---|---:|---:|---:|
| `solve_only` | `762` / `15,859,307` | `587` / `11,747,825` | `503` / `10,277,990` |
| `decomposition_induction_only` (`8/8` Plain, `8/8` Hegel) | `0` / `0` | `15` / `444,568` | `15` / `440,597` |
| `lazy_retroactive_projection_only` (`B3`) | `0` / `0` | `0` / `0` | `0` / `0` |
| `guard_induction_only` | `61` / `1,147,180` | `33` / `399,908` | `42` / `621,672` |
| **`Scope-1` In-Stream Total** | **`823` / `17,006,487`** | **`635` / `12,592,301`** | **`560` / `11,340,259`** |
| Shared `Stage S` Source (`8/8` public+private pass) | `78` / `1,447,612` (shared across 3 arms) | same | same |

---

## 3. Cluster-by-Cluster Confirmation Breakdown & Paired Statistical Inference (`C3 / Q06`)

### 3.1 Per-Cluster Results (`n = 8` Independent Confirmation Source Clusters)

| World ID | Family | Order Mode | `Monolithic` Pass (`0`-LLM) | `Plain` Pass (`0`-LLM) | `Hegel` Pass (`0`-LLM) | `Mono` `Scope-1 v2` | `Plain` `Scope-1 v2` | `Hegel` `Scope-1 v2` | `Hegel - Mono` `v2` | `Hegel - Plain` `v2` | `Hegel` Withdrawals |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `alpha_m100_c01` | `alpha` | `sep_first` | `10/10` (`2`) | `9/10` (`4`) | **`10/10` (`5`)** | `1,959,574` | `1,680,065` | **`1,299,879`** | `-659,695` (-33.67%) | `-380,186` (-22.63%) | `1` |
| `alpha_m100_c02` | `alpha` | `coupled_first` | `9/10` (`2`) | `9/10` (`4`) | **`10/10` (`5`)** | `2,161,629` | `1,527,679` | **`1,316,094`** | `-845,535` (-39.12%) | `-211,585` (-13.85%) | `1` |
| `alpha_m100_c03` | `alpha` | `sep_first` | `5/10` (`1`) | `9/10` (`4`) | **`8/10` (`3`)** | `2,316,370` | `1,474,417` | **`1,988,171`** | `-328,199` (-14.17%) | `513,754` (+34.84%) | `1` |
| `alpha_m100_c04` | `alpha` | `coupled_first` | `10/10` (`2`) | `8/10` (`4`) | **`10/10` (`5`)** | `2,055,719` | `1,490,387` | **`1,514,499`** | `-541,220` (-26.33%) | `24,112` (+1.62%) | `1` |
| `beta_m100_c01` | `beta` | `sep_first` | `10/10` (`2`) | `9/10` (`4`) | **`10/10` (`5`)** | `2,037,798` | `1,380,511` | **`1,165,037`** | `-872,761` (-42.83%) | `-215,474` (-15.61%) | `1` |
| `beta_m100_c02` | `beta` | `coupled_first` | `9/10` (`2`) | `10/10` (`3`) | **`10/10` (`5`)** | `2,240,789` | `1,685,436` | **`1,203,963`** | `-1,036,826` (-46.27%) | `-481,473` (-28.57%) | `1` |
| `beta_m100_c03` | `beta` | `sep_first` | `9/10` (`2`) | `10/10` (`3`) | **`10/10` (`5`)** | `2,088,754` | `1,731,132` | **`1,297,125`** | `-791,629` (-37.90%) | `-434,007` (-25.07%) | `1` |
| `beta_m100_c04` | `beta` | `coupled_first` | `10/10` (`2`) | `10/10` (`3`) | **`9/10` (`4`)** | `2,145,854` | `1,622,674` | **`1,555,491`** | `-590,363` (-27.51%) | `-67,183` (-4.14%) | `1` |

### 3.2 Paired Cluster-Level Exact Two-Sided Sign-Flip & Bootstrap Inference (`n = 8`)

| Comparison (`Hegel` minus Baseline) | Mean Cluster `Scope-1 v2` Diff (95% Boot CI) | Cost Exact 2-Sided `p` (`All / Alpha / Beta`) | `sep_first` vs `coupled_first` Mean `v2` Diff | Quality Rate Diff/Ep (95% Boot CI; `delta_NI = -0.05`) | Quality Exact 2-Sided `p` | `0`-LLM Closure Cluster Diffs & Exact 2-Sided `p` |
|---|---|---|---|---|---:|---|
| **`Hegel - Monolithic`** | **`-708,278.50`** (`[-848,648.82, -555,651.29]`) | **`0.0078125`** (`0.125` / `0.125`) | `-663,071.00` / `-753,486.00` | **`+0.0625`** (`[-0.01, 0.14]`, **NI PASS**) | `0.3125` | `[3.0, 3.0, 2.0, 3.0, 3.0, 3.0, 3.0, 2.0]`, **`p = 0.0078125`** |
| **`Hegel - Plain`** | **`-156,505.25`** (`[-340,623.00, 77,101.88]`) | `0.21875` (`1.0` / `0.125`) | `-128,978.25` / `-184,032.25` | **`+0.0375`** (`[-0.04, 0.1]`, **NI PASS**) | `0.53125` | `[1.0, 1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 1.0]`, **`p = 0.0546875`** |

---

## 4. Cumulative Amortization Curves $\Delta C(K)$ (`K = 1..10`) & Empirical Break-Even $K^*$ (`D2`)

### 4.1 `Hegel` minus `Flat_MonolithicConditionalTable_Online` (`Empirical K* = 3`)

| Step `K` | `Hegel` Cum `Scope-1 v2` | `Monolithic` Cum `Scope-1 v2` | Total `Delta C_mono(K)` | Mean Cluster `Delta v2` | Clusters Cheaper (`/8`) | Cum `strict_pass` (`Hegel` vs `Mono`) | Quality Diff (`>= -0.05` Check) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `2,318,465` | `1,935,783` | `382,682` | `47,835.25` | `0` | `7/8` vs `6/8` | `+0.1250` (`PASS`) |
| `2` | `4,175,556` | `3,886,709` | `288,847` | `36,105.88` | `2` | `15/16` vs `14/16` | `+0.0625` (`PASS`) |
| `3` | `4,975,781` | `5,781,119` | `-805,338` | `-100,667.25` | `5` | `23/24` vs `22/24` | `+0.0417` (`PASS`) |
| `4` | `6,268,865` | `7,773,491` | `-1,504,626` | `-188,078.25` | `7` | `30/32` vs `29/32` | `+0.0312` (`PASS`) |
| `5` | `8,539,180` | `10,552,522` | `-2,013,342` | `-251,667.75` | `7` | `37/40` vs `36/40` | `+0.0250` (`PASS`) |
| `6` | `9,075,950` | `12,481,655` | `-3,405,705` | `-425,713.12` | `7` | `45/48` vs `43/48` | `+0.0417` (`PASS`) |
| `7` | `10,278,123` | `14,342,987` | `-4,064,864` | `-508,108.00` | `8` | `53/56` vs `50/56` | `+0.0536` (`PASS`) |
| `8` | `11,340,259` | `16,754,839` | `-5,414,580` | `-676,822.50` | `8` | `61/64` vs `57/64` | `+0.0625` (`PASS`) |
| `9` | `11,340,259` | `16,754,839` | `-5,414,580` | `-676,822.50` | `8` | `69/72` vs `65/72` | `+0.0556` (`PASS`) |
| `10` | `11,340,259` | `17,006,487` | `-5,666,228` | `-708,278.50` | `8` | `77/80` vs `72/80` | `+0.0625` (`PASS`) |

### 4.2 `Hegel` minus `Flat_PlainModuleRegistry_Online` (`Empirical K* = 9`)

| Step `K` | `Hegel` Cum `Scope-1 v2` | `Plain` Cum `Scope-1 v2` | Total `Delta C_plain(K)` | Mean Cluster `Delta v2` | Clusters Cheaper (`/8`) | Cum `strict_pass` (`Hegel` vs `Plain`) | Quality Diff (`>= -0.05` Check) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `2,318,465` | `2,405,789` | `-87,324` | `-10,915.50` | `5` | `7/8` vs `8/8` | `-0.1250` (`FAIL`) |
| `2` | `4,175,556` | `4,324,743` | `-149,187` | `-18,648.38` | `6` | `15/16` vs `16/16` | `-0.0625` (`FAIL`) |
| `3` | `4,975,781` | `5,167,353` | `-191,572` | `-23,946.50` | `6` | `23/24` vs `24/24` | `-0.0417` (`PASS`) |
| `4` | `6,268,865` | `6,119,348` | `149,517` | `18,689.62` | `6` | `30/32` vs `31/32` | `-0.0312` (`PASS`) |
| `5` | `8,539,180` | `8,212,706` | `326,474` | `40,809.25` | `4` | `37/40` vs `39/40` | `-0.0500` (`PASS`) |
| `6` | `9,075,950` | `8,212,706` | `863,244` | `107,905.50` | `4` | `45/48` vs `47/48` | `-0.0417` (`PASS`) |
| `7` | `10,278,123` | `9,137,284` | `1,140,839` | `142,604.88` | `4` | `53/56` vs `55/56` | `-0.0357` (`PASS`) |
| `8` | `11,340,259` | `10,131,364` | `1,208,895` | `151,111.88` | `4` | `61/64` vs `63/64` | `-0.0312` (`PASS`) |
| `9` | `11,340,259` | `11,896,029` | `-555,770` | `-69,471.25` | `5` | `69/72` vs `66/72` | `+0.0417` (`PASS`) |
| `10` | `11,340,259` | `12,592,301` | `-1,252,042` | `-156,505.25` | `6` | `77/80` vs `74/80` | `+0.0375` (`PASS`) |

---

## 5. Container Probe Elapsed Wall-Time Breakdown (`Q11`) & `Gate 3` Dev Summary

### 5.1 Non-LLM Container Probe & Verifier Elapsed Wall-Time (`OMP_NUM_THREADS=1`, Elapsed Wall-Sec, NOT `/proc` CPU Time)

| Arm (`Confirmation, 80 Episodes`) | Pre-LLM (`sec`) | Post-LLM (`sec`) | Localization (`sec`) | `2x2` Decomp (`sec`) | Empirical Coverage (`sec`) | Official Verifier (`sec`) | Total Container Elapsed (`sec`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicConditionalTable_Online` | `505.59` | `441.99` | `0.00` | `0.00` | `1,569.44` | `415.33` | `2,932.35` |
| `Flat_PlainModuleRegistry_Online` | `519.81` | `340.27` | `41.12` | `205.65` | `376.69` | `422.88` | `1,906.42` |
| **`Hegel_SelectiveConditionExpansion_Online`** | **`520.76`** | **`293.08`** | **`34.64`** | **`205.83`** | **`259.56`** | **`422.44`** | **`1,736.30`** |

Because `Monolithic` accumulates `O(R^2)` pairwise empirical coverage checks across unfactored `(U, D)` cells (`1,569.44s`), `Hegel` (`1,736.30s` total) simultaneously lowers LLM cost **and** lowers total container probe elapsed wall-time by **`40.79%`** vs `Monolithic` and **`8.92%`** vs `Plain`.

### 5.2 Summary of Sealed `Gate 3` Dev Calibration (`n = 4`, `971000..974000`, `40` Episodes/Arm)

- `Monolithic`: `38/40` `strict_pass` (`9/40` `0`-LLM, `8,387,547` `Scope-1 v2`).
- `Plain`: `37/40` `strict_pass` (`15/40` `0`-LLM, `6,841,612` `Scope-1 v2`).
- `Hegel`: `37/40` `strict_pass` (`18/40` `0`-LLM, `6,164,083` `Scope-1 v2`, `K* = 4` vs `Monolithic`, `K* = 9` vs `Plain`, `8/8` plain-dict candidate-SHA parity).

### 5.3 Remaining Open Frontiers

1. **Open-Topology Multi-Stage / Unknown-Boundary Decomposition**: `M10.0` supplies the two-stage coordinate contract `(calibrate_upstream, solve_downstream)` symmetrically across modular arms; discovering unknown graph depth, arity, or module boundaries from raw code traces without a prescribed split remains open.
2. **Finite-Witness Coverage Scope**: Empirical coverage receipts (`EmpiricalCoverageReceipt`) and guard partitions remain bounded by finite observed campaign witnesses rather than universal symbolic equivalence proofs.
3. **Legacy `R15` Beta KKT Public Adoption**: `R14` is retired; legacy `R15` KKT adoption remains documented as open in `docs/55b_*`.
