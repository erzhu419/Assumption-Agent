# Hegel Machine `M9.9.1` Audit Closure, `M10.0` Reliable Modular Amortization, and `M10.0.1` Calibration Errata Report

> **Deterministic Generation Notice (`Q05 / Q12 / T01–T11`)**: Generated strictly from sealed SHA-256-verified JSON ledgers (`8` files in `m9_9_1_audit_closure_and_forensic_reverification_20261128`, `3946` files in `m10_0_dev_economic_calibration_20261128`, `7880` files in `m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_20261128`, and `44` files in `m10_0_1_audit_closure_and_probe_reverification_20261128`) with zero hand-edited metrics.

---

## 1. Executive Summary & Calibrated Claim Boundary (`Q09 / Q10 / T01–T04`)

Following the independent GPT-6 audits of `M9.9` (`d4b1cd7f0`) and `M10.0` (`9a83c2751`), this report records strictly separated forensic, prospective, and errata layers without mutating any sealed historical archive:
1. **`M9.9.1` Forensic Closure (`0` Paid LLM Calls)**: Re-executed all `30/30` historical off-diagonal `M9.9` decomposition pairs and `7/7` withdrawal counterexamples in `hegel-clean-scipy:m7v1`; verified `bind_alpha_m99_c04_up_b02 ∘ dn_b03` on `alpha_m99_c04/E7` (`strict_pass = True`, diagnostic only); isolated `alpha_m99_c02/E7` (`0.071267 -> 0.004857`) to missing pooled `ref_coupons` standardization/imputation; and recovered cancelled `M9.9` dev runs (`333` total calls = `327` completed + `6` `in_progress` calls, `8,220,922 + y` `proxy_v2`) into the `53,904,542 + x + y` full ledger (`T10`, see `docs/55b_*`, `docs/55c_*`). See Section 6 for itemized `Q01–Q14` closure status (`T08`).
2. **`M10.0` Prospective `3 × 3` Bidirectional Modular Amortization (`n = 4` Dev Calibration $\to$ Zero-Code-Change Freeze $\to$ Single-Shot `n = 8` Unexposed Confirmation)**:
   - **Scoped Mechanism Claim (`Q09 / Q10`)**: Under a **prescribed two-stage interface** (`calibrate_upstream`, `solve_downstream`), `Hegel_SelectiveConditionExpansion_Online` tests **counterexample-driven selective condition-set expansion (`dn_obs -> up_obs + dn_obs`) and bidirectional separability withdrawal** across `{U0, U1, U2} × {D0, D1, D2}` (`U1 ⊗ D2 -> D2_coupled`), balanced `50% / 50%` between `Order A (sep_first)` and `Order B (coupled_first)`, with symmetric `B1/B2/B3` upgrades across modular arms.
   - **Plain-Dict Algorithmic Equivalence (`Q10`)**: On the frozen `step_08` prefix across all confirmation clusters, a plain Python dictionary evaluating the same learned modules, joint `15`-observable vectors, and Kleene `all_of` rules reproduces **`16/16` (`100.0%`)** of `Hegel`'s composed candidate SHA-256 hashes (`8/8 = 100.0%` on `dev`). Gains belong to the **selective condition-expansion & withdrawal algorithm + symmetric gates**, NOT the `PersistentMechanismGraph` container class or open-topology DAG discovery.
3. **`M10.0.1` Probe-Persistence & Statistical Calibration Errata (`T01–T11`, `0` Paid LLM Calls)**:
   - Explicitly records that `M10.0`'s original runner omitted raw JSON persistence for **`41/216`** confirmation witnesses (`32 pprobe + 9 uprobe`, including `8/8` withdrawal `prior_positive` witnesses) and **`19/107`** dev witnesses (`T01`).
   - All **`41/41`** missing confirmation probes (including **`8/8`** withdrawal `prior_positive` witnesses) were post-hoc replayed in real Docker containers (`raw_measurement_status = "backfilled_posthoc_replay"`) and matched their recorded verdicts `100%`; future runs enforce atomic `ProbeReceiptStore` persistence + strict 3-way external FK (`docs/56b_*`).

### Headline `M10.0` Single-Shot Confirmation Results (`n = 8` Independent Unexposed Clusters, `80` Stream Episodes/Arm, `x = 0` Unknown Attempts)

| Arm (`Equal Tools, Same Sandbox & Guard Compiler`) | Official `strict_pass` (`80` eps) | `0`-LLM Passed Closures | Solver Turns | `Scope-1` Calls | `Scope-1` `proxy_v2` | `Scope-Total` (`+Stage S`) `proxy_v2` | Container Probe + Verifier Elapsed (`sec`) | Verified Withdrawals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicConditionalTable_Online` | `72/80` (`90.00%`) | `15/80` | `762` | `823` | `17,006,487` | `18,454,099` | `2,932.35s` | `0` |
| `Flat_PlainModuleRegistry_Online` (`B1+B2+B3`) | `74/80` (`92.50%`) | `29/80` | `587` | `635` | `12,592,301` | `14,039,913` | `1,906.42s` | `0` |
| **`Hegel_SelectiveConditionExpansion_Online` (`B1+B2+B3`)** | **`77/80` (`96.25%`)** | **`37/80`** | **`503`** | **`560`** | **`11,340,259`** (`-33.32%` vs Mono / `-9.94%` vs Plain) | **`12,787,871`** (`-30.70%` / `-8.92%`) | **`1,736.30s`** | **`8`** |

Calibrated statistical takeaways on the `n = 8` unexposed confirmation split (`981000..988000`, frozen commit `042ab4299`, zero `.py` changes after `dev`):
- **Quality Non-Inferiority (`delta_NI = -5.00pp`, `T02` — NO Statistical Quality Superiority Claim)**:
  - `Hegel` achieved **`77/80` (`96.25%`)**, numerically above `Flat_MonolithicConditionalTable_Online` (`72/80 = 90.00%`, diff **`+6.25pp`**, unrounded 95% bootstrap CI **`[-1.25pp, +13.75pp]`**, small-sample paired `t(df=7)` 95% CI **`[-3.68pp, +16.18pp]`**, two-sided sign-flip `p = 0.3125`) and `Flat_PlainModuleRegistry_Online` (`74/80 = 92.50%`, diff **`+3.75pp`**, unrounded 95% bootstrap CI **`[-3.75pp, +10.00pp]`**, paired `t(df=7)` 95% CI **`[-5.12pp, +12.62pp]`**, two-sided sign-flip `p = 0.53125`).
  - **Calibrated reading**: Both point estimates and 95% bootstrap lower bounds (`-1.25pp`, `-3.75pp`) clear `-5.00pp` (non-inferiority vs `Monolithic` also clears under `t(df=7)` at `-3.68pp`; vs `Plain` the `t(df=7)` lower bound is `-5.12pp`, `0.12pp` below `-5.00pp`, so `t(df=7)` sensitivity does NOT clear `-5.00pp`). Because both intervals span `0` and `p = 0.3125 / 0.53125`, we claim **numerical advantage and bootstrap non-inferiority**, NOT statistically significant quality superiority. Note (`T03`): standalone pre-confirmation CI decision-rule JSON files began in `M10.1` (`m10_1_preregistered_analysis_protocol.json`); for `M10.0` the pre-confirmation CI rule location is marked `unverified_prior_to_m10_1_protocol`.
- **Decisive Cost Amortization vs Monolithic Primary Benchmark (`8/8` Clusters, `p = 0.0078125`)**: `Hegel` reduced `Scope-1` `proxy_v2` on **`8/8` (`100%`)** clusters (`Alpha` `-27.96%`, `p = 0.125`; `Beta` `-38.66%`, `p = 0.125`; combined exact two-sided **`p = 0.0078125`**), saving **`5,666,228` `proxy_v2` (`-33.32%`)** with finite 10-step pooled `K* = 3`.
- **Modest Pooled Cost Reduction vs Equal-Tool Plain Modular Registry (`T03` Family Concentration Disclosed)**: `Hegel` saved **`1,252,042` `proxy_v2` (`-9.94%`)** vs `Plain` (`6/8` clusters cheaper, pooled exact two-sided **`p = 0.21875` — NOT statistically significant at `alpha = 0.05`**). Family decomposition shows this pooled gain is heavily concentrated in **Beta** (`-1,198,137`, **`-18.66%`**, `4/4` clusters, **`95.69%`** of total `H-Plain` savings), whereas **Alpha** is near break-even (`-53,905`, **`-0.87%`**, `2/4` clusters). Meanwhile, `0`-LLM passed closures increased from `29/80` to **`37/80`** (`7/8` clusters positive, exact two-sided `p = 0.0546875`; vs Monolithic `15/80`, `8/8` positive, `p = 0.0078125`).
- **Step-08 Constrained Routing Ceiling**: On all `16` colliding `D2` revisit episodes (`S09, S10`), `hegel_selective_condition_expansion` achieved **`16/16` (`100.0%`)** `0`-LLM private `strict_pass`, vs **`8/16` (`50.0%`)** for `dn_only_fixed_prior_d2` and **`8/16` (`50.0%`)** for `dn_only_fixed_latest_d2`.

---

## 2. Execution Provenance, Clocks, & Full R&D Cost Ledger (`C1–C2 / Q05 / Q12 / T10–T11`)

### 2.1 Immutable Start/Completion/Ablation Provenance (`0` Post-Dev Source Edits)

| Split | `execution_run_id` | Start Commit | Dirty at Start? | `.py` Source Tree SHA-256 (`src+scripts+tests`) | Tree Unchanged During Run? | Monotonic Elapsed (`sec`, `T11`) | Ablation Elapsed (`sec`) | Concurrency (`cap / peak`) | Unknown Attempts |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `dev` (`n=4`) | `m100_dev_20261128T021932Z_f5bac8c4` | `1697f4d04` | `False` | `8b13f56969f8219f...` | `True` | `3,877.17s` | `103.66s` | `32 / 12` | `0` |
| `confirmation` (`n=8`) | `m100_confirmation_20261128T032437Z_6dc6e01a` | `042ab4299` | `False` | `8b13f56969f8219f...` | `True` | `3,824.73s` | `245.27s` | `32 / 24` | `0` |

- **Clock Domain Clarification (`T11`)**: `execution_elapsed_wall_sec` is measured via `time.monotonic()` (`3,877.17s` dev, `3,824.73s` confirmation); UTC timestamps (`utc_started_at`, `utc_ended_at`) are recorded separately (`3,630.54s` dev UTC span, `3,570.28s` confirmation UTC span; historical `254.45s` WSL wall-clock vs monotonic drift on `M10.0` confirmation is preserved transparently as unadjusted host clock skew rather than overwritten).
- **Complete `M10.0` Paid R&D Ledger (`0` Cancelled Runs)**:
  - `M10.0` `dev` (`n = 4`, `124` episodes): `1,028` calls, `26,451,981` total tokens, **`22,295,301` `proxy_v2`** (`0` unknown attempts).
  - `M10.0` `confirmation` (`n = 8`, `248` episodes): `2,096` calls, `52,977,017` total tokens, **`42,386,659` `proxy_v2`** (`0` unknown attempts).
  - `M10.0` Grand Total (`dev + confirmation`): **`3,124` calls / `64,681,960` exact `proxy_v2`**.
  - Historical `M9.9` Full Ledger (`T10`): `53,904,542 + x + y` `proxy_v2` across `M9.9` dev/confirm + recovered cancelled dev logs (`333` total = `327` completed + `6` `in_progress` calls).

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

## 3. Cluster-by-Cluster Confirmation Breakdown & Paired Statistical Inference (`C3 / Q06 / T02–T03`)

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

### 3.2 Paired Cluster-Level Exact Two-Sided Sign-Flip, Unrounded Bootstrap, & `t(df=7)` Inference (`n = 8`)

| Comparison (`Hegel` minus Baseline) | Mean Cluster `Scope-1 v2` Diff (95% Boot CI) | Cost Exact 2-Sided `p` (`All / Alpha / Beta`) | `sep_first` vs `coupled_first` Mean `v2` Diff | Quality Rate Diff (Unrounded Boot 95% CI / `t(df=7)` 95% CI; `delta_NI = -5.00pp`) | Quality Exact 2-Sided `p` | `0`-LLM Closure Cluster Diffs & Exact 2-Sided `p` |
|---|---|---|---|---|---:|---|
| **`Hegel - Monolithic`** | **`-708,278.50`** (`[-848,648.82, -555,651.29]`) | **`0.0078125`** (`0.125` / `0.125`) | `-663,071.00` / `-753,486.00` | **`+6.25pp`** (`[-1.25pp, +13.75pp]` / `[-3.68pp, +16.18pp]`, **NI PASS**) | `0.3125` | `[3.0, 3.0, 2.0, 3.0, 3.0, 3.0, 3.0, 2.0]`, **`p = 0.0078125`** |
| **`Hegel - Plain`** | **`-156,505.25`** (`[-340,623.00, 77,101.88]`) | `0.21875` (`1.0` / `0.125`) | `-128,978.25` / `-184,032.25` | **`+3.75pp`** (`[-3.75pp, +10.00pp]` / `[-5.12pp, +12.62pp]`, **Boot NI PASS / `t` borderline**) | `0.53125` | `[1.0, 1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 1.0]`, **`p = 0.0546875`** |

---

## 4. Cumulative Amortization Curves $\Delta C(K)$ (`K = 1..10`) & Finite-Horizon Break-Even $K^*$ (`D2 / T04`)

> **Finite-Horizon `K*` Definition (`T04`)**: `K*` is defined strictly over the observed `10`-step horizon as the smallest step $K \in \{1..10\}$ such that cumulative pooled cost difference $\Delta C(k) < 0$ for every $k \in \{K..10\}$ AND cumulative quality at $K=10$ meets `delta_NI >= -0.05`. It does NOT assume perpetual savings beyond $K = 10$ without recurring revisits.

### 4.1 `Hegel` minus `Flat_MonolithicConditionalTable_Online` (`Pooled K* = 3`)

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

### 4.2 `Hegel` minus `Flat_PlainModuleRegistry_Online` (`Pooled K* = 9`)

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

Notice the exact `T04` payback mechanism: through `K = 8`, `Hegel` carries a pooled upfront selective-expansion investment of **`+1,208,895` `proxy_v2`** over `Plain`; payback occurs specifically at **`K = 9` and `K = 10`** when colliding-`dn_obs` revisits (`S09, S10`) arrive!

### 4.3 Per-Cluster `K*` & Right-Censoring Table (`T04`)

| World ID | Family | Order Mode | Cluster `K*` vs `Mono` | Cluster `K*` vs `Plain` | `Delta C_plain(K=8)` | `Delta C_plain(K=10)` |
|---|---|---|---:|---:|---:|---:|
| `alpha_m100_c01` | `alpha` | `sep_first` | `4` | `1` | `-184,131` | `-380,186` |
| `alpha_m100_c02` | `alpha` | `coupled_first` | `3` | `1` | `-12,916` | `-211,585` |
| `alpha_m100_c03` | `alpha` | `sep_first` | `7` | `not_reached` | `779,376` | `513,754` |
| `alpha_m100_c04` | `alpha` | `coupled_first` | `2` | `not_reached` | `229,700` | `24,112` |
| `beta_m100_c01` | `beta` | `sep_first` | `3` | `1` | `-8,534` | `-215,474` |
| `beta_m100_c02` | `beta` | `coupled_first` | `2` | `1` | `-66,205` | `-481,473` |
| `beta_m100_c03` | `beta` | `sep_first` | `4` | `9` | `80,830` | `-434,007` |
| `beta_m100_c04` | `beta` | `coupled_first` | `3` | `10` | `390,775` | `-67,183` |

---

## 5. `B2`/`B3` Execution Funnel (`T07`), Container Wall-Time (`Q11`), & `Gate 3` Dev Summary

### 5.1 `B2` & `B3` Observed Trigger Funnel on `M10.0` Confirmation (`T07`)

| Arm | `B2` Alt Direct Probes | `B2` Alt Joint Probes | `B2` Direct Pre-LLM Rescues | `B3` Raw Archive Replays | `B3` Lazy Projections |
|---|---:|---:|---:|---:|---:|
| `Flat_PlainModuleRegistry_Online` | `10` | `1` | `0` | `0` | `0` |
| `Hegel_SelectiveConditionExpansion_Online` | `9` | `0` | `0` | `0` | `0` |

- **`B2` Interpretation**: On confirmation, `B2` probed an alternate verified upstream sibling (`10` times on `Plain`, `9` times on `Hegel`), promoted `1` alternate upstream into a joint probe on `Plain` and `0` on `Hegel`, and achieved `0` direct pre-LLM rescues (diagnostic localization only).
- **`B3` Interpretation**: Because initial `2×2` decomposition succeeded on `16/16` (`100%`) confirmation modular streams (`8/8` `Plain`, `8/8` `Hegel`), `B3` had **`0` natural triggers** in `dev` and `confirmation`; its correctness is verified via dedicated synthetic unit tests (`test_m100_b2_sibling_upstream_fallback_and_b3_deferred_lazy_projection` and `test_m10_0_1_audit_closure_suite.py`).

### 5.2 Non-LLM Container Probe & Verifier Elapsed Wall-Time (`OMP_NUM_THREADS=1`, Elapsed Wall-Sec)

| Arm (`Confirmation, 80 Episodes`) | Pre-LLM (`sec`) | Post-LLM (`sec`) | Localization (`sec`) | `2x2` Decomp (`sec`) | Empirical Coverage (`sec`) | Official Verifier (`sec`) | Total Container Elapsed (`sec`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicConditionalTable_Online` | `505.59` | `441.99` | `0.00` | `0.00` | `1,569.44` | `415.33` | `2,932.35` |
| `Flat_PlainModuleRegistry_Online` | `519.81` | `340.27` | `41.12` | `205.65` | `376.69` | `422.88` | `1,906.42` |
| **`Hegel_SelectiveConditionExpansion_Online`** | **`520.76`** | **`293.08`** | **`34.64`** | **`205.83`** | **`259.56`** | **`422.44`** | **`1,736.30`** |

### 5.3 Summary of Sealed `Gate 3` Dev Calibration (`n = 4`, `971000..974000`, `40` Episodes/Arm)

- `Monolithic`: `38/40` `strict_pass` (`9/40` `0`-LLM, `8,387,547` `Scope-1 v2`).
- `Plain`: `37/40` `strict_pass` (`15/40` `0`-LLM, `6,841,612` `Scope-1 v2`).
- `Hegel`: `37/40` `strict_pass` (`18/40` `0`-LLM, `6,164,083` `Scope-1 v2`, `K* = 4` vs `Monolithic`, `K* = 9` vs `Plain`, `8/8` plain-dict parity).

---

## 6. Itemized `Q01–Q14` & `T01–T11` Audit Closure Status (`T08`)

| ID | Status | Evidence / Resolution Summary |
|---|---|---|
| `Q01 / T01` | **Closed in `M10.0.1`** | `2×2` decomposition executed in `M10.0`; `M10.0.1` adds atomic `ProbeReceiptStore`, migrates all bare `record_tested_witness` calls, enforces 3-way external receipt FK on withdrawals, and post-hoc back-verifies `41/41` missing `M10.0` confirm probes (`docs/56b_*`). |
| `Q02 / B1` | **Closed** | Public pooled `ref_coupons` exchange invariance active symmetrically across all arms (`require_pooled_coupon_check=True`). |
| `Q03 / B2` | **Closed (Calibrated `T07`)** | Symmetric `1`-sibling upstream fallback implemented and dual primary/alt diagnostics preserved; funnel disclosed (`0` direct rescues on confirmation). |
| `Q04 / B3` | **Closed (Test-Verified `T07`)** | Unfactored monolithic fallback + lazy projection implemented symmetrically; `0` natural triggers in `M10.0` (`16/16` immediate decomp), covered by unit tests. |
| `Q05 / T11` | **Closed** | Immutable write-once `execution_manifest.json` + separate `execution_completion.json` & `ablation_manifest.json`; UTC vs monotonic clock fields separated. |
| `Q06 / T02–T03` | **Closed in `M10.0.1`** | Unrounded bootstrap CIs + `t(df=7)` sensitivity intervals, family decomposition (`Alpha -0.87%` vs `Beta -18.66%`), and non-superiority calibration added. |
| `Q07 / T01` | **Closed in `M10.0.1`** | 6-link ledger FK upgraded to 3-way external `ProbeReceiptStore` FK rejecting GPT-6's exact forged `prior_positive.probe_id` negative control. |
| `Q08` | **Closed** | Idempotent legacy migration round-trip verified in `test_m9_9_1_audit_closure_suite.py`. |
| `Q09 / Q10` | **Closed** | Calibrated scope (`SelectiveConditionExpansion` under prescribed 2-stage split) & `16/16` (`100%`) plain-dict candidate-SHA parity reported upfront. |
| `Q11` | **Closed** | Explicit elapsed wall-time (not `/proc` CPU time) reported for all container probe categories. |
| `Q12 / T10` | **Closed** | Full `M9.9` (`53,904,542 + x + y`; `327` completed + `6` `in_progress` cancelled dev calls) + `M10.0` (`64,681,960` exact, `0` unknown) ledgers reconciled. |
| `Q13 / Q14` | **Closed** | Equal-info archive & controlled step-08 ablation (`16/16` vs `8/16`) verified; zero `M9.9` archived folder mutations. |
| `T05 / T06` | **Closed in `M10.0.1`** | Generic Alpha near-zero derivative/pilot consistency check (`0` `log1p` hardcoding) + Beta calibration-to-KKT perturbation check & non-`0.0` failure aggregator fix. |
| `T09` | **Closed in `M10.0.1`** | Mono/Plain/Hegel row IDs and branch tags neutralized (`mono_rXX`, `plain_up_rXX`, `plain_dn_rXX`, `up_branch_XX`, `dn_branch_XX`). |
