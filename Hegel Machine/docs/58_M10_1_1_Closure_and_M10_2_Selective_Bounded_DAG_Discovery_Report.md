# Hegel `M10.1.1` Audit Closure & `M10.2` De-Scaffolded Equal-Probe-Cap Selective Bounded DAG Discovery (`H = 16`)

> **Zero Hardcoded Constants & Strict Source-Tree Freeze (`U01 / U10`)**: `dev` (`n=4, H=16`) and unexposed `confirmation` (`n=8, H=16`) share byte-identical Python `source_tree_sha256` (`4a086673a3bbab4b85d403b24a6ec2c439a76246484743f9204f32c134e2a8a9`), `working_tree_dirty_at_start = False`, `source_private_previously_exposed = False`, `protocol_amended_after_source_results = False`; `dev` `usage_completeness = complete` (`0` unknown), `confirmation` `partial_unknown_usage` (`14` tail retry unknown attempts across `3,309` calls, `K* = unknown_partial_usage`).

## 1. Executive Summary & Three Separate Statistical Verdicts (`H = 16` Primary Endpoint)

1. **Equal-Probe-Cap & De-Scaffolded Single-Factor Design (`U02 / U05 / U06`)**: All 3 arms share `MAX_PRE_LLM_PROBES_PER_STEP = 4`, role-blind `stat_00..stat_17`, de-scaffolded prompts (`0` `panel_a/b/c` splitting recipes), `M10.1.1` `0`-LLM guard satisfiability pre-check, and runtime `_TrackedPayloadDict` edge-consumption & anti-wrapper-shell checks (with public calibration sub-tables retained symmetrically across all 3 arms as pre-registered). `Eager_BoundedDAG_Online` and `Hegel_SelectiveBoundedDAG_Online` differ solely in `evaluate_m102_selective_investment_gate`.
2. **Primary Economic Endpoint (`Hegel_Selective − Flat_Monolithic` at `H = 16`)**: Strict pass **`120/128` vs `125/128`** (Bootstrap 95% CI `[-11.00pp, +1.00pp]`, `t(7)` 95% CI `[-11.77pp, +3.96pp]`); known `Scope-1` cost delta **`+4.75%`** (`+1,117,083 v2`, `3/8` clusters cheaper, exact `p = 0.4531`, Bootstrap 95% CI `[-120070, +448519]`). Pooled verdicts: `cost_superiority_supported = False`, `quality_noninferiority_bootstrap_supported = False`, `finite_observed_break_even_reached = False` (`K* = unknown_partial_usage`). On held-out **Stratum B (`b01..b04`, `64` episodes)**: **`63/64` vs `63/64`** (`quality_noninferiority_bootstrap_supported = True`) with **`-3.49%`** (`-437,377 v2`).
3. **Single-Factor Mechanism Endpoint (`Hegel_Selective − Eager_BoundedDAG` at `H = 16`)**: Strict pass **`120/128` vs `114/128`** (`+4.69pp`, Bootstrap 95% CI `[-4.00pp, +16.00pp]`, `quality_noninferiority_bootstrap_supported = True`); failed graph surgeries cut `7` vs `14`; known `Scope-1` cost delta **`-4.93%`** (`-1,277,984 v2`, `p = 0.8516`; Stratum B **`-13.09%` / `-1,820,758 v2`**, `63/64` vs `56/64`).
4. **Complete `M10.2` R&D Ledger**: Dev `1,540` calls / `37,005,845 v2` (`0` unknown) + Confirmation `3,309` calls / `76,272,930+x v2` (`14` unknown attempts: `Mono=0, Eager=8, Hegel=6`) = **`4,849` calls / `113,278,775+x v2`** (`14` unknown attempts).

## 2. Unexposed Confirmation (`n = 8, H = 16`, `128` Episodes/Arm) Master Table

| Arm | Strict Pass | Solver-Free Pass | **All-Role Zero-Call Pass** | Comp. Reuse | Max(Tot) Pre-Probes | Total Calls | Solver Turns | Scope-1 v2 | stream_solve | dag_prop | dag_synth | dag_failed | dag_merge | guard_ind | Container Sec | Max K/d/fo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicArchive_Online` | **125/128 (97.7%)** | 43/128 | **38/128** | 0 | 4 (346) | 1020 | 943 | **23,536,248** | 20,624,120 | 0 | 0 | 0 | 0 | 2,912,128 | 1,376.68s | 1/1/0 |
| `Eager_BoundedDAG_Online` | **114/128 (89.1%)** | 49/128 | **37/128** | 13 | 4 (367) | 1119 | 953 | **25,931,315** | 0 | 1,319,768 | 13,191,856 | 3,756,943 | 4,965,954 | 2,696,794 | 1,697.80s | 4/3/2 |
| `Hegel_SelectiveBoundedDAG_Online` | **120/128 (93.8%)** | 46/128 | **39/128** | 5 | 4 (366) | 1080 | 928 | **24,653,331** | 4,660,314 | 1,276,275 | 10,908,073 | 2,162,214 | 2,400,950 | 3,245,505 | 1,752.35s | 4/3/2 |

## 3. Nested Horizon Amortization Trajectory (`H = 4, 8, 12, 16`; `H = 16` Primary Endpoint)

### 3.1 `Hegel_SelectiveBoundedDAG_Online − Flat_MonolithicArchive_Online`

| Horizon | Passes (H vs Mono) | Quality Boot 95% CI (`t` 95% CI) | Scope-1 % (Total Δv2) | Cluster Wins | Cost `p` | Cost Mean Boot 95% CI | Cost Sup / Boot NI |
|---|---|---|---:|---:|---:|---|---|
| `H = 4` | `32/32` vs `31/32` | `[+0.00pp, +9.00pp]` (`t`: `[-4.26pp, +10.51pp]`) | **`-8.65%`** (`-576,140`) | `5/8` | `0.5156` | `[-257772, +117541]` | `False` / `True` |
| `H = 8` | `62/64` vs `61/64` | `[-3.00pp, +6.00pp]` (`t`: `[-5.13pp, +8.26pp]`) | **`+10.89%`** (`+1,226,547`) | `3/8` | `0.4297` | `[-113350, +451513]` | `False` / `True` |
| `H = 12` | `93/96` vs `93/96` | `[-4.00pp, +4.00pp]` (`t`: `[-5.27pp, +5.27pp]`) | **`+4.72%`** (`+837,258`) | `4/8` | `0.5312` | `[-154231, +388608]` | `False` / `True` |
| `H = 16` **(Primary)** | `120/128` vs `125/128` | `[-11.00pp, +1.00pp]` (`t`: `[-11.77pp, +3.96pp]`) | **`+4.75%`** (`+1,117,083`) | `3/8` | `0.4531` | `[-120070, +448519]` | `False` / `False` |

### 3.2 `Hegel_SelectiveBoundedDAG_Online − Eager_BoundedDAG_Online`

| Horizon | Passes (H vs Eager) | Quality Boot 95% CI (`t` 95% CI) | Scope-1 % (Total Δv2) | Cluster Wins | Cost `p` | Cost Mean Boot 95% CI | Cost Sup / Boot NI |
|---|---|---|---:|---:|---:|---|---|
| `H = 4` | `32/32` vs `32/32` | `[+0.00pp, +0.00pp]` (`t`: `[+0.00pp, +0.00pp]`) | **`+5.55%`** (`+319,640`) | `3/8` | `0.7812` | `[-187994, +277266]` | `False` / `True` |
| `H = 8` | `62/64` vs `57/64` | `[+0.00pp, +16.00pp]` (`t`: `[-1.76pp, +17.39pp]`) | **`-13.03%`** (`-1,871,405`) | `5/8` | `0.4375` | `[-718647, +157184]` | `False` / `True` |
| `H = 12` | `93/96` vs `86/96` | `[+1.00pp, +16.00pp]` (`t`: `[-2.16pp, +16.74pp]`) | **`-3.36%`** (`-646,952`) | `4/8` | `0.6797` | `[-530825, +318185]` | `False` / `True` |
| `H = 16` **(Primary)** | `120/128` vs `114/128` | `[-4.00pp, +16.00pp]` (`t`: `[-7.72pp, +17.10pp]`) | **`-4.93%`** (`-1,277,984`) | `3/8` | `0.8516` | `[-840276, +337731]` | `False` / `True` |

## 4. Stratified Primary Endpoint (`H = 16`) Inference

| Comparison | Stratum | Passes | Quality Boot 95% CI | Cost % (Total Δv2) | Cost `p` | Cost Mean Boot 95% CI | Cost Sup / Boot NI |
|---|---|---|---|---:|---:|---|---|
| Hegel − Monolithic | Pooled all clusters (n=8) | `120/128` vs `125/128` | `[-11.00pp, +1.00pp]` | **`+4.75%`** (`+1,117,083`) | `0.4531` | `[-120070, +448519]` | `False` / `False` |
| Hegel − Eager DAG | Pooled all clusters (n=8) | `120/128` vs `114/128` | `[-4.00pp, +16.00pp]` | **`-4.93%`** (`-1,277,984`) | `0.8516` | `[-840276, +337731]` | `False` / `True` |
| Eager DAG − Monolithic | Pooled all clusters (n=8) | `114/128` vs `125/128` | `[-18.00pp, -2.00pp]` | **`+10.18%`** (`+2,395,067`) | `0.4375` | `[-213006, +927538]` | `False` / `False` |
| Hegel − Monolithic | Stratum A seen families (n=4) | `57/64` vs `62/64` | `[-19.00pp, +0.00pp]` | **`+14.12%`** (`+1,554,460`) | `0.2500` | `[+40390, +789666]` | `False` / `False` |
| Hegel − Eager DAG | Stratum A seen families (n=4) | `57/64` vs `58/64` | `[-9.00pp, +6.00pp]` | **`+4.51%`** (`+542,774`) | `0.6250` | `[-272818, +621406]` | `False` / `False` |
| Eager DAG − Monolithic | Stratum A seen families (n=4) | `58/64` vs `62/64` | `[-11.00pp, -2.00pp]` | **`+9.19%`** (`+1,011,686`) | `0.2500` | `[+24042, +457638]` | `False` / `False` |
| Hegel − Monolithic | Stratum B1 unseen static topology (n=2) | `32/32` vs `32/32` | `[+0.00pp, +0.00pp]` | **`-1.33%`** (`-94,883`) | `1.0000` | `[-131711, +36828]` | `False` / `True` |
| Hegel − Eager DAG | Stratum B1 unseen static topology (n=2) | `32/32` vs `25/32` | `[+6.00pp, +38.00pp]` | **`-24.96%`** (`-2,347,261`) | `0.5000` | `[-2287611, -59650]` | `False` / `True` |
| Eager DAG − Monolithic | Stratum B1 unseen static topology (n=2) | `25/32` vs `32/32` | `[-38.00pp, -6.00pp]` | **`+31.49%`** (`+2,252,378`) | `0.5000` | `[+96478, +2155900]` | `False` / `False` |
| Hegel − Monolithic | Stratum B2 order/regime switch (n=2) | `31/32` vs `31/32` | `[-6.00pp, +6.00pp]` | **`-6.37%`** (`-342,494`) | `1.0000` | `[-466185, +123691]` | `False` / `False` |
| Hegel − Eager DAG | Stratum B2 order/regime switch (n=2) | `31/32` vs `31/32` | `[+0.00pp, +0.00pp]` | **`+11.69%`** (`+526,503`) | `0.5000` | `[+52524, +473979]` | `False` / `True` |
| Eager DAG − Monolithic | Stratum B2 order/regime switch (n=2) | `31/32` vs `31/32` | `[-6.00pp, +6.00pp]` | **`-16.17%`** (`-868,997`) | `1.0000` | `[-940164, +71167]` | `False` / `False` |
| Hegel − Monolithic | Stratum B combined (n=4) | `63/64` vs `63/64` | `[-5.00pp, +5.00pp]` | **`-3.49%`** (`-437,377`) | `0.6250` | `[-340432, +80260]` | `False` / `True` |
| Hegel − Eager DAG | Stratum B combined (n=4) | `63/64` vs `56/64` | `[+0.00pp, +28.00pp]` | **`-13.09%`** (`-1,820,758`) | `0.7500` | `[-1702577, +340572]` | `False` / `True` |
| Eager DAG − Monolithic | Stratum B combined (n=4) | `56/64` vs `63/64` | `[-30.00pp, +3.00pp]` | **`+11.04%`** (`+1,383,381`) | `0.6250` | `[-681004, +1634717]` | `False` / `False` |
| Hegel − Monolithic | Monolithic negative control (n=1) | `15/16` vs `16/16` | `[-6.00pp, -6.00pp]` | **`-4.08%`** (`-36,840`) | `1.0000` | `[-36840, -36840]` | `False` / `False` |
| Hegel − Eager DAG | Monolithic negative control (n=1) | `15/16` vs `16/16` | `[-6.00pp, -6.00pp]` | **`+8.53%`** (`+68,164`) | `1.0000` | `[+68164, +68164]` | `False` / `False` |
| Eager DAG − Monolithic | Monolithic negative control (n=1) | `16/16` vs `16/16` | `[+0.00pp, +0.00pp]` | **`-11.62%`** (`-105,004`) | `1.0000` | `[-105004, -105004]` | `False` / `True` |
| Hegel − Monolithic | Multi-node non-monolithic (n=7) | `105/112` vs `109/112` | `[-12.00pp, +2.00pp]` | **`+5.10%`** (`+1,153,923`) | `0.4062` | `[-130226, +503397]` | `False` / `False` |
| Hegel − Eager DAG | Multi-node non-monolithic (n=7) | `105/112` vs `98/112` | `[-3.00pp, +18.00pp]` | **`-5.36%`** (`-1,346,148`) | `0.8281` | `[-953956, +375093]` | `False` / `True` |
| Eager DAG − Monolithic | Multi-node non-monolithic (n=7) | `98/112` vs `109/112` | `[-21.00pp, -3.00pp]` | **`+11.05%`** (`+2,500,071`) | `0.4062` | `[-206937, +1052310]` | `False` / `False` |

## 5. Per-Cluster Confirmation Audit Table & Exact `K*`

| World ID | Stratum | Hidden Topology | Mono Pass (v2) | Eager Pass (v2) | **Hegel Pass (v2)** | H−Mono% | H−Eager% | Solver-Free / **All-Zero** | Max K/d/fo | Shared exec=1 | `K*` (vs M / E) |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| `conf_m102_a01` | `stratum_a_seen_topology` | `monolithic_coupled` | 16/16 (903,879) | 16/16 (798,875) | **15/16 (867,039)** | `-4.08%` | `+8.53%` | 13 / **13** | K=1,d=1,fo=0 | — | `not_reached / not_reached` |
| `conf_m102_a02` | `stratum_a_seen_topology` | `chain_k3_variable_cut` | 15/16 (4,008,657) | 14/16 (4,512,756) | **15/16 (4,126,278)** | `+2.93%` | `-8.56%` | 2 / **2** | K=2,d=2,fo=1 | — | `not_reached / 1` |
| `conf_m102_a03` | `stratum_a_seen_topology` | `chain_k3_reversed_cut` | 15/16 (3,483,986) | 13/16 (3,685,399) | **11/16 (4,497,667)** | `+29.10%` | `+22.04%` | 2 / **2** | K=3,d=3,fo=2 | YES | `not_reached / not_reached` |
| `conf_m102_a04` | `stratum_a_seen_topology` | `diamond_k4_shared_prefix` | 16/16 (2,613,706) | 15/16 (3,024,884) | **16/16 (3,073,704)** | `+17.60%` | `+1.61%` | 6 / **4** | K=4,d=3,fo=2 | YES | `not_reached / not_reached` |
| `conf_m102_b01` | `stratum_b1_unseen_static_topology` | `chain_k4_deep_cascade` | 16/16 (3,161,534) | 15/16 (3,258,012) | **16/16 (3,198,362)** | `+1.16%` | `-1.83%` | 6 / **3** | K=2,d=2,fo=1 | — | `not_reached / 15` |
| `conf_m102_b02` | `stratum_b1_unseen_static_topology` | `asymmetric_skip_dag_k4` | 16/16 (3,990,261) | 10/16 (6,146,161) | **16/16 (3,858,550)** | `-3.30%` | `-37.22%` | 3 / **2** | K=4,d=3,fo=2 | YES | `9 / 5` |
| `conf_m102_b03` | `stratum_b2_shift_order_and_regime_switch` | `diamond_k4_coupled_first` | 15/16 (2,845,083) | 16/16 (1,904,919) | **16/16 (2,378,898)** | `-16.39%` | `+24.88%` | 6 / **6** | K=2,d=2,fo=1 | — | `1 / not_reached` |
| `conf_m102_b04` | `stratum_b2_shift_order_and_regime_switch` | `hybrid_mono_to_split_shift` | 16/16 (2,529,142) | 15/16 (2,600,309) | **15/16 (2,652,833)** | `+4.89%` | `+2.02%` | 8 / **7** | K=2,d=2,fo=1 | — | `not_reached / not_reached` |

## 6. Pre-Registered Structural Verification Gates (`U06 / U11`)

| Gate ID | Criterion | Observed | Verdict |
|---|---|---|---|
| `G1_monolithic_economic_non_fragmentation` | On `monolithic_coupled`, `Hegel_SelectiveBoundedDAG_Online` retains `max_K == 1` via 0-LLM investment gate with non-inferior quality vs Monolithic. | conf_m102_a01: max_K=1, passes=15/16 vs Mono 16/16 | **`FAIL`** |
| `G2_descaffolded_multi_stage_and_shared_fanout_discovery` | Under de-scaffolded prompts & `stat_00..stat_17`, discover `K >= 3` on `>= 4` clusters, shared fan-out `exec_count == 1` on `>= 1` cluster, and `>= 1` solver-free compositional reuse pass. | K>=3 clusters=['conf_m102_a03', 'conf_m102_a04', 'conf_m102_b02'] (3), shared_fanout_exec_1=['conf_m102_a03', 'conf_m102_a04', 'conf_m102_b02'] (3), compositional_reuse_passes=5 | **`FAIL`** |
| `G3_strict_semantic_withdrawal_and_runtime_edge_consumption` | 100% traced episodes satisfy `all_exec_count_eq_1 == True` & `all_declared_edges_consumed == True`; 100% issued `CutWithdrawalMergeReceipt` pass v2 4-way semantic FK. | all_traced_exec_1_and_edges_consumed=True, verified_v2_withdrawals=2 | **`PASS`** |
| `G4_step14_cold_hydration_and_plain_dict_parity` | 100% cold-hydration FK verification and 100.0% plain-dict SHA-256 routing parity on frozen Step-14 checkpoints across S15 & S16. | hydrated=8/8, parity=16/16 (100.0%) | **`PASS`** |

## 7. Dev Calibration (`n = 4, H = 16`) Summary

| Arm | Strict Pass | Solver-Free Pass | **All-Role Zero-Call Pass** | Comp. Reuse | Max(Tot) Pre-Probes | Total Calls | Solver Turns | Scope-1 v2 | stream_solve | dag_prop | dag_synth | dag_failed | dag_merge | guard_ind | Container Sec | Max K/d/fo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicArchive_Online` | **60/64 (93.8%)** | 25/64 | **20/64** | 0 | 4 (168) | 477 | 435 | **11,254,822** | 9,724,995 | 0 | 0 | 0 | 0 | 1,529,827 | 638.66s | 1/1/0 |
| `Eager_BoundedDAG_Online` | **61/64 (95.3%)** | 24/64 | **19/64** | 5 | 4 (186) | 531 | 446 | **12,615,482** | 0 | 934,332 | 7,390,166 | 969,120 | 1,646,342 | 1,675,522 | 844.62s | 4/4/2 |
| `Hegel_SelectiveBoundedDAG_Online` | **59/64 (92.2%)** | 27/64 | **21/64** | 3 | 4 (177) | 490 | 417 | **11,849,513** | 2,273,021 | 703,721 | 4,918,832 | 952,326 | 1,391,541 | 1,610,072 | 753.23s | 3/3/2 |
