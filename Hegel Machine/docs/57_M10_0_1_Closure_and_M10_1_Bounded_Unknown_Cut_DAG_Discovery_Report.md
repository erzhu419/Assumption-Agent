# Hegel `M10.0.1` Closure, `M10.1` Panel-Assisted Bounded DAG Discovery, and `M10.1.1` Calibrated Report (`U01–U12`)

> **Zero Hardcoded Derived Constants (`U10`)**: Every percentage, `K*`, CI, dual zero-call count, and structural gate verdict is computed directly from sealed JSONs with consistency assertions.

## 1. Calibrated Executive Summary & Claim Scope (`U01 / U06 / U07 / U08`)

1. **Provenance (`U01`)**: `source_private_previously_exposed = True`, `protocol_amended_after_source_results = True`, `downstream_streams_previously_exposed = False` (`0/192` stream episodes run at Stage S abort; prospective online stream validation frozen at `cf1ace15e`).
2. **Prompt & Probe Scope (`U06`)**: Eliminates fixed `(calibrate_upstream, solve_downstream)` slots and verifies `K=3/4` DAGs, `a04` shared-prefix `exec_count==1`, and `5/5` solver-free unseen node compositions, under calibration-panel and prompt decomposition scaffolding (`pre-LLM probe cap = 2` Mono vs `5` modular). De-scaffolded equal-probe-cap evaluation is isolated in `M10.2`.
3. **Statistical Verdicts (`U07 / U08`)**: `Hegel` `61/64 (95.3%)` vs `Mono` `60/64 (93.8%)` and `Fixed2` `60/64 (93.8%)`; `solver_free_successes = 19 / 18 / 22`, **`all_role_call_free_successes = 17 / 10 / 11`**. Vs `Mono`: `+3.32%` (`p = 0.8281`), quality bootstrap 95% CI `[-6.00pp, +9.00pp]` (`< -5.00pp`, non-inferiority NOT established, `K* = not_reached`). Vs `Fixed2`: `-15.32%` (`p = 0.1484`), quality bootstrap 95% CI `[-3.00pp, +6.00pp]` (`t(7)` lower `-5.13pp`, `K* = 4`). Grand R&D ledger: `2,797` calls / `61,477,706 v2` (`0` unknown).

## 2. Confirmation (`n = 8`) 3-Arm Master Table (`U08`)

| Arm | Strict Pass | Solver-Free Pass | **All-Role Zero-Call Pass** | Total Calls | Solver Turns | Scope-1 v2 | stream_solve | dag_prop | dag_synth | dag_failed | dag_merge | guard_ind | Container Sec | Max K/d/fo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicArchive_Online` | **60/64 (93.8%)** | 19/64 | **17/64** | 546 | 491 | **12,076,813** | 11,094,551 | 0 | 0 | 0 | 0 | 982,262 | 529.36s | 1/1/0 |
| `Fixed_TwoStage_Modular_Online` | **60/64 (93.8%)** | 18/64 | **10/64** | 671 | 517 | **14,735,484** | 2,056,530 | 802,806 | 6,597,620 | 1,055,805 | 2,231,646 | 1,991,077 | 981.00s | 2/2/1 |
| `Hegel_UnknownCut_GraphDiscovery_Online` | **61/64 (95.3%)** | 22/64 | **11/64** | 581 | 440 | **12,477,436** | 1,893,758 | 681,869 | 4,481,736 | 723,732 | 2,935,017 | 1,761,324 | 898.96s | 4/3/2 |

## 3. Stratified Paired Cluster Inference (`U07 / U10`)

| Comparison | Stratum | Passes (H vs Base) | Quality Boot 95% CI | Cost % Change (Total Δv2) | Cost `p` | Cost Mean Boot 95% CI | Cost Sup / Boot NI |
|---|---|---|---|---:|---:|---|---|
| Hegel − Monolithic | Pooled all clusters (n=8) | `61/64` vs `60/64` | `[-6.00pp, +9.00pp]` | `+3.32%` (`+400,623`) | `0.8281` | `[-358642, +454756]` | `False` / `False` |
| Hegel − Fixed2 | Pooled all clusters (n=8) | `61/64` vs `60/64` | `[-3.00pp, +6.00pp]` | `-15.32%` (`-2,258,048`) | `0.1484` | `[-602847, +21359]` | `False` / `True` |
| Hegel − Monolithic | Stratum A seen families (n=4) | `30/32` vs `29/32` | `[-9.00pp, +19.00pp]` | `-14.06%` (`-859,324`) | `0.5000` | `[-710346, +280684]` | `False` / `False` |
| Hegel − Fixed2 | Stratum A seen families (n=4) | `30/32` vs `30/32` | `[-9.00pp, +9.00pp]` | `-10.25%` (`-600,083`) | `0.5000` | `[-486101, +186060]` | `False` / `False` |
| Hegel − Monolithic | Stratum B unseen topology/order (n=4) | `31/32` vs `31/32` | `[-9.00pp, +9.00pp]` | `+21.12%` (`+1,259,947`) | `0.3750` | `[-199074, +829048]` | `False` / `False` |
| Hegel − Fixed2 | Stratum B unseen topology/order (n=4) | `31/32` vs `30/32` | `[+0.00pp, +9.00pp]` | `-18.66%` (`-1,657,965`) | `0.5000` | `[-880601, +51618]` | `False` / `True` |
| Hegel − Monolithic | Monolithic negative control (n=1) | `8/8` vs `6/8` | `[+25.00pp, +25.00pp]` | `+45.36%` (`+435,032`) | `1.0000` | `[+435032, +435032]` | `False` / `False` |
| Hegel − Fixed2 | Monolithic negative control (n=1) | `8/8` vs `7/8` | `[+12.00pp, +12.00pp]` | `+1.28%` (`+17,675`) | `1.0000` | `[+17675, +17675]` | `False` / `False` |
| Hegel − Monolithic | Multi-node non-monolithic (n=7) | `53/56` vs `54/56` | `[-7.00pp, +4.00pp]` | `-0.31%` (`-34,409`) | `0.9844` | `[-455120, +444035]` | `False` / `False` |
| Hegel − Fixed2 | Multi-node non-monolithic (n=7) | `53/56` vs `53/56` | `[-5.00pp, +5.00pp]` | `-17.04%` (`-2,275,723`) | `0.1406` | `[-673408, +17248]` | `False` / `True` |

## 4. Per-Cluster Audit Table & Exact `K*`

| World ID | Stratum | Hidden Topology | Mono Pass (v2) | Fixed2 Pass (v2) | **Hegel Pass (v2)** | H−Mono% | H−Fixed2% | Solver-Free | Max K/d/fo | Shared exec=1 | `K*` (vs M / F) |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| `conf_m101_a01` | `stratum_a_seen_topology` | `monolithic_coupled` | 6/8 (959,148) | 7/8 (1,376,505) | **8/8 (1,394,180)** | `+45.36%` | `+1.28%` | 4/8 | K=2,d=2,fo=1 | — | `not_reached / not_reached` |
| `conf_m101_a02` | `stratum_a_seen_topology` | `chain_k3_variable_cut` | 7/8 (1,890,301) | 7/8 (1,697,377) | **6/8 (1,248,092)** | `-33.97%` | `-26.47%` | 1/8 | K=3,d=3,fo=2 | YES | `not_reached / not_reached` |
| `conf_m101_a03` | `stratum_a_seen_topology` | `chain_k3_reversed_cut` | 8/8 (1,480,017) | 8/8 (1,251,909) | **8/8 (1,606,353)** | `+8.54%` | `+28.31%` | 3/8 | K=3,d=3,fo=2 | YES | `not_reached / not_reached` |
| `conf_m101_a04` | `stratum_a_seen_topology` | `diamond_k4_shared_prefix` | 8/8 (1,782,275) | 8/8 (1,526,709) | **8/8 (1,003,792)** | `-43.68%` | `-34.25%` | 4/8 | K=4,d=3,fo=2 | YES | `2 / 1` |
| `conf_m101_b01` | `stratum_b_unseen_topology` | `chain_k4_deep_cascade` | 8/8 (1,747,227) | 7/8 (2,369,553) | **8/8 (1,272,945)** | `-27.14%` | `-46.28%` | 3/8 | K=3,d=3,fo=2 | YES | `1 / 5` |
| `conf_m101_b02` | `stratum_b_unseen_topology` | `asymmetric_skip_dag_k4` | 8/8 (1,464,644) | 7/8 (3,037,549) | **7/8 (2,372,955)** | `+62.02%` | `-21.88%` | 2/8 | K=3,d=3,fo=1 | — | `not_reached / 6` |
| `conf_m101_b03` | `stratum_b_unseen_topology` | `diamond_k4_coupled_first` | 8/8 (1,204,055) | 8/8 (1,868,740) | **8/8 (1,953,840)** | `+62.27%` | `+4.55%` | 3/8 | K=4,d=3,fo=2 | YES | `not_reached / not_reached` |
| `conf_m101_b04` | `stratum_b_unseen_topology` | `hybrid_mono_to_split_shift` | 7/8 (1,549,146) | 8/8 (1,607,142) | **8/8 (1,625,279)** | `+4.91%` | `+1.13%` | 2/8 | K=4,d=3,fo=2 | YES | `not_reached / not_reached` |

## 5. Pre-Registered Structural Verification Gates (`U06`)

| Gate ID | Criterion | Observed | Verdict |
|---|---|---|---|
| `monolithic_k1_non_fragmentation` | Retain K=1 on monolithic_coupled (`conf_m101_a01`) without decomposing | strict_pass=8/8, max_active_nodes_observed=2 (adopted K=2 on S5/S7/S8) | **`FAIL`** |
| `multi_stage_unknown_cut_discovery_and_reuse` | Discover K>=3 on K=3/K=4 cascades and pass unseen combinations without node re-synthesis | K>=3 discovered on a02/a03/b01; 5/5 solver-free unseen node compositions passed (with paid guard maintenance calls; prompt had panel scaffolding) | **`PARTIAL_PASS_WITH_SCAFFOLDING`** |
| `shared_prefix_diamond_and_skip_exec_count_1` | Verify max_fanout==2 and exec_count==1 on BOTH diamond_k4 (`a04`) and asymmetric_skip_dag_k4 (`b02`) | a04: max_fanout=2, shared_exec_1=True; b02: max_fanout=1, shared_exec_1=False | **`FAIL`** |
| `cut_withdrawal_and_old_domain_preservation` | 4-link external FK withdrawal receipt + old separable domain preserved on colliding revisits | 2/6 Hegel tickets code-identical (`a02/S2, a04/S6`), 4/6 had prior/fail code-version mismatch; `a02/S7` revisit missed private threshold due to empty-guard tiebreak | **`FAIL`** |
| `plain_dict_substrate_parity_ablation` | 100% candidate SHA-256 routing parity on frozen Step-06 checkpoints | 16/16 (100.0%) | **`PASS`** |
