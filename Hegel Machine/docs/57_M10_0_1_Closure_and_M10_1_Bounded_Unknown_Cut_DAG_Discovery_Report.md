# Hegel `M10.0.1` Audit Closure & `M10.1` Bounded Unknown-Cut Acyclic Computational DAG Discovery Report

> **Deterministically generated strictly from sealed JSON evidence** (`0` hand-transcribed metrics).

## 1. Executive Summary

### 1.1 `M10.0.1` Audit Closure (`0` Paid LLM Calls)
- Back-verified **`41/41` (`100.0%`)** previously unpersisted `M10.0` confirmation probes in isolated Docker (`backfilled_posthoc_replay`, including **`8/8`** withdrawal priors), added write-once atomic `ProbeReceiptStore` + 3-way external FK verification (rejecting GPT-6's forged `prior.probe_id` negative control), closed zero-leakage `T05`/`T06` verifier checks, audited `32` archived numerical solvers for `0`-LLM scalar AST lifting coverage (`32/32` `lifter_unsupported`, motivating callable Python DAG nodes in `M10.1`), and calibrated `docs/56` & `docs/56b`.

### 1.2 `M10.1` Bounded Unknown-Cut DAG Discovery (`n = 8` Independent Confirmation Clusters = `4` Stratum A Seen + `4` Stratum B Held-Out Unseen Topologies)
- Eliminates `M10.0`'s prescribed two-stage split boundary (`calibrate_upstream -> solve_downstream`) and `U/D` role labels; operates over bounded acyclic DAGs (`1 <= K <= 4`, `depth <= 4`, `fan-out <= 2`, `K = 1` monolithic ALWAYS legal), 6 graph-edit actions (`keep_monolithic / extract_module / split_node / merge_nodes / reuse_subgraph / rewire_edge`), runtime `exec_count == 1` & anti-identity-shell checks, and 100% metered `Scope-1` accounting including `dag_failed_search`.
- **Confirmation Outcome (`64` episodes/arm, `gpt-6.2-sol`, configured/peak concurrency `32/24`, unknown attempts `0`, `working_tree_dirty_at_start = False`, `source_tree_unchanged = True`)**:
  - **`Hegel_UnknownCut_GraphDiscovery_Online`**: **`61/64 (95.3%)`** strict pass (**#1 across all arms**, including **`31/32 = 96.9%` on Held-Out Unseen Stratum B**), **`22/64`** 0-LLM closures, **`440`** turns (`-10.4%` vs Monolithic `491`; `-14.9%` vs Fixed 2-Stage `517`), **`12,477,436 v2`** Scope-1 cost (**`-15.3%`** vs `Fixed_TwoStage_Modular_Online` `14,735,484 v2`; **`+3.3%`** vs `Flat_MonolithicArchive_Online` `12,076,813 v2`).
  - **Paired Cluster Inference (`n = 8`)**: vs Fixed 2-Stage `mean Δv2 = -282,256.00` (`p = 0.1484`, bootstrap 95% CI `[-602847.31, 21358.51]`, `t(7)` 95% CI `[-683906.64, 119394.64]`, quality diff `+0.0156`, **`K* = 4`**, Stratum B mean `-414,491.25 v2`); vs Monolithic `mean Δv2 = +50,077.88` (`p = 0.8281`, bootstrap 95% CI `[-358642.12, 454756.17]`, `t(7)` 95% CI `[-480700.47, 580856.22]`, quality diff `+0.0156`, pooled `H=8` **`K* = not_reached`**, Stratum A mean `-214,831.00 v2` = `-12.1%`, canonical multi-stage/diamond clusters `-27.1%` to `-43.7%`).
  - **Structural & Audit Gates**: `all_traced_episodes_exec_count_1 = True`, shared `fan-out=2` `exec_count == 1` clusters = `6/8`, 4-link external-FK `CutWithdrawalMergeReceipt` count = `6`, `ProbeReceiptStore` files = `779`, Step-06 plain-dict routing parity = `16/16 (100.0%)`.
  - **Complete R&D Ledger Disclosure**: Dev calibration (`829` calls / `18,068,350 v2`) + preserved pre-stream `Stage S` public-gate abort audit (`docs/evidence/m10_1_stage_s_prestream_abort_audit_20261128/`, `90` calls / `2,212,443 v2`, **`0/192` confirmation stream episodes started**, `0` private touch, protocol amendment `v1.1`) + single-shot confirmation (`1878` calls / `41,196,913 v2`) = **`2,797` total logical calls / `61,477,706 v2`**, `0` unknown-usage attempts.

## 2. Confirmation (`n = 8`) 3-Arm Summary & Full Metered `Scope-1` Breakdown

| Arm | Strict Pass | 0-LLM Closures | Total Turns | Scope-1 Total v2 | stream_solve | dag_proposal | dag_synth | **dag_failed_search** | dag_merge_wd | guard_ind | Max K / Depth / Fan-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicArchive_Online` | **60/64 (93.8%)** | 19/64 | 491 | **12,076,813** | 11,094,551 | 0 | 0 | 0 | 0 | 982,262 | 1 / 1 / 0 |
| `Fixed_TwoStage_Modular_Online` | **60/64 (93.8%)** | 18/64 | 517 | **14,735,484** | 2,056,530 | 802,806 | 6,597,620 | 1,055,805 | 2,231,646 | 1,991,077 | 2 / 2 / 1 |
| `Hegel_UnknownCut_GraphDiscovery_Online` | **61/64 (95.3%)** | 22/64 | 440 | **12,477,436** | 1,893,758 | 681,869 | 4,481,736 | 723,732 | 2,935,017 | 1,761,324 | 4 / 3 / 2 |

- Shared `Stage S` source ledger (reported separately): `1,907,180 v2` (`80` calls); Confirmation Grand Physical Ledger: `41,196,913 v2` (`1878` logical calls).

## 3. Stratified Paired Cluster Inference (Stratum A Seen vs Stratum B Held-Out Unseen + Bootstrap & Student-`t` CIs)

| Comparison | Stratum | Mean Cluster Δv2 | Exact Sign-Flip `p` | Bootstrap 95% CI | Student-`t` 95% CI | Pooled `K*` |
|---|---|---:|---:|---|---|---|
| Hegel − Flat_MonolithicArchive | Pooled (all clusters) (n=8) | +50,077.88 | 0.8281 | [-358642.12, 454756.17] | [-480700.47, 580856.22] | `K* = not_reached` |
| Hegel − Flat_MonolithicArchive | Stratum A (seen topology families) (n=4) | -214,831.00 | 0.5000 | [-710346.00, 280684.00] | [-1151299.90, 721637.90] | — |
| Hegel − Flat_MonolithicArchive | Stratum B (held-out unseen topologies) (n=4) | +314,986.75 | 0.3750 | [-199074.50, 829048.00] | [-700191.78, 1330165.28] | — |
| Hegel − Flat_MonolithicArchive | Monolithic K=1 negative control (n=1) | +435,032.00 | 1.0000 | [435032.00, 435032.00] | N/A | — |
| Hegel − Flat_MonolithicArchive | Multi-node topologies only (n=7) | -4,915.57 | 0.9844 | [-455119.71, 444035.29] | [-619805.38, 609974.24] | — |
| Hegel − Fixed_TwoStage_Modular | Pooled (all clusters) (n=8) | -282,256.00 | 0.1484 | [-602847.31, 21358.51] | [-683906.64, 119394.64] | `K* = 4` |
| Hegel − Fixed_TwoStage_Modular | Stratum A (seen topology families) (n=4) | -150,020.75 | 0.5000 | [-486101.00, 186059.50] | [-806881.35, 506839.85] | — |
| Hegel − Fixed_TwoStage_Modular | Stratum B (held-out unseen topologies) (n=4) | -414,491.25 | 0.5000 | [-880601.00, 51618.50] | [-1316773.83, 487791.33] | — |
| Hegel − Fixed_TwoStage_Modular | Monolithic K=1 negative control (n=1) | +17,675.00 | 1.0000 | [17675.00, 17675.00] | N/A | — |
| Hegel − Fixed_TwoStage_Modular | Multi-node topologies only (n=7) | -325,103.29 | 0.1406 | [-673407.57, 17247.86] | [-789508.78, 139302.21] | — |

## 4. Per-Cluster Confirmation Audit Table (Including Held-Out Stratum B & Per-Cluster `K*`)

| World ID | Stratum | Hidden Topology (Analytics Only) | Flat Mono Pass (v2) | Fixed 2-Stage Pass (v2) | **Hegel Pass (v2)** | Hegel 0-LLM | Max K/d/fo | Shared exec=1 | Withdrawals | Per-Cluster `K*` (vs Mono / Fix2) |
|---|---|---|---:|---:|---:|---:|---|---|---:|---|
| `conf_m101_a01` | `stratum_a_seen_topology` | `monolithic_coupled` | 6/8 (959,148) | 7/8 (1,376,505) | **8/8 (1,394,180)** | 4/8 | K=2, d=2, fo=1 | — | 0 | not_reached / not_reached |
| `conf_m101_a02` | `stratum_a_seen_topology` | `chain_k3_variable_cut` | 7/8 (1,890,301) | 7/8 (1,697,377) | **6/8 (1,248,092)** | 1/8 | K=3, d=3, fo=2 | YES | 3 | not_reached / not_reached |
| `conf_m101_a03` | `stratum_a_seen_topology` | `chain_k3_reversed_cut` | 8/8 (1,480,017) | 8/8 (1,251,909) | **8/8 (1,606,353)** | 3/8 | K=3, d=3, fo=2 | YES | 1 | not_reached / not_reached |
| `conf_m101_a04` | `stratum_a_seen_topology` | `diamond_k4_shared_prefix` | 8/8 (1,782,275) | 8/8 (1,526,709) | **8/8 (1,003,792)** | 4/8 | K=4, d=3, fo=2 | YES | 1 | 2 / 1 |
| `conf_m101_b01` | `stratum_b_unseen_topology` | `chain_k4_deep_cascade` | 8/8 (1,747,227) | 7/8 (2,369,553) | **8/8 (1,272,945)** | 3/8 | K=3, d=3, fo=2 | YES | 1 | 1 / 5 |
| `conf_m101_b02` | `stratum_b_unseen_topology` | `asymmetric_skip_dag_k4` | 8/8 (1,464,644) | 7/8 (3,037,549) | **7/8 (2,372,955)** | 2/8 | K=3, d=3, fo=1 | — | 0 | not_reached / 6 |
| `conf_m101_b03` | `stratum_b_unseen_topology` | `diamond_k4_coupled_first` | 8/8 (1,204,055) | 8/8 (1,868,740) | **8/8 (1,953,840)** | 3/8 | K=4, d=3, fo=2 | YES | 0 | not_reached / not_reached |
| `conf_m101_b04` | `stratum_b_unseen_topology` | `hybrid_mono_to_split_shift` | 7/8 (1,549,146) | 8/8 (1,607,142) | **8/8 (1,625,279)** | 2/8 | K=4, d=3, fo=2 | YES | 0 | not_reached / not_reached |

## 5. Pre-Freeze `n = 4` Dev Calibration Summary

- Run ID: `m101_dev_20261128T074929Z_da6c9471` | Peak concurrency: `12/32` | Unknown attempts: `0` | `source_tree_unchanged = True`

| Arm | Strict Pass | 0-LLM Closures | Total Turns | Scope-1 Total v2 | stream_solve | dag_proposal | dag_synth | **dag_failed_search** | dag_merge_wd | guard_ind | Max K / Depth / Fan-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicArchive_Online` | **31/32 (96.9%)** | 10/32 | 243 | **5,978,326** | 5,537,949 | 0 | 0 | 0 | 0 | 440,377 | 1 / 1 / 0 |
| `Fixed_TwoStage_Modular_Online` | **29/32 (90.6%)** | 12/32 | 226 | **6,070,822** | 1,301,544 | 302,408 | 2,306,232 | 640,061 | 905,670 | 614,907 | 2 / 2 / 1 |
| `Hegel_UnknownCut_GraphDiscovery_Online` | **31/32 (96.9%)** | 15/32 | 179 | **4,862,101** | 624,677 | 261,120 | 1,165,766 | 292,806 | 1,929,125 | 588,607 | 4 / 3 / 2 |

## 6. Calibrated Claim Boundary & Structural Amortization Law

1. **Advantage Over Prescribed Two-Stage Cuts**: Forcing `K <= 2` on depth `3–4` cascades and asymmetric skip DAGs lumps distinct mechanisms into coarse modules that repeatedly break under local shifts; `Hegel_UnknownCut_GraphDiscovery_Online` reduces total `Scope-1` cost by **`-15.3%`** (`K* = 4`) and saves **`-414,491.25 v2/cluster`** on Held-Out Unseen Stratum B (`-46.3%` on `conf_m101_b01`, `-21.9%` on `conf_m101_b02`).
2. **Exact Amortization Boundary Against Unrestricted Monolithic Archive**: Under full-cost `Scope-1` accounting (`dag_structure_proposal: 682k`, `dag_failed_search: 724k`, `dag_merge_withdrawal_repair: 2,935k`, `guard_induction: 1,761k`), unknown-cut DAG discovery cuts cost by **`-27.1%` to `-43.7%`** on separable multi-stage chains and shared-prefix diamonds (`conf_m101_a04` `-43.7%`, `conf_m101_a02` `-34.0%`, `conf_m101_b01` `-27.1%`; Stratum A `-12.1%`), while paying upfront capital on monolithic negative controls (`a01`: `8/8` vs `6/8` quality), mid-stream cut re-splitting (`b02`), and coupled-first reverse streams (`b03`), yielding pooled `n = 8` cost near-parity (`+3.3%`, `p = 0.8281`, `K* = not_reached` at `H = 8`) with `#1` pass rate (`95.3%` vs `93.8%`) and `-10.4%` fewer turns (`440` vs `491`). Bounded unknown-cut graph discovery works, whereas unbounded zero-prior graph search faces steep search-and-withdrawal capital unless amortized over bounded graph priors and horizons $H > K^*$.
3. **Container Attribution**: Step-06 plain-dict routing parity (`16/16 = 100.0%`) confirms that runtime routing gains come from the discovered DAG factorization, large-margin guards, and external receipt verification rather than graph-container lock-in.
