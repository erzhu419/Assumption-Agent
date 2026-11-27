# M9.6.1 Audit Errata, Semantic Closure & Paid-Pool Attribution (`0` Paid LLM Calls)

- **Evidence Directory**: `docs/evidence/m9_6_1_semantic_closure_and_paid_pool_attribution_20261127/` (`SHA256SUMS.txt` verified; all historical `M9.4`, `M9.5`, `M9.5.1`, and `M9.6` bundles untouched)
- **CostPolicy Version**: `hegel_cost_policy_v2_m961_call_uuid_conserved` (`sum(episodes) == sum(events) == sum(worlds) == sum(disjoint_pairs) == arm_ledger` asserted across `v1` and `v2`)
- **Shared Verification Kernel**: `hegel_shared_math_handoff_kernel_v3_m961_three_layer` (`validate_output_structure` + `validate_public_witness` + `validate_truth_accuracy`)

## 1. Reconciled M9.6 Event, World & Disjoint-Pair Cost Statistics (`N01` Errata)

- **Event-Level `proxy_v2` (Corrected from Call-UUID Raw Tuples)**:
  - `T`: `Flat = 2,531,206` vs `Full = 3,095,591` (`N, G1, R, U, G2, M = 0`)
  - `C`: `Flat = 2,472,490` vs `Full = 3,531,535`
  - **Scope 1 Sum Check**: `Flat = 5,003,696` (`== 5,003,696`), `Full = 6,627,126` (`== 6,627,126`). In-stream L2 compile accounts for **`89.55%`** (`1,453,819 / 1,623,430`) of the `v2` excess.
- **Primary Cluster-Level Inference (`n = 4` Disjoint Source Pairs)**:
  - Pair `v2` deltas (`Full - Flat`): `[122073, 401432, 474908, 625017]` (`4/4` higher; two-sided sign-flip `p = 0.125`).
- **Secondary World-Level Sensitivity (`n = 8` Worlds)**:
  - World `v2` deltas: `{'alpha_m96_w01': 129381, 'alpha_m96_w02': 291293, 'alpha_m96_w03': 243453, 'alpha_m96_w04': 393279, 'beta_m96_w01': -7308, 'beta_m96_w02': 110139, 'beta_m96_w03': 231455, 'beta_m96_w04': 231738}` (`7/8` worlds higher; `beta_m96_w01 = -7,308`; mean diff `202,928.75`, 95% bootstrap CI `[119139.49, 280925.88]`, exact two-sided sign-flip `p = 0.015625`).
- **Zero-LLM Episode Clarification**: Both arms executed `48/64` zero-LLM episodes (`47/64` passed private verification; `1/64` failed private verification on `alpha_m96_w03 / M`; non-zero Docker/CPU probe time is tracked separately).

## 2. Three-Layer Contract Forensics on `alpha_m96_w03 / M` (`N02`) & M9.4 `68/68` Regrade Errata (`N03`)

- **`N02` (`alpha_m96_w03 / M` True Root Cause)**: Official `M9.6` sealed scores remain `63/64` vs `63/64`. `alpha_m96_w03 / M` had `preload_scale = 0.0` (`has_initial_memory_states = False`); the `0.1128 > 0.08` failure on seed `973861` arose from evaluating algebraic curve-rate self-consistency using hidden ground-truth stresses rather than the candidate's validated `CalibrationWitness` coordinates. Under `hegel_shared_math_handoff_kernel_v3_m961_three_layer`, Layer 2 estimated-coordinate consistency residual is `<= 2.01e-14` (`4/4` seed-arm pairs pass all 3 layers while spoofed `CalibrationWitness`es are `100%` rejected).
- **`N03` (`M9.4` Full `68`-Instance Unified-Kernel Regrade)**: Replacing script-local `gt_obj` key dispatch with `infer_contract_kind` (`gt.get("terminal_field_B") is not None`) and re-running all `34` `M9.4` episodes (`68` private instances) via pure-JSON worker yields **`64/68` private instances (`30/34` episodes)**.

## 3. Qualification Identity (`N04`), Multi-Conflict Lineage (`N07`), Reconstructed Funnel (`N09/N10`) & Paid-Pool Controls (`N11`)

- **`N04`**: Exact `executable_kind` code SHA matching rejects unexecuted-module receipts (`rejected = True`), content-recomputed `receipt_id` rejects mutated receipts (`rejected = True`), and secondary-parent version bumps invalidate stale `parent_parameter_versions` (`rejected = True`).
- **`N07`**: `ImmutableBranchLineageController` rejects splits without a tested scientific negative witness (`rejected = True`), never fabricates negative witnesses from `C1` selection (`True`), and preserves `C0 -> C1 -> C2 -> restore C0 -> restore C1` (`{'restore_C0_x_0_2': 'bind_dev_b00', 'restore_C1_x_1_1': 'bind_dev_b01', 'route_C2_x_2_4': 'bind_dev_b02'}`).
- **`N09 + N10`**: Reconstructed `Full` `T` funnel from `04d_skill_contract_inspections.jsonl`: new `L1` search/describe = `5/8`, `3/8`; new `L2` search/describe = `6/8`, `4/8`; any new module described = `7/8`; dynamic calls = `0/8`; exact retention = `0/8` (`4` Alpha cross-domain KKT opportunities vs `4` Beta same-family multi-budget opportunities).
- **`N11`**: Sequential paid-pool replay distinguishes `OldestFirst_Probes1` (`32/48`) from `LatestFirst_Probes1` (`40/48`) and shows `Flat_SharedConditionalTable_Probes1` (`48/48`) matches `Hegel_MechanismGraph_Probes1` (`48/48`) on 2-wing fixed-wiring `M9.5`, motivating `M9.7`'s multi-conflict non-nested + counterfactual-history + permuted-parent benchmark.
