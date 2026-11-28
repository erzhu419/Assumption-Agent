# `M10.0.1` Audit Closure & Probe-Persistence Errata (`T01–T11`, `0` Paid LLM Calls)

> **Sealed Post-Hoc Reverification Directory**: `docs/evidence/m10_0_1_audit_closure_and_probe_reverification_20261128/` (`44` SHA-256-verified files). Zero existing `m9_9*` or `m10_0_*` sealed files or scores (`72/74/77`) were modified.

## 1. `T01` Missing Raw Probe Receipt Inventory & `41/41` Post-Hoc Docker Back-Verification

- **Historical Gap Inventory (`01_m100_probe_persistence_inventory.json`)**:
  - `M10.0` `dev`: `19/107` witnesses marked `recorded_without_raw_measurement` (`{'pprobe': 15, 'uprobe': 4}`).
  - `M10.0` `confirmation`: **`41/216`** witnesses marked `recorded_without_raw_measurement` (`{'pprobe': 32, 'uprobe': 9}`), including **`8/8`** withdrawal `prior_positive` witnesses.
- **Post-Hoc Real Docker Back-Verification (`02_m100_41_missing_probes_docker_backverification.json`)**:
  - All **`41`** missing confirmation probes (`32 pprobe` + `9 uprobe`) were re-executed in `hegel-clean-scipy:m7v1` against the exact sealed binding code, parameter epoch, public batch, and dependency bundle (`96.09s` elapsed, `0` paid LLM calls).
  - **Outcome**: **`41/41` (`100.0%`)** replayed verdicts match the recorded `M10.0` witness ledger, including **`8/8` (`100.0%`)** withdrawal `prior_positive` witnesses. Every replayed receipt is explicitly labeled `raw_measurement_status = "backfilled_posthoc_replay"` with a fresh `backfill_*` ID and `supersedes_probe_id`.
- **Forward Architectural Enforcement**: `ProbeReceiptStore`, schema-v3 `probe_receipt_sha256` witness digests, migration of all `run_m10_0` bare witness calls to `record_tested_witness_from_probe_receipt`, dual `primary/alternate` `B2` diagnostic retention, and strict 3-way external FK verification in `issue_measured_withdrawal` & `hydrate_separability_withdrawals_from_graph` (verified against GPT-6's exact forged-`prior.probe_id` negative control in `tests/test_m10_0_1_audit_closure_suite.py`).

## 2. `0`-LLM Archived Solver Lifting & Anti-Unification Coverage Inventory (`03_m100_archived_solvers_lifting_coverage_inventory.json`)

- Audited `32` archived `M10.0` confirmation monolithic solvers through the legacy `M6.8` scalar AST expression lifter (`lift_python_scientific_expressions_to_programs`) and `anti_unify_programs`:
  - `category_counts`: `{'extracted_unadopted': 0, 'lifter_unsupported': 32, 'no_shared_skeleton': 0}`
  - `total_lifted_scalar_programs`: `0`
  - `non_trivial_anti_unified_cross_cluster_pairs`: `0`
- **Conclusion**: Scientific solvers in `M10.0` are predominantly imperative NumPy/SciPy routines (`lifter_unsupported` for scalar AST lifting), confirming that open multi-module discovery (`M10.1`) must operate over executable Python callable nodes and measured I/O contracts rather than scalar symbolic AST lifting alone.

## 3. Complete `41/41` Missing Confirmation Probe Post-Hoc Replay Ledger

| World ID | Witness ID | Origin | Original `probe_id` | Backfill `probe_id` | Withdrawal Prior? | Recorded vs Replayed | Match? |
|---|---|---|---|---|---:|---|---|
| `alpha_m100_c01` | `wit_0005_bind_alpha_m100_c01_up_b01_e8ce6588` | `joint_task_probe` | `pprobe_9cfe4611634442db` | `backfill_pprobe_9cfe4611634442db` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c01` | `wit_0006_bind_alpha_m100_c01_dn_b02_cfe8ecc8` | `joint_task_probe` | `pprobe_b7b53350516d4026` | `backfill_pprobe_b7b53350516d4026` | `True` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c01` | `wit_0008_bind_alpha_m100_c01_up_b01_2d495579` | `direct_component_probe` | `uprobe_563241be8db84ff5` | `backfill_uprobe_563241be8db84ff5` | `False` | `failed` vs `failed` | **`True`** |
| `alpha_m100_c01` | `wit_0009_bind_alpha_m100_c01_up_b02_2d495579` | `joint_task_probe` | `pprobe_381c2b6144df4513` | `backfill_pprobe_381c2b6144df4513` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c01` | `wit_0011_bind_alpha_m100_c01_dn_b03_31b8c081` | `joint_task_probe` | `pprobe_c0041f8fcfca4c5a` | `backfill_pprobe_c0041f8fcfca4c5a` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c02` | `wit_0004_bind_alpha_m100_c02_up_b01_2e36a86d` | `joint_task_probe` | `pprobe_12cd75eb32924c45` | `backfill_pprobe_12cd75eb32924c45` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c02` | `wit_0007_bind_alpha_m100_c02_dn_b02_7a35fe59` | `joint_task_probe` | `pprobe_8f0a149cff7a4561` | `backfill_pprobe_8f0a149cff7a4561` | `True` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c02` | `wit_0008_bind_alpha_m100_c02_up_b01_a00ea06d` | `direct_component_probe` | `uprobe_78b5844273284d8f` | `backfill_uprobe_78b5844273284d8f` | `False` | `failed` vs `failed` | **`True`** |
| `alpha_m100_c02` | `wit_0009_bind_alpha_m100_c02_up_b02_a00ea06d` | `joint_task_probe` | `pprobe_a57b3696f1374dd8` | `backfill_pprobe_a57b3696f1374dd8` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c02` | `wit_0010_bind_alpha_m100_c02_dn_b03_b74aecdd` | `joint_task_probe` | `pprobe_ed8002da7b1e4748` | `backfill_pprobe_ed8002da7b1e4748` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c03` | `wit_0005_bind_alpha_m100_c03_up_b01_35a330cf` | `joint_task_probe` | `pprobe_2fcef5a9fc4e4bc1` | `backfill_pprobe_2fcef5a9fc4e4bc1` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c03` | `wit_0006_bind_alpha_m100_c03_dn_b02_c2b83bba` | `joint_task_probe` | `pprobe_44d9575412c24720` | `backfill_pprobe_44d9575412c24720` | `True` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c03` | `wit_0008_bind_alpha_m100_c03_up_b01_3e170931` | `direct_component_probe` | `uprobe_eb9046b06c144754` | `backfill_uprobe_eb9046b06c144754` | `False` | `failed` vs `failed` | **`True`** |
| `alpha_m100_c03` | `wit_0009_bind_alpha_m100_c03_up_b02_3e170931` | `joint_task_probe` | `pprobe_1df4bf8e0c564897` | `backfill_pprobe_1df4bf8e0c564897` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c03` | `wit_0011_bind_alpha_m100_c03_dn_b03_d6f2f32c` | `joint_task_probe` | `pprobe_7e00ebbcf9864a9c` | `backfill_pprobe_7e00ebbcf9864a9c` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c04` | `wit_0004_bind_alpha_m100_c04_up_b01_f2244f00` | `joint_task_probe` | `pprobe_7641e285f0654580` | `backfill_pprobe_7641e285f0654580` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c04` | `wit_0007_bind_alpha_m100_c04_dn_b02_b493573f` | `joint_task_probe` | `pprobe_1fdc31a3cd7e4d7d` | `backfill_pprobe_1fdc31a3cd7e4d7d` | `True` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c04` | `wit_0008_bind_alpha_m100_c04_up_b01_dd8649e5` | `direct_component_probe` | `uprobe_6a8bd8fc6c304956` | `backfill_uprobe_6a8bd8fc6c304956` | `False` | `failed` vs `failed` | **`True`** |
| `alpha_m100_c04` | `wit_0009_bind_alpha_m100_c04_up_b02_dd8649e5` | `joint_task_probe` | `pprobe_270fea190d7c40c8` | `backfill_pprobe_270fea190d7c40c8` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c04` | `wit_0010_bind_alpha_m100_c04_dn_b03_f3c69c5d` | `joint_task_probe` | `pprobe_f35231f9373048a8` | `backfill_pprobe_f35231f9373048a8` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c01` | `wit_0005_bind_beta_m100_c01_up_b01_7528934e` | `joint_task_probe` | `pprobe_dd600d0799114516` | `backfill_pprobe_dd600d0799114516` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c01` | `wit_0006_bind_beta_m100_c01_dn_b02_27b4710e` | `joint_task_probe` | `pprobe_3dbd6d2de2ca4a2d` | `backfill_pprobe_3dbd6d2de2ca4a2d` | `True` | `passed` vs `passed` | **`True`** |
| `beta_m100_c01` | `wit_0008_bind_beta_m100_c01_up_b01_e7ec38f6` | `direct_component_probe` | `uprobe_31a67c992f0e4e67` | `backfill_uprobe_31a67c992f0e4e67` | `False` | `failed` vs `failed` | **`True`** |
| `beta_m100_c01` | `wit_0009_bind_beta_m100_c01_up_b02_e7ec38f6` | `joint_task_probe` | `pprobe_1000490ab5074d6a` | `backfill_pprobe_1000490ab5074d6a` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c01` | `wit_0011_bind_beta_m100_c01_dn_b03_3f8467fe` | `joint_task_probe` | `pprobe_bfde598286af47bb` | `backfill_pprobe_bfde598286af47bb` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c02` | `wit_0004_bind_beta_m100_c02_up_b01_a06703f6` | `joint_task_probe` | `pprobe_0d6e020f6216497b` | `backfill_pprobe_0d6e020f6216497b` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c02` | `wit_0007_bind_beta_m100_c02_dn_b02_b9a60bfe` | `joint_task_probe` | `pprobe_195e88b7b5ca4610` | `backfill_pprobe_195e88b7b5ca4610` | `True` | `passed` vs `passed` | **`True`** |
| `beta_m100_c02` | `wit_0008_bind_beta_m100_c02_up_b01_c4142fb8` | `direct_component_probe` | `uprobe_4665ecb98718492c` | `backfill_uprobe_4665ecb98718492c` | `False` | `failed` vs `failed` | **`True`** |
| `beta_m100_c02` | `wit_0009_bind_beta_m100_c02_up_b02_c4142fb8` | `joint_task_probe` | `pprobe_a299909c771b4923` | `backfill_pprobe_a299909c771b4923` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c02` | `wit_0010_bind_beta_m100_c02_dn_b03_e563d504` | `joint_task_probe` | `pprobe_c746c9e36f9a4da5` | `backfill_pprobe_c746c9e36f9a4da5` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c03` | `wit_0005_bind_beta_m100_c03_up_b01_ddbc9e27` | `joint_task_probe` | `pprobe_85004a39ab544feb` | `backfill_pprobe_85004a39ab544feb` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c03` | `wit_0006_bind_beta_m100_c03_dn_b02_7d7e5c94` | `joint_task_probe` | `pprobe_7e6426367ca8408e` | `backfill_pprobe_7e6426367ca8408e` | `True` | `passed` vs `passed` | **`True`** |
| `beta_m100_c03` | `wit_0008_bind_beta_m100_c03_up_b01_72168d3c` | `direct_component_probe` | `uprobe_149f45eb92984c2b` | `backfill_uprobe_149f45eb92984c2b` | `False` | `failed` vs `failed` | **`True`** |
| `beta_m100_c03` | `wit_0009_bind_beta_m100_c03_up_b02_72168d3c` | `joint_task_probe` | `pprobe_08b46660d7914687` | `backfill_pprobe_08b46660d7914687` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c03` | `wit_0011_bind_beta_m100_c03_dn_b03_6f9fba05` | `joint_task_probe` | `pprobe_cbb6e782e2864ca8` | `backfill_pprobe_cbb6e782e2864ca8` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c04` | `wit_0004_bind_beta_m100_c04_up_b01_fc22afcb` | `joint_task_probe` | `pprobe_25457f9a1bed4d16` | `backfill_pprobe_25457f9a1bed4d16` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c04` | `wit_0007_bind_beta_m100_c04_dn_b02_7cda1fab` | `joint_task_probe` | `pprobe_e719d2a817fe479e` | `backfill_pprobe_e719d2a817fe479e` | `True` | `passed` vs `passed` | **`True`** |
| `beta_m100_c04` | `wit_0008_bind_beta_m100_c04_up_b01_54615d04` | `direct_component_probe` | `uprobe_0421fbdfa59846b5` | `backfill_uprobe_0421fbdfa59846b5` | `False` | `failed` vs `failed` | **`True`** |
| `beta_m100_c04` | `wit_0009_bind_beta_m100_c04_dn_b03_3b4d43a5` | `joint_task_probe` | `pprobe_4a1046e8970e4fbe` | `backfill_pprobe_4a1046e8970e4fbe` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c04` | `wit_0010_bind_beta_m100_c04_up_b01_bb129c73` | `direct_component_probe` | `uprobe_fa8a98379f344079` | `backfill_uprobe_fa8a98379f344079` | `False` | `failed` vs `failed` | **`True`** |
| `beta_m100_c04` | `wit_0011_bind_beta_m100_c04_up_b02_bb129c73` | `joint_task_probe` | `pprobe_16f03b58993a4f13` | `backfill_pprobe_16f03b58993a4f13` | `False` | `passed` vs `passed` | **`True`** |
