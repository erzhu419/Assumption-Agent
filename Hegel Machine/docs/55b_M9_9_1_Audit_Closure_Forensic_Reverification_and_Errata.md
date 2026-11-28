# Hegel M9.9.1 Independent Audit Closure, 30-Probe Docker Back-Verification, Forensic Root-Cause Calibration & Errata (`Q01–Q14` + Original `R01–R16`)

- **Date**: `2026-11-28`
- **Audited Baseline Commit**: `d4b1cd7f06a08db8d967a43913003527d21a0354` (`Hegel_M9_9_Audit`)
- **Paid LLM Calls Used in `M9.9.1`**: **`0`** (`100%` deterministic code hardening + real air-gapped Docker back-verification)
- **Original Sealed Evidence Status**: `docs/evidence/m9_8_1_*`, `docs/evidence/m9_9_dev_live_calibration_20261128`, and `docs/evidence/m9_9_autonomous_decomposition_and_interaction_withdrawal_20261128` remain `100%` byte-untouched; all `M9.9.1` back-verification receipts are independently sealed in `docs/evidence/m9_9_1_audit_closure_and_forensic_reverification_20261128/`.

---

## 1. Complete `Hegel_M9_9_Audit` (`Q01–Q14`) Resolution Register

| Finding | Priority | Audit Diagnosis | `M9.9.1` Code Fix, Container Back-Verification & Claim Calibration | Status |
|---|---|---|---|---|
| `Q01` | `P1` | The original factorization inferred two off-diagonal outcomes. | Completed 30/30 historical probes. Scientific/domain rejections 30, old witnesses linked 8/8. Full matrix required for new joint guard evidence. | Retrospective replay |
| `Q02` | `P1` | alpha c02/E7 failed. | Frozen downstream feature repair reduced MAE from 0.071267 to 0.004857. The public swap check tests one necessary pooling invariant. The frozen preprocessing counterfactual changes pooling and imputation together. | Component attribution |
| `Q03` | `P1` | The guard omitted an available upstream solver in c04/E7. | Prefix pair up_b02 + dn_b03. Public acceptance True, official pass True. The prospective bounded fallback requires online evaluation. | Frozen candidate replay |
| `Q04` | `P1` | Delayed factorization hid archived code. A missing separable branch was aliased to a coupled one in the ablation. | Prospective archive fallback and explicit missing-state accounting. Historical costs and grades are still reported under their original policy. | Engineering / future online test |
| `Q05` | `P1` | `--finalize-existing` overwrote `run_id`, `wall_sec`, and `source_tree_sha256_at_start` (`be7190fb...` vs freeze `34b01d19...`). | Separated write-once `execution_manifest.json` from `finalization_manifest.json` in runner; reconstructed exact freeze tree SHA (`34b01d19ef125692...`) and raw call envelopes (`3,718.765s` confirm, `2,259.938s` dev). | **CLOSED (`Code + Provenance Ledger`)** |
| `Q06` | `P1` | Eight paired clusters. | Hegel versus Monolithic quality p = 0.5, cost p = 0.765625. Versus Plain, cost p = 0.0625 at x = 0. The sensitivity table uses the same frozen eight pairs. | Inference corrected |
| `Q07` | `P1` | A legacy withdrawal can be internally consistent while its original scientific receipt is missing. | Original joint scientific receipts 0/7. Matched negative probes in the separate replay 7/7. Typed evidence is required for prospective authority. | Historical and replay evidence reported separately |
| `Q08` | `P2` | Legacy `v1` downgrade failed second `save -> reload` round-trip; `M9.8` `compatibility_inference` could write a downstream scientific negative. | Recomputes migrated `v2` `witness_content_sha256` & `unknown_requires_reverification` `receipt_id` (`snap_1 == snap_2` idempotent, `0` authority); blocked `compatibility_inference` from minting downstream scientific negatives. | **CLOSED** |
| `Q09` | `P1` | Zero host-seeded source code described too broadly as zero host-seeded decomposition. | Scoped throughout `docs/55_*` & hub as **online module formation under a host-prescribed two-stage interface** (open DAG discovery deferred). | **CLOSED** |
| `Q10` | `P1` | The frozen ablation restricted features available to fixed selectors. | With the same host-reconstructed policy inputs, a plain dictionary reproduces 16/16 candidate hashes. Actual Docker public files are tracked separately. | Prefix replay |
| `Q11` | `P1` | Ideal grid count written as an exact LLM cost law. | Replaced with conditional `N_{sep,unseen} = card((U x D) \ (R u X))` and empirical quality-constrained `ΔC(K)` / `K*` protocol. | **CLOSED** |
| `Q12` | `P2` | Cancelled development cost was missing. | Recovered 333 calls and 8,220,922 known v2, with 6 unclosed calls. Total 53904542 + x + y. | Partial billing recovery |
| `Q13` | `P2` | `E4/E5` `v2/v3` swap, `E1` `7/8, 6/8` vs cumulative `8/8, 7/8`, and elapsed wall-ms (`1.421x` total / `3.320x` coverage) vs CPU wording. | Fixed in `scripts/generate_m9_9_report_from_sealed_json.py`, `docs/55_*`, and hub. | **CLOSED** |
| `Q14` | `P2` | `R01–R16` ID drift in `docs/55` §7 and broken `54b` link in reuse hub. | Restored exact original `R01–R16` IDs (retaining retired `R14` and open `R15`) and fixed `54b` link in `docs/hegel_machine_reuse_hub_20261006.md`. | **CLOSED** |

---

## 2. Real Docker Back-Verification of All `30` Historical Cross-Diagonal Decomposition Cells (`Q01`)

- **Ledger**: `01_m99_q01_30_cross_diagonal_decomposition_probes.json` (`total_probe_wall_ms = 146,840.91 ms`, `0` paid LLM calls)
- **Outcome Breakdown Across `30` Probes (`15` Accepted Decompositions × `2` Off-Diagonal Cells)**:
  - `accepted = True`: **`0/30`**
  - `scientific_or_domain` rejected (`completed`): **`30/30`**
  - `schema_or_interface` rejected (`completed`): **`0/30`**
  - `infra_or_timeout`: **`0/30`**
  - Superseded original unexecuted `Hegel` `d0@E` witnesses (`errata_relation = supersedes_unexecuted_projection_with_executed_probe`): **`8/8`**

| Arm | Cluster (`Decomp Step`) | Cross-Diagonal Cell | `probe_id` | Status | Accepted | Failure Kind | Wall (`ms`) | Supersedes Unexecuted Witness |
|---|---|---|---|---|---|---|---:|---|
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c01` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_207e31f71a9a4bfd` | `completed` | `False` | `scientific_or_numerical` | `4952.0` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c01` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_ebe609a20564467c` | `completed` | `False` | `scientific_or_numerical` | `4904.9` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c02` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_284af950f9464dcf` | `completed` | `False` | `scientific_or_numerical` | `4964.0` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c02` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_c4dfb183d1714fca` | `completed` | `False` | `scientific_or_numerical` | `4963.1` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c04` (`s2`) | `u0_plus_d0_on_current_E` | `xprobe_1e633a14a73d49c4` | `completed` | `False` | `scientific_or_numerical` | `5029.2` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c04` (`s2`) | `u0_plus_d1_on_Stage_S` | `xprobe_8c0458eeabb74ca8` | `completed` | `False` | `scientific_or_numerical` | `5432.3` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c01` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_7341a09e5b54400b` | `completed` | `False` | `scientific_or_numerical` | `4753.4` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c01` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_ca69051b451e46e5` | `completed` | `False` | `scientific_or_numerical` | `4766.6` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c02` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_54cfc758b4ff4a33` | `completed` | `False` | `scientific_or_numerical` | `4883.2` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c02` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_17979b37c7264b54` | `completed` | `False` | `scientific_or_numerical` | `4770.1` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c03` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_bd0601848b7c4811` | `completed` | `False` | `scientific_or_numerical` | `5085.5` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c03` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_c9c1a1f1d45a43fc` | `completed` | `False` | `scientific_or_numerical` | `5313.0` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c04` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_2da92e05d2564f54` | `completed` | `False` | `scientific_or_numerical` | `4771.7` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c04` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_2b4eba4007e3482d` | `completed` | `False` | `scientific_or_numerical` | `4807.3` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c01` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_b83f6acb78b74a3c` | `completed` | `False` | `scientific_or_numerical` | `4937.4` | `wit_0002_bind_alpha_m99_c01_dn_b00_5a38161b` |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c01` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_3ca15e58450842ca` | `completed` | `False` | `scientific_or_numerical` | `4966.2` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c02` (`s2`) | `u0_plus_d0_on_current_E` | `xprobe_34560cf5d69b460f` | `completed` | `False` | `scientific_or_numerical` | `5121.8` | `wit_0002_bind_alpha_m99_c02_dn_b00_ed77e0c2` |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c02` (`s2`) | `u0_plus_d1_on_Stage_S` | `xprobe_4b152e625ff045b5` | `completed` | `False` | `scientific_or_numerical` | `5079.4` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c03` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_30803ee5863a46b0` | `completed` | `False` | `scientific_or_numerical` | `4545.1` | `wit_0002_bind_alpha_m99_c03_dn_b00_03b4a8a3` |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c03` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_a039987d7dc64097` | `completed` | `False` | `scientific_or_numerical` | `4537.6` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c04` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_47532080a84b48fc` | `completed` | `False` | `scientific_or_numerical` | `5119.5` | `wit_0002_bind_alpha_m99_c04_dn_b00_3b352674` |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c04` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_24384cd5773c41af` | `completed` | `False` | `scientific_or_numerical` | `5148.2` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c01` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_1cb070deb60e4a2e` | `completed` | `False` | `scientific_or_numerical` | `4623.3` | `wit_0002_bind_beta_m99_c01_dn_b00_dc9859ed` |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c01` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_be0d9b11609b4276` | `completed` | `False` | `scientific_or_numerical` | `4497.6` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c02` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_1375835636ba4d81` | `completed` | `False` | `scientific_or_numerical` | `5004.8` | `wit_0002_bind_beta_m99_c02_dn_b00_c95627f7` |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c02` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_5ea807d92f224f76` | `completed` | `False` | `scientific_or_numerical` | `4990.9` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c03` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_aa2aa8723a38492d` | `completed` | `False` | `scientific_or_numerical` | `4884.7` | `wit_0002_bind_beta_m99_c03_dn_b00_83808d89` |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c03` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_5664ada0fc764d48` | `completed` | `False` | `scientific_or_numerical` | `4955.8` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c04` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_bbf71289896548e7` | `completed` | `False` | `scientific_or_numerical` | `4529.1` | `wit_0002_bind_beta_m99_c04_dn_b00_387320b1` |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c04` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_c4d8b74456b443be` | `completed` | `False` | `scientific_or_numerical` | `4503.6` | — |

---

## 3. Provenance Reconciliation (`Q05`), Unknown `x` Sign-Flip Ladder (`Q06`), & `c04/E7` Container Check (`Q03`)

1. **Git Tree & Timing Separation (`Q05`)**:
   - Declared paid-run freeze commit `7624c43bf72d283e82442504eff35ff6ee02b40d` (`2700` `.py` files): `34b01d19ef12569240f013bd137e5818c449d68bdcb2ebb285697099c1234cdd`
   - Post-freeze finalizer tree recorded in `m9_9_continual_summary.json`: `be7190fbfab8df60c5da023c3133f59fa0f71c6dde95ebc967367de3fd9d40bc`
   - Raw mirrored call time envelope: Confirmation `3,718.765s` (`m99_confirmation_20261127T193020Z` vs finalizer `219.26s`); Dev `2,259.938s` (`m99_dev_20261127T183538Z`).
2. **Container Verification of Prefix-Available `bind_alpha_m99_c04_up_b02` on `alpha_m99_c04 / E7` (`Q03`)**:
   - Direct `56/24` upstream component probe: `execution_status = completed`, **`chosen_up_holdout_rmse = 0.00007655`** (`<= 0.03`).
   - Composed `(bind_alpha_m99_c04_up_b02 + bind_alpha_m99_c04_dn_b03)` 3-layer public gate: **`accepted = True`** (`4977.58 ms`).
   - Official private verifier (diagnostic check only; original `M9.9` confirmation score/cost unchanged): **`strict_pass = True`** (`4227.89 ms`).
3. **Exact Two-Sided Sign-Flip Sensitivity Ladder for Unknown Attempt `x` (`call_uuid = 0391d702b0c547a488608847b5dff224`)**:

| Unknown `Plain` Attempt Cost `x` (`v2`) | Exact Two-Sided Sign-Flip `p` (`Hegel - Plain`) |
|---:|---:|
| `0` | `0.06250000` |
| `17,652` | `0.06250000` |
| `17,653` | `0.05468750` |
| `53,450` | `0.05468750` |
| `53,451` | `0.04687500` |
| `148,364` | `0.04687500` |
| `148,365` | `0.03906250` |
| `184,162` | `0.03906250` |
| `184,163` | `0.03125000` |
| `217,083` | `0.03125000` |
| `217,084` | `0.02343750` |
| `252,881` | `0.02343750` |
| `252,882` | `0.01562500` |

4. **Elapsed Probe Wall-Clock Ratios (`Q13`)**: Total non-LLM agent probe elapsed wall-ms `Monolithic / Hegel = 1.420765×` (`1,271,833.04 ms / 895,174.81 ms`); empirical coverage sub-item `= 3.319855×` (`689,259.6 ms / 207,604.6 ms`).
