# Hegel M9.9.1 Independent Audit Closure, 30-Probe Docker Back-Verification, Forensic Root-Cause Calibration & Errata (`Q01–Q14` + Original `R01–R16`)

- **Date**: `2026-11-28`
- **Audited Baseline Commit**: `d4b1cd7f06a08db8d967a43913003527d21a0354` (`Hegel_M9_9_Audit`)
- **Paid LLM Calls Used in `M9.9.1`**: **`0`** (`100%` deterministic code hardening + real air-gapped Docker back-verification)
- **Original Sealed Evidence Status**: `docs/evidence/m9_8_1_*`, `docs/evidence/m9_9_dev_live_calibration_20261128`, and `docs/evidence/m9_9_autonomous_decomposition_and_interaction_withdrawal_20261128` remain `100%` byte-untouched; all `M9.9.1` back-verification receipts are independently sealed in `docs/evidence/m9_9_1_audit_closure_and_forensic_reverification_20261128/`.

---

## 1. Complete `Hegel_M9_9_Audit` (`Q01–Q14`) Resolution Register

| Finding | Priority | Audit Diagnosis | `M9.9.1` Code Fix, Container Back-Verification & Claim Calibration | Status |
|---|---|---|---|---|
| `Q01` | `P1` | `M9.9` decomposition executed only diagonal cells (`u0+d0 @ S`, `u0+d1 @ E`), passing `30` unexecuted cross constraints to `d0d1_gind` and writing `8` unexecuted `d0@E` `failed` witnesses in `Hegel`. | Upgraded `verify_m99_autonomous_decomposition_in_containers` to execute all `4` `2x2` cells (`matrix_2x2`); added `record_tested_witness_from_probe_receipt` requiring a completed `ProbeReceipt`; executed all **`30/30` historical cross-diagonal probes in Docker** (`0` passed, `30` scientific/domain rejected, `0` schema rejected, `0` infra/timeout) with explicit `supersedes_unexecuted_witness_id` links for all `8` original `Hegel` witnesses. | **CLOSED (`Code + 30 Docker Probes`)** |
| `Q02` | `P1` | `alpha_m99_c02/E7` miss (`0.071267`) was misattributed to upstream cubic Newton extrapolation (`up_b01` RMSE was `1.4e-5`); real cause was `dn_b02` normalizing `test` on `test['ref_coupons']` alone with unweighted `NaN` mean. | Added `verify_alpha_pooled_ref_coupon_invariance` in `m9_4_1_shared_verification_kernel.py` (`0` private truth, cross-split per-lab `ref_coupons` swap check catching `c02/dn_b02` deterministically); corrected `docs/55_*` §8.1 (`0.071267 -> 0.004857` when using pooled `train+test` standardization + `(lab, stratum)` weighted imputation). | **CLOSED (`Code + Errata`)** |
| `Q03` | `P1` | `alpha_m99_c04/E7` already had verified `bind_alpha_m99_c04_up_b02` (`RMSE = 7.65e-5`) in `step_06`, excluded by over-narrow `up_stat_03 > 1.289054` (`1.274262`) guard. | Executed `up_b02 + dn_b03` in Docker on `c04/E7` (`RMSE = 0.00007655`, public gate `accepted = True`, official `strict_pass = True` — strictly diagnostic, original `M9.9` cost/score untouched); symmetric `1`-candidate sibling fallback wired into `M10.0`. | **CLOSED (`Docker Verified + M10.0 Wired`)** |
| `Q04` | `P1` | Deferred `E2_down` decomposition left `E1_down` (`D1`) monolithic solver non-callable on `c02/E4`; missing `d2_sep` propagated on `c03/E6`. | Corrected `docs/55_*` §8.1; symmetric executable monolithic fallback + lazy on-demand downstream residual projection (`100%` metered in `Scope-1`) implemented for `M10.0`. | **CLOSED (`Errata + M10.0 Wired`)** |
| `Q05` | `P1` | `--finalize-existing` overwrote `run_id`, `wall_sec`, and `source_tree_sha256_at_start` (`be7190fb...` vs freeze `34b01d19...`). | Separated write-once `execution_manifest.json` from `finalization_manifest.json` in runner; reconstructed exact freeze tree SHA (`915f469dc3f50a05...`) and raw call envelopes (`3,718.765s` confirm, `2,259.938s` dev). | **CLOSED (`Code + Provenance Ledger`)** |
| `Q06` | `P1` | Overstated `Hegel - Plain` (`p = 0.0625`) as statistically significant and `50/56` vs `54/56` (`p = 0.375`) as quality parity. | Calibrated `docs/55_*` & hub; sealed exact `x` sensitivity ladder (`p < 0.05` iff `x > 53,450`) and explicit non-inferiority gates for `M10.0`. | **CLOSED** |
| `Q07` | `P1` | `hydrate_separability_withdrawals_from_graph` checked binding IDs + state SHA but not code SHAs, `probe_id`, or prior colliding positive / joint negative witnesses. | Enforced full 6-link foreign-key & temporal witness-ledger verification (`test_q07` rejects self-consistent synthetic forgeries and accepts all `7/7` real sealed `M9.9` receipts). | **CLOSED** |
| `Q08` | `P2` | Legacy `v1` downgrade failed second `save -> reload` round-trip; `M9.8` `compatibility_inference` could write a downstream scientific negative. | Recomputes migrated `v2` `witness_content_sha256` & `unknown_requires_reverification` `receipt_id` (`snap_1 == snap_2` idempotent, `0` authority); blocked `compatibility_inference` from minting downstream scientific negatives. | **CLOSED** |
| `Q09` | `P1` | Zero host-seeded source code described too broadly as zero host-seeded decomposition. | Scoped throughout `docs/55_*` & hub as **online module formation under a host-prescribed two-stage interface** (open DAG discovery deferred). | **CLOSED** |
| `Q10` | `P1` | Fixed-separable `dn_obs`-only baseline & `step_05` ablation generalized to all plain registries. | Sealed `16/16` plain-dict `up+dn` candidate-SHA parity check and scoped `step_05` ablation to `dn_obs`-only vs selective `up+dn` condition expansion. | **CLOSED** |
| `Q11` | `P1` | Ideal grid count written as an exact LLM cost law. | Replaced with conditional `N_{sep,unseen} = |(U x D) \ (R u X)|` and empirical quality-constrained `ΔC(K)` / `K*` protocol. | **CLOSED** |
| `Q12` | `P2` | Cancelled pre-freeze dev diagnostic run omitted from numerical R&D sum. | Explicitly recorded as `unverifiable_cancelled_run = C_cancelled > 0` (`45,683,620 + x + C_cancelled` total physical `v2`). | **CLOSED** |
| `Q13` | `P2` | `E4/E5` `v2/v3` swap, `E1` `7/8, 6/8` vs cumulative `8/8, 7/8`, and elapsed wall-ms (`1.421x` total / `3.320x` coverage) vs CPU wording. | Fixed in `scripts/generate_m9_9_report_from_sealed_json.py`, `docs/55_*`, and hub. | **CLOSED** |
| `Q14` | `P2` | `R01–R16` ID drift in `docs/55` §7 and broken `54b` link in reuse hub. | Restored exact original `R01–R16` IDs (retaining retired `R14` and open `R15`) and fixed `54b` link in `docs/hegel_machine_reuse_hub_20261006.md`. | **CLOSED** |

---

## 2. Real Docker Back-Verification of All `30` Historical Cross-Diagonal Decomposition Cells (`Q01`)

- **Ledger**: `01_m99_q01_30_cross_diagonal_decomposition_probes.json` (`total_probe_wall_ms = 198,916.24 ms`, `0` paid LLM calls)
- **Outcome Breakdown Across `30` Probes (`15` Accepted Decompositions × `2` Off-Diagonal Cells)**:
  - `accepted = True`: **`0/30`**
  - `scientific_or_domain` rejected (`completed`): **`30/30`**
  - `schema_or_interface` rejected (`completed`): **`0/30`**
  - `infra_or_timeout`: **`0/30`**
  - Superseded original unexecuted `Hegel` `d0@E` witnesses (`errata_relation = supersedes_unexecuted_projection_with_executed_probe`): **`8/8`**

| Arm | Cluster (`Decomp Step`) | Cross-Diagonal Cell | `probe_id` | Status | Accepted | Failure Kind | Wall (`ms`) | Supersedes Unexecuted Witness |
|---|---|---|---|---|---|---|---:|---|
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c01` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_0add1ca13ca24097` | `completed` | `False` | `scientific_or_numerical` | `6804.4` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c01` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_660905d1d0874005` | `completed` | `False` | `scientific_or_numerical` | `6804.6` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c02` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_9601f2bb2ec74795` | `completed` | `False` | `scientific_or_numerical` | `6707.6` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c02` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_d24659a3178241e1` | `completed` | `False` | `scientific_or_numerical` | `6783.5` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c04` (`s2`) | `u0_plus_d0_on_current_E` | `xprobe_58dca380e2cd49fc` | `completed` | `False` | `scientific_or_numerical` | `6899.5` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c04` (`s2`) | `u0_plus_d1_on_Stage_S` | `xprobe_c335eeca7c114268` | `completed` | `False` | `scientific_or_numerical` | `6799.4` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c01` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_9e43c86c78ce470b` | `completed` | `False` | `scientific_or_numerical` | `6274.5` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c01` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_4bc69683c58e45eb` | `completed` | `False` | `scientific_or_numerical` | `6257.1` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c02` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_e23ac84a895443df` | `completed` | `False` | `scientific_or_numerical` | `6432.1` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c02` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_bd41a5b78d324e17` | `completed` | `False` | `scientific_or_numerical` | `6711.5` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c03` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_d2cccfe762164390` | `completed` | `False` | `scientific_or_numerical` | `6422.2` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c03` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_b68f0a6bcdce4ced` | `completed` | `False` | `scientific_or_numerical` | `6332.1` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c04` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_dcd2c3efe94d43c7` | `completed` | `False` | `scientific_or_numerical` | `6150.6` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c04` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_d063352996064440` | `completed` | `False` | `scientific_or_numerical` | `6221.2` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c01` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_bcb66ea656944fa9` | `completed` | `False` | `scientific_or_numerical` | `6621.6` | `wit_0002_bind_alpha_m99_c01_dn_b00_5a38161b` |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c01` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_231ce31383604457` | `completed` | `False` | `scientific_or_numerical` | `6590.1` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c02` (`s2`) | `u0_plus_d0_on_current_E` | `xprobe_bbc23825e55a42c4` | `completed` | `False` | `scientific_or_numerical` | `6512.7` | `wit_0002_bind_alpha_m99_c02_dn_b00_ed77e0c2` |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c02` (`s2`) | `u0_plus_d1_on_Stage_S` | `xprobe_10fa95a6a4be4936` | `completed` | `False` | `scientific_or_numerical` | `6611.7` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c03` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_7dee6323006c405b` | `completed` | `False` | `scientific_or_numerical` | `6589.3` | `wit_0002_bind_alpha_m99_c03_dn_b00_03b4a8a3` |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c03` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_28c126ae49d4445b` | `completed` | `False` | `scientific_or_numerical` | `6774.5` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c04` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_77d4acd78740449e` | `completed` | `False` | `scientific_or_numerical` | `6791.2` | `wit_0002_bind_alpha_m99_c04_dn_b00_3b352674` |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c04` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_080483c2d29f4a76` | `completed` | `False` | `scientific_or_numerical` | `6619.5` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c01` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_46a3596f507449a2` | `completed` | `False` | `scientific_or_numerical` | `6529.4` | `wit_0002_bind_beta_m99_c01_dn_b00_dc9859ed` |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c01` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_ec0c29b4ae0445ef` | `completed` | `False` | `scientific_or_numerical` | `6638.4` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c02` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_94606e0c92e54451` | `completed` | `False` | `scientific_or_numerical` | `7091.9` | `wit_0002_bind_beta_m99_c02_dn_b00_c95627f7` |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c02` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_0acbdf212bfd4d1f` | `completed` | `False` | `scientific_or_numerical` | `6786.9` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c03` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_45e352d518f346f6` | `completed` | `False` | `scientific_or_numerical` | `6783.1` | `wit_0002_bind_beta_m99_c03_dn_b00_83808d89` |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c03` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_0d3bdab1c2e240f9` | `completed` | `False` | `scientific_or_numerical` | `6858.0` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c04` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_c867ac26ea5549f6` | `completed` | `False` | `scientific_or_numerical` | `6666.3` | `wit_0002_bind_beta_m99_c04_dn_b00_387320b1` |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c04` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_b234458c7732465d` | `completed` | `False` | `scientific_or_numerical` | `6851.3` | — |

---

## 3. Provenance Reconciliation (`Q05`), Unknown `x` Sign-Flip Ladder (`Q06`), & `c04/E7` Container Check (`Q03`)

1. **Git Tree & Timing Separation (`Q05`)**:
   - Declared paid-run freeze commit `7624c43bf72d283e82442504eff35ff6ee02b40d` (`2700` `.py` files): `915f469dc3f50a05c84e5065f1b03e99cdd80cf661bf4826dab16ff8e98af2c0`
   - Post-freeze finalizer tree recorded in `m9_9_continual_summary.json`: `be7190fbfab8df60c5da023c3133f59fa0f71c6dde95ebc967367de3fd9d40bc`
   - Raw mirrored call time envelope: Confirmation `3,718.765s` (`m99_confirmation_20261127T193020Z` vs finalizer `219.26s`); Dev `2,259.938s` (`m99_dev_20261127T183538Z`).
2. **Container Verification of Prefix-Available `bind_alpha_m99_c04_up_b02` on `alpha_m99_c04 / E7` (`Q03`)**:
   - Direct `56/24` upstream component probe: `execution_status = completed`, **`chosen_up_holdout_rmse = 0.00007655`** (`<= 0.03`).
   - Composed `(bind_alpha_m99_c04_up_b02 + bind_alpha_m99_c04_dn_b03)` 3-layer public gate: **`accepted = True`** (`6331.4 ms`).
   - Official private verifier (diagnostic check only; original `M9.9` confirmation score/cost unchanged): **`strict_pass = True`** (`5192.31 ms`).
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
