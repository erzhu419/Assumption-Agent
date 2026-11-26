# Hegel M9.4.1 Deterministic Closure, Auto-Generated Errata & Counterfactual Attribution Addendum

> **Machine-Generated Provenance Notice**: Every numerical figure in this document is deterministically populated from sealed JSON evidence via `scripts/generate_m9_4_1_audit_closure_and_counterfactual_receipts.py` (`0` manual copy-paste numbers, `0` paid LLM calls).

## 1. Machine-Verified Errata for `docs/48b` and `docs/50` (`A10` Closed)

| Item | Corrected Machine-Verified Value | Source JSON |
| :--- | :--- | :--- |
| **`M9.3` 8-Episode Successor Proxy (`B / K / E`)** | `B = 1,950,809`, `K = 1,909,591`, `E = 1,927,667` (`E - B = -23,142`, `-1.1863%`; `E - K = +18,076`) | `m9_3_online_source_freeze_lifecycle_20261126/**/00_episode_summary.json` |
| **`M9.3` New-Source `L2` Positive Episodes** | `[{"episode": "rep_01/equal_source_executable_library/rec02_R__original_separable_kronecker_kkt_heldout_recovery", "arm": "stage_R", "calls": 3}, {"episode": "rep_01/suc01_C__viscoplastic_bingham_overstress_counterexample_revision", "arm": "equal_source_executable_library", "calls": 1018}, {"episode": "rep_02/suc02_C__nonseparable_vortex_cross_coupling_counterexample_revision", "arm": "equal_source_executable_library", "calls": 3}]` (`rep_02/suc02_C` has `3` calls, NOT `suc02_T`) | `m9_3_source_aware_l1_l2_adoption_funnel.json` |
| **`M9.3` Unknown-Usage Timeout Files** | `[{"rel_path": "stage_S_sources/attempt_01/src02__kronecker_sylvester_multishift_and_box_simplex_kkt_source/01b_provider_attempts/attempts_003.json", "call_seq": 3, "usage_unknown_attempt_count": 1}, {"rel_path": "stage_S_sources/attempt_01/src02__kronecker_sylvester_multishift_and_box_simplex_kkt_source/01b_provider_attempts/attempts_012.json", "call_seq": 12, "usage_unknown_attempt_count": 1}, {"rel_path": "stage_TC_successors/rep_01/equal_source_knowledge_only/suc01_C__viscoplastic_bingham_overstress_counterexample_revision/01b_provider_attempts/attempts_011.json", "call_seq": 11, "usage_unknown_attempt_count": 1}]` | `m9_3_online_source_freeze_lifecycle_20261126/**/attempts_*.json` |
| **`M9.3` Historical Stage C Split Parent** | Old wrong parent: `node_auto_07_control_panel_deconfounding`; true `src01` parent: `node_auto_10_hereditary_spectrum_identifi` | `m9_3_post_counterexample_bifurcated_bundle.json` |
| **`M9.4` Raw Provider Attempt Coverage** | `321` call summaries total: `291` raw episode `attempts_*.json` archived in bundle + `30` compiler summaries whose raw stage paths pointed to host temp (`fixed in M9.4.1 via immediate_archive_dir`) | `all_provider_call_receipts.json` |
| **`M9.4` Superseded Development Run Disclosure** | Commit `d44b3314f557`: `superseded_development_run_evidence_unrecoverable_on_disk` — `Transient child mechanism `mech_branch_default_cal_*` remained attached after `ep01`, suppressing `AncestorReweight` (`0/16` updates); directory was overwritten before unique-run-dir enforcement (`Gate C`). Development token cost is explicitly marked unknown/excluded from single-run `6,825,968` proxy.` | `gate_c_reconciliation_and_errata.json` |

## 2. Summary of `M9.4.1` Gates `A–D` (`0` LLM Calls)

- **Gate A (Shared Verification Kernel & Full 7-Coefficient Handoff)**:
  - `hereditary_weights = zeros(7)` rejected by public gate: `True` and private verifier: `True`.
  - `hereditary_weights[2:7] = 1_000_000` rejected by public gate: `True` and private verifier: `True`.
  - Offline vNext regrade of all `34` sealed `M9.4` episodes: `34 / 34` pass the full 7-coefficient + canonical KKT shared kernel (`0` discrepancies).
- **Gate B (Fail-Closed 3-Valued Guards, Binding Foreign Keys, AST Domain Guards, Stratified `F08`, `F05` Exact-AND-Reachable)**:
  - All 6 guard edge cases (`missing_yield`, `nan_history`, `nan_yield`, `missing_history`, `valid_subyield`, `residual_damage`): `{"valid_subyield": {"state": "TRUE", "measured_value": 0.0}, "missing_yield": {"state": "UNKNOWN", "measured_value": null}, "nan_history_valid_hold": {"state": "UNKNOWN", "measured_value": null}, "nan_yield_valid_history": {"state": "UNKNOWN", "measured_value": null}, "missing_history_current_below": {"state": "UNKNOWN", "measured_value": null}, "valid_history_residual_damage": {"state": "FALSE", "measured_value": 0.4}}`.
  - Mismatched SHA rejected: `True`; invalid qualification rejected: `True`; `TypedMechanism` registered in `graph.mechanisms`: `True`.
  - AST `Program("domain_guard", ...)` raises `DomainViolationError` outside domain: `True`.
  - Stratified per-function `F08` rejects `24 stable + 1 unseeded` module: `True`.
  - `F05` adapted reachable module classified as `adapted_closure_reachable` (not `exact`): `True`.
- **Gate D (Real `Stage S` Counterfactual Attribution)**:
  - `Real Stage S solvers already self-calibrate when full calibration sweeps are included in every batch (`alias-only` passes Alpha G, Alpha R, Beta G; Beta R passes with the same public additive `V_background` adapter without M8). Therefore M9.5 MUST provide Equal-Tool Auto-Replay to ALL arms AND separate immediate-calibration-sufficient negative-control steps from sparse-current-evidence history-necessary steps under unified output schemas!`
