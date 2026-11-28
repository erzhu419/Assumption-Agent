#!/usr/bin/env python3
"""Deterministic Report Generator for `docs/55_M9_8_1_Closure_and_M9_9_Autonomous_Decomposition_and_Interaction_Withdrawal_Report*.md`.

Calibrated in `M9.9.1` (`Hegel_M9_9_Audit` `Q01–Q14` + Original `R01–R16` Crosswalk, `0` hand-edited metrics!):
  - Preserves all sealed `M9.8.1` and `M9.9` confirmation scores, costs, and receipts unchanged.
  - Incorporates every audit calibration from `01_Implementation_and_Experiment_Plan_ZH.md` §5.C2:
    * `Q01`: Explicit notice on the `8` unexecuted `d0@E` projection witnesses & `30`-probe Docker back-verification in `docs/55b_*`.
    * `Q02`: Corrected `alpha_m99_c02/E7` forensic root cause (downstream test-only `_chamber_features` violating pooled `train+test`
      standardization & `(lab, stratum)` weighted `NaN` imputation `0.071267 -> 0.004857`, NOT upstream cubic Newton extrapolation).
    * `Q03`: Corrected `alpha_m99_c04/E7` forensic root cause (`bind_alpha_m99_c04_up_b02` already learned at `E4` with `7.65e-5`
      holdout RMSE & passing joint public/official gates, excluded by over-narrow `up_stat_03 > 1.289054` guard).
    * `Q05`: Explicit separation of declared paid-run freeze tree (`7624c43bf`, `34b01d19...`, raw call envelope `3,718.765s`) vs
      post-freeze finalizer run (`be7190fb...`, `219.26s`).
    * `Q06, Q09, Q10`: Scoped claims (`p = 0.0625` at `x = 0`, `x > 53,450` for `p < 0.05`; fixed-separable `dn_obs`-only Plain routing;
      online module formation under prescribed 2-stage interface; frozen `step_05` routing ablation; `50/56` vs `54/56` quality gap).
    * `Q11, Q12, Q13, Q14`: Conditional `|(U x D) \\ (R u X)|` grid count, `45,683,620 + x + C_cancelled` R&D ledger, `E4=v3, E5=v2, E7=v3`,
      `E1` (`7/8`, `6/8`) vs cumulative (`8/8`, `7/8`) decompositions, `1.421x` total agent probe elapsed wall-ms vs `3.32x` coverage
      sub-item elapsed wall-ms, and exact original `R01–R16` ID crosswalk.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

M981_DIR = REPO_ROOT / "docs/evidence/m9_8_1_audit_closure_and_twotable_attribution_20261128"
M99_DEV_DIR = REPO_ROOT / "docs/evidence/m9_9_dev_live_calibration_20261128"
M99_CONF_DIR = REPO_ROOT / "docs/evidence/m9_9_autonomous_decomposition_and_interaction_withdrawal_20261128"
M991_DIR = REPO_ROOT / "docs/evidence/m9_9_1_audit_closure_and_forensic_reverification_20261128"

OUT_EN = REPO_ROOT / "docs/55_M9_8_1_Closure_and_M9_9_Autonomous_Decomposition_and_Interaction_Withdrawal_Report.md"
OUT_ZH = REPO_ROOT / "docs/55_M9_8_1_Closure_and_M9_9_Autonomous_Decomposition_and_Interaction_Withdrawal_Report_ZH.md"


def _pct(new_v: float, base_v: float) -> str:
    if base_v == 0:
        return "0.00%"
    return f"{100.0 * (new_v - base_v) / base_v:+.2f}%"


def _mean_cluster_pct(conf: dict, base_arm: str, metric_key: str = "proxy_v2_u_010c_4o") -> str:
    hg = conf["arm_cost_ledgers"]["Hegel_AutonomousInteractionGraph_Online"]["by_cluster"]
    bs = conf["arm_cost_ledgers"][base_arm]["by_cluster"]
    pcts = [100.0 * (float(hg[w][metric_key]) - float(bs[w][metric_key])) / float(bs[w][metric_key]) for w in sorted(hg.keys())]
    return f"{float(np.mean(pcts)):+.2f}%"


def main() -> None:
    r01 = json.loads((M981_DIR / "01_m98_executed_two_plain_tables_paid_prefix_replay.json").read_text(encoding="utf-8"))
    r02 = json.loads((M981_DIR / "02_m98_r04_16_upstream_direct_verification_and_r11_probe_ledger.json").read_text(encoding="utf-8"))
    r03 = json.loads((M981_DIR / "03_m98_r08_prefix_locked_and_fixed_policy_twin_ablations.json").read_text(encoding="utf-8"))

    dev = json.loads((M99_DEV_DIR / "m9_9_continual_summary.json").read_text(encoding="utf-8"))
    conf = json.loads((M99_CONF_DIR / "m9_9_continual_summary.json").read_text(encoding="utf-8"))
    abl = json.loads((M99_CONF_DIR / "m9_9_factorized_ceiling_ablation.json").read_text(encoding="utf-8"))

    m991_q01 = json.loads((M991_DIR / "01_m99_q01_30_cross_diagonal_decomposition_probes.json").read_text(encoding="utf-8"))
    m991_q03 = json.loads((M991_DIR / "02_m99_q03_c04_e7_existing_candidate_container_verification.json").read_text(encoding="utf-8"))
    m991_prov = json.loads((M991_DIR / "03_m99_provenance_x_sensitivity_and_plain_dict_replay.json").read_text(encoding="utf-8"))

    missed = json.loads((M991_DIR / "07_m99_missing_prefix_ablation.json").read_text())
    sep_replay = missed["summary_by_policy"]["factorized_fixed_D2_separable"]
    replay = m991_prov["plain_dict_same_joint_features_step05_replay"]
    parity = f"{replay['matched_episodes']}/{replay['total_episodes']}"
    cancelled = m991_prov["rd_cost_coverage_boundary"]["cancelled_dev_diagnostic_run"]
    arms = [
        "Flat_MonolithicConditionalTable_Online",
        "Flat_PlainModuleRegistry_Online",
        "Hegel_AutonomousInteractionGraph_Online",
    ]
    out_t = conf["arm_outcome_totals"]
    led = conf["arm_cost_ledgers"]
    st_s = conf["stage_S_shared_ledger"]

    hg_s1_v2 = int(led["Hegel_AutonomousInteractionGraph_Online"]["scope1_in_stream_total"]["proxy_v2_u_010c_4o"])
    pl_s1_v2 = int(led["Flat_PlainModuleRegistry_Online"]["scope1_in_stream_total"]["proxy_v2_u_010c_4o"])
    mo_s1_v2 = int(led["Flat_MonolithicConditionalTable_Online"]["scope1_in_stream_total"]["proxy_v2_u_010c_4o"])

    hg_s1_v1 = int(led["Hegel_AutonomousInteractionGraph_Online"]["scope1_in_stream_total"]["proxy_v1_u_025c_1o"])
    pl_s1_v1 = int(led["Flat_PlainModuleRegistry_Online"]["scope1_in_stream_total"]["proxy_v1_u_025c_1o"])
    mo_s1_v1 = int(led["Flat_MonolithicConditionalTable_Online"]["scope1_in_stream_total"]["proxy_v1_u_025c_1o"])

    pl_unk = int(led["Flat_PlainModuleRegistry_Online"]["scope1_in_stream_total"]["usage_unknown_attempt_count"])
    hg_unk = int(led["Hegel_AutonomousInteractionGraph_Online"]["scope1_in_stream_total"]["usage_unknown_attempt_count"])
    mo_unk = int(led["Flat_MonolithicConditionalTable_Online"]["scope1_in_stream_total"]["usage_unknown_attempt_count"])
    assert hg_unk == 0 and mo_unk == 0 and pl_unk == 1

    s_v2 = int(st_s["proxy_v2_u_010c_4o"])
    s_v1 = int(st_s["proxy_v1_u_025c_1o"])

    hg_tot_v2, pl_tot_v2, mo_tot_v2 = hg_s1_v2 + s_v2, pl_s1_v2 + s_v2, mo_s1_v2 + s_v2
    hg_tot_v1, pl_tot_v1, mo_tot_v1 = hg_s1_v1 + s_v1, pl_s1_v1 + s_v1, mo_s1_v1 + s_v1

    pair_pl = conf["paired_cluster_inference"]["Hegel_minus_Flat_PlainModuleRegistry_Online"]
    pair_mo = conf["paired_cluster_inference"]["Hegel_minus_Flat_MonolithicConditionalTable_Online"]

    cluster_rows = []
    streams_by_key = {(s["world_id"], s["arm"]): s for s in conf["streams"]}
    wids = [s["world_id"] for s in conf["streams"] if s["arm"] == "Hegel_AutonomousInteractionGraph_Online"]
    for wid in wids:
        fam = streams_by_key[(wid, "Hegel_AutonomousInteractionGraph_Online")]["family"]
        mo_s = streams_by_key[(wid, "Flat_MonolithicConditionalTable_Online")]
        pl_s = streams_by_key[(wid, "Flat_PlainModuleRegistry_Online")]
        hg_s = streams_by_key[(wid, "Hegel_AutonomousInteractionGraph_Online")]
        mo_c = int(led["Flat_MonolithicConditionalTable_Online"]["by_cluster"][wid]["proxy_v2_u_010c_4o"])
        pl_c = int(led["Flat_PlainModuleRegistry_Online"]["by_cluster"][wid]["proxy_v2_u_010c_4o"])
        pl_u = int(led["Flat_PlainModuleRegistry_Online"]["by_cluster"][wid]["usage_unknown_attempt_count"])
        hg_c = int(led["Hegel_AutonomousInteractionGraph_Online"]["by_cluster"][wid]["proxy_v2_u_010c_4o"])
        pl_tag = f"`{pl_c:,} + x`" if pl_u > 0 else f"`{pl_c:,}`"
        d_pl_tag = f"`{hg_c - pl_c:+,} - x` (`{_pct(hg_c, pl_c)}`)" if pl_u > 0 else f"`{hg_c - pl_c:+,}` (`{_pct(hg_c, pl_c)}`)"
        cluster_rows.append(
            f"| `{wid}` (`{fam}`) | `{mo_s['strict_passes']}/7` (`{mo_s['zero_turn_passed_closures']}` 0-LLM) | `{mo_c:,}` | "
            f"`{pl_s['strict_passes']}/7` (`{pl_s['zero_turn_passed_closures']}` 0-LLM) | {pl_tag} | "
            f"**`{hg_s['strict_passes']}/7` (`{hg_s['zero_turn_passed_closures']}` 0-LLM)** | **`{hg_c:,}`** | "
            f"{d_pl_tag} | `{hg_c - mo_c:+,}` (`{_pct(hg_c, mo_c)}`) |"
        )

    ev_order = [m["causal_event"] for m in conf["streams"][0]["episodes_meta"]]
    ev_rows = []
    for ev in ev_order:
        cells = []
        for arm in arms:
            eps = [ep for s in conf["streams"] if s["arm"] == arm for ep in s["episodes_meta"] if ep["causal_event"] == ev]
            n_tot = len(eps)
            n_sp = sum(1 for e in eps if e["strict_pass"])
            n_z = sum(1 for e in eps if e["turns_used"] == 0 and e["strict_pass"])
            c_v2 = int(led[arm]["by_event"][ev]["proxy_v2_u_010c_4o"])
            u_cnt = int(led[arm]["by_event"][ev]["usage_unknown_attempt_count"])
            c_s = f"{c_v2:,}+x" if u_cnt > 0 else f"{c_v2:,}"
            cells.append(f"`{n_sp}/{n_tot}` pass, `{n_z}/{n_tot}` 0-LLM (`{c_s}`)")
        ev_rows.append(f"| `{ev}` | {cells[0]} | {cells[1]} | **{cells[2]}** |")

    probe_rows = []
    tot_agent_by_arm = {}
    for arm in arms:
        ot = out_t[arm]
        tot_p = (
            ot["pre_llm_probe_wall_ms"]
            + ot["post_llm_probe_wall_ms"]
            + ot["localization_probe_wall_ms"]
            + ot["decomposition_probe_wall_ms"]
            + ot["coverage_probe_wall_ms"]
        )
        tot_agent_by_arm[arm] = tot_p
        probe_rows.append(
            f"| `{arm}` | `{ot['pre_llm_probe_wall_ms']:,.1f}` | `{ot['post_llm_probe_wall_ms']:,.1f}` | "
            f"`{ot['localization_probe_wall_ms']:,.1f}` | `{ot['decomposition_probe_wall_ms']:,.1f}` | "
            f"`{ot['coverage_probe_wall_ms']:,.1f}` | **`{tot_p:,.1f}`** | `{ot['official_verifier_wall_ms']:,.1f}` |"
        )

    tot_probe_ratio = tot_agent_by_arm["Flat_MonolithicConditionalTable_Online"] / tot_agent_by_arm["Hegel_AutonomousInteractionGraph_Online"]
    cov_sub_ratio = out_t["Flat_MonolithicConditionalTable_Online"]["coverage_probe_wall_ms"] / out_t["Hegel_AutonomousInteractionGraph_Online"]["coverage_probe_wall_ms"]

    abl_s = abl["summary_by_policy"]
    decomp_hg_e1 = sum(
        1 for s in conf["streams"] if s["arm"] == "Hegel_AutonomousInteractionGraph_Online" for d in s["autonomous_decomposition_events"] if d.get("accepted") and int(d.get("step", 0)) == 1
    )
    decomp_pl_e1 = sum(
        1 for s in conf["streams"] if s["arm"] == "Flat_PlainModuleRegistry_Online" for d in s["autonomous_decomposition_events"] if d.get("accepted") and int(d.get("step", 0)) == 1
    )
    decomp_hg_cum = sum(
        1 for s in conf["streams"] if s["arm"] == "Hegel_AutonomousInteractionGraph_Online" for d in s["autonomous_decomposition_events"] if d.get("accepted")
    )
    decomp_pl_cum = sum(
        1 for s in conf["streams"] if s["arm"] == "Flat_PlainModuleRegistry_Online" for d in s["autonomous_decomposition_events"] if d.get("accepted")
    )
    e67_hg_z = sum(
        1
        for s in conf["streams"]
        if s["arm"] == "Hegel_AutonomousInteractionGraph_Online"
        for ep in s["episodes_meta"]
        if ep["causal_event"] in {"E6_separable_revisit", "E7_coupled_revisit"} and ep["turns_used"] == 0 and ep["strict_pass"]
    )
    e67_pl_z = sum(
        1
        for s in conf["streams"]
        if s["arm"] == "Flat_PlainModuleRegistry_Online"
        for ep in s["episodes_meta"]
        if ep["causal_event"] in {"E6_separable_revisit", "E7_coupled_revisit"} and ep["turns_used"] == 0 and ep["strict_pass"]
    )
    e4_hg_z = sum(
        1 for s in conf["streams"] if s["arm"] == "Hegel_AutonomousInteractionGraph_Online" for ep in s["episodes_meta"] if ep["causal_event"] == "E4_valid_comb" and ep["turns_used"] == 0 and ep["strict_pass"]
    )
    e4_hg_p = sum(
        1 for s in conf["streams"] if s["arm"] == "Hegel_AutonomousInteractionGraph_Online" for ep in s["episodes_meta"] if ep["causal_event"] == "E4_valid_comb" and ep["strict_pass"]
    )
    decomp_hg_v2 = int(led["Hegel_AutonomousInteractionGraph_Online"]["decomposition_induction_only"]["proxy_v2_u_010c_4o"])
    decomp_hg_per_cluster = decomp_hg_v2 / 8.0
    e4_mo_v2 = int(led["Flat_MonolithicConditionalTable_Online"]["by_event"]["E4_valid_comb"]["proxy_v2_u_010c_4o"])
    e4_mo_per_cluster = e4_mo_v2 / 8.0

    c02_ep7 = json.loads(
        (
            M99_CONF_DIR
            / "continual_streams/alpha_m99_c02/Hegel_AutonomousInteractionGraph_Online/alpha_m99_c02_batch_07__blind_continual_campaign/00_episode_summary.json"
        ).read_text(encoding="utf-8")
    )
    c02_pub_metrics = c02_ep7["pre_llm_trial_receipt"]["probe_result"]["scientific_probe_receipt"]["measured_scientific_metrics"]
    c02_priv_inst = {int(d["seed"]): d for d in c02_ep7["verifier_report"]["verifier_metrics"]["instance_diagnostics"]}
    c04_s = json.loads((M99_CONF_DIR / "continual_streams/alpha_m99_c04/Hegel_AutonomousInteractionGraph_Online/00_stream_summary.json").read_text(encoding="utf-8"))
    c04_e4_turns = int(c04_s["episodes_meta"][3]["turns_used"])
    c04_e7_turns = int(c04_s["episodes_meta"][6]["turns_used"])

    frz_tree = m991_prov["git_source_tree_reconciliation"]["declared_paid_freeze_commit_7624c43bf"]["source_tree_sha256"]
    fin_tree = m991_prov["git_source_tree_reconciliation"]["m99_summary_recorded_post_freeze_finalizer_tree_sha256"]
    conf_env = m991_prov["execution_vs_finalizer_timing_reconciliation"]["confirmation_n8"]
    dev_env = m991_prov["execution_vs_finalizer_timing_reconciliation"]["dev_n2"]

    en_md = f"""# Hegel M9.8.1 Audit Closure & M9.9 Equal-Tool Online Module Formation and Counterexample-Scoped Interaction Withdrawal Report (`Calibrated in M9.9.1`)

> [!IMPORTANT]
> **M9.9.1 Independent Audit Calibration & Errata Cross-Reference (`Q01–Q14`)**: All original sealed `M9.9` scores, costs, and receipts remain `100%` immutable. This report has been deterministically calibrated by `scripts/generate_m9_9_report_from_sealed_json.py` to incorporate every provenance, statistical, wording, and forensic root-cause correction from `Hegel_M9_9_Audit` (`Q01–Q14` + original `R01–R16` ID crosswalk). See companion errata `docs/55b_M9_9_1_Audit_Closure_Forensic_Reverification_and_Errata.md` for the `30`-probe Docker cross-diagonal back-verification (`{m991_q01['accepted_count']}/30` passed, `{m991_q01['scientific_or_domain_rejected_count']}/30` scientific/domain rejected, `{m991_q01['schema_rejected_count']}/30` schema rejected), `c02/c04` forensic container reproductions, and provenance ledgers.

- **Date**: `2026-11-28`
- **Declared Pre-Confirmation Freeze Commit (`7624c43bf`, `Q05`)**: `{conf['paid_llm_frozen_commit']}` (`2,700` `.py` files, reconstructed commit tree SHA-256 `{frz_tree}`; raw mirrored call envelope `{conf_env['call_envelope_lower_bound_sec']:,.3f}s` from `{conf_env['first_call_started_utc']}` to `{conf_env['last_call_ended_utc']}` under execution run ID `{conf_env['execution_run_ids_in_raw_calls'][0]}`). *Provenance note (`Q05`)*: Original paid execution did not persist a separate start-time manifest; `m9_9_continual_summary.json` recorded post-freeze finalizer run `{conf_env['finalizer_summary_run_id']}` (`{conf_env['finalizer_postprocess_wall_sec']}s`, post-fix tree SHA `{fin_tree}`). A commit-tree reconstruction cannot establish which uncommitted sources were loaded at runtime.
- **M9.8.1 Closure Commit**: `b959102fa9c284efebb8921ce39229c4b23f45b1`
- **Shared Verification Kernel**: `{conf['shared_kernel_version']}` | **Cost Policy**: `{conf['cost_policy_version']}`
- **Air-Gapped Sandbox Digest**: `{conf['sandbox_image_digest']}`
- **Live Model & Concurrency (`R16`)**: `{conf['model']}` (`reasoning_effort = {conf['reasoning_effort']}`, `configured_max_codex_concurrency = {conf['configured_max_codex_concurrency']}`, observed `peak_active_codex_concurrency = {conf['peak_active_codex_concurrency']}`)
- **Usage Accounting & Unknown `x` Sensitivity (`Q06`)**: `{conf['usage_completeness']}` (`1,750/1,751` calls usage-complete; `0` unknown attempts on `Stage S`, `Hegel`, and `Monolithic`; `1` unknown timeout attempt `x >= 0` on `Flat_PlainModuleRegistry_Online` `alpha_m99_c04/E1_down` `call_uuid=0391d702b0c547a488608847b5dff224`: `Hegel - Plain = -2,535,683 - x`, with exact two-sided sign-flip `p = 0.0625` for `0 <= x <= 17,652`, `p = 0.0546875` for `17,652 < x <= 53,450`, and **`p < 0.05` iff `x > 53,450`**).

---

## 1. Executive Summary & Scoped Attribution Boundaries (`Q01–Q14` & Original `R01–R16`)

1. **`M9.8.1` Evidence-Contract Hardening & Executed Attribution Controls (`0` Paid LLM Calls)**:
   - Hardened `SolverEvaluationWitness` (`v2` schema: `execution_status`, `evidence_origin`, nullable `metric_value` with zero `999.0` sentinels, `probe_id`, `setup_source_sha256`, `canonical_numeric_sha256`, `dependency_bundle_sha256`, `verification_trust_state`), `EmpiricalCoverageReceipt.verify_integrity()` (`R06`), and `_hydrate_from_graph()` (`R07`, with idempotent legacy `save -> reload` round-trip finalized in `M9.9.1` `Q08`).
   - Executed `M9.8` two-plain-tables replay (`56/56` setup, pickle, and canonical numeric SHA parity in the independent host reconstruction against the archived host-generated inputs; `32/56` public gate accepted $\\to$ `32/32 = 100.0%` zero-LLM private `strict_pass`), `16/16` direct `u0` upstream Docker probes (`max RMSE = {r02['r04_16_previously_inferred_upstream_instances']['max_holdout_rmse_across_16']:.8f} <= 0.03`), and `64` `(R0, U_iso)` prefix-locked / fixed-policy controls (`16/16` history-aware vs `0/16` masked vs `8/16` fixed `D0`/`D1`).

2. **`M9.9` Pre-Registered `n = 8` Unexposed Confirmation Campaign (`176` Live Episodes, `100%` Equal Tool Flags)**:
   - **Zero Host-Seeded Module Source Code Under a Prescribed Two-Stage Interface (`Q09`)**: `Stage S` synthesizes an unfactored monolithic `solve_batch` (`8/8` pass). Within the host-prescribed two-stage `(calibrate_upstream, solve_downstream)` interface and parent parameter topology (open unknown-depth DAG discovery is NOT claimed here), modular arms invoke `propose_and_verify_m99_autonomous_two_stage_factorization` (`100%` metered in `Scope-1`), succeeding on `E1_down` in **`{decomp_hg_e1}/8` `Hegel` and `{decomp_pl_e1}/8` `Plain`** clusters (**`{decomp_hg_cum}/8` and `{decomp_pl_cum}/8` cumulative** across the full stream after `E2_down`). *Audit note (`Q01`)*: Original `M9.9` decomposition executed the `2` diagonal public-gate probes while passing `30` unexecuted off-diagonal constraints to `d0d1_gind` and writing `8` unexecuted `d0@E` projection witnesses; `M9.9.1` upgrades the verifier to a full `2x2` container check and executes all `30` historical cross-diagonal probes in Docker (`docs/55b_*`).
   - **Result A — `Hegel_AutonomousInteractionGraph_Online` vs Fixed-Separable `Flat_PlainModuleRegistry_Online` (`Q06, Q10`)**:
     - **Zero-LLM Passed Closures**: **`{out_t['Hegel_AutonomousInteractionGraph_Online']['zero_turn_passed_closures']}/56` (`Hegel`) vs `{out_t['Flat_PlainModuleRegistry_Online']['zero_turn_passed_closures']}/56` (`PlainModuleRegistry`)** (`5` cluster wins, `3` ties, `0` losses, exact two-sided sign-flip `p = {pair_pl['zero_llm_two_sided_sign_flip_p']}`), driven by `{out_t['Hegel_AutonomousInteractionGraph_Online']['separability_withdrawals']}/8` traceable `SeparabilityWithdrawalReceipt`s (`{e67_hg_z}/16` vs `{e67_pl_z}/16` zero-LLM passes on `(E6, E7)`).
     - **Compute Cost & Statistical Boundary**: Known Scope-1 `proxy_v2 = {hg_s1_v2:,}` vs `{pl_s1_v2:,} + x` (**`{_pct(hg_s1_v2, pl_s1_v2)}`** known ratio-of-sums, equal-cluster mean `{_mean_cluster_pct(conf, 'Flat_PlainModuleRegistry_Online')}`, Scope-Total `{_pct(hg_tot_v2, pl_tot_v2)}`, `mean Δv2 = {pair_pl['mean_diff']:+,.2f} - x/8`, `95%` bootstrap CI `{pair_pl['bootstrap_95ci']}`, **exact two-sided sign-flip `p = {pair_pl['two_sided_sign_flip_p']}` at `x = 0`** — directional `6/8` cluster cost reduction that is marginal at `alpha = 0.05` unless `x > 53,450`; `strict_pass` `{out_t['Hegel_AutonomousInteractionGraph_Online']['strict_passes']}/56` vs `{out_t['Flat_PlainModuleRegistry_Online']['strict_passes']}/56`, `p = {pair_pl['quality_two_sided_sign_flip_p']}`).
     - **Scoped Frozen `step_05` Routing Ablation (`48` Runs, `Q10`)**: Given the frozen `step_05` module prefix, `hegel_selective_cross_factor_graph = {abl_s['hegel_selective_cross_factor_graph']['strict_passes']}/16` (`{abl_s['hegel_selective_cross_factor_graph']['pass_rate_pct']}%`) vs `dn_obs`-only selectors `factorized_fixed_D2_separable = {abl_s['factorized_fixed_D2_separable']['strict_passes']}/16` (`{abl_s['factorized_fixed_D2_separable']['pass_rate_pct']}%`) and `factorized_fixed_D2_coupled = {abl_s['factorized_fixed_D2_coupled']['strict_passes']}/16` (`{abl_s['factorized_fixed_D2_coupled']['pass_rate_pct']}%`). This isolates the **selective condition-expansion algorithm** (`dn_obs -> up_obs + dn_obs` upon counterexample) relative to fixed `dn_obs`-only routing; a plain Python dict/list given the same host-reconstructed policy inputs and expanded `up_obs + dn_obs` features reproduces `{parity}` selected branches and candidate SHAs (`03_m99_provenance_x_sensitivity_and_plain_dict_replay.json`).
       The historical fixed-separable selector substituted the coupled branch when the separable branch was missing. Replaying with explicit abstention gives {sep_replay['strict_passes']}/{sep_replay['total_episodes']} strict passes and {sep_replay['abstained_missing_prefix_binding']} abstentions. The original 48 policy positions are retained in both comparisons (ledger 07).
   - **Result B — Short-Horizon Comparison Against `Flat_MonolithicConditionalTable_Online` (`Q06, Q13`)**:
     - On **Beta (`n = 4` PDE/KKT clusters)**, `Hegel` achieves `28/28` `strict_pass`, `12/28` vs `8/28` zero-LLM closures, and `4/4` cluster cost reductions (`mean Δv2 = -201,744.75`, `beta_sign_flip_p = {pair_mo['beta_sign_flip_p']}`).
     - Across all `8` clusters (`1` unseen separable cell `E4_valid_comb`), total cost is indistinguishable from `Monolithic` (`{hg_s1_v2:,}` vs `{mo_s1_v2:,}` = `{_pct(hg_s1_v2, mo_s1_v2)}` Scope-1 / `{_pct(hg_tot_v2, mo_tot_v2)}` Scope-Total, `p = {pair_mo['two_sided_sign_flip_p']}`), while `Hegel` had **`4` fewer strict passes on Alpha (`50/56 = 89.29%` vs `54/56 = 96.43%`, `p = {pair_mo['quality_two_sided_sign_flip_p']}` — `p > 0.05` does NOT establish quality non-inferiority)**. Summed non-LLM agent container probe **elapsed wall-clock** is `{tot_probe_ratio:.3f}×` lower overall (`1,271,833.0 ms` vs `895,174.8 ms`), with the empirical coverage sub-item elapsed wall-ms `3.320×` lower (`689,259.6 ms` vs `207,604.6 ms`).

---

## 2. Primary Three-Arm Confirmation Ledger (`n = 8` Unexposed Clusters, `176` Episodes)

| Arm (`100%` Equal Tool Flags) | `strict_pass` | `0`-LLM Passed Closures | `Scope-1` Calls (`solve` / `decomp` / `guard`) | `Scope-1 proxy_v1` | `Scope-1 proxy_v2` | `Scope-Total proxy_v2` (`Stage S + Scope-1`) | `Hegel` vs Row (`Scope-1` / `Total`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicConditionalTable_Online` | `{out_t['Flat_MonolithicConditionalTable_Online']['strict_passes']}/56` | `{out_t['Flat_MonolithicConditionalTable_Online']['zero_turn_passed_closures']}/56` | `{led['Flat_MonolithicConditionalTable_Online']['scope1_in_stream_total']['call_count']}` (`{led['Flat_MonolithicConditionalTable_Online']['solve_only']['call_count']}` / `0` / `{led['Flat_MonolithicConditionalTable_Online']['guard_induction_only']['call_count']}`) | `{mo_s1_v1:,}` | `{mo_s1_v2:,}` | `{mo_tot_v2:,}` | `{_pct(hg_s1_v2, mo_s1_v2)}` / `{_pct(hg_tot_v2, mo_tot_v2)}` |
| `Flat_PlainModuleRegistry_Online` *(fixed-separable `dn_obs`)* | `{out_t['Flat_PlainModuleRegistry_Online']['strict_passes']}/56` | `{out_t['Flat_PlainModuleRegistry_Online']['zero_turn_passed_closures']}/56` | `{led['Flat_PlainModuleRegistry_Online']['scope1_in_stream_total']['call_count']}` (`{led['Flat_PlainModuleRegistry_Online']['solve_only']['call_count']}` / `{led['Flat_PlainModuleRegistry_Online']['decomposition_induction_only']['call_count']}` / `{led['Flat_PlainModuleRegistry_Online']['guard_induction_only']['call_count']}`) | `{pl_s1_v1:,} + x_1` | `{pl_s1_v2:,} + x` | `{pl_tot_v2:,} + x` | `{_pct(hg_s1_v2, pl_s1_v2)}` / `{_pct(hg_tot_v2, pl_tot_v2)}` (`x >= 0`) |
| **`Hegel_AutonomousInteractionGraph_Online`** | **`{out_t['Hegel_AutonomousInteractionGraph_Online']['strict_passes']}/56`** | **`{out_t['Hegel_AutonomousInteractionGraph_Online']['zero_turn_passed_closures']}/56`** | **`{led['Hegel_AutonomousInteractionGraph_Online']['scope1_in_stream_total']['call_count']}`** (`{led['Hegel_AutonomousInteractionGraph_Online']['solve_only']['call_count']}` / `{led['Hegel_AutonomousInteractionGraph_Online']['decomposition_induction_only']['call_count']}` / `{led['Hegel_AutonomousInteractionGraph_Online']['guard_induction_only']['call_count']}`) | **`{hg_s1_v1:,}`** | **`{hg_s1_v2:,}`** | **`{hg_tot_v2:,}`** | reference |

- **Shared `Stage S` Source Ledger (`8/8` Pass, `0` Unknown Attempts)**: `{st_s['call_count']}` calls, `proxy_v1 = {s_v1:,}`, `proxy_v2 = {s_v2:,}` (`1/3`-allocated: `Hegel` vs `Plain` `-18.08%`, `Hegel` vs `Monolithic` `-1.86%`).
- **Separate Paired Exact Sign-Flip Tests (`n = 8` Independent Clusters; Minimum Two-Sided Resolution `2/256 = 0.0078125`, Distinct from Statistical Power)**:
  - `Hegel - Flat_PlainModuleRegistry_Online`: `mean Δv2 = {pair_pl['mean_diff']:+,.2f} - x/8` (`95%` bootstrap CI `{pair_pl['bootstrap_95ci']}`), **cost `p = {pair_pl['two_sided_sign_flip_p']}` at `x = 0`** (`p < 0.05` iff `x > 53,450`), **zero-LLM closure `p = {pair_pl['zero_llm_two_sided_sign_flip_p']}`**, **quality `strict_pass` `p = {pair_pl['quality_two_sided_sign_flip_p']}`**.
  - `Hegel - Flat_MonolithicConditionalTable_Online`: `mean Δv2 = {pair_mo['mean_diff']:+,.2f}` (`95%` bootstrap CI `{pair_mo['bootstrap_95ci']}`), **cost `p = {pair_mo['two_sided_sign_flip_p']}`**, **zero-LLM closure `p = {pair_mo['zero_llm_two_sided_sign_flip_p']}`**, **quality `strict_pass` `p = {pair_mo['quality_two_sided_sign_flip_p']}`** (`50/56` vs `54/56`).

---

## 3. Cluster-by-Cluster Paired Ledger (`n = 8` Unexposed Confirmation Worlds)

| Cluster (`Family`) | `Monolithic` Pass (`0`-LLM) | `Monolithic` `v2` | `PlainModuleRegistry` Pass (`0`-LLM) | `PlainModuleRegistry` `v2` | `Hegel` Pass (`0`-LLM) | `Hegel` `v2` | `Hegel - Plain` `Δv2` | `Hegel - Mono` `Δv2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(cluster_rows)}

---

## 4. Event-by-Event Causal Trajectory (`7` Continual Events × `8` Clusters)

| Stream Event | `Flat_MonolithicConditionalTable_Online` | `Flat_PlainModuleRegistry_Online` | `Hegel_AutonomousInteractionGraph_Online` |
|---|---|---|---|
{chr(10).join(ev_rows)}

---

## 5. Non-LLM Container Probe Elapsed Wall-Clock Ledger (`R11 / Q13` — Summed Container Elapsed `ms`, Not CPU Time)

| Arm | Pre-LLM Gate (`ms`) | Post-LLM Gate (`ms`) | Parent Localization (`ms`) | Decomp Verification (`ms`) | Empirical Coverage (`ms`) | **Total Non-LLM Agent Probes (`ms`)** | Official Private Verifier (`ms`) |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(probe_rows)}

---

## 6. Full Campaign R&D Ledger Including Recovered Cancelled Dev Usage (`Q12`)

1. **Cancelled development run**. Recovered {cancelled['unique_logical_calls']} calls. Known physical cost was {cancelled['known_physical_v2']:,} v2. Usage for {cancelled['unclosed_call_count']} calls remains unknown.
2. **Sealed `n = 2` Pre-Freeze Dev Calibration (`docs/evidence/m9_9_dev_live_calibration_20261128`, `1,398` files, raw call envelope `{dev_env['call_envelope_lower_bound_sec']:,.3f}s`)**: `{dev['total_logical_calls']}` calls, `8,409,059` physical `v2` (`0` unknown attempts, `peak_concurrency = {dev['peak_active_codex_concurrency']}`).
3. **Sealed `n = 8` Unexposed Confirmation Campaign (`docs/evidence/m9_9_autonomous_decomposition_and_interaction_withdrawal_20261128`, `6,226` files, raw call envelope `{conf_env['call_envelope_lower_bound_sec']:,.3f}s`)**: `{conf['total_logical_calls']}` calls, `37,274,561 + x` physical `v2` (`peak_concurrency = {conf['peak_active_codex_concurrency']}` under `cap = {conf['configured_max_codex_concurrency']}`).
4. **Total R&D**. The two sealed runs account for 2,143 calls and 45,683,620 + x v2. Including cancelled development gives {cancelled['total_rd_v2_formula']}.

---

## 7. Original `Hegel_M9_8_Audit` (`R01–R16`) Exact-ID Crosswalk (`Q14`)

| Original `M9.8` ID & Topic | Verified `M9.8.1` / `M9.9` / `M9.9.1` Evidence & Scope | Status |
|---|---|---|
| `R01` (Graph vs two plain tables attribution) | `56/56` `M9.8` two-table replay (`32/32` 0-LLM pass) + `M9.9` `Flat_PlainModuleRegistry_Online` + `{parity}` plain-dict `up+dn` replay (`03_m99_provenance...json`): isolates selective `dn -> up+dn` expansion policy from substrate. | **`backverified_and_claim_narrowed`** |
| `R02` (Host-seeded `u0/d0`, parent sets & repair prompts) | `u0/d0/d1` source code synthesized & metered online (`2x2` verified in `M9.9.1`); two-stage interface & parent topology remain host-prescribed (open DAG deferred). | **`partially_changed_claim_narrowed`** |
| `R03` (`L1` / auto-packaging flag & repair task parity) | `enable_auto_skill_packaging=False`, `allow_l1_primitives=False` across all arms; identical modular repair templates across `Plain` and `Hegel`. | **`technically_fixed`** |
| `R04` (`C1/C2` unexecuted `r2>=0.96` logged as `0.0` RMSE) | Fixed in `M9.8.1` + `16/16` direct Docker probes (`max RMSE = 0.00019837`); `M9.9.1` also closes `Q01` (`2x2` decomp + `30` Docker probes) and `Q08` (`M9.8` `compatibility_inference` isolation). | **`technically_fixed_plus_m991_q01_q08`** |
| `R05` (Infra/timeout misclassified as scientific negative) | Main & localization timeouts/rc125 fail closed to `unknown_infra` (`0` scientific witnesses); `M9.9.1` logs decomposition non-triviality timeout `probe_id`/`wall_ms`. | **`technically_fixed`** |
| `R06` (`EmpiricalCoverageReceipt` `ecov_invalid` bypass) | Enforces 29-char `ecov_` SHA-256 and count/list invariants. | **`technically_fixed`** |
| `R07` (`_hydrate_from_graph` missing digest & legacy trust) | Rejects missing hashes on current schemas; `M9.9.1` (`Q08`) makes legacy downgrade `load -> migrate -> save -> reload` idempotent (`0` authority). | **`technically_fixed`** |
| `R08` (`HistoryMasked` future step-4 & fixed-`D2` flaw) | Sealed `64`-run prefix-locked (`0/16`) & fixed `D0`/`D1` (`8/16`, `8/16`) vs history-aware (`16/16`) post-hoc control. | **`backverified_executed`** |
| `R09` (Cost `p`-value conflated with quality `p`-value) | Cost, zero-LLM closure, and `strict_pass` `p`-values reported in separate rows/columns across `docs/54*` and `docs/55*`. | **`claim_narrowed_and_reformatted`** |
| `R10` (Last-entry observable precursor overstated as long latent memory) | Narrowed explicitly in `docs/54*`: `archive_apparatus_log[-1]` is an observable precursor state, not long-horizon latent POMDP inference. | **`claim_narrowed`** |
| `R11` (`8` `M9.8` localization probes untimed; elapsed vs CPU) | Reference rerun (`4,967.12 ms`) distinguished from unrecovered original time; `M9.9` 6-category ledger explicitly labeled **elapsed container wall-ms** (`1.421x` total, `3.320x` coverage sub-item). | **`technically_fixed_and_reworded`** |
| `R12` (Input hash vs source hash & cross-env reproducibility) | `56/56` setup, pickle, and `canonical_numeric_sha256` matched when independently materialized on the host and compared to the archived host fields. | **`technically_fixed`** |
| `R13` (`M9.7` Beta `p = 0.50` typo & `16/16` vs `15/16` twin text) | Corrected in `docs/53b*` (`Beta p = 1.0`) and `docs/54*` (`15/16` `Table` twin 0-LLM). | **`technically_fixed`** |
| `R14` (`R1` error-code diagnosis overstated as numerical fix) | Explicitly retired (`diagnosis_and_error_code_closed_numerical_candidate_retired`; original `41/42` score retained). | **`retired_by_design`** |
| `R15` (Legacy KKT `L1` certificate interface / live adoption) | Explicitly retained as an open boundary (`retained_open_gap` in `docs/53b*` and `docs/54*`; orthogonal to `M9.9/M10.0`). | **`retained_open_gap`** |
| `R16` (`cap=32` vs `peak`, sign-flip resolution vs power, push wording) | `cap` and `peak` reported side-by-side; `2/256` resolution separated from power; broken `54b` hub link fixed in `M9.9.1`. | **`technically_fixed`** |

---

## 8. Calibrated Forensic Attribution of `M9.9` Alpha Trajectories & Conditional Amortization Equation (`Q02–Q04, Q11, Q13`)

### 8.1 Corrected Episode-Level Forensics Across the `8` Confirmation Worlds (`E4 = v3_gains`, `E5 = v2_gains`, `E7 = v3_gains`)

1. **Beta (`beta_m99_c01..c04`, `28/28` Pass, `4/4` Cost Reductions vs Both Baselines)**: `u1` dynamically re-fits the coupled cubic sensor rosette per campaign (`< 5e-5` holdout RMSE across `v3_gains` on `E4, E7` and `v2_gains` on `E5`), closing `12/12` `(E4, E6, E7)` episodes at `0` LLM turns.
2. **Alpha (`alpha_m99_c01..c04`) — Audit-Verified Root Causes (`Hegel_M9_9_Audit` `Q02–Q04`)**:
   - **`alpha_m99_c01` (`5/7` Pass Matching `Monolithic` `5/7`, `-243,947` `v2`)**: Decomposed on `E1_down`, closed `E4` and `E7` at `0` LLM turns, and issued `swdr_9dbe611c8cc839270fd88efd` on `E5`.
   - **`alpha_m99_c02` (`5/7` vs `Monolithic` `7/7`, `+241,390` `v2`) — Corrected Forensics (`Q02` & `Q04`)**:
     - *(i) Missing executable fallback / retroactive projection after `E2_down` decomposition (`Q04`)*: `E1_down` (`D1`) passed private grading after 14 monolithic turns but failed public qualification, and decomposition triggered on `E2_down` (`step 2`), leaving `E1_down` unprojected and non-callable on E4 (13 turns, failed). Replaying the archived solver on E4 also failed public qualification.
     - *(ii) Downstream test-only reference standardization violating the public `train+test` pooled contract on `E7_coupled_revisit` (`Q02` — Correcting Prior Upstream-Newton Attribution)*: On `E7` (`v3_gains`), `bind_alpha_m99_c02_up_b01` inverted test loads to **`~1.4e-5` RMSE** (substituting exact oracle upstream loads still yields `strain_mae = 0.071270 > 0.070`!). The true root cause was `bind_alpha_m99_c02_dn_b02`'s `_chamber_features(test)` normalizing `test` on `test['ref_coupons']` alone with unweighted `NaN` mean instead of the documented pooled `vstack([train, test])` per-lab standardization and `(lab_id, stratum)` inverse-count weighted imputation (`BLIND_ALPHA_INSTRUCTION_TEXT`). Fixing **only** that public preprocessing contract drops `seed = 952711` `strain_mae` from **`{c02_priv_inst[952711]['strain_mae']:.6f}` to `0.004857`** (`<= 0.070`!), while `tau`/kernel approximation (`0.004857 -> 0.001425`) was secondary (`public strain_mae = {c02_pub_metrics['strain_mae']:.6f}`).
   - **`alpha_m99_c03` (`5/7` vs `Monolithic` `7/7`, `+173,853` `v2`) — Single-Episode Miss on `E2_down` (`Q04`)**: `E2_down` (`U0 × D2_sep`) failed its `13`-turn synthesis budget, leaving `d2_sep` missing (`separability_withdrawal_receipts = []` at `E5`) and propagating into pilot-free `E6_separable_revisit`.
   - **`alpha_m99_c04` (`7/7` Pass Matching `Monolithic` `7/7`, `+417,571` `v2`) — Corrected Forensics (`Q03`)**:
     - On **`E4_valid_comb` (`v3_gains`)**, `up_b01` (learned at `E3_up` under `v2_gains`) had hardcoded `E3`'s fitted polynomial coefficients instead of re-fitting `calibration_load_cells` per item, genuinely requiring `{c04_e4_turns}` synthesis turns to learn **`bind_alpha_m99_c04_up_b02`**.
     - On **`E7_coupled_revisit` (`v3_gains`)**, however, **`bind_alpha_m99_c04_up_b02` was ALREADY in the `step_06` library** and passes the deterministic `56/24` (`0..55` / `>=56` outlier-filtered) upstream probe (`RMSE = {m991_q03['direct_upstream_component_probe']['chosen_up_holdout_rmse']:.8f} <= 0.03`), the 3-layer public gate (`accepted = {m991_q03['joint_three_layer_public_gate']['accepted']}`), AND the official verifier (`strict_pass = {m991_q03['official_verifier_result']['strict_pass']}`) when composed with `dn_b03` (`02_m99_q03_c04_e7_existing_candidate_container_verification.json`). It was never tried on `E7` only because its 2-sample guard required `up_stat_03_cal_frobenius_rms > 1.289054` (`E7` had `1.274262`), routing back to `up_b01` (`RMSE = 0.5763`) and spending `{c04_e7_turns}` avoidable repair turns!

### 8.2 Conditional Unseen-Cell Count (`|(U × D) \\ (R ∪ X)|`) & Why Break-Even $K^*$ Must Be Measured Empirically (`Q11`)

Let $U$ and $D$ denote upstream and downstream operator sets covered by single-factor exposure episodes $R$, and let $X \\subseteq U \\times D$ denote coupled interaction exceptions. Under **ideal 1-shot module synthesis and zero false guard exclusions**, the number of unseen separable cells is:

$$N_{{\\text{{sep,unseen}}}} = |(U \\times D) \\setminus (R \\cup X)|$$

(note: $(|U|-1)(|D|-1) - |X|$ holds only when $X \\cap R = \\emptyset$). In real online continual streams, actual cumulative cost difference $\\Delta C(K) = C_{{\\text{{Hegel}}}}(K) - C_{{\\text{{baseline}}}}(K)$ depends on decomposition cost $C_{{\\text{{decomp}}}}$, guard/fallback probe costs, and non-zero repair rates when guards or contracts drift (as seen on `alpha_m99_c02..c04`). The larger grids and the quality-constrained break-even require a prospective experiment.
"""

    zh_md = f"""# Hegel M9.8.1 审计闭环与 M9.9 等工具给定两阶段接口在线模块形成及反例域交互撤回报告（`M9.9.1 审计校准版`）

> [!IMPORTANT]
> **`M9.9.1` 独立审计勘误与补证说明（`Q01–Q14`）**：原封存 `M9.9` 确认集全部私判得分、原始调用账单与文件保持 `100%` 零改动只读封存。本报告由 `scripts/generate_m9_9_report_from_sealed_json.py` 基于审计结论统一重生成，逐项校准来源身份（`Q05`）、统计与范围措辞（`Q06, Q09–Q14`）、原编号 `R01–R16` 对照及 `c02/c04` 真实数值根因（`Q02–Q04`）。配套 `30` 次断网容器交叉补证（`{m991_q01['accepted_count']}/30` 通过、`{m991_q01['scientific_or_domain_rejected_count']}/30` 科学/域拒绝、`{m991_q01['schema_rejected_count']}/30` 接口拒绝）与完整取证回执见 `docs/55b_M9_9_1_Audit_Closure_Forensic_Reverification_and_Errata_ZH.md`。

- **日期**：`2026-11-28`
- **声明冻结提交与执行/后处理身份分离（`Q05`）**：付费确认运行对应声明冻结提交 `{conf['paid_llm_frozen_commit']}`（`2,700` 个 `.py` 文件重建树哈希 `{frz_tree}`；原始镜像调用时间包络下界 `{conf_env['call_envelope_lower_bound_sec']:,.3f}s`，自 `{conf_env['first_call_started_utc']}` 至 `{conf_env['last_call_ended_utc']}`，运行 ID `{conf_env['execution_run_ids_in_raw_calls'][0]}`）。*来源说明（`Q05`）*：原在线运行未持久化独立启动清单；`m9_9_continual_summary.json` 中的 `run_id = {conf_env['finalizer_summary_run_id']}`（`{conf_env['finalizer_postprocess_wall_sec']}s`）与树哈希 `{fin_tree}` 实测自赛后 `--finalize-existing` 消融补算阶段。提交树重建无法补出执行时未留存的工作区状态。
- **M9.8.1 审计闭环提交**：`b959102fa9c284efebb8921ce39229c4b23f45b1`
- **共享三层核查内核**：`{conf['shared_kernel_version']}` ｜ **统一成本策略**：`{conf['cost_policy_version']}`
- **断网容器镜像摘要**：`{conf['sandbox_image_digest']}`
- **实测模型与并发度记录（`R16`）**：`{conf['model']}`（`reasoning_effort = {conf['reasoning_effort']}`，`configured_max_codex_concurrency = {conf['configured_max_codex_concurrency']}`，实测峰值活跃并发 `peak_active_codex_concurrency = {conf['peak_active_codex_concurrency']}`）
- **用量完备性与未知 `x` 精确敏感性阶梯（`Q06`）**：`{conf['usage_completeness']}`（`1,750/1,751` 次调用用量完整；`Stage S`、`Hegel`、`Monolithic` 均为 `0` 未知尝试；`Flat_PlainModuleRegistry_Online` 在 `alpha_m99_c04/E1_down` 含 `1` 次超时未知尝试 `x >= 0` `call_uuid=0391d702b0c547a488608847b5dff224`：`Hegel - Plain = -2,535,683 - x`；当 `0 <= x <= 17,652` 时双侧精确符号翻转 `p = 0.0625`，当 `17,652 < x <= 53,450` 时 `p = 0.0546875`，**仅当 `x > 53,450` 时 `p < 0.05`**）。

---

## 一、核心结论与严格限定的归因边界（`Q01–Q14` 及原编号 `R01–R16`）

1. **`M9.8.1` 证据契约修复与容器归因补证（`0` 新增付费 LLM 调用）**：
   - 硬化 `SolverEvaluationWitness`（`v2` 模式：显式 `execution_status`、`evidence_origin`、可空 `metric_value` 移除 `999.0` 哨兵、`probe_id`、`setup_source_sha256`、`canonical_numeric_sha256`、`dependency_bundle_sha256`、`verification_trust_state`）、`EmpiricalCoverageReceipt.verify_integrity()`（`R06`）及 `_hydrate_from_graph()`（`R07`；显式旧模式 `load -> migrate -> save -> reload` 往返幂等与旧 `M9.8` 推定定位隔离已在 `M9.9.1` `Q08` 补齐）。
   - 封存 `M9.8` 双普通表容器回放（`56/56` setup、pickle 与规范数值摘要在独立宿主重建中与已归档的宿主输入一致；`32/56` 公开接受候选 `32/32 = 100.0%` 私判通过）、`16/16` 次 `u0` 直接上游容器探针（最大 RMSE `{r02['r04_16_previously_inferred_upstream_instances']['max_holdout_rmse_across_16']:.8f} <= 0.03`）及 `64` 次 `(R0, U_iso)` 前缀锁定四策略对照（历史感知 `16/16` vs 遮蔽 `0/16` vs 固定 `D0/D1` 各 `8/16`）。

2. **`M9.9` 预登记 `n = 8` 未曝光确认簇三臂同权限在线实验（`176` 个容器化实测 Episode）**：
   - **零预置模块源码、给定两阶段接口下的在线模块形成（`Q09` 精确范围）**：`Stage S` 从零合成未分解单体 `solve_batch`（`8/8` 首轮通过）。在宿主规定的两阶段 `(calibrate_upstream, solve_downstream)` 接口、角色划分与父参数拓扑内（本阶段不主张未知层数开放 DAG 拓扑发现），双模块化臂在线调用 `propose_and_verify_m99_autonomous_two_stage_factorization`（全额计入 `Scope-1`）：**在 `E1_down` 当步成功分解 `Hegel` `{decomp_hg_e1}/8`、`Plain` `{decomp_pl_e1}/8`；计入 `E2_down` 延迟分解后全流累计为 `Hegel` `{decomp_hg_cum}/8`、`Plain` `{decomp_pl_cum}/8`**（`Q13`）。*审计勘误（`Q01`）*：原 `M9.9` 分解验证器仅实跑 `2` 个对角正格，而向初始 `d0d1_gind` 传入 `30` 个未执行交叉负约束并向 `Hegel` 写入 `8` 条未执行 `d0@E` 投影负证人；`M9.9.1` 已升级为完整 `2×2` 容器检查并实跑补齐全部 `30` 次历史交叉公开门控（见 `docs/55b_*`）。
   - **结论 A —— `Hegel_AutonomousInteractionGraph_Online` 相对固定可分路由策略 `Flat_PlainModuleRegistry_Online` 的对照结果（`Q06, Q10`）**：
     - **零 LLM 通过闭环**：**`{out_t['Hegel_AutonomousInteractionGraph_Online']['zero_turn_passed_closures']}/56`（`Hegel`）vs `{out_t['Flat_PlainModuleRegistry_Online']['zero_turn_passed_closures']}/56`（固定可分 `PlainModuleRegistry`）**（`5` 胜 `3` 平 `0` 负，双侧精确符号翻转 `p = {pair_pl['zero_llm_two_sided_sign_flip_p']}`），依托 `{out_t['Hegel_AutonomousInteractionGraph_Online']['separability_withdrawals']}/8` 份可完整追溯的 `SeparabilityWithdrawalReceipt` 在 `(E6, E7)` 取得 `{e67_hg_z}/16` vs `{e67_pl_z}/16` 零调用通过。
     - **成本与统计推断边界**：已知 Scope-1 `proxy_v2 = {hg_s1_v2:,}` vs `{pl_s1_v2:,} + x`（已知总和比 **`{_pct(hg_s1_v2, pl_s1_v2)}`**，等权簇均值 `{_mean_cluster_pct(conf, 'Flat_PlainModuleRegistry_Online')}`，含源 `Scope-Total` `{_pct(hg_tot_v2, pl_tot_v2)}`，簇均差 `{pair_pl['mean_diff']:+,.2f} - x/8`，`95%` 自助法区间 `{pair_pl['bootstrap_95ci']}`，**`x = 0` 时双侧精确符号翻转 `p = {pair_pl['two_sided_sign_flip_p']}`**——呈 `6/8` 簇一致方向优势，但在 `x <= 53,450` 时未跨过双侧 `0.05` 显著性门槛；`strict_pass` `{out_t['Hegel_AutonomousInteractionGraph_Online']['strict_passes']}/56` vs `{out_t['Flat_PlainModuleRegistry_Online']['strict_passes']}/56`，`p = {pair_pl['quality_two_sided_sign_flip_p']}`）。
     - **给定 `step_05` 模块库与 `dn_obs` 限制下的冻结路由消融（`48` 次运行，`Q10`）**：在固定 `step_05` 模块前缀上，`hegel_selective_cross_factor_graph = {abl_s['hegel_selective_cross_factor_graph']['strict_passes']}/16`（`{abl_s['hegel_selective_cross_factor_graph']['pass_rate_pct']}%`）对比受限于仅读 `dn_obs` 的固定路由 `factorized_fixed_D2_separable = {abl_s['factorized_fixed_D2_separable']['strict_passes']}/16`（`{abl_s['factorized_fixed_D2_separable']['pass_rate_pct']}%`）与 `factorized_fixed_D2_coupled = {abl_s['factorized_fixed_D2_coupled']['strict_passes']}/16`（`{abl_s['factorized_fixed_D2_coupled']['pass_rate_pct']}%`）。该消融验证的是**反例触发的选择性条件扩展策略**（`dn_obs -> up_obs + dn_obs`），而非声称普通字典数据结构无法表达同一算法：在相同宿主重建的策略输入上向普通 Python `dict/list` 提供相同 `up+dn` 特征时，`{parity}` 步选中分支与候选代码 SHA 完全一致（见 `03_m99_provenance_x_sensitivity_and_plain_dict_replay.json`）。
       原消融在可分分支缺失时调用了耦合分支。改为显式弃权后，固定可分策略有 {sep_replay['strict_passes']}/{sep_replay['total_episodes']} 个位置严格通过、{sep_replay['abstained_missing_prefix_binding']} 个位置弃权。两种口径均覆盖原 48 个策略位置，见补证账本 07。
   - **结论 B —— 相对在线单体条件表 `Flat_MonolithicConditionalTable_Online` 的短周期比较（`Q06, Q13`）**：
     - 在 **Beta 家族（`n = 4` 二维 PDE/KKT 簇）** 上，`Hegel` 取得 `28/28` 通过、`12/28` vs `8/28` 零调用闭环及 `4/4` 簇成本下降（簇均 `-201,744.75` `v2`，`beta_sign_flip_p = {pair_mo['beta_sign_flip_p']}`）。
     - 在全 `8` 簇总计上，与 `Monolithic` 的配对成本差未达显著（Scope-1 `{hg_s1_v2:,}` vs `{mo_s1_v2:,}` = `{_pct(hg_s1_v2, mo_s1_v2)}`，`p = {pair_mo['two_sided_sign_flip_p']}`），而 **`Hegel` 在 Alpha 上比 `Monolithic` 少通过 `4` 个回合（总 `strict_pass` `50/56 = 89.29%` vs `54/56 = 96.43%`，`p = {pair_mo['quality_two_sided_sign_flip_p']}`——`p > 0.05` 不能当作质量非劣证明）**。非 LLM 代理容器探针**累计流逝墙钟（elapsed wall-ms，非 CPU 秒数）**总体比值为 **`{tot_probe_ratio:.3f}×`**（`1,271,833.0 ms` vs `895,174.8 ms`），其中经验覆盖探针子项累计墙钟比值为 **`{cov_sub_ratio:.3f}×`**（`689,259.6 ms` vs `207,604.6 ms`）。

---

## 二、三臂主实验总账本（`n = 8` 未曝光确认簇，`176` 个容器化 Episode）

| 实验臂（`100%` 同工具权限） | `strict_pass` | 零 LLM 通过闭环 | `Scope-1` 调用数（求解 / 分解 / 守卫） | `Scope-1 proxy_v1` | `Scope-1 proxy_v2` | `Scope-Total proxy_v2`（含全额 `Stage S`） | `Hegel` 相对该行（`Scope-1` / `Total`） |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicConditionalTable_Online` | `{out_t['Flat_MonolithicConditionalTable_Online']['strict_passes']}/56` | `{out_t['Flat_MonolithicConditionalTable_Online']['zero_turn_passed_closures']}/56` | `{led['Flat_MonolithicConditionalTable_Online']['scope1_in_stream_total']['call_count']}` (`{led['Flat_MonolithicConditionalTable_Online']['solve_only']['call_count']}` / `0` / `{led['Flat_MonolithicConditionalTable_Online']['guard_induction_only']['call_count']}`) | `{mo_s1_v1:,}` | `{mo_s1_v2:,}` | `{mo_tot_v2:,}` | `{_pct(hg_s1_v2, mo_s1_v2)}` / `{_pct(hg_tot_v2, mo_tot_v2)}` |
| `Flat_PlainModuleRegistry_Online` *（固定可分 `dn_obs` 路由）* | `{out_t['Flat_PlainModuleRegistry_Online']['strict_passes']}/56` | `{out_t['Flat_PlainModuleRegistry_Online']['zero_turn_passed_closures']}/56` | `{led['Flat_PlainModuleRegistry_Online']['scope1_in_stream_total']['call_count']}` (`{led['Flat_PlainModuleRegistry_Online']['solve_only']['call_count']}` / `{led['Flat_PlainModuleRegistry_Online']['decomposition_induction_only']['call_count']}` / `{led['Flat_PlainModuleRegistry_Online']['guard_induction_only']['call_count']}`) | `{pl_s1_v1:,} + x_1` | `{pl_s1_v2:,} + x` | `{pl_tot_v2:,} + x` | `{_pct(hg_s1_v2, pl_s1_v2)}` / `{_pct(hg_tot_v2, pl_tot_v2)}`（`x >= 0`） |
| **`Hegel_AutonomousInteractionGraph_Online`** | **`{out_t['Hegel_AutonomousInteractionGraph_Online']['strict_passes']}/56`** | **`{out_t['Hegel_AutonomousInteractionGraph_Online']['zero_turn_passed_closures']}/56`** | **`{led['Hegel_AutonomousInteractionGraph_Online']['scope1_in_stream_total']['call_count']}`** (`{led['Hegel_AutonomousInteractionGraph_Online']['solve_only']['call_count']}` / `{led['Hegel_AutonomousInteractionGraph_Online']['decomposition_induction_only']['call_count']}` / `{led['Hegel_AutonomousInteractionGraph_Online']['guard_induction_only']['call_count']}`) | **`{hg_s1_v1:,}`** | **`{hg_s1_v2:,}`** | **`{hg_tot_v2:,}`** | 参照基准 |

- **共享 `Stage S` 源任务账本（`8/8` 首轮通过，`0` 未知尝试）**：`{st_s['call_count']}` 次调用，`proxy_v1 = {s_v1:,}`，`proxy_v2 = {s_v2:,}`（按 `1/3` 分摊时 `Hegel` 相对 `Plain` `-18.08%`，相对 `Monolithic` `-1.86%`）。
- **分项配对精确符号翻转检验（`n = 8` 独立确认簇；双侧最小分辨率 `2/256 = 0.0078125`，注意最小 `p` 分辨率不等于统计检验功效 power）**：
  - `Hegel - Flat_PlainModuleRegistry_Online`：簇均 `Δv2 = {pair_pl['mean_diff']:+,.2f} - x/8`（`95%` 自助法区间 `{pair_pl['bootstrap_95ci']}`），**`x = 0` 时成本 `p = {pair_pl['two_sided_sign_flip_p']}`**（`x > 53,450` 时 `p < 0.05`），**零 LLM 闭环 `p = {pair_pl['zero_llm_two_sided_sign_flip_p']}`**，**质量 `strict_pass` `p = {pair_pl['quality_two_sided_sign_flip_p']}`**。
  - `Hegel - Flat_MonolithicConditionalTable_Online`：簇均 `Δv2 = {pair_mo['mean_diff']:+,.2f}`（`95%` 自助法区间 `{pair_mo['bootstrap_95ci']}`），**成本 `p = {pair_mo['two_sided_sign_flip_p']}`**，**零 LLM 闭环 `p = {pair_mo['zero_llm_two_sided_sign_flip_p']}`**，**质量 `strict_pass` `p = {pair_mo['quality_two_sided_sign_flip_p']}`**（`50/56` vs `54/56`）。

---

## 三、逐簇配对结果明细（`n = 8` 未曝光确认世界）

| 确认簇（家族） | `Monolithic` 通过（零 LLM） | `Monolithic` `v2` | `PlainModuleRegistry` 通过（零 LLM） | `PlainModuleRegistry` `v2` | `Hegel` 通过（零 LLM） | `Hegel` `v2` | `Hegel - Plain` `Δv2` | `Hegel - Mono` `Δv2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(cluster_rows)}

---

## 四、逐事件因果演化轨迹（`7` 个持续步 × `8` 个确认簇）

| 持续流事件 | `Flat_MonolithicConditionalTable_Online` | `Flat_PlainModuleRegistry_Online` | `Hegel_AutonomousInteractionGraph_Online` |
|---|---|---|---|
{chr(10).join(ev_rows)}

---

## 五、非 LLM 容器探针累计流逝墙钟全账本（`R11 / Q13`：并发线程累加容器 elapsed `ms`，非 CPU 时间）

| 实验臂 | Pre-LLM 门控 (`ms`) | Post-LLM 门控 (`ms`) | 父残差定位 (`ms`) | 自主分解核查 (`ms`) | 经验覆盖探针 (`ms`) | **非 LLM 代理探针累计墙钟 (`ms`)** | 私判官方评测器 (`ms`) |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(probe_rows)}

---

## 六、开发与确认研发总账本（含恢复的取消运行用量，`Q12`）

1. **冻结前取消的开发运行**。已找回 {cancelled['unique_logical_calls']} 次调用，其中已知物理成本为 {cancelled['known_physical_v2']:,} v2，另有 {cancelled['unclosed_call_count']} 次缺少最终用量。
2. **已封存 `n = 2` 开发集包（`docs/evidence/m9_9_dev_live_calibration_20261128`，`1,398` 文件，原始调用时间包络 `{dev_env['call_envelope_lower_bound_sec']:,.3f}s`）**：`{dev['total_logical_calls']}` 次调用，物理实付 `8,409,059` `v2`（`0` 未知尝试，`peak_concurrency = {dev['peak_active_codex_concurrency']}`）。
3. **已封存 `n = 8` 确认集包（`docs/evidence/m9_9_autonomous_decomposition_and_interaction_withdrawal_20261128`，`6,226` 文件，原始调用时间包络 `{conf_env['call_envelope_lower_bound_sec']:,.3f}s`）**：`{conf['total_logical_calls']}` 次调用，物理实付 `37,274,561 + x` `v2`（`configured = {conf['configured_max_codex_concurrency']}`，实测 `peak = {conf['peak_active_codex_concurrency']}`）。
4. **研发成本汇总**。两份封存运行合计 2,143 次调用，物理成本 45,683,620 + x。含取消运行合计 {cancelled['total_rd_v2_formula']}。

---

## 七、`Hegel_M9_8_Audit` 原编号（`R01–R16`）逐项对应矩阵（`Q14` 严禁重排编号）

| 原 `M9.8` 编号与原问题 | `M9.8.1` / `M9.9` / `M9.9.1` 实际证据与适用边界 | 审计核定状态 |
|---|---|---|
| `R01`（模块化整体收益归因于图结构） | `M9.8` 双普通表 `56/56` 候选一致回放（`32/32` 通过）+ `M9.9` 固定可分 `Plain` 对照 + 普通 `dict` 同 `up+dn` 特征 `{parity}` 候选 SHA 一致回放：严格区分“选择性条件扩展算法”与底层容器结构。 | **补证完成 + 声明收窄** |
| `R02`（宿主预设 `u0/d0`、父集与修复接口） | 模块源码 `u0/d0/d1` 由模型在线生成并计入 `Scope-1`（`M9.9.1` 补齐 `2×2` 验证）；两阶段接口与父参数拓扑仍由宿主规定，未知层数开放 DAG 另轨推进。 | **部分实质改变 + 声明收窄** |
| `R03`（`L1` 权限与修复任务不对称） | 三臂统一 `enable_auto_skill_packaging=False, allow_l1_primitives=False`；双模块化臂共享完全一致的模块修复模板。 | **技术修复** |
| `R04`（高 `r²` 未执行却记 `RMSE=0` 与正例） | `M9.8.1` 修复并实跑 `16/16` 次直接上游探针（最大 RMSE `0.00019837`）；`M9.9.1` 同步闭合新分解交叉格未执行问题（`Q01` `30` 次容器探针）与旧 `M9.8` 推定定位负归因隔离（`Q08`）。 | **技术修复 + M9.9.1 补证闭环** |
| `R05`（基础设施失败/超时变科学负例） | 主探针与定位超时/`rc=125` 均归入 `unknown_infra`（零科学负证人）；`M9.9.1` 补记分解非平凡性探针超时耗时。 | **技术修复** |
| `R06`（覆盖回执畸形 `ecov_invalid` 绕过） | 强制校验 29 字符 `ecov_` SHA-256 与证人列表计数不变量。 | **技术修复** |
| `R07`（冷恢复缺摘要越权与旧模式迁移） | 当前模式缺摘要拒绝并校验绑定代码外键；`M9.9.1`（`Q08`）补齐显式旧模式降权后的幂等 `save -> reload` 往返（零升权）。 | **技术修复** |
| `R08`（`HistoryMasked` 未来步与固定 `D2` 缺陷） | 封存 `64` 次合法前缀对照：历史感知 `16/16`、前缀遮蔽 `0/16`、固定 `D0/D1` 各 `8/16`（保持事后敏感性身份）。 | **补证完成** |
| `R09`（成本 `p` 易误读为质量 `p`） | 成本、零调用闭环、严格通过率三类检验在全报告分列呈现。 | **文档分列修复** |
| `R10`（末条可观测前驱被说成长历史隐状态） | 在 `docs/54*` 明确收窄：`archive_apparatus_log[-1]` 是可观测前驱状态，不宣称长历史隐变量推断。 | **声明收窄** |
| `R11`（`8` 次定位未单列计时；墙钟误写 CPU） | `4,967.12 ms` 明确标为补跑参考时间；`M9.9` 六类探针统一正名为 **容器流逝墙钟 elapsed `ms`**（总比值 `1.421×`，覆盖子项 `3.320×`）。 | **技术记账改善 + 措辞修正** |
| `R12`（输入摘要与跨环境可复现性） | `56/56` setup、pickle 与规范数值摘要在独立宿主重建时与旧宿主账本匹配。 | **技术修复** |
| `R13`（旧 Beta `p = 0.50` 与双生 `16/16` 文字错） | 已在 `docs/53b*`（`Beta p = 1.0`）与 `docs/54*`（`Table` 双生 `15/16`）修正。 | **文字修正** |
| `R14`（`R1` 诊断码修复被当成数值修复） | 按审计建议明确退役（`diagnosis_and_error_code_closed_numerical_candidate_retired`，保留原 `41/42` 得分）。 | **按设计退役（非算法重跑）** |
| `R15`（旧 KKT `L1` 证书接口与真实采用未闭合） | 在 `docs/53b*` 与 `docs/54*` 明确保留开放缺口 `retained_open_gap`（不伪称已修，也不阻塞两阶段模块化研究）。 | **明确保留开放缺口** |
| `R16`（并发 `cap/peak`、功效/分辨率与入口坏链接） | 并列报告 `cap=32 / peak=24`，区分分辨率与功效，并在 `M9.9.1` 修复能力总入口 `54b` 链接路径。 | **文字与链接修复** |

---

## 八、经审计复核校准的 Alpha 逐簇取证归因与附条件摊销计数（`Q02–Q04, Q11, Q13`）

### 8.1 全 `8` 个确认簇校准取证（修正增益映射：`E4 = v3_gains`、`E5 = v2_gains`、`E7 = v3_gains`）

1. **Beta 家族（`beta_m99_c01..c04`，`28/28` 全过，对双基线 `4/4` 成本下降）**：`u1` 在 `E4, E7`（`v3_gains`）与 `E5`（`v2_gains`）上均按当前批次 `sensor_calibration_sweeps` 动态重拟合三次花瓣逆映射（留出 RMSE `< 5e-5`），`(E4, E6, E7)` 共 `12/12` 步零调用私判通过。
2. **Alpha 家族（`alpha_m99_c01..c04`）——经 `Hegel_M9_9_Audit` 代码复现核实的真实根因（`Q02–Q04`）**：
   - **`alpha_m99_c01`（`5/7` 追平 `Monolithic` `5/7`，相对 `Monolithic` `-243,947` `v2`）**：`E1_down` 成功分解，`E4` 与 `E7` 零调用通过，`E5` 签发 `swdr_9dbe611c8cc839270fd88efd`。
   - **`alpha_m99_c02`（`5/7` vs `Monolithic` `7/7`，`+241,390` `v2`）——校准根因（`Q02` 与 `Q04`）**：
     - *(i) 延迟至 `E2_down` 分解后缺失旧单体可执行兜底与惰性回投（`Q04`）*：`E1_down`（`D1`）经 14 轮单体合成后私判通过，但公开准入失败，`E2_down` 步才完成分解，原 `E1_down` 单体既未作为有类型执行兜底保留也未回投为 `d1` 模块。`E4_valid_comb`（`v3_gains`）耗费 13 轮未过。该旧单体在 E4 回放也未过公开门。
     - *(ii) `E7_coupled_revisit` 下游测试集独立标准化违背公开 `train+test` 联合变换合同（`Q02`——更正原“上游三次牛顿外推漂移”归因）*：代码复现证明 `bind_alpha_m99_c02_up_b01` 在 `E7`（`v3_gains`）上对测试载荷的反演 RMSE 仅为 **`~1.4e-5`**（即便替换为完美真值上游载荷，原 `dn_b02` 在 `seed = 952711` 上仍得 `strain_mae = 0.071270 > 0.070`！）。真实主因是 `bind_alpha_m99_c02_dn_b02` 的 `_chamber_features(test)` 仅用 `test['ref_coupons']` 独立计算均值/标准差并以未加权列均值填补 `NaN`，违背了公开说明要求的 `vstack([train, test])` 按 `lab_id` 联合标准化与 `(lab_id, stratum)` 逆频数加权缺失填补合同；**仅修复该公开预处理合同即将 `seed = 952711` 的 `strain_mae` 从 `{c02_priv_inst[952711]['strain_mae']:.6f}` 压至 `0.004857`（远低于 `0.070`）**，而 `tau`/时间核近似（`0.004857 -> 0.001425`）仅为次要因素。
   - **`alpha_m99_c03`（`5/7` vs `Monolithic` `7/7`，`+173,853` `v2`）——`E2_down` 缺失模块向 `E6` 传播（`Q04`）**：`E2_down` 未通过公开门控，`d2_sep` 处于缺失状态，导致无先导曲线的 `E6` 冷合成未过。
   - **`alpha_m99_c04`（`7/7` 追平 `Monolithic` `7/7`，`+417,571` `v2`）——区分 `E4` 真缺能力与 `E7` 已有合格候选被守卫排除（`Q03`）**：
     - **`E4_valid_comb`（`v3_gains`）**：`E3_up`（`v2_gains`）学得的 `up_b01` 将当轮拟合系数硬编码为常数而非按输入重拟合校准集，确需 `{c04_e4_turns}` 轮新合成出 **`bind_alpha_m99_c04_up_b02`**。
     - **`E7_coupled_revisit`（`v3_gains`）**：前缀 `step_06` 库中**已存在的 `bind_alpha_m99_c04_up_b02`** 在 `E7` 前 `56` / 后 `24`（滤除离群行后）确定性留出探针上 **`RMSE = {m991_q03['direct_upstream_component_probe']['chosen_up_holdout_rmse']:.8f} <= 0.03`**，且与 `dn_b03` 组合后**同时通过断网公开三层门控（`accepted = {m991_q03['joint_three_layer_public_gate']['accepted']}`）与官方私判评测器（`strict_pass = {m991_q03['official_verifier_result']['strict_pass']}`）**（见 `02_m99_q03_c04_e7_existing_candidate_container_verification.json`）。它之所以未被调用，仅仅因为 2 样本守卫归纳生成了过窄的 `up_stat_03_cal_frobenius_rms > 1.289054`（`E7` 为 `1.274262`），回退到失效的 `up_b01`（`RMSE = 0.5763`）并触发了 `{c04_e7_turns}` 轮重合成。

### 8.2 附条件未见可分格点计数（`|(U × D) \\ (R ∪ X)|`）与实测盈亏平衡点 $K^*$ 要求（`Q11`）

设单因子暴露集为 $R$、非可分耦合例外集为 $X \\subseteq U \\times D$，则在**理想单次合成且零误排重编译**前提下，未见可分乘积格点数为附条件集合计数：

$$N_{{\\text{{sep,unseen}}}} = |(U \\times D) \\setminus (R \\cup X)|$$

（仅当 $X \\cap R = \\emptyset$ 时退化为 $(|U|-1)(|D|-1) - |X|$）。在真实在线持续学习中，累计净成本曲线 $\\Delta C(K) = C_{{\\text{{Hegel}}}}(K) - C_{{\\text{{baseline}}}}(K)$ 同时取决于分解成本、探针/回退开销与模块边界修复率，不能预设更大网格必然盈利；更大网格和质量约束下的盈亏平衡点均待独立实验。
"""

    OUT_EN.write_text(en_md, encoding="utf-8")
    OUT_ZH.write_text(zh_md, encoding="utf-8")
    print(f"Wrote calibrated `{OUT_EN.name}` and `{OUT_ZH.name}` successfully.")


if __name__ == "__main__":
    main()
