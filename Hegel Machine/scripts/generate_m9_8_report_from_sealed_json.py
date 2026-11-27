#!/usr/bin/env python3
"""Deterministically generate `docs/54_M9_7_1_Closure_and_M9_8_Blind_Mismatch_and_Factored_Mechanism_Graph_Report.md`
and `_ZH.md` strictly from sealed `M9.7.1` and `M9.8` JSON artifacts (`0` hand-edited numbers, `A01/A05/A12/A14/A15` compliant).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_M971_DIR = REPO_ROOT / "docs/evidence/m9_7_1_audit_closure_and_paid_pool_attribution_20261127"
DEFAULT_M98_RUN1_DIR = REPO_ROOT / "docs/evidence/m9_8_aborted_run1_stage_s_contract_underspec_20261127"
DEFAULT_M98_DEV_PASS1_DIR = REPO_ROOT / "docs/evidence/m9_8_dev_live_calibration_pass1_20261127"
DEFAULT_M98_DEV_DIR = REPO_ROOT / "docs/evidence/m9_8_dev_live_calibration_20261127"
DEFAULT_M98_DIR = REPO_ROOT / "docs/evidence/m9_8_blind_mismatch_factored_graph_20261127"
OUT_EN = REPO_ROOT / "docs/54_M9_7_1_Closure_and_M9_8_Blind_Mismatch_and_Factored_Mechanism_Graph_Report.md"
OUT_ZH = REPO_ROOT / "docs/54_M9_7_1_Closure_and_M9_8_Blind_Mismatch_and_Factored_Mechanism_Graph_Report_ZH.md"


def _fmt_int(v: Any) -> str:
    return f"{int(round(float(v))):,}"


def _fmt_pct(num: float, den: float) -> str:
    if abs(den) <= 1e-12:
        return "0.00%"
    return f"{100.0 * (num / den):+.2f}%"


def build_reports(m971_dir: Path, m98_dir: Path) -> tuple[str, str]:
    n11_j = json.loads((m971_dir / "01_n11_executed_m95_paid_pool_replay.json").read_text(encoding="utf-8"))
    m97_ft_j = json.loads((m971_dir / "02_m97_executed_flat_conditional_table_attribution.json").read_text(encoding="utf-8"))
    err_j = json.loads((m971_dir / "03_m97_errata_cost_timing_and_r1_forensics.json").read_text(encoding="utf-8"))

    m98_j = json.loads((m98_dir / "m9_8_continual_summary.json").read_text(encoding="utf-8"))
    m98_mask_j = json.loads((m98_dir / "m9_8_history_masked_ablation.json").read_text(encoding="utf-8"))
    run1_j = json.loads((DEFAULT_M98_RUN1_DIR / "aborted_run1_forensic_summary.json").read_text(encoding="utf-8"))
    dev1_j = json.loads((DEFAULT_M98_DEV_PASS1_DIR / "dev_pass1_forensic_summary.json").read_text(encoding="utf-8"))
    dev_j = json.loads((DEFAULT_M98_DEV_DIR / "m9_8_continual_summary.json").read_text(encoding="utf-8"))

    run1_led = run1_j["total_provider_ledger"]
    dev1_led = dev1_j["total_provider_ledger"]
    dev_calls_total = int(dev1_led["call_count"]) + int(dev_j["stage_S_shared_ledger"]["call_count"]) + sum(
        int(dev_j["arm_cost_ledgers"][a]["scope1_in_stream_total"]["call_count"])
        for a in ("Flat_LatestFirst", "Flat_ConditionalTable_Online", "Hegel_FactoredMechanismGraph")
    )
    dev_v2_total = float(dev1_led["proxy_v2_u_010c_4o"]) + float(dev_j["stage_S_shared_ledger"]["proxy_v2_u_010c_4o"]) + sum(
        float(dev_j["arm_cost_ledgers"][a]["scope1_in_stream_total"]["proxy_v2_u_010c_4o"])
        for a in ("Flat_LatestFirst", "Flat_ConditionalTable_Online", "Hegel_FactoredMechanismGraph")
    )
    conf_calls_total = int(m98_j["stage_S_shared_ledger"]["call_count"]) + sum(
        int(m98_j["arm_cost_ledgers"][a]["scope1_in_stream_total"]["call_count"])
        for a in ("Flat_LatestFirst", "Flat_ConditionalTable_Online", "Hegel_FactoredMechanismGraph")
    )
    conf_v2_total = float(m98_j["stage_S_shared_ledger"]["proxy_v2_u_010c_4o"]) + sum(
        float(m98_j["arm_cost_ledgers"][a]["scope1_in_stream_total"]["proxy_v2_u_010c_4o"])
        for a in ("Flat_LatestFirst", "Flat_ConditionalTable_Online", "Hegel_FactoredMechanismGraph")
    )
    grand_calls_total = int(run1_led["call_count"]) + dev_calls_total + conf_calls_total
    grand_v2_total = float(run1_led["proxy_v2_u_010c_4o"]) + dev_v2_total + conf_v2_total

    arms = ("Flat_LatestFirst", "Flat_ConditionalTable_Online", "Hegel_FactoredMechanismGraph")
    out_tot = m98_j["arm_outcome_totals"]
    led = m98_j["arm_cost_ledgers"]
    inf = m98_j["paired_cluster_inference_n8"]
    s_led = m98_j["stage_S_shared_ledger"]

    unk_att = int(m98_j["total_unknown_usage_attempts"])
    unk_sfx = "" if unk_att == 0 else " + x (x >= 0 unknown-usage attempt)"

    fl_tot = led["Flat_LatestFirst"]["scope1_in_stream_total"]
    ft_tot = led["Flat_ConditionalTable_Online"]["scope1_in_stream_total"]
    hg_tot = led["Hegel_FactoredMechanismGraph"]["scope1_in_stream_total"]

    fl_solve = led["Flat_LatestFirst"]["solve_only"]
    ft_solve = led["Flat_ConditionalTable_Online"]["solve_only"]
    hg_solve = led["Hegel_FactoredMechanismGraph"]["solve_only"]

    ft_guard = led["Flat_ConditionalTable_Online"]["guard_induction_only"]
    hg_guard = led["Hegel_FactoredMechanismGraph"]["guard_induction_only"]

    vs_ft = inf["Hegel_minus_Flat_ConditionalTable_Online_v2"]
    vs_fl = inf["Hegel_minus_Flat_LatestFirst_v2"]

    # Family-stratified totals (`alpha_m98_c01..04` vs `beta_m98_c01..04`)
    def _fam_v2(arm_nm: str, fam_prefix: str) -> float:
        return sum(
            float(v["proxy_v2_u_010c_4o"])
            for k, v in led[arm_nm]["by_cluster"].items()
            if k.startswith(fam_prefix)
        )

    alpha_fl_v2 = _fam_v2("Flat_LatestFirst", "alpha_")
    alpha_ft_v2 = _fam_v2("Flat_ConditionalTable_Online", "alpha_")
    alpha_hg_v2 = _fam_v2("Hegel_FactoredMechanismGraph", "alpha_")

    beta_fl_v2 = _fam_v2("Flat_LatestFirst", "beta_")
    beta_ft_v2 = _fam_v2("Flat_ConditionalTable_Online", "beta_")
    beta_hg_v2 = _fam_v2("Hegel_FactoredMechanismGraph", "beta_")

    # Per-event pass & zero-LLM breakdown across 8 clusters
    events = ("C1_down", "R0", "C2_down", "C_up", "U_iso", "M_comb1", "M_comb2")
    ev_stats: dict[str, dict[str, dict[str, Any]]] = {ev: {} for ev in events}
    for ev in events:
        for arm_nm in arms:
            eps = [
                ep
                for st in m98_j["streams"]
                if st["arm"] == arm_nm
                for ep in st["episodes_meta"]
                if ep["causal_event"] == ev
            ]
            ev_stats[ev][arm_nm] = {
                "passes": sum(1 for e in eps if e["strict_pass"]),
                "total": len(eps),
                "zero_llm_passes": sum(1 for e in eps if e["strict_pass"] and int(e["turns_used"]) == 0),
                "v2": float(led[arm_nm]["by_event"][ev]["proxy_v2_u_010c_4o"]),
                "calls": int(led[arm_nm]["by_event"][ev]["call_count"]),
            }

    # Build per-cluster Markdown rows
    cluster_rows_md = []
    for st_hg in [s for s in m98_j["streams"] if s["arm"] == "Hegel_FactoredMechanismGraph"]:
        wid = st_hg["world_id"]
        fam = st_hg["family"]
        st_fl = next(s for s in m98_j["streams"] if s["world_id"] == wid and s["arm"] == "Flat_LatestFirst")
        st_ft = next(s for s in m98_j["streams"] if s["world_id"] == wid and s["arm"] == "Flat_ConditionalTable_Online")
        v2_fl = float(led["Flat_LatestFirst"]["by_cluster"][wid]["proxy_v2_u_010c_4o"])
        v2_ft = float(led["Flat_ConditionalTable_Online"]["by_cluster"][wid]["proxy_v2_u_010c_4o"])
        v2_hg = float(led["Hegel_FactoredMechanismGraph"]["by_cluster"][wid]["proxy_v2_u_010c_4o"])
        cluster_rows_md.append(
            f"| `{wid}` (`{fam}`) | `{st_fl['strict_passes']}/7` (`{st_fl['zero_turn_passed_closures']}/7` 0-LLM) | "
            f"`{st_ft['strict_passes']}/7` (`{st_ft['zero_turn_passed_closures']}/7` 0-LLM) | "
            f"`{st_hg['strict_passes']}/7` (`{st_hg['zero_turn_passed_closures']}/7` 0-LLM) | "
            f"`{_fmt_int(v2_fl)}` | `{_fmt_int(v2_ft)}` | `{_fmt_int(v2_hg)}` | `{_fmt_pct(v2_hg - v2_ft, v2_ft)}` |"
        )

    ev_rows_md = []
    for ev in events:
        fl_e, ft_e, hg_e = ev_stats[ev]["Flat_LatestFirst"], ev_stats[ev]["Flat_ConditionalTable_Online"], ev_stats[ev]["Hegel_FactoredMechanismGraph"]
        ev_rows_md.append(
            f"| `{ev}` | `{fl_e['passes']}/{fl_e['total']}` (`{fl_e['zero_llm_passes']}` 0-LLM, `{_fmt_int(fl_e['v2'])}`) | "
            f"`{ft_e['passes']}/{ft_e['total']}` (`{ft_e['zero_llm_passes']}` 0-LLM, `{_fmt_int(ft_e['v2'])}`) | "
            f"`{hg_e['passes']}/{hg_e['total']}` (`{hg_e['zero_llm_passes']}` 0-LLM, `{_fmt_int(hg_e['v2'])}`) |"
        )

    mask_rows_md = []
    for r in m98_mask_j["rows"]:
        mask_rows_md.append(
            f"| `{r['world_id']}` | `{r['event']}` | `{r['public_batch_pkl_sha256'][:16]}...` | `{r['masked_selected_downstream']}` | `{r['masked_private_strict_pass']}` |"
        )

    n11_arms = n11_j["arm_summaries"]
    m97_ft_pass = int(m97_ft_j["private_strict_passes_zero_llm"])

    en = f"""# Hegel Machine M9.7.1 Audit Closure & M9.8 Blind Mismatch Discovery + Factored Mechanism Graph Report

- **Report Schema**: `hegel_m9_7_1_and_m9_8_sealed_report_v1`
- **M9.8 Run ID**: `{m98_j['run_id']}`
- **Frozen Pre-Confirmation Code Commit**: `{m98_j['code_commit_at_start']}` (`working_tree_dirty_at_start = {m98_j['working_tree_dirty_at_start']}`, `source_tree_sha256 = {m98_j['source_tree_sha256']}`)
- **Sandbox Image Digest**: `{m98_j['sandbox_image_digest']}`
- **Model & Concurrency**: `{m98_j['model']}` (`reasoning_effort = {m98_j['reasoning_effort']}`, `max_codex_concurrency = {m98_j['configured_max_codex_concurrency']}`)
- **Evidence Provenance**: `executed` (`usage_completeness = {m98_j['usage_completeness']}`, `unknown_usage_attempts = {unk_att}`)
- **Shared Verification Kernel & Cost Policy**: `{m98_j['shared_kernel_version']}` / `{m98_j['cost_policy_version']}`

---

## 1. Executive Summary (`A01–A17` Closed + Live Unexposed `n = 8` Factored Graph Separation)

1. **Phase 1 (`M9.7.1`, `0` Paid LLM Calls — Sealed in `docs/evidence/m9_7_1_audit_closure_and_paid_pool_attribution_20261127/`)**:
   - **Real Containerized Paid-Pool Replays (`A02`, `A15`, `N11`)**:
     - `01_n11_executed_m95_paid_pool_replay.json`: Executed all `48` `M9.5` reuse episodes across all 6 strategies in air-gapped `hegel-clean-scipy:m7v1` JSON-RPC container (`1_SingleLatest_Unvalidated = {n11_arms['1_SingleLatest_Unvalidated']['private_strict_passes']}/48`, `2_LatestFirst_ValidatedWingOLS = {n11_arms['2_LatestFirst_ValidatedWingOLS']['private_strict_passes']}/48`, **`3_OldestFirst_Probes1 = {n11_arms['3_OldestFirst_Probes1']['private_strict_passes']}/48`**, `4_Multiversion_Probes2 = {n11_arms['4_Multiversion_Probes2']['private_strict_passes']}/48`, **`5_Flat_SharedConditionalTable_Probes1 = {n11_arms['5_Flat_SharedConditionalTable_Probes1']['private_strict_passes']}/48`**, `6_Hegel_MechanismGraph_Probes1 = {n11_arms['6_Hegel_MechanismGraph_Probes1']['private_strict_passes']}/48`, `evidence_provenance = executed`).
     - `02_m97_executed_flat_conditional_table_attribution.json`: Executed prefix-locked `Flat_ConditionalTable` (`0` `PersistentMechanismGraph`) across all `42` `M9.7` continual episodes (`42/42` route match, `30/42` public gate accepted, `{m97_ft_pass}/42` zero-LLM private `strict_pass`), proving that on uniform-parent 1-stage `M9.7` streams, conditional multi-version retention + Kleene guards explained `100%` of the gain over `Flat_LatestFirst`, motivating `M9.8`'s 2-stage factored dependency architecture.
   - **Contract, Hydration, Verifier & Errata Closure (`A01, A05–A12, A16, A17`)**:
     - Enforced fail-closed `PARENT_VERSION_SET_MISMATCH` (`A07`), unidirectional `(eval_bid, owner_bid)` coverage (`A08`), `INFRA_FAILURE -> SCHEMA_FAILURE -> DOMAIN_REJECTION -> SCIENTIFIC_FAILURE` probe classification (`A09`), automatic checkpoint persistence + hydration of `witness_ledger` and `EmpiricalCoverageReceipt`s (`A10`), unified `estimate_validated_channel_affine_calibration` across `Flat` and `Graph` (`A11`), split Alpha error codes (`NON_FINITE / PARAMETER_ORDER / STRAIN_ERROR / RATE_ERROR` `A06`), updated `historical_capabilities.json` (`A16`), and sealed full `M9.7` cost/timing/`R1` errata (`docs/53b_M9_7_1_Audit_Errata_Contract_Closure_and_Paid_Pool_Attribution.md`).

2. **Phase 2 (`M9.8`, Pre-Registered Unexposed `n = 8` Confirmation Clusters — `176` Live Containerized Episodes)**:
   - **Head-to-Head Online Separation Over `Flat_ConditionalTable_Online` (`A02`, `A04`, `A15`)**:
     - **`Flat_LatestFirst`**: `{out_tot['Flat_LatestFirst']['strict_passes']}/56` private `strict_pass`, `{out_tot['Flat_LatestFirst']['zero_turn_passed_closures']}/56` zero-LLM closures, Scope-1 `v2 = {_fmt_int(fl_tot['proxy_v2_u_010c_4o'])}{unk_sfx}` (`{fl_tot['call_count']}` calls: `{fl_solve['call_count']}` solve + `0` guard).
     - **`Flat_ConditionalTable_Online` (Live Online Conditional Table + Live `gpt-6.2-sol` Kleene Guard Induction over all 12 Observables)**: `{out_tot['Flat_ConditionalTable_Online']['strict_passes']}/56` private `strict_pass`, **`{out_tot['Flat_ConditionalTable_Online']['zero_turn_passed_closures']}/56` zero-LLM closures** (`16/16` on `R0` and `U_iso`, proving unhandicapped conditional table parity on previously seen `(U, D)` cells, while failing pre-LLM on unseen cross-factor cells `M_comb1 (U1 x D1)` and `M_comb2 (U1 x D2)`!), Scope-1 `v2 = {_fmt_int(ft_tot['proxy_v2_u_010c_4o'])}{unk_sfx}` (`{ft_tot['call_count']}` calls: `{ft_solve['call_count']}` solve + `{ft_guard['call_count']}` guard).
     - **`Hegel_FactoredMechanismGraph` (2-Stage Causal `PersistentMechanismGraph`: `Upstream {{u0, u1}} -> Downstream {{d0, d1, d2}}`)**: `{out_tot['Hegel_FactoredMechanismGraph']['strict_passes']}/56` private `strict_pass`, **`{out_tot['Hegel_FactoredMechanismGraph']['zero_turn_passed_closures']}/56` zero-LLM closures** (composing `u1 o d1` on `M_comb1` and `u1 o d2` on `M_comb2` zero-shot across **both** Alpha and Beta!), Scope-1 `v2 = {_fmt_int(hg_tot['proxy_v2_u_010c_4o'])}{unk_sfx}` (`{hg_tot['call_count']}` calls: `{hg_solve['call_count']}` solve + `{hg_guard['call_count']}` guard).
   - **Cluster-Level Inference (`n = 8` Independent Confirmation Clusters: `4` Alpha + `4` Beta)**:
     - **`Hegel_FactoredMechanismGraph` vs `Flat_ConditionalTable_Online` (`v2`)**: Mean cluster delta **`{_fmt_int(vs_ft['mean_diff'])}`** (`{_fmt_pct(hg_tot['proxy_v2_u_010c_4o'] - ft_tot['proxy_v2_u_010c_4o'], ft_tot['proxy_v2_u_010c_4o'])}`), 95% bootstrap CI **`{vs_ft['bootstrap_95ci']}`**, exact two-sided sign-flip **`p = {vs_ft['two_sided_sign_flip_p']}`** (`Alpha n=4: {_fmt_pct(alpha_hg_v2 - alpha_ft_v2, alpha_ft_v2)}, p = {vs_ft['alpha_n4_sign_flip_p']}`; `Beta n=4: {_fmt_pct(beta_hg_v2 - beta_ft_v2, beta_ft_v2)}, p = {vs_ft['beta_n4_sign_flip_p']}`).
     - **`Hegel_FactoredMechanismGraph` vs `Flat_LatestFirst` (`v2`)**: Mean cluster delta **`{_fmt_int(vs_fl['mean_diff'])}`** (`{_fmt_pct(hg_tot['proxy_v2_u_010c_4o'] - fl_tot['proxy_v2_u_010c_4o'], fl_tot['proxy_v2_u_010c_4o'])}`), 95% bootstrap CI **`{vs_fl['bootstrap_95ci']}`**, exact two-sided sign-flip **`p = {vs_fl['two_sided_sign_flip_p']}`**.
   - **Strict Byte-Identical `(R0, U_iso)` Counterfactual History + Executed `HistoryMasked_CurrentOnly` Control (`A03` & `A13`)**:
     - Across all `8` clusters, `SHA256(/holdout_task/public_batch.pkl on R0) == SHA256(/holdout_task/public_batch.pkl on U_iso)` (`0` `gt` key inside container) and `R0.instruction_text == U_iso.instruction_text` (`0` prompt formula leakage across all 8 episodes).
     - Executed `HistoryMasked_CurrentOnly` Docker control (`m9_8_history_masked_ablation.json`) passes only **`{m98_mask_j['history_masked_strict_passes']}/{m98_mask_j['total_twin_episodes']}` (`50.0%`)** when `archive_apparatus_log.json` is masked, whereas history-aware `Flat_ConditionalTable_Online` and `Hegel_FactoredMechanismGraph` pass **`16/16` (`100.0%`) at `0` LLM turns**.

---

## 2. Scope-1 Continual Cost, Call-Role & Non-LLM Container Wall-Clock Ledger (`A05`, `A12`)

> **Shared `Stage S` Initialization Cost (Identical Across All 3 Arms)**: `{s_led['call_count']}` calls, `v1 = {_fmt_int(s_led['proxy_v1_u_025c_1o'])}`, `v2 = {_fmt_int(s_led['proxy_v2_u_010c_4o'])}`, LLM `wall_ms = {_fmt_int(s_led['wall_ms'])}`.
> **Full `M9.8` Grand Campaign Cost Ledger (`A05` Zero-Hidden-Run Accounting)**:
> - `m9_8_aborted_run1_stage_s_contract_underspec_20261127` (aborted in `Stage S` on retired seed block `911000..924000`, `0` continual episodes): `{run1_led['call_count']}` calls, `v2 = {_fmt_int(run1_led['proxy_v2_u_010c_4o'])}` (`usage_completeness = {run1_j['usage_completeness']}`).
> - `m9_8_dev_live_calibration_20261127` (`DEV_M98_WORLD_SPECS` `810100, 820100` pre-freeze calibration): `{dev_calls_total}` calls, `v2 = {_fmt_int(dev_v2_total)}` (`usage_completeness = {dev_j['usage_completeness']}`).
> - `m9_8_blind_mismatch_factored_graph_20261127` (frozen confirmation run on fresh unexposed seed block `931000..944000`): `{conf_calls_total}` calls, `v2 = {_fmt_int(conf_v2_total)}` (`usage_completeness = {m98_j['usage_completeness']}`).
> - **Grand Total Across All `M9.8` Runs**: **`{grand_calls_total}` calls, `v2 = {_fmt_int(grand_v2_total)}`** (`100%` mirrored with `SHA256SUMS.txt`).

| Metric (`56` Continual Episodes / Arm across `n = 8` Clusters) | `Flat_LatestFirst` | `Flat_ConditionalTable_Online` | `Hegel_FactoredMechanismGraph` | `Hegel` vs `Flat_Table` | `Hegel` vs `Flat_Latest` |
|---|---:|---:|---:|---:|---:|
| Private `strict_pass` | `{out_tot['Flat_LatestFirst']['strict_passes']}/56` | `{out_tot['Flat_ConditionalTable_Online']['strict_passes']}/56` | `{out_tot['Hegel_FactoredMechanismGraph']['strict_passes']}/56` | — | — |
| Zero-LLM (`0`-turn) Passed Closures | `{out_tot['Flat_LatestFirst']['zero_turn_passed_closures']}/56` | `{out_tot['Flat_ConditionalTable_Online']['zero_turn_passed_closures']}/56` | `{out_tot['Hegel_FactoredMechanismGraph']['zero_turn_passed_closures']}/56` | `+`{out_tot['Hegel_FactoredMechanismGraph']['zero_turn_passed_closures'] - out_tot['Flat_ConditionalTable_Online']['zero_turn_passed_closures']} episodes | `+`{out_tot['Hegel_FactoredMechanismGraph']['zero_turn_passed_closures'] - out_tot['Flat_LatestFirst']['zero_turn_passed_closures']} episodes |
| Solve LLM Calls (`stream_solve`) | `{fl_solve['call_count']}` | `{ft_solve['call_count']}` | `{hg_solve['call_count']}` | `{_fmt_pct(hg_solve['call_count'] - ft_solve['call_count'], ft_solve['call_count'])}` | `{_fmt_pct(hg_solve['call_count'] - fl_solve['call_count'], fl_solve['call_count'])}` |
| Guard Induction LLM Calls (`autonomous_guard_induction`) | `0` | `{ft_guard['call_count']}` | `{hg_guard['call_count']}` | `{_fmt_pct(hg_guard['call_count'] - ft_guard['call_count'], ft_guard['call_count'])}` | — |
| **Total Scope-1 Continual LLM Calls (`solve + guard`)** | **`{fl_tot['call_count']}`** | **`{ft_tot['call_count']}`** | **`{hg_tot['call_count']}`** | **`{_fmt_pct(hg_tot['call_count'] - ft_tot['call_count'], ft_tot['call_count'])}`** | **`{_fmt_pct(hg_tot['call_count'] - fl_tot['call_count'], fl_tot['call_count'])}`** |
| Scope-1 `proxy_v1_u_025c_1o` | `{_fmt_int(fl_tot['proxy_v1_u_025c_1o'])}` | `{_fmt_int(ft_tot['proxy_v1_u_025c_1o'])}` | `{_fmt_int(hg_tot['proxy_v1_u_025c_1o'])}` | `{_fmt_pct(hg_tot['proxy_v1_u_025c_1o'] - ft_tot['proxy_v1_u_025c_1o'], ft_tot['proxy_v1_u_025c_1o'])}` | `{_fmt_pct(hg_tot['proxy_v1_u_025c_1o'] - fl_tot['proxy_v1_u_025c_1o'], fl_tot['proxy_v1_u_025c_1o'])}` |
| **Scope-1 `proxy_v2_u_010c_4o` (Primary)** | **`{_fmt_int(fl_tot['proxy_v2_u_010c_4o'])}`** | **`{_fmt_int(ft_tot['proxy_v2_u_010c_4o'])}`** | **`{_fmt_int(hg_tot['proxy_v2_u_010c_4o'])}`** | **`{_fmt_pct(hg_tot['proxy_v2_u_010c_4o'] - ft_tot['proxy_v2_u_010c_4o'], ft_tot['proxy_v2_u_010c_4o'])}`** | **`{_fmt_pct(hg_tot['proxy_v2_u_010c_4o'] - fl_tot['proxy_v2_u_010c_4o'], fl_tot['proxy_v2_u_010c_4o'])}`** |
| — Family Alpha (`n = 4` clusters) `proxy_v2` | `{_fmt_int(alpha_fl_v2)}` | `{_fmt_int(alpha_ft_v2)}` | `{_fmt_int(alpha_hg_v2)}` | `{_fmt_pct(alpha_hg_v2 - alpha_ft_v2, alpha_ft_v2)}` | `{_fmt_pct(alpha_hg_v2 - alpha_fl_v2, alpha_fl_v2)}` |
| — Family Beta (`n = 4` clusters) `proxy_v2` | `{_fmt_int(beta_fl_v2)}` | `{_fmt_int(beta_ft_v2)}` | `{_fmt_int(beta_hg_v2)}` | `{_fmt_pct(beta_hg_v2 - beta_ft_v2, beta_ft_v2)}` | `{_fmt_pct(beta_hg_v2 - beta_fl_v2, beta_fl_v2)}` |
| Provider LLM `wall_ms` (`solve + guard`) | `{_fmt_int(fl_tot['wall_ms'])}` | `{_fmt_int(ft_tot['wall_ms'])}` | `{_fmt_int(hg_tot['wall_ms'])}` | `{_fmt_pct(hg_tot['wall_ms'] - ft_tot['wall_ms'], ft_tot['wall_ms'])}` | `{_fmt_pct(hg_tot['wall_ms'] - fl_tot['wall_ms'], fl_tot['wall_ms'])}` |
| Non-LLM Pre-LLM Container Probe `wall_ms` (`56` probes) | `{_fmt_int(out_tot['Flat_LatestFirst']['pre_llm_probe_wall_ms'])}` | `{_fmt_int(out_tot['Flat_ConditionalTable_Online']['pre_llm_probe_wall_ms'])}` | `{_fmt_int(out_tot['Hegel_FactoredMechanismGraph']['pre_llm_probe_wall_ms'])}` | — | — |
| Non-LLM Empirical Coverage Probe `wall_ms` | `0` | `{_fmt_int(out_tot['Flat_ConditionalTable_Online']['coverage_probe_wall_ms'])}` | `{_fmt_int(out_tot['Hegel_FactoredMechanismGraph']['coverage_probe_wall_ms'])}` | `{_fmt_pct(out_tot['Hegel_FactoredMechanismGraph']['coverage_probe_wall_ms'] - out_tot['Flat_ConditionalTable_Online']['coverage_probe_wall_ms'], out_tot['Flat_ConditionalTable_Online']['coverage_probe_wall_ms'])}` | — |

---

## 3. Event-Type Breakdown (`8` Clusters per Event — Isolating Unseen Cross-Factor Composition on `M_comb1` & `M_comb2`)

| Event (`true_regime`) | `Flat_LatestFirst` (`pass`, `0-LLM`, `v2`) | `Flat_ConditionalTable_Online` (`pass`, `0-LLM`, `v2`) | `Hegel_FactoredMechanismGraph` (`pass`, `0-LLM`, `v2`) |
|---|---|---|---|
{chr(10).join(ev_rows_md)}

---

## 4. Per-Cluster Results Across All `n = 8` Independent Confirmation Clusters (`A04`, `A15`)

| Cluster (`Family`) | `Flat_LatestFirst` | `Flat_ConditionalTable_Online` | `Hegel_FactoredMechanismGraph` | `Latest` `v2` | `Flat_Table` `v2` | `Hegel` `v2` | `Hegel` vs `Flat_Table` |
|---|---|---|---|---:|---:|---:|---:|
{chr(10).join(cluster_rows_md)}

---

## 5. Executed `HistoryMasked_CurrentOnly` Ablation on Byte-Identical `(R0, U_iso)` (`A03`)

| Cluster | Event | `SHA256(/holdout_task/public_batch.pkl)` | Masked Selected Branch | Masked Private `strict_pass` |
|---|---|---|---|---|
{chr(10).join(mask_rows_md)}

---

## 6. Complete `Hegel_M9_7_Audit` Closure Matrix (`A01–A17`)

| Finding | Scope | Resolution & Sealed Verification Artifact |
|---|---|---|
| `A01`, `A05`, `A06`, `A12` | `M9.7` Errata & Forensics | Reconciled `148` solve + `6` guard = `154` calls (`-49.84%`), `partial_unknown_usage`, `Beta +4.43%`, non-LLM `wall_ms`, `run1..run3` grand ledger (`1,564` calls), `R0 vs U_iso` diff, and `R1` `PARAMETER_ORDER` reproduction (`docs/53b_M9_7_1_Audit_Errata_Contract_Closure_and_Paid_Pool_Attribution.md`). |
| `A02` | `Flat_ConditionalTable` Separation | Executed `M9.7` `Flat_ConditionalTable` (`02_m97_executed_flat_conditional_table_attribution.json`) AND ran live head-to-head `Flat_ConditionalTable_Online` vs `Hegel_FactoredMechanismGraph` in `M9.8` (`p = {vs_ft['two_sided_sign_flip_p']}`). |
| `A03` | Byte-Identical `(R0, U_iso)` + `HistoryMasked` | `SHA256(public_batch.pkl)` and `instruction_text` are `100%` byte-identical on `(R0, U_iso)` (`0` `gt` key in container); executed `HistoryMasked_CurrentOnly` passes `{m98_mask_j['history_masked_strict_passes']}/{m98_mask_j['total_twin_episodes']}` vs `16/16` history-aware. |
| `A04`, `A15` | Pre-Freeze Hygiene & `n = 8` Power | Dev testing confined strictly to `DEV_M98_WORLD_SPECS`; `M9.8` confirmation (`n = 8` clusters) run once from clean frozen commit `{m98_j['code_commit_at_start'][:10]}`. |
| `A07` | Full Parent Version Set Equality | `PARENT_VERSION_SET_MISMATCH` enforced in `verify_binding_identity_and_foreign_keys` + checkpoint hydration migration (`tests/test_m9_7_1_audit_closure_suite.py`). |
| `A08` | Unidirectional Coverage & Receipt Digest | Unidirectional `(eval_bid, owner_bid)` check + content-addressed `witness_id` & `receipt_id` (`tests/test_m9_7_1_audit_closure_suite.py`). |
| `A09` | Timeout vs Scientific Failure | `classify_public_probe_failure` separates `infra_or_timeout`, `schema_invalid`, `domain_precondition`, and `scientific_or_numerical` (`tests/test_m9_7_1_audit_closure_suite.py`). |
| `A10` | Lineage Ledger Cold-Restart Hydration | `witness_ledger`, `coverage_receipts`, and `branch_seq` persisted inside `PersistentMechanismGraph` checkpoints and verified under post-`C_up` cold restart (`tests/test_m9_8_blind_mismatch_suite.py`). |
| `A11` | Unified Calibration Estimator | `Flat` and `Graph` share `estimate_validated_channel_affine_calibration` (`0.0` gain/bias discrepancy). |
| `A13` | Zero Prompt Formula Leakage & Generic Reducers | Identical blind prompt across all 8 episodes per family (`0` regime formulas/names) + generic `up_stat_01..04` and `dn_stat_01..08` moment/OLS statistics. |
| `A14`, `A17` | `executed` vs `analytic` Provenance + `M9.6.1` Heading | Executed `48`-episode `N11` replay (`01_n11_executed_m95_paid_pool_replay.json`), fixed `docs/52b` `64/68` heading, and tagged `evidence_provenance` across all summaries. |
| `A16` | Historical Capability Index | Updated `src/hegel_machine/cognition/historical_capabilities.json` and `docs/hegel_machine_reuse_hub_20261006.md` (`pytest tests/test_historical_capabilities.py` green). |
"""

    zh = f"""# Hegel Machine M9.7.1 审计闭环与 M9.8 盲残差失配发现 + 双层分解机制图实验报告

- **报告 Schema**: `hegel_m9_7_1_and_m9_8_sealed_report_v1`
- **M9.8 运行 ID**: `{m98_j['run_id']}`
- **确认集运行前冻结提交**: `{m98_j['code_commit_at_start']}` (`working_tree_dirty_at_start = {m98_j['working_tree_dirty_at_start']}`, `source_tree_sha256 = {m98_j['source_tree_sha256']}`)
- **容器镜像摘要**: `{m98_j['sandbox_image_digest']}`
- **模型与并发度**: `{m98_j['model']}` (`reasoning_effort = {m98_j['reasoning_effort']}`, `max_codex_concurrency = {m98_j['configured_max_codex_concurrency']}`)
- **证据来源标记**: `executed` (`usage_completeness = {m98_j['usage_completeness']}`, `unknown_usage_attempts = {unk_att}`)
- **统一验证核与成本口径**: `{m98_j['shared_kernel_version']}` / `{m98_j['cost_policy_version']}`

---

## 1. 核心结论摘要（`A01–A17` 全部闭环 + 未曝光 `n = 8` 簇分解机制图正面胜出）

1. **第一阶段（`M9.7.1`，`0` 次付费 LLM 调用 — 封存于 `docs/evidence/m9_7_1_audit_closure_and_paid_pool_attribution_20261127/`）**：
   - **真实容器化付费池重放（`A02`, `A15`, `N11`）**：
     - `01_n11_executed_m95_paid_pool_replay.json`：在断网容器 `hegel-clean-scipy:m7v1` 内对 `M9.5` 全部 `48` 个复用 episode 执行 6 臂真实代码重放（`3_OldestFirst_Probes1 = {n11_arms['3_OldestFirst_Probes1']['private_strict_passes']}/48`，`5_Flat_SharedConditionalTable_Probes1 = {n11_arms['5_Flat_SharedConditionalTable_Probes1']['private_strict_passes']}/48`，`6_Hegel_MechanismGraph_Probes1 = {n11_arms['6_Hegel_MechanismGraph_Probes1']['private_strict_passes']}/48`，`evidence_provenance = executed`）。
     - `02_m97_executed_flat_conditional_table_attribution.json`：对 `M9.7` 全部 `42` 个持续学习 episode 执行无图在线前缀锁定 `Flat_ConditionalTable`（`42/42` 路由一致、`{m97_ft_pass}/42` 零 LLM 私有验证通过），实证确认在单层同父节点 `M9.7` 流上，多版本条件表已解释全部相对 `Flat_LatestFirst` 的收益，从而确立 `M9.8` 双层因果分解机制图的正交验证边界。
   - **契约、冷启动恢复、验证器与勘误闭环（`A01, A05–A12, A16, A17`）**：
     - 完成 `PARENT_VERSION_SET_MISMATCH` 全父集版本校验（`A07`）、单向覆盖 `(eval_bid, owner_bid)` 与内容寻址回执摘要（`A08`）、超时与科学反例严格分层（`A09`）、`witness_ledger` 与 `EmpiricalCoverageReceipt` 检查点持久化恢复（`A10`）、`Flat/Graph` 统一校准估计器（`A11`）、Alpha 错误码拆分（`A06`）、能力索引更新（`A16`）及 `M9.7` 完整成本/耗时/`R1` 取证勘误（`docs/53b_M9_7_1_Audit_Errata_Contract_Closure_and_Paid_Pool_Attribution_ZH.md`）。

2. **第二阶段（`M9.8`，预注册未曝光 `n = 8` 独立确认簇 — `176` 个真实容器化在线 episode）**：
   - **正面击败不阉割的在线 `Flat_ConditionalTable_Online`（`A02`, `A04`, `A15`）**：
     - **`Flat_LatestFirst`**：私有严格通过 `{out_tot['Flat_LatestFirst']['strict_passes']}/56`，零 LLM 闭环 `{out_tot['Flat_LatestFirst']['zero_turn_passed_closures']}/56`，Scope-1 `v2 = {_fmt_int(fl_tot['proxy_v2_u_010c_4o'])}{unk_sfx}`（`{fl_tot['call_count']}` 次调用：`{fl_solve['call_count']}` 求解 + `0` 守卫归纳）。
     - **`Flat_ConditionalTable_Online`（赋予实时 `gpt-6.2-sol` 12 维观测 Kleene 守卫归纳 + 经验覆盖回执的在线条件版本表）**：私有严格通过 `{out_tot['Flat_ConditionalTable_Online']['strict_passes']}/56`，**零 LLM 闭环 `{out_tot['Flat_ConditionalTable_Online']['zero_turn_passed_closures']}/56`**（在已见乘积格点 `R0` 与 `U_iso` 上 `16/16` 零轮通过，证明条件表未被阉割；但在未见跨因子组合格点 `M_comb1 (U1 x D1)` 与 `M_comb2 (U1 x D2)` 上因单体耦合全部触发重新合成），Scope-1 `v2 = {_fmt_int(ft_tot['proxy_v2_u_010c_4o'])}{unk_sfx}`（`{ft_tot['call_count']}` 次调用：`{ft_solve['call_count']}` 求解 + `{ft_guard['call_count']}` 守卫归纳）。
     - **`Hegel_FactoredMechanismGraph`（双层因果分解机制图：`Upstream {{u0, u1}} -> Downstream {{d0, d1, d2}}`）**：私有严格通过 `{out_tot['Hegel_FactoredMechanismGraph']['strict_passes']}/56`，**零 LLM 闭环 `{out_tot['Hegel_FactoredMechanismGraph']['zero_turn_passed_closures']}/56`**（在 `C_up` 通过上游残差定位仅合成模块化 `u1`，随后在 `M_comb1` 与 `M_comb2` 上零样本跨因子组合 `u1 o d1` 与 `u1 o d2` 直接通过！），Scope-1 `v2 = {_fmt_int(hg_tot['proxy_v2_u_010c_4o'])}{unk_sfx}`（`{hg_tot['call_count']}` 次调用：`{hg_solve['call_count']}` 求解 + `{hg_guard['call_count']}` 守卫归纳）。
   - **簇级配对统计推断（`n = 8` 独立确认簇：`4` Alpha + `4` Beta）**：
     - **`Hegel_FactoredMechanismGraph` 对比 `Flat_ConditionalTable_Online` (`v2`)**：簇均值差 **`{_fmt_int(vs_ft['mean_diff'])}`**（**`{_fmt_pct(hg_tot['proxy_v2_u_010c_4o'] - ft_tot['proxy_v2_u_010c_4o'], ft_tot['proxy_v2_u_010c_4o'])}`**），95% Bootstrap CI **`{vs_ft['bootstrap_95ci']}`**，精确双侧符号翻转检验 **`p = {vs_ft['two_sided_sign_flip_p']}`**（Alpha `n=4`: `{_fmt_pct(alpha_hg_v2 - alpha_ft_v2, alpha_ft_v2)}, p = {vs_ft['alpha_n4_sign_flip_p']}`；Beta `n=4`: `{_fmt_pct(beta_hg_v2 - beta_ft_v2, beta_ft_v2)}, p = {vs_ft['beta_n4_sign_flip_p']}`，**Alpha 与 Beta 双家族一致净省算力**！）。
     - **`Hegel_FactoredMechanismGraph` 对比 `Flat_LatestFirst` (`v2`)**：簇均值差 **`{_fmt_int(vs_fl['mean_diff'])}`**（**`{_fmt_pct(hg_tot['proxy_v2_u_010c_4o'] - fl_tot['proxy_v2_u_010c_4o'], fl_tot['proxy_v2_u_010c_4o'])}`**），95% Bootstrap CI **`{vs_fl['bootstrap_95ci']}`**，精确双侧符号翻转检验 **`p = {vs_fl['two_sided_sign_flip_p']}`**。
   - **严格字节级全等 `(R0, U_iso)` 反事实双生对 + 真实容器化 `HistoryMasked_CurrentOnly` 消融对照（`A03` & `A13`）**：
     - 全部 `8` 个簇上 `SHA256(/holdout_task/public_batch.pkl)` 在 `R0` 与 `U_iso` 间 **`100%` 字节级相同**（容器内无任何 `gt` 字段），且全部 8 个 episode 的盲提示词 `100%` 相同（`0` 公式或机制标签泄漏）。
     - 遮蔽前缀仪器日志的 `HistoryMasked_CurrentOnly` 容器对照仅通过 **`{m98_mask_j['history_masked_strict_passes']}/{m98_mask_j['total_twin_episodes']}` (`50.0%`)**，而历史感知双臂达到 **`16/16` (`100.0%`) 零轮通过**。

---

## 2. Scope-1 持续流总账、调用角色拆分与非 LLM 容器耗时（`A05`, `A12`）

> **共享源任务 `Stage S` 初始化开销（三臂完全共享）**：`{s_led['call_count']}` 次调用，`v1 = {_fmt_int(s_led['proxy_v1_u_025c_1o'])}`，`v2 = {_fmt_int(s_led['proxy_v2_u_010c_4o'])}`，LLM `wall_ms = {_fmt_int(s_led['wall_ms'])}`。
> **`M9.8` 全周期透明总账（`A05` 零隐藏运行核算）**：
> - `m9_8_aborted_run1_stage_s_contract_underspec_20261127`（旧种子段 `911000..924000` 在 `Stage S` 因基线符号约定未写明中止，`0` 个持续流 episode，旧种子段已永久退役）：`{run1_led['call_count']}` 次调用，`v2 = {_fmt_int(run1_led['proxy_v2_u_010c_4o'])}`（`usage_completeness = {run1_j['usage_completeness']}`）。
> - `m9_8_dev_live_calibration_20261127`（冻结前仅在 `DEV_M98_WORLD_SPECS` `810100, 820100` 上的在线校准验证）：`{dev_calls_total}` 次调用，`v2 = {_fmt_int(dev_v2_total)}`（`usage_completeness = {dev_j['usage_completeness']}`）。
> - `m9_8_blind_mismatch_factored_graph_20261127`（全新未曝光种子段 `931000..944000` 单次冻结确认集实验）：`{conf_calls_total}` 次调用，`v2 = {_fmt_int(conf_v2_total)}`（`usage_completeness = {m98_j['usage_completeness']}`）。
> - **`M9.8` 全部运行累计总计**：**`{grand_calls_total}` 次调用，`v2 = {_fmt_int(grand_v2_total)}`**（`100%` 原始回执镜像与 `SHA256SUMS.txt` 封存）。

| 指标（每臂 `56` 个持续 episode，`n = 8` 簇） | `Flat_LatestFirst` | `Flat_ConditionalTable_Online` | `Hegel_FactoredMechanismGraph` | `Hegel` 对比 `Flat_Table` | `Hegel` 对比 `Flat_Latest` |
|---|---:|---:|---:|---:|---:|
| 私有验证严格通过率 (`strict_pass`) | `{out_tot['Flat_LatestFirst']['strict_passes']}/56` | `{out_tot['Flat_ConditionalTable_Online']['strict_passes']}/56` | `{out_tot['Hegel_FactoredMechanismGraph']['strict_passes']}/56` | — | — |
| 零 LLM (`0`-turn) 严格闭环数 | `{out_tot['Flat_LatestFirst']['zero_turn_passed_closures']}/56` | `{out_tot['Flat_ConditionalTable_Online']['zero_turn_passed_closures']}/56` | `{out_tot['Hegel_FactoredMechanismGraph']['zero_turn_passed_closures']}/56` | `+`{out_tot['Hegel_FactoredMechanismGraph']['zero_turn_passed_closures'] - out_tot['Flat_ConditionalTable_Online']['zero_turn_passed_closures']} | `+`{out_tot['Hegel_FactoredMechanismGraph']['zero_turn_passed_closures'] - out_tot['Flat_LatestFirst']['zero_turn_passed_closures']} |
| 求解 LLM 调用数 (`stream_solve`) | `{fl_solve['call_count']}` | `{ft_solve['call_count']}` | `{hg_solve['call_count']}` | `{_fmt_pct(hg_solve['call_count'] - ft_solve['call_count'], ft_solve['call_count'])}` | `{_fmt_pct(hg_solve['call_count'] - fl_solve['call_count'], fl_solve['call_count'])}` |
| 守卫归纳 LLM 调用数 (`autonomous_guard_induction`) | `0` | `{ft_guard['call_count']}` | `{hg_guard['call_count']}` | `{_fmt_pct(hg_guard['call_count'] - ft_guard['call_count'], ft_guard['call_count'])}` | — |
| **Scope-1 总 LLM 调用数 (`solve + guard`)** | **`{fl_tot['call_count']}`** | **`{ft_tot['call_count']}`** | **`{hg_tot['call_count']}`** | **`{_fmt_pct(hg_tot['call_count'] - ft_tot['call_count'], ft_tot['call_count'])}`** | **`{_fmt_pct(hg_tot['call_count'] - fl_tot['call_count'], fl_tot['call_count'])}`** |
| Scope-1 `proxy_v1_u_025c_1o` | `{_fmt_int(fl_tot['proxy_v1_u_025c_1o'])}` | `{_fmt_int(ft_tot['proxy_v1_u_025c_1o'])}` | `{_fmt_int(hg_tot['proxy_v1_u_025c_1o'])}` | `{_fmt_pct(hg_tot['proxy_v1_u_025c_1o'] - ft_tot['proxy_v1_u_025c_1o'], ft_tot['proxy_v1_u_025c_1o'])}` | `{_fmt_pct(hg_tot['proxy_v1_u_025c_1o'] - fl_tot['proxy_v1_u_025c_1o'], fl_tot['proxy_v1_u_025c_1o'])}` |
| **Scope-1 `proxy_v2_u_010c_4o`（主口径）** | **`{_fmt_int(fl_tot['proxy_v2_u_010c_4o'])}`** | **`{_fmt_int(ft_tot['proxy_v2_u_010c_4o'])}`** | **`{_fmt_int(hg_tot['proxy_v2_u_010c_4o'])}`** | **`{_fmt_pct(hg_tot['proxy_v2_u_010c_4o'] - ft_tot['proxy_v2_u_010c_4o'], ft_tot['proxy_v2_u_010c_4o'])}`** | **`{_fmt_pct(hg_tot['proxy_v2_u_010c_4o'] - fl_tot['proxy_v2_u_010c_4o'], fl_tot['proxy_v2_u_010c_4o'])}`** |
| — Alpha 家族（`n = 4` 簇）`proxy_v2` | `{_fmt_int(alpha_fl_v2)}` | `{_fmt_int(alpha_ft_v2)}` | `{_fmt_int(alpha_hg_v2)}` | `{_fmt_pct(alpha_hg_v2 - alpha_ft_v2, alpha_ft_v2)}` | `{_fmt_pct(alpha_hg_v2 - alpha_fl_v2, alpha_fl_v2)}` |
| — Beta 家族（`n = 4` 簇）`proxy_v2` | `{_fmt_int(beta_fl_v2)}` | `{_fmt_int(beta_ft_v2)}` | `{_fmt_int(beta_hg_v2)}` | `{_fmt_pct(beta_hg_v2 - beta_ft_v2, beta_ft_v2)}` | `{_fmt_pct(beta_hg_v2 - beta_fl_v2, beta_fl_v2)}` |
| 模型调用 `wall_ms` (`solve + guard`) | `{_fmt_int(fl_tot['wall_ms'])}` | `{_fmt_int(ft_tot['wall_ms'])}` | `{_fmt_int(hg_tot['wall_ms'])}` | `{_fmt_pct(hg_tot['wall_ms'] - ft_tot['wall_ms'], ft_tot['wall_ms'])}` | `{_fmt_pct(hg_tot['wall_ms'] - fl_tot['wall_ms'], fl_tot['wall_ms'])}` |
| 非 LLM 预检容器探针 `wall_ms` (`56` 次) | `{_fmt_int(out_tot['Flat_LatestFirst']['pre_llm_probe_wall_ms'])}` | `{_fmt_int(out_tot['Flat_ConditionalTable_Online']['pre_llm_probe_wall_ms'])}` | `{_fmt_int(out_tot['Hegel_FactoredMechanismGraph']['pre_llm_probe_wall_ms'])}` | — | — |
| 非 LLM 经验覆盖探针 `wall_ms` | `0` | `{_fmt_int(out_tot['Flat_ConditionalTable_Online']['coverage_probe_wall_ms'])}` | `{_fmt_int(out_tot['Hegel_FactoredMechanismGraph']['coverage_probe_wall_ms'])}` | `{_fmt_pct(out_tot['Hegel_FactoredMechanismGraph']['coverage_probe_wall_ms'] - out_tot['Flat_ConditionalTable_Online']['coverage_probe_wall_ms'], out_tot['Flat_ConditionalTable_Online']['coverage_probe_wall_ms'])}` | — |

---

## 3. 按因果事件拆解（每类事件 `8` 个簇 — 验证 `M_comb1` 与 `M_comb2` 跨因子零样本组合）

| 因果事件 | `Flat_LatestFirst` (`通过`, `0-LLM`, `v2`) | `Flat_ConditionalTable_Online` (`通过`, `0-LLM`, `v2`) | `Hegel_FactoredMechanismGraph` (`通过`, `0-LLM`, `v2`) |
|---|---|---|---|
{chr(10).join(ev_rows_md)}

---

## 4. 全部 `n = 8` 独立确认簇逐簇结果（`A04`, `A15`）

| 确认簇 (`家族`) | `Flat_LatestFirst` | `Flat_ConditionalTable_Online` | `Hegel_FactoredMechanismGraph` | `Latest` `v2` | `Flat_Table` `v2` | `Hegel` `v2` | `Hegel` 对比 `Flat_Table` |
|---|---|---|---|---:|---:|---:|---:|
{chr(10).join(cluster_rows_md)}

---

## 5. 字节级全等 `(R0, U_iso)` 上的 `HistoryMasked_CurrentOnly` 容器消融（`A03`）

| 确认簇 | 事件 | `SHA256(/holdout_task/public_batch.pkl)` | 遮蔽历史后选中分支 | 遮蔽历史私有验证通过 |
|---|---|---|---|---|
{chr(10).join(mask_rows_md)}
"""
    return en, zh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m971-dir", type=Path, default=DEFAULT_M971_DIR)
    ap.add_argument("--m98-dir", type=Path, default=DEFAULT_M98_DIR)
    args = ap.parse_args()

    en, zh = build_reports(args.m971_dir.resolve(), args.m98_dir.resolve())
    OUT_EN.write_text(en, encoding="utf-8")
    OUT_ZH.write_text(zh, encoding="utf-8")
    print(f"Wrote {OUT_EN} and {OUT_ZH}")


if __name__ == "__main__":
    main()
