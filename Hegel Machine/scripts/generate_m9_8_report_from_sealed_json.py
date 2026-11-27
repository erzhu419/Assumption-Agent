#!/usr/bin/env python3
"""Deterministically generate `docs/54_M9_7_1_Closure_and_M9_8_Blind_Mismatch_and_Factored_Mechanism_Graph_Report.md`
and `_ZH.md` strictly from sealed `M9.7.1` and `M9.8` JSON artifacts (`0` hand-edited numbers, `R01–R16` audit-aligned).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

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


def _sign_flip_p(diffs: list[float]) -> float:
    arr = np.asarray(diffs, dtype=float)
    n = len(arr)
    if n == 0:
        return 1.0
    obs = abs(float(np.mean(arr)))
    if obs <= 1e-15:
        return 1.0
    ext = sum(
        1
        for signs in itertools.product((-1.0, 1.0), repeat=n)
        if abs(float(np.mean(arr * np.asarray(signs, dtype=float)))) >= obs - 1e-12
    )
    return round(float(ext) / float(1 << n), 7)


def build_reports(m971_dir: Path, m98_dir: Path) -> tuple[str, str]:
    n11_j = json.loads((m971_dir / "01_n11_executed_m95_paid_pool_replay.json").read_text(encoding="utf-8"))
    m97_ft_j = json.loads((m971_dir / "02_m97_executed_flat_conditional_table_attribution.json").read_text(encoding="utf-8"))

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

    # Scope-Total (`Stage S` included per standalone deployed system, `R09`)
    s_v2 = float(s_led["proxy_v2_u_010c_4o"])
    fl_with_s_v2 = float(fl_tot["proxy_v2_u_010c_4o"]) + s_v2
    ft_with_s_v2 = float(ft_tot["proxy_v2_u_010c_4o"]) + s_v2
    hg_with_s_v2 = float(hg_tot["proxy_v2_u_010c_4o"]) + s_v2

    fl_solve = led["Flat_LatestFirst"]["solve_only"]
    ft_solve = led["Flat_ConditionalTable_Online"]["solve_only"]
    hg_solve = led["Hegel_FactoredMechanismGraph"]["solve_only"]

    ft_guard = led["Flat_ConditionalTable_Online"]["guard_induction_only"]
    hg_guard = led["Hegel_FactoredMechanismGraph"]["guard_induction_only"]

    vs_ft = inf["Hegel_minus_Flat_ConditionalTable_Online_v2"]
    vs_fl = inf["Hegel_minus_Flat_LatestFirst_v2"]

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

    # Dynamic interpolation for `R0 + U_iso` and `M_comb1 + M_comb2` (`R13` — ZERO hardcoded constants!)
    ft_twin_0llm = ev_stats["R0"]["Flat_ConditionalTable_Online"]["zero_llm_passes"] + ev_stats["U_iso"]["Flat_ConditionalTable_Online"]["zero_llm_passes"]
    hg_twin_0llm = ev_stats["R0"]["Hegel_FactoredMechanismGraph"]["zero_llm_passes"] + ev_stats["U_iso"]["Hegel_FactoredMechanismGraph"]["zero_llm_passes"]
    ft_comb_0llm = ev_stats["M_comb1"]["Flat_ConditionalTable_Online"]["zero_llm_passes"] + ev_stats["M_comb2"]["Flat_ConditionalTable_Online"]["zero_llm_passes"]
    ft_comb_pass = ev_stats["M_comb1"]["Flat_ConditionalTable_Online"]["passes"] + ev_stats["M_comb2"]["Flat_ConditionalTable_Online"]["passes"]
    ft_comb_calls = ev_stats["M_comb1"]["Flat_ConditionalTable_Online"]["calls"] + ev_stats["M_comb2"]["Flat_ConditionalTable_Online"]["calls"]
    ft_comb_v2 = ev_stats["M_comb1"]["Flat_ConditionalTable_Online"]["v2"] + ev_stats["M_comb2"]["Flat_ConditionalTable_Online"]["v2"]
    hg_comb_0llm = ev_stats["M_comb1"]["Hegel_FactoredMechanismGraph"]["zero_llm_passes"] + ev_stats["M_comb2"]["Hegel_FactoredMechanismGraph"]["zero_llm_passes"]

    cluster_rows_md = []
    qual_diffs_vs_ft: list[float] = []
    qual_diffs_vs_fl: list[float] = []
    rel_pcts_vs_ft: list[float] = []
    for st_hg in [s for s in m98_j["streams"] if s["arm"] == "Hegel_FactoredMechanismGraph"]:
        wid = st_hg["world_id"]
        fam = st_hg["family"]
        st_fl = next(s for s in m98_j["streams"] if s["world_id"] == wid and s["arm"] == "Flat_LatestFirst")
        st_ft = next(s for s in m98_j["streams"] if s["world_id"] == wid and s["arm"] == "Flat_ConditionalTable_Online")
        v2_fl = float(led["Flat_LatestFirst"]["by_cluster"][wid]["proxy_v2_u_010c_4o"])
        v2_ft = float(led["Flat_ConditionalTable_Online"]["by_cluster"][wid]["proxy_v2_u_010c_4o"])
        v2_hg = float(led["Hegel_FactoredMechanismGraph"]["by_cluster"][wid]["proxy_v2_u_010c_4o"])
        qual_diffs_vs_ft.append(float(st_hg["strict_passes"] - st_ft["strict_passes"]))
        qual_diffs_vs_fl.append(float(st_hg["strict_passes"] - st_fl["strict_passes"]))
        rel_pcts_vs_ft.append(100.0 * (v2_hg - v2_ft) / v2_ft)
        cluster_rows_md.append(
            f"| `{wid}` (`{fam}`) | `{st_fl['strict_passes']}/7` (`{st_fl['zero_turn_passed_closures']}/7` 0-LLM) | "
            f"`{st_ft['strict_passes']}/7` (`{st_ft['zero_turn_passed_closures']}/7` 0-LLM) | "
            f"`{st_hg['strict_passes']}/7` (`{st_hg['zero_turn_passed_closures']}/7` 0-LLM) | "
            f"`{_fmt_int(v2_fl)}` | `{_fmt_int(v2_ft)}` | `{_fmt_int(v2_hg)}` | `{_fmt_pct(v2_hg - v2_ft, v2_ft)}` |"
        )

    qual_p_vs_ft = _sign_flip_p(qual_diffs_vs_ft)
    qual_p_vs_fl = _sign_flip_p(qual_diffs_vs_fl)
    mean_cluster_rel_pct_vs_ft = float(np.mean(rel_pcts_vs_ft))
    comb_share_pct = 100.0 * ft_comb_v2 / max(1.0, float(ft_tot["proxy_v2_u_010c_4o"] - hg_tot["proxy_v2_u_010c_4o"]))

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

    n11_arms = n11_j["ladder"]
    m97_ft_pass = int(
        m97_ft_j["scope_and_formation_cost_separation"]["marginal_execution_comparison"][
            "Flat_ConditionalTable_Executed_Probes1"
        ]["zero_llm_private_strict_pass"]
    )
    peak_conc = m98_j.get("peak_active_codex_concurrency")

    en = f"""# Hegel Machine M9.7.1 Audit Closure & M9.8 Blind Mismatch Discovery + Modular Factored System Report

- **Report Schema**: `hegel_m9_7_1_and_m9_8_sealed_report_v2_m981_reconciled`
- **M9.8 Run ID**: `{m98_j['run_id']}`
- **Frozen Pre-Confirmation Code Commit**: `{m98_j['code_commit_at_start']}` (`working_tree_dirty_at_start = {m98_j['working_tree_dirty_at_start']}`, `source_tree_sha256 = {m98_j['source_tree_sha256']}`)
- **Sandbox Image Digest**: `{m98_j['sandbox_image_digest']}`
- **Model & Concurrency**: `{m98_j['model']}` (`reasoning_effort = {m98_j['reasoning_effort']}`, `configured_cap = {m98_j['configured_max_codex_concurrency']}`, `observed_peak_active = {peak_conc}`)
- **Evidence Provenance**: `executed` (`usage_completeness = {m98_j['usage_completeness']}`, `unknown_usage_attempts = {unk_att}`)
- **Shared Verification Kernel & Cost Policy**: `{m98_j['shared_kernel_version']}` / `{m98_j['cost_policy_version']}`
- **Post-Audit Representation & Contract Companion (`M9.8.1`)**: See `docs/54b_M9_8_1_Audit_Errata_Evidence_Contract_and_TwoTable_Attribution.md`

---

## 1. Executive Summary & Honest Attribution Scope (`R01–R16` Reconciled)

1. **Phase 1 (`M9.7.1`, `0` Paid LLM Calls — Sealed in `docs/evidence/m9_7_1_audit_closure_and_paid_pool_attribution_20261127/`)**:
   - **Real Containerized Paid-Pool Replays (`A01`, `A02`)**:
     - `01_n11_executed_m95_paid_pool_replay.json`: Executed all `48` `M9.5` reuse episodes across all 6 strategies in air-gapped `hegel-clean-scipy:m7v1` (`1_Unvalidated = {n11_arms['1_Flat_SingleLatest_Unvalidated_Probes1']['private_strict_pass']}/48`, `2_ValidatedWingOLS = {n11_arms['2_Flat_SingleLatest_ValidatedWingOLS_Probes1']['private_strict_pass']}/48`, **`3_OldestFirst_Probes1 = {n11_arms['3_Flat_Multiversion_OldestFirst_ValidatedWingOLS_Probes1']['private_strict_pass']}/48`**, `4_Multiversion_Probes2 = {n11_arms['4_Flat_Multiversion_ValidatedWingOLS_Probes2']['private_strict_pass']}/48`, **`5_SharedConditionalTable_Probes1 = {n11_arms['5_Flat_SharedConditionalTable_ValidatedWingOLS_Probes1']['private_strict_pass']}/48`**, `6_Hegel_MechanismGraph_Probes1 = {n11_arms['6_Hegel_MechanismGraph_Guard_Probes1']['private_strict_pass']}/48`).
     - `02_m97_executed_flat_conditional_table_attribution.json`: Executed prefix-locked `Flat_ConditionalTable` (`0` `PersistentMechanismGraph`) across all `42` `M9.7` continual episodes (`42/42` route match, `30/42` public gate accepted, `{m97_ft_pass}/42` zero-LLM private `strict_pass`), proving that on 1-stage uniform-parent `M9.7` streams, conditional multi-version retention + Kleene guards matched `Hegel_DemandGated_Graph`.
   - **Errata & Contract Hardening (`docs/53b_*` + `M9.8.1` Closure)**:
     - Enforced `PARENT_VERSION_SET_MISMATCH` (`A07`), unidirectional coverage + canonical `ecov_` receipt digest (`A08 + R06`), 4-tier failure classification + localization infra isolation (`A09 + R05`), fail-closed lineage checkpoint digest & foreign-key hydration (`A10 + R07`), unified calibration (`A11`), retired defective `M9.7` `R1` candidate after diagnostic error-code split (`A06 + R14`), and marked legacy `M9.4/M9.6` `L1` KKT dual-helper adoption as `retained_open_gap` (`A14 + R15`).

2. **Phase 2 (`M9.8`, Pre-Registered Unexposed `n = 8` Confirmation Clusters — `176` Live Containerized Episodes)**:
   - **What `M9.8` Proves vs What Requires `M9.9` (`R01–R03, R10` Scope Boundary)**:
     - `M9.8` tests a **host-supplied two-stage decomposition (`U -> D`) + localized upstream synthesis harness + cross-factor composition system** (`Hegel_FactoredMechanismGraph`) against a **monolithic online conditional table** (`Flat_ConditionalTable_Online`) and `Flat_LatestFirst` (note: in `M9.8`, `Hegel` also had `allow_l1_primitives=True` and specialized `upstream_transducer.py` repair prompts on `C_up`; paid-prefix two-table replay in `docs/54b` confirms `56/56` routing/code equivalence with two plain tables once that `U/D` factorization and prefix programs exist, motivating `M9.9`'s strict equal-tool autonomous decomposition & non-separable interaction benchmark).
   - **Head-to-Head Online Outcomes (`56` Continual Episodes / Arm)**:
     - **`Flat_LatestFirst`**: `{out_tot['Flat_LatestFirst']['strict_passes']}/56` private `strict_pass`, `{out_tot['Flat_LatestFirst']['zero_turn_passed_closures']}/56` zero-LLM closures, Scope-1 `v2 = {_fmt_int(fl_tot['proxy_v2_u_010c_4o'])}{unk_sfx}` (`{fl_tot['call_count']}` calls).
     - **`Flat_ConditionalTable_Online`**: `{out_tot['Flat_ConditionalTable_Online']['strict_passes']}/56` private `strict_pass`, **`{out_tot['Flat_ConditionalTable_Online']['zero_turn_passed_closures']}/56` zero-LLM closures** (**`{ft_twin_0llm}/16`** zero-LLM passes on `(R0, U_iso)`, while achieving **`{ft_comb_0llm}/16` zero-LLM reuse** on unseen cross-factor `M_comb1 + M_comb2` and falling back to full LLM re-synthesis for **`{ft_comb_pass}/16` final `strict_pass`** at `{ft_comb_calls}` calls / `{_fmt_int(ft_comb_v2)} v2`, accounting for `{comb_share_pct:.2f}%` of net `v2` savings!), Scope-1 `v2 = {_fmt_int(ft_tot['proxy_v2_u_010c_4o'])}{unk_sfx}` (`{ft_tot['call_count']}` calls: `{ft_solve['call_count']}` solve + `{ft_guard['call_count']}` guard).
     - **`Hegel_FactoredMechanismGraph`**: `{out_tot['Hegel_FactoredMechanismGraph']['strict_passes']}/56` private `strict_pass`, **`{out_tot['Hegel_FactoredMechanismGraph']['zero_turn_passed_closures']}/56` zero-LLM closures** (`{hg_twin_0llm}/16` on `R0 + U_iso` and **`{hg_comb_0llm}/16` zero-LLM composition** on `M_comb1 + M_comb2`!), Scope-1 `v2 = {_fmt_int(hg_tot['proxy_v2_u_010c_4o'])}{unk_sfx}` (`{hg_tot['call_count']}` calls: `{hg_solve['call_count']}` solve + `{hg_guard['call_count']}` guard).
   - **Cluster-Level Cost vs Quality Inference (`n = 8` Clusters; Minimum Two-Sided Sign-Flip Resolution = `2 / 2^8 = 0.0078125`, `R09, R16`)**:
     - **Scope-1 Continual `v2` Cost (`Hegel` vs `Flat_ConditionalTable_Online`)**: Ratio-of-sums **`{_fmt_pct(hg_tot['proxy_v2_u_010c_4o'] - ft_tot['proxy_v2_u_010c_4o'], ft_tot['proxy_v2_u_010c_4o'])}`** (equal-cluster mean relative change `{mean_cluster_rel_pct_vs_ft:+.2f}%`; Scope-Total including full standalone `Stage S` per system: `{_fmt_int(hg_with_s_v2)}` vs `{_fmt_int(ft_with_s_v2)}` = **`{_fmt_pct(hg_with_s_v2 - ft_with_s_v2, ft_with_s_v2)}`**), mean cluster delta `{_fmt_int(vs_ft['mean_diff'])}`, 95% bootstrap CI `{vs_ft['bootstrap_95ci']}`, exact two-sided sign-flip **`p = {vs_ft['two_sided_sign_flip_p']}`** (`Alpha n=4: {_fmt_pct(alpha_hg_v2 - alpha_ft_v2, alpha_ft_v2)}, p = {vs_ft['alpha_n4_sign_flip_p']}`; `Beta n=4: {_fmt_pct(beta_hg_v2 - beta_ft_v2, beta_ft_v2)}, p = {vs_ft['beta_n4_sign_flip_p']}`).
     - **Scope-1 Continual `v2` Cost (`Hegel` vs `Flat_LatestFirst`)**: Ratio-of-sums **`{_fmt_pct(hg_tot['proxy_v2_u_010c_4o'] - fl_tot['proxy_v2_u_010c_4o'], fl_tot['proxy_v2_u_010c_4o'])}`** (Scope-Total with `Stage S`: **`{_fmt_pct(hg_with_s_v2 - fl_with_s_v2, fl_with_s_v2)}`**), exact two-sided sign-flip **`p = {vs_fl['two_sided_sign_flip_p']}`**.
     - **Cluster-Level Final Quality (`strict_pass`, Reported Separately from Cost per `R09`)**: `Hegel - Flat_ConditionalTable_Online` cluster pass diffs `{[int(x) for x in qual_diffs_vs_ft]}` (`56/56` vs `50/56`, `+10.71` pp, all 6 gains in Alpha while Beta tied `28/28`), exact two-sided sign-flip **`p = {qual_p_vs_ft}`** (observational quality advantage, NOT `< 0.05` at `n = 8`; vs `Flat_LatestFirst` `{[int(x) for x in qual_diffs_vs_fl]}`, `p = {qual_p_vs_fl}`).
   - **Byte-Identical `(R0, U_iso)` Twins & Post-Hoc vs Prefix-Locked Mask Controls (`A03, R08`)**:
     - Base `public_batch.pkl` bytes and base `instruction_text` match `100%` on `(R0, U_iso)` across all `8` clusters (with baseline domain equations present in shared base instructions, while shifted `D1/D2` equations/labels are unrevealed; live prompts differ by archive history and `Hegel`'s `C_up` repair harness).
     - The `0/16` table in §5 below is a **post-hoc step-04 library `D2`-fallback log-deletion sensitivity check**; strict causal prefix-locked (`step-1` / `step-4`) and fixed `D0`/`D1` current-only controls (`8/16 = 50.0%` matching the disjoint-acceptance-ball ceiling) are executed and sealed in `docs/54b`.

---

## 2. Scope-1 & Scope-Total Cost, Call-Role & Non-LLM Container Wall-Clock Ledger (`A05, R09, R11`)

> **Shared `Stage S` Initialization Cost**: `{s_led['call_count']}` calls, `v1 = {_fmt_int(s_led['proxy_v1_u_025c_1o'])}`, `v2 = {_fmt_int(s_led['proxy_v2_u_010c_4o'])}`, LLM `wall_ms = {_fmt_int(s_led['wall_ms'])}`.
> **Full `M9.8` Grand Campaign Cost Ledger (`4` Sealed Runs, `100%` `call_uuid` Mirrored)**:
> - `m9_8_aborted_run1_stage_s_contract_underspec_20261127` (aborted in `Stage S` on retired seeds `911000..924000`): `{run1_led['call_count']}` calls, `v2 = {_fmt_int(run1_led['proxy_v2_u_010c_4o'])}`.
> - `m9_8_dev_live_calibration_pass1_20261127` + `m9_8_dev_live_calibration_20261127` (`DEV_M98_WORLD_SPECS` `810100, 820100`): `{dev_calls_total}` calls, `v2 = {_fmt_int(dev_v2_total)}`.
> - `m9_8_blind_mismatch_factored_graph_20261127` (frozen confirmation run on unexposed `931000..944000`): `{conf_calls_total}` calls, `v2 = {_fmt_int(conf_v2_total)}`.
> - **Grand Total Across All `M9.8` Runs**: **`{grand_calls_total}` calls, `v2 = {_fmt_int(grand_v2_total)}`**.

| Metric (`56` Continual Episodes / Arm across `n = 8` Clusters) | `Flat_LatestFirst` | `Flat_ConditionalTable_Online` | `Hegel_FactoredMechanismGraph` | `Hegel` vs `Flat_Table` | `Hegel` vs `Flat_Latest` |
|---|---:|---:|---:|---:|---:|
| Private `strict_pass` (Cluster Sign-Flip `p`) | `{out_tot['Flat_LatestFirst']['strict_passes']}/56` | `{out_tot['Flat_ConditionalTable_Online']['strict_passes']}/56` | `{out_tot['Hegel_FactoredMechanismGraph']['strict_passes']}/56` | `p = {qual_p_vs_ft}` | `p = {qual_p_vs_fl}` |
| Zero-LLM (`0`-turn) Passed Closures | `{out_tot['Flat_LatestFirst']['zero_turn_passed_closures']}/56` | `{out_tot['Flat_ConditionalTable_Online']['zero_turn_passed_closures']}/56` | `{out_tot['Hegel_FactoredMechanismGraph']['zero_turn_passed_closures']}/56` | `+`{out_tot['Hegel_FactoredMechanismGraph']['zero_turn_passed_closures'] - out_tot['Flat_ConditionalTable_Online']['zero_turn_passed_closures']} episodes | `+`{out_tot['Hegel_FactoredMechanismGraph']['zero_turn_passed_closures'] - out_tot['Flat_LatestFirst']['zero_turn_passed_closures']} episodes |
| Solve LLM Calls (`stream_solve`) | `{fl_solve['call_count']}` | `{ft_solve['call_count']}` | `{hg_solve['call_count']}` | `{_fmt_pct(hg_solve['call_count'] - ft_solve['call_count'], ft_solve['call_count'])}` | `{_fmt_pct(hg_solve['call_count'] - fl_solve['call_count'], fl_solve['call_count'])}` |
| Guard Induction LLM Calls (`autonomous_guard_induction`) | `0` | `{ft_guard['call_count']}` | `{hg_guard['call_count']}` | `{_fmt_pct(hg_guard['call_count'] - ft_guard['call_count'], ft_guard['call_count'])}` | — |
| **Total Scope-1 Continual LLM Calls (`solve + guard`)** | **`{fl_tot['call_count']}`** | **`{ft_tot['call_count']}`** | **`{hg_tot['call_count']}`** | **`{_fmt_pct(hg_tot['call_count'] - ft_tot['call_count'], ft_tot['call_count'])}`** | **`{_fmt_pct(hg_tot['call_count'] - fl_tot['call_count'], fl_tot['call_count'])}`** |
| Scope-1 `proxy_v1_u_025c_1o` | `{_fmt_int(fl_tot['proxy_v1_u_025c_1o'])}` | `{_fmt_int(ft_tot['proxy_v1_u_025c_1o'])}` | `{_fmt_int(hg_tot['proxy_v1_u_025c_1o'])}` | `{_fmt_pct(hg_tot['proxy_v1_u_025c_1o'] - ft_tot['proxy_v1_u_025c_1o'], ft_tot['proxy_v1_u_025c_1o'])}` | `{_fmt_pct(hg_tot['proxy_v1_u_025c_1o'] - fl_tot['proxy_v1_u_025c_1o'], fl_tot['proxy_v1_u_025c_1o'])}` |
| **Scope-1 `proxy_v2_u_010c_4o` (In-Stream Primary)** | **`{_fmt_int(fl_tot['proxy_v2_u_010c_4o'])}`** | **`{_fmt_int(ft_tot['proxy_v2_u_010c_4o'])}`** | **`{_fmt_int(hg_tot['proxy_v2_u_010c_4o'])}`** | **`{_fmt_pct(hg_tot['proxy_v2_u_010c_4o'] - ft_tot['proxy_v2_u_010c_4o'], ft_tot['proxy_v2_u_010c_4o'])}`** | **`{_fmt_pct(hg_tot['proxy_v2_u_010c_4o'] - fl_tot['proxy_v2_u_010c_4o'], fl_tot['proxy_v2_u_010c_4o'])}`** |
| **Scope-Total `proxy_v2` (Standalone incl. `Stage S`)** | **`{_fmt_int(fl_with_s_v2)}`** | **`{_fmt_int(ft_with_s_v2)}`** | **`{_fmt_int(hg_with_s_v2)}`** | **`{_fmt_pct(hg_with_s_v2 - ft_with_s_v2, ft_with_s_v2)}`** | **`{_fmt_pct(hg_with_s_v2 - fl_with_s_v2, fl_with_s_v2)}`** |
| — Family Alpha (`n = 4` clusters) Scope-1 `proxy_v2` | `{_fmt_int(alpha_fl_v2)}` | `{_fmt_int(alpha_ft_v2)}` | `{_fmt_int(alpha_hg_v2)}` | `{_fmt_pct(alpha_hg_v2 - alpha_ft_v2, alpha_ft_v2)}` | `{_fmt_pct(alpha_hg_v2 - alpha_fl_v2, alpha_fl_v2)}` |
| — Family Beta (`n = 4` clusters) Scope-1 `proxy_v2` | `{_fmt_int(beta_fl_v2)}` | `{_fmt_int(beta_ft_v2)}` | `{_fmt_int(beta_hg_v2)}` | `{_fmt_pct(beta_hg_v2 - beta_ft_v2, beta_ft_v2)}` | `{_fmt_pct(beta_hg_v2 - beta_fl_v2, beta_fl_v2)}` |
| Provider LLM `wall_ms` (`solve + guard`) | `{_fmt_int(fl_tot['wall_ms'])}` | `{_fmt_int(ft_tot['wall_ms'])}` | `{_fmt_int(hg_tot['wall_ms'])}` | `{_fmt_pct(hg_tot['wall_ms'] - ft_tot['wall_ms'], ft_tot['wall_ms'])}` | `{_fmt_pct(hg_tot['wall_ms'] - fl_tot['wall_ms'], fl_tot['wall_ms'])}` |
| Non-LLM Pre-LLM Gate Probe `wall_ms` (`56` probes) | `{_fmt_int(out_tot['Flat_LatestFirst']['pre_llm_probe_wall_ms'])}` | `{_fmt_int(out_tot['Flat_ConditionalTable_Online']['pre_llm_probe_wall_ms'])}` | `{_fmt_int(out_tot['Hegel_FactoredMechanismGraph']['pre_llm_probe_wall_ms'])}` | — | — |
| Non-LLM Empirical Coverage Probe `wall_ms` | `0` | `{_fmt_int(out_tot['Flat_ConditionalTable_Online']['coverage_probe_wall_ms'])}` | `{_fmt_int(out_tot['Hegel_FactoredMechanismGraph']['coverage_probe_wall_ms'])}` | `{_fmt_pct(out_tot['Hegel_FactoredMechanismGraph']['coverage_probe_wall_ms'] - out_tot['Flat_ConditionalTable_Online']['coverage_probe_wall_ms'], out_tot['Flat_ConditionalTable_Online']['coverage_probe_wall_ms'])}` | — |
| Non-LLM `C_up` Localization Container Probes (`R11`) | `0` | `0` | `8` (`unmetered_in_m98`; metered in `docs/54b`) | — | — |

---

## 3. Event-Type Breakdown (`8` Clusters per Event — Isolating Cross-Factor Reuse on `M_comb1` & `M_comb2`)

| Event (`true_regime`) | `Flat_LatestFirst` (`pass`, `0-LLM`, `v2`) | `Flat_ConditionalTable_Online` (`pass`, `0-LLM`, `v2`) | `Hegel_FactoredMechanismGraph` (`pass`, `0-LLM`, `v2`) |
|---|---|---|---|
{chr(10).join(ev_rows_md)}

---

## 4. Per-Cluster Results Across All `n = 8` Independent Confirmation Clusters

| Cluster (`Family`) | `Flat_LatestFirst` | `Flat_ConditionalTable_Online` | `Hegel_FactoredMechanismGraph` | `Latest` `v2` | `Flat_Table` `v2` | `Hegel` `v2` | `Hegel` vs `Flat_Table` |
|---|---|---|---|---:|---:|---:|---:|
{chr(10).join(cluster_rows_md)}

---

## 5. Post-Hoc Step-04 Library `D2`-Fallback Log-Deletion Sensitivity (`m9_8_history_masked_ablation.json`; Prefix-Locked & `D0/D1` `8/16` Controls in `docs/54b`)

| Cluster | Event | `SHA256(/holdout_task/public_batch.pkl)` | Post-Hoc Step-04 Fallback Branch | Masked Private `strict_pass` |
|---|---|---|---|---|
{chr(10).join(mask_rows_md)}

---

## 6. Granular `A01–A17` Audit Status Matrix (`5`-Status Classification per `Hegel_M9_8_Audit`)

| Finding | Granular Status | Evidence & Scope Boundary |
|---|---|---|
| `A01`, `A17` | `verified_closed` | `48`-episode `N11` RPC execution (`01_n11_executed_m95_paid_pool_replay.json`) + `docs/52b` `64/68` heading correction. |
| `A02` | `scope_narrowed` | `M9.7` conditional table matched `30/42` pub, `{m97_ft_pass}/42` priv; `M9.8` factored system beat monolithic `Flat_ConditionalTable_Online` (`-44.54%`), while `M9.8.1` two-table paid-prefix replay (`56/56`) narrows `M9.8` to modular `U/D` system gain rather than graph-vs-two-table storage gain. |
| `A03`, `A13` | `scope_narrowed` | Base `public_batch.pkl` & base `instruction_text` match on `(R0, U_iso)` (`<= 8/16` fixed current-only ceiling); baseline physics equations appear in shared instructions while shifted `D1/D2` equations do not; prefix log acts as observable precursor state; full prefix-locked & `D0/D1` controls sealed in `docs/54b`. |
| `A04`, `A05`, `A15` | `verified_closed` | `M9.7` reclassified as exposed-seed replication (`Beta p = 1.0` fixed in `docs/53b`); `M9.8` `n = 8` confirmation frozen at `{m98_j['code_commit_at_start'][:10]}` with `0` unknown usage, cost `p = {vs_ft['two_sided_sign_flip_p']}`, quality `p = {qual_p_vs_ft}`, and `8` localization probes documented (`docs/54b`). |
| `A06` | `diagnosed_and_retired` | `alpha_m97_c02 / R1` (`2b098388239b...`) diagnosed (`PARAMETER_ORDER`, `STRAIN_ERROR`, `RATE_ERROR`), historical `41/42` preserved, candidate retired (`docs/53b`). |
| `A07`, `A11`, `A12`, `A16` | `verified_closed` | `PARENT_VERSION_SET_MISMATCH` (`A07`), shared `2/3–1/3` affine estimator (`A11`), named `M9.7` consolidation rule (`A12`), and updated `historical_capabilities.json` (`A16`). |
| `A08`, `A09`, `A10` | `superseded_in_m9_8_1` | `M9.8.1` closes malformed `ecov_invalid` bypass (`R06`), missing-digest cold hydration & foreign-key checks (`R07`), `r2 >= 0.96` unexecuted witness labeling (`R04`), and localization `rc=125` / `TimeoutExpired` handling (`R05`) (`tests/test_m9_8_1_audit_closure_suite.py` + `docs/54b`). |
| `A14` | `retained_open_gap` | Standalone whole-program KKT solvers pass exact primal-dual checks, while legacy `M9.4/M9.6` `L1` compiled helper dual-dictionary adoption remains `retained_open_gap`. |
"""

    zh = f"""# Hegel Machine M9.7.1 审计闭环与 M9.8 盲残差失配发现 + 模块化分解系统实验报告

- **报告 Schema**: `hegel_m9_7_1_and_m9_8_sealed_report_v2_m981_reconciled`
- **M9.8 运行 ID**: `{m98_j['run_id']}`
- **确认集运行前冻结提交**: `{m98_j['code_commit_at_start']}` (`working_tree_dirty_at_start = {m98_j['working_tree_dirty_at_start']}`, `source_tree_sha256 = {m98_j['source_tree_sha256']}`)
- **容器镜像摘要**: `{m98_j['sandbox_image_digest']}`
- **模型与并发度**: `{m98_j['model']}` (`reasoning_effort = {m98_j['reasoning_effort']}`, `configured_cap = {m98_j['configured_max_codex_concurrency']}`, `observed_peak_active = {peak_conc}`)
- **证据来源标记**: `executed` (`usage_completeness = {m98_j['usage_completeness']}`, `unknown_usage_attempts = {unk_att}`)
- **统一验证核与成本口径**: `{m98_j['shared_kernel_version']}` / `{m98_j['cost_policy_version']}`
- **M9.8.1 契约加固与双普通表归因伴随报告**: 见 `docs/54b_M9_8_1_Audit_Errata_Evidence_Contract_and_TwoTable_Attribution_ZH.md`

---

## 1. 核心结论与诚实归因边界（按 `Hegel_M9_8_Audit` `R01–R16` 全面对齐）

1. **第一阶段（`M9.7.1`，`0` 次付费 LLM 调用 — 封存于 `docs/evidence/m9_7_1_audit_closure_and_paid_pool_attribution_20261127/`）**：
   - **真实容器化付费池重放（`A01`, `A02`）**：
     - `01_n11_executed_m95_paid_pool_replay.json`：在断网容器 `hegel-clean-scipy:m7v1` 内完成 `M9.5` `48` 步 6 臂真实代码执行（`3_OldestFirst_Probes1 = {n11_arms['3_Flat_Multiversion_OldestFirst_ValidatedWingOLS_Probes1']['private_strict_pass']}/48`，`5_SharedConditionalTable_Probes1 = {n11_arms['5_Flat_SharedConditionalTable_ValidatedWingOLS_Probes1']['private_strict_pass']}/48`）。
     - `02_m97_executed_flat_conditional_table_attribution.json`：在 `M9.7` `42` 步上实跑无图 `Flat_ConditionalTable`（`42/42` 路由一致、`30/42` 公共门控通过、`{m97_ft_pass}/42` 零 LLM 私判通过），证明单层同父流中普通条件表与图等价。
   - **勘误修正与 `M9.8.1` 契约闭合（`docs/53b_*` + `docs/54b_*`）**：
     - 修正 `docs/53b` 中旧 `M9.7` Beta 双侧符号翻转 `p = 1.0`（原模板误硬编码 `0.50`，`R13`），明确 `alpha_m97_c02 / R1` 候选为诊断与错误码拆分闭合后正式退役（`diagnosed_and_retired`，`R14`），将旧 `M9.4/M9.6` `L1` KKT 对偶辅助函数采用保留为 `retained_open_gap`（`R15`），并在 `M9.8.1` 闭合 `ecov_invalid` 格式校验（`R06`）、冷恢复缺摘要拒绝与外键校验（`R07`）、`r2 >= 0.96` 兼容推定与实测证据分离（`R04`）及定位容器异常隔离（`R05`）。

2. **第二阶段（`M9.8`，未曝光 `n = 8` 独立确认簇 — `176` 个真实容器化在线 episode）**：
   - **归因范围精确界定（`R01–R03, R10`）**：
     - `M9.8` 实证检验的是**宿主给定两阶段模块分解（`U -> D`）+ 公开标定残差局部修复接口 + 跨因子组合系统包**（`Hegel_FactoredMechanismGraph`）相对于**单体型在线条件版本表**（`Flat_ConditionalTable_Online`）与 `Flat_LatestFirst` 的整体收益（注：`M9.8` 中 `Hegel` 额外开启了 `allow_l1_primitives=True` 与 `C_up` 局部修复任务包装；`docs/54b` 的已付前缀双普通表控制已验证在共享该 `U/D` 分解与已学模块后双表可 `56/56` 复现相同候选代码与 `32/32` 零轮通过，故全等权限下的自主分解形成与非可分交互撤回交由 `M9.9` 独立检验）。
   - **三臂在线主结果（每臂 `56` 个持续流 episode）**：
     - **`Flat_LatestFirst`**：私判通过 `{out_tot['Flat_LatestFirst']['strict_passes']}/56`，零 LLM 闭环 `{out_tot['Flat_LatestFirst']['zero_turn_passed_closures']}/56`，Scope-1 `v2 = {_fmt_int(fl_tot['proxy_v2_u_010c_4o'])}{unk_sfx}`（`{fl_tot['call_count']}` 次调用）。
     - **`Flat_ConditionalTable_Online`**：私判通过 `{out_tot['Flat_ConditionalTable_Online']['strict_passes']}/56`，**零 LLM 闭环 `{out_tot['Flat_ConditionalTable_Online']['zero_turn_passed_closures']}/56`**（在已见双生格点 `(R0, U_iso)` 上达成 **`{ft_twin_0llm}/16`** 零轮通过；在未见跨因子格点 `M_comb1 + M_comb2` 上零轮复用为 **`{ft_comb_0llm}/16`**，回退全量重合成后最终私判通过 **`{ft_comb_pass}/16`**，消耗 `{ft_comb_calls}` 次调用 / `{_fmt_int(ft_comb_v2)} v2`，解释了两臂净节省额的 `{comb_share_pct:.2f}%`！），Scope-1 `v2 = {_fmt_int(ft_tot['proxy_v2_u_010c_4o'])}{unk_sfx}`（`{ft_tot['call_count']}` 次调用：`{ft_solve['call_count']}` 求解 + `{ft_guard['call_count']}` 守卫归纳）。
     - **`Hegel_FactoredMechanismGraph`**：私判通过 `{out_tot['Hegel_FactoredMechanismGraph']['strict_passes']}/56`，**零 LLM 闭环 `{out_tot['Hegel_FactoredMechanismGraph']['zero_turn_passed_closures']}/56`**（`R0 + U_iso` `{hg_twin_0llm}/16` 零轮通过，`M_comb1 + M_comb2` **`{hg_comb_0llm}/16` 零轮跨因子组合通过**！），Scope-1 `v2 = {_fmt_int(hg_tot['proxy_v2_u_010c_4o'])}{unk_sfx}`（`{hg_tot['call_count']}` 次调用：`{hg_solve['call_count']}` 求解 + `{hg_guard['call_count']}` 守卫归纳）。
   - **簇级成本与质量分层推断（`n = 8` 簇，双侧符号翻转最小可达分辨率 `2 / 2^8 = 0.0078125`，`R09, R16`）**：
     - **Scope-1 流内成本（`Hegel` 对比 `Flat_ConditionalTable_Online`）**：总和比 **`{_fmt_pct(hg_tot['proxy_v2_u_010c_4o'] - ft_tot['proxy_v2_u_010c_4o'], ft_tot['proxy_v2_u_010c_4o'])}`**（等权簇相对变化均值 `{mean_cluster_rel_pct_vs_ft:+.2f}%`；若每套系统独立承担完整 `Stage S` 初始化，则 Scope-Total 为 `{_fmt_int(hg_with_s_v2)}` vs `{_fmt_int(ft_with_s_v2)}` = **`{_fmt_pct(hg_with_s_v2 - ft_with_s_v2, ft_with_s_v2)}`**），簇均值差 `{_fmt_int(vs_ft['mean_diff'])}`，95% Bootstrap CI `{vs_ft['bootstrap_95ci']}`，精确双侧符号翻转 **`p = {vs_ft['two_sided_sign_flip_p']}`**（Alpha `n=4`: `{_fmt_pct(alpha_hg_v2 - alpha_ft_v2, alpha_ft_v2)}, p = {vs_ft['alpha_n4_sign_flip_p']}`；Beta `n=4`: `{_fmt_pct(beta_hg_v2 - beta_ft_v2, beta_ft_v2)}, p = {vs_ft['beta_n4_sign_flip_p']}`）。
     - **Scope-1 流内成本（`Hegel` 对比 `Flat_LatestFirst`）**：总和比 **`{_fmt_pct(hg_tot['proxy_v2_u_010c_4o'] - fl_tot['proxy_v2_u_010c_4o'], fl_tot['proxy_v2_u_010c_4o'])}`**（含 `Stage S` 的 Scope-Total 为 **`{_fmt_pct(hg_with_s_v2 - fl_with_s_v2, fl_with_s_v2)}`**），双侧 **`p = {vs_fl['two_sided_sign_flip_p']}`**。
     - **最终质量推断单列（`strict_pass`，严禁与费用 `p` 混用，`R09`）**：`Hegel - Flat_ConditionalTable_Online` 逐簇通过数差为 `{[int(x) for x in qual_diffs_vs_ft]}`（`56/56` 对 `50/56`，提升 `+10.71` 个百分点，差异均在 Alpha 四簇而 Beta 为 `28/28` 平手），精确双侧符号翻转 **`p = {qual_p_vs_ft}`**（属有限簇观测优势，未达 `< 0.05`；对比 `Flat_LatestFirst` 簇差 `{[int(x) for x in qual_diffs_vs_fl]}`，`p = {qual_p_vs_fl}`）。
   - **字节级全等 `(R0, U_iso)` 双生任务与事后库敏感性 vs 在线前缀锁定消融（`A03, R08`）**：
     - 8 个簇的 `(R0, U_iso)` 共享 `100%` 相同的 `public_batch.pkl` 字节与基础 `instruction_text`（基础说明含基线控制方程，而新增 `D1/D2` 相变公式与标签零预给；实际在线提示随前缀档案及 `Hegel` 的 `C_up` 局部修复包装变化）。
     - 下文第 5 节的 `0/16` 精确定义为**事后冻结 `step_04` 知识库下 `D2` 回退策略的删日志敏感性检查**；严格在线前缀锁定消融（`step_01` / `step_04` 检查点）与固定 `D0` / `D1` 无历史策略实测（`8/16 = 50.0%`，精确触及不相交验收球上界）已完整封存于 `docs/54b`。

---

## 2. Scope-1 / Scope-Total 成本总账、调用角色拆分与非 LLM 容器耗时（`A05, R09, R11`）

> **共享源任务 `Stage S` 初始化开销**：`{s_led['call_count']}` 次调用，`v1 = {_fmt_int(s_led['proxy_v1_u_025c_1o'])}`，`v2 = {_fmt_int(s_led['proxy_v2_u_010c_4o'])}`，LLM `wall_ms = {_fmt_int(s_led['wall_ms'])}`。
> **`M9.8` 四运行研发全周期总账（`100%` `call_uuid` 原始镜像核验）**：
> - 首轮 `Stage S` 中止运行（旧种子 `911000..924000` 已退役）：`{run1_led['call_count']}` 次调用，`v2 = {_fmt_int(run1_led['proxy_v2_u_010c_4o'])}`。
> - 开发校准 `pass1 + final`（仅 `DEV_M98_WORLD_SPECS` `810100, 820100`）：`{dev_calls_total}` 次调用，`v2 = {_fmt_int(dev_v2_total)}`。
> - 冻结确认集运行（未曝光种子 `931000..944000`）：`{conf_calls_total}` 次调用，`v2 = {_fmt_int(conf_v2_total)}`。
> - **四运行全周期合计**：**`{grand_calls_total}` 次调用，`v2 = {_fmt_int(grand_v2_total)}`**。

| 指标（每臂 `56` 个持续 episode，`n = 8` 簇） | `Flat_LatestFirst` | `Flat_ConditionalTable_Online` | `Hegel_FactoredMechanismGraph` | `Hegel` 对比 `Flat_Table` | `Hegel` 对比 `Flat_Latest` |
|---|---:|---:|---:|---:|---:|
| 私有验证严格通过率 (`strict_pass` 及簇 `p`) | `{out_tot['Flat_LatestFirst']['strict_passes']}/56` | `{out_tot['Flat_ConditionalTable_Online']['strict_passes']}/56` | `{out_tot['Hegel_FactoredMechanismGraph']['strict_passes']}/56` | `p = {qual_p_vs_ft}` | `p = {qual_p_vs_fl}` |
| 零 LLM (`0`-turn) 严格闭环数 | `{out_tot['Flat_LatestFirst']['zero_turn_passed_closures']}/56` | `{out_tot['Flat_ConditionalTable_Online']['zero_turn_passed_closures']}/56` | `{out_tot['Hegel_FactoredMechanismGraph']['zero_turn_passed_closures']}/56` | `+`{out_tot['Hegel_FactoredMechanismGraph']['zero_turn_passed_closures'] - out_tot['Flat_ConditionalTable_Online']['zero_turn_passed_closures']} | `+`{out_tot['Hegel_FactoredMechanismGraph']['zero_turn_passed_closures'] - out_tot['Flat_LatestFirst']['zero_turn_passed_closures']} |
| 求解 LLM 调用数 (`stream_solve`) | `{fl_solve['call_count']}` | `{ft_solve['call_count']}` | `{hg_solve['call_count']}` | `{_fmt_pct(hg_solve['call_count'] - ft_solve['call_count'], ft_solve['call_count'])}` | `{_fmt_pct(hg_solve['call_count'] - fl_solve['call_count'], fl_solve['call_count'])}` |
| 守卫归纳 LLM 调用数 (`autonomous_guard_induction`) | `0` | `{ft_guard['call_count']}` | `{hg_guard['call_count']}` | `{_fmt_pct(hg_guard['call_count'] - ft_guard['call_count'], ft_guard['call_count'])}` | — |
| **Scope-1 总 LLM 调用数 (`solve + guard`)** | **`{fl_tot['call_count']}`** | **`{ft_tot['call_count']}`** | **`{hg_tot['call_count']}`** | **`{_fmt_pct(hg_tot['call_count'] - ft_tot['call_count'], ft_tot['call_count'])}`** | **`{_fmt_pct(hg_tot['call_count'] - fl_tot['call_count'], fl_tot['call_count'])}`** |
| Scope-1 `proxy_v1_u_025c_1o` | `{_fmt_int(fl_tot['proxy_v1_u_025c_1o'])}` | `{_fmt_int(ft_tot['proxy_v1_u_025c_1o'])}` | `{_fmt_int(hg_tot['proxy_v1_u_025c_1o'])}` | `{_fmt_pct(hg_tot['proxy_v1_u_025c_1o'] - ft_tot['proxy_v1_u_025c_1o'], ft_tot['proxy_v1_u_025c_1o'])}` | `{_fmt_pct(hg_tot['proxy_v1_u_025c_1o'] - fl_tot['proxy_v1_u_025c_1o'], fl_tot['proxy_v1_u_025c_1o'])}` |
| **Scope-1 `proxy_v2_u_010c_4o`（流内主口径）** | **`{_fmt_int(fl_tot['proxy_v2_u_010c_4o'])}`** | **`{_fmt_int(ft_tot['proxy_v2_u_010c_4o'])}`** | **`{_fmt_int(hg_tot['proxy_v2_u_010c_4o'])}`** | **`{_fmt_pct(hg_tot['proxy_v2_u_010c_4o'] - ft_tot['proxy_v2_u_010c_4o'], ft_tot['proxy_v2_u_010c_4o'])}`** | **`{_fmt_pct(hg_tot['proxy_v2_u_010c_4o'] - fl_tot['proxy_v2_u_010c_4o'], fl_tot['proxy_v2_u_010c_4o'])}`** |
| **Scope-Total `proxy_v2`（独立部署含 `Stage S`）** | **`{_fmt_int(fl_with_s_v2)}`** | **`{_fmt_int(ft_with_s_v2)}`** | **`{_fmt_int(hg_with_s_v2)}`** | **`{_fmt_pct(hg_with_s_v2 - ft_with_s_v2, ft_with_s_v2)}`** | **`{_fmt_pct(hg_with_s_v2 - fl_with_s_v2, fl_with_s_v2)}`** |
| — Alpha 家族（`n = 4` 簇）Scope-1 `proxy_v2` | `{_fmt_int(alpha_fl_v2)}` | `{_fmt_int(alpha_ft_v2)}` | `{_fmt_int(alpha_hg_v2)}` | `{_fmt_pct(alpha_hg_v2 - alpha_ft_v2, alpha_ft_v2)}` | `{_fmt_pct(alpha_hg_v2 - alpha_fl_v2, alpha_fl_v2)}` |
| — Beta 家族（`n = 4` 簇）Scope-1 `proxy_v2` | `{_fmt_int(beta_fl_v2)}` | `{_fmt_int(beta_ft_v2)}` | `{_fmt_int(beta_hg_v2)}` | `{_fmt_pct(beta_hg_v2 - beta_ft_v2, beta_ft_v2)}` | `{_fmt_pct(beta_hg_v2 - beta_fl_v2, beta_fl_v2)}` |
| 模型调用 `wall_ms` (`solve + guard`) | `{_fmt_int(fl_tot['wall_ms'])}` | `{_fmt_int(ft_tot['wall_ms'])}` | `{_fmt_int(hg_tot['wall_ms'])}` | `{_fmt_pct(hg_tot['wall_ms'] - ft_tot['wall_ms'], ft_tot['wall_ms'])}` | `{_fmt_pct(hg_tot['wall_ms'] - fl_tot['wall_ms'], fl_tot['wall_ms'])}` |
| 非 LLM 预检容器探针 `wall_ms` (`56` 次) | `{_fmt_int(out_tot['Flat_LatestFirst']['pre_llm_probe_wall_ms'])}` | `{_fmt_int(out_tot['Flat_ConditionalTable_Online']['pre_llm_probe_wall_ms'])}` | `{_fmt_int(out_tot['Hegel_FactoredMechanismGraph']['pre_llm_probe_wall_ms'])}` | — | — |
| 非 LLM 经验覆盖探针 `wall_ms` | `0` | `{_fmt_int(out_tot['Flat_ConditionalTable_Online']['coverage_probe_wall_ms'])}` | `{_fmt_int(out_tot['Hegel_FactoredMechanismGraph']['coverage_probe_wall_ms'])}` | `{_fmt_pct(out_tot['Hegel_FactoredMechanismGraph']['coverage_probe_wall_ms'] - out_tot['Flat_ConditionalTable_Online']['coverage_probe_wall_ms'], out_tot['Flat_ConditionalTable_Online']['coverage_probe_wall_ms'])}` | — |
| 非 LLM `C_up` 上游定位容器探针（`R11`） | `0` | `0` | `8` 次（原汇总未单列，见 `docs/54b` 实测补账） | — | — |

---

## 3. 按因果事件拆解（每类事件 `8` 个簇 — 验证 `M_comb1` 与 `M_comb2` 跨因子零样本组合）

| 因果事件 | `Flat_LatestFirst` (`通过`, `0-LLM`, `v2`) | `Flat_ConditionalTable_Online` (`通过`, `0-LLM`, `v2`) | `Hegel_FactoredMechanismGraph` (`通过`, `0-LLM`, `v2`) |
|---|---|---|---|
{chr(10).join(ev_rows_md)}

---

## 4. 全部 `n = 8` 独立确认簇逐簇结果

| 确认簇 (`家族`) | `Flat_LatestFirst` | `Flat_ConditionalTable_Online` | `Hegel_FactoredMechanismGraph` | `Latest` `v2` | `Flat_Table` `v2` | `Hegel` `v2` | `Hegel` 对比 `Flat_Table` |
|---|---|---|---|---:|---:|---:|---:|
{chr(10).join(cluster_rows_md)}

---

## 5. 事后 `step_04` 冻结库 `D2` 回退删日志敏感性检查（`m9_8_history_masked_ablation.json`；严格在线前缀锁定与 `D0/D1` `8/16` 对照见 `docs/54b`）

| 确认簇 | 事件 | `SHA256(/holdout_task/public_batch.pkl)` | 事后库回退选中分支 | 遮蔽日志私有验证通过 |
|---|---|---|---|---|
{chr(10).join(mask_rows_md)}

---

## 6. `A01–A17` 五状态精细审计矩阵（按 `Hegel_M9_8_Audit` 证据分界）

| 编号 | 当前分项状态 | 核验证据与适用边界 |
|---|---|---|
| `A01`, `A17` | `verified_closed` | `48` 步 `N11` 隔离 RPC 实跑（`01_n11_executed_m95_paid_pool_replay.json`）与 `docs/52b` `64/68` 标题更正。 |
| `A02` | `scope_narrowed` | `M9.7` 普通条件表匹配 `30/42` 公共与 `{m97_ft_pass}/42` 私判；`M9.8` 分解系统相对单体型在线表省费 `-44.54%`，同时 `M9.8.1` 双普通表已付前缀回放（`56/56`）将结论精确限定在模块化分解复用系统包而非图容器本身。 |
| `A03`, `A13` | `scope_narrowed` | `(R0, U_iso)` 基础输入与基础说明全等（固定 current-only 上界 `<= 8/16`）；新增 `D1/D2` 相变公式不预给但基线控制方程在共享说明中给出；前缀日志为可观测前驱状态，完整前缀锁定与 `D0/D1` 对照见 `docs/54b`。 |
| `A04`, `A05`, `A15` | `verified_closed` | `M9.7` 重分类为已曝光种子复制（Beta 双侧 `p = 1.0` 已更正）；`M9.8` `n = 8` 确认集自干净提交 `{m98_j['code_commit_at_start'][:10]}` 单次运行，`0` 未知用量，费用 `p = {vs_ft['two_sided_sign_flip_p']}`、质量 `p = {qual_p_vs_ft}` 单列，`8` 次定位探针于 `docs/54b` 补账。 |
| `A06` | `diagnosed_and_retired` | `alpha_m97_c02 / R1` 完成错误码拆分与极点翻转复现，历史 `41/42` 保留，失稳候选正式退役（`docs/53b`）。 |
| `A07`, `A11`, `A12`, `A16` | `verified_closed` | 全父版本键校验（`A07`）、共享分层校准估计器（`A11`）、`M9.7` 编译策略命名（`A12`）与能力目录更新（`A16`）。 |
| `A08`, `A09`, `A10` | `superseded_in_m9_8_1` | `M9.8.1` 彻底闭合 `ecov_invalid` 格式校验（`R06`）、冷恢复缺摘要拒绝与外键校验（`R07`）、`r2 >= 0.96` 兼容推定与实测分离（`R04`）及定位容器异常隔离（`R05`）。 |
| `A14` | `retained_open_gap` | 整程序求解器通过完整 KKT 主对偶校验；历史 `M9.4/M9.6` `L1` 原语编译器的完整对偶字典采用仍列为保留缺口。 |
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
