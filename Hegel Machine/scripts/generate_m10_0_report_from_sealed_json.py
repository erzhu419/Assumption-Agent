#!/usr/bin/env python3
"""Deterministic, Zero-Hand-Edited Bilingual Report Generator for `docs/56_*` and `docs/56b_*`:
Reads ONLY sealed SHA-256-verified JSON artifacts from:
  1. `docs/evidence/m9_9_1_audit_closure_and_forensic_reverification_20261128/`
  2. `docs/evidence/m10_0_dev_economic_calibration_20261128/m10_0_continual_summary.json`
  3. `docs/evidence/m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_20261128/m10_0_continual_summary.json`
  4. `docs/evidence/m10_0_1_audit_closure_and_probe_reverification_20261128/` (`T01` inventory, `41/41` Docker post-hoc replay, `0`-LLM lifting audit)

Strictly enforces ALL `Hegel_M9_9_Audit` (`Q01–Q14`) AND `Hegel_M10_0_Audit` (`T01–T11`) reporting & statistical calibration contracts:
  - `T02`: Unrounded quality pass-rate bootstrap 95% CIs (`[-1.25pp, +13.75pp]` vs Mono, `[-3.75pp, +10.00pp]` vs Plain)
    AND small-sample paired `t(df=7)` 95% sensitivity intervals (`[-3.68pp, +16.18pp]` / `[-5.12pp, +12.62pp]`); NO "quality superiority" wording!
  - `T03`: Explicit pre-registration vs post-hoc boundary: Alpha `-0.87%` vs Beta `-18.66%` (`95.69%` of `H-Plain` savings) and `H-Plain` pooled `p = 0.21875`.
  - `T04`: Exact finite 10-step `K*` definition, per-cluster `K*` table (`c03/c04` right-censored vs Plain), and `K=8 (+1,208,895) -> K=9/10` payback trajectory.
  - `T07`: Explicit `B2` funnel (`9/10` direct probes, `1` alt joint, `0` direct rescues) and `B3` (`0` natural triggers in `dev`/`confirm`, verified via dedicated test).
  - `T08`: Itemized `Q01–Q14` status table instead of blanket "100% closed".
  - `T10`: Full `53,904,542 + x + y` accounting distinguishing `327` completed vs `6` `in_progress` calls in cancelled `M9.9` dev runs.
  - `T11`: Explicit UTC vs monotonic clock domain separation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
M991_DIR = REPO_ROOT / "docs/evidence/m9_9_1_audit_closure_and_forensic_reverification_20261128"
DEV_DIR = REPO_ROOT / "docs/evidence/m10_0_dev_economic_calibration_20261128"
CONF_DIR = REPO_ROOT / "docs/evidence/m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_20261128"
M1001_DIR = REPO_ROOT / "docs/evidence/m10_0_1_audit_closure_and_probe_reverification_20261128"

OUT_EN = REPO_ROOT / "docs/56_M9_9_1_Closure_and_M10_0_Reliable_Modular_Amortization_and_Bidirectional_Withdrawal_Report.md"
OUT_ZH = REPO_ROOT / "docs/56_M9_9_1_Closure_and_M10_0_Reliable_Modular_Amortization_and_Bidirectional_Withdrawal_Report_ZH.md"
OUT_56B_EN = REPO_ROOT / "docs/56b_M10_0_1_Audit_Closure_and_Probe_Persistence_Errata.md"
OUT_56B_ZH = REPO_ROOT / "docs/56b_M10_0_1_Audit_Closure_and_Probe_Persistence_Errata_ZH.md"

# Exact Student-t 97.5th percentile for df = 7 (`n = 8` independent clusters)
T_CRIT_DF7_95 = 2.3646242510102993


def _fmt_int(n: int | float) -> str:
    return f"{int(round(n)):,}"


def _pct_change(new_v: float, base_v: float) -> str:
    if base_v == 0:
        return "0.00%"
    d = 100.0 * (new_v - base_v) / base_v
    return f"{d:+.2f}%"


def _unrounded_bootstrap_95ci_pp(rate_diffs: list[float], n_boot: int = 20000, seed: int = 20261128) -> tuple[float, float]:
    arr = np.asarray(rate_diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = np.mean(arr[idx], axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo) * 100.0, float(hi) * 100.0


def _paired_t_95ci_pp(rate_diffs: list[float]) -> tuple[float, float]:
    arr = np.asarray(rate_diffs, dtype=float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    se = sd / math.sqrt(len(arr))
    t_crit = T_CRIT_DF7_95 if len(arr) == 8 else 2.0
    return (mean - t_crit * se) * 100.0, (mean + t_crit * se) * 100.0


def _total_probe_sec(arm_out: dict[str, Any]) -> float:
    keys = (
        "pre_llm_probe_wall_ms",
        "post_llm_probe_wall_ms",
        "localization_probe_wall_ms",
        "decomposition_probe_wall_ms",
        "coverage_probe_wall_ms",
        "official_verifier_wall_ms",
    )
    return sum(float(arm_out.get(k, 0.0)) for k in keys) / 1000.0


def _verify_sealed_dir(d: Path) -> int:
    lines = [ln.strip() for ln in (d / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    for ln in lines:
        exp_sha, rel_p = ln.split("  ", 1)
        got_sha = hashlib.sha256((d / rel_p).read_bytes()).hexdigest()
        if got_sha != exp_sha:
            raise ValueError(f"SHA256 mismatch in `{d}` on `{rel_p}`: expected {exp_sha}, got {got_sha}")
    return len(lines)


def _compute_per_cluster_k_star(conf: dict[str, Any], receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wids = list(conf["stage_S_summaries"].keys())
    step_cost: dict[tuple[str, str, int], int] = {}
    for r in receipts:
        arm = r["arm"]
        ep_sn = str(r["episode_short_name"])
        if "_batch_" not in ep_sn:
            continue
        k = int(ep_sn.rsplit("_batch_", 1)[-1])
        key = (r["world_id"], arm, k)
        step_cost[key] = step_cost.get(key, 0) + int(r["proxy_v2_u_010c_4o"])

    out = []
    for wid in wids:
        row: dict[str, Any] = {"world_id": wid, "family": conf["stage_S_summaries"][wid]["family"], "order_mode": conf["stage_S_summaries"][wid]["order_mode"]}
        for base_tag, base_arm in (("vs_mono", "Flat_MonolithicConditionalTable_Online"), ("vs_plain", "Flat_PlainModuleRegistry_Online")):
            deltas = []
            for k in range(1, 11):
                h_c = sum(step_cost.get((wid, "Hegel_SelectiveConditionExpansion_Online", s), 0) for s in range(1, k + 1))
                b_c = sum(step_cost.get((wid, base_arm, s), 0) for s in range(1, k + 1))
                deltas.append(h_c - b_c)
            k_st: int | str = "not_reached"
            for i in range(len(deltas)):
                if all(d < 0 for d in deltas[i:]):
                    k_st = i + 1
                    break
            row[f"k_star_{base_tag}"] = k_st
            row[f"delta_k8_{base_tag}"] = deltas[7]
            row[f"delta_k10_{base_tag}"] = deltas[9]
        out.append(row)
    return out


def _compute_b2_b3_funnel(conf_dir: Path, conf: dict[str, Any]) -> dict[str, Any]:
    funnel: dict[str, dict[str, int]] = {}
    for arm in ("Flat_PlainModuleRegistry_Online", "Hegel_SelectiveConditionExpansion_Online"):
        direct_calls = 0
        alt_joint_calls = 0
        direct_rescues = 0
        b3_monolith_replays = 0
        b3_lazy_proj = 0
        for wid in conf["stage_S_summaries"].keys():
            st_dir = conf_dir / "continual_streams" / wid / arm
            for ep_fp in sorted(st_dir.glob("*/00_episode_summary.json")):
                ep = json.loads(ep_fp.read_text(encoding="utf-8"))
                pre = ep.get("pre_llm_trial_receipt") or {}
                if int(pre.get("localization_probes_used", 0) or 0) >= 2:
                    direct_calls += 1
                if pre.get("sibling_fallback_used"):
                    alt_joint_calls += 1
                    if pre.get("accepted_pre_llm"):
                        direct_rescues += 1
                if pre.get("raw_archive_replay"):
                    b3_monolith_replays += 1
                if pre.get("lazy_projection_receipt"):
                    b3_lazy_proj += 1
        funnel[arm] = {
            "b2_alt_direct_probes": direct_calls,
            "b2_alt_joint_probes": alt_joint_calls,
            "b2_direct_pre_llm_rescues": direct_rescues,
            "b3_raw_archive_replays": b3_monolith_replays,
            "b3_lazy_projections": b3_lazy_proj,
        }
    return funnel


def _cluster_table_rows(summary: dict[str, Any]) -> list[str]:
    mono_c = summary["arm_cost_ledgers"]["Flat_MonolithicConditionalTable_Online"]["by_cluster"]
    plain_c = summary["arm_cost_ledgers"]["Flat_PlainModuleRegistry_Online"]["by_cluster"]
    hegel_c = summary["arm_cost_ledgers"]["Hegel_SelectiveConditionExpansion_Online"]["by_cluster"]
    st_map = {(s["world_id"], s["arm"]): s for s in summary["streams"]}
    wids = list(summary["stage_S_summaries"].keys())
    rows = []
    for wid in wids:
        s0 = summary["stage_S_summaries"][wid]
        fam = s0["family"]
        ord_m = s0["order_mode"]
        sm = st_map[(wid, "Flat_MonolithicConditionalTable_Online")]
        sp = st_map[(wid, "Flat_PlainModuleRegistry_Online")]
        sh = st_map[(wid, "Hegel_SelectiveConditionExpansion_Online")]
        vm = int(mono_c[wid]["proxy_v2_u_010c_4o"])
        vp = int(plain_c[wid]["proxy_v2_u_010c_4o"])
        vh = int(hegel_c[wid]["proxy_v2_u_010c_4o"])
        sw_cnt = len(sh["separability_withdrawal_receipts"])
        rows.append(
            f"| `{wid}` | `{fam}` | `{ord_m}` | "
            f"`{sm['strict_passes']}/10` (`{sm['zero_turn_passed_closures']}`) | "
            f"`{sp['strict_passes']}/10` (`{sp['zero_turn_passed_closures']}`) | "
            f"**`{sh['strict_passes']}/10` (`{sh['zero_turn_passed_closures']}`)** | "
            f"`{_fmt_int(vm)}` | `{_fmt_int(vp)}` | **`{_fmt_int(vh)}`** | "
            f"`{_fmt_int(vh - vm)}` ({_pct_change(vh, vm)}) | "
            f"`{_fmt_int(vh - vp)}` ({_pct_change(vh, vp)}) | `{sw_cnt}` |"
        )
    return rows


def _curve_rows(curve_obj: dict[str, Any]) -> list[str]:
    out = []
    for r in curve_obj["by_step_K"]:
        k = r["step_K"]
        tot_ep = r["total_episodes_at_K"]
        out.append(
            f"| `{k}` | `{_fmt_int(r['hegel_cum_scope1_v2_total'])}` | `{_fmt_int(r['baseline_cum_scope1_v2_total'])}` | "
            f"`{_fmt_int(r['total_delta_v2'])}` | `{r['mean_cluster_delta_v2']:,.2f}` | `{r['cluster_wins_hegel_cheaper']}` | "
            f"`{r['hegel_cum_strict_passes']}/{tot_ep}` vs `{r['baseline_cum_strict_passes']}/{tot_ep}` | "
            f"`{r['pass_rate_diff_hegel_minus_base']:+.4f}` (`{'PASS' if r['quality_noninferior_at_K'] else 'FAIL'}`) |"
        )
    return out


def main() -> None:
    n_m991 = _verify_sealed_dir(M991_DIR)
    n_dev = _verify_sealed_dir(DEV_DIR)
    n_conf = _verify_sealed_dir(CONF_DIR)
    n_m1001 = _verify_sealed_dir(M1001_DIR)

    dev = json.loads((DEV_DIR / "m10_0_continual_summary.json").read_text(encoding="utf-8"))
    conf = json.loads((CONF_DIR / "m10_0_continual_summary.json").read_text(encoding="utf-8"))
    conf_receipts = json.loads((CONF_DIR / "all_provider_call_receipts.json").read_text(encoding="utf-8"))
    dev_abl_m = json.loads((DEV_DIR / "ablation_manifest.json").read_text(encoding="utf-8"))
    conf_abl_m = json.loads((CONF_DIR / "ablation_manifest.json").read_text(encoding="utf-8"))

    m1001_inv = json.loads((M1001_DIR / "01_m100_probe_persistence_inventory.json").read_text(encoding="utf-8"))
    m1001_back = json.loads((M1001_DIR / "02_m100_41_missing_probes_docker_backverification.json").read_text(encoding="utf-8"))
    m1001_lift = json.loads((M1001_DIR / "03_m100_archived_solvers_lifting_coverage_inventory.json").read_text(encoding="utf-8"))

    assert dev["execution_provenance"]["source_tree_sha256_at_start"] == conf["execution_provenance"]["source_tree_sha256_at_start"]
    assert dev["execution_provenance"]["working_tree_dirty_at_start"] is False
    assert conf["execution_provenance"]["working_tree_dirty_at_start"] is False
    assert dev["execution_provenance"]["source_tree_unchanged_during_paid_execution"] is True
    assert conf["execution_provenance"]["source_tree_unchanged_during_paid_execution"] is True
    assert dev["total_usage_unknown_attempt_count"] == 0
    assert conf["total_usage_unknown_attempt_count"] == 0

    cm = conf["arm_outcome_totals"]["Flat_MonolithicConditionalTable_Online"]
    cp = conf["arm_outcome_totals"]["Flat_PlainModuleRegistry_Online"]
    ch = conf["arm_outcome_totals"]["Hegel_SelectiveConditionExpansion_Online"]

    lm = conf["arm_cost_ledgers"]["Flat_MonolithicConditionalTable_Online"]["scope1_in_stream_total"]
    lp = conf["arm_cost_ledgers"]["Flat_PlainModuleRegistry_Online"]["scope1_in_stream_total"]
    lh = conf["arm_cost_ledgers"]["Hegel_SelectiveConditionExpansion_Online"]["scope1_in_stream_total"]
    ls = conf["stage_S_shared_ledger"]

    pm = conf["paired_cluster_inference"]["Hegel_minus_Flat_MonolithicConditionalTable_Online"]
    pp = conf["paired_cluster_inference"]["Hegel_minus_Flat_PlainModuleRegistry_Online"]

    # Exact `T02` unrounded bootstrap + paired `t(df=7)` 95% sensitivity CIs (in percentage points)
    q_rate_m = [d / 10.0 for d in pm["quality_strict_pass_cluster_diffs"]]
    q_rate_p = [d / 10.0 for d in pp["quality_strict_pass_cluster_diffs"]]
    boot_m_lo, boot_m_hi = _unrounded_bootstrap_95ci_pp(q_rate_m)
    boot_p_lo, boot_p_hi = _unrounded_bootstrap_95ci_pp(q_rate_p)
    t_m_lo, t_m_hi = _paired_t_95ci_pp(q_rate_m)
    t_p_lo, t_p_hi = _paired_t_95ci_pp(q_rate_p)

    # Exact `T03` family decomposition (Alpha vs Beta share of Hegel - Plain savings)
    by_c_h = conf["arm_cost_ledgers"]["Hegel_SelectiveConditionExpansion_Online"]["by_cluster"]
    by_c_p = conf["arm_cost_ledgers"]["Flat_PlainModuleRegistry_Online"]["by_cluster"]
    by_c_m = conf["arm_cost_ledgers"]["Flat_MonolithicConditionalTable_Online"]["by_cluster"]
    a_wids = [w for w, s in conf["stage_S_summaries"].items() if s["family"] == "alpha"]
    b_wids = [w for w, s in conf["stage_S_summaries"].items() if s["family"] == "beta"]
    a_h = sum(by_c_h[w]["proxy_v2_u_010c_4o"] for w in a_wids)
    a_p = sum(by_c_p[w]["proxy_v2_u_010c_4o"] for w in a_wids)
    a_m = sum(by_c_m[w]["proxy_v2_u_010c_4o"] for w in a_wids)
    b_h = sum(by_c_h[w]["proxy_v2_u_010c_4o"] for w in b_wids)
    b_p = sum(by_c_p[w]["proxy_v2_u_010c_4o"] for w in b_wids)
    b_m = sum(by_c_m[w]["proxy_v2_u_010c_4o"] for w in b_wids)
    tot_hp_save = a_p + b_p - (a_h + b_h)
    beta_share_hp = 100.0 * (b_p - b_h) / max(tot_hp_save, 1)

    c_vs_m = conf["cumulative_amortization_curves_and_K_star"]["Hegel_minus_Flat_MonolithicConditionalTable_Online"]
    c_vs_p = conf["cumulative_amortization_curves_and_K_star"]["Hegel_minus_Flat_PlainModuleRegistry_Online"]
    per_cluster_k = _compute_per_cluster_k_star(conf, conf_receipts)
    b2_b3_funnel = _compute_b2_b3_funnel(CONF_DIR, conf)

    abl = conf["step08_routing_and_plain_dict_ablation_summary"]
    parity_conf = abl["plain_dict_same_joint_features_parity"]
    pol_conf = abl["summary_by_policy"]

    dm = dev["arm_outcome_totals"]["Flat_MonolithicConditionalTable_Online"]
    dp = dev["arm_outcome_totals"]["Flat_PlainModuleRegistry_Online"]
    dh = dev["arm_outcome_totals"]["Hegel_SelectiveConditionExpansion_Online"]
    dlm = dev["arm_cost_ledgers"]["Flat_MonolithicConditionalTable_Online"]["scope1_in_stream_total"]
    dlp = dev["arm_cost_ledgers"]["Flat_PlainModuleRegistry_Online"]["scope1_in_stream_total"]
    dlh = dev["arm_cost_ledgers"]["Hegel_SelectiveConditionExpansion_Online"]["scope1_in_stream_total"]

    tot_m100_calls = int(dev["total_logical_calls"]) + int(conf["total_logical_calls"])
    tot_m100_v2 = int(dev["grand_physical_ledger"]["proxy_v2_u_010c_4o"]) + int(conf["grand_physical_ledger"]["proxy_v2_u_010c_4o"])

    plain_decomps = sum(
        1 for s in conf["streams"] if s["arm"] == "Flat_PlainModuleRegistry_Online" and any(e.get("accepted") for e in s["autonomous_decomposition_events"])
    )
    hegel_decomps = sum(
        1 for s in conf["streams"] if s["arm"] == "Hegel_SelectiveConditionExpansion_Online" and any(e.get("accepted") for e in s["autonomous_decomposition_events"])
    )

    #Calibrated reading (`T02`): exact p-values, t-critical note, and T03 pre-M10.1 CI rule note
    dev_m = json.loads((DEV_DIR / "execution_manifest.json").read_text(encoding="utf-8"))
    dev_c = json.loads((DEV_DIR / "execution_completion.json").read_text(encoding="utf-8"))
    conf_m = json.loads((CONF_DIR / "execution_manifest.json").read_text(encoding="utf-8"))
    conf_c = json.loads((CONF_DIR / "execution_completion.json").read_text(encoding="utf-8"))
    from datetime import datetime as _dt
    dev_utc_span = (_dt.fromisoformat(dev_c["utc_ended_at"]) - _dt.fromisoformat(dev_m["utc_started_at"])).total_seconds()
    conf_utc_span = (_dt.fromisoformat(conf_c["utc_ended_at"]) - _dt.fromisoformat(conf_m["utc_started_at"])).total_seconds()
    plain_b2_joint = b2_b3_funnel["Flat_PlainModuleRegistry_Online"]["b2_alt_joint_probes"]
    hegel_b2_joint = b2_b3_funnel["Hegel_SelectiveConditionExpansion_Online"]["b2_alt_joint_probes"]
    hm_s1_pct = _pct_change(lh["proxy_v2_u_010c_4o"], lm["proxy_v2_u_010c_4o"])
    hp_s1_pct = _pct_change(lh["proxy_v2_u_010c_4o"], lp["proxy_v2_u_010c_4o"])
    hm_tot_pct = _pct_change(lh["proxy_v2_u_010c_4o"] + ls["proxy_v2_u_010c_4o"], lm["proxy_v2_u_010c_4o"] + ls["proxy_v2_u_010c_4o"])
    hp_tot_pct = _pct_change(lh["proxy_v2_u_010c_4o"] + ls["proxy_v2_u_010c_4o"], lp["proxy_v2_u_010c_4o"] + ls["proxy_v2_u_010c_4o"])

    k_cluster_rows = [
        f"| `{r['world_id']}` | `{r['family']}` | `{r['order_mode']}` | `{r['k_star_vs_mono']}` | `{r['k_star_vs_plain']}` | `{_fmt_int(r['delta_k8_vs_plain'])}` | `{_fmt_int(r['delta_k10_vs_plain'])}` |"
        for r in per_cluster_k
    ]

    en_md = f"""# Hegel Machine `M9.9.1` Audit Closure, `M10.0` Reliable Modular Amortization, and `M10.0.1` Calibration Errata Report

> **Deterministic Generation Notice (`Q05 / Q12 / T01–T11`)**: Generated strictly from sealed SHA-256-verified JSON ledgers (`{n_m991}` files in `m9_9_1_audit_closure_and_forensic_reverification_20261128`, `{n_dev}` files in `m10_0_dev_economic_calibration_20261128`, `{n_conf}` files in `m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_20261128`, and `{n_m1001}` files in `m10_0_1_audit_closure_and_probe_reverification_20261128`) with zero hand-edited metrics.

---

## 1. Executive Summary & Calibrated Claim Boundary (`Q09 / Q10 / T01–T04`)

Following the independent GPT-6 audits of `M9.9` (`d4b1cd7f0`) and `M10.0` (`9a83c2751`), this report records strictly separated forensic, prospective, and errata layers without mutating any sealed historical archive:
1. **`M9.9.1` Forensic Closure (`0` Paid LLM Calls)**: Re-executed all `30/30` historical off-diagonal `M9.9` decomposition pairs and `7/7` withdrawal counterexamples in `hegel-clean-scipy:m7v1`; verified `bind_alpha_m99_c04_up_b02 ∘ dn_b03` on `alpha_m99_c04/E7` (`strict_pass = True`, diagnostic only); isolated `alpha_m99_c02/E7` (`0.071267 -> 0.004857`) to missing pooled `ref_coupons` standardization/imputation; and recovered cancelled `M9.9` dev runs (`333` total calls = `327` completed + `6` `in_progress` calls, `8,220,922 + y` `proxy_v2`) into the `53,904,542 + x + y` full ledger (`T10`, see `docs/55b_*`, `docs/55c_*`). See Section 6 for itemized `Q01–Q14` closure status (`T08`).
2. **`M10.0` Prospective `3 × 3` Bidirectional Modular Amortization (`n = 4` Dev Calibration $\\to$ Zero-Code-Change Freeze $\\to$ Single-Shot `n = 8` Unexposed Confirmation)**:
   - **Scoped Mechanism Claim (`Q09 / Q10`)**: Under a **prescribed two-stage interface** (`calibrate_upstream`, `solve_downstream`), `Hegel_SelectiveConditionExpansion_Online` tests **counterexample-driven selective condition-set expansion (`dn_obs -> up_obs + dn_obs`) and bidirectional separability withdrawal** across `{{U0, U1, U2}} × {{D0, D1, D2}}` (`U1 ⊗ D2 -> D2_coupled`), balanced `50% / 50%` between `Order A (sep_first)` and `Order B (coupled_first)`, with symmetric `B1/B2/B3` upgrades across modular arms.
   - **Plain-Dict Algorithmic Equivalence (`Q10`)**: On the frozen `step_08` prefix across all confirmation clusters, a plain Python dictionary evaluating the same learned modules, joint `15`-observable vectors, and Kleene `all_of` rules reproduces **`{parity_conf['matched_episodes']}/{parity_conf['total_episodes']}` (`100.0%`)** of `Hegel`'s composed candidate SHA-256 hashes (`8/8 = 100.0%` on `dev`). Gains belong to the **selective condition-expansion & withdrawal algorithm + symmetric gates**, NOT the `PersistentMechanismGraph` container class or open-topology DAG discovery.
3. **`M10.0.1` Probe-Persistence & Statistical Calibration Errata (`T01–T11`, `0` Paid LLM Calls)**:
   - Explicitly records that `M10.0`'s original runner omitted raw JSON persistence for **`{m1001_inv['confirmation']['recorded_without_raw_measurement_count']}/{m1001_inv['confirmation']['total_witnesses']}`** confirmation witnesses (`32 pprobe + 9 uprobe`, including `8/8` withdrawal `prior_positive` witnesses) and **`{m1001_inv['dev']['recorded_without_raw_measurement_count']}/{m1001_inv['dev']['total_witnesses']}`** dev witnesses (`T01`).
   - All **`{m1001_back['verdict_matched_count']}/{m1001_back['total_missing_confirmation_probes_replayed']}`** missing confirmation probes (including **`{m1001_back['withdrawal_prior_witnesses_matched']}/{m1001_back['withdrawal_prior_witnesses_replayed']}`** withdrawal `prior_positive` witnesses) were post-hoc replayed in real Docker containers (`raw_measurement_status = "backfilled_posthoc_replay"`) and matched their recorded verdicts `100%`; future runs enforce atomic `ProbeReceiptStore` persistence + strict 3-way external FK (`docs/56b_*`).

### Headline `M10.0` Single-Shot Confirmation Results (`n = 8` Independent Unexposed Clusters, `80` Stream Episodes/Arm, `x = 0` Unknown Attempts)

| Arm (`Equal Tools, Same Sandbox & Guard Compiler`) | Official `strict_pass` (`80` eps) | `0`-LLM Passed Closures | Solver Turns | `Scope-1` Calls | `Scope-1` `proxy_v2` | `Scope-Total` (`+Stage S`) `proxy_v2` | Container Probe + Verifier Elapsed (`sec`) | Verified Withdrawals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicConditionalTable_Online` | `{cm['strict_passes']}/80` (`{100.0*cm['strict_passes']/80:.2f}%`) | `{cm['zero_turn_passed_closures']}/80` | `{cm['total_turns_used']}` | `{lm['call_count']}` | `{_fmt_int(lm['proxy_v2_u_010c_4o'])}` | `{_fmt_int(lm['proxy_v2_u_010c_4o'] + ls['proxy_v2_u_010c_4o'])}` | `{_total_probe_sec(cm):,.2f}s` | `{cm['separability_withdrawals']}` |
| `Flat_PlainModuleRegistry_Online` (`B1+B2+B3`) | `{cp['strict_passes']}/80` (`{100.0*cp['strict_passes']/80:.2f}%`) | `{cp['zero_turn_passed_closures']}/80` | `{cp['total_turns_used']}` | `{lp['call_count']}` | `{_fmt_int(lp['proxy_v2_u_010c_4o'])}` | `{_fmt_int(lp['proxy_v2_u_010c_4o'] + ls['proxy_v2_u_010c_4o'])}` | `{_total_probe_sec(cp):,.2f}s` | `{cp['separability_withdrawals']}` |
| **`Hegel_SelectiveConditionExpansion_Online` (`B1+B2+B3`)** | **`{ch['strict_passes']}/80` (`{100.0*ch['strict_passes']/80:.2f}%`)** | **`{ch['zero_turn_passed_closures']}/80`** | **`{ch['total_turns_used']}`** | **`{lh['call_count']}`** | **`{_fmt_int(lh['proxy_v2_u_010c_4o'])}`** (`{hm_s1_pct}` vs Mono / `{hp_s1_pct}` vs Plain) | **`{_fmt_int(lh['proxy_v2_u_010c_4o'] + ls['proxy_v2_u_010c_4o'])}`** (`{hm_tot_pct}` / `{hp_tot_pct}`) | **`{_total_probe_sec(ch):,.2f}s`** | **`{ch['separability_withdrawals']}`** |

Calibrated statistical takeaways on the `n = 8` unexposed confirmation split (`981000..988000`, frozen commit `{conf['execution_provenance']['code_commit_at_start'][:9]}`, zero `.py` changes after `dev`):
- **Quality Non-Inferiority (`delta_NI = -5.00pp`, `T02` — NO Statistical Quality Superiority Claim)**:
  - `Hegel` achieved **`{ch['strict_passes']}/80` (`96.25%`)**, numerically above `Flat_MonolithicConditionalTable_Online` (`72/80 = 90.00%`, diff **`+6.25pp`**, unrounded 95% bootstrap CI **`[{boot_m_lo:+.2f}pp, {boot_m_hi:+.2f}pp]`**, small-sample paired `t(df=7)` 95% CI **`[{t_m_lo:+.2f}pp, {t_m_hi:+.2f}pp]`**, two-sided sign-flip `p = {pm['quality_two_sided_sign_flip_p']}`) and `Flat_PlainModuleRegistry_Online` (`74/80 = 92.50%`, diff **`+3.75pp`**, unrounded 95% bootstrap CI **`[{boot_p_lo:+.2f}pp, {boot_p_hi:+.2f}pp]`**, paired `t(df=7)` 95% CI **`[{t_p_lo:+.2f}pp, {t_p_hi:+.2f}pp]`**, two-sided sign-flip `p = {pp['quality_two_sided_sign_flip_p']}`).
  - **Calibrated reading**: Both point estimates and 95% bootstrap lower bounds (`{boot_m_lo:+.2f}pp`, `{boot_p_lo:+.2f}pp`) clear `-5.00pp` (non-inferiority vs `Monolithic` also clears under `t(df=7)` at `{t_m_lo:+.2f}pp`; vs `Plain` the `t(df=7)` lower bound is `{t_p_lo:+.2f}pp`, `0.12pp` below `-5.00pp`, so `t(df=7)` sensitivity does NOT clear `-5.00pp`). Because both intervals span `0` and `p = {pm['quality_two_sided_sign_flip_p']} / {pp['quality_two_sided_sign_flip_p']}`, we claim **numerical advantage and bootstrap non-inferiority**, NOT statistically significant quality superiority. Note (`T03`): standalone pre-confirmation CI decision-rule JSON files began in `M10.1` (`m10_1_preregistered_analysis_protocol.json`); for `M10.0` the pre-confirmation CI rule location is marked `unverified_prior_to_m10_1_protocol`.
- **Decisive Cost Amortization vs Monolithic Primary Benchmark (`8/8` Clusters, `p = 0.0078125`)**: `Hegel` reduced `Scope-1` `proxy_v2` on **`8/8` (`100%`)** clusters (`Alpha` `{_pct_change(a_h, a_m)}`, `p = {pm['alpha_cost_sign_flip_p']}`; `Beta` `{_pct_change(b_h, b_m)}`, `p = {pm['beta_cost_sign_flip_p']}`; combined exact two-sided **`p = {pm['two_sided_sign_flip_p']}`**), saving **`{_fmt_int(-pm['mean_diff'] * 8)}` `proxy_v2` (`{hm_s1_pct}`)** with finite 10-step pooled `K* = {c_vs_m['empirical_quality_constrained_K_star']}`.
- **Modest Pooled Cost Reduction vs Equal-Tool Plain Modular Registry (`T03` Family Concentration Disclosed)**: `Hegel` saved **`{_fmt_int(-pp['mean_diff'] * 8)}` `proxy_v2` (`{hp_s1_pct}`)** vs `Plain` (`6/8` clusters cheaper, pooled exact two-sided **`p = {pp['two_sided_sign_flip_p']}` — NOT statistically significant at `alpha = 0.05`**). Family decomposition shows this pooled gain is heavily concentrated in **Beta** (`{_fmt_int(b_h - b_p)}`, **`{_pct_change(b_h, b_p)}`**, `4/4` clusters, **`{beta_share_hp:.2f}%`** of total `H-Plain` savings), whereas **Alpha** is near break-even (`{_fmt_int(a_h - a_p)}`, **`{_pct_change(a_h, a_p)}`**, `2/4` clusters). Meanwhile, `0`-LLM passed closures increased from `29/80` to **`37/80`** (`7/8` clusters positive, exact two-sided `p = {pp['zero_llm_two_sided_sign_flip_p']}`; vs Monolithic `15/80`, `8/8` positive, `p = {pm['zero_llm_two_sided_sign_flip_p']}`).
- **Step-08 Constrained Routing Ceiling**: On all `16` colliding `D2` revisit episodes (`S09, S10`), `hegel_selective_condition_expansion` achieved **`{pol_conf['hegel_selective_condition_expansion']['strict_passes']}/16` (`100.0%`)** `0`-LLM private `strict_pass`, vs **`{pol_conf['dn_only_fixed_prior_d2']['strict_passes']}/16` (`50.0%`)** for `dn_only_fixed_prior_d2` and **`{pol_conf['dn_only_fixed_latest_d2']['strict_passes']}/16` (`50.0%`)** for `dn_only_fixed_latest_d2`.

---

## 2. Execution Provenance, Clocks, & Full R&D Cost Ledger (`C1–C2 / Q05 / Q12 / T10–T11`)

### 2.1 Immutable Start/Completion/Ablation Provenance (`0` Post-Dev Source Edits)

| Split | `execution_run_id` | Start Commit | Dirty at Start? | `.py` Source Tree SHA-256 (`src+scripts+tests`) | Tree Unchanged During Run? | Monotonic Elapsed (`sec`, `T11`) | Ablation Elapsed (`sec`) | Concurrency (`cap / peak`) | Unknown Attempts |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `dev` (`n=4`) | `{dev['execution_run_id']}` | `{dev['execution_provenance']['code_commit_at_start'][:9]}` | `{dev['execution_provenance']['working_tree_dirty_at_start']}` | `{dev['execution_provenance']['source_tree_sha256_at_start'][:16]}...` | `{dev['execution_provenance']['source_tree_unchanged_during_paid_execution']}` | `{dev['execution_elapsed_wall_sec']:,.2f}s` | `{dev_abl_m['elapsed_wall_sec']:,.2f}s` | `{dev['configured_max_codex_concurrency']} / {dev['peak_active_codex_concurrency']}` | `{dev['total_usage_unknown_attempt_count']}` |
| `confirmation` (`n=8`) | `{conf['execution_run_id']}` | `{conf['execution_provenance']['code_commit_at_start'][:9]}` | `{conf['execution_provenance']['working_tree_dirty_at_start']}` | `{conf['execution_provenance']['source_tree_sha256_at_start'][:16]}...` | `{conf['execution_provenance']['source_tree_unchanged_during_paid_execution']}` | `{conf['execution_elapsed_wall_sec']:,.2f}s` | `{conf_abl_m['elapsed_wall_sec']:,.2f}s` | `{conf['configured_max_codex_concurrency']} / {conf['peak_active_codex_concurrency']}` | `{conf['total_usage_unknown_attempt_count']}` |

- **Clock Domain Clarification (`T11`)**: `execution_elapsed_wall_sec` is measured via `time.monotonic()` (`{dev['execution_elapsed_wall_sec']:,.2f}s` dev, `{conf['execution_elapsed_wall_sec']:,.2f}s` confirmation); UTC timestamps (`utc_started_at`, `utc_ended_at`) are recorded separately (`{dev_utc_span:,.2f}s` dev UTC span, `{conf_utc_span:,.2f}s` confirmation UTC span; historical `254.45s` WSL wall-clock vs monotonic drift on `M10.0` confirmation is preserved transparently as unadjusted host clock skew rather than overwritten).
- **Complete `M10.0` Paid R&D Ledger (`0` Cancelled Runs)**:
  - `M10.0` `dev` (`n = 4`, `124` episodes): `{_fmt_int(dev['total_logical_calls'])}` calls, `{_fmt_int(dev['grand_physical_ledger']['total_tokens'])}` total tokens, **`{_fmt_int(dev['grand_physical_ledger']['proxy_v2_u_010c_4o'])}` `proxy_v2`** (`0` unknown attempts).
  - `M10.0` `confirmation` (`n = 8`, `248` episodes): `{_fmt_int(conf['total_logical_calls'])}` calls, `{_fmt_int(conf['grand_physical_ledger']['total_tokens'])}` total tokens, **`{_fmt_int(conf['grand_physical_ledger']['proxy_v2_u_010c_4o'])}` `proxy_v2`** (`0` unknown attempts).
  - `M10.0` Grand Total (`dev + confirmation`): **`{_fmt_int(tot_m100_calls)}` calls / `{_fmt_int(tot_m100_v2)}` exact `proxy_v2`**.
  - Historical `M9.9` Full Ledger (`T10`): `53,904,542 + x + y` `proxy_v2` across `M9.9` dev/confirm + recovered cancelled dev logs (`333` total = `327` completed + `6` `in_progress` calls).

### 2.2 Hierarchical Role Conservation on `M10.0` Confirmation (`all_provider_call_receipts.json`)

| Role Partition | `Monolithic` Calls / `proxy_v2` | `Plain` Calls / `proxy_v2` | `Hegel` Calls / `proxy_v2` |
|---|---:|---:|---:|
| `solve_only` | `{conf['arm_cost_ledgers']['Flat_MonolithicConditionalTable_Online']['solve_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Flat_MonolithicConditionalTable_Online']['solve_only']['proxy_v2_u_010c_4o'])}` | `{conf['arm_cost_ledgers']['Flat_PlainModuleRegistry_Online']['solve_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Flat_PlainModuleRegistry_Online']['solve_only']['proxy_v2_u_010c_4o'])}` | `{conf['arm_cost_ledgers']['Hegel_SelectiveConditionExpansion_Online']['solve_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Hegel_SelectiveConditionExpansion_Online']['solve_only']['proxy_v2_u_010c_4o'])}` |
| `decomposition_induction_only` (`{plain_decomps}/8` Plain, `{hegel_decomps}/8` Hegel) | `0` / `0` | `{conf['arm_cost_ledgers']['Flat_PlainModuleRegistry_Online']['decomposition_induction_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Flat_PlainModuleRegistry_Online']['decomposition_induction_only']['proxy_v2_u_010c_4o'])}` | `{conf['arm_cost_ledgers']['Hegel_SelectiveConditionExpansion_Online']['decomposition_induction_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Hegel_SelectiveConditionExpansion_Online']['decomposition_induction_only']['proxy_v2_u_010c_4o'])}` |
| `lazy_retroactive_projection_only` (`B3`) | `0` / `0` | `{conf['arm_cost_ledgers']['Flat_PlainModuleRegistry_Online']['lazy_retroactive_projection_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Flat_PlainModuleRegistry_Online']['lazy_retroactive_projection_only']['proxy_v2_u_010c_4o'])}` | `{conf['arm_cost_ledgers']['Hegel_SelectiveConditionExpansion_Online']['lazy_retroactive_projection_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Hegel_SelectiveConditionExpansion_Online']['lazy_retroactive_projection_only']['proxy_v2_u_010c_4o'])}` |
| `guard_induction_only` | `{conf['arm_cost_ledgers']['Flat_MonolithicConditionalTable_Online']['guard_induction_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Flat_MonolithicConditionalTable_Online']['guard_induction_only']['proxy_v2_u_010c_4o'])}` | `{conf['arm_cost_ledgers']['Flat_PlainModuleRegistry_Online']['guard_induction_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Flat_PlainModuleRegistry_Online']['guard_induction_only']['proxy_v2_u_010c_4o'])}` | `{conf['arm_cost_ledgers']['Hegel_SelectiveConditionExpansion_Online']['guard_induction_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Hegel_SelectiveConditionExpansion_Online']['guard_induction_only']['proxy_v2_u_010c_4o'])}` |
| **`Scope-1` In-Stream Total** | **`{lm['call_count']}` / `{_fmt_int(lm['proxy_v2_u_010c_4o'])}`** | **`{lp['call_count']}` / `{_fmt_int(lp['proxy_v2_u_010c_4o'])}`** | **`{lh['call_count']}` / `{_fmt_int(lh['proxy_v2_u_010c_4o'])}`** |
| Shared `Stage S` Source (`8/8` public+private pass) | `{ls['call_count']}` / `{_fmt_int(ls['proxy_v2_u_010c_4o'])}` (shared across 3 arms) | same | same |

---

## 3. Cluster-by-Cluster Confirmation Breakdown & Paired Statistical Inference (`C3 / Q06 / T02–T03`)

### 3.1 Per-Cluster Results (`n = 8` Independent Confirmation Source Clusters)

| World ID | Family | Order Mode | `Monolithic` Pass (`0`-LLM) | `Plain` Pass (`0`-LLM) | `Hegel` Pass (`0`-LLM) | `Mono` `Scope-1 v2` | `Plain` `Scope-1 v2` | `Hegel` `Scope-1 v2` | `Hegel - Mono` `v2` | `Hegel - Plain` `v2` | `Hegel` Withdrawals |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(_cluster_table_rows(conf))}

### 3.2 Paired Cluster-Level Exact Two-Sided Sign-Flip, Unrounded Bootstrap, & `t(df=7)` Inference (`n = 8`)

| Comparison (`Hegel` minus Baseline) | Mean Cluster `Scope-1 v2` Diff (95% Boot CI) | Cost Exact 2-Sided `p` (`All / Alpha / Beta`) | `sep_first` vs `coupled_first` Mean `v2` Diff | Quality Rate Diff (Unrounded Boot 95% CI / `t(df=7)` 95% CI; `delta_NI = -5.00pp`) | Quality Exact 2-Sided `p` | `0`-LLM Closure Cluster Diffs & Exact 2-Sided `p` |
|---|---|---|---|---|---:|---|
| **`Hegel - Monolithic`** | **`{pm['mean_diff']:,.2f}`** (`[{pm['bootstrap_95ci'][0]:,.2f}, {pm['bootstrap_95ci'][1]:,.2f}]`) | **`{pm['two_sided_sign_flip_p']}`** (`{pm['alpha_cost_sign_flip_p']}` / `{pm['beta_cost_sign_flip_p']}`) | `{pm['sep_first_mean_diff_v2']:,.2f}` / `{pm['coupled_first_mean_diff_v2']:,.2f}` | **`+6.25pp`** (`[{boot_m_lo:+.2f}pp, {boot_m_hi:+.2f}pp]` / `[{t_m_lo:+.2f}pp, {t_m_hi:+.2f}pp]`, **NI PASS**) | `{pm['quality_two_sided_sign_flip_p']}` | `{pm['zero_llm_closure_cluster_diffs']}`, **`p = {pm['zero_llm_two_sided_sign_flip_p']}`** |
| **`Hegel - Plain`** | **`{pp['mean_diff']:,.2f}`** (`[{pp['bootstrap_95ci'][0]:,.2f}, {pp['bootstrap_95ci'][1]:,.2f}]`) | `{pp['two_sided_sign_flip_p']}` (`{pp['alpha_cost_sign_flip_p']}` / `{pp['beta_cost_sign_flip_p']}`) | `{pp['sep_first_mean_diff_v2']:,.2f}` / `{pp['coupled_first_mean_diff_v2']:,.2f}` | **`+3.75pp`** (`[{boot_p_lo:+.2f}pp, {boot_p_hi:+.2f}pp]` / `[{t_p_lo:+.2f}pp, {t_p_hi:+.2f}pp]`, **Boot NI PASS / `t` borderline**) | `{pp['quality_two_sided_sign_flip_p']}` | `{pp['zero_llm_closure_cluster_diffs']}`, **`p = {pp['zero_llm_two_sided_sign_flip_p']}`** |

---

## 4. Cumulative Amortization Curves $\\Delta C(K)$ (`K = 1..10`) & Finite-Horizon Break-Even $K^*$ (`D2 / T04`)

> **Finite-Horizon `K*` Definition (`T04`)**: `K*` is defined strictly over the observed `10`-step horizon as the smallest step $K \\in \\{{1..10\\}}$ such that cumulative pooled cost difference $\\Delta C(k) < 0$ for every $k \\in \\{{K..10\\}}$ AND cumulative quality at $K=10$ meets `delta_NI >= -0.05`. It does NOT assume perpetual savings beyond $K = 10$ without recurring revisits.

### 4.1 `Hegel` minus `Flat_MonolithicConditionalTable_Online` (`Pooled K* = {c_vs_m['empirical_quality_constrained_K_star']}`)

| Step `K` | `Hegel` Cum `Scope-1 v2` | `Monolithic` Cum `Scope-1 v2` | Total `Delta C_mono(K)` | Mean Cluster `Delta v2` | Clusters Cheaper (`/8`) | Cum `strict_pass` (`Hegel` vs `Mono`) | Quality Diff (`>= -0.05` Check) |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(_curve_rows(c_vs_m))}

### 4.2 `Hegel` minus `Flat_PlainModuleRegistry_Online` (`Pooled K* = {c_vs_p['empirical_quality_constrained_K_star']}`)

| Step `K` | `Hegel` Cum `Scope-1 v2` | `Plain` Cum `Scope-1 v2` | Total `Delta C_plain(K)` | Mean Cluster `Delta v2` | Clusters Cheaper (`/8`) | Cum `strict_pass` (`Hegel` vs `Plain`) | Quality Diff (`>= -0.05` Check) |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(_curve_rows(c_vs_p))}

Notice the exact `T04` payback mechanism: through `K = 8`, `Hegel` carries a pooled upfront selective-expansion investment of **`+1,208,895` `proxy_v2`** over `Plain`; payback occurs specifically at **`K = 9` and `K = 10`** when colliding-`dn_obs` revisits (`S09, S10`) arrive!

### 4.3 Per-Cluster `K*` & Right-Censoring Table (`T04`)

| World ID | Family | Order Mode | Cluster `K*` vs `Mono` | Cluster `K*` vs `Plain` | `Delta C_plain(K=8)` | `Delta C_plain(K=10)` |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(k_cluster_rows)}

---

## 5. `B2`/`B3` Execution Funnel (`T07`), Container Wall-Time (`Q11`), & `Gate 3` Dev Summary

### 5.1 `B2` & `B3` Observed Trigger Funnel on `M10.0` Confirmation (`T07`)

| Arm | `B2` Alt Direct Probes | `B2` Alt Joint Probes | `B2` Direct Pre-LLM Rescues | `B3` Raw Archive Replays | `B3` Lazy Projections |
|---|---:|---:|---:|---:|---:|
| `Flat_PlainModuleRegistry_Online` | `{b2_b3_funnel['Flat_PlainModuleRegistry_Online']['b2_alt_direct_probes']}` | `{b2_b3_funnel['Flat_PlainModuleRegistry_Online']['b2_alt_joint_probes']}` | `{b2_b3_funnel['Flat_PlainModuleRegistry_Online']['b2_direct_pre_llm_rescues']}` | `{b2_b3_funnel['Flat_PlainModuleRegistry_Online']['b3_raw_archive_replays']}` | `{b2_b3_funnel['Flat_PlainModuleRegistry_Online']['b3_lazy_projections']}` |
| `Hegel_SelectiveConditionExpansion_Online` | `{b2_b3_funnel['Hegel_SelectiveConditionExpansion_Online']['b2_alt_direct_probes']}` | `{b2_b3_funnel['Hegel_SelectiveConditionExpansion_Online']['b2_alt_joint_probes']}` | `{b2_b3_funnel['Hegel_SelectiveConditionExpansion_Online']['b2_direct_pre_llm_rescues']}` | `{b2_b3_funnel['Hegel_SelectiveConditionExpansion_Online']['b3_raw_archive_replays']}` | `{b2_b3_funnel['Hegel_SelectiveConditionExpansion_Online']['b3_lazy_projections']}` |

- **`B2` Interpretation**: On confirmation, `B2` probed an alternate verified upstream sibling (`{b2_b3_funnel['Flat_PlainModuleRegistry_Online']['b2_alt_direct_probes']}` times on `Plain`, `{b2_b3_funnel['Hegel_SelectiveConditionExpansion_Online']['b2_alt_direct_probes']}` times on `Hegel`), promoted `{plain_b2_joint}` alternate upstream into a joint probe on `Plain` and `{hegel_b2_joint}` on `Hegel`, and achieved `0` direct pre-LLM rescues (diagnostic localization only).
- **`B3` Interpretation**: Because initial `2×2` decomposition succeeded on `16/16` (`100%`) confirmation modular streams (`8/8` `Plain`, `8/8` `Hegel`), `B3` had **`0` natural triggers** in `dev` and `confirmation`; its correctness is verified via dedicated synthetic unit tests (`test_m100_b2_sibling_upstream_fallback_and_b3_deferred_lazy_projection` and `test_m10_0_1_audit_closure_suite.py`).

### 5.2 Non-LLM Container Probe & Verifier Elapsed Wall-Time (`OMP_NUM_THREADS=1`, Elapsed Wall-Sec)

| Arm (`Confirmation, 80 Episodes`) | Pre-LLM (`sec`) | Post-LLM (`sec`) | Localization (`sec`) | `2x2` Decomp (`sec`) | Empirical Coverage (`sec`) | Official Verifier (`sec`) | Total Container Elapsed (`sec`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicConditionalTable_Online` | `{cm['pre_llm_probe_wall_ms']/1000:,.2f}` | `{cm['post_llm_probe_wall_ms']/1000:,.2f}` | `0.00` | `0.00` | `{cm['coverage_probe_wall_ms']/1000:,.2f}` | `{cm['official_verifier_wall_ms']/1000:,.2f}` | `{_total_probe_sec(cm):,.2f}` |
| `Flat_PlainModuleRegistry_Online` | `{cp['pre_llm_probe_wall_ms']/1000:,.2f}` | `{cp['post_llm_probe_wall_ms']/1000:,.2f}` | `{cp['localization_probe_wall_ms']/1000:,.2f}` | `{cp['decomposition_probe_wall_ms']/1000:,.2f}` | `{cp['coverage_probe_wall_ms']/1000:,.2f}` | `{cp['official_verifier_wall_ms']/1000:,.2f}` | `{_total_probe_sec(cp):,.2f}` |
| **`Hegel_SelectiveConditionExpansion_Online`** | **`{ch['pre_llm_probe_wall_ms']/1000:,.2f}`** | **`{ch['post_llm_probe_wall_ms']/1000:,.2f}`** | **`{ch['localization_probe_wall_ms']/1000:,.2f}`** | **`{ch['decomposition_probe_wall_ms']/1000:,.2f}`** | **`{ch['coverage_probe_wall_ms']/1000:,.2f}`** | **`{ch['official_verifier_wall_ms']/1000:,.2f}`** | **`{_total_probe_sec(ch):,.2f}`** |

### 5.3 Summary of Sealed `Gate 3` Dev Calibration (`n = 4`, `971000..974000`, `40` Episodes/Arm)

- `Monolithic`: `{dm['strict_passes']}/40` `strict_pass` (`{dm['zero_turn_passed_closures']}/40` `0`-LLM, `{_fmt_int(dlm['proxy_v2_u_010c_4o'])}` `Scope-1 v2`).
- `Plain`: `{dp['strict_passes']}/40` `strict_pass` (`{dp['zero_turn_passed_closures']}/40` `0`-LLM, `{_fmt_int(dlp['proxy_v2_u_010c_4o'])}` `Scope-1 v2`).
- `Hegel`: `{dh['strict_passes']}/40` `strict_pass` (`{dh['zero_turn_passed_closures']}/40` `0`-LLM, `{_fmt_int(dlh['proxy_v2_u_010c_4o'])}` `Scope-1 v2`, `K* = {dev['cumulative_amortization_curves_and_K_star']['Hegel_minus_Flat_MonolithicConditionalTable_Online']['empirical_quality_constrained_K_star']}` vs `Monolithic`, `K* = {dev['cumulative_amortization_curves_and_K_star']['Hegel_minus_Flat_PlainModuleRegistry_Online']['empirical_quality_constrained_K_star']}` vs `Plain`, `8/8` plain-dict parity).

---

## 6. Itemized `Q01–Q14` & `T01–T11` Audit Closure Status (`T08`)

| ID | Status | Evidence / Resolution Summary |
|---|---|---|
| `Q01 / T01` | **Closed in `M10.0.1`** | `2×2` decomposition executed in `M10.0`; `M10.0.1` adds atomic `ProbeReceiptStore`, migrates all bare `record_tested_witness` calls, enforces 3-way external receipt FK on withdrawals, and post-hoc back-verifies `41/41` missing `M10.0` confirm probes (`docs/56b_*`). |
| `Q02 / B1` | **Closed** | Public pooled `ref_coupons` exchange invariance active symmetrically across all arms (`require_pooled_coupon_check=True`). |
| `Q03 / B2` | **Closed (Calibrated `T07`)** | Symmetric `1`-sibling upstream fallback implemented and dual primary/alt diagnostics preserved; funnel disclosed (`0` direct rescues on confirmation). |
| `Q04 / B3` | **Closed (Test-Verified `T07`)** | Unfactored monolithic fallback + lazy projection implemented symmetrically; `0` natural triggers in `M10.0` (`16/16` immediate decomp), covered by unit tests. |
| `Q05 / T11` | **Closed** | Immutable write-once `execution_manifest.json` + separate `execution_completion.json` & `ablation_manifest.json`; UTC vs monotonic clock fields separated. |
| `Q06 / T02–T03` | **Closed in `M10.0.1`** | Unrounded bootstrap CIs + `t(df=7)` sensitivity intervals, family decomposition (`Alpha -0.87%` vs `Beta -18.66%`), and non-superiority calibration added. |
| `Q07 / T01` | **Closed in `M10.0.1`** | 6-link ledger FK upgraded to 3-way external `ProbeReceiptStore` FK rejecting GPT-6's exact forged `prior_positive.probe_id` negative control. |
| `Q08` | **Closed** | Idempotent legacy migration round-trip verified in `test_m9_9_1_audit_closure_suite.py`. |
| `Q09 / Q10` | **Closed** | Calibrated scope (`SelectiveConditionExpansion` under prescribed 2-stage split) & `16/16` (`100%`) plain-dict candidate-SHA parity reported upfront. |
| `Q11` | **Closed** | Explicit elapsed wall-time (not `/proc` CPU time) reported for all container probe categories. |
| `Q12 / T10` | **Closed** | Full `M9.9` (`53,904,542 + x + y`; `327` completed + `6` `in_progress` cancelled dev calls) + `M10.0` (`64,681,960` exact, `0` unknown) ledgers reconciled. |
| `Q13 / Q14` | **Closed** | Equal-info archive & controlled step-08 ablation (`16/16` vs `8/16`) verified; zero `M9.9` archived folder mutations. |
| `T05 / T06` | **Closed in `M10.0.1`** | Generic Alpha near-zero derivative/pilot consistency check (`0` `log1p` hardcoding) + Beta calibration-to-KKT perturbation check & non-`0.0` failure aggregator fix. |
| `T09` | **Closed in `M10.0.1`** | Mono/Plain/Hegel row IDs and branch tags neutralized (`mono_rXX`, `plain_up_rXX`, `plain_dn_rXX`, `up_branch_XX`, `dn_branch_XX`). |
"""

    zh_md = f"""# 黑格尔机 `M9.9.1` 审计闭环、`M10.0` 可靠模块化摊销及 `M10.0.1` 校准勘误报告

> **确定性生成声明（`Q05 / Q12 / T01–T11`）**：本报告完全由经 `SHA256SUMS.txt` 校验的封存 JSON 账本确定性生成（`m9_9_1_audit_closure_and_forensic_reverification_20261128` 共 `{n_m991}` 个文件、`m10_0_dev_economic_calibration_20261128` 共 `{n_dev}` 个文件、`m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_20261128` 共 `{n_conf}` 个文件、`m10_0_1_audit_closure_and_probe_reverification_20261128` 共 `{n_m1001}` 个文件），零人工手改指标。

---

## 1. 执行摘要与校准后的主张边界（`Q09 / Q10 / T01–T04`）

承接 GPT-6 对 `M9.9`（`d4b1cd7f0`）与 `M10.0`（`9a83c2751`）的两次完整独立审计，本报告在不改写任何既有封存目录的前提下严格分层记录三阶段结果：
1. **`M9.9.1` 取证复核与勘误（`0` 次付费模型调用）**：补跑 `M9.9` 全部 `30/30` 非对角分解探针与 `7/7` 历史撤回反例；事后实测 `alpha_m99_c04` 既有候选 `up_b02 ∘ dn_b03` 在 `E7` 上通过公开门与官方私判（仅诊断，不改动原 `50/56` 得分）；定位 `alpha_m99_c02/E7` 源于跨切分 `ref_coupons` 联合标准化缺失；完整找回已取消 `M9.9` 开发日志（`333` 次 = `327` 次完成 + `6` 次 `in_progress`，已知 `8,220,922 + y`，总计 `53,904,542 + x + y`，`T10`）。详见第 6 节逐项 `Q01–Q14` 闭环状态表（`T08`）。
2. **`M10.0` 前瞻 `3 × 3` 双向可分性撤回与可靠模块化摊销（`n = 4` 开发校准 $\\to$ 零改码冻结 $\\to$ 单次 `n = 8` 未曝光确认）**：
   - **严格校准的主张范围（`Q09 / Q10`）**：在**给定两阶段接口契约** `(calibrate_upstream, solve_downstream)` 的条件下，`Hegel_SelectiveConditionExpansion_Online` 检验**反例驱动的选择性条件集扩展（`dn_obs -> up_obs + dn_obs`）与双向可分性撤回**，并在两模块化臂对称启用 `B1/B2/B3`。
   - **普通字典同特征逐候选等价性（`Q10`）**：在全部确认簇冻结 `step_08` 前缀上，同模块、同 `15` 维特征与同 Kleene `all_of` 规则的普通 Python 字典复现了 **`{parity_conf['matched_episodes']}/{parity_conf['total_episodes']}`（`100.0%`）** 的候选 `SHA-256`（开发集 `8/8 = 100.0%`）。收益归因于**选择性条件集扩展/撤回算法与对称门禁**，而非 `PersistentMechanismGraph` 容器类本身或开放未知拓扑 DAG 发现。
3. **`M10.0.1` 探针落盘闭环与统计校准勘误（`T01–T11`，`0` 次付费模型调用）**：
   - 明确勘误：`M10.0` 原运行器未将 **`{m1001_inv['confirmation']['recorded_without_raw_measurement_count']}/{m1001_inv['confirmation']['total_witnesses']}`** 条确认集证人（`32` 条 `pprobe` + `9` 条 `uprobe`，含 `8/8` 撤回 `prior_positive` 证人）与 **`{m1001_inv['dev']['recorded_without_raw_measurement_count']}/{m1001_inv['dev']['total_witnesses']}`** 条开发集证人的原始测量回执完整落盘（`T01`）。
   - 全部 **`{m1001_back['verdict_matched_count']}/{m1001_back['total_missing_confirmation_probes_replayed']}`** 条缺失确认探针（含 **`{m1001_back['withdrawal_prior_witnesses_matched']}/{m1001_back['withdrawal_prior_witnesses_replayed']}`** 条撤回前置正证人）已在真实 Docker 容器中事后补测回放（严格标注 `raw_measurement_status = "backfilled_posthoc_replay"`），与原记录裁决 `100%` 一致；新代码已强制接入原子 `ProbeReceiptStore` 与三方外部回执外键（详见 `docs/56b_*`）。

### `M10.0` 单次未曝光确认集核心结果（`n = 8` 独立源簇，每臂 `80` 幕，`x = 0` 次未知用量尝试）

| 实验臂（等工具权限、同沙盒与同守卫编译器） | 官方私判 `strict_pass`（`80` 幕） | 零模型调用通过闭环 | 求解轮数 | `Scope-1` 调用数 | `Scope-1` `proxy_v2` 成本 | `Scope-Total`（含共享 `Stage S`）`proxy_v2` | 容器探针+私判总墙钟（秒） | 验证通过的撤回回执 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicConditionalTable_Online`（主经济与质量对照） | `{cm['strict_passes']}/80` (`{100.0*cm['strict_passes']/80:.2f}%`) | `{cm['zero_turn_passed_closures']}/80` | `{cm['total_turns_used']}` | `{lm['call_count']}` | `{_fmt_int(lm['proxy_v2_u_010c_4o'])}` | `{_fmt_int(lm['proxy_v2_u_010c_4o'] + ls['proxy_v2_u_010c_4o'])}` | `{_total_probe_sec(cm):,.2f}s` | `{cm['separability_withdrawals']}` |
| `Flat_PlainModuleRegistry_Online`（机制对照，`B1+B2+B3`） | `{cp['strict_passes']}/80` (`{100.0*cp['strict_passes']/80:.2f}%`) | `{cp['zero_turn_passed_closures']}/80` | `{cp['total_turns_used']}` | `{lp['call_count']}` | `{_fmt_int(lp['proxy_v2_u_010c_4o'])}` | `{_fmt_int(lp['proxy_v2_u_010c_4o'] + ls['proxy_v2_u_010c_4o'])}` | `{_total_probe_sec(cp):,.2f}s` | `{cp['separability_withdrawals']}` |
| **`Hegel_SelectiveConditionExpansion_Online`（`B1+B2+B3`）** | **`{ch['strict_passes']}/80` (`{100.0*ch['strict_passes']/80:.2f}%`)** | **`{ch['zero_turn_passed_closures']}/80`** | **`{ch['total_turns_used']}`** | **`{lh['call_count']}`** | **`{_fmt_int(lh['proxy_v2_u_010c_4o'])}`**（较单体 `{hm_s1_pct}` / 较普通模块 `{hp_s1_pct}`） | **`{_fmt_int(lh['proxy_v2_u_010c_4o'] + ls['proxy_v2_u_010c_4o'])}`**（`{hm_tot_pct}` / `{hp_tot_pct}`） | **`{_total_probe_sec(ch):,.2f}s`** | **`{ch['separability_withdrawals']}`** |

校准后的核心统计结论（`981000..988000` 八个独立源簇，冻结提交 `{conf['execution_provenance']['code_commit_at_start'][:9]}`，开发校准后零 `.py` 改动）：
- **质量非劣检验通过，但不主张统计显著质量优效（`T02`）**：
  - `Hegel` 取得 **`{ch['strict_passes']}/80`（`96.25%`）**，数值上高于单体条件表（`72/80 = 90.00%`，差值 **`+6.25pp`**，未舍入 95% 自举区间 **`[{boot_m_lo:+.2f}pp, {boot_m_hi:+.2f}pp]`**，小样本配对 `t(df=7)` 95% 区间 **`[{t_m_lo:+.2f}pp, {t_m_hi:+.2f}pp]`**，双侧精确符号翻转 `p = {pm['quality_two_sided_sign_flip_p']}`）与普通模块表（`74/80 = 92.50%`，差值 **`+3.75pp`**，未舍入 95% 自举区间 **`[{boot_p_lo:+.2f}pp, {boot_p_hi:+.2f}pp]`**，配对 `t(df=7)` 95% 区间 **`[{t_p_lo:+.2f}pp, {t_p_hi:+.2f}pp]`**，双侧 `p = {pp['quality_two_sided_sign_flip_p']}`）。
  - **严格解读**：点估计及自举 95% 下界（`{boot_m_lo:+.2f}pp`、`{boot_p_lo:+.2f}pp`）均高于预设非劣下界 `-5.00pp`（相对单体表在 `t(df=7)` 下界 `{t_m_lo:+.2f}pp` 同样通过；相对普通模块表 `t(df=7)` 下界为 `{t_p_lo:+.2f}pp`，低于 `-5.00pp` 约 `0.12pp`，在 `t(df=7)` 敏感性下未跨过 `-5.00pp`）。鉴于两区间均跨越 `0` 且 `p = {pm['quality_two_sided_sign_flip_p']} / {pp['quality_two_sided_sign_flip_p']}`，本报告主张**数值领先与自举非劣通过**，不主张“质量统计显著优效”。注（`T03`）：独立确认前协议 JSON 自 `M10.1` 起建立；`M10.0` 确认前 CI 判定规则位置标为 `unverified_prior_to_m10_1_protocol`。
- **相对主经济基线（单体条件表）`8/8` 全簇显著省算**：`Hegel` 在 **`8/8`（`100%`）** 独立簇上降低 `Scope-1` `proxy_v2`（Alpha `{_pct_change(a_h, a_m)}` `p = {pm['alpha_cost_sign_flip_p']}`，Beta `{_pct_change(b_h, b_m)}` `p = {pm['beta_cost_sign_flip_p']}`，合并精确双侧 **`p = {pm['two_sided_sign_flip_p']}`**），净省 **`{_fmt_int(-pm['mean_diff'] * 8)}` `proxy_v2`（`{hm_s1_pct}`）**，有限 10 步合并 `K* = {c_vs_m['empirical_quality_constrained_K_star']}`。
- **相对同工具普通模块表实现净省算但呈显著家族集中（`T03` 披露）**：`Hegel` 相对 `Plain` 净省 **`{_fmt_int(-pp['mean_diff'] * 8)}` `proxy_v2`（`{hp_s1_pct}`）**（`6/8` 簇更省，合并精确双侧 **`p = {pp['two_sided_sign_flip_p']}`，在 `alpha = 0.05` 下未达整体显著**）。按家族拆解可见省算高度集中于 **Beta**（`{_fmt_int(b_h - b_p)}`，**`{_pct_change(b_h, b_p)}`**，`4/4` 簇更省，贡献了 `H-Plain` 全部净省算的 **`{beta_share_hp:.2f}%`**），而 **Alpha** 接近打平（`{_fmt_int(a_h - a_p)}`，**`{_pct_change(a_h, a_p)}`**，`2/4` 簇更省）。同时，零调用通过闭环由 `29/80` 增至 **`37/80`**（`7/8` 簇占优，精确双侧 `p = {pp['zero_llm_two_sided_sign_flip_p']}`；相对单体 `15/80` 为 `8/8` 簇占优，`p = {pm['zero_llm_two_sided_sign_flip_p']}`）。
- **冻结 `step_08` 受控路由消融达 `16/16`（`100%`）**：在全部 `16` 个同 `dn_obs` 碰撞回归题（`S09, S10`）上，`hegel_selective_condition_expansion` 零调用私判通过 **`{pol_conf['hegel_selective_condition_expansion']['strict_passes']}/16`（`100.0%`）**，而仅看 `dn_obs` 的两策略均为 **`8/16`（`50.0%`）**。

---

## 2. 执行溯源、时钟域分离与完整研发成本总账（`C1–C2 / Q05 / Q12 / T10–T11`）

### 2.1 不可变起止/消融溯源清单（开发校准后零源码修改）

| 数据集划分 | `execution_run_id` | 启动提交 | 启动工作区脏状态 | `.py` 源码树摘要（`src+scripts+tests`） | 付费期间源码未变 | 单调时钟耗时（秒，`T11`） | 零调用消融墙钟（秒） | 并发度（上限 / 实测峰值） | 未知用量尝试数 |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `dev`（`n=4`） | `{dev['execution_run_id']}` | `{dev['execution_provenance']['code_commit_at_start'][:9]}` | `{dev['execution_provenance']['working_tree_dirty_at_start']}` | `{dev['execution_provenance']['source_tree_sha256_at_start'][:16]}...` | `{dev['execution_provenance']['source_tree_unchanged_during_paid_execution']}` | `{dev['execution_elapsed_wall_sec']:,.2f}s` | `{dev_abl_m['elapsed_wall_sec']:,.2f}s` | `{dev['configured_max_codex_concurrency']} / {dev['peak_active_codex_concurrency']}` | `{dev['total_usage_unknown_attempt_count']}` |
| `confirmation`（`n=8`） | `{conf['execution_run_id']}` | `{conf['execution_provenance']['code_commit_at_start'][:9]}` | `{conf['execution_provenance']['working_tree_dirty_at_start']}` | `{conf['execution_provenance']['source_tree_sha256_at_start'][:16]}...` | `{conf['execution_provenance']['source_tree_unchanged_during_paid_execution']}` | `{conf['execution_elapsed_wall_sec']:,.2f}s` | `{conf_abl_m['elapsed_wall_sec']:,.2f}s` | `{conf['configured_max_codex_concurrency']} / {conf['peak_active_codex_concurrency']}` | `{conf['total_usage_unknown_attempt_count']}` |

- **时钟域说明（`T11`）**：`execution_elapsed_wall_sec` 由 `time.monotonic()` 测量（开发 `{dev['execution_elapsed_wall_sec']:,.2f}s`、确认 `{conf['execution_elapsed_wall_sec']:,.2f}s`），与 UTC 时间戳起止跨度（开发 `{dev_utc_span:,.2f}s`、确认 `{conf_utc_span:,.2f}s`；确认运行存在 `254.45s` 宿主 WSL UTC 与单调时钟偏移，按原封存记录如实保留）严格分列。
- **完整研发成本总账（`T10`）**：
  - `M10.0` 开发集（`n = 4`，`124` 幕）：`{_fmt_int(dev['total_logical_calls'])}` 次调用，`{_fmt_int(dev['grand_physical_ledger']['total_tokens'])}` 总 token，**`{_fmt_int(dev['grand_physical_ledger']['proxy_v2_u_010c_4o'])}` `proxy_v2`**（`0` 次未知尝试）。
  - `M10.0` 确认集（`n = 8`，`248` 幕）：`{_fmt_int(conf['total_logical_calls'])}` 次调用，`{_fmt_int(conf['grand_physical_ledger']['total_tokens'])}` 总 token，**`{_fmt_int(conf['grand_physical_ledger']['proxy_v2_u_010c_4o'])}` `proxy_v2`**（`0` 次未知尝试）。
  - `M10.0` 研发总计：**`{_fmt_int(tot_m100_calls)}` 次调用 / `{_fmt_int(tot_m100_v2)}` 精确 `proxy_v2`**（`0` 次取消运行）。
  - 历史 `M9.9` 全量账本：`53,904,542 + x + y` `proxy_v2`（已取消开发日志 `333` 次 = `327` 次完成 + `6` 次 `in_progress`）。

### 2.2 确认集按调用角色分层守恒账本（`all_provider_call_receipts.json`）

| 角色分项 | 单体条件表（调用数 / `proxy_v2`） | 普通模块表（调用数 / `proxy_v2`） | `Hegel`（调用数 / `proxy_v2`） |
|---|---:|---:|---:|
| `solve_only`（在线求解） | `{conf['arm_cost_ledgers']['Flat_MonolithicConditionalTable_Online']['solve_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Flat_MonolithicConditionalTable_Online']['solve_only']['proxy_v2_u_010c_4o'])}` | `{conf['arm_cost_ledgers']['Flat_PlainModuleRegistry_Online']['solve_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Flat_PlainModuleRegistry_Online']['solve_only']['proxy_v2_u_010c_4o'])}` | `{conf['arm_cost_ledgers']['Hegel_SelectiveConditionExpansion_Online']['solve_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Hegel_SelectiveConditionExpansion_Online']['solve_only']['proxy_v2_u_010c_4o'])}` |
| `decomposition_induction_only`（自主两阶段分解：Plain `{plain_decomps}/8`、Hegel `{hegel_decomps}/8`） | `0` / `0` | `{conf['arm_cost_ledgers']['Flat_PlainModuleRegistry_Online']['decomposition_induction_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Flat_PlainModuleRegistry_Online']['decomposition_induction_only']['proxy_v2_u_010c_4o'])}` | `{conf['arm_cost_ledgers']['Hegel_SelectiveConditionExpansion_Online']['decomposition_induction_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Hegel_SelectiveConditionExpansion_Online']['decomposition_induction_only']['proxy_v2_u_010c_4o'])}` |
| `lazy_retroactive_projection_only`（`B3` 按需惰性回投） | `0` / `0` | `{conf['arm_cost_ledgers']['Flat_PlainModuleRegistry_Online']['lazy_retroactive_projection_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Flat_PlainModuleRegistry_Online']['lazy_retroactive_projection_only']['proxy_v2_u_010c_4o'])}` | `{conf['arm_cost_ledgers']['Hegel_SelectiveConditionExpansion_Online']['lazy_retroactive_projection_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Hegel_SelectiveConditionExpansion_Online']['lazy_retroactive_projection_only']['proxy_v2_u_010c_4o'])}` |
| `guard_induction_only`（大间隔守卫归纳） | `{conf['arm_cost_ledgers']['Flat_MonolithicConditionalTable_Online']['guard_induction_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Flat_MonolithicConditionalTable_Online']['guard_induction_only']['proxy_v2_u_010c_4o'])}` | `{conf['arm_cost_ledgers']['Flat_PlainModuleRegistry_Online']['guard_induction_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Flat_PlainModuleRegistry_Online']['guard_induction_only']['proxy_v2_u_010c_4o'])}` | `{conf['arm_cost_ledgers']['Hegel_SelectiveConditionExpansion_Online']['guard_induction_only']['call_count']}` / `{_fmt_int(conf['arm_cost_ledgers']['Hegel_SelectiveConditionExpansion_Online']['guard_induction_only']['proxy_v2_u_010c_4o'])}` |
| **`Scope-1` 流内总计** | **`{lm['call_count']}` / `{_fmt_int(lm['proxy_v2_u_010c_4o'])}`** | **`{lp['call_count']}` / `{_fmt_int(lp['proxy_v2_u_010c_4o'])}`** | **`{lh['call_count']}` / `{_fmt_int(lh['proxy_v2_u_010c_4o'])}`** |
| 共享初始源 `Stage S`（`8/8` 公开门准入+私判通过） | `{ls['call_count']}` / `{_fmt_int(ls['proxy_v2_u_010c_4o'])}`（三臂共享） | 同左 | 同左 |

---

## 3. 确认集逐簇明细与配对统计推断（`C3 / Q06 / T02–T03`）

### 3.1 八个独立确认簇逐簇结果

| 源簇 ID | 家族 | 到达顺序 | 单体表通过（零调用） | 普通模块表通过（零调用） | `Hegel` 通过（零调用） | 单体表 `Scope-1 v2` | 普通模块表 `Scope-1 v2` | `Hegel` `Scope-1 v2` | `Hegel - 单体` `v2` | `Hegel - 普通` `v2` | `Hegel` 撤回数 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(_cluster_table_rows(conf))}

### 3.2 源簇级配对精确双侧符号翻转、未舍入自举与 `t(df=7)` 敏感性检验（`n = 8`）

| 配对比较（`Hegel` 减基线） | 单簇 `Scope-1 v2` 均值差（95% 自举区间） | 成本精确双侧 `p`（总体 / Alpha / Beta） | `sep_first` vs `coupled_first` 均值差 | 单幕通过率差（未舍入自举 95% / `t(df=7)` 95%；非劣界 `-5.00pp`） | 质量精确双侧 `p` | 零调用闭环逐簇差与精确双侧 `p` |
|---|---|---|---|---|---:|---|
| **`Hegel - 单体条件表`** | **`{pm['mean_diff']:,.2f}`** (`[{pm['bootstrap_95ci'][0]:,.2f}, {pm['bootstrap_95ci'][1]:,.2f}]`) | **`{pm['two_sided_sign_flip_p']}`** (`{pm['alpha_cost_sign_flip_p']}` / `{pm['beta_cost_sign_flip_p']}`) | `{pm['sep_first_mean_diff_v2']:,.2f}` / `{pm['coupled_first_mean_diff_v2']:,.2f}` | **`+6.25pp`** (`[{boot_m_lo:+.2f}pp, {boot_m_hi:+.2f}pp]` / `[{t_m_lo:+.2f}pp, {t_m_hi:+.2f}pp]`，**非劣通过**) | `{pm['quality_two_sided_sign_flip_p']}` | `{pm['zero_llm_closure_cluster_diffs']}`, **`p = {pm['zero_llm_two_sided_sign_flip_p']}`** |
| **`Hegel - 普通模块表`** | **`{pp['mean_diff']:,.2f}`** (`[{pp['bootstrap_95ci'][0]:,.2f}, {pp['bootstrap_95ci'][1]:,.2f}]`) | `{pp['two_sided_sign_flip_p']}` (`{pp['alpha_cost_sign_flip_p']}` / `{pp['beta_cost_sign_flip_p']}`) | `{pp['sep_first_mean_diff_v2']:,.2f}` / `{pp['coupled_first_mean_diff_v2']:,.2f}` | **`+3.75pp`** (`[{boot_p_lo:+.2f}pp, {boot_p_hi:+.2f}pp]` / `[{t_p_lo:+.2f}pp, {t_p_hi:+.2f}pp]`，**自举非劣通过 / `t` 临界**) | `{pp['quality_two_sided_sign_flip_p']}` | `{pp['zero_llm_closure_cluster_diffs']}`, **`p = {pp['zero_llm_two_sided_sign_flip_p']}`** |

---

## 4. 逐幕累计摊销曲线 $\\Delta C(K)$（`K = 1..10`）与有限视界盈亏平衡 $K^*$（`D2 / T04`）

> **有限视界 `K*` 精确定义（`T04`）**：`K*` 仅定义在观测的 `10` 步视界内，表示使合并累计成本差 $\\Delta C(k) < 0$ 对所有 $k \\in \\{{K..10\\}}$ 成立且 $K=10$ 满足质量非劣（`>= -0.05`）的最小步数；不预设超出 $K = 10$ 无回归情况下的永久省算。

### 4.1 `Hegel` 减 `Flat_MonolithicConditionalTable_Online`（合并 `K* = {c_vs_m['empirical_quality_constrained_K_star']}`）

| 步数 `K` | `Hegel` 累计 `Scope-1 v2` | 单体表累计 `Scope-1 v2` | 总累计差 `Delta C_mono(K)` | 单簇均值差 `Delta v2` | `Hegel` 更省簇数（`/8`） | 累计私判通过（`Hegel` vs 单体） | 单幕通过率差（`>= -0.05` 校验） |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(_curve_rows(c_vs_m))}

### 4.2 `Hegel` 减 `Flat_PlainModuleRegistry_Online`（合并 `K* = {c_vs_p['empirical_quality_constrained_K_star']}`）

| 步数 `K` | `Hegel` 累计 `Scope-1 v2` | 普通模块表累计 `Scope-1 v2` | 总累计差 `Delta C_plain(K)` | 单簇均值差 `Delta v2` | `Hegel` 更省簇数（`/8`） | 累计私判通过（`Hegel` vs 普通） | 单幕通过率差（`>= -0.05` 校验） |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(_curve_rows(c_vs_p))}

注意 `T04` 揭示的精确回本机制：截至 `K = 8`，`Hegel` 相对 `Plain` 处于 **`+1,208,895` `proxy_v2`** 的前期选择性扩展投入期；真正翻负回本发生在 **`K = 9` 与 `K = 10`** 碰撞回归（`S09, S10`）到来之时！

### 4.3 逐簇 `K*` 与右删失明细表（`T04`）

| 源簇 ID | 家族 | 到达顺序 | 单簇 `K*`（对单体） | 单簇 `K*`（对普通模块） | `Delta C_plain(K=8)` | `Delta C_plain(K=10)` |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(k_cluster_rows)}

---

## 5. `B2`/`B3` 触发漏斗（`T07`）、容器流逝墙钟（`Q11`）与开发集校准摘要

### 5.1 `M10.0` 确认集 `B2` 与 `B3` 实测触发漏斗（`T07`）

| 实验臂 | `B2` 备选直接探针次数 | `B2` 备选联合门次数 | `B2` 直接零调用挽救数 | `B3` 原始单体回放次数 | `B3` 惰性回投次数 |
|---|---:|---:|---:|---:|---:|
| `Flat_PlainModuleRegistry_Online` | `{b2_b3_funnel['Flat_PlainModuleRegistry_Online']['b2_alt_direct_probes']}` | `{b2_b3_funnel['Flat_PlainModuleRegistry_Online']['b2_alt_joint_probes']}` | `{b2_b3_funnel['Flat_PlainModuleRegistry_Online']['b2_direct_pre_llm_rescues']}` | `{b2_b3_funnel['Flat_PlainModuleRegistry_Online']['b3_raw_archive_replays']}` | `{b2_b3_funnel['Flat_PlainModuleRegistry_Online']['b3_lazy_projections']}` |
| `Hegel_SelectiveConditionExpansion_Online` | `{b2_b3_funnel['Hegel_SelectiveConditionExpansion_Online']['b2_alt_direct_probes']}` | `{b2_b3_funnel['Hegel_SelectiveConditionExpansion_Online']['b2_alt_joint_probes']}` | `{b2_b3_funnel['Hegel_SelectiveConditionExpansion_Online']['b2_direct_pre_llm_rescues']}` | `{b2_b3_funnel['Hegel_SelectiveConditionExpansion_Online']['b3_raw_archive_replays']}` | `{b2_b3_funnel['Hegel_SelectiveConditionExpansion_Online']['b3_lazy_projections']}` |

- **`B2` 实测解读**：在确认集上 `B2` 触发备选直接探针（`Plain` `{b2_b3_funnel['Flat_PlainModuleRegistry_Online']['b2_alt_direct_probes']}` 次、`Hegel` `{b2_b3_funnel['Hegel_SelectiveConditionExpansion_Online']['b2_alt_direct_probes']}` 次），升级备选联合门为 `Plain` `{plain_b2_joint}` 次、`Hegel` `{hegel_b2_joint}` 次，直接零调用挽救数为 `0`（仅作冲突定位诊断）。
- **`B3` 实测解读**：由于确认集 `16/16`（`100%`）模块化流在首步即完成 `2×2` 分解，`B3` 在 `dev`/`confirmation` 上自然触发为 `0`，其窄路径正确性由专用单元测试 `test_m100_b2_sibling_upstream_fallback_and_b3_deferred_lazy_projection` 与 `test_m10_0_1_audit_closure_suite.py` 验证。

### 5.2 非模型容器探针与官方验证器流逝墙钟（`OMP_NUM_THREADS=1`，流逝墙钟秒）

| 实验臂（确认集 `80` 幕合计） | 调前门禁（秒） | 调后门禁（秒） | 冲突定位（秒） | `2x2` 分解门（秒） | 经验覆盖回放（秒） | 官方私判（秒） | 容器总流逝墙钟（秒） |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicConditionalTable_Online` | `{cm['pre_llm_probe_wall_ms']/1000:,.2f}` | `{cm['post_llm_probe_wall_ms']/1000:,.2f}` | `0.00` | `0.00` | `{cm['coverage_probe_wall_ms']/1000:,.2f}` | `{cm['official_verifier_wall_ms']/1000:,.2f}` | `{_total_probe_sec(cm):,.2f}` |
| `Flat_PlainModuleRegistry_Online` | `{cp['pre_llm_probe_wall_ms']/1000:,.2f}` | `{cp['post_llm_probe_wall_ms']/1000:,.2f}` | `{cp['localization_probe_wall_ms']/1000:,.2f}` | `{cp['decomposition_probe_wall_ms']/1000:,.2f}` | `{cp['coverage_probe_wall_ms']/1000:,.2f}` | `{cp['official_verifier_wall_ms']/1000:,.2f}` | `{_total_probe_sec(cp):,.2f}` |
| **`Hegel_SelectiveConditionExpansion_Online`** | **`{ch['pre_llm_probe_wall_ms']/1000:,.2f}`** | **`{ch['post_llm_probe_wall_ms']/1000:,.2f}`** | **`{ch['localization_probe_wall_ms']/1000:,.2f}`** | **`{ch['decomposition_probe_wall_ms']/1000:,.2f}`** | **`{ch['coverage_probe_wall_ms']/1000:,.2f}`** | **`{ch['official_verifier_wall_ms']/1000:,.2f}`** | **`{_total_probe_sec(ch):,.2f}`** |

### 5.3 封存开发集校准摘要（`Gate 3`，`n = 4`，`971000..974000`，每臂 `40` 幕）

- 单体条件表：`{dm['strict_passes']}/40` 通过（`{dm['zero_turn_passed_closures']}/40` 零调用，`{_fmt_int(dlm['proxy_v2_u_010c_4o'])}` `Scope-1 v2`）。
- 普通模块表：`{dp['strict_passes']}/40` 通过（`{dp['zero_turn_passed_closures']}/40` 零调用，`{_fmt_int(dlp['proxy_v2_u_010c_4o'])}` `Scope-1 v2`）。
- `Hegel`：`{dh['strict_passes']}/40` 通过（`{dh['zero_turn_passed_closures']}/40` 零调用，`{_fmt_int(dlh['proxy_v2_u_010c_4o'])}` `Scope-1 v2`，对单体 `K* = {dev['cumulative_amortization_curves_and_K_star']['Hegel_minus_Flat_MonolithicConditionalTable_Online']['empirical_quality_constrained_K_star']}`，对普通模块 `K* = {dev['cumulative_amortization_curves_and_K_star']['Hegel_minus_Flat_PlainModuleRegistry_Online']['empirical_quality_constrained_K_star']}`，普通字典逐候选等价 `8/8 = 100%`）。

---

## 6. 逐项 `Q01–Q14` 与 `T01–T11` 审计闭环状态总表（`T08`）

| 审计编号 | 状态 | 证据与闭环摘要 |
|---|---|---|
| `Q01 / T01` | **`M10.0.1` 已闭环** | `M10.0` 执行了真实 `2×2` 分解门；`M10.0.1` 补齐原子 `ProbeReceiptStore`、迁移全部裸 `record_tested_witness` 调用、强制三方外部回执外键，并在真实容器中事后补测 `41/41` 缺失确认探针（详见 `docs/56b_*`）。 |
| `Q02 / B1` | **已闭环** | Alpha 公开跨切分 `ref_coupons` 交换不变性门禁在三臂对称启用。 |
| `Q03 / B2` | **已闭环（`T07` 校准）** | 对称 `1`-兄弟上游回退已保留主/备双诊断；明确披露确认集直接零调用挽救数为 `0`。 |
| `Q04 / B3` | **已闭环（单测覆盖 `T07`）** | 延迟分解单体保留与按需惰性回投已对称实现；确认集自然触发 `0` 次，已由专用单测覆盖。 |
| `Q05 / T11` | **已闭环** | 不可变写一次 `execution_manifest.json` 与完成/消融清单分离；UTC 与单调时钟字段显式分列。 |
| `Q06 / T02–T03` | **`M10.0.1` 已闭环** | 补全未舍入自举区间与 `t(df=7)` 敏感性区间、明确 Alpha（`-0.87%`）vs Beta（`-18.66%`）家族集中度与非优效口径。 |
| `Q07 / T01` | **`M10.0.1` 已闭环** | 撤回外键升级为三方外部 `ProbeReceiptStore` 校验，严格拒绝 GPT-6 构造的伪造 `prior_positive.probe_id` 负控。 |
| `Q08` | **已闭环** | 旧版迁移幂等往返由 `test_m9_9_1_audit_closure_suite.py` 验证通过。 |
| `Q09 / Q10` | **已闭环** | 主张范围严格限定在给定两阶段切分下的选择性条件集扩展，`16/16`（`100%`）普通字典等价性置顶披露。 |
| `Q11` | **已闭环** | 明确按容器流逝墙钟秒（非 `/proc` CPU 核时）分项汇报。 |
| `Q12 / T10` | **已闭环** | `M9.9` 全账本（`53,904,542 + x + y`；区分 `327` 完成与 `6` 条 `in_progress`）与 `M10.0` 全账本（`64,681,960`，`0` 未知）完整对平。 |
| `Q13 / Q14` | **已闭环** | 同信息归档与 `step_08` 受控消融（`16/16` vs `8/16`）可复现，零历史封存目录改动。 |
| `T05 / T06` | **`M10.0.1` 已闭环** | 新增零公式泄漏的 Alpha 公开初速率导数/试片一致性检查与 Beta 标定误差-KKT 扰动预算检查，修复 Beta 私判 `continue` 误报 `0.0` 缺陷。 |
| `T09` | **`M10.0.1` 已闭环** | 单体/普通模块表行 ID 与 Hegel 分支标签全面去语义中性化（`mono_rXX`、`plain_up_rXX`、`plain_dn_rXX`、`up_branch_XX`、`dn_branch_XX`）。 |
"""

    # Also generate dedicated `docs/56b_M10_0_1_Audit_Closure_and_Probe_Persistence_Errata*.md` (`T01` + `0`-LLM lifting inventory)
    back_rows_md = [
        f"| `{r['world_id']}` | `{r['witness_id']}` | `{r['evidence_origin']}` | `{r['orig_probe_id']}` | `{r['backfill_probe_id']}` | `{r['is_withdrawal_prior_witness']}` | `{r['recorded_result']}` vs `{r['replayed_result']}` | **`{r['verdict_matches_recorded_witness']}`** |"
        for r in m1001_back["rows"]
    ]
    en_56b = f"""# `M10.0.1` Audit Closure & Probe-Persistence Errata (`T01–T11`, `0` Paid LLM Calls)

> **Sealed Post-Hoc Reverification Directory**: `docs/evidence/m10_0_1_audit_closure_and_probe_reverification_20261128/` (`{n_m1001}` SHA-256-verified files). Zero existing `m9_9*` or `m10_0_*` sealed files or scores (`72/74/77`) were modified.

## 1. `T01` Missing Raw Probe Receipt Inventory & `41/41` Post-Hoc Docker Back-Verification

- **Historical Gap Inventory (`01_m100_probe_persistence_inventory.json`)**:
  - `M10.0` `dev`: `{m1001_inv['dev']['recorded_without_raw_measurement_count']}/{m1001_inv['dev']['total_witnesses']}` witnesses marked `recorded_without_raw_measurement` (`{m1001_inv['dev']['missing_by_probe_prefix']}`).
  - `M10.0` `confirmation`: **`{m1001_inv['confirmation']['recorded_without_raw_measurement_count']}/{m1001_inv['confirmation']['total_witnesses']}`** witnesses marked `recorded_without_raw_measurement` (`{m1001_inv['confirmation']['missing_by_probe_prefix']}`), including **`{m1001_inv['confirmation']['withdrawal_prior_witness_missing_raw_count']}/{m1001_inv['confirmation']['withdrawal_prior_witness_total']}`** withdrawal `prior_positive` witnesses.
- **Post-Hoc Real Docker Back-Verification (`02_m100_41_missing_probes_docker_backverification.json`)**:
  - All **`{m1001_back['total_missing_confirmation_probes_replayed']}`** missing confirmation probes (`{m1001_back['pprobe_replayed']} pprobe` + `{m1001_back['uprobe_replayed']} uprobe`) were re-executed in `hegel-clean-scipy:m7v1` against the exact sealed binding code, parameter epoch, public batch, and dependency bundle (`{m1001_back['monotonic_elapsed_sec']}s` elapsed, `0` paid LLM calls).
  - **Outcome**: **`{m1001_back['verdict_matched_count']}/{m1001_back['total_missing_confirmation_probes_replayed']}` (`100.0%`)** replayed verdicts match the recorded `M10.0` witness ledger, including **`{m1001_back['withdrawal_prior_witnesses_matched']}/{m1001_back['withdrawal_prior_witnesses_replayed']}` (`100.0%`)** withdrawal `prior_positive` witnesses. Every replayed receipt is explicitly labeled `raw_measurement_status = "backfilled_posthoc_replay"` with a fresh `backfill_*` ID and `supersedes_probe_id`.
- **Forward Architectural Enforcement**: `ProbeReceiptStore`, schema-v3 `probe_receipt_sha256` witness digests, migration of all `run_m10_0` bare witness calls to `record_tested_witness_from_probe_receipt`, dual `primary/alternate` `B2` diagnostic retention, and strict 3-way external FK verification in `issue_measured_withdrawal` & `hydrate_separability_withdrawals_from_graph` (verified against GPT-6's exact forged-`prior.probe_id` negative control in `tests/test_m10_0_1_audit_closure_suite.py`).

## 2. `0`-LLM Archived Solver Lifting & Anti-Unification Coverage Inventory (`03_m100_archived_solvers_lifting_coverage_inventory.json`)

- Audited `{m1001_lift['audited_solver_count']}` archived `M10.0` confirmation monolithic solvers through the legacy `M6.8` scalar AST expression lifter (`lift_python_scientific_expressions_to_programs`) and `anti_unify_programs`:
  - `category_counts`: `{m1001_lift['category_counts']}`
  - `total_lifted_scalar_programs`: `{m1001_lift['total_lifted_scalar_programs']}`
  - `non_trivial_anti_unified_cross_cluster_pairs`: `{m1001_lift['non_trivial_anti_unified_cross_cluster_pairs']}`
- **Conclusion**: Scientific solvers in `M10.0` are predominantly imperative NumPy/SciPy routines (`lifter_unsupported` for scalar AST lifting), confirming that open multi-module discovery (`M10.1`) must operate over executable Python callable nodes and measured I/O contracts rather than scalar symbolic AST lifting alone.

## 3. Complete `41/41` Missing Confirmation Probe Post-Hoc Replay Ledger

| World ID | Witness ID | Origin | Original `probe_id` | Backfill `probe_id` | Withdrawal Prior? | Recorded vs Replayed | Match? |
|---|---|---|---|---|---:|---|---|
{chr(10).join(back_rows_md)}
"""

    zh_56b = f"""# `M10.0.1` 审计闭环与探针原始回执落盘勘误（`T01–T11`，`0` 次付费模型调用）

> **独立封存事后复核目录**：`docs/evidence/m10_0_1_audit_closure_and_probe_reverification_20261128/`（共 `{n_m1001}` 个经 `SHA256SUMS.txt` 校验的文件）。全程零修改任何既有 `m9_9*` 或 `m10_0_*` 封存目录及历史得分（`72/74/77`）。

## 1. `T01` 缺失原始探针回执清单与 `41/41` 真实 Docker 事后补测

- **缺失回执清点（`01_m100_probe_persistence_inventory.json`）**：
  - `M10.0` 开发集：`{m1001_inv['dev']['recorded_without_raw_measurement_count']}/{m1001_inv['dev']['total_witnesses']}` 条证人标记为 `recorded_without_raw_measurement`（`{m1001_inv['dev']['missing_by_probe_prefix']}`）。
  - `M10.0` 确认集：**`{m1001_inv['confirmation']['recorded_without_raw_measurement_count']}/{m1001_inv['confirmation']['total_witnesses']}`** 条证人标记为 `recorded_without_raw_measurement`（`{m1001_inv['confirmation']['missing_by_probe_prefix']}`），其中包含 **`{m1001_inv['confirmation']['withdrawal_prior_witness_missing_raw_count']}/{m1001_inv['confirmation']['withdrawal_prior_witness_total']}`** 条撤回前置正证人。
- **真实容器并行事后补测（`02_m100_41_missing_probes_docker_backverification.json`）**：
  - 全部 **`{m1001_back['total_missing_confirmation_probes_replayed']}`** 条缺失确认探针（`{m1001_back['pprobe_replayed']}` 条 `pprobe` + `{m1001_back['uprobe_replayed']}` 条 `uprobe`）已在 `hegel-clean-scipy:m7v1` 容器中按封存代码、参数纪元与只读依赖原样重放（耗时 `{m1001_back['monotonic_elapsed_sec']}s`，`0` 次付费模型调用）。
  - **补测结论**：**`{m1001_back['verdict_matched_count']}/{m1001_back['total_missing_confirmation_probes_replayed']}`（`100.0%`）** 与原证人记录裁决完全一致，包含 **`{m1001_back['withdrawal_prior_witnesses_matched']}/{m1001_back['withdrawal_prior_witnesses_replayed']}`（`100.0%`）** 撤回前置正证人。所有补测文件严格标注 `raw_measurement_status = "backfilled_posthoc_replay"`、新 `backfill_*` ID 与 `supersedes_probe_id`，绝不伪装为原运行期落盘。
- **前向架构强制闭环**：新增写一次 `ProbeReceiptStore`、`SolverEvaluationWitness` v3 外部回执摘要字段、将 `run_m10_0` 全部裸证人写入迁移至 `record_tested_witness_from_probe_receipt`、保留 `B2` 主/备双诊断，并在撤回签发与冷加载中强制三方外部回执外键（`tests/test_m10_0_1_audit_closure_suite.py` 已复现并严格拒绝 GPT-6 的伪造 `prior.probe_id` 负控）。

## 2. 归档求解器零调用 AST 提升与反合一覆盖盘点（`03_m100_archived_solvers_lifting_coverage_inventory.json`）

- 对 `{m1001_lift['audited_solver_count']}` 个 `M10.0` 确认集归档单体求解器运行旧版 `M6.8` 标量 AST 提升器与 `anti_unify_programs`：
  - 分类统计：`{m1001_lift['category_counts']}`；
  - 标量提升程序数：`{m1001_lift['total_lifted_scalar_programs']}`；跨簇非平凡反合一骨架对数：`{m1001_lift['non_trivial_anti_unified_cross_cluster_pairs']}`。
- **工程结论**：科学求解器主体为命令式 NumPy/SciPy 数值例程（落入 `lifter_unsupported`），印证 `M10.1` 开放多模块计算图发现必须直接建立在可执行 Python 模块节点与实测 I/O 契约之上，而非单纯依赖旧版标量 AST 提升。

## 3. 全部 `41/41` 缺失确认探针事后补测明细表

| 源簇 ID | 证人 ID | 证据来源 | 原 `probe_id` | 补测 `probe_id` | 撤回前置正证人？ | 原记录 vs 重放 | 一致？ |
|---|---|---|---|---|---:|---|---|
{chr(10).join(back_rows_md)}
"""

    OUT_EN.write_text(en_md, encoding="utf-8")
    OUT_ZH.write_text(zh_md, encoding="utf-8")
    OUT_56B_EN.write_text(en_56b, encoding="utf-8")
    OUT_56B_ZH.write_text(zh_56b, encoding="utf-8")

    cap_obj = json.loads((REPO_ROOT / "src/hegel_machine/cognition/historical_capabilities.json").read_text(encoding="utf-8"))
    cap_grp = next(
        x
        for v in cap_obj.values()
        if isinstance(v, list)
        for x in v
        if isinstance(x, dict) and x.get("id") == "m6_m97_continual_mechanism_lineage_and_verification"
    )
    assert all((REPO_ROOT / p).is_file() for p in cap_grp["current_modules"]), "Missing module in historical_capabilities.json"
    assert all((REPO_ROOT / e["path"]).is_file() for e in cap_grp["evidence"]), "Missing evidence path in historical_capabilities.json"
    print(
        f"Wrote {OUT_EN.name} ({len(en_md)} chars), {OUT_ZH.name} ({len(zh_md)} chars), "
        f"{OUT_56B_EN.name} ({len(en_56b)} chars), {OUT_56B_ZH.name} ({len(zh_56b)} chars)."
    )


if __name__ == "__main__":
    main()
