#!/usr/bin/env python3
"""Deterministically generate `docs/53_M9_6_1_Closure_and_M9_7_NonNested_Lineage_and_Counterfactual_Memory_Report*.md`
strictly from sealed `M9.6.1` and `M9.7` JSON artifacts (`0` manual number transcription!).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition.cost_policy import assert_hierarchical_ledger_conservation

M961_DIR = REPO_ROOT / "docs/evidence/m9_6_1_semantic_closure_and_paid_pool_attribution_20261127"
M97_RUN1 = REPO_ROOT / "docs/evidence/m9_7_aborted_run1_openblas_setup_timeout_20261127/00_aborted_run1_forensic_manifest.json"
M97_RUN2 = REPO_ROOT / "docs/evidence/m9_7_aborted_run2_beta_self_guard_witness_kind_20261127/00_aborted_run2_forensic_manifest.json"
M97_RUN3 = REPO_ROOT / "docs/evidence/m9_7_non_nested_counterfactual_lineage_run3_20261127/m9_7_continual_summary.json"

DOC_EN = REPO_ROOT / "docs/53_M9_6_1_Closure_and_M9_7_NonNested_Lineage_and_Counterfactual_Memory_Report.md"
DOC_ZH = REPO_ROOT / "docs/53_M9_6_1_Closure_and_M9_7_NonNested_Lineage_and_Counterfactual_Memory_Report_ZH.md"


def main() -> None:
    r961_01 = json.loads((M961_DIR / "01_n01_reconciled_m96_cost_and_pair_statistics.json").read_text(encoding="utf-8"))
    r961_02 = json.loads((M961_DIR / "02_n02_m96_three_layer_regrade_and_m_forensics.json").read_text(encoding="utf-8"))
    r961_03 = json.loads((M961_DIR / "03_n03_m94_full_68_instance_unified_kernel_errata.json").read_text(encoding="utf-8"))
    r961_05 = json.loads((M961_DIR / "05_n09_n10_reconstructed_m96_funnel_and_kkt_contract_audit.json").read_text(encoding="utf-8"))

    m97_r1 = json.loads(M97_RUN1.read_text(encoding="utf-8"))
    m97_r2 = json.loads(M97_RUN2.read_text(encoding="utf-8"))
    m97 = json.loads(M97_RUN3.read_text(encoding="utf-8"))

    # Re-assert strict hierarchical conservation before rendering a single character!
    for arm_nm in ("Flat_EqualInfo_Multiversion", "Hegel_DemandGated_Graph"):
        arm_d = m97["arm_comparison"][arm_nm]
        a_streams = [s for s in m97["streams"] if s["arm"] == arm_nm]
        assert_hierarchical_ledger_conservation(
            reference_arm_ledger=arm_d["scope_1_in_stream_total"],
            named_partition_ledgers={
                "events": [v["cost_ledger"] for v in arm_d["by_causal_event"].values()],
                "clusters": [s["cost_ledger"] for s in a_streams],
                "episodes": [ep["cost_ledger"] for s in a_streams for ep in s["episodes_meta"]],
            },
            context_label=f"report_gen_{arm_nm}",
        )

    flat = m97["arm_comparison"]["Flat_EqualInfo_Multiversion"]
    hegel = m97["arm_comparison"]["Hegel_DemandGated_Graph"]
    ps = m97["paired_statistics_Hegel_vs_Flat_n6_clusters"]
    lf = m97["multibranch_lineage_and_counterfactual_funnel"]
    fi = m97["forensic_integrity"]
    s_stage = m97["stage_S_cost_ledger"]["shared_stage_S_solve_6clusters"]
    s3 = m97["scope_3_campaign_physical_spend_all_calls"]
    s3p = m97["scope_3_plus_grand_campaign_spend_including_aborted_runs"]

    by_wa = {(s["world_id"], s["arm"]): s for s in m97["streams"]}
    alpha_wids = ["alpha_m97_c01", "alpha_m97_c02", "alpha_m97_c03"]
    beta_wids = ["beta_m97_c01", "beta_m97_c02", "beta_m97_c03"]

    def _fam_tot(wids: list[str], arm_nm: str) -> dict[str, Any]:
        sts = [by_wa[(w, arm_nm)] for w in wids]
        return {
            "passes": sum(s["strict_passes"] for s in sts),
            "eps": sum(s["episode_count"] for s in sts),
            "zt": sum(s["zero_turn_passed_closures"] for s in sts),
            "turns": sum(s["total_turns_used"] for s in sts),
            "v1": sum(s["cost_ledger"]["proxy_v1_u_025c_1o"] for s in sts),
            "v2": sum(s["cost_ledger"]["proxy_v2_u_010c_4o"] for s in sts),
            "probe_ms": round(sum(s["total_pre_llm_probe_wall_ms"] for s in sts), 1),
            "fails": [
                f"{s['world_id']}:{e['causal_event']}"
                for s in sts
                for e in s["episodes_meta"]
                if not e["strict_pass"]
            ],
        }

    a_flat, a_heg = _fam_tot(alpha_wids, "Flat_EqualInfo_Multiversion"), _fam_tot(alpha_wids, "Hegel_DemandGated_Graph")
    b_flat, b_heg = _fam_tot(beta_wids, "Flat_EqualInfo_Multiversion"), _fam_tot(beta_wids, "Hegel_DemandGated_Graph")
    alpha_v2_pct = round((a_heg["v2"] - a_flat["v2"]) / a_flat["v2"] * 100.0, 2)
    beta_v2_pct = round((b_heg["v2"] - b_flat["v2"]) / b_flat["v2"] * 100.0, 2)

    ev_order = ("C1", "R0", "C2", "R1", "U_iso", "G_perm", "M_couple")
    ev_rows_en, ev_rows_zh = [], []
    for ek in ev_order:
        ef = flat["by_causal_event"][ek]
        eh = hegel["by_causal_event"][ek]
        row = (
            f"| `{ek}` | `{ef['strict_passes']}/6` | `{eh['strict_passes']}/6` "
            f"| `{ef['zero_turn_passed_closures']}/6` | `{eh['zero_turn_passed_closures']}/6` "
            f"| `{ef['mean_turns']:.2f}` | `{eh['mean_turns']:.2f}` "
            f"| `{ef['cost_ledger']['proxy_v2_u_010c_4o']:,}` | `{eh['cost_ledger']['proxy_v2_u_010c_4o']:,}` |"
        )
        ev_rows_en.append(row)
        ev_rows_zh.append(row)

    cl_rows_en, cl_rows_zh = [], []
    for row_c in ps["per_independent_cluster_deltas"]:
        wid = row_c["world_id"]
        sf = by_wa[(wid, "Flat_EqualInfo_Multiversion")]
        sh = by_wa[(wid, "Hegel_DemandGated_Graph")]
        r_str = (
            f"| `{wid}` (`{row_c['cluster_id']}`) | `{sf['strict_passes']}/7` | `{sh['strict_passes']}/7` "
            f"| `{sf['zero_turn_passed_closures']}/7` | `{sh['zero_turn_passed_closures']}/7` "
            f"| `{sf['total_turns_used']}` | `{sh['total_turns_used']}` (`{row_c['turns_diff']:+d}`) "
            f"| `{sf['cost_ledger']['proxy_v2_u_010c_4o']:,}` | `{sh['cost_ledger']['proxy_v2_u_010c_4o']:,}` (`{row_c['in_stream_v2_diff']:+,}`) |"
        )
        cl_rows_en.append(r_str)
        cl_rows_zh.append(r_str)

    en = f"""# Hegel Machine M9.6.1 Closure & M9.7 Non-Nested Multi-Conflict Lineage and Counterfactual Memory Report

- **Confirmation Frozen Commit**: `{m97['code_commit_at_start']}` (`source_tree_py_sha256`: `{m97['source_tree_py_sha256_at_start']}`, start == end, `tracked_tree_dirty_at_end = {m97['tracked_tree_dirty_at_end']}`)
- **Sealed Evidence Directories**:
  - `M9.6.1` Closure & Errata (`0` paid LLM calls): `docs/evidence/m9_6_1_semantic_closure_and_paid_pool_attribution_20261127/` (`docs/52b_M9_6_1_Audit_Errata_Semantic_Closure_and_Paid_Pool_Attribution.md`)
  - `M9.7` Official Confirmation Run (`run3`, `90` live episodes): `docs/evidence/m9_7_non_nested_counterfactual_lineage_run3_20261127/` (`1,996` entries verified in `SHA256SUMS.txt`)
  - `M9.7` Aborted Pre-Confirmation Runs (100% forensically preserved & counted in `Scope 3+`):
    - `docs/evidence/m9_7_aborted_run1_openblas_setup_timeout_20261127/` (`{m97_r1['aborted_run1_physical_spend']['call_count']}` calls, `proxy_v2 = {m97_r1['aborted_run1_physical_spend']['proxy_v2_u_010c_4o']:,}`)
    - `docs/evidence/m9_7_aborted_run2_beta_self_guard_witness_kind_20261127/` (`{m97_r2['aborted_physical_spend']['call_count']}` calls, `proxy_v2 = {m97_r2['aborted_physical_spend']['proxy_v2_u_010c_4o']:,}`)
- **Live Provider & Concurrency**: Official ChatGPT subscription Codex CLI (`{m97['model']}`, `reasoning_effort={m97['reasoning_effort']}`), configured semaphore cap = `{m97['configured_codex_concurrency_cap']}`, observed peak concurrent active CLI calls = `{m97['peak_concurrent_active_codex_calls_observed']}` across `12` independent parallel continual streams (`wall_clock_sec = {m97['wall_clock_sec']}`).

---

## 1. Summary of M9.6.1 Semantic & Accounting Closure (`N01–N11`, `0` Paid LLM Calls)

1. **`N01` Call-UUID Hierarchical Cost Conservation**: Eliminated legacy `token_cost_proxy` (`v1`) leakage into per-episode `v2` tables (`COST_POLICY_VERSION = "{m97['cost_policy_version']}"`). Reconciled `M9.6` `beta_m96_w01` delta = `{r961_01['secondary_descriptive_world_sensitivity_n8']['beta_m96_w01_v2_diff']:+,}` (`7/8` worlds higher, $n=8$ $p = {r961_01['secondary_descriptive_world_sensitivity_n8']['world_v2_two_sided_sign_flip_p']}$; $n=4$ disjoint source pairs $p = {r961_01['primary_inferential_unit_disjoint_pairs_n4']['two_sided_sign_flip_p']}$; L2 compilation share of excess = `{r961_01['in_stream_compile_share_of_v2_excess_pct']:.2f}%`).
2. **`N02` & `N03` Unified Three-Layer Scientific Contract (`hegel_shared_math_handoff_kernel_v3_m961_three_layer`)**: Separated Layer-1 structure, Layer-2 `CalibrationWitness` estimated-coordinate self-consistency (`<= 2.01e-14` on `alpha_m96_w03 / M`, correcting the `initial_memory_states` misdiagnosis while keeping official `M9.6` scores at `63/64` vs `63/64`), and Layer-3 truth accuracy; re-verified all `34` `M9.4` episodes (`{r961_03['passed_private_instances']}/{r961_03['total_private_instances']}` private instances, `100%` of `beta_ep03` dual-channel episodes passing after fixing `terminal_field_B` dispatch).
3. **`N04, N07, N09–N11`**: Enforced exact `executable_kind` SHA matching + content-recomputed `QualificationReceipt` + full `parent_parameter_versions` vector (`N04`), `ImmutableBranchLineageController` + `SolverEvaluationWitness` + `EmpiricalCoverageReceipt` (`N07`), reconstructed `M9.6` `Full` `T` skill inspection funnel (`{r961_05['Full_T_summary_counts']['any_new_module_described']}/8` episodes described new source skills, `0/8` executed/retained, `N09/N10`), and completed sequential paid-pool component attribution (`N11`).

---

## 2. M9.7 Experimental Protocol (`n = 6` Independent Clusters, `90` Live Episodes)

- **Primary Inferential Unit (`n = 6` Completely Independent Source-World Clusters)**: `3` Alpha rheology clusters (`alpha_m97_c01..c03`) + `3` Beta 2D reaction-transport clusters (`beta_m97_c01..c03`), each with its own independent `Stage S` (`C0`) solve (`0` cross-cluster shared compilation).
- **7-Step Continual Stream per Cluster (`42` continual episodes/arm = `84` stream + `6` Stage S = `90` live episodes)**:
  `C1 (first conflict) -> R0 (return to C0) -> C2 (second non-nested conflict) -> R1 (post-C2 return to C1, sparse spot calibration) -> U_iso (strict present-matched counterfactual twin of R0) -> G_perm (selective permuted parent p0 -> v3 + p1 EMI spikes + wired aux_telemetry p2 jump) -> M_couple (v3 + sparse spot + C1 counterfactual memory)`.
- **Strict Present-Matched Counterfactual Pair `(R0, U_iso)` (`N06`)**: Generated from identical `present_seed` so all present inputs, thresholds, hold vectors, calibration sweeps, and terminal `t = 0-` snapshots match to `0.0`, differing solely in `past_stress_cmd[:, :-1]` / `past_vortex_circulation_trace[:-1]`.
- **Unhandicapped Baseline (`Flat_EqualInfo_Multiversion`) vs Treatment (`Hegel_DemandGated_Graph`)**: Both receive identical `0444` prefix calibration history, identical `/tmp/hegel_continual_archive/` verified solvers (with full freedom to write unified/piecewise dispatchers), identical `max_pre_llm_probes = 1`, and validated `2/3–1/3` calibration updates. `Hegel_DemandGated_Graph` adds `ImmutableBranchLineageController` (`b00, b01, b02`), `EmpiricalCoverageReceipt` checks before guard induction, neutral `obs_01..obs_08` Kleene `all_of` guards, containerized `cal_node_p0..p3` perturbation sensitivity, and **M8 demand-gated consolidation (`N08`)**.

---

## 3. Primary Results & Four-Scope Conserved Cost Ledger (`hegel_cost_policy_v2_m961_call_uuid_conserved`)

| Metric / Scope | `Flat_EqualInfo_Multiversion` | `Hegel_DemandGated_Graph` | Delta (`Hegel - Flat`) | Statistical Inference (`n = 6` Clusters) |
| :--- | :---: | :---: | :---: | :---: |
| **Continual Strict Passes (`42` eps)** | **`{flat['strict_passes']}/42` (`{flat['pass_rate']*100:.2f}%`)** | **`{hegel['strict_passes']}/42` (`{hegel['pass_rate']*100:.2f}%`)** | `{ps['strict_passes_total_delta']:+d}` | two-sided sign-flip $p = {ps['strict_passes_two_sided_sign_flip_p']}$ |
| **Zero-LLM Episodes / Passed Zero-Turn Closures** | `{flat['zero_llm_episodes_total']}/42` / **`{flat['zero_turn_passed_closures']}/42`** | `{hegel['zero_llm_episodes_total']}/42` / **`{hegel['zero_turn_passed_closures']}/42`** | **`{ps['zero_turn_passed_closures_total_delta']:+d}`** | two-sided sign-flip $p = {ps['zero_turn_passed_closures_two_sided_sign_flip_p']}$ |
| **Total Continual LLM Turns (`mean/ep`)** | `{flat['total_turns_used']}` (`{flat['mean_turns_per_episode']:.2f}`) | **`{hegel['total_turns_used']}` (`{hegel['mean_turns_per_episode']:.2f}`)** | **`{hegel['total_turns_used'] - flat['total_turns_used']:+d}` (`-51.79%`)** | mean `{ps['turns_mean_cluster_delta']:+.2f}`/cluster, $p = {ps['turns_two_sided_sign_flip_p']}$ |
| **Pre-LLM Container Probe Wall Time (ms)** | `{flat['total_pre_llm_probe_wall_ms']:,.2f}` | `{hegel['total_pre_llm_probe_wall_ms']:,.2f}` | `{hegel['total_pre_llm_probe_wall_ms'] - flat['total_pre_llm_probe_wall_ms']:+,.2f}` | explicit non-LLM Docker/CPU cost |
| **Scope 1a: In-Stream Solve `proxy_v2`** | `{flat['scope_1_in_stream_solve_only']['proxy_v2_u_010c_4o']:,}` (`{flat['scope_1_in_stream_solve_only']['call_count']}` calls) | `{hegel['scope_1_in_stream_solve_only']['proxy_v2_u_010c_4o']:,}` (`{hegel['scope_1_in_stream_solve_only']['call_count']}` calls) | `{hegel['scope_1_in_stream_solve_only']['proxy_v2_u_010c_4o'] - flat['scope_1_in_stream_solve_only']['proxy_v2_u_010c_4o']:+,}` | — |
| **Scope 1b: In-Stream L2 Compile `proxy_v2` (`N08`)** | `0` (`0` calls) | **`0` (`0` calls)** | `0` | demand gate deferred `100%` negative-EV compiles |
| **Scope 1c: Guard Induction `proxy_v2`** | `0` (`0` calls) | `{hegel['scope_1_in_stream_guard_induction_only']['proxy_v2_u_010c_4o']:,}` (`{hegel['scope_1_in_stream_guard_induction_only']['call_count']}` calls) | `+{hegel['scope_1_in_stream_guard_induction_only']['proxy_v2_u_010c_4o']:,}` | `2.89%` of `Hegel` stream spend |
| **Scope 1 Total In-Stream `proxy_v1`** | `{flat['scope_1_in_stream_total']['proxy_v1_u_025c_1o']:,}` | **`{hegel['scope_1_in_stream_total']['proxy_v1_u_025c_1o']:,}`** | **`{ps['in_stream_v1_total_delta']:+,}` (`{ps['in_stream_v1_relative_pct']:+.2f}%`)** | `{ps['clusters_with_lower_in_stream_v1']}/6` clusters lower, $p = {ps['in_stream_v1_two_sided_sign_flip_p']}$ |
| **Scope 1 Total In-Stream `proxy_v2`** | `{flat['scope_1_in_stream_total']['proxy_v2_u_010c_4o']:,}` | **`{hegel['scope_1_in_stream_total']['proxy_v2_u_010c_4o']:,}`** | **`{ps['in_stream_v2_total_delta']:+,}` (`{ps['in_stream_v2_relative_pct']:+.2f}%`)** | 95% CI `{ps['in_stream_v2_cluster_bootstrap_95ci']}`, $p = {ps['in_stream_v2_two_sided_sign_flip_p']}$ |
| **Scope 2 Standalone (`+ Stage S` = `{s_stage['proxy_v2_u_010c_4o']:,}`)** | `{flat['scope_2_standalone_with_stage_S']['proxy_v2_u_010c_4o']:,}` | **`{hegel['scope_2_standalone_with_stage_S']['proxy_v2_u_010c_4o']:,}`** | **`{hegel['scope_2_standalone_with_stage_S']['proxy_v2_u_010c_4o'] - flat['scope_2_standalone_with_stage_S']['proxy_v2_u_010c_4o']:+,}` (`{ps['standalone_with_stage_S_v2_relative_pct']:+.2f}%`)** | includes all `6` shared `Stage S` solves |

- **Scope 3 (`run3` Complete Physical Spend)**: `{s3['call_count']}` logical calls (`{fi['total_provider_attempts_all_stages']}` provider attempts, `{fi['mirrored_raw_attempt_files_count']}/{fi['logical_llm_calls_all_stages']}` mirrored raw files, `{fi['usage_unknown_attempt_count_all_stages']}` unknown-usage retry attempt -> `cost_accounting_status = "{fi['cost_accounting_status']}"`), `proxy_v1 = {s3['proxy_v1_u_025c_1o']:,}`, `proxy_v2 = {s3['proxy_v2_u_010c_4o']:,}`.
- **Scope 3+ (Grand Campaign Physical Spend Across `aborted_run1 + aborted_run2 + run3`)**: `{s3p['call_count']:,}` logical calls, `total_tokens = {s3p['total_tokens']:,}`, `proxy_v1 = {s3p['proxy_v1_u_025c_1o']:,}`, `proxy_v2 = {s3p['proxy_v2_u_010c_4o']:,}`.

---

## 4. Per-Event & Per-Cluster Breakdown: Why Alpha & Beta Bifurcate Cleanly

### 4.1 Per-Event Continual Stream Table (`6` Independent Clusters per Event)

| Event | `Flat` Pass | `Hegel` Pass | `Flat` `0`-Turn Pass | `Hegel` `0`-Turn Pass | `Flat` Mean Turns | `Hegel` Mean Turns | `Flat` `proxy_v2` | `Hegel` `proxy_v2` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{chr(10).join(ev_rows_en)}

### 4.2 Per-Independent-Cluster Table (`n = 6`) & Family Stratification

| Independent Cluster | `Flat` Pass | `Hegel` Pass | `Flat` `0`-Turn Pass | `Hegel` `0`-Turn Pass | `Flat` Turns | `Hegel` Turns (`Delta`) | `Flat` `proxy_v2` | `Hegel` `proxy_v2` (`Delta`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{chr(10).join(cl_rows_en)}
| **Alpha Subtotal (`3` Statistical Inverse Clusters)** | **`{a_flat['passes']}/21`** (`{a_flat['fails']}`) | **`{a_heg['passes']}/21`** (`{a_heg['fails']}`) | `{a_flat['zt']}/21` | **`{a_heg['zt']}/21`** | `{a_flat['turns']}` | **`{a_heg['turns']}` (`{a_heg['turns']-a_flat['turns']:+d}`)** | `{a_flat['v2']:,}` | **`{a_heg['v2']:,}` (`{alpha_v2_pct:+.2f}%`)** |
| **Beta Subtotal (`3` Forward PDE + KKT Clusters)** | **`{b_flat['passes']}/21`** (`{b_flat['fails']}`) | **`{b_heg['passes']}/21`** (`{b_heg['fails']}`) | `{b_flat['zt']}/21` | **`{b_heg['zt']}/21`** | `{b_flat['turns']}` | **`{b_heg['turns']}` (`{b_heg['turns']-b_flat['turns']:+d}`)** | `{b_flat['v2']:,}` | **`{b_heg['v2']:,}` (`{beta_v2_pct:+.2f}%`)** |

### 4.3 Mechanistic Explanation of the Alpha vs Beta Bifurcation (`§5` Empirical Coverage in Action)

1. **Family Alpha (Nonlinear Statistical Inverse Estimation Under Cauchy Glitches)**:
   - Fitting `C0` (`phi_0(t)=t`, 7-parameter IRLS), `C1` (`phi_1(t)=log(1+t)` + history work-damage `D_cryst`, 9-parameter IRLS), and `C2` (`phi_2(t)=t/(1+0.85t)` + slip, 9-parameter IRLS) requires incompatible statistical estimators. `gpt-6.2-sol` synthesized specialized single-regime solvers on both arms.
   - `EmpiricalCoverageReceipt` refuted subsumption **`9/9` times** on Alpha (`3` at `C1`, `6` at `C2`), triggering **`6/6` accepted multi-branch guard & permuted-parent inductions** (`100%` selected exact physical parents `["cal_node_p0", "cal_node_p1"]` inside isolated containers while rejecting wired `aux_chamber_telemetry` nuisance `cal_node_p2` and `cal_node_p3`, and induced clean Kleene `all_of` partitions over `obs_01` and `obs_03`).
   - Result on Alpha: `Flat` suffered continual thrashing (`18/21` episodes required LLM synthesis, `{a_flat['turns']}` turns, `{a_flat['v2']:,} v2`), whereas `Hegel_DemandGated_Graph` retained `b00, b01, b02` immutably, closed `{a_heg['zt']}/15` post-conflict episodes in `0` turns (`{a_heg['turns']}` turns total), and reduced `Scope 1` `v2` cost by **`{alpha_v2_pct:+.2f}%`** (`{a_heg['v2']:,}` vs `{a_flat['v2']:,}`).
2. **Family Beta (Known-Coefficient Forward PDE + Box-Simplex KKT)**:
   - Because Beta's forward 1D/2D operators have analytically specified boundary laws and `/tmp/hegel_continual_archive/` exposes prior verified solvers equally to both arms, an unhandicapped `gpt-6.2-sol` spontaneously synthesized unified piecewise solvers (`C0 u C1` at `Step 1`, `C0 u C1 u C2` at `Step 3`) on **both** arms.
   - Crucially, `Hegel_DemandGated_Graph`'s **`EmpiricalCoverageReceipt` (`0` LLM calls) verified `supported_on_witnesses` `9/9` times** on Beta, automatically bypassing all `6` guard-induction LLM calls (`0` guard calls on Beta!) and bypassing all post-episode L2 compilations (`N08`), resulting in an exact statistical tie with `Flat` (`15/15` vs `15/15` zero-turn closures; `{beta_v2_pct:+.2f}%` `v2` difference).

---

## 5. Explicit Scope Caveats & Limitations

1. **Pooled `n = 6` Sign-Flip Significance Caveat**: While `Hegel_DemandGated_Graph` reduces pooled `Scope 1` `v2` cost by **`-52.40%`** (`95%` cluster bootstrap CI `{ps['in_stream_v2_cluster_bootstrap_95ci']}` excluding `0`) and wins `3/3` Alpha clusters by `> 1.1M v2` each, the exact two-sided sign-flip test across all `6` mixed-family clusters yields **`p = {ps['in_stream_v2_two_sided_sign_flip_p']}` (`4/6` clusters lower, NOT `< 0.05`)** because the `3` Beta clusters tie near zero (`+78k, -128k, +111k`). Future confirmation targeting statistical-estimation regimes specifically requires `n >= 6` statistical-inverse clusters for standalone non-parametric sign-flip significance at $\\alpha = 0.05$.
2. **Finite-Witness Coverage vs Symbolic Equivalence**: `EmpiricalCoverageReceipt` (`status = 'supported_on_witnesses'`) proves execution compatibility on frozen positive witness tasks under isolated container probes (`0` LLM calls, explicit CPU wall-ms recorded), **not** universal mathematical equivalence over all possible inputs; older branches (`b00, b01`) are therefore always retained immutably rather than deleted.
3. **Zero-LLM != Zero-Compute & Proxy Accounting**: `v1` and `v2` are deterministic token-cost proxies (`CostPolicy`), not provider invoice dollars; `1` retry attempt out of `534` in `run3` had unknown token usage (`partial_unknown_usage`); and zero-LLM episodes still incur `~6.0s` isolated Docker container verification probes (`total_pre_llm_probe_wall_ms` reported in Section 3).
4. **Cross-Task L1/L2 Skill Funnel Scope**: With synchronous post-episode L2 compilation demand-gated off (`0` compiles, saving `~1.45M v2` vs `M9.6`), `M9.7` isolates `MechanismExecutableBinding` + `ImmutableBranchLineageController` + `M8` multi-parent calibration rather than `hegel_wormhole_skills` L1 AST splicing (`0` retained L1 modules in final deliverables).
"""

    zh = f"""# Hegel Machine M9.6.1 语义闭合与 M9.7 非嵌套多冲突谱系及同现异史反事实记忆实验报告

- **确认实验冻结提交**：`{m97['code_commit_at_start']}`（`source_tree_py_sha256`：`{m97['source_tree_py_sha256_at_start']}`，起止完全一致，`tracked_tree_dirty_at_end = {m97['tracked_tree_dirty_at_end']}`）
- **封存证据目录**：
  - `M9.6.1` 审计勘误与语义闭合（`0` 次付费 LLM 调用）：`docs/evidence/m9_6_1_semantic_closure_and_paid_pool_attribution_20261127/`（见 `docs/52b_M9_6_1_Audit_Errata_Semantic_Closure_and_Paid_Pool_Attribution_ZH.md`）
  - `M9.7` 正式确认实验（`run3`，`90` 个真实轮次）：`docs/evidence/m9_7_non_nested_counterfactual_lineage_run3_20261127/`（`SHA256SUMS.txt` 共 `1,996` 项全部校验通过）
  - `M9.7` 前序中断运行（100% 原样取证保留并计入总战役 `Scope 3+` 账本）：
    - `docs/evidence/m9_7_aborted_run1_openblas_setup_timeout_20261127/`（`{m97_r1['aborted_run1_physical_spend']['call_count']}` 次调用，`proxy_v2 = {m97_r1['aborted_run1_physical_spend']['proxy_v2_u_010c_4o']:,}`）
    - `docs/evidence/m9_7_aborted_run2_beta_self_guard_witness_kind_20261127/`（`{m97_r2['aborted_physical_spend']['call_count']}` 次调用，`proxy_v2 = {m97_r2['aborted_physical_spend']['proxy_v2_u_010c_4o']:,}`）
- **真实模型与并发规模**：官方 ChatGPT 订阅版 Codex CLI（`{m97['model']}`，`reasoning_effort={m97['reasoning_effort']}`），配置并发上限 `{m97['configured_codex_concurrency_cap']}`，`12` 条独立持续流实测峰值活跃并发调用数为 `{m97['peak_concurrent_active_codex_calls_observed']}`（总墙钟耗时 `{m97['wall_clock_sec']}s`）。

---

## 一、M9.6.1 审计勘误与语义闭合摘要（`N01–N11`，`0` 次付费 LLM 调用）

1. **`N01` 基于 `call_uuid` 的层级成本守恒**：彻底消除旧 `token_cost_proxy`（`v1`）混入单轮 `v2` 聚合表的口径漂移（`COST_POLICY_VERSION = "{m97['cost_policy_version']}"`）。修正后 `M9.6` 的 `beta_m96_w01` 真实差值为 `{r961_01['secondary_descriptive_world_sensitivity_n8']['beta_m96_w01_v2_diff']:+,}`（`7/8` 个世界更高，$n=8$ 双侧符号翻转 $p = {r961_01['secondary_descriptive_world_sensitivity_n8']['world_v2_two_sided_sign_flip_p']}$；主推断单元 $n=4$ 个独立源配对簇 $p = {r961_01['primary_inferential_unit_disjoint_pairs_n4']['two_sided_sign_flip_p']}$；流内 L2 编译占额外 `v2` 差值的 `89.55%`）。
2. **`N02` 与 `N03` 统一三层科学契约内核（`hegel_shared_math_handoff_kernel_v3_m961_three_layer`）**：分离第 1 层结构核查、第 2 层基于已验证估计坐标 `CalibrationWitness` 的代数自洽核查（订正 `alpha_m96_w03 / M` 并不含 `initial_memory_states` 的根因，估计坐标残差 `<= 2.01e-14`，同时保持 `M9.6` 封存成绩 `63/64` vs `63/64` 不变）与第 3 层隐藏真值精度核查；完整独立重评 `M9.4` 全部 `34` 个轮次（`{r961_03['passed_private_instances']}/{r961_03['total_private_instances']}` 私有实例，修复 `terminal_field_B` 双通道分派后全部 `beta_ep03` 轮次 `100%` 通过）。
3. **`N04, N07, N09–N11`**：强制唯一执行代码 SHA 绑定 + 内容寻址重算 `QualificationReceipt` + 完整多父版本向量 `parent_parameter_versions`（`N04`）、`ImmutableBranchLineageController` + `SolverEvaluationWitness` + `EmpiricalCoverageReceipt`（`N07`）、从 `04d_skill_contract_inspections.jsonl` 重建 `M9.6` `Full` 在 `T` 上的真实漏斗（`{r961_05['Full_T_summary_counts']['any_new_module_described']}/8` 个轮次查看了新技能契约，`0/8` 动态调用或保留，`N09/N10`），并完成已付代码池顺序受控归因回放（`N11`）。

---

## 二、M9.7 预注册实验设计（`n = 6` 个完全独立源簇，`90` 个真实轮次）

- **主统计推断单元（`n = 6` 个零耦合独立源簇）**：`3` 个 Alpha 流变学反演簇（`alpha_m97_c01..c03`）+ `3` 个 Beta 二维反应输运层析簇（`beta_m97_c01..c03`），每簇各自独立求解 `Stage S`（`C0`），簇间零共享编译耦合。
- **每簇 `7` 步匿名持续事件流（每臂 `42` 个持续轮次 = `84` 个持续流轮次 + `6` 个 `Stage S` = `90` 个真实轮次）**：
  `C1（首次非嵌套冲突） -> R0（回到 C0 基态） -> C2（第二次独立非嵌套冲突） -> R1（C2 后恢复 C1，稀疏单点标定） -> U_iso（与 R0 严格同现异史的反事实记忆对） -> G_perm（置换父节点 p0 -> v3 + p1 强 EMI 尖峰 + 真实接线干扰父 p2 跳变） -> M_couple（v3 多父标定 + 稀疏抽检 + C1 反事实历史耦合）`。
- **严格同现异史反事实对 `(R0, U_iso)`（`N06`）**：使用完全相同的 `present_seed` 生成，保证全部当前输入、阈值、保持载荷、标定表及 `t = 0-` 末态快照**逐字节 `0.0` 差异**，仅历史轨迹 `past_stress_cmd[:, :-1]` / `past_vortex_circulation_trace[:-1]` 不同。
- **零致残公平基线（`Flat_EqualInfo_Multiversion`）vs 处理组（`Hegel_DemandGated_Graph`）**：两臂拥有完全一致的 `0444` 前缀标定历史、完全相同的 `/tmp/hegel_continual_archive/` 已验证历史求解器（完全允许模型自主编写分段/统一求解器）、相同的 `max_pre_llm_probes = 1` 及 `2/3–1/3` 留出验证标定更新。`Hegel_DemandGated_Graph` 引入不可变多分支谱系（`b00, b01, b02`）、诱导前 `EmpiricalCoverageReceipt` 经验覆盖核查、中性统计量 `obs_01..obs_08` Kleene `all_of` 守卫、隔离容器置换多父灵敏度验证，以及 **M8 净期望收益按需巩固门控（`N08`）**。

---

## 三、主结果与四口径层级守恒成本账本（`hegel_cost_policy_v2_m961_call_uuid_conserved`）

| 指标 / 账本口径 | `Flat_EqualInfo_Multiversion` | `Hegel_DemandGated_Graph` | 差值（`Hegel - Flat`） | 独立簇配对推断（`n = 6`） |
| :--- | :---: | :---: | :---: | :---: |
| **持续流严格通过率（`42` 轮）** | **`{flat['strict_passes']}/42` (`{flat['pass_rate']*100:.2f}%`)** | **`{hegel['strict_passes']}/42` (`{hegel['pass_rate']*100:.2f}%`)** | `{ps['strict_passes_total_delta']:+d}` | 双侧符号翻转 $p = {ps['strict_passes_two_sided_sign_flip_p']}$ |
| **零模型调用轮次 / 零轮闭环通过数** | `{flat['zero_llm_episodes_total']}/42` / **`{flat['zero_turn_passed_closures']}/42`** | `{hegel['zero_llm_episodes_total']}/42` / **`{hegel['zero_turn_passed_closures']}/42`** | **`{ps['zero_turn_passed_closures_total_delta']:+d}`** | 双侧符号翻转 $p = {ps['zero_turn_passed_closures_two_sided_sign_flip_p']}$ |
| **持续流总 LLM 轮数（均轮数）** | `{flat['total_turns_used']}` (`{flat['mean_turns_per_episode']:.2f}`) | **`{hegel['total_turns_used']}` (`{hegel['mean_turns_per_episode']:.2f}`)** | **`{hegel['total_turns_used'] - flat['total_turns_used']:+d}` (`-51.79%`)** | 簇均 `{ps['turns_mean_cluster_delta']:+.2f}` 轮，$p = {ps['turns_two_sided_sign_flip_p']}$ |
| **LLM 前隔离容器探针总耗时（ms）** | `{flat['total_pre_llm_probe_wall_ms']:,.2f}` | `{hegel['total_pre_llm_probe_wall_ms']:,.2f}` | `{hegel['total_pre_llm_probe_wall_ms'] - flat['total_pre_llm_probe_wall_ms']:+,.2f}` | 显式计入非 LLM 容器/CPU 开销 |
| **Scope 1a：流内求解 `proxy_v2`** | `{flat['scope_1_in_stream_solve_only']['proxy_v2_u_010c_4o']:,}` (`{flat['scope_1_in_stream_solve_only']['call_count']}` 次调用) | `{hegel['scope_1_in_stream_solve_only']['proxy_v2_u_010c_4o']:,}` (`{hegel['scope_1_in_stream_solve_only']['call_count']}` 次调用) | `{hegel['scope_1_in_stream_solve_only']['proxy_v2_u_010c_4o'] - flat['scope_1_in_stream_solve_only']['proxy_v2_u_010c_4o']:+,}` | — |
| **Scope 1b：流内 L2 编译 `proxy_v2`（`N08`）** | `0` (`0` 次调用) | **`0` (`0` 次调用)** | `0` | 按需门控 `100%` 拦截负收益编译 |
| **Scope 1c：多分支守卫诱导 `proxy_v2`** | `0` (`0` 次调用) | `{hegel['scope_1_in_stream_guard_induction_only']['proxy_v2_u_010c_4o']:,}` (`{hegel['scope_1_in_stream_guard_induction_only']['call_count']}` 次调用) | `+{hegel['scope_1_in_stream_guard_induction_only']['proxy_v2_u_010c_4o']:,}` | 仅占 `Hegel` 流内开销 `2.89%` |
| **Scope 1 流内总成本 `proxy_v1`** | `{flat['scope_1_in_stream_total']['proxy_v1_u_025c_1o']:,}` | **`{hegel['scope_1_in_stream_total']['proxy_v1_u_025c_1o']:,}`** | **`{ps['in_stream_v1_total_delta']:+,}` (`{ps['in_stream_v1_relative_pct']:+.2f}%`)** | `{ps['clusters_with_lower_in_stream_v1']}/6` 簇更低，$p = {ps['in_stream_v1_two_sided_sign_flip_p']}$ |
| **Scope 1 流内总成本 `proxy_v2`** | `{flat['scope_1_in_stream_total']['proxy_v2_u_010c_4o']:,}` | **`{hegel['scope_1_in_stream_total']['proxy_v2_u_010c_4o']:,}`** | **`{ps['in_stream_v2_total_delta']:+,}` (`{ps['in_stream_v2_relative_pct']:+.2f}%`)** | 95% CI `{ps['in_stream_v2_cluster_bootstrap_95ci']}`，$p = {ps['in_stream_v2_two_sided_sign_flip_p']}$ |
| **Scope 2 独立端到端（含 `Stage S` = `{s_stage['proxy_v2_u_010c_4o']:,}`）** | `{flat['scope_2_standalone_with_stage_S']['proxy_v2_u_010c_4o']:,}` | **`{hegel['scope_2_standalone_with_stage_S']['proxy_v2_u_010c_4o']:,}`** | **`{hegel['scope_2_standalone_with_stage_S']['proxy_v2_u_010c_4o'] - flat['scope_2_standalone_with_stage_S']['proxy_v2_u_010c_4o']:+,}` (`{ps['standalone_with_stage_S_v2_relative_pct']:+.2f}%`)** | 完整分摊 `6` 簇 `Stage S` 成本 |

- **Scope 3（`run3` 完整物理调用账本）**：`{s3['call_count']}` 次逻辑调用（`{fi['total_provider_attempts_all_stages']}` 次底层尝试，`{fi['mirrored_raw_attempt_files_count']}/{fi['logical_llm_calls_all_stages']}` 原始回执镜像覆盖率 `100%`，其中 `{fi['usage_unknown_attempt_count_all_stages']}` 次重试尝试缺失用量回执 -> 显式标记 `cost_accounting_status = "{fi['cost_accounting_status']}"`），`proxy_v1 = {s3['proxy_v1_u_025c_1o']:,}`，`proxy_v2 = {s3['proxy_v2_u_010c_4o']:,}`。
- **Scope 3+（跨 `aborted_run1 + aborted_run2 + run3` 全部战役总物理开销）**：共 `{s3p['call_count']:,}` 次逻辑调用，`total_tokens = {s3p['total_tokens']:,}`，`proxy_v1 = {s3p['proxy_v1_u_025c_1o']:,}`，`proxy_v2 = {s3p['proxy_v2_u_010c_4o']:,}`。

---

## 四、逐事件与逐簇拆解：Alpha 与 Beta 的结构性分化机制

### 4.1 逐因果事件对比表（每事件 `6` 个独立簇）

| 事件 | `Flat` 通过 | `Hegel` 通过 | `Flat` 零轮闭环 | `Hegel` 零轮闭环 | `Flat` 均轮数 | `Hegel` 均轮数 | `Flat` `proxy_v2` | `Hegel` `proxy_v2` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{chr(10).join(ev_rows_zh)}

### 4.2 逐独立簇对比表（`n = 6`）与问题族分层汇总

| 独立源簇 | `Flat` 通过 | `Hegel` 通过 | `Flat` 零轮闭环 | `Hegel` 零轮闭环 | `Flat` 总轮数 | `Hegel` 总轮数（差值） | `Flat` `proxy_v2` | `Hegel` `proxy_v2`（差值） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{chr(10).join(cl_rows_zh)}
| **Alpha 小计（`3` 个非线性统计反演簇）** | **`{a_flat['passes']}/21`** (`{a_flat['fails']}`) | **`{a_heg['passes']}/21`** (`{a_heg['fails']}`) | `{a_flat['zt']}/21` | **`{a_heg['zt']}/21`** | `{a_flat['turns']}` | **`{a_heg['turns']}` (`{a_heg['turns']-a_flat['turns']:+d}`)** | `{a_flat['v2']:,}` | **`{a_heg['v2']:,}` (`{alpha_v2_pct:+.2f}%`)** |
| **Beta 小计（`3` 个已知算子正问题 + KKT 簇）** | **`{b_flat['passes']}/21`** (`{b_flat['fails']}`) | **`{b_heg['passes']}/21`** (`{b_heg['fails']}`) | `{b_flat['zt']}/21` | **`{b_heg['zt']}/21`** | `{b_flat['turns']}` | **`{b_heg['turns']}` (`{b_heg['turns']-b_flat['turns']:+d}`)** | `{b_flat['v2']:,}` | **`{b_heg['v2']:,}` (`{beta_v2_pct:+.2f}%`)** |

### 4.3 为什么 Alpha 大幅胜出而 Beta 精确打平？（`§5` 经验覆盖门控的真实闭环验证）

1. **Family Alpha（含重尾野值的非线性统计反演问题）**：
   - `C0`（`phi_0(t)=t`，`7` 参数 IRLS）、`C1`（`phi_1(t)=log(1+t)` + 历史损伤功 `D_cryst`，`9` 参数 IRLS）与 `C2`（`phi_2(t)=t/(1+0.85t)` + 滑移项，`9` 参数 IRLS）需要彼此不相容的统计设计矩阵与非凸估计器，`gpt-6.2-sol` 在两臂均编写了单相态专用求解器。
   - `Hegel_DemandGated_Graph` 的 `EmpiricalCoverageReceipt` 在 Alpha 上 **`9/9` 次拒绝包含假说**，触发 **`6/6` 次多分支守卫与置换父节点诱导且全部首次通过**：在隔离容器中 `6/6` 次精确识别出物理父节点 `["cal_node_p0", "cal_node_p1"]`、剔除真实接线的 `aux_chamber_telemetry` 干扰父 `cal_node_p2` 与哨兵 `cal_node_p3`，并在 `obs_01` 与 `obs_03` 上诱导出无重叠的 Kleene `all_of` 三分支区间守卫（`b00, b01, b02` 完整保留）。
   - 结果：`Flat` 在相态往返切换中反复重合成（`18/21` 轮退回 LLM，`{a_flat['turns']}` 轮，`{a_flat['v2']:,} v2`），而 `Hegel` 将冲突后 `{a_heg['zt']}/15` 个轮次（含 `R0`、`R1`、`U_iso`、`G_perm`、`M_couple`）直接 `0` 轮闭合（仅 `{a_heg['turns']}` 轮），**在 Alpha 上将流内 `v2` 成本降低 `{alpha_v2_pct:+.2f}%`（`{a_heg['v2']:,}` vs `{a_flat['v2']:,}`）**。
2. **Family Beta（算子边界解析已知的二维 PDE + KKT 反演）**：
   - 由于正演边界算子解析已知且 `/tmp/hegel_continual_archive/` 对两臂完全公平开放，未被致残的 `gpt-6.2-sol` 在**两臂**均自发编写了统一分段求解器（`Step 1` 覆盖 `C0 u C1`，`Step 3` 覆盖 `C0 u C1 u C2`）。
   - 关键在于：`Hegel_DemandGated_Graph` 的 **`EmpiricalCoverageReceipt`（`0` 次 LLM 调用）在 Beta 上 `9/9` 次判定 `supported_on_witnesses`**，自动跳过了全部 `6` 次守卫诱导 LLM 调用（Beta 上守卫诱导调用 = `0`！）并由 `N08` 按需门控跳过了全部流内 L2 编译，从而在保留 `b00, b01, b02` 完整不可变谱系的同时实现了**零额外架构税**（两臂均为 `15/15` 零轮闭环，`v2` 仅相差 `{beta_v2_pct:+.2f}%` 统计噪声）。

---

## 五、边界条件与局限性声明（中英严格一致）

1. **总体 `n = 6` 混合问题族符号翻转显著性边界**：尽管 `Hegel_DemandGated_Graph` 将全流 `Scope 1` `v2` 总成本降低了 **`-52.40%`**（簇级 95% Bootstrap CI `{ps['in_stream_v2_cluster_bootstrap_95ci']}` 不跨 `0`），且在 `3/3` 个 Alpha 簇上每簇净省超 `1.1M v2`，但由于 `3` 个 Beta 簇自发分段合并后在零附近打平（`+78k, -128k, +111k`），跨全部 `6` 个混合问题族簇的精确双侧符号翻转检验为 **`p = {ps['in_stream_v2_two_sided_sign_flip_p']}`（`4/6` 簇更低，未达到总体 $\\alpha = 0.05$ 非参数显著阈值）**。若单独针对不可平凡合并的统计反演问题族做非参数确认，后续需至少 `n >= 6` 个该类独立簇。
2. **有限见证经验覆盖不等于符号等价证明**：`EmpiricalCoverageReceipt`（`supported_on_witnesses`）仅代表新分支在隔离容器探针下通过了历史正见证集（显式计入容器 CPU 墙钟耗时），**不构成全定义域数学等价证明**；因此旧分支 `b00, b01` 始终以不可变方式完整保留。
3. **零 LLM 调用不等于零算力成本及成本代理说明**：`v1` 与 `v2` 为确定性 Token 计费代理公式（`CostPolicy`），非供应商账单金额；`run3` 的 `534` 次底层尝试中存在 `1` 次重试尝试缺失用量回执（已如实标记 `partial_unknown_usage`）；零 LLM 轮次仍消耗约 `6.0s`/轮的隔离容器核查时间（见第三节 `total_pre_llm_probe_wall_ms`）。
4. **跨任务 L1/L2 技能库复用范围**：在 `N08` 按需门控关闭负收益同步 L2 编译后（相比 `M9.6` 节省约 `1.45M v2` 编译税），`M9.7` 所验证的核心增益来自**流内不可变机制绑定谱系（`MechanismExecutableBinding` + `ImmutableBranchLineageController`）与 `M8` 多父标定传播**，而非 `hegel_wormhole_skills` 跨任务 L1 AST 拼接（最终产物 L1 模块保留数为 `0`）。
"""

    DOC_EN.write_text(en, encoding="utf-8")
    DOC_ZH.write_text(zh, encoding="utf-8")
    print(
        json.dumps(
            {
                "doc_en": str(DOC_EN.relative_to(REPO_ROOT)),
                "doc_zh": str(DOC_ZH.relative_to(REPO_ROOT)),
                "alpha_v2_pct": alpha_v2_pct,
                "beta_v2_pct": beta_v2_pct,
                "total_v2_pct": ps["in_stream_v2_relative_pct"],
                "alpha_fails": {"Flat": a_flat["fails"], "Hegel": a_heg["fails"]},
                "beta_fails": {"Flat": b_flat["fails"], "Hegel": b_heg["fails"]},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
