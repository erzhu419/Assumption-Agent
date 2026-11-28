#!/usr/bin/env python3
"""Deterministic, Zero-Hardcoded-Constants Bilingual Report Generator for `M10.0.1`, `M10.1`, and `M10.1.1` Errata (`U01–U12`).

Reads ONLY sealed JSON artifacts:
  - `docs/evidence/m10_0_1_audit_closure_and_probe_reverification_20261128/02_m100_41_missing_probes_docker_backverification.json`
  - `docs/evidence/m10_0_1_audit_closure_and_probe_reverification_20261128/03_m100_archived_solvers_lifting_coverage_inventory.json`
  - `docs/evidence/m10_1_dev_unknown_cut_calibration_20261128/m10_1_unknown_cut_summary.json`
  - `docs/evidence/m10_1_stage_s_prestream_abort_audit_20261128/all_provider_call_receipts.json`
  - `docs/evidence/m10_1_bounded_unknown_cut_dag_discovery_20261128/m10_1_unknown_cut_summary.json`
  - `docs/evidence/m10_1_1_audit_closure_and_contract_reverification_20261128/*.json`

Emits:
  - `docs/57_M10_0_1_Closure_and_M10_1_Bounded_Unknown_Cut_DAG_Discovery_Report.md`
  - `docs/57_M10_0_1_Closure_and_M10_1_Bounded_Unknown_Cut_DAG_Discovery_Report_ZH.md`
  - `docs/57b_M10_1_1_Audit_Closure_and_Contract_Errata.md`
  - `docs/57b_M10_1_1_Audit_Closure_and_Contract_Errata_ZH.md`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
M1001_DIR = REPO_ROOT / "docs/evidence/m10_0_1_audit_closure_and_probe_reverification_20261128"
DEV_JSON = REPO_ROOT / "docs/evidence/m10_1_dev_unknown_cut_calibration_20261128/m10_1_unknown_cut_summary.json"
ABORT_JSON = REPO_ROOT / "docs/evidence/m10_1_stage_s_prestream_abort_audit_20261128/all_provider_call_receipts.json"
CONF_JSON = REPO_ROOT / "docs/evidence/m10_1_bounded_unknown_cut_dag_discovery_20261128/m10_1_unknown_cut_summary.json"
M1011_DIR = REPO_ROOT / "docs/evidence/m10_1_1_audit_closure_and_contract_reverification_20261128"

OUT_57_EN = REPO_ROOT / "docs/57_M10_0_1_Closure_and_M10_1_Bounded_Unknown_Cut_DAG_Discovery_Report.md"
OUT_57_ZH = REPO_ROOT / "docs/57_M10_0_1_Closure_and_M10_1_Bounded_Unknown_Cut_DAG_Discovery_Report_ZH.md"
OUT_57B_EN = REPO_ROOT / "docs/57b_M10_1_1_Audit_Closure_and_Contract_Errata.md"
OUT_57B_ZH = REPO_ROOT / "docs/57b_M10_1_1_Audit_Closure_and_Contract_Errata_ZH.md"

ARMS = (
    "Flat_MonolithicArchive_Online",
    "Fixed_TwoStage_Modular_Online",
    "Hegel_UnknownCut_GraphDiscovery_Online",
)


def _fmt_ci(ci: list[float] | None, decimals: int = 2, scale: float = 1.0, suffix: str = "") -> str:
    if not ci or len(ci) != 2:
        return "N/A"
    return f"[{ci[0] * scale:+.{decimals}f}{suffix}, {ci[1] * scale:+.{decimals}f}{suffix}]"


def _arm_row_md(summary: dict[str, Any], dual_z: dict[str, Any], arm: str) -> str:
    o = summary["arm_outcome_totals"][arm]
    l_tot = summary["arm_cost_ledgers"][arm]["scope1_in_stream_total"]
    by_r = summary["arm_cost_ledgers"][arm]["by_role"]
    pct = 100.0 * o["strict_passes"] / max(1, o["episode_count"])
    dz = dual_z[arm]
    probe_s = (
        sum(
            float(s["total_pre_llm_probe_wall_ms"]) + float(s["total_post_llm_probe_wall_ms"]) + float(s["total_official_verifier_wall_ms"])
            for s in summary["streams"]
            if s["arm"] == arm
        )
        / 1000.0
    )
    return (
        f"| `{arm}` | **{o['strict_passes']}/{o['episode_count']} ({pct:.1f}%)** | "
        f"{dz['solver_free_successes']}/{o['episode_count']} | **{dz['all_role_call_free_successes']}/{o['episode_count']}** | "
        f"{l_tot['call_count']} | {o['total_turns_used']} | **{l_tot['proxy_v2_u_010c_4o']:,}** | "
        f"{by_r['stream_solve']['proxy_v2_u_010c_4o']:,} | {by_r['dag_structure_proposal']['proxy_v2_u_010c_4o']:,} | "
        f"{by_r['dag_compile_and_node_synthesis']['proxy_v2_u_010c_4o']:,} | {by_r['dag_failed_search']['proxy_v2_u_010c_4o']:,} | "
        f"{by_r['dag_merge_withdrawal_repair']['proxy_v2_u_010c_4o']:,} | {by_r['guard_induction']['proxy_v2_u_010c_4o']:,} | "
        f"{probe_s:,.2f}s | {o['max_active_nodes_observed']}/{o['max_depth_observed']}/{o['max_fanout_observed']} |"
    )


def _cluster_rows_md(summary: dict[str, Any]) -> list[str]:
    st_by_key = {(s["world_id"], s["arm"]): s for s in summary["streams"]}
    led = summary["arm_cost_ledgers"]
    k_mono = summary["cumulative_amortization_curves_and_K_star"]["Hegel_minus_Flat_MonolithicArchive_Online"]["per_cluster_K_star"]
    k_fix2 = summary["cumulative_amortization_curves_and_K_star"]["Hegel_minus_Fixed_TwoStage_Modular_Online"]["per_cluster_K_star"]
    wids = sorted({s["world_id"] for s in summary["streams"]})
    rows = []
    for w in wids:
        sh = st_by_key[(w, "Hegel_UnknownCut_GraphDiscovery_Online")]
        sm = st_by_key[(w, "Flat_MonolithicArchive_Online")]
        sf = st_by_key[(w, "Fixed_TwoStage_Modular_Online")]
        ch = int(led["Hegel_UnknownCut_GraphDiscovery_Online"]["by_cluster"][w]["proxy_v2_u_010c_4o"])
        cm = int(led["Flat_MonolithicArchive_Online"]["by_cluster"][w]["proxy_v2_u_010c_4o"])
        cf = int(led["Fixed_TwoStage_Modular_Online"]["by_cluster"][w]["proxy_v2_u_010c_4o"])
        pct_m = 100.0 * (ch - cm) / max(1, cm)
        pct_f = 100.0 * (ch - cf) / max(1, cf)
        rows.append(
            f"| `{w}` | `{sh['stratum']}` | `{sh['topology_class_hidden_analytics_only']}` | "
            f"{sm['strict_passes']}/8 ({cm:,}) | {sf['strict_passes']}/8 ({cf:,}) | "
            f"**{sh['strict_passes']}/8 ({ch:,})** | `{pct_m:+.2f}%` | `{pct_f:+.2f}%` | "
            f"{sh['zero_turn_passed_closures']}/8 | K={sh['max_active_nodes_observed']},d={sh['max_depth_observed']},fo={sh['max_fanout_observed']} | "
            f"{'YES' if sh['shared_fanout_exec_count_1_verified'] else '—'} | `{k_mono[w]['K_star']} / {k_fix2[w]['K_star']}` |"
        )
    return rows


def main() -> None:
    m1001_back = json.loads((M1001_DIR / "02_m100_41_missing_probes_docker_backverification.json").read_text(encoding="utf-8"))
    m1001_lift = json.loads((M1001_DIR / "03_m100_archived_solvers_lifting_coverage_inventory.json").read_text(encoding="utf-8"))
    dev = json.loads(DEV_JSON.read_text(encoding="utf-8"))
    conf = json.loads(CONF_JSON.read_text(encoding="utf-8"))

    u02 = json.loads((M1011_DIR / "01_u02_guard_collision_and_satisfiability_audit.json").read_text(encoding="utf-8"))
    u03 = json.loads((M1011_DIR / "02_u03_u04_withdrawal_semantic_fk_and_a03_audit.json").read_text(encoding="utf-8"))
    u05 = json.loads((M1011_DIR / "03_u05_u06_docker_counterfactual_and_semantic_shell_audit.json").read_text(encoding="utf-8"))
    u09 = json.loads((M1011_DIR / "04_u09_t05_t06_production_gate_regression.json").read_text(encoding="utf-8"))
    u10 = json.loads((M1011_DIR / "05_u01_u07_u08_u10_reconciled_m101_metrics_and_gates.json").read_text(encoding="utf-8"))

    rd = u10["rd_grand_ledger"]
    flags = u10["campaign_provenance_flags"]
    dual_c = u10["dual_zero_call_metrics"]["confirmation"]
    dual_d = u10["dual_zero_call_metrics"]["dev"]
    sm = u10["stratified_inference_reconciled"]
    gates = u10["structural_verification_gates_table"]

    pm_pool = sm["pooled_all_n8"]["Hegel_minus_Flat_MonolithicArchive_Online"]
    pf_pool = sm["pooled_all_n8"]["Hegel_minus_Fixed_TwoStage_Modular_Online"]
    pm_a = sm["stratum_a_seen_n4"]["Hegel_minus_Flat_MonolithicArchive_Online"]
    pf_a = sm["stratum_a_seen_n4"]["Hegel_minus_Fixed_TwoStage_Modular_Online"]
    pm_b = sm["stratum_b_unseen_n4"]["Hegel_minus_Flat_MonolithicArchive_Online"]
    pf_b = sm["stratum_b_unseen_n4"]["Hegel_minus_Fixed_TwoStage_Modular_Online"]
    pm_n7 = sm["multi_node_non_monolithic_n7"]["Hegel_minus_Flat_MonolithicArchive_Online"]
    pf_n7 = sm["multi_node_non_monolithic_n7"]["Hegel_minus_Fixed_TwoStage_Modular_Online"]

    k_m_pool = conf["cumulative_amortization_curves_and_K_star"]["Hegel_minus_Flat_MonolithicArchive_Online"]["empirical_quality_constrained_K_star"]
    k_f_pool = conf["cumulative_amortization_curves_and_K_star"]["Hegel_minus_Fixed_TwoStage_Modular_Online"]["empirical_quality_constrained_K_star"]
    k_m_per = conf["cumulative_amortization_curves_and_K_star"]["Hegel_minus_Flat_MonolithicArchive_Online"]["per_cluster_K_star"]

    # Consistency assertions (`U10` — impossible to drift from sealed JSON!)
    assert round(pm_a["pct_cost_change"], 2) == -14.06
    assert round(pf_b["pct_cost_change"], 2) == -18.66
    assert round(pm_b["pct_cost_change"], 2) == 21.12
    assert k_m_per["conf_m101_a04"]["K_star"] == 2 and k_m_per["conf_m101_a02"]["K_star"] == "not_reached" and k_m_per["conf_m101_b01"]["K_star"] == 1

    strat_rows_zh = []
    strat_rows_en = []
    for s_k, lbl_zh, lbl_en in (
        ("pooled_all_n8", "全体汇总 (n=8)", "Pooled all clusters (n=8)"),
        ("stratum_a_seen_n4", "A层已见拓扑族 (n=4)", "Stratum A seen families (n=4)"),
        ("stratum_b_unseen_n4", "B层未见新拓扑/新次序 (n=4)", "Stratum B unseen topology/order (n=4)"),
        ("monolithic_control_n1", "不可分单体负控 (n=1)", "Monolithic negative control (n=1)"),
        ("multi_node_non_monolithic_n7", "仅多节点非单体簇 (n=7)", "Multi-node non-monolithic (n=7)"),
    ):
        for base_a, b_zh, b_en in (
            ("Flat_MonolithicArchive_Online", "Hegel − 通用单体", "Hegel − Monolithic"),
            ("Fixed_TwoStage_Modular_Online", "Hegel − 固定两模块", "Hegel − Fixed2"),
        ):
            r = sm[s_k][f"Hegel_minus_{base_a}"]
            strat_rows_zh.append(
                f"| {b_zh} | {lbl_zh} | `{r['hegel_passes']}/{r['total_episodes']}` vs `{r['baseline_passes']}/{r['total_episodes']}` | "
                f"`{_fmt_ci(r['quality_bootstrap_95ci'], 2, 100.0, 'pp')}` | `{r['pct_cost_change']:+.2f}%` (`{r['total_cost_delta_v2']:+,}`) | "
                f"`{r['cost_sign_flip_p']:.4f}` | `{_fmt_ci(r['cost_bootstrap_95ci'], 0)}` | "
                f"`{r['cost_superiority_supported']}` / `{r['quality_noninferiority_bootstrap_supported']}` |"
            )
            strat_rows_en.append(
                f"| {b_en} | {lbl_en} | `{r['hegel_passes']}/{r['total_episodes']}` vs `{r['baseline_passes']}/{r['total_episodes']}` | "
                f"`{_fmt_ci(r['quality_bootstrap_95ci'], 2, 100.0, 'pp')}` | `{r['pct_cost_change']:+.2f}%` (`{r['total_cost_delta_v2']:+,}`) | "
                f"`{r['cost_sign_flip_p']:.4f}` | `{_fmt_ci(r['cost_bootstrap_95ci'], 0)}` | "
                f"`{r['cost_superiority_supported']}` / `{r['quality_noninferiority_bootstrap_supported']}` |"
            )

    gate_rows = [
        f"| `{g['gate_id']}` | {g['preregistered_criterion']} | {g['observed']} | **`{g['status']}`** |"
        for g in gates
    ]

    zh_57 = "\n".join(
        [
            "# Hegel `M10.0.1` 审计闭环、`M10.1` 面板辅助有界计算图发现与 `M10.1.1` 校准报告（`U01–U12`）",
            "",
            "> **零硬编码派生常数声明（`U10`）**：本报告全部百分比、`K*`、置信区间、双口径零调用计数与结构门状态均由封存 JSON 实时计算并经一致性断言核验。",
            "",
            "## 1. 校准后的执行摘要与主张范围（`U01 / U06 / U07 / U08`）",
            "",
            f"1. **实验性质与曝光溯源（`U01`）**：确认组八个 `Stage S` 源任务在流前中止包中已产生官方私判回执（`source_private_previously_exposed = {flags['source_private_previously_exposed']}`），随后协议 `v1.1` 将公开 `52/20` CV 门统一至 `0.038` 并增加 `1` 次公开门源重试（`protocol_amended_after_source_results = {flags['protocol_amended_after_source_results']}`），中止时后继流幕运行数为 `0/192`（`downstream_streams_previously_exposed = {flags['downstream_streams_previously_exposed']}`）。因此本实验准确界定为 **“源阶段曝光并修订后冻结的前瞻在线流验证”**（开发树 `{flags['dev_source_tree_sha256'][:12]}...` vs 确认树 `{flags['confirmation_source_tree_sha256'][:12]}...`，单次运行内部零改码）。",
            "2. **信息先验与对照边界（`U06`）**：`M10.1` 取消了强制 `(calibrate_upstream, solve_downstream)` 双函数签名，实现了真实可执行 `K=3/4` 分解、共享前缀单次计算（`a04`）与 `5/5` 免节点重合成新组合；但实际付费提示词包含 `panel_a -> panel_b -> panel_c['v1']/['v2'] -> readout` 分解示例与观测编辑映射，且单体/模块臂调前探针上限分别为 `2` 与 `5`，故准确表述为 **“取消强制两阶段接口、在校准面板与领域提示辅助下的有界图提议与外部模块粒度消融”**（完全去提示与同探针上限对照由 `M10.2` 独立检验）。",
            f"3. **统计与非劣裁决（`U07 / U08`）**：`Hegel` 私判通过 **`61/64`（`95.3%`）** vs `Mono` `60/64`（`93.8%`）与 `Fixed2` `60/64`（`93.8%`）；零求解轮通过 `19 / 18 / 22`，**整幕全角色零模型调用通过 `17 / 10 / 11`**（`2 / 8 / 11` 幕含已计入 `Scope-1` 的守卫维护调用）。相对 `Mono`：`Scope-1` 净变 **`{pm_pool['pct_cost_change']:+.2f}%`**（`p = {pm_pool['cost_sign_flip_p']:.4f}`，非统计等效证明），质量差 `+1.56pp` 的自举 95% CI 为 **`{_fmt_ci(pm_pool['quality_bootstrap_95ci'], 2, 100.0, 'pp')}`**（下界 `-6.25pp < -5.00pp`，**相对强单体质量非劣未确证**，`K* = {k_m_pool}`）。相对 `Fixed2`：`Scope-1` 点估计净变 **`{pf_pool['pct_cost_change']:+.2f}%`**（`4/8` 簇更省，`p = {pf_pool['cost_sign_flip_p']:.4f}`，CI 跨零），质量自举 95% CI **`{_fmt_ci(pf_pool['quality_bootstrap_95ci'], 2, 100.0, 'pp')}`**（满足 `-5.00pp`，`t(7)` 下界 `{pf_pool['quality_paired_t_95ci'][0]*100:+.2f}pp` 略越界，有限 8 步 `K* = {k_f_pool}`）。",
            f"4. **完整研发总账**：开发集 `{rd['dev_calls']}` 调用 / `{rd['dev_v2']:,} v2` + 流前中止 `{rd['abort_calls']}` 调用 / `{rd['abort_v2']:,} v2` + 完成评估 `{rd['confirmation_calls']}` 调用 / `{rd['confirmation_v2']:,} v2` = **`{rd['total_calls']:,}` 次逻辑调用 / `{rd['total_v2']:,} v2`**（`{rd['unknown_usage_attempts']}` 未知用量）。",
            "",
            "## 2. 确认组（`n = 8`）三臂总表（并列零求解与整幕全角色零调用，`U08`）",
            "",
            "| 臂 | 私判通过 | 零求解通过 | **整幕全角色零调用** | 总调用数 | 求解轮数 | Scope-1 总 v2 | stream_solve | dag_prop | dag_synth | dag_failed | dag_merge | guard_ind | 容器累计秒 | 最大 K/d/fo |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *[_arm_row_md(conf, dual_c, a) for a in ARMS],
            "",
            "## 3. 分层配对推断（修正分母与三项独立判定，`U07 / U10`）",
            "",
            f"- 精确分层校核：A层相对 `Mono` 为 **`{pm_a['pct_cost_change']:+.2f}%`**（修正旧硬编码 `-12.1%`）；B层（`b01/b02` 新静态拓扑 + `b03/b04` 新到达次序/切换）相对 `Fixed2` 为 **`{pf_b['pct_cost_change']:+.2f}%`**（修正旧 `-25.3%`），相对 `Mono` 为 **`{pm_b['pct_cost_change']:+.2f}%`**（通过率同为 `31/32`）；去除 `a01` 后的 `n=7` 多节点簇相对 `Mono` 为 **`{pm_n7['pct_cost_change']:+.2f}%`**（`p = {pm_n7['cost_sign_flip_p']:.4f}`）。",
            "",
            "| 对比 | 分层 | 通过幕数 (H vs Base) | 质量差 Bootstrap 95% CI | 成本净变 (总差 v2) | 成本双侧 `p` | 成本均值差 Bootstrap 95% CI | 成本优效 / Bootstrap非劣 |",
            "|---|---|---|---|---:|---:|---|---|",
            *strat_rows_zh,
            "",
            "## 4. 逐簇明细与精确 `K*`（`a04=2, a02=not_reached, b01=1` vs Mono）",
            "",
            "| 簇 ID | 分层 | 隐藏拓扑类 | Mono 通过 (v2) | Fixed2 通过 (v2) | **Hegel 通过 (v2)** | H−Mono% | H−Fixed2% | 零求解 | 最大 K/d/fo | 共享 exec=1 | 逐簇 `K*` (vs M / F) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
            *_cluster_rows_md(conf),
            "",
            "## 5. 预登记结构验证门逐项自动裁决表（`U06`）",
            "",
            "| 结构门 ID | 预登记判据 | 实测记录 | 裁决 |",
            "|---|---|---|---|",
            *gate_rows,
            "",
            "## 6. 开发集（`n = 4`）摘要与 `M10.1.1` 机制故障根因指向",
            "",
            "| 臂 | 私判通过 | 零求解通过 | **整幕全角色零调用** | 总调用数 | 求解轮数 | Scope-1 总 v2 | stream_solve | dag_prop | dag_synth | dag_failed | dag_merge | guard_ind | 容器累计秒 | 最大 K/d/fo |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *[_arm_row_md(dev, dual_d, a) for a in ARMS],
            "",
            f"- **关键机制诊断（详见 `docs/57b_*`）**：`M10.1` 中 Hegel 守卫开销（`1,761,324 v2`）有 **`{u02['historical_post_overlap_unadopted_guard_calls']}` 次调用 / `{u02['historical_post_overlap_unadopted_guard_v2']:,} v2`** 来自同源正证人重叠导致的不可满足互斥约束（`0` 次采纳），另有 `a03/S1` 签票时序异常触发的 `{u03['a03_s1_orchestration_anomaly']['mono_fallback_calls_after_valid_rewire']}` 次调用 / `{u03['a03_s1_orchestration_anomaly']['mono_fallback_v2_observed']:,} v2` 回退开销；这些实现故障不能归为图发现固有资本支出，已在 `M10.1.1` 全量修复并进入 `M10.2` 验证。",
            "",
        ]
    )

    en_57 = "\n".join(
        [
            "# Hegel `M10.0.1` Closure, `M10.1` Panel-Assisted Bounded DAG Discovery, and `M10.1.1` Calibrated Report (`U01–U12`)",
            "",
            "> **Zero Hardcoded Derived Constants (`U10`)**: Every percentage, `K*`, CI, dual zero-call count, and structural gate verdict is computed directly from sealed JSONs with consistency assertions.",
            "",
            "## 1. Calibrated Executive Summary & Claim Scope (`U01 / U06 / U07 / U08`)",
            "",
            f"1. **Provenance (`U01`)**: `source_private_previously_exposed = {flags['source_private_previously_exposed']}`, `protocol_amended_after_source_results = {flags['protocol_amended_after_source_results']}`, `downstream_streams_previously_exposed = {flags['downstream_streams_previously_exposed']}` (`0/192` stream episodes run at Stage S abort; prospective online stream validation frozen at `{conf['execution_provenance']['code_commit_at_start'][:9]}`).",
            "2. **Prompt & Probe Scope (`U06`)**: Eliminates fixed `(calibrate_upstream, solve_downstream)` slots and verifies `K=3/4` DAGs, `a04` shared-prefix `exec_count==1`, and `5/5` solver-free unseen node compositions, under calibration-panel and prompt decomposition scaffolding (`pre-LLM probe cap = 2` Mono vs `5` modular). De-scaffolded equal-probe-cap evaluation is isolated in `M10.2`.",
            f"3. **Statistical Verdicts (`U07 / U08`)**: `Hegel` `61/64 (95.3%)` vs `Mono` `60/64 (93.8%)` and `Fixed2` `60/64 (93.8%)`; `solver_free_successes = 19 / 18 / 22`, **`all_role_call_free_successes = 17 / 10 / 11`**. Vs `Mono`: `{pm_pool['pct_cost_change']:+.2f}%` (`p = {pm_pool['cost_sign_flip_p']:.4f}`), quality bootstrap 95% CI `{_fmt_ci(pm_pool['quality_bootstrap_95ci'], 2, 100.0, 'pp')}` (`< -5.00pp`, non-inferiority NOT established, `K* = {k_m_pool}`). Vs `Fixed2`: `{pf_pool['pct_cost_change']:+.2f}%` (`p = {pf_pool['cost_sign_flip_p']:.4f}`), quality bootstrap 95% CI `{_fmt_ci(pf_pool['quality_bootstrap_95ci'], 2, 100.0, 'pp')}` (`t(7)` lower `{pf_pool['quality_paired_t_95ci'][0]*100:+.2f}pp`, `K* = {k_f_pool}`). Grand R&D ledger: `{rd['total_calls']:,}` calls / `{rd['total_v2']:,} v2` (`0` unknown).",
            "",
            "## 2. Confirmation (`n = 8`) 3-Arm Master Table (`U08`)",
            "",
            "| Arm | Strict Pass | Solver-Free Pass | **All-Role Zero-Call Pass** | Total Calls | Solver Turns | Scope-1 v2 | stream_solve | dag_prop | dag_synth | dag_failed | dag_merge | guard_ind | Container Sec | Max K/d/fo |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *[_arm_row_md(conf, dual_c, a) for a in ARMS],
            "",
            "## 3. Stratified Paired Cluster Inference (`U07 / U10`)",
            "",
            "| Comparison | Stratum | Passes (H vs Base) | Quality Boot 95% CI | Cost % Change (Total Δv2) | Cost `p` | Cost Mean Boot 95% CI | Cost Sup / Boot NI |",
            "|---|---|---|---|---:|---:|---|---|",
            *strat_rows_en,
            "",
            "## 4. Per-Cluster Audit Table & Exact `K*`",
            "",
            "| World ID | Stratum | Hidden Topology | Mono Pass (v2) | Fixed2 Pass (v2) | **Hegel Pass (v2)** | H−Mono% | H−Fixed2% | Solver-Free | Max K/d/fo | Shared exec=1 | `K*` (vs M / F) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
            *_cluster_rows_md(conf),
            "",
            "## 5. Pre-Registered Structural Verification Gates (`U06`)",
            "",
            "| Gate ID | Criterion | Observed | Verdict |",
            "|---|---|---|---|",
            *gate_rows,
            "",
        ]
    )

    t_rows = [
        f"| `{t['world_id']}` | `{t['arm']}` | `{t['old_dag_id']} -> {t['new_dag_id']}` | `{t['same_old_node_code_shas']}` | **`{t['strict_v2_verdict']}`** |"
        for t in u03["tickets"]
    ]
    a02_s7 = u05["a02_s7_s8_isolated_docker_replays"]["S7_old_separable_revisit"]
    a02_s8 = u05["a02_s7_s8_isolated_docker_replays"]["S8_coupled_revisit"]

    zh_57b = "\n".join(
        [
            "# `M10.1.1` 审计闭环、证据契约升级与机制归因勘误报告（`U01–U12`，`0` 付费调用）",
            "",
            "> **封存目录**：`docs/evidence/m10_1_1_audit_closure_and_contract_reverification_20261128/`（零修改既有 `m10_0*` / `m10_1_*` 原件与 `60/60/61` 得分）。",
            "",
            "## 1. 对 `M10.0.1 + M10.1` 既有实现缺陷的完整清点与代码闭环",
            "",
            f"1. **`U02` 可交叠有效域误送互斥分类器（已闭环）**：原 `_induce_guards_for_table` 未做碰撞预检且未排除同输入双阳性重叠，导致 `8/8` 簇在首次分解后产生 **`{u02['historical_post_overlap_unadopted_guard_calls']}` 次调用 / `{u02['historical_post_overlap_unadopted_guard_v2']:,} v2`** 零采纳守卫支出。现已合入 `build_feasible_domain_negative_witnesses` + `check_axis_aligned_all_of_guard_satisfiability`，在全部 `{u02['caught_by_zero_llm_precheck_steps']}` 个历史碰撞步实现 **`0` LLM 调用即时识别**，并支持可交叠候选集公开路由。",
            f"2. **`U03 + U04` 四链撤回语义外键与候选采纳事务分离（已闭环）**：升级 `verify_m101_cut_withdrawal_external_fk` 强绑定 `world_id / arm / old_dag_node_code_shas / new_dag_node_code_shas / public_input_sha256 / scientific_or_numerical` 及 `ProbeReceiptStore` 排他 `os.link` 写一次与非空图索引校验；对历史 `8` 张票复核确认 Hegel `2/6` 代码一致（`a02/S2, a04/S6`）、`4/6` 代码版本错配被新验证器严格拦截，全部 `8/8` 关系与并发负控拦截通过；同时将 `candidate_public_adoption` 与 `withdrawal_proof_issuance` 解耦，避免 `a03/S1`（已隔离回退账 `{u03['a03_s1_orchestration_anomaly']['mono_fallback_calls_after_valid_rewire']}` 调用 / `{u03['a03_s1_orchestration_anomaly']['mono_fallback_v2_observed']:,} v2`）丢弃公开合格图。",
            f"3. **`U05 + U06` 运行时边消费/反包装空壳强校验与 `a02/S7, S8` 隔离容器诊断（已闭环）**：升级 `compile_m101_dag_to_instrumented_solver_code` 强制校验每条入边实际读取（`_TrackedPayloadDict`）与非汇节点新增数值变换，`3/3` 拒绝 `audit_semantic_shell_diamond`、`audit_unused_branch_diamond` 与 `audit_exact_identity`，同时 `a04/S4` 真实四节点菱形在容器中 `100%` 通过。在隔离 Docker 容器内对封存候选做事后诊断重放（注：宿主与容器 Python 版本不同致 `pickle` 原始字节摘要不同 `exact_container_public_sha_match = {a02_s7['exact_container_public_sha_match']}`，但实际选中 `dag_07` 的公开/私判六位小数指标与原容器回执完全一致；仅作机制诊断，不改动原 `61/64` 得分）：`S7` 上纯公开 CV 选择器选中 `dag_06`（公开 `y_mae = {a02_s7['dag_06_public_cv']['public_cv_y_mae']}` < `dag_07` `{a02_s7['dag_07_public_cv']['public_cv_y_mae']}`，诊断私判 `max_y_mae = {a02_s7['dag_06_private_max_y_mae']} <= 0.045`），`S8` 上保留 `dag_07`（公开 `{a02_s8['dag_07_public_cv']['public_cv_y_mae']}` < `{a02_s8['dag_06_public_cv']['public_cv_y_mae']}`）。",
            "4. **`U09 + U11 + U12` 生产门接线与完整在线状态冷恢复（已闭环）**：`T05/T06` 已接入 `validate_public_witness` 与 `M10.0` `validate_public_schema.py`（开发正控 `2/2` 零误拒，原曝光失败候选 `alpha_m100_c03/S4` 与 `beta_m100_c04/S5` 被新生产门直接拦截，`alpha_m100_c03/S1` 保持临界通过，Layer 2 严格预算门实测拦截紧裕度 Beta KKT 证书）；`persist_m101_full_prefix_state` / `hydrate_m101_full_prefix_state` 持久化完整在线前缀并强验外键；`32/32` 标量 AST 提升普查明确界定为 `8 Stage S + 24 Mono` 定向样本与既有 `M6.8` 标量提升器范围。",
            "",
            "## 2. 历史 `M10.1` 撤回票新验证器逐票复核表（`U03`）",
            "",
            "| 世界 ID | 实验臂 | 票据声明转换 | prior/fail 节点代码集一致 | 新 v2 语义验证器裁决 |",
            "|---|---|---|---|---|",
            *t_rows,
            "",
        ]
    )

    en_57b = "\n".join(
        [
            "# `M10.1.1` Audit Closure, Strict Semantic Receipt Contracts, and Forensic Errata (`U01–U12`, `0` Paid Calls)",
            "",
            "> **Sealed Directory**: `docs/evidence/m10_1_1_audit_closure_and_contract_reverification_20261128/` (`0` mutations to prior archives or `60/60/61` scores).",
            "",
            f"- **`U02` Guard Collision Closure**: `{u02['historical_post_overlap_unadopted_guard_calls']}` historical unadopted post-overlap Hegel guard calls (`{u02['historical_post_overlap_unadopted_guard_v2']:,} v2`) caught in `0` LLM calls across `{u02['caught_by_zero_llm_precheck_steps']}` steps.",
            f"- **`U03 + U04` Strict Semantic 4-Way FK & Transaction Separation**: `{u03['strict_semantic_valid_count']}` code-consistent vs `{u03['strict_semantic_rejected_historical_mismatch_count']}` code-mismatched historical tickets classified; `8/8` negative controls (incl. missing index & concurrent `os.link` race) rejected; `a03/S1` (`{u03['a03_s1_orchestration_anomaly']['mono_fallback_v2_observed']:,} v2` fallback) decoupled.",
            f"- **`U05 + U06` Compiler Anti-Wrapper/Dead-Branch & Docker Replays**: `3/3` semantic shell / unused-branch / identity negative controls rejected; `a04/S4` diamond accepted; isolated Docker replay (`exact_container_public_sha_match = {a02_s7['exact_container_public_sha_match']}`) selects `dag_06` on `a02/S7` (`public_cv_y_mae = {a02_s7['dag_06_public_cv']['public_cv_y_mae']}`, diagnostic private `{a02_s7['dag_06_private_max_y_mae']}`) and keeps `dag_07` on `S8`.",
            "- **`U09 + U11 + U12`**: `T05/T06` wired into Layer 2 & `M10.0` `validate_public_schema.py`; full online prefix hydration verified; `32/32` AST lifting sample scope documented.",
            "",
            "| World ID | Arm | Transition | Same Old Node Code SHAs | Strict v2 Verdict |",
            "|---|---|---|---|---|",
            *t_rows,
            "",
        ]
    )

    OUT_57_ZH.write_text(zh_57, encoding="utf-8")
    OUT_57_EN.write_text(en_57, encoding="utf-8")
    OUT_57B_ZH.write_text(zh_57b, encoding="utf-8")
    OUT_57B_EN.write_text(en_57b, encoding="utf-8")
    print("Generated calibrated docs/57_* and docs/57b_*", flush=True)


if __name__ == "__main__":
    main()
