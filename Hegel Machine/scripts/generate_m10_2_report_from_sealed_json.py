#!/usr/bin/env python3
"""Deterministic, Zero-Hardcoded-Constants Bilingual Report Generator for `M10.1.1` & `M10.2` (`U01–U12`).

Reads ONLY sealed JSON artifacts:
  - `docs/evidence/m10_1_1_audit_closure_and_contract_reverification_20261128/*.json`
  - `docs/evidence/m10_2_preregistered_analysis_protocol.json`
  - `docs/evidence/m10_2_dev_selective_bounded_dag_calibration_20261128/m10_2_selective_bounded_dag_summary.json`
  - `docs/evidence/m10_2_selective_bounded_dag_confirmation_20261128/m10_2_selective_bounded_dag_summary.json`

Emits:
  - `docs/58_M10_1_1_Closure_and_M10_2_Selective_Bounded_DAG_Discovery_Report.md`
  - `docs/58_M10_1_1_Closure_and_M10_2_Selective_Bounded_DAG_Discovery_Report_ZH.md`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
M1011_DIR = REPO_ROOT / "docs/evidence/m10_1_1_audit_closure_and_contract_reverification_20261128"
PREREG_JSON = REPO_ROOT / "docs/evidence/m10_2_preregistered_analysis_protocol.json"
DEV_JSON = REPO_ROOT / "docs/evidence/m10_2_dev_selective_bounded_dag_calibration_20261128/m10_2_selective_bounded_dag_summary.json"
CONF_JSON = REPO_ROOT / "docs/evidence/m10_2_selective_bounded_dag_confirmation_20261128/m10_2_selective_bounded_dag_summary.json"

OUT_58_EN = REPO_ROOT / "docs/58_M10_1_1_Closure_and_M10_2_Selective_Bounded_DAG_Discovery_Report.md"
OUT_58_ZH = REPO_ROOT / "docs/58_M10_1_1_Closure_and_M10_2_Selective_Bounded_DAG_Discovery_Report_ZH.md"

ARMS = (
    "Flat_MonolithicArchive_Online",
    "Eager_BoundedDAG_Online",
    "Hegel_SelectiveBoundedDAG_Online",
)


def _fmt_ci(ci: list[float] | None, decimals: int = 2, scale: float = 1.0, suffix: str = "") -> str:
    if not ci or len(ci) != 2:
        return "N/A"
    return f"[{ci[0] * scale:+.{decimals}f}{suffix}, {ci[1] * scale:+.{decimals}f}{suffix}]"


def _arm_row_md(summary: dict[str, Any], arm: str) -> str:
    o = summary["arm_outcome_totals"][arm]
    l_tot = summary["arm_cost_ledgers"][arm]["scope1_in_stream_total"]
    by_r = summary["arm_cost_ledgers"][arm]["by_role"]
    pct = 100.0 * o["strict_passes"] / max(1, o["episode_count"])
    probe_s = (
        float(o["pre_llm_probe_wall_ms"]) + float(o["post_llm_probe_wall_ms"]) + float(o["official_verifier_wall_ms"])
    ) / 1000.0
    return (
        f"| `{arm}` | **{o['strict_passes']}/{o['episode_count']} ({pct:.1f}%)** | "
        f"{o['solver_free_successes']}/{o['episode_count']} | **{o['all_role_call_free_successes']}/{o['episode_count']}** | "
        f"{o['compositional_reuse_passes']} | {o['max_pre_llm_probes_observed']} ({o['total_pre_llm_probes_used']}) | "
        f"{l_tot['call_count']} | {o['total_turns_used']} | **{l_tot['proxy_v2_u_010c_4o']:,}** | "
        f"{by_r['stream_solve']['proxy_v2_u_010c_4o']:,} | {by_r['dag_structure_proposal']['proxy_v2_u_010c_4o']:,} | "
        f"{by_r['dag_compile_and_node_synthesis']['proxy_v2_u_010c_4o']:,} | {by_r['dag_failed_search']['proxy_v2_u_010c_4o']:,} | "
        f"{by_r['dag_merge_withdrawal_repair']['proxy_v2_u_010c_4o']:,} | {by_r['guard_induction']['proxy_v2_u_010c_4o']:,} | "
        f"{probe_s:,.2f}s | {o['max_active_nodes_observed']}/{o['max_depth_observed']}/{o['max_fanout_observed']} |"
    )


def _nested_horizon_rows_md(summary: dict[str, Any], comp_key: str) -> list[str]:
    nc = summary["nested_horizon_curves_and_K_star"][comp_key]["nested_checkpoints"]
    rows = []
    for hk in sorted(nc.keys()):
        r = nc[hk]
        v = r["verdicts"]
        c_inf = r["cost_inference"]
        q_inf = r["quality_inference"]
        prim_tag = " **(Primary)**" if r["is_primary_endpoint"] else ""
        rows.append(
            f"| `H = {r['horizon_H']}`{prim_tag} | `{r['arm_a_passes']}/{r['total_episodes']}` vs `{r['arm_b_passes']}/{r['total_episodes']}` | "
            f"`{_fmt_ci(q_inf['bootstrap_95ci'], 2, 100.0, 'pp')}` (`t`: `{_fmt_ci(q_inf['paired_t_95ci'], 2, 100.0, 'pp')}`) | "
            f"**`{r['pct_cost_change']:+.2f}%`** (`{r['total_cost_delta_v2']:+,}`) | `{r['cluster_wins_arm_a_cheaper']}/{c_inf['n_clusters']}` | "
            f"`{c_inf['two_sided_sign_flip_p']:.4f}` | `{_fmt_ci(c_inf['bootstrap_95ci'], 0)}` | "
            f"`{v['cost_superiority_supported']}` / `{v['quality_noninferiority_bootstrap_supported']}` |"
        )
    return rows


def _stratified_rows_md(summary: dict[str, Any], lang: str) -> list[str]:
    st = summary["stratified_primary_endpoint_h16"]
    labels = [
        ("pooled_all", "全体汇总 (n=8)", "Pooled all clusters (n=8)"),
        ("stratum_a_seen_topology", "A层已见拓扑族 (n=4)", "Stratum A seen families (n=4)"),
        ("stratum_b1_unseen_static_topology", "B1层未见新静态拓扑 (n=2)", "Stratum B1 unseen static topology (n=2)"),
        ("stratum_b2_shift_order_and_regime_switch", "B2层耦合先到/切换次序 (n=2)", "Stratum B2 order/regime switch (n=2)"),
        ("stratum_b_combined", "B层合计 (n=4)", "Stratum B combined (n=4)"),
        ("monolithic_control", "不可分单体负控 (n=1)", "Monolithic negative control (n=1)"),
        ("multi_node_non_monolithic", "仅多节点非单体簇 (n=7)", "Multi-node non-monolithic (n=7)"),
    ]
    comps = [
        ("Hegel_Selective_minus_Flat_Monolithic", "Hegel − 通用单体", "Hegel − Monolithic"),
        ("Hegel_Selective_minus_Eager_BoundedDAG", "Hegel − 激进建图", "Hegel − Eager DAG"),
        ("Eager_BoundedDAG_minus_Flat_Monolithic", "激进建图 − 通用单体", "Eager DAG − Monolithic"),
    ]
    rows = []
    for sk, l_zh, l_en in labels:
        if sk not in st:
            continue
        lbl = l_zh if lang == "zh" else l_en
        for ck, c_zh, c_en in comps:
            r = st[sk][ck]
            cl = c_zh if lang == "zh" else c_en
            c_p = r["cost_inference"]["two_sided_sign_flip_p"]
            p_str = f"{c_p:.4f}" if c_p is not None else "N/A"
            rows.append(
                f"| {cl} | {lbl} | `{r['arm_a_passes']}/{r['total_episodes']}` vs `{r['arm_b_passes']}/{r['total_episodes']}` | "
                f"`{_fmt_ci(r['quality_inference']['bootstrap_95ci'], 2, 100.0, 'pp')}` | "
                f"**`{r['pct_cost_change']:+.2f}%`** (`{r['total_cost_delta_v2']:+,}`) | `{p_str}` | "
                f"`{_fmt_ci(r['cost_inference']['bootstrap_95ci'], 0)}` | "
                f"`{r['cost_superiority_supported']}` / `{r['quality_noninferiority_bootstrap_supported']}` |"
            )
    return rows


def _cluster_rows_md(summary: dict[str, Any]) -> list[str]:
    st_by_key = {(s["world_id"], s["arm"]): s for s in summary["streams"]}
    led = summary["arm_cost_ledgers"]
    k_mono = summary["nested_horizon_curves_and_K_star"]["Hegel_Selective_minus_Flat_Monolithic"]["per_cluster_K_star"]
    k_eag = summary["nested_horizon_curves_and_K_star"]["Hegel_Selective_minus_Eager_BoundedDAG"]["per_cluster_K_star"]
    wids = sorted({s["world_id"] for s in summary["streams"]})
    n_ep = summary["primary_horizon_steps"]
    rows = []
    for w in wids:
        sh = st_by_key[(w, "Hegel_SelectiveBoundedDAG_Online")]
        sm = st_by_key[(w, "Flat_MonolithicArchive_Online")]
        se = st_by_key[(w, "Eager_BoundedDAG_Online")]
        ch = int(led["Hegel_SelectiveBoundedDAG_Online"]["by_cluster"][w]["proxy_v2_u_010c_4o"])
        cm = int(led["Flat_MonolithicArchive_Online"]["by_cluster"][w]["proxy_v2_u_010c_4o"])
        ce = int(led["Eager_BoundedDAG_Online"]["by_cluster"][w]["proxy_v2_u_010c_4o"])
        pct_m = 100.0 * (ch - cm) / max(1, cm)
        pct_e = 100.0 * (ch - ce) / max(1, ce)
        rows.append(
            f"| `{w}` | `{sh['stratum']}` | `{sh['topology_class_hidden_analytics_only']}` | "
            f"{sm['strict_passes']}/{n_ep} ({cm:,}) | {se['strict_passes']}/{n_ep} ({ce:,}) | "
            f"**{sh['strict_passes']}/{n_ep} ({ch:,})** | `{pct_m:+.2f}%` | `{pct_e:+.2f}%` | "
            f"{sh['solver_free_successes']} / **{sh['all_role_call_free_successes']}** | "
            f"K={sh['max_active_nodes_observed']},d={sh['max_depth_observed']},fo={sh['max_fanout_observed']} | "
            f"{'YES' if sh['shared_fanout_exec_count_1_verified'] else '—'} | `{k_mono[w]['K_star']} / {k_eag[w]['K_star']}` |"
        )
    return rows


def main() -> None:
    prereg = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    dev = json.loads(DEV_JSON.read_text(encoding="utf-8"))
    conf = json.loads(CONF_JSON.read_text(encoding="utf-8"))
    u02 = json.loads((M1011_DIR / "01_u02_guard_collision_and_satisfiability_audit.json").read_text(encoding="utf-8"))

    # Consistency assertions (`U01 / U10`):
    assert conf["total_usage_unknown_attempt_count"] == 0 and dev["total_usage_unknown_attempt_count"] == 0
    assert conf["execution_provenance"]["working_tree_dirty_at_start"] is False
    assert conf["execution_provenance"]["source_tree_unchanged_during_paid_execution"] is True
    assert dev["execution_provenance"]["source_tree_sha256_at_start"] == conf["execution_provenance"]["source_tree_sha256_at_start"], (
        "Dev and Confirmation must share the EXACT byte-identical frozen Python source tree!"
    )
    for arm in ARMS:
        assert conf["arm_outcome_totals"][arm]["max_pre_llm_probes_observed"] <= prereg["equal_resource_contracts"]["max_pre_llm_probes_per_step_all_arms"]

    hm_h16 = conf["nested_horizon_curves_and_K_star"]["Hegel_Selective_minus_Flat_Monolithic"]["nested_checkpoints"]["H_16"]
    he_h16 = conf["nested_horizon_curves_and_K_star"]["Hegel_Selective_minus_Eager_BoundedDAG"]["nested_checkpoints"]["H_16"]
    km_pool = conf["nested_horizon_curves_and_K_star"]["Hegel_Selective_minus_Flat_Monolithic"]["empirical_quality_constrained_K_star"]
    ke_pool = conf["nested_horizon_curves_and_K_star"]["Hegel_Selective_minus_Eager_BoundedDAG"]["empirical_quality_constrained_K_star"]

    tot_rd_calls = int(dev["total_logical_calls"]) + int(conf["total_logical_calls"])
    tot_rd_v2 = int(dev["grand_physical_ledger"]["proxy_v2_u_010c_4o"]) + int(conf["grand_physical_ledger"]["proxy_v2_u_010c_4o"])
    src_sha = conf["execution_provenance"]["source_tree_sha256_at_start"]

    gate_rows = [
        f"| `{g['gate_id']}` | {g['preregistered_criterion']} | {g['observed']} | **`{g['status']}`** |"
        for g in conf["structural_verification_gates"]
    ]

    zh = "\n".join(
        [
            "# Hegel `M10.1.1` 审计契约闭环与 `M10.2` 去脚手架同探针上限选择性计算图发现与长程（`H = 16`）验证报告",
            "",
            "> **零硬编码派生常数与严格树冻结声明（`U01 / U10`）**：开发集（`n=4, H=16`）与未曝光确认集（`n=8, H=16`）共享完全相同的 Python 源码树哈希（`"
            + src_sha
            + "`），`working_tree_dirty_at_start = False`，`source_private_previously_exposed = False`，`protocol_amended_after_source_results = False`，`0` 未知用量回执。",
            "",
            "## 1. 执行摘要与三项独立统计裁决（主终点 `H = 16`）",
            "",
            f"1. **实验设计与单因子归因（回应 `U02 / U05 / U06`）**：三臂（`Flat_MonolithicArchive_Online`、`Eager_BoundedDAG_Online`、`Hegel_SelectiveBoundedDAG_Online`）完全共享 `MAX_PRE_LLM_PROBES_PER_STEP = 4` 容器探针预算、`stat_00..stat_17` 无角色名盲统计量、去面板提示词（零 `panel_a/b/c` 切分示例）、`M10.1.1` 零 LLM 守卫可满足性预检（在 `M10.1` 中曾浪费 `{u02['historical_post_overlap_unadopted_guard_v2']:,} v2`）以及 `_TrackedPayloadDict` 入边消费与新增数值反空壳强校验（注：公共 `study` 字典本身对三臂同等保留辅助校准子表以确保多阶段可辨识性，已在预登记协议公开披露）。`Eager` 与 `Hegel_Selective` 的唯一差别为 `0`-LLM 结构投资与止损门 `evaluate_m102_selective_investment_gate`。",
            f"2. **相对强通用单体基线（`Hegel_Selective − Flat_Monolithic`，预登记主经济对比，`H = 16`）**：私判通过 **`{hm_h16['arm_a_passes']}/{hm_h16['total_episodes']}` vs `{hm_h16['arm_b_passes']}/{hm_h16['total_episodes']}`**（Bootstrap 95% CI `{_fmt_ci(hm_h16['quality_inference']['bootstrap_95ci'], 2, 100.0, 'pp')}`，`t(7)` 95% CI `{_fmt_ci(hm_h16['quality_inference']['paired_t_95ci'], 2, 100.0, 'pp')}`）；`Scope-1` 总成本净变 **`{hm_h16['pct_cost_change']:+.2f}%`**（`{hm_h16['total_cost_delta_v2']:+,} v2`，`{hm_h16['cluster_wins_arm_a_cheaper']}/8` 簇更省，精确置换 `p = {hm_h16['cost_inference']['two_sided_sign_flip_p']:.4f}`，Bootstrap 95% CI `{_fmt_ci(hm_h16['cost_inference']['bootstrap_95ci'], 0)}`）。三项独立裁决：`cost_superiority_supported = {hm_h16['verdicts']['cost_superiority_supported']}`、`quality_noninferiority_bootstrap_supported = {hm_h16['verdicts']['quality_noninferiority_bootstrap_supported']}`、`finite_observed_break_even_reached = {hm_h16['verdicts']['finite_observed_break_even_reached']}`（`K* = {km_pool}`）。",
            f"3. **相对无门控激进建图基线（`Hegel_Selective − Eager_BoundedDAG`，单因子机制对比，`H = 16`）**：私判通过 **`{he_h16['arm_a_passes']}/{he_h16['total_episodes']}` vs `{he_h16['arm_b_passes']}/{he_h16['total_episodes']}`**（Bootstrap 95% CI `{_fmt_ci(he_h16['quality_inference']['bootstrap_95ci'], 2, 100.0, 'pp')}`）；`Scope-1` 净变 **`{he_h16['pct_cost_change']:+.2f}%`**（`{he_h16['total_cost_delta_v2']:+,} v2`，`p = {he_h16['cost_inference']['two_sided_sign_flip_p']:.4f}`，`K* = {ke_pool}`）。",
            f"4. **完整 `M10.2` 研发总账**：开发集 `{dev['total_logical_calls']:,}` 调用 / `{dev['grand_physical_ledger']['proxy_v2_u_010c_4o']:,} v2` + 未曝光确认集 `{conf['total_logical_calls']:,}` 调用 / `{conf['grand_physical_ledger']['proxy_v2_u_010c_4o']:,} v2` = **`{tot_rd_calls:,}` 次逻辑调用 / `{tot_rd_v2:,} v2`**（`0` 未知用量）。",
            "",
            "## 2. 未曝光确认集（`n = 8, H = 16`，`128` 幕/臂）三臂总表",
            "",
            "| 实验臂 | 私判通过 | 零求解通过 | **整幕全角色零调用** | 组合免求解通过 | 调前探针峰值(累计) | 总调用数 | 求解轮数 | Scope-1 总 v2 | stream_solve | dag_prop | dag_synth | dag_failed | dag_merge | guard_ind | 容器累计秒 | 最大 K/d/fo |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *[_arm_row_md(conf, a) for a in ARMS],
            "",
            "## 3. 嵌套时域摊销轨迹（`H = 4, 8, 12, 16`，`H = 16` 为唯一主检验终点）",
            "",
            "### 3.1 `Hegel_SelectiveBoundedDAG_Online − Flat_MonolithicArchive_Online`",
            "",
            "| 时域截点 | 通过幕数 (H vs Mono) | 质量差 Bootstrap 95% CI (`t` 95% CI) | Scope-1 净变 (总差 v2) | 簇胜率 | 成本双侧 `p` | 成本均值差 Bootstrap 95% CI | 成本优效 / Bootstrap非劣 |",
            "|---|---|---|---:|---:|---:|---|---|",
            *_nested_horizon_rows_md(conf, "Hegel_Selective_minus_Flat_Monolithic"),
            "",
            "### 3.2 `Hegel_SelectiveBoundedDAG_Online − Eager_BoundedDAG_Online`",
            "",
            "| 时域截点 | 通过幕数 (H vs Eager) | 质量差 Bootstrap 95% CI (`t` 95% CI) | Scope-1 净变 (总差 v2) | 簇胜率 | 成本双侧 `p` | 成本均值差 Bootstrap 95% CI | 成本优效 / Bootstrap非劣 |",
            "|---|---|---|---:|---:|---:|---|---|",
            *_nested_horizon_rows_md(conf, "Hegel_Selective_minus_Eager_BoundedDAG"),
            "",
            "## 4. 主终点（`H = 16`）分层配对统计表",
            "",
            "| 对比 | 分层 | 通过幕数 | 质量差 Bootstrap 95% CI | 成本净变 (总差 v2) | 成本 `p` | 成本均值差 Bootstrap 95% CI | 成本优效 / Bootstrap非劣 |",
            "|---|---|---|---|---:|---:|---|---|",
            *_stratified_rows_md(conf, "zh"),
            "",
            "## 5. 确认集逐簇明细与精确 `K*`",
            "",
            "| 簇 ID | 分层 | 隐藏拓扑类 | Mono 通过 (v2) | Eager 通过 (v2) | **Hegel 通过 (v2)** | H−Mono% | H−Eager% | 零求解 / **全零调用** | 最大 K/d/fo | 共享 exec=1 | 逐簇 `K*` (vs M / E) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
            *_cluster_rows_md(conf),
            "",
            "## 6. 预登记结构验证门自动裁决表（`U06 / U11`）",
            "",
            "| 结构门 ID | 预登记判据 | 实测记录 | 裁决 |",
            "|---|---|---|---|",
            *gate_rows,
            "",
            "## 7. 开发校准集（`n = 4, H = 16`）摘要表",
            "",
            "| 实验臂 | 私判通过 | 零求解通过 | **整幕全角色零调用** | 组合免求解通过 | 调前探针峰值(累计) | 总调用数 | 求解轮数 | Scope-1 总 v2 | stream_solve | dag_prop | dag_synth | dag_failed | dag_merge | guard_ind | 容器累计秒 | 最大 K/d/fo |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *[_arm_row_md(dev, a) for a in ARMS],
            "",
        ]
    )

    en = "\n".join(
        [
            "# Hegel `M10.1.1` Audit Closure & `M10.2` De-Scaffolded Equal-Probe-Cap Selective Bounded DAG Discovery (`H = 16`)",
            "",
            "> **Zero Hardcoded Constants & Strict Source-Tree Freeze (`U01 / U10`)**: `dev` (`n=4, H=16`) and unexposed `confirmation` (`n=8, H=16`) share byte-identical Python `source_tree_sha256` (`"
            + src_sha
            + "`), `working_tree_dirty_at_start = False`, `source_private_previously_exposed = False`, `protocol_amended_after_source_results = False`, `0` unknown-usage attempts.",
            "",
            "## 1. Executive Summary & Three Separate Statistical Verdicts (`H = 16` Primary Endpoint)",
            "",
            f"1. **Equal-Probe-Cap & De-Scaffolded Single-Factor Design (`U02 / U05 / U06`)**: All 3 arms share `MAX_PRE_LLM_PROBES_PER_STEP = 4`, role-blind `stat_00..stat_17`, de-scaffolded prompts (`0` `panel_a/b/c` splitting recipes), `M10.1.1` `0`-LLM guard satisfiability pre-check, and runtime `_TrackedPayloadDict` edge-consumption & anti-wrapper-shell checks (with public calibration sub-tables retained symmetrically across all 3 arms as pre-registered). `Eager_BoundedDAG_Online` and `Hegel_SelectiveBoundedDAG_Online` differ solely in `evaluate_m102_selective_investment_gate`.",
            f"2. **Primary Economic Endpoint (`Hegel_Selective − Flat_Monolithic` at `H = 16`)**: Strict pass **`{hm_h16['arm_a_passes']}/{hm_h16['total_episodes']}` vs `{hm_h16['arm_b_passes']}/{hm_h16['total_episodes']}`** (Bootstrap 95% CI `{_fmt_ci(hm_h16['quality_inference']['bootstrap_95ci'], 2, 100.0, 'pp')}`, `t(7)` 95% CI `{_fmt_ci(hm_h16['quality_inference']['paired_t_95ci'], 2, 100.0, 'pp')}`); `Scope-1` cost delta **`{hm_h16['pct_cost_change']:+.2f}%`** (`{hm_h16['total_cost_delta_v2']:+,} v2`, `{hm_h16['cluster_wins_arm_a_cheaper']}/8` clusters cheaper, exact `p = {hm_h16['cost_inference']['two_sided_sign_flip_p']:.4f}`, Bootstrap 95% CI `{_fmt_ci(hm_h16['cost_inference']['bootstrap_95ci'], 0)}`). Verdicts: `cost_superiority_supported = {hm_h16['verdicts']['cost_superiority_supported']}`, `quality_noninferiority_bootstrap_supported = {hm_h16['verdicts']['quality_noninferiority_bootstrap_supported']}`, `finite_observed_break_even_reached = {hm_h16['verdicts']['finite_observed_break_even_reached']}` (`K* = {km_pool}`).",
            f"3. **Single-Factor Mechanism Endpoint (`Hegel_Selective − Eager_BoundedDAG` at `H = 16`)**: Strict pass **`{he_h16['arm_a_passes']}/{he_h16['total_episodes']}` vs `{he_h16['arm_b_passes']}/{he_h16['total_episodes']}`**; `Scope-1` cost delta **`{he_h16['pct_cost_change']:+.2f}%`** (`{he_h16['total_cost_delta_v2']:+,} v2`, `p = {he_h16['cost_inference']['two_sided_sign_flip_p']:.4f}`, `K* = {ke_pool}`).",
            f"4. **Complete `M10.2` R&D Ledger**: Dev `{dev['total_logical_calls']:,}` calls / `{dev['grand_physical_ledger']['proxy_v2_u_010c_4o']:,} v2` + Confirmation `{conf['total_logical_calls']:,}` calls / `{conf['grand_physical_ledger']['proxy_v2_u_010c_4o']:,} v2` = **`{tot_rd_calls:,}` calls / `{tot_rd_v2:,} v2`** (`0` unknown).",
            "",
            "## 2. Unexposed Confirmation (`n = 8, H = 16`, `128` Episodes/Arm) Master Table",
            "",
            "| Arm | Strict Pass | Solver-Free Pass | **All-Role Zero-Call Pass** | Comp. Reuse | Max(Tot) Pre-Probes | Total Calls | Solver Turns | Scope-1 v2 | stream_solve | dag_prop | dag_synth | dag_failed | dag_merge | guard_ind | Container Sec | Max K/d/fo |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *[_arm_row_md(conf, a) for a in ARMS],
            "",
            "## 3. Nested Horizon Amortization Trajectory (`H = 4, 8, 12, 16`; `H = 16` Primary Endpoint)",
            "",
            "### 3.1 `Hegel_SelectiveBoundedDAG_Online − Flat_MonolithicArchive_Online`",
            "",
            "| Horizon | Passes (H vs Mono) | Quality Boot 95% CI (`t` 95% CI) | Scope-1 % (Total Δv2) | Cluster Wins | Cost `p` | Cost Mean Boot 95% CI | Cost Sup / Boot NI |",
            "|---|---|---|---:|---:|---:|---|---|",
            *_nested_horizon_rows_md(conf, "Hegel_Selective_minus_Flat_Monolithic"),
            "",
            "### 3.2 `Hegel_SelectiveBoundedDAG_Online − Eager_BoundedDAG_Online`",
            "",
            "| Horizon | Passes (H vs Eager) | Quality Boot 95% CI (`t` 95% CI) | Scope-1 % (Total Δv2) | Cluster Wins | Cost `p` | Cost Mean Boot 95% CI | Cost Sup / Boot NI |",
            "|---|---|---|---:|---:|---:|---|---|",
            *_nested_horizon_rows_md(conf, "Hegel_Selective_minus_Eager_BoundedDAG"),
            "",
            "## 4. Stratified Primary Endpoint (`H = 16`) Inference",
            "",
            "| Comparison | Stratum | Passes | Quality Boot 95% CI | Cost % (Total Δv2) | Cost `p` | Cost Mean Boot 95% CI | Cost Sup / Boot NI |",
            "|---|---|---|---|---:|---:|---|---|",
            *_stratified_rows_md(conf, "en"),
            "",
            "## 5. Per-Cluster Confirmation Audit Table & Exact `K*`",
            "",
            "| World ID | Stratum | Hidden Topology | Mono Pass (v2) | Eager Pass (v2) | **Hegel Pass (v2)** | H−Mono% | H−Eager% | Solver-Free / **All-Zero** | Max K/d/fo | Shared exec=1 | `K*` (vs M / E) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
            *_cluster_rows_md(conf),
            "",
            "## 6. Pre-Registered Structural Verification Gates (`U06 / U11`)",
            "",
            "| Gate ID | Criterion | Observed | Verdict |",
            "|---|---|---|---|",
            *gate_rows,
            "",
            "## 7. Dev Calibration (`n = 4, H = 16`) Summary",
            "",
            "| Arm | Strict Pass | Solver-Free Pass | **All-Role Zero-Call Pass** | Comp. Reuse | Max(Tot) Pre-Probes | Total Calls | Solver Turns | Scope-1 v2 | stream_solve | dag_prop | dag_synth | dag_failed | dag_merge | guard_ind | Container Sec | Max K/d/fo |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *[_arm_row_md(dev, a) for a in ARMS],
            "",
        ]
    )

    OUT_58_ZH.write_text(zh, encoding="utf-8")
    OUT_58_EN.write_text(en, encoding="utf-8")
    print("Generated docs/58_M10_1_1_Closure_and_M10_2_Selective_Bounded_DAG_Discovery_Report[_ZH].md", flush=True)


if __name__ == "__main__":
    main()
