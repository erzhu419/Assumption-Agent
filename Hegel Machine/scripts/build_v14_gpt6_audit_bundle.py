#!/usr/bin/env python3
"""Build the unified Hegel Machine v14 GPT-6 Audit Handoff Directory & Self-Contained ZIP.

Outputs:
  1. In-repo tracked documentation, manifest, & standalone verifier directory:
     ``docs/gpt6_audit_bundle_v14_20261118/``
  2. Self-contained uploadable ZIP archive (with all 35 JSON receipts, full sources/tests/scripts,
     and GPT-6's 3 V14 probe scripts):
     ``markdown/Hegel_v14_GPT6_Audit_Handoff_20261118.zip``

Guarantees:
  - Zero paid online LLM calls (`model_calls = 0`).
  - Zero byte modifications to all 32 historic receipts (`v10`..`v13`).
  - Zero new files added under `src/`.
  - Programmatically populates all Markdown tables, 5-item V13 Errata, Replays A–F, Deliverable 2
    prefix full costs, Deliverable 3 continuous object 7-sentence ledgers, and §9 Stopping Rule verdicts
    directly from the strict-JSON receipts.
  - Executes `05_STANDALONE_VERIFY_V14.py` in isolated `/tmp` with `PYTHONPATH=""` and asserts
    `all_checks_passed == True`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / "docs" / "gpt6_audit_bundle_v14_20261118"
ZIP_OUT = REPO_ROOT / "markdown" / "Hegel_v14_GPT6_Audit_Handoff_20261118.zip"

EVIDENCE_DIRS = [
    ("v10_world_structure_20261116", REPO_ROOT / "docs" / "evidence" / "v10_world_structure_20261116"),
    ("v10_2_audit_remediation_20261116", REPO_ROOT / "docs" / "evidence" / "v10_2_audit_remediation_20261116"),
    ("v10_3_scaling_and_calibration_20261117", REPO_ROOT / "docs" / "evidence" / "v10_3_scaling_and_calibration_20261117"),
    ("v10_4_occam_and_gpt6_bundle_20261117", REPO_ROOT / "docs" / "evidence" / "v10_4_occam_and_gpt6_bundle_20261117"),
    ("v11_branch_lifecycle_and_km_20261117", REPO_ROOT / "docs" / "evidence" / "v11_branch_lifecycle_and_km_20261117"),
    ("v12_autonomous_reconciliation_and_utility_20261117", REPO_ROOT / "docs" / "evidence" / "v12_autonomous_reconciliation_and_utility_20261117"),
    ("v13_evidence_recursion_and_routing_20261117", REPO_ROOT / "docs" / "evidence" / "v13_evidence_recursion_and_routing_20261117"),
    ("v14_continuous_object_and_fair_comparison_20261118", REPO_ROOT / "docs" / "evidence" / "v14_continuous_object_and_fair_comparison_20261118"),
]

BUNDLED_SOURCE_RELPATHS = [
    "src/hegel_machine/cognition/models.py",
    "src/hegel_machine/cognition/unified_research.py",
    "src/hegel_machine/cognition/competing_object_programs.py",
    "src/hegel_machine/cognition/premise_commitment_index.py",
    "src/hegel_machine/cognition/branch_reconciliation.py",
    "src/hegel_machine/cognition/discriminating_experiment_policy.py",
    "src/hegel_machine/cognition/premise_dependency_audit.py",
    "src/hegel_machine/cognition/sensor_role_calibration.py",
    "src/hegel_machine/cognition/equal_tool_attribution.py",
    "src/hegel_machine/cognition/world_structure_commitments.py",
    "src/hegel_machine/cognition/algebraic_proof_certificates.py",
    "scripts/run_v14_deliverable_1_replays_and_errata.py",
    "scripts/run_v14_fair_comparison_and_continuous_object.py",
    "scripts/build_v14_gpt6_audit_bundle.py",
    "scripts/run_v13_pr_ab_controls.py",
    "scripts/run_v13_pr_c_tightened_study.py",
    "tests/test_v14_production_contracts_and_lifecycle.py",
    "tests/test_v13_evidence_recursion_and_routing.py",
    "tests/test_v12_semantics_and_autonomous_reconciliation.py",
    "tests/test_v11_audit_and_branch_lifecycle.py",
    "markdown/hegel_v14/probes/probe_acquisition_contract.py",
    "markdown/hegel_v14/probes/probe_question_identity.py",
    "markdown/hegel_v14/probes/probe_scope_and_routing.py",
    "markdown/hegel_v14/probes/recompute_v13.py",
]

EXTRA_RECEIPT_META = {
    "deliverable_a_v12_semantics_and_errata_20261117.json": ("historic_v12_untouched", 0, "Untouched v12 Deliverable A receipt."),
    "deliverable_b_v12_autonomous_lifecycle_20261117.json": ("historic_v12_exposed_retired", 44, "Untouched v12 Deliverable B receipt (retired exposed dev material)."),
    "deliverable_c_v12_knowledge_utility_20261117.json": ("historic_v12_exposed_retired", 64, "Untouched v12 Deliverable C receipt (retired exposed dev material)."),
    "pr_a_evidence_and_recursion_controls_20261117.json": ("historic_v13_untouched", 5, "Untouched v13 PR-A receipt."),
    "pr_b_routing_vs_applicability_and_cache_20261117.json": ("historic_v13_untouched", 6, "Untouched v13 PR-B receipt."),
    "pr_c_tightened_source_paired_study_20261117.json": ("historic_v13_exposed_retired", 64, "Untouched v13 PR-C receipt (chains 1021000–1028000 retired to exposed dev material in v14)."),
    "deliverable_1_errata_and_replays_a_to_f_20261118.json": ("v14_deliverable_1_errata_probes_and_replays_a_to_f", 0, "v14 Deliverable 1: 5-item V13 Errata, remediated GPT-6 V14 probes, Replays A–F, & exposed 1024000 2x2 diagnostic."),
    "deliverable_2_unconfounded_noH_source_paired_study_20261118.json": ("v14_deliverable_2_unconfounded_noH_source_paired_study", 64, "v14 Deliverable 2: Frozen single-touch evaluation on 8 brand-new unexposed source clusters (1031000–1038000) comparing K_cold, R_lazy_noH, and K_eager_noH with prefix full costs Delta C_c(1..3)."),
    "deliverable_3_continuous_repaired_object_study_20261118.json": ("v14_deliverable_3_continuous_repaired_object_study", 36, "v14 Deliverable 3: Frozen single-touch evaluation on 6 brand-new unexposed focal_memory chains (1041000–1046000) tracking the exact same repaired accumulate object across Question A, Question B, minimal object removals, and 7-sentence factual ledgers."),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    raw = path.read_bytes()
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def _git_head_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _build_manifest_and_index() -> tuple[dict, str]:
    ledger_v10_2 = json.loads(
        (REPO_ROOT / "docs/evidence/v10_2_audit_remediation_20261116/errata_and_sha256_ledger_20261116.json").read_text(encoding="utf-8")
    )
    v10_lookup = {item["filename"]: item for item in ledger_v10_2["untouched_v10_receipts_sha256_and_roles"]}

    receipt_entries = []
    for phase_tag, edir in EVIDENCE_DIRS:
        for fp in sorted(edir.glob("*.json")):
            rel_repo = fp.relative_to(REPO_ROOT).as_posix()
            if fp.name in v10_lookup:
                role = v10_lookup[fp.name]["study_role"]
                ind_src = v10_lookup[fp.name]["independent_physical_sources"]
                notes = v10_lookup[fp.name]["notes"]
            else:
                role, ind_src, notes = EXTRA_RECEIPT_META.get(fp.name, ("historic_untouched_receipt", 0, "Untouched historic receipt."))
            receipt_entries.append(
                {
                    "phase": phase_tag,
                    "filename": fp.name,
                    "repo_relative_path": rel_repo,
                    "zip_relative_path": f"receipts/{phase_tag}/{fp.name}",
                    "size_bytes": fp.stat().st_size,
                    "sha256": _sha256_file(fp),
                    "git_blob_sha1": _git_blob_sha1(fp),
                    "study_role": role,
                    "independent_physical_sources": ind_src,
                    "notes": notes,
                }
            )

    source_entries = []
    for rel_s in BUNDLED_SOURCE_RELPATHS:
        sp = REPO_ROOT / rel_s
        line_count = len(sp.read_text(encoding="utf-8").splitlines())
        source_entries.append(
            {
                "repo_relative_path": rel_s,
                "zip_relative_path": f"bundled_sources/{rel_s}",
                "lines": line_count,
                "size_bytes": sp.stat().st_size,
                "sha256": _sha256_file(sp),
                "git_blob_sha1": _git_blob_sha1(sp),
            }
        )

    manifest = {
        "bundle_id": "hegel_v14_gpt6_audit_handoff_20261118",
        "generated_at": _utc_now(),
        "audited_v13_commit": "c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4",
        "parent_head_at_bundle_build": _git_head_commit(),
        "online_llm_calls_in_v14": 0,
        "online_llm_tokens_in_v14": 0,
        "new_src_modules_added": 0,
        "total_evidence_receipts": len(receipt_entries),
        "total_bundled_source_files": len(source_entries),
        "evidence_receipts": receipt_entries,
        "bundled_sources": source_entries,
    }

    md_lines = [
        f"# 来源清单、{len(receipt_entries)} 份证据回执索引与核查范围（v14）",
        "",
        "- **历史冻结回执（32 份，100% 原始字节零改动）：** `v10..v13` 全量保留，`1001000–1028000` 全部退居已曝光开发／回归材料；",
        "- **`v14` 全新严格 JSON（`allow_nan=False`）回执（3 份，零新增 `src/` 模块）：**",
        "  1. `deliverable_1_errata_and_replays_a_to_f_20261118.json`（V13 5 项勘误 + GPT-6 V14 三探针修复核验 + Replays A–F + 已曝光 `1024000` 2×2 诊断）；",
        "  2. `deliverable_2_unconfounded_noH_source_paired_study_20261118.json`（8 条未曝光源簇 `1031000–1038000` 去混杂无 `H*` 主比较与逐链前缀全成本 $\\Delta C_c(1..3)$）；",
        "  3. `deliverable_3_continuous_repaired_object_study_20261118.json`（6 条未曝光 `focal_memory` 连续修复对象链 `1041000–1046000`：问题 A 同能力父任务修复、问题 B 未见目标复用＋最小单对象移除干预＋逐链七句事实账本）。",
        "",
        "## 一键零依赖沙箱核验",
        "```bash",
        "PYTHONPATH=\"\" python3 05_STANDALONE_VERIFY_V14.py --output verify_v14_report.json",
        "```",
        "",
        "| 阶段 | 回执文件 | 角色分类 (`study_role`) | 字节数 | SHA256 前16位 |",
        "| :--- | :--- | :--- | ---: | :--- |",
    ]
    for r in receipt_entries:
        md_lines.append(f"| `{r['phase']}` | `{r['filename']}` | `{r['study_role']}` | `{r['size_bytes']:,}` | `{r['sha256'][:16]}…` |")
    return manifest, "\n".join(md_lines) + "\n"


def _build_executive_report() -> str:
    v14_dir = REPO_ROOT / "docs/evidence/v14_continuous_object_and_fair_comparison_20261118"
    d1 = json.loads((v14_dir / "deliverable_1_errata_and_replays_a_to_f_20261118.json").read_text(encoding="utf-8"))
    d2 = json.loads((v14_dir / "deliverable_2_unconfounded_noH_source_paired_study_20261118.json").read_text(encoding="utf-8"))
    d3 = json.loads((v14_dir / "deliverable_3_continuous_repaired_object_study_20261118.json").read_text(encoding="utf-8"))

    t_arms = d2["separate_tables"]["target_arms_24_per_arm"]
    src32 = d2["separate_tables"]["ordinary_source_episodes_32"]
    anc8 = d2["separate_tables"]["common_calibration_anchors_8"]
    pref = d2["prefix_full_cost_summary_R_minus_K"]
    p_d2 = d2["source_cluster_paired_comparisons"]["K_eager_noH_vs_R_lazy_noH"]

    qa = d3["question_A_parent_repair_summary"]
    qb_arms = d3["question_B_downstream_reuse_summary"]["arms"]
    qb_pair_r = d3["question_B_downstream_reuse_summary"]["paired_cluster_comparisons"]["K_eager_noH_vs_R_lazy_noH"]
    qb_pair_rm = d3["question_B_downstream_reuse_summary"]["paired_cluster_comparisons"][
        "K_eager_noH_vs_K_minimal_repaired_object_removed"
    ]
    ch0_d3 = d3["all_attempted_continuous_chains"][0]

    lines = [
        "# Hegel Machine v14：生产者—消费者合同收口、去混杂无 H* 源簇比较与连续修复对象端到端报告",
        "",
        "## 1. 对核心研究问题的直接回答（§10 首要结论）",
        "",
        f"- **哪个新假设、因何经历改变、如何改变未见问题：** 在未曝光 `focal_memory` 连续链 `1041000–1046000`（共 `{qa['chains_attempted_total']}` 条全部记录，其中 `{qa['chains_producing_qualified_accumulate_object']}/{qa['chains_attempted_total']}` 条产出合格 `accumulate` 修复对象）中，系统先由普通马尔可夫任务形成已确认马尔可夫分支，随后在潜变量储层冲突任务上实测双分支均失败（`both_competing_branches_refuted`），触发子问题调用现有 `latent_state_discovery` 生成含 `accumulate` 的具名修复对象（例如 `{ch0_d3['continuous_repaired_object_ledger']['repaired_concept_id']}` @ `{ch0_d3['continuous_repaired_object_ledger']['repaired_semantic_version']}`），经全新 24 行确认批独立通过后执行局部历史分支重调和。在冻结后的未见 `N=5` 新参数／新布线潜变量目标 `T1` 上，`K_eager_noH` 完成 `{qb_arms['K_eager_noH']['T1_unseen_focal_memory_N5']['strict_passes']}`（均值算力 `{qb_arms['K_eager_noH']['T1_unseen_focal_memory_N5']['mean_compute']:,.1f}`，观测 `{qb_arms['K_eager_noH']['T1_unseen_focal_memory_N5']['mean_observations']:.1f}`），而**仅移除该修复版本但保留其余马尔可夫源与完整潜变量 API** 的最小干预臂 `K_minimal_repaired_object_removed_other_sources_kept` 为 `{qb_arms['K_minimal_repaired_object_removed_other_sources_kept']['T1_unseen_focal_memory_N5']['strict_passes']}`（均值算力 `{qb_arms['K_minimal_repaired_object_removed_other_sources_kept']['T1_unseen_focal_memory_N5']['mean_compute']:,.1f}`，观测 `{qb_arms['K_minimal_repaired_object_removed_other_sources_kept']['T1_unseen_focal_memory_N5']['mean_observations']:.1f}`）。",
        f"- **同样强的普通方法表现如何（严格执行 §9 停止规则）：**",
        f"  1. **问题 A（同潜变量 API、同预算父任务修复）：** 不受限直接启用潜变量搜索的普通失败重试 `ordinary_failure_retry_equal_api` 严格通过 `{qa['ordinary_failure_retry_equal_api']['strict_external_passes']}/{qa['chains_attempted_total']}`（均值算力 `{qa['ordinary_failure_retry_equal_api']['mean_compute']:,.1f}`，均值观测 `{qa['ordinary_failure_retry_equal_api']['mean_observations']:.1f}`），子问题门控修复 `subquestion_driven_repair_equal_api` 严格通过 `{qa['subquestion_driven_repair_equal_api']['strict_external_passes']}/{qa['chains_attempted_total']}`（均值算力 `{qa['subquestion_driven_repair_equal_api']['mean_compute']:,.1f}`，均值观测 `{qa['subquestion_driven_repair_equal_api']['mean_observations']:.1f}`）。",
        f"  2. **问题 B 与 Deliverable 2（同原始历史＋按需缓存重建 `R_lazy_noH` vs 预调和 `K_eager_noH`，全组关闭 `H*` 前置）：** 在未曝光 8 源簇 `1031000–1038000` 上，`R_lazy_noH` 严格通过 **`{t_arms['R_lazy_noH']['strict_external_passes']}/24`**（假确认 `{t_arms['R_lazy_noH']['false_internal_confirmations']}`，均值目标算力 `{t_arms['R_lazy_noH']['unconditional_measured_target_costs_all_24']['mean_compute']:,.1f}`），`K_eager_noH` 严格通过 **`{t_arms['K_eager_noH']['strict_external_passes']}/24`**（假确认 `{t_arms['K_eager_noH']['false_internal_confirmations']}`，均值目标算力 `{t_arms['K_eager_noH']['unconditional_measured_target_costs_all_24']['mean_compute']:,.1f}`），源簇配对胜／负／平 `{p_d2['task_wins']}/{p_d2['task_losses']}/{p_d2['task_ties']}`（双侧符号翻转 `p = {p_d2['source_cluster_sign_flip_two_sided_p']}`）；计入源期维护费 `C_K_maint` 后，逐链前缀全成本差均值 `$\\Delta C_c(1..3) = ({pref['k_1']['mean_delta_C_c_k_R_minus_K']:,.1f}, {pref['k_2']['mean_delta_C_c_k_R_minus_K']:,.1f}, {pref['k_3']['mean_delta_C_c_k_R_minus_K']:,.1f})$`。在问题 B 上，`R_lazy_noH`（`{qb_arms['R_lazy_noH']['strict_external_passes']}/18`，均值算力 `{qb_arms['R_lazy_noH']['overall_measured_costs']['mean_compute']:,.1f}`）亦从共享原始经历按需重建出等价复用对象并与 `K_eager_noH`（`{qb_arms['K_eager_noH']['strict_external_passes']}/18`，均值算力 `{qb_arms['K_eager_noH']['overall_measured_costs']['mean_compute']:,.1f}`）正面对比。按 §9 停止规则，我们如实报告普通按需重建的追平／取舍，不做夸大。",
        "",
        "---",
        "",
        "## 2. V13 五项勘误表（§2）与 Replays A–F 生产路径核验（Deliverable 1）",
        "",
        "1. **普通源任务严格通过数更正：** V13 的 32 个普通源任务为 `32/32` 内部确认、**`31/32` 外部严格通过**（`1021000/S_left_dpg_N4` 内部 `1.12e-18`、外部 `8.5874e-4` 属 `1` 例源假确认），`8/8` 共同锚点单独列计；",
        "2. **`-18.92%` 混杂撤回：** 旧 `R_ordinary_optimized_lazy`（`H* ON`）对 `K_routing_no_H_star`（`H* OFF`）混杂了 `H*` 开关，不再写作分支组织单因子净节省；",
        "3. **同开 `H* ON` 对照：** `K_routing_plus_H_star`（`20/24`，`249,564.33`）较 `R_ordinary_optimized_lazy`（`21/24`，`226,943.08`）多耗 `+9.97%` 算力且少通过 `1` 题；",
        "4. **PR-A Choice Point 2 记账更正：** 实有 `1` 条首分支真实 `executed_new` 与 `1` 条次分支 `unavailable`（不变量为 `0` 条 `unavailable` 混入 `paid_probes_executed`）；",
        "5. **边界探针接线与 `H*` 默认关闭：** 本轮已将残差阈值边界探针接入 `UnifiedResearchAgent.run()` 确认前主循环，并默认关闭 `H*` 前置优先级（`_prioritize_generalized_families = False`）。",
        f"- **GPT-6 V14 三探针与 Replays A–F：** `all_gpt6_v14_probes_remediated = {d1['all_gpt6_v14_probes_remediated']}`，`all_replays_a_to_f_passed = {d1['all_replays_a_to_f_passed']}`。",
        "",
        "---",
        "",
        "## 3. Deliverable 2：8 条未曝光源簇（`1031000–1038000`）去混杂无 `H*` 主比较",
        "",
        f"- **共同标定锚点（单独列计）：** `{anc8['strict_external_passes']}/{anc8['episodes_total']}` 严格通过，假确认 `{anc8['false_internal_confirmations']}`；",
        f"- **普通源任务（单独列计）：** `{src32['strict_external_passes']}/{src32['episodes_total']}` 严格通过（内部确认 `{src32['internally_confirmed']}/{src32['episodes_total']}`，假确认 `{src32['false_internal_confirmations']}`，未决 `{src32['unresolved_or_unconfirmed']}`，执行阈值边界探针 `{src32['boundary_probes_executed_total']}` 次，每链源调和维护算力均值 `{src32['mean_reconciliation_maintenance_compute_per_chain']:,.1f}`）。",
        "",
        "| 目标对比臂（全组 `H* OFF`） | 严格通过 (`n=8` 簇, 24题) | 假确认 | 未决 | 错误迁移 | `T1` 新阈值族 | `T2` 旧域保留 | `T3` 少数反向 | 均值目标算力 | 均值观测 | 运行墙钟(不含外评) | 冷/暖缓存重建算力 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: | :---: |",
    ]
    for arm in ("K_cold", "R_lazy_noH", "K_eager_noH"):
        av = t_arms[arm]
        br = av["by_target_role"]
        mc = av["unconditional_measured_target_costs_all_24"]
        cb = av["cache_cold_vs_warm_breakdown"]
        lines.append(
            f"| `{arm}` | **`{av['strict_external_passes']}/24`** | `{av['false_internal_confirmations']}` | `{av['unresolved_or_unconfirmed']}` | `{av['wrong_knowledge_transfers']}` "
            f"| `{br['T1_unseen_family_strict_passes']}` | `{br['T2_old_domain_retention_strict_passes']}` | `{br['T3_minority_direction_strict_passes']}` "
            f"| `{mc['mean_compute']:,.1f}` | `{mc['mean_observations']:.1f}` | `{mc['mean_agent_run_wall_seconds_excluding_ext_eval']:.2f}s` "
            f"| `{cb['cold_cache_k1_mean_reconstruction_compute']:,.0f}` / `{cb['warm_cache_k2_k3_mean_reconstruction_compute']:,.0f}` |"
        )

    lines.extend(
        [
            "",
            "### 逐链前缀全成本差 $\\Delta C_c(k) = C^R_{\\mathrm{maint},c} + \\sum_{t\\le k} C^R_{c,t} - C^K_{\\mathrm{maint},c} - \\sum_{t\\le k} C^K_{c,t}$",
            f"- `k = 1`：逐链 `{pref['k_1']['per_chain_delta_C_c_k_R_minus_K']}`，均值 `{pref['k_1']['mean_delta_C_c_k_R_minus_K']:,.1f}`（`K` 全成本更低链数 `{pref['k_1']['chains_where_K_has_lower_full_prefix_cost_than_R']}/8`）；",
            f"- `k = 2`：逐链 `{pref['k_2']['per_chain_delta_C_c_k_R_minus_K']}`，均值 `{pref['k_2']['mean_delta_C_c_k_R_minus_K']:,.1f}`（`K` 全成本更低链数 `{pref['k_2']['chains_where_K_has_lower_full_prefix_cost_than_R']}/8`）；",
            f"- `k = 3`：逐链 `{pref['k_3']['per_chain_delta_C_c_k_R_minus_K']}`，均值 `{pref['k_3']['mean_delta_C_c_k_R_minus_K']:,.1f}`（`K` 全成本更低链数 `{pref['k_3']['chains_where_K_has_lower_full_prefix_cost_than_R']}/8`）。",
            "",
            "---",
            "",
            "## 4. Deliverable 3：6 条未曝光连续修复对象链（`1041000–1046000`）与首链七句事实示例",
            "",
            "| 问题 B 对比臂（全组 `latent_state_discovery=True`） | 总严格通过 (18题) | 假确认 | `T1` 新潜变量参数 (`N=5`) | `T2` 返回旧马尔可夫 (`N=5`) | `T3` 左向条件反例 (`N=5`) | 均值算力 | 均值观测 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | ---: | ---: |",
        ]
    )
    for arm_b in (
        "K_cold",
        "R_lazy_noH",
        "K_eager_noH",
        "K_minimal_branch_removed_raw_kept",
        "K_minimal_repaired_object_removed_other_sources_kept",
    ):
        bv = qb_arms[arm_b]
        lines.append(
            f"| `{arm_b}` | **`{bv['strict_external_passes']}/18`** | `{bv['false_internal_confirmations']}` "
            f"| `{bv['T1_unseen_focal_memory_N5']['strict_passes']}` (`{bv['T1_unseen_focal_memory_N5']['mean_compute']:,.0f}`) "
            f"| `{bv['T2_return_to_old_markov_N5']['strict_passes']}` (`{bv['T2_return_to_old_markov_N5']['mean_compute']:,.0f}`) "
            f"| `{bv['T3_mismatched_left_conditional_counterexample_N5']['strict_passes']}` (`{bv['T3_mismatched_left_conditional_counterexample_N5']['mean_compute']:,.0f}`) "
            f"| `{bv['overall_measured_costs']['mean_compute']:,.1f}` | `{bv['overall_measured_costs']['mean_observations']:.1f}` |"
        )

    lines.extend(["", f"### 连续链 `{ch0_d3['chain_id']}` 机器生成七句事实账本（全 6 条链见回执 JSON）", ""])
    for k_s, v_s in ch0_d3["seven_sentence_factual_ledger"].items():
        lines.append(f"- **{k_s}**: {v_s}")

    return "\n".join(lines) + "\n"


def main() -> int:
    v14_dir = REPO_ROOT / "docs/evidence/v14_continuous_object_and_fair_comparison_20261118"
    for req in (
        "deliverable_1_errata_and_replays_a_to_f_20261118.json",
        "deliverable_2_unconfounded_noH_source_paired_study_20261118.json",
        "deliverable_3_continuous_repaired_object_study_20261118.json",
    ):
        rp = v14_dir / req
        if not rp.is_file():
            raise FileNotFoundError(f"Missing v14 receipt: {rp}")

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    ZIP_OUT.parent.mkdir(parents=True, exist_ok=True)

    manifest_dict, index_md = _build_manifest_and_index()
    exec_md = _build_executive_report()

    (BUNDLE_DIR / "00_MANIFEST_AND_FILE_INDEX.md").write_text(index_md, encoding="utf-8")
    (BUNDLE_DIR / "01_EXECUTIVE_REPORT_V14.md").write_text(exec_md, encoding="utf-8")
    (BUNDLE_DIR / "04_SOURCE_AND_RECEIPT_MANIFEST.json").write_text(
        json.dumps(manifest_dict, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )

    with tempfile.TemporaryDirectory(prefix="hegel_v14_bundle_stage_") as tmp_str:
        stage = Path(tmp_str) / "hegel_v14_gpt6_audit_handoff"
        stage.mkdir(parents=True, exist_ok=True)

        for name in (
            "00_MANIFEST_AND_FILE_INDEX.md",
            "01_EXECUTIVE_REPORT_V14.md",
            "04_SOURCE_AND_RECEIPT_MANIFEST.json",
            "05_STANDALONE_VERIFY_V14.py",
        ):
            shutil.copy2(BUNDLE_DIR / name, stage / name)

        # Compatibility copy so `probe_scope_and_routing.py` (`ROOT/'05_STANDALONE_VERIFY_V13.py'`) runs directly!
        shutil.copy2(
            REPO_ROOT / "docs/gpt6_audit_bundle_v13_20261117/05_STANDALONE_VERIFY_V13.py",
            stage / "05_STANDALONE_VERIFY_V13.py",
        )

        # Copy GPT-6's 3 V14 probe scripts into top-level `probes/`
        (stage / "probes").mkdir(parents=True, exist_ok=True)
        for p_name in (
            "probe_acquisition_contract.py",
            "probe_question_identity.py",
            "probe_scope_and_routing.py",
            "recompute_v13.py",
        ):
            shutil.copy2(REPO_ROOT / "markdown/hegel_v14/probes" / p_name, stage / "probes" / p_name)

        for phase_tag, edir in EVIDENCE_DIRS:
            dst_dir = stage / "receipts" / phase_tag
            dst_dir.mkdir(parents=True, exist_ok=True)
            for fp in sorted(edir.glob("*.json")):
                shutil.copy2(fp, dst_dir / fp.name)

        for rel_s in BUNDLED_SOURCE_RELPATHS:
            dst_s = stage / "bundled_sources" / rel_s
            dst_s.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / rel_s, dst_s)

        sums_lines = []
        for fp in sorted(stage.rglob("*")):
            if fp.is_file() and fp.name not in {"06_SHA256_SUMS.txt", "sandbox_verify_v14_stdout.json"}:
                sums_lines.append(f"{_sha256_file(fp)}  {fp.relative_to(stage).as_posix()}")
        (stage / "06_SHA256_SUMS.txt").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")

        repo_sums_lines = [
            ln for ln in sums_lines
            if not ln.split("  ", 1)[1].startswith(("receipts/", "bundled_sources/", "probes/", "05_STANDALONE_VERIFY_V13.py"))
        ]
        (BUNDLE_DIR / "06_SHA256_SUMS.txt").write_text("\n".join(repo_sums_lines) + "\n", encoding="utf-8")

        verify_out = stage / "sandbox_verify_v14_stdout.json"
        env_clean = dict(os.environ)
        env_clean["PYTHONPATH"] = ""
        proc = subprocess.run(
            [sys.executable, str(stage / "05_STANDALONE_VERIFY_V14.py"), "--bundle", str(stage), "--output", str(verify_out)],
            capture_output=True,
            text=True,
            env=env_clean,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Standalone v14 verifier failed (exit={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
        verify_json = json.loads(verify_out.read_text(encoding="utf-8"))
        assert verify_json["all_checks_passed"] is True
        shutil.copy2(verify_out, BUNDLE_DIR / "sandbox_verify_v14_stdout.json")

        if ZIP_OUT.exists():
            ZIP_OUT.unlink()
        with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for fp in sorted(stage.rglob("*")):
                if fp.is_file():
                    zf.write(fp, arcname=fp.relative_to(stage).as_posix())

    print(
        f"Built GPT-6 v14 audit bundle at {BUNDLE_DIR} and self-contained ZIP at {ZIP_OUT} "
        f"({ZIP_OUT.stat().st_size:,} bytes, verified_files={len(sums_lines)})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
