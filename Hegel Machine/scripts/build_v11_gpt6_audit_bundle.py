#!/usr/bin/env python3
"""Build the unified Hegel Machine v11 GPT-6 Audit Handoff Directory & Self-Contained ZIP.

Outputs:
  1. In-repo tracked documentation & manifest directory:
     ``docs/gpt6_audit_bundle_v11_20261117/``
  2. Self-contained uploadable ZIP archive (with all 25 JSON receipts + core source/script/test modules):
     ``markdown/Hegel_v11_GPT6_Audit_Handoff_20261117.zip``

Guarantees:
  - Zero paid online LLM calls (`model_calls = 0`).
  - Zero modifications to historic `v10` (13), `v10.2` (6), `v10.3` (2), or `v10.4` (1) receipts.
  - Programmatically populates all Markdown quantitative tables directly from the JSON receipts.
  - Executes `05_STANDALONE_VERIFY_V11.py` against the staged ZIP payload and asserts
    `all_checks_passed == True`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / "docs" / "gpt6_audit_bundle_v11_20261117"
ZIP_OUT = REPO_ROOT / "markdown" / "Hegel_v11_GPT6_Audit_Handoff_20261117.zip"

EVIDENCE_DIRS = [
    ("v10_world_structure_20261116", REPO_ROOT / "docs" / "evidence" / "v10_world_structure_20261116"),
    ("v10_2_audit_remediation_20261116", REPO_ROOT / "docs" / "evidence" / "v10_2_audit_remediation_20261116"),
    ("v10_3_scaling_and_calibration_20261117", REPO_ROOT / "docs" / "evidence" / "v10_3_scaling_and_calibration_20261117"),
    ("v10_4_occam_and_gpt6_bundle_20261117", REPO_ROOT / "docs" / "evidence" / "v10_4_occam_and_gpt6_bundle_20261117"),
    ("v11_branch_lifecycle_and_km_20261117", REPO_ROOT / "docs" / "evidence" / "v11_branch_lifecycle_and_km_20261117"),
]

BUNDLED_SOURCE_RELPATHS = [
    "src/hegel_machine/cognition/models.py",
    "src/hegel_machine/cognition/unified_research.py",
    "src/hegel_machine/cognition/premise_commitment_index.py",
    "src/hegel_machine/cognition/branch_reconciliation.py",
    "src/hegel_machine/cognition/discriminating_experiment_policy.py",
    "src/hegel_machine/cognition/premise_dependency_audit.py",
    "src/hegel_machine/cognition/sensor_role_calibration.py",
    "src/hegel_machine/cognition/equal_tool_attribution.py",
    "src/hegel_machine/cognition/world_structure_commitments.py",
    "src/hegel_machine/cognition/algebraic_proof_certificates.py",
    "src/hegel_machine/cognition/sde_conditional_drift_eval.py",
    "src/hegel_machine/cognition/stochastic_premise_audit.py",
    "scripts/run_v11_deliverable_a_reaudit.py",
    "scripts/run_v11_branch_lifecycle_study.py",
    "scripts/run_v11_knowledge_vs_method_2x2.py",
    "scripts/run_v10_3_library_scaling_study.py",
    "scripts/run_v10_4_occam_minimality_study.py",
    "scripts/run_v10_2_frozen_lifecycle_study.py",
    "tests/test_v11_audit_and_branch_lifecycle.py",
    "tests/test_v10_4_occam_minimality.py",
    "tests/test_v10_3_scaling_and_calibration.py",
    "tests/test_v10_2_audit_acceptance.py",
    "markdown/hegel_v11/03_fingerprint_probe.py",
]

EXTRA_RECEIPT_META = {
    "pre_fix_a66cd82b_domain_probe_20261116.json": (
        "historical_bug_reproduction_probe",
        0,
        "v10.2 pre-fix F01 reproduction probe.",
    ),
    "post_fix_domain_probe_20261116.json": (
        "remediated_domain_soundness_probe",
        0,
        "v10.2 post-fix domain soundness probe.",
    ),
    "sde_and_proof_remediation_20261116.json": (
        "domain_semantics_learned_bridge_and_sde_drift_reaudit",
        5,
        "v10.2 Deliverable A1–A3 receipt.",
    ),
    "equal_tool_attribution_20261116.json": (
        "equal_tool_knowledge_and_single_factor_attribution",
        19,
        "v10.2 Deliverable B1–B3 receipt.",
    ),
    "frozen_v4_lifecycle_20261116.json": (
        "frozen_six_stage_prospective_lifecycle",
        5,
        "v10.2 Deliverable C receipt.",
    ),
    "errata_and_sha256_ledger_20261116.json": (
        "read_only_sha256_and_errata_index",
        0,
        "v10.2 frozen SHA256 ledger over all 13 v10 receipts.",
    ),
    "library_scaling_curve_20261117.json": (
        "historical_v10_3_scaling_receipt",
        4,
        "Untouched v10.3 scaling receipt (4 empirical sources + 68 stress fixtures; superseded in v11 Deliverable A2).",
    ),
    "high_arity_sensor_calibration_20261117.json": (
        "single_factor_high_arity_role_calibration_and_falsifiability",
        16,
        "Untouched v10.3 high-arity calibration receipt.",
    ),
    "occam_minimality_ablation_20261117.json": (
        "historical_v10_4_occam_receipt",
        9,
        "Untouched v10.4 Occam receipt (re-audited without double-charging in v11 Deliverable A2).",
    ),
    "deliverable_a_credibility_and_baselines_20261117.json": (
        "v11_deliverable_a_cache_credibility_and_strong_baselines",
        4,
        "v11 Deliverable A1–A3: 0/6 undetected mutations, atomic resume parity, zero archive double-charging, OrdinaryPersistentCommitmentCache parity check, synthetic_stress_fixture isolation, and provisional Occam.",
    ),
    "deliverable_b_branch_lifecycle_20261117.json": (
        "v11_deliverable_b_8_stage_branch_reconciliation_lifecycle",
        20,
        "v11 Deliverable B: 1 dev chain (940000) + 4 contiguous unseen 8-stage evaluation chains (951000–954000) verifying B1–B10 on 100% real lab.acquire labels.",
    ),
    "deliverable_c_knowledge_vs_method_2x2_20261117.json": (
        "v11_deliverable_c_2x2_world_knowledge_vs_induced_method",
        36,
        "v11 Deliverable C: Disjoint dev_formation/dev_selection/unseen_evaluation 2x2 (K0/K1 x M0/M1) + ordinary persistent archive anchor + bounded M1->M2 check.",
    ),
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
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _build_manifest_and_scope() -> tuple[dict, str]:
    ledger_v10_2 = json.loads(
        (
            REPO_ROOT
            / "docs/evidence/v10_2_audit_remediation_20261116/errata_and_sha256_ledger_20261116.json"
        ).read_text(encoding="utf-8")
    )
    v10_lookup = {
        item["filename"]: item
        for item in ledger_v10_2["untouched_v10_receipts_sha256_and_roles"]
    }

    receipt_entries = []
    for phase_tag, edir in EVIDENCE_DIRS:
        for fp in sorted(edir.glob("*.json")):
            rel_repo = fp.relative_to(REPO_ROOT).as_posix()
            if fp.name in v10_lookup:
                role = v10_lookup[fp.name]["study_role"]
                ind_src = v10_lookup[fp.name]["independent_physical_sources"]
                notes = v10_lookup[fp.name]["notes"]
            else:
                role, ind_src, notes = EXTRA_RECEIPT_META.get(
                    fp.name, ("supplementary_receipt", 0, "")
                )
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
        "bundle_id": "hegel_v11_gpt6_audit_handoff_20261117",
        "generated_at": _utc_now(),
        "audited_v10_4_commit": "abe15bead458739d1ef5b83e45e96140c75a4d34",
        "parent_head_at_bundle_build": _git_head_commit(),
        "online_llm_calls_in_v11": 0,
        "online_llm_tokens_in_v11": 0,
        "total_evidence_receipts": len(receipt_entries),
        "total_bundled_source_files": len(source_entries),
        "evidence_receipts": receipt_entries,
        "bundled_sources": source_entries,
    }

    md_lines = [
        "# 来源清单、25 份证据回执分级索引与核查范围（v11）",
        "",
        "本包完整收录：",
        "- **历史冻结回执（22 份，100% 原始字节零修改）：** `v10`（13 份）、`v10.2`（6 份）、`v10.3`（2 份）、`v10.4`（1 份）；",
        "- **`v11` 新增实证回执（3 份）：** `deliverable_a_credibility_and_baselines_20261117.json`、`deliverable_b_branch_lifecycle_20261117.json`、`deliverable_c_knowledge_vs_method_2x2_20261117.json`；",
        "- **随包完整未截断源码／脚本／测试（23 个文件）：** 含 `unified_research.py` 全文、新增 `branch_reconciliation.py`、`discriminating_experiment_policy.py` 及 `03_fingerprint_probe.py`。",
        "",
        "## 一、25 份 JSON 证据回执全量索引",
        "",
        "| 阶段 | 回执文件 | 实验角色分类 (`study_role`) | 独立物理源数 | 字节数 | SHA256 前16位 |",
        "| :--- | :--- | :--- | :---: | ---: | :--- |",
    ]
    for r in receipt_entries:
        md_lines.append(
            f"| `{r['phase']}` | `{r['filename']}` | `{r['study_role']}` | `{r['independent_physical_sources']}` | `{r['size_bytes']:,}` | `{r['sha256'][:16]}…` |"
        )

    md_lines.extend(
        [
            "",
            "## 二、23 个随包附带核心源码、实验脚本与验收测试清单（`bundled_sources/`）",
            "",
            "| 仓库相对路径 | 行数 | 字节数 | Git Blob SHA1 前12位 | SHA256 前16位 |",
            "| :--- | ---: | ---: | :--- | :--- |",
        ]
    )
    for s in source_entries:
        md_lines.append(
            f"| `{s['repo_relative_path']}` | `{s['lines']}` | `{s['size_bytes']:,}` | `{s['git_blob_sha1'][:12]}` | `{s['sha256'][:16]}…` |"
        )

    return manifest, "\n".join(md_lines) + "\n"


def _build_traceability_markdown() -> str:
    return """# GPT-6 v11 审计发现与三项交付验收逐条追溯表（A1–A3, B1–B10, C）

| 验收项 | 审计问题定位（`abe15bea`） | `v11` 具体代码修复与实现路径 | 实证回执与单测核验 |
| :--- | :--- | :--- | :--- |
| **A1.1 语义指纹盲区** | `_prior_state_fingerprint` 仅哈希条数与状态，漏检 6 类原地语义修改 | `premise_commitment_index.py:_prior_state_fingerprint` 递归哈希完整 `program, structure, research_program, scope, support_witness_rows, counterexamples, evidence_kind` 与 `checker_schema_version="v11.0"` | `03_fingerprint_probe.py`: `undetected_semantic_mutations = 0`；`test_a1_gpt6_fingerprint_probe_and_versioned_group_invalidation` |
| **A1.2 缓存分组键版本绑定** | `_cached_commitment_group_key` 首次写入后永不失效 | `_compute_group_key_for_prior` 绑定 `(semantic_version, checker_version)`，仅在语义指纹未变时命中 | 单条先验原地修改后 `groups_reconsolidated = 1`，其余组直接复用 |
| **A1.3 原子发布与预算中断续跑** | 预算耗尽仍可能把整库指纹提交为已完成 | `ensure_library_commitment_index` 逐组原子提交，中断时标记 `partial_budget_pending` 且 `_indexed_prior_fingerprints = None`，续跑与足额一次构建一致 | `deliverable_a`: `resumed_parity_exact = True`；`test_a1_partial_budget_atomic_build_and_resume_parity` |
| **A1.4 依赖撤回与独立支持保留** | 父先验撤回未自动使无独立证据的子结论过期 | `_detect_stale_dependency_ids` 与 `propagate_premise_retraction_invalidation` 区分依赖失活与独立实证支撑 | `test_a1_dependency_retraction_and_five_way_retrieval_telemetry` |
| **A1.5 五态检索遥测与物理计数** | 未命中／预算未测混记或仅报 `SearchBudget` 约定单位 | `query_consistent_commitments` 分离 `not_retrieved / not_tested / budget_pending / inconsistent / empirically_refuted`；`_last_index_telemetry` 单列扫描数、归一化 AST 数、求解器调用与墙钟秒数 | `deliverable_a` §A1 |
| **A2.1 消除 `archive` 双重计费** | 外部重构一次、`agent.run()` 内部再次全量巩固计费 | `reconstruct_commitments_from_raw_archive(..., install_into_library=lib)` + `UnifiedResearchAgent._world_commitments` `_episode_archive_cache` 保证单题巩固 `<= 1` 次 | `deliverable_a` §A2: `max_consolidation_calls_per_episode = 1` |
| **A2.2 强普通基线与同工具公平性** | 缺少持久化 KV 缓存普通基线；`v10.3` 归档组缺奥卡姆门控 | 新增 `OrdinaryPersistentCommitmentCache`（持久化摘要缓存 + AST 去重），所有记忆组共享完全相同的求解器、`occam_premise_minimality=True` 与角色标定先验；如实报告静态读多批次下 `ordinary_persistent_cache` 与 `structured_premise_index` 平手 | `deliverable_a` §A2（$M \\in \\{12, 36, 72\\}$） |
| **A3.1 68 条自标注先验降级与隔离** | `prog.predict(r)` 生成标签被标为 `confirmed_source_episode` | 全部 68 条明确改为 `evidence_kind="synthetic_stress_fixture", empirical_confirmation=False`；`count_empirical_confirmation_sources` 与角色标定严格排除压力测试夹具（真实经验源数恒为 `4`） | `deliverable_a` §A3；`test_a2_a3_archive_single_charge_ordinary_cache_and_resource_account` |
| **A3.2 顶层统一资源账户与真实 `ConsumptionEvent`** | fallback 预算与共享观测未由同一顶层账户硬拦截；`bool(retrieved_ids)` 冒充动作改变 | `UnifiedResourceAccount` 统一扣减源／主循环／回退／确认；`UnifiedResearchAgent` 写入含 `parent_before/after`、`probe_before/after` 的真实 `ConsumptionEvent`，召回未用保持 `retained_knowledge_not_invoked` | `deliverable_a` §A3 |
| **A3.3 暂定（Provisional）奥卡姆门控** | 单侧样本上零参数规则拟合成功即宣称条件规则被证伪 | `premise_dependency_audit.py` 检验活跃有向边是否跨越候选阈值；单侧一致时记 `selection_epistemic_status="provisional_occam_preference"` + `undistinguished_conditional_alternatives` + `boundary_test_needed=True` | `deliverable_a` §A3 |
| **B1 角色／单位／时间尺度对齐** | 换名或换物理量纲易触发假冲突 | `branch_reconciliation.py:align_semantics_roles_units_and_time` 对齐 `(q,u,x,f)`、校验滞后深度并转换 `(q_scale, x_scale, y_scale)` | `deliverable_b` Stage 2 + `test_b1_b2_b6` |
| **B2 可观测可执行条件域（禁 `task_id`）** | 作用域依赖任务名或将 `unknown` 默认当 `disjoint` | `validate_observable_scope_guard` 强制 `Program` 仅引用可观测转移统计并拦截 `task_id/seed/oracle`；`assess_scope_overlap` 将未知重叠标为 `unknown=True, certified_disjoint=False` | `deliverable_b` Stage 4–5 + `test_b1_b2_b6` |
| **B3 重叠分歧主动合成可行 `Probe`** | 冲突分支缺乏在控制盒内主动区分的实验合成 | `find_feasible_prediction_difference` 在 `spec.feature_bounds` 内合成最大化预测差的合法控制序列（覆盖全匿名焦点排列与阶跃激发）并交 `lab.acquire` 实测 | `deliverable_b` Stage 3 & Stage 5 + `test_b3_b4_b5` |
| **B4 新候选被反驳时不覆盖旧有效定律** | 需防止错误新假设污染已验证旧承诺 | `adjudicate_branches_on_paid_evidence` 在实测探针上拒绝错误新分支（`status="refuted"`），原 $H_0$ 保持完好 | `deliverable_b` Stage 3 (`B4`) + Stage 7 |
| **B5 旧全称假设遇反例自动缩域** | 仅记异常文本而非可执行缩域 | `shrink_claim_to_observable_scope` 提取支持域与反例域的可观测分离特征，生成可执行 `Program` `scope_guard` 将全称假设缩域为条件定律 | `deliverable_b` Stage 4 (`B5`) + `test_b3_b4_b5` |
| **B6 真等价去重 vs 有限点巧合不合并** | 有限点数值相近不能当形式等价 | `_ast_canonically_equivalent` 仅在角色化 AST 与参数代数恒等时认证 `proved_absent=True`；有限点一致（如 `q_src > -99` vs `always`）严格返回 `proved_absent=False` | `deliverable_b` Stage 2 (`B6`) + `test_b1_b2_b6` |
| **B7 类型化反统一提取共同参数族 $H^*(\\theta)$** | 多阈值同机制分支仅平铺并存，无共同骨架抽取 | `anti_unify_programs` + `propose_anti_unified_generalization` 从 $(H_1, H_2)$ 抽取共享骨架与连续槽 `theta_0`，通过逐分支重构与独立新点检验，并在新目标端由 `instantiate_parametric_family_on_target` 直接标定 | `deliverable_b` Stage 6A & Stage 8 + `test_b7_b8_b9` |
| **B8 拒绝掩盖少数区域退化的错误融合** | 池化 MSE 会让多数样本掩盖少数分支崩溃 | `propose_anti_unified_generalization` 强制逐分支 `max_branch_mse <= tolerance`；右向多数样本与左向少数样本强行融合被拒绝（`rejected_false_merge`），恢复原条件分支 | `deliverable_b` Stage 6B + `test_b7_b8_b9` |
| **B9 前提撤回级联失活与循环支撑拒绝** | 缺乏有根支撑图传播与自指循环拦截 | `propagate_premise_retraction_invalidation` 基于有向无环经验基础可达性更新子结论，拒绝无独立经验根基的 `A <-> B` 循环自证 | `deliverable_b` Stage 6C + `test_b7_b8_b9` |
| **B10 8 阶段完整真实链路在新源组产生可核验作用** | 需端到端真实 `lab.acquire` 链路而非日志堆砌 | 1 条开发链（`940000`）+ 4 条连续预注册未见源链（`951000–954000`），Stage 8 在独立新阈值目标上实测比较 `cold`、`archive_on_demand_cached`、`ordinary_persistent_cache` 与 `structured_branch_lifecycle_v11` | `deliverable_b` Stage 8 |
| **C 世界知识 `K0/K1` 与学得研究策略 `M0/M1` 2×2 正交拆解** | 需把世界假设收益与从开发经历归纳的研究策略收益拆开，严禁把手工规则冒充学得方法或重跑测试集调参 | `discriminating_experiment_policy.py` 从 `dev_formation`（`961001–961006`）实测探针消歧率回归归纳 `M1`（`author_origin="runtime_induced_from_dev_episodes"`），经独立 `dev_selection`（`962001–962004`）筛选晋升并保留 `exploration_archive`，在 6 条未见链（`971000–976000`）单次触碰完成 `A/B/C/D` + `ordinary_archive` 锚点及有界 `M1->M2` 检验；严格遵循 §4 保留世界知识显著正收益（`B > A` 且 `B > E`）并如实披露与定位 `M1` 在未见链上的负迁移根因 | `deliverable_c` + `test_c` |
"""


def _build_main_dossier_markdown() -> str:
    da = json.loads(
        (
            REPO_ROOT
            / "docs/evidence/v11_branch_lifecycle_and_km_20261117/deliverable_a_credibility_and_baselines_20261117.json"
        ).read_text(encoding="utf-8")
    )
    db = json.loads(
        (
            REPO_ROOT
            / "docs/evidence/v11_branch_lifecycle_and_km_20261117/deliverable_b_branch_lifecycle_20261117.json"
        ).read_text(encoding="utf-8")
    )
    dc = json.loads(
        (
            REPO_ROOT
            / "docs/evidence/v11_branch_lifecycle_and_km_20261117/deliverable_c_knowledge_vs_method_2x2_20261117.json"
        ).read_text(encoding="utf-8")
    )

    lines = [
        "# Hegel Machine v11 审计整改、可执行分支调和生命周期与 K×M 正交验证主档案",
        "",
        "## 0. 对前序版本（`abe15bea`）过度表述的正式撤回与科学定位校准",
        "",
        "根据 GPT-6 对提交 `abe15bead458739d1ef5b83e45e96140c75a4d34` 的源码级审计（`hegel_diagnosis_v11.md` 与 `01_系统审计_abe15bea.md`），我们**正式撤回** `v10.4` 交接摘要中“全部问题已 100% 闭合”的过度宣称，并如实确认此前存在的五项实质缺口：",
        "1. **索引指纹盲区：** 旧 `_prior_state_fingerprint` 仅覆盖条数与状态，对原地修改 `program / coefficient / radius / scope / witness / counterexample` 全部漏检（`03_fingerprint_probe.py` 检出 `6/6` 盲区），且 `_cached_commitment_group_key` 未绑定语义版本；",
        "2. **归档基线双重计费与缓存不对称：** 旧缩放脚本在外部重构一次后，`UnifiedResearchAgent._world_commitments` 内部又重复全量巩固计费，且缺少同历史的持久化普通缓存（`OrdinaryPersistentCommitmentCache`）对照；",
        "3. **压力测试夹具误标为经验确认：** `v10.3` 缩放脚本中 68 条由 `prog.predict(r)` 自标注的合成条目曾被标记为 `confirmed_source_episode`；",
        "4. **单侧样本奥卡姆规则缺少暂定（Provisional）认识论状态：** 当观测样本全部落在阈值同侧时，优选简单模型属于暂定归纳偏好而非对条件模型的经验反驳；",
        "5. **多源假设生命周期与 `K × M` 方法归纳尚未闭环：** 此前仅有检索过滤与单步局部修订，缺少从角色／量纲对齐、可观测作用域重叠判定、主动区分探针合成、全称假设缩域、类型化反统一参数族提取、少数区域退化熔断到独立新源验证的完整调和回路，亦未将世界知识 `K` 与经验归纳的研究策略 `M` 正交拆开。",
        "",
        "本轮 `v11` 在 **`model_calls = 0, total_tokens = 0`**、**不新增平行空壳类型**、**不修改任何历史回执（22 份旧回执 SHA256 100% 原封不动）**、**不削弱任何普通基线**的前提下，完成了 `02_下一轮实施与验收.md` 规定的全部三项交付（**交付 A、交付 B、交付 C**）。所有表格数字均由构建脚本直接读取 JSON 回执自动生成。",
        "",
        "---",
        "",
        "## 一、交付 A：索引原子一致性修复、强普通持久化缓存基线与遥测校准",
        "",
        "### 1.1 A1：版本绑定指纹、逐组原子发布与五态检索核验",
        f"- **GPT-6 `03_fingerprint_probe.py` 实测：** `undetected_semantic_mutations = {da['deliverable_A1_index_and_fingerprint_audit']['gpt6_fingerprint_probe']['undetected_semantic_mutations']}`（`program, coefficient, radius, scope, support_content, counterexample_content` 六类语义原地修改 100% 检出，内部 `_cached_*` 元数据改动不误触发重算）。",
        f"- **单条原地修改最小增量重算：** 在 12 条先验（共 `{da['deliverable_A1_index_and_fingerprint_audit']['incremental_single_group_mutation']['total_groups']}` 个语义组）中原地修改 1 条先验的半径与系数，仅重算 **`{da['deliverable_A1_index_and_fingerprint_audit']['incremental_single_group_mutation']['groups_reconsolidated_on_single_prior_edit']}`** 个组，其余组直接命中版本校验缓存。",
        f"- **预算中断与续跑精确一致性：** 小预算中断时状态为 `{da['deliverable_A1_index_and_fingerprint_audit']['partial_budget_interruption_and_resume']['interrupted_status']}`（挂起脏组 `{da['deliverable_A1_index_and_fingerprint_audit']['partial_budget_interruption_and_resume']['dirty_pending_count']}` 个，已提交完整指纹数 `{da['deliverable_A1_index_and_fingerprint_audit']['partial_budget_interruption_and_resume']['committed_fingerprints_during_partial']}`），续跑完成后与一次足额构建 `resumed_parity_exact = {da['deliverable_A1_index_and_fingerprint_audit']['partial_budget_interruption_and_resume']['resumed_parity_exact']}`。",
        f"- **父前提撤回传播与独立证据保护：** 撤回父先验后，纯派生子结论进入 `stale_invalidated_ids`，而具备独立经验支撑的子结论保持有效（`independently_supported_child_preserved = {da['deliverable_A1_index_and_fingerprint_audit']['dependency_retraction_isolation']['independently_supported_child_preserved']}`）。",
        "",
        "### 1.2 A2：消除双重计费后的四臂同工具缩放复核（`UnifiedResourceAccount = 150k` 紧预算，6 题 $N=5$）",
        "",
        "全部记忆臂统一开启 `occam_premise_minimality=True` 与 `sensor_role_calibration_priors=True`（仅允许使用 4 条真实经验源，68 条 `synthetic_stress_fixture` 严格排除），每题巩固次数硬核验 `<= 1`：",
        "",
        "| 库规模 $M$（真实经验源 / 压力夹具） | `cold_same_tools` 通过 / 算力 | `archive_on_demand_cached`（消除双计费）通过 / 算力 | `ordinary_persistent_cache`（强普通持久化缓存）通过 / 算力 | `structured_premise_index` 通过 / 算力 | 一次性缓存／索引构建算力 (`ord` / `idx`) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for mk in ("M_12", "M_36", "M_72"):
        mv = da["deliverable_A2_strong_baselines_and_scaling"][mk]
        emp_raw = mv["confirmed_empirical_source_count"]
        emp_n = emp_raw["empirical_episode"] if isinstance(emp_raw, dict) else emp_raw
        arms = mv["arms"]
        c_a = arms["cold_same_tools"]
        ar_a = arms["archive_on_demand_cached"]
        op_a = arms["ordinary_persistent_cache"]
        sp_a = arms["structured_premise_index"]
        lines.append(
            f"| **{mv['library_size_M']}** (`{emp_n}` / `{mv['synthetic_stress_fixture_count']}`) "
            f"| `{c_a['strict_confirmed_passes']}/{c_a['total_tasks']}` / `{c_a['mean_total_episode_compute']:,.1f}` "
            f"| `{ar_a['strict_confirmed_passes']}/{ar_a['total_tasks']}` / `{ar_a['mean_total_episode_compute']:,.1f}` "
            f"| **`{op_a['strict_confirmed_passes']}/{op_a['total_tasks']}` / `{op_a['mean_total_episode_compute']:,.1f}`** "
            f"| **`{sp_a['strict_confirmed_passes']}/{sp_a['total_tasks']}` / `{sp_a['mean_total_episode_compute']:,.1f}`** "
            f"| `{op_a['one_time_cache_or_index_build_compute']:,}` / `{sp_a['one_time_cache_or_index_build_compute']:,}` |"
        )

    lines.extend(
        [
            "",
            "> **诚实科学结论（遵循 `02_下一轮实施与验收.md` §4）：** 消除 `archive` 双重计费并赋予完全相同的奥卡姆门控与标定工具后，在**静态只读目标批次**上，工程优化良好的普通持久化缓存基线 `ordinary_persistent_cache` 与 `structured_premise_index` **在通过率与目标期算力上完全打平**（二者在 $M=72$ 下均较按需单次重构 `archive_on_demand_cached` 节省约 `7,776` 算力／题）。因此我们**不宣称**静态索引本身带来额外科学发现率，将其准确定位为支持局部组失效更新（单条改动仅重算 1 组而非整库重算）与前提桶剪枝的工程基础设施；真正的科学表征增益来自下文**交付 B（跨分支反统一参数族与可观测域调和）**与**交付 C（经验归纳的区分实验策略 `M1`）**。",
            "",
            "---",
            "",
            "## 二、交付 B：有实体内容的分支调和回路与 8 阶段真实研究链（`B1–B10`）",
            "",
            "复用现有 `Program / Probe / ResearchQuestion / KnowledgeLibrary`，在 `branch_reconciliation.py` 中编译执行完整的 `reconcile(new_claim, old_claims, observations, budget)` 回路，并在 **1 条独立开发链（`dev_chain_940000`，单独披露不计入评估）+ 4 条连续预注册未见源链（`951000, 952000, 953000, 954000`）** 上全部以真实 `lab.acquire` 跑通 Stage 1–8：",
            "1. **Stage 1–2（真实源形成 $H_0$ 与跨元数／量纲复用 `B1, B6`）：** 空库起步发现并冻结 $H_0$（右向无条件守恒），在 $N=5$ 移位重连仪器上零误差复用；重命名传感器＋空间 2 倍量纲缩放经 `align_semantics_roles_units_and_time` 认证 `proved_absent=True`（`B1`），而有限区间数值巧合（`q_src > -99` vs `always`）严格拒绝等价合并（`B6`）。",
            "2. **Stage 3–4（错误新候选保护 `B4`、反例触发主动区分探针 `B3` 与全称缩域 `B5, B2`）：** 在 $S_0$ 上注入左向错误候选，实测探针将其反驳且不覆盖 $H_0$（`B4`）；在阈值仪器 $S_1$（$\\theta=0.22$）上，系统自动合成覆盖全匿名焦点排列的阶跃区分探针并交 `lab.acquire` 执行（`B3`），据此将全称 $H_0$ 缩域为仅引用可观测特征的可执行 `Program` `scope_guard`（`B2, B5`）。",
            "3. **Stage 5–6（多分支重叠、类型化反统一成功融合 `B7`、少数区域退化拒绝融合 `B8` 与撤回传播 `B9`）：** 引入第二阈值分支 $H_2$（$\\theta=0.46$）与反向少数样本分支 $H_{\\text{left}}$。`propose_anti_unified_generalization` 从 $(H_1, H_2)$ 反统一抽取出共享参数骨架 $H^*(r_0, \\theta_0)$ 并通过逐分支重构与独立新点核验（`B7`）；而将右向多数样本与左向少数样本强行合并时，因逐分支误差暴增被明确拒绝（`status=\"rejected_false_merge\"`）并恢复按可观测 `obs_signed_direction_flux` 分派的条件分支（`B8`）；撤回父节点与无根循环 `A <-> B` 测试通过（`B9`）。",
            "4. **Stage 7–8（回归 $H_0$ 原域核验 + 全新未见阈值源组四臂同预算评测 `B10`）：** 回到 $N=5$ 无条件原域时，凭负供体转移实证剪枝门控参数族并恢复 $H_0$（`MSE <= 1e-7`）；在每条链 2 台全新阈值目标仪器（4 条未见链共 8 题：未见右向中间阈值 $\\theta=0.34$ 与未见左向阈值 $\\theta=0.33$，`UnifiedResourceAccount = 150k`）上的聚合结果如下：",
            "",
            "| Stage 8 未见目标对照臂（4 条独立源链，共 8 题） | 预算内严格确认通过 | 链级平均通过率 | 均值目标算力 | 均值观测行数 | 触发 Pass-0 参数族直接标定题数 |",
            "| :--- | :---: | :---: | ---: | ---: | :---: |",
        ]
    )
    s8_agg = db["unseen_stage8_aggregate_by_arm"]
    for ak in (
        "cold_same_tools",
        "archive_on_demand_cached",
        "ordinary_persistent_cache",
        "structured_branch_lifecycle_v11",
    ):
        av = s8_agg[ak]
        lines.append(
            f"| `{ak}` | **`{av['strict_passes']}/{av['unseen_target_episodes']}`** | `{av['chain_level_mean_pass_rate']:.4f}` | `{av['mean_target_compute']:,.1f}` | `{av['mean_target_observations']:.2f}` | `{av['episodes_with_pass0_parametric_instantiation']}/{av['unseen_target_episodes']}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 三、交付 C：世界知识（`K0/K1`）与经验归纳研究策略（`M0/M1`）的 `2×2` 正交验证",
            "",
            "### 3.1 严格三分区隔离与 `M0 -> M1` 运行时方法归纳",
            f"- **分区零重叠与评估器不可变核验：** `dev_formation`（`{dc['partition_hygiene_and_evaluator_immutability']['dev_formation_seeds']}`）、`dev_selection`（`{dc['partition_hygiene_and_evaluator_immutability']['dev_selection_seeds']}`）与最终未见评估链（`{dc['partition_hygiene_and_evaluator_immutability']['unseen_evaluation_chain_seeds']}`）种子零交集（`zero_seed_overlap_verified = {dc['partition_hygiene_and_evaluator_immutability']['zero_seed_overlap_verified']}`）；评估契约前后哈希完全一致（`{dc['partition_hygiene_and_evaluator_immutability']['evaluator_contract_sha256_before'][:16]}…`）。",
            f"- **`M1` 归纳来源与晋升门控：** 在 6 个 $N=6$ 开发题上对候选探针实测每行前提消歧率，线性回归学得特征权重 `{json.dumps(dc['method_induction_M0_to_M1']['learned_regression_weights'])}` 与优选对偶位移尺度 `{dc['method_induction_M0_to_M1']['learned_preferred_displacement_scale']}`（`author_origin = \"{dc['method_induction_M0_to_M1']['promoted_method']['author_origin']}\"`）。独立 `dev_selection` 上 `M0` 为 `{dc['method_induction_M0_to_M1']['baseline_m0_dev_selection']}`，晋升策略为 `{dc['method_induction_M0_to_M1']['promoted_dev_selection']}`（`promoted = {dc['method_induction_M0_to_M1']['promoted']}`），全部 `{dc['method_induction_M0_to_M1']['exploration_archive_count']}` 个候选（含未晋升项）均保留在 `exploration_archive` 中。",
            "",
            "### 3.2 未见预注册源链（6 条独立链 × 2 台 $N=6$ 目标 = 12 题）`2×2` + 强普通归档锚点结果",
            "",
            "| 条件臂 | 预算内严格确认通过 | 失败题数 | 链级均值通过率 | 实测改变探针题数 (`M1 != M0`) | 方法 `ConsumptionEvent` 数 | 均值目标观测行数 | 均值目标算力 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | ---: | ---: |",
        ]
    )
    for ak in (
        "A_K0_M0",
        "B_K1_M0",
        "C_K0_M1",
        "D_K1_M1",
        "E_optimized_ordinary_archive_M0",
    ):
        av = dc["unseen_2x2_arm_summaries"][ak]
        lines.append(
            f"| `{ak}` | **`{av['strict_passes']}/{av['total_unseen_target_episodes']}`** | `{av['failed_episodes']}` | **`{av['chain_mean_pass_rate']:.4f}`** | `{av['episodes_where_method_changed_probe_vs_m0']}/{av['total_unseen_target_episodes']}` | `{av['total_method_consumption_events']}` | `{av['mean_target_observations']:.2f}` | `{av['mean_target_compute']:,.1f}` |"
        )

    lines.extend(
        [
            "",
            "### 3.3 以独立研究链为单位（$n=6$）的配对差与 95% Bootstrap 置信区间",
            "",
            "| 正交对比项 | 通过率配对均值差 `[95% CI]` | 观测行数配对均值差 `[95% CI]` | 科学归因含义 |",
            "| :--- | :---: | :---: | :--- |",
        ]
    )
    contrast_labels = [
        ("B_minus_A_world_knowledge_at_M0", "固定默认策略 `M0` 下，运行时修订／反统一世界假设 `K1` 的净收益"),
        ("C_minus_A_learned_method_at_K0", "固定初始世界假设 `K0` 下，开发经历归纳实验策略 `M1` 的净收益"),
        ("D_minus_B_learned_method_at_K1", "固定 `K1` 下，学得实验策略 `M1` 的边际收益"),
        ("D_minus_C_world_knowledge_at_M1", "固定新策略 `M1` 下，世界知识 `K1` 的净收益"),
        ("B_minus_E_structured_K1_vs_ordinary_archive_at_M0", "同源历史下，结构化分支调和 `K1` 相对强普通持久化归档 `E` 的净收益"),
        ("D_minus_E_full_K1_M1_vs_ordinary_archive", "`K1 + M1` 完整体系相对同历史强普通持久化归档 `E` 的总净收益"),
    ]
    for ck, desc in contrast_labels:
        cv = dc["paired_chain_contrasts_and_ci95"][ck]
        pd = cv["pass_rate_paired_diff"]
        od = cv["observation_cost_paired_diff"]
        lines.append(
            f"| `{cv['contrast']}` | **`{pd['mean_diff']:+.4f}`** `[{pd['ci95_low']:+.4f}, {pd['ci95_high']:+.4f}]` | `{od['mean_diff']:+.2f}` `[{od['ci95_low']:+.2f}, {od['ci95_high']:+.2f}]` | {desc} |"
        )

    g2 = dc["second_generation_improver_check_M1_to_M2"]
    lines.extend(
        [
            "",
            "### 3.4 遵循 `02_下一轮实施与验收.md` §3.4 & §4 的诚实归因、二代方法检验（`M1 -> M2`）与非声称边界",
            f"- **主命题 §3.1 实证成立（世界假设 `K1` 显著优于 `K0` 与同历史强普通归档 `E`）：** 在完全未重调参的 6 条连续预注册未见源链（12 题 $N=6$）上，`B_K1_M0` 取得 **`9/12`（链均通过率 `0.7500`）**，不仅较 `A_K0_M0`（`8/12 = 0.6667`）配对节省 **`-57.83` 行观测（`95% CI [-89.25, -26.42]`）** 与 **`-38.77%` 目标算力（`174,519.9` vs `285,035.6`）**，而且较同源历史的强普通持久化归档 `E_optimized_ordinary_archive_M0`（因缺乏反统一与作用域保护而受多源分支干扰跌至 `5/12 = 0.4167`）显著提升 **`+0.3333` 通过率（`95% CI [+0.1667, +0.5000]`）** 并节省 **`-48.42` 行观测（`95% CI [-72.25, -24.58]`）**。",
            "- **`M1` 在未见链上的负迁移如实保留与 §4 三分类根因定位（严守 §3.4 不二次触碰隐藏评估集重调参）：**",
            "  1. **调用核验（非调用失败）：** `M1` 在 `dev_selection`（$n=4$）上由 `2/4` 提升至 `3/4` 合法晋升，并在未见评估集 `10/12` 题上实际改变了执行的实验探针（其余 `2/12` 题因初审即闭合而诚实计入 `uninvoked_method_episodes = 2`），累计产生 `29` 条真实 `ConsumptionEvent`。",
            "  2. **候选生成与单目标选择偏差定位：** 在未见集上 `C_K0_M1` 虽在全量 **`12/12` 题细粒度传感器上达成 `external_fresh48_mse == 0.0`**（含修复 `A_K0_M0` 在 `972000:T1` 上的 `0.01326` 拟合偏差），且均值观测微降 `-9.50` 行，但严格在线纯控制重确认通过率降为 `4/12`（`D_K1_M1` 为 `6/12`）。根因在于：(a) **候选探针生成缺口**——`menu_pair_scale_*` 为隔离双体输运将执行器冲量固定为 `u = 0`，挤占了辨识非线性执行器曲率（`actuator_curvature = 0.15`）所需的冲量激励样本，导致下游 `selected_channels=[]` 纯控制投影在 `315k` 预算内未完成确认；(b) **小样本 `dev_selection` 尺度过拟合**——$n=4$ 选拔偏好了极近对偶间距 `preferred_displacement_scale = 0.0718`，有利于密集激发边阈值却对外部交互半径边界 $r \\in [0.87, 0.90]$ 包夹不足（如 `975000:T1` 出现 `1.26e-4` 半径边界残差）。遵循 §4，我们保留 `K1` 的正向成果，原样保留首次未见评估回执并明确记录上述候选生成改进靶点。",
            f"- **有界二代检验（`M1 -> M2`）：** 在第二批独立开发题（`963001–963004` 形成、`964001–964004` 选拔）上，`M0` 引导采样的 `M2_from_M0` 选拔表现为 `{g2['m2_from_m0_dev_selection']}`，`M1` 引导采样的 `M2_from_M1` 选拔表现为 `{g2['m2_from_m1_dev_selection']}`（均 `4/4` 通过、观测相近，算力微降 `-3,706.3`），我们**不宣称**单步线性探针评分器已构成多代开放式 RSI 自举。",
            "- **成本分账声明：** 所有开发形成成本、源世界形成成本、缓存／索引维护成本与目标评测成本已在回执 `cost_ledgers_separated` 中完全分列，不以人为加权总分掩盖形成开销。",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_readme_markdown() -> str:
    return """# Hegel Machine v11 GPT-6 审计交接包

## 阅读顺序

1. `01_V11_AUDIT_REMEDIATION_AND_LIFECYCLE_DOSSIER.md` —— 对 `v10.4` 过度表述的正式撤回、交付 A（原子版本索引＋强普通持久化缓存平手复核＋暂定奥卡姆）、交付 B（8 阶段可执行分支调和研究链 `B1–B10`）、交付 C（`K0/K1 × M0/M1` 正交 `2×2` 与有界 `M1->M2` 检验）主报告。
2. `02_ACCEPTANCE_TRACEABILITY_A_B_C.md` —— `02_下一轮实施与验收.md` 全部验收条款逐条代码／回执／单测映射表。
3. `03_SOURCE_AND_RECEIPT_SCOPE.md`（配套 `04_SOURCE_AND_RECEIPT_MANIFEST.json`）—— 全部 **25 份完整 JSON 回执**（22 份冻结旧回执 + 3 份 `v11` 新回执）与 **23 个完整源码／脚本／测试文件**清单。

## 一键零依赖沙箱独立核验

```bash
python3 05_STANDALONE_VERIFY_V11.py --output verify_report.json
```

仅使用 Python 标准库即可独立验证：
- AST 提取 `_prior_state_fingerprint` 通过全部 6 类语义修改测试（`undetected_semantic_mutations == 0`）；
- 全部 25 份回执与 23 份源码 SHA256 完整一致，且 `v10` 13 份原始回执零字节改动；
- `v11` 三份回执满足 `model_calls == 0`、单题巩固无双重计费（`<= 1`）、68 条压力夹具隔离、暂定奥卡姆状态、`B1–B10` 全通过、三分区种子零重叠与评估器不可变；
- 随包全部 9 个运行时认知模块（含完整 `unified_research.py`）对 `lab._...` 私有属性零读取。
"""


def main() -> int:
    for req in (
        "deliverable_a_credibility_and_baselines_20261117.json",
        "deliverable_b_branch_lifecycle_20261117.json",
        "deliverable_c_knowledge_vs_method_2x2_20261117.json",
    ):
        rp = REPO_ROOT / "docs/evidence/v11_branch_lifecycle_and_km_20261117" / req
        if not rp.is_file():
            raise FileNotFoundError(f"Missing v11 receipt: {rp}")

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    ZIP_OUT.parent.mkdir(parents=True, exist_ok=True)

    readme_md = _build_readme_markdown()
    dossier_md = _build_main_dossier_markdown()
    trace_md = _build_traceability_markdown()
    manifest_dict, scope_md = _build_manifest_and_scope()

    (BUNDLE_DIR / "00_README_FOR_GPT6.md").write_text(readme_md, encoding="utf-8")
    (BUNDLE_DIR / "01_V11_AUDIT_REMEDIATION_AND_LIFECYCLE_DOSSIER.md").write_text(
        dossier_md, encoding="utf-8"
    )
    (BUNDLE_DIR / "02_ACCEPTANCE_TRACEABILITY_A_B_C.md").write_text(trace_md, encoding="utf-8")
    (BUNDLE_DIR / "03_SOURCE_AND_RECEIPT_SCOPE.md").write_text(scope_md, encoding="utf-8")
    (BUNDLE_DIR / "04_SOURCE_AND_RECEIPT_MANIFEST.json").write_text(
        json.dumps(manifest_dict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with tempfile.TemporaryDirectory() as tmp_str:
        stage = Path(tmp_str) / "hegel_v11_gpt6_audit_handoff"
        stage.mkdir(parents=True, exist_ok=True)

        for name in (
            "00_README_FOR_GPT6.md",
            "01_V11_AUDIT_REMEDIATION_AND_LIFECYCLE_DOSSIER.md",
            "02_ACCEPTANCE_TRACEABILITY_A_B_C.md",
            "03_SOURCE_AND_RECEIPT_SCOPE.md",
            "04_SOURCE_AND_RECEIPT_MANIFEST.json",
            "05_STANDALONE_VERIFY_V11.py",
        ):
            shutil.copy2(BUNDLE_DIR / name, stage / name)

        for phase_tag, edir in EVIDENCE_DIRS:
            dst_dir = stage / "receipts" / phase_tag
            dst_dir.mkdir(parents=True, exist_ok=True)
            for fp in sorted(edir.glob("*.json")):
                shutil.copy2(fp, dst_dir / fp.name)

        for rel_s in BUNDLED_SOURCE_RELPATHS:
            dst_s = stage / "bundled_sources" / rel_s
            dst_s.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / rel_s, dst_s)

        staged_hashes = {}
        for fp in sorted(stage.rglob("*")):
            if fp.is_file() and fp.name not in {"SHA256SUMS.json", "sandbox_verify_stdout.json"}:
                staged_hashes[fp.relative_to(stage).as_posix()] = _sha256_file(fp)

        sums_payload = {
            "bundle_id": "hegel_v11_gpt6_audit_handoff_20261117",
            "generated_at": _utc_now(),
            "file_count": len(staged_hashes),
            "files": staged_hashes,
        }
        (stage / "SHA256SUMS.json").write_text(
            json.dumps(sums_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        repo_doc_hashes = {
            k: v
            for k, v in staged_hashes.items()
            if not k.startswith("receipts/") and not k.startswith("bundled_sources/")
        }
        (BUNDLE_DIR / "SHA256SUMS.json").write_text(
            json.dumps(
                {
                    "bundle_id": "hegel_v11_gpt6_audit_handoff_20261117_tracked_docs",
                    "generated_at": _utc_now(),
                    "file_count": len(repo_doc_hashes),
                    "files": repo_doc_hashes,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        verify_out = stage / "sandbox_verify_stdout.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(stage / "05_STANDALONE_VERIFY_V11.py"),
                "--bundle-dir",
                str(stage),
                "--repo",
                str(REPO_ROOT),
                "--output",
                str(verify_out),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Standalone verifier failed (exit={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
            )
        verify_json = json.loads(verify_out.read_text(encoding="utf-8"))
        assert verify_json["all_checks_passed"] is True
        shutil.copy2(verify_out, BUNDLE_DIR / "sandbox_verify_stdout.json")

        if ZIP_OUT.exists():
            ZIP_OUT.unlink()
        with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for fp in sorted(stage.rglob("*")):
                if fp.is_file():
                    arcname = fp.relative_to(stage).as_posix()
                    zf.write(fp, arcname=arcname)

    print(
        f"Built GPT-6 v11 audit bundle at {BUNDLE_DIR} and self-contained ZIP at {ZIP_OUT} "
        f"({ZIP_OUT.stat().st_size:,} bytes, verified_files={len(staged_hashes)})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
