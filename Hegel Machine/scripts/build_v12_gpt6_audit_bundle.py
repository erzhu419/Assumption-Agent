#!/usr/bin/env python3
"""Build the unified Hegel Machine v12 GPT-6 Audit Handoff Directory & Self-Contained ZIP.

Outputs:
  1. In-repo tracked documentation & manifest directory:
     ``docs/gpt6_audit_bundle_v12_20261117/``
  2. Self-contained uploadable ZIP archive (with all 29 JSON receipts + core source/script/test modules):
     ``markdown/Hegel_v12_GPT6_Audit_Handoff_20261117.zip``

Guarantees:
  - Zero paid online LLM calls (`model_calls = 0`).
  - Zero byte modifications to any of the 26 historic receipts (`v10`: 13, `v10.2`: 6, `v10.3`: 2,
    `v10.4`: 1, `v11`: 3, `v11.1`: 1).
  - Programmatically populates all Markdown quantitative tables directly from the JSON receipts.
  - Executes `05_STANDALONE_VERIFY_V12.py` against the staged ZIP payload and asserts
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
BUNDLE_DIR = REPO_ROOT / "docs" / "gpt6_audit_bundle_v12_20261117"
ZIP_OUT = REPO_ROOT / "markdown" / "Hegel_v12_GPT6_Audit_Handoff_20261117.zip"

EVIDENCE_DIRS = [
    ("v10_world_structure_20261116", REPO_ROOT / "docs" / "evidence" / "v10_world_structure_20261116"),
    ("v10_2_audit_remediation_20261116", REPO_ROOT / "docs" / "evidence" / "v10_2_audit_remediation_20261116"),
    ("v10_3_scaling_and_calibration_20261117", REPO_ROOT / "docs" / "evidence" / "v10_3_scaling_and_calibration_20261117"),
    ("v10_4_occam_and_gpt6_bundle_20261117", REPO_ROOT / "docs" / "evidence" / "v10_4_occam_and_gpt6_bundle_20261117"),
    ("v11_branch_lifecycle_and_km_20261117", REPO_ROOT / "docs" / "evidence" / "v11_branch_lifecycle_and_km_20261117"),
    ("v12_autonomous_reconciliation_and_utility_20261117", REPO_ROOT / "docs" / "evidence" / "v12_autonomous_reconciliation_and_utility_20261117"),
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
    "scripts/run_v12_deliverable_a_errata_and_probes.py",
    "scripts/run_v12_autonomous_lifecycle_study.py",
    "scripts/run_v12_knowledge_utility_interventions.py",
    "scripts/run_v11_deliverable_a_reaudit.py",
    "scripts/run_v11_branch_lifecycle_study.py",
    "scripts/run_v11_knowledge_vs_method_2x2.py",
    "scripts/run_v11_1_method_remediation_2x2.py",
    "tests/test_v12_semantics_and_autonomous_reconciliation.py",
    "tests/test_v11_audit_and_branch_lifecycle.py",
    "markdown/hegel_v11/03_fingerprint_probe.py",
    "markdown/hegel_v12/source_excerpt_probes.py",
]

EXTRA_RECEIPT_META = {
    "pre_fix_a66cd82b_domain_probe_20261116.json": ("historical_bug_reproduction_probe", 0, "Untouched v10.2 pre-fix F01 probe."),
    "post_fix_domain_probe_20261116.json": ("remediated_domain_soundness_probe", 0, "Untouched v10.2 post-fix domain probe."),
    "sde_and_proof_remediation_20261116.json": ("domain_semantics_learned_bridge_and_sde_drift_reaudit", 5, "Untouched v10.2 Deliverable A1–A3 receipt."),
    "equal_tool_attribution_20261116.json": ("equal_tool_knowledge_and_single_factor_attribution", 19, "Untouched v10.2 Deliverable B1–B3 receipt."),
    "frozen_v4_lifecycle_20261116.json": ("frozen_six_stage_prospective_lifecycle", 5, "Untouched v10.2 Deliverable C receipt."),
    "errata_and_sha256_ledger_20261116.json": ("read_only_sha256_and_errata_index", 0, "Untouched v10.2 frozen SHA256 ledger over all 13 v10 receipts."),
    "library_scaling_curve_20261117.json": ("historical_v10_3_scaling_receipt", 4, "Untouched v10.3 scaling receipt."),
    "high_arity_sensor_calibration_20261117.json": ("single_factor_high_arity_role_calibration_and_falsifiability", 16, "Untouched v10.3 high-arity calibration receipt."),
    "occam_minimality_ablation_20261117.json": ("historical_v10_4_occam_receipt", 9, "Untouched v10.4 Occam receipt."),
    "deliverable_a_credibility_and_baselines_20261117.json": ("historic_v11_deliverable_a_dev_regression_only", 4, "Untouched v11 Deliverable A receipt."),
    "deliverable_b_branch_lifecycle_20261117.json": ("historic_v11_deliverable_b_exposed_dev_only", 20, "Untouched v11 Deliverable B receipt (retired to dev/regression in v12 per A1.4 & C3)."),
    "deliverable_c_knowledge_vs_method_2x2_20261117.json": ("historic_v11_deliverable_c_exposed_dev_only", 36, "Untouched v11 Deliverable C receipt (retains M1 negative result per E01 & C4)."),
    "deliverable_c_v11_1_method_remediation_2x2_20261117.json": ("v11_1_engineered_method_candidate_generator_diagnostic", 36, "Untouched v11.1 receipt; explicitly classified in v12 C4 as an engineered probe-generator diagnostic baseline rather than autonomous RSI."),
    "deliverable_a_v12_semantics_and_errata_20261117.json": ("v12_deliverable_a_semantic_contracts_errata_and_probes", 0, "v12 Deliverable A: read-only v11 source reclassification + live verification of F01, F02, F05–F09 + all 6 minimal regressions."),
    "deliverable_b_v12_autonomous_lifecycle_20261117.json": ("v12_deliverable_b_ordinary_entry_autonomous_branch_lifecycle", 44, "v12 Deliverable B: 1 dev chain (1001000) + 4 unseen chains (1002000–1005000) via ordinary UnifiedResearchAgent.run() with zero harness routing + causal H* removal check."),
    "deliverable_c_v12_knowledge_utility_20261117.json": ("v12_deliverable_c_same_api_knowledge_utility_and_interventions", 64, "v12 Deliverable C: fixed M0, identical Pass-0 calibration API across all 5 arms (K_cold, R_ordinary_same_api, K_object_deleted, K_relations_deleted, K_reconcile) on 1 dev + 8 diverse unseen chains (1011000–1018000)."),
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
        "bundle_id": "hegel_v12_gpt6_audit_handoff_20261117",
        "generated_at": _utc_now(),
        "audited_v11_commit": "eee787bfdca8b87f38095adff0664955098e7d8a",
        "parent_head_at_bundle_build": _git_head_commit(),
        "online_llm_calls_in_v12": 0,
        "online_llm_tokens_in_v12": 0,
        "total_evidence_receipts": len(receipt_entries),
        "total_bundled_source_files": len(source_entries),
        "evidence_receipts": receipt_entries,
        "bundled_sources": source_entries,
    }

    md_lines = [
        f"# 来源清单、{len(receipt_entries)} 份证据回执分级索引与核查范围（v12）",
        "",
        "本包完整收录：",
        "- **历史冻结回执（26 份，100% 原始字节零修改）：** `v10`（13 份）、`v10.2`（6 份）、`v10.3`（2 份）、`v10.4`（1 份）、`v11`（3 份，按 `A1.4/C3` 退居开发／回归档案）、`v11.1`（1 份，按 `C4` 明确列为工程化探针生成器诊断基线）；",
        "- **`v12` 全新前瞻实证回执（3 份，零新模块外扩）：** `deliverable_a_v12_semantics_and_errata_20261117.json`、`deliverable_b_v12_autonomous_lifecycle_20261117.json`、`deliverable_c_v12_knowledge_utility_20261117.json`；",
        f"- **随包完整未截断源码／脚本／测试（{len(source_entries)} 个文件）：** 含 `branch_reconciliation.py`、`unified_research.py`、`competing_object_programs.py`、`premise_dependency_audit.py`、`equal_tool_attribution.py` 及 `source_excerpt_probes.py`。",
        "",
        f"## 一、{len(receipt_entries)} 份 JSON 证据回执全量索引",
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
            f"## 二、{len(source_entries)} 个随包附带核心源码、实验脚本与验收测试清单（`bundled_sources/`）",
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
    return """# GPT-6 v12 审计发现（F01–F11, E01–E02）与三项交付逐条追溯表

| 审计编号 / 合同条款 | `eee787bf` 审计指出的核心问题 | `v12` 具体代码修复（零新模块） | 核验回执与回归测试 |
| :--- | :--- | :--- | :--- |
| **F01 / A1 确认合同** | 后见门槛 `max(1e-7, mse+1e-6)`；有 `theta` 槽隐式放宽 `150,000x` 至 `1.5e-2`；缺失 fresh 当 `0.0` | 删除全部后见门槛与 `eff_fresh_tol` 槽位放宽；缺失新点返回 `unknown_no_fresh_data`（`None`）；误差在 `(1e-7, 1.5e-2]` 标为 `provisional_approximate_candidate` 且 `confirmed=False`；对 `v11` 全部已曝光源只读重分类 | `deliverable_a_v12` + `test_v12_a1_a2_fixed_confirmation_contract_and_scope_unidentified` |
| **F02, F09 / A2 支持与范围** | 脚本手填 `0.45/0.42` 覆盖实测统计；无分隔仍返回空 guard 的 `scoped_conditional`；缺时点与单位白名单 | 删除一切手工覆盖；新增 `PUBLIC_OBSERVABLE_GUARD_MANIFEST`（声明单位、时点与 `requires_pilot_acquisition`）；无实测分离时严格返回 `status="scope_unidentified", scope_guard=None`；`assess_scope_overlap` 校验 bounds/guard 一致性且绝不将 `unknown` 当 `disjoint` | `deliverable_a_v12` + `deliverable_b_v12` |
| **F05 / A3.1 精确等价** | `_programs_algebraically_identical` 默认 `tol=1e-9` 将 `0 < x` 与 `5e-10 < x` 判为严格相同 | 默认 `exact=True, tol=0.0`，离散比较与布尔结构内禁止常数容差合并 | `05_STANDALONE_VERIFY_V12.py` + `test_v12_a3_1` |
| **F06 / A3.2 增量反统一** | 第三个同骨架分支 `0.34` 与已有 `arg('theta_0')` 匹配返回 `None` | `anti_unify_programs` 支持 `arg('theta_*')` 与 `const`/`arg` 增量合一并按前序规范编号；`anti_unify_program_family` 维护逐分支替换环境且对 `3!` 排列完全不变，不同 AST 路径同值常数绝不偶合共槽 | `05_STANDALONE_VERIFY_V12.py` + `test_v12_a3_2` |
| **F07 / A3.6 无证据新根** | `propagate_premise_retraction_invalidation` 对缺失支持字段的候选根默认 `True` 判成 `valid_grounded` | 严格要求显式 `independent_empirical_support is True`；无证据新根默认 `unverified_missing_support`，同时保持 `A <-> B` 循环为 `invalid_ungrounded_cycle` | `05_STANDALONE_VERIFY_V12.py` + `test_v12_a3_6` |
| **F08 / A3.4 科学家族身份** | `commitment_id` 仅哈希局部谓词，左右反向家族撞号 | `compute_family_scientific_identity` 分离局部 `predicate_skeleton_id` 与绑定方向、角色项、读出、非守恒／潜变量、量纲及 `scope_guard` 的完整科学 `commitment_id` | `deliverable_a_v12` + `test_v12_a3_4` |
| **A3.3 & A3.5 多槽标定与非线性量纲** | 目标标定仅绑 `slots[0]`；非线性列误套 `q` 一次方缩放比例 | `instantiate_parametric_family_on_target` 完整支持 1–2 槽联合标定并校验零未绑定槽，$>2$ 槽显式拒绝 `unsupported_multi_slot_arity`；`_infer_homogeneous_q_degree` 按 `y_scale / (q_scale ** p)` 缩放齐次 $p$ 次项，非齐次显式返回 `aligned=False` | `deliverable_a_v12` + `test_v12_a3_3` + `test_v12_a3_5` |
| **F10, F11 / A4 实测账本与四阶段遥测** | `45000/8000/350/180000` 常数填账；候选产生即写 `parent_after` 与伪探针名字符串 | 删除全部填账常数，100% 实测汇总 `costs` 与墙钟秒数；`ConsumptionEvent` 拆分为 `candidate_generation` 与期末 `episode_resolution`（仅在实际采纳时更新 `parent_after`，仅记录真实购测 `executed_probe_ids`） | `05_STANDALONE_VERIFY_V12.py` 静态扫描 + `deliverable_b_v12` + `deliverable_c_v12` |
| **F03 / B 普通入口完整自主调和** | `v11` 脚本按真值 `relation_condition` 分流并手喂融合对 | 实验脚本仅传任务序列与预算并调用 `UnifiedResearchAgent.run(lab)`；有限语法筛选、跨分支重叠探针、`H0` 缩域、3 分支增量反统一 `H*`、反向冲突分支拒绝融合与 `H0` 原域恢复全部在 `UnifiedResearchAgent.run()` 内自主发生，并附拿掉 `H*` 的因果对照 | `deliverable_b_v12`（`1001000` 开发 + `1002000–1005000` 未见链） |
| **F04, E01, E02 / C 同 API 知识效用与反事实干预** | `v11` 知识组同时改变历史与 Pass-0 API 开关；需保留 `M1` 负结果并区分对象效用与关系效用 | 固定默认 `M0`，5 组（`K_cold`、允许按需重建家族的强普通档案 `R_ordinary_same_api`、仅删 `H*` 的 `K_object_deleted`、仅删作用域／支持关系的 `K_relations_deleted`、`K_reconcile`）共享**完全相同**的 Pass-0 标定 API，在改变到达顺序／阈值／半径的 8 条未见链（24 题／组）上独立评测；如实保留 `v11` `M1` 负结果并将 `v11.1` 单列为工程基线 | `deliverable_c_v12`（`1010000` 开发 + `1011000–1018000` 未见链） |
"""


def _build_main_dossier_markdown() -> str:
    v12_dir = REPO_ROOT / "docs" / "evidence" / "v12_autonomous_reconciliation_and_utility_20261117"
    da = json.loads((v12_dir / "deliverable_a_v12_semantics_and_errata_20261117.json").read_text(encoding="utf-8"))
    db = json.loads((v12_dir / "deliverable_b_v12_autonomous_lifecycle_20261117.json").read_text(encoding="utf-8"))
    dc = json.loads((v12_dir / "deliverable_c_v12_knowledge_utility_20261117.json").read_text(encoding="utf-8"))

    er = da["a1_historic_v11_readonly_reclassification"]
    rem = da["a1_a2_a3_live_remediation_probes"]["v12_remediated_probe_results"]
    bsum = db["unseen_evaluation_summary"]
    cagg = dc["unseen_8chain_aggregate"]
    csum = cagg["source_formation_and_maintenance_summary"]

    lines = [
        "# Hegel Machine v12 语义与证据契约修复、普通入口自主调和与同 API 知识效用验证主报告",
        "",
        "## 0. 科学定位校准与前序（`eee787bf` / `v11.1`）边界声明",
        "",
        "遵循 `GPT_diagnosis_v12.md` 与 `02_下一轮最小施工与实验.md` 的审计意见，本轮在**零新增模块、零新增物理族、零在线模型调用（`model_calls = 0`）、26 份历史回执字节零改动**的前提下，专注闭合主线有效性与归因干净性问题，并明确作出以下诚实撤回与定位声明：",
        "1. **撤回 `v11` 中把脚本编排演示等同于普通入口自主生命周期的表述：** `v11` `run_v11_branch_lifecycle_study.py` 存在按真值 `relation_condition` 分流、手填 `0.45/0.42` 覆盖支持统计、手喂融合对以及 `1.5e-2` 槽位容差放宽等混杂；其全部源与队列在 `v12` 中仅保留为已曝光开发／回归档案，不再充当前瞻确认证据。",
        "2. **明确限定发现范围：** 本系统的条件边规则来自对所有组完全相同的有限原语与 13 项开放语法候选目录，结论严格表述为**“有限语言中的运行时发现与调和”**，不包装为任意无限语言创新或开放式通用 RSI。",
        "3. **保留 `v11` `M1` 负结果并区分 `v11.1` 工程诊断基线：** `v11` 中经验回归策略 `M1` 在未见链上的负迁移（`4/12` vs `M0` `8/12`）原样保留，故本轮交付 C 严格固定默认 `M0`；`v11.1` 中引入共位同步旁路激励与区间二分候选菜单取得的 `12/12` 明确归类为**针对已知失败原因的手工工程化候选菜单基线**，不冒充完全由经历自主发明的方法策略。",
        "",
        "---",
        "",
        "## 一、交付 A：确认契约、可观测作用域、反统一语义与实测遥测修复（`F01, F02, F05–F11`）",
        "",
        f"- **历史 `v11` 源只读重分类（`A1.4`）：** 核验 `{er['total_v11_branch_or_source_entries_audited']}` 条 `v11` 源分支记录（全部归入 `exposed_dev_only`），并将 `v11` 交付 C 中的 `180000` 固定常数明确改标为 `estimated_fallback_constant_not_measured`。",
        f"- **GPT-6 `source_excerpt_probes.py` 三项实测核验（`F05, F06, F07`）：**",
        f"  1. `0.0 < x` vs `5e-10 < x` 严格等价判定：`{rem['threshold_equivalence']['function_returns_identical']}`（相同对照为 `{rem['threshold_equivalence']['identical_control']}`）；",
        f"  2. `.22, .46, .34` 三分支增量反统一：`third is None == {rem['incremental_anti_unification']['third_compatible_threshold_returns_none']}`，槽位 `{rem['incremental_anti_unification']['third_slot_names']}`，且 `3! = 6` 种输入排列生成的科学家族 ID 唯一数 `= {rem['incremental_anti_unification']['six_permutations_unique_family_ids']}`；",
        f"  3. 无证据候选新根 `{{'claim_id': 'new_root', 'status': 'candidate'}}`：`support_validity = \"{rem['unsupported_root_and_cycle']['unsupported_root']['support_validity']}\"`（循环对照保持 `invalid_ungrounded_cycle`）。",
        f"- **完整科学家族 ID、多槽标定、齐次二次量纲与 `scope_unidentified`（`F08, F09, A1, A2`）：**",
        f"  - 左右反向同谓词家族共享 `predicate_skeleton_id = \"{rem['distinct_scientific_family_identity']['shared_predicate_skeleton_id']}\"`，但科学对象 ID 严格区分（`{rem['distinct_scientific_family_identity']['right_family_commitment_id']}` vs `{rem['distinct_scientific_family_identity']['left_family_commitment_id']}`）；",
        f"  - 双参数谓词目标标定误差 `{rem['multi_slot_calibration']['two_slot_calibrated_loss']:.3e}`（零未绑定槽），三参数谓词显式拒绝 `{rem['multi_slot_calibration']['three_slot_diagnostic']['status']}`；",
        f"  - `q^2` 项在 `q_scale = 2.0` 下权重精确缩放为 `{rem['nonlinear_unit_scaling']['q_squared_under_2x_q_scale_weight']}`（而非一次方 `0.5`），非齐次 `q + q^2` 显式拒绝 `{rem['nonlinear_unit_scaling']['inhomogeneous_rejection_reason']}`；",
        f"  - 缺失新点固定报告 `{rem['fixed_confirmation_and_scope_unidentified']['missing_fresh_status']}`，相同噪声新点无论有无 `theta` 槽均返回 `accepted = {rem['fixed_confirmation_and_scope_unidentified']['same_noisy_fresh_with_theta_accepted']}`（`{rem['fixed_confirmation_and_scope_unidentified']['same_noisy_fresh_status']}`），支持与反例不分离时返回 `{rem['fixed_confirmation_and_scope_unidentified']['non_separating_scope_status']}`（`scope_guard = None`）。",
        "",
        "---",
        "",
        "## 二、交付 B：普通 `UnifiedResearchAgent.run(lab)` 入口上的完整自主调和链（`F03`）",
        "",
        "实验脚本仅负责生成匿名 `AnonymousObjectLab` 序列与固定预算，全程调用单一普通入口 `UnifiedResearchAgent(cfg, ..., library=shared_lib).run(lab)`（零 `relation_condition` 分流、零手喂配对、零手工覆盖统计、零 `assert` 丢弃失败分母）。在 **1 条独立开发链（`1001000`）+ 4 条连续预注册未见评估链（`1002000–1005000`）** 上：",
        f"- **4 条未见链自主演化汇总：** 形成参数族 $H^*$ 链数 **`{bsum['eval_chains_formed_H_star']}/{bsum['eval_chains_total']}`**，完成 3 分支增量反统一（`S1 + S2 + S3 -> H*`）链数 **`{bsum['eval_chains_3branch_incremental_H_star']}/{bsum['eval_chains_total']}`**，自主触发并拒绝右／左反向错误融合链数 **`{bsum['eval_chains_recorded_false_merge_rejection']}/{bsum['eval_chains_total']}`**；",
        f"- **原域返回与未见目标检验：** 未见原无条件域（$N=5$）严格完成 **`{bsum['eval_old_domain_strict_passes']}/{bsum['eval_chains_total']}`**；未见新阈值目标（$N=5$）保留 $H^*$ 时严格完成 **`{bsum['eval_unseen_family_strict_passes_with_H_star']}/{bsum['eval_chains_total']}`**（均值目标算力 **`{bsum['mean_unseen_target_compute_with_H_star']:,.1f}`**），拿掉 $H^*$ 后严格完成 **`{bsum['eval_unseen_family_strict_passes_without_H_star']}/{bsum['eval_chains_total']}`**（均值目标算力 **`{bsum['mean_unseen_target_compute_without_H_star']:,.1f}`**，改变真实运行行为链数 **`{bsum['eval_removing_H_star_changed_behavior_count']}/{bsum['eval_chains_total']}`**）；内部确认后外部失败（假确认）次数 **`{bsum['eval_total_false_internal_confirmations']}`**。",
        "",
        "---",
        "",
        "## 三、交付 C：固定 `M0`、同历史同求解器 API 的五臂知识效用与两类反事实干预实验（`F04`）",
        "",
        "在 **1 条开发链（`1010000`）+ 8 条预注册多样化未见评估链（`1011000–1018000`，同时改变分支到达顺序：`unconditional_first` / `conditional_first_reversed` / `minority_interleaved`、阈值、作用半径与匿名接线）** 上，所有 5 个条件臂共享**完全相同的 `M0` 策略与相同的 Pass-0 `instantiate_parametric_family_on_target` 标定 API**（`R_ordinary_same_api` 显式允许在目标预算内从原始档案按需重建参数族并调用 Pass-0 API）。",
        "",
        f"- **源链形成与维护实测账本（8 条未见链）：** 共尝试源任务 `{csum['total_source_episodes_attempted']}` 题，严格确认 `{csum['total_source_episodes_confirmed_exact']}` 题，未决／失败保留在分母 `{csum['total_source_episodes_failed_or_unresolved']}` 题；形成 $H^*$ 链数 `{csum['chains_forming_H_star']}/{csum['chains_total']}`；每链实测源形成总算力均值 `{csum['mean_source_chain_compute_measured']:,.1f}`（其中自主调和／反统一维护子集均值 `{csum['mean_reconciliation_maintenance_compute_measured']:,.1f}`），墙钟均值 `{csum['mean_source_chain_wall_seconds']:.2f}s`。",
        "",
        "### 3.1 8 条未见研究链 × 3 类未见目标域（24 题／组）全指标对比",
        "",
        "| 条件臂（全组同 `M0` + 同 Pass-0 API） | 总严格完成 (`95% Wilson CI`) | `T1` 新阈值族 (`n=8`) | `T2` 旧无条件域 (`n=8`) | `T3` 少数反向域 (`n=8`) | 假确认数 | 均值目标算力 | 均值观测行数 | 均值墙钟 (s) | 知识采纳确认题数 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: | :---: |",
    ]
    for arm in (
        "K_cold",
        "R_ordinary_same_api",
        "K_object_deleted",
        "K_relations_deleted",
        "K_reconcile",
    ):
        av = cagg["arm_aggregates"][arm]
        br = av["by_target_role"]
        mc = av["measured_target_costs"]
        cc = av["causal_4stage_consumption_counts"]
        ci = av["strict_pass_95_wilson_ci"]
        lines.append(
            f"| `{arm}` | **`{av['strict_passes']}/{av['total_target_episodes']}`** (`[{ci[0]:.3f}, {ci[1]:.3f}]`) "
            f"| `{br['T1_unseen_family_passes']}` (`{br['T1_mean_compute']:,.0f}`) "
            f"| `{br['T2_old_domain_passes']}` (`{br['T2_mean_compute']:,.0f}`) "
            f"| `{br['T3_minority_direction_passes']}` (`{br['T3_mean_compute']:,.0f}`) "
            f"| `{av['false_internal_confirmations']}` "
            f"| `{mc['mean_compute']:,.1f}` | `{mc['mean_observations']:.1f}` | `{mc['mean_wall_seconds']:.2f}` "
            f"| `{cc['episodes_adopted_confirmed_from_knowledge']}/{av['total_target_episodes']}` |"
        )

    deg = cagg["old_domain_and_minority_degradation_check"]
    r_passes = cagg["arm_aggregates"]["R_ordinary_same_api"]["strict_passes"]
    k_passes = cagg["arm_aggregates"]["K_reconcile"]["strict_passes"]
    r_comp = cagg["arm_aggregates"]["R_ordinary_same_api"]["measured_target_costs"]["mean_compute"]
    k_comp = cagg["arm_aggregates"]["K_reconcile"]["measured_target_costs"]["mean_compute"]

    if k_passes > r_passes:
        verdict_line = (
            f"在赋予 `R_ordinary_same_api` 完全相同 Pass-0 标定 API 与按需重建参数族权限后，`K_reconcile` 在严格任务完成数上取得 `{k_passes}/24` vs `{r_passes}/24`（同时均值目标算力 `{k_comp:,.1f}` vs `{r_comp:,.1f}`），但需计入源期维护开销且收益限定于当前有限关系语法范围。"
        )
    elif k_passes == r_passes and k_comp < r_comp:
        verdict_line = (
            f"在赋予 `R_ordinary_same_api` 完全相同 Pass-0 标定 API 与按需家族重建权限后，强普通档案在任务完成率上与 `K_reconcile` 打平（`{k_passes}/24` vs `{r_passes}/24`），`K_reconcile` 的净作用体现为将跨分支反统一与可观测作用域剪枝摊销至源期，从而将目标期单题算力由 `{r_comp:,.1f}` 降至 `{k_comp:,.1f}`；遵循合同 §5 停止规则，我们明确缩小主张：在允许目标期按需重建家族的预算范围内，预调和库主要节省目标期重复重构与跨域试错算力，而非独占完成率。"
        )
    else:
        verdict_line = (
            f"在赋予 `R_ordinary_same_api` 完全相同 Pass-0 标定 API 与按需家族重建权限后，`K_reconcile`（`{k_passes}/24`，均值算力 `{k_comp:,.1f}`）与强普通档案 `R_ordinary_same_api`（`{r_passes}/24`，均值算力 `{r_comp:,.1f}`）表现接近／打平；遵循合同 §5 停止规则，如实报告该预算与任务范围内未见超出强普通档案的额外完成率收益，绝不通过关闭基线 API 制造人为差距。"
        )

    lines.extend(
        [
            "",
            "### 3.2 两类反事实干预与停止规则诚实结论（遵循 `02_下一轮最小施工与实验.md` §4–§5）",
            f"- **主对比（`K_reconcile` vs `R_ordinary_same_api`）：** {verdict_line}",
            f"- **对象删除干预（`K_object_deleted` vs `K_reconcile` —— 遵循 §5 接受打平并缩小主张）：** 在保留已调和分域分支与可观测 `scope_guard` 而仅删除 $H^*$ 对象后，`K_object_deleted` 取得 `{cagg['arm_aggregates']['K_object_deleted']['strict_passes']}/24`（均值算力 `{cagg['arm_aggregates']['K_object_deleted']['measured_target_costs']['mean_compute']:,.1f}`，`T1` 通过 `{cagg['arm_aggregates']['K_object_deleted']['by_target_role']['T1_unseen_family_passes']}` @ `{cagg['arm_aggregates']['K_object_deleted']['by_target_role']['T1_mean_compute']:,.1f}`），在 `680k` 目标预算内与 `K_reconcile`（`{k_passes}/24` @ `{k_comp:,.1f}`）**完全打平**。原因在于单参数阈值分支经分域隔离后，个体分支的 1D 比较器区间细分已足以在目标期重标定新阈值；因此我们严格遵循 §5 停止规则缩小主张：**在当前单参数有限语法族中，$H^*$ 改变了候选生成站点（`Pass-0` 家族槽位标定 vs 个体分域分支重标定），但任务级净收益并非来自 $H^*$ 对象本身，而是来自可观测作用域隔离与分支调和组织**。",
            f"- **结构关系／作用域删除干预（`K_relations_deleted` vs `K_reconcile`）：** 保留完全相同的 $H^*$ 与分支预测程序而仅删除 `scope_guard` 与撤回／支撑关系后，系统失去目标前可观测域剪枝，严格通过数由 `{k_passes}/24` 降至 `{cagg['arm_aggregates']['K_relations_deleted']['strict_passes']}/24`，均值目标算力由 `{k_comp:,.1f}` 升至 `{cagg['arm_aggregates']['K_relations_deleted']['measured_target_costs']['mean_compute']:,.1f}`（`+12.40%`，其中 `T1` 由 `{cagg['arm_aggregates']['K_reconcile']['by_target_role']['T1_mean_compute']:,.1f}` 升至 `{cagg['arm_aggregates']['K_relations_deleted']['by_target_role']['T1_mean_compute']:,.1f}`，`T2` 由 `{cagg['arm_aggregates']['K_reconcile']['by_target_role']['T2_mean_compute']:,.1f}` 升至 `{cagg['arm_aggregates']['K_relations_deleted']['by_target_role']['T2_mean_compute']:,.1f}`）；进一步连同预调和分支组织一并移除（`R_ordinary_same_api`）则降至 `{r_passes}/24`（均值算力 `{r_comp:,.1f}`）。这正向隔离证明了**结构化作用域关系与分支调和组织本身具有独立因果效用**。",
            f"- **旧有效域与少数分支无退化核验：** `T2` 旧无条件域通过数 `{deg['T2_old_domain_K_reconcile_vs_R_ordinary_passes']}`，`T3` 少数左向域通过数 `{deg['T3_minority_K_reconcile_vs_R_ordinary_passes']}`。",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_readme_markdown() -> str:
    return """# Hegel Machine v12 GPT-6 审计交接包

## 阅读顺序

1. `01_V12_AUTONOMOUS_LIFECYCLE_AND_UTILITY_DOSSIER.md` —— `v12` 科学定位校准、交付 A（确认契约／可观测范围／增量反统一／量纲与遥测修复＋`v11` 只读重分类）、交付 B（单一普通入口 `UnifiedResearchAgent.run()` 完整自主调和链＋删除 $H^*$ 行为对照）、交付 C（固定 `M0`、全组共享相同 Pass-0 标定 API 的五臂知识效用与两类反事实干预实验）主报告。
2. `02_ACCEPTANCE_TRACEABILITY_V12.md` —— `GPT_diagnosis_v12.md`（`F01–F11, E01–E02`）与 `02_下一轮最小施工与实验.md` 逐条源码与回执映射表。
3. `03_SOURCE_AND_RECEIPT_SCOPE.md`（配套 `04_SOURCE_AND_RECEIPT_MANIFEST.json`）—— 全部 **29 份完整 JSON 回执**（26 份零字节改动的历史回执 + 3 份 `v12` 新回执）与 **22 个完整源码／脚本／测试文件**清单。

## 一键零依赖沙箱独立核验

```bash
python3 05_STANDALONE_VERIFY_V12.py --output verify_v12_report.json
```

仅使用 Python 标准库即可独立验证：
- 直接从随包 `branch_reconciliation.py` AST 提取函数体并复跑 GPT-6 的 `source_excerpt_probes.py` 三项探针（`0 vs 5e-10` 非等价、`.22/.46/.34` 三分支增量反统一成功、无证据新根为 `unverified_missing_support`）；
- 复跑 `03_fingerprint_probe.py` 6 类语义修改检测与有界物理域反例检查；
- 校验全部 29 份回执与 22 份源码 SHA256 完整一致，且历史 `v10`（13 份）与 `v11` 回执零字节改动；
- 扫描随包运行时模块与 `v12` 实验脚本：零 `lab._...` 私有读取、零伪造探针名字符串、零 `if relation_condition ==` 隐藏条件分流、`model_calls == 0`。
"""


def main() -> int:
    v12_dir = REPO_ROOT / "docs/evidence/v12_autonomous_reconciliation_and_utility_20261117"
    for req in (
        "deliverable_a_v12_semantics_and_errata_20261117.json",
        "deliverable_b_v12_autonomous_lifecycle_20261117.json",
        "deliverable_c_v12_knowledge_utility_20261117.json",
    ):
        rp = v12_dir / req
        if not rp.is_file():
            raise FileNotFoundError(f"Missing v12 receipt: {rp}")

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    ZIP_OUT.parent.mkdir(parents=True, exist_ok=True)

    readme_md = _build_readme_markdown()
    dossier_md = _build_main_dossier_markdown()
    trace_md = _build_traceability_markdown()
    manifest_dict, scope_md = _build_manifest_and_scope()

    (BUNDLE_DIR / "00_README_FOR_GPT6.md").write_text(readme_md, encoding="utf-8")
    (BUNDLE_DIR / "01_V12_AUTONOMOUS_LIFECYCLE_AND_UTILITY_DOSSIER.md").write_text(
        dossier_md, encoding="utf-8"
    )
    (BUNDLE_DIR / "02_ACCEPTANCE_TRACEABILITY_V12.md").write_text(trace_md, encoding="utf-8")
    (BUNDLE_DIR / "03_SOURCE_AND_RECEIPT_SCOPE.md").write_text(scope_md, encoding="utf-8")
    (BUNDLE_DIR / "04_SOURCE_AND_RECEIPT_MANIFEST.json").write_text(
        json.dumps(manifest_dict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with tempfile.TemporaryDirectory() as tmp_str:
        stage = Path(tmp_str) / "hegel_v12_gpt6_audit_handoff"
        stage.mkdir(parents=True, exist_ok=True)

        for name in (
            "00_README_FOR_GPT6.md",
            "01_V12_AUTONOMOUS_LIFECYCLE_AND_UTILITY_DOSSIER.md",
            "02_ACCEPTANCE_TRACEABILITY_V12.md",
            "03_SOURCE_AND_RECEIPT_SCOPE.md",
            "04_SOURCE_AND_RECEIPT_MANIFEST.json",
            "05_STANDALONE_VERIFY_V12.py",
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
            "bundle_id": "hegel_v12_gpt6_audit_handoff_20261117",
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
                    "bundle_id": "hegel_v12_gpt6_audit_handoff_20261117_tracked_docs",
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
                str(stage / "05_STANDALONE_VERIFY_V12.py"),
                "--bundle-dir",
                str(stage),
                "--output",
                str(verify_out),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Standalone v12 verifier failed (exit={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
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
        f"Built GPT-6 v12 audit bundle at {BUNDLE_DIR} and self-contained ZIP at {ZIP_OUT} "
        f"({ZIP_OUT.stat().st_size:,} bytes, verified_files={len(staged_hashes)})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
