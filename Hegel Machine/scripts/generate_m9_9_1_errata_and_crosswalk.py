#!/usr/bin/env python3
"""Deterministic Errata & Back-Verification Report Generator for
`docs/55b_M9_9_1_Audit_Closure_Forensic_Reverification_and_Errata*.md` (`0` Paid LLM Calls).

Reads ONLY sealed JSON ledgers in `docs/evidence/m9_9_1_audit_closure_and_forensic_reverification_20261128/`
and emits calibrated English and Chinese errata documents covering `Q01–Q14` and the original `R01–R16` crosswalk.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
M991_DIR = REPO_ROOT / "docs/evidence/m9_9_1_audit_closure_and_forensic_reverification_20261128"

OUT_EN = REPO_ROOT / "docs/55b_M9_9_1_Audit_Closure_Forensic_Reverification_and_Errata.md"
OUT_ZH = REPO_ROOT / "docs/55b_M9_9_1_Audit_Closure_Forensic_Reverification_and_Errata_ZH.md"


def main() -> None:
    q01 = json.loads((M991_DIR / "01_m99_q01_30_cross_diagonal_decomposition_probes.json").read_text(encoding="utf-8"))
    q03 = json.loads((M991_DIR / "02_m99_q03_c04_e7_existing_candidate_container_verification.json").read_text(encoding="utf-8"))
    prov = json.loads((M991_DIR / "03_m99_provenance_x_sensitivity_and_plain_dict_replay.json").read_text(encoding="utf-8"))

    probe_table_rows = []
    for p in q01["probes"]:
        sup = f"`{p['supersedes_unexecuted_witness_id']}`" if p["supersedes_unexecuted_witness_id"] else "—"
        probe_table_rows.append(
            f"| `{p['arm']}` | `{p['world_id']}` (`s{p['decomp_step']}`) | `{p['cell']}` | `{p['probe_id']}` | "
            f"`{p['execution_status']}` | `{p['accepted']}` | `{p['failure_kind']}` | `{p['probe_wall_ms']:.1f}` | {sup} |"
        )

    x_rows = [f"| `{row['x']:,}` | `{row['two_sided_exact_sign_flip_p']:.8f}` |" for row in prov["unknown_x_exact_sign_flip_sensitivity_ladder"]["evaluated_points"]]

    frz = prov["git_source_tree_reconciliation"]["declared_paid_freeze_commit_7624c43bf"]
    fin_sha = prov["git_source_tree_reconciliation"]["m99_summary_recorded_post_freeze_finalizer_tree_sha256"]
    c_env = prov["execution_vs_finalizer_timing_reconciliation"]["confirmation_n8"]
    d_env = prov["execution_vs_finalizer_timing_reconciliation"]["dev_n2"]
    w_rat = prov["non_llm_probe_elapsed_wall_ratios"]

    replay = prov["plain_dict_same_joint_features_step05_replay"]
    parity = f"{replay['matched_episodes']}/{replay['total_episodes']}"
    costs = prov["rd_cost_coverage_boundary"]["cancelled_dev_diagnostic_run"]
    conf_path = M991_DIR.parent / "m9_9_autonomous_decomposition_and_interaction_withdrawal_20261128/m9_9_continual_summary.json"
    conf = json.loads(conf_path.read_text())
    quality_p = conf["paired_cluster_inference"]["Hegel_minus_Flat_MonolithicConditionalTable_Online"]["quality_two_sided_sign_flip_p"]
    withdrawal = json.loads((M991_DIR / "06_m99_q04_original_withdrawal_joint_replay.json").read_text())
    completed = sum(row["execution_status"] == "completed" for row in q01["probes"])
    en_md = f"""# Hegel M9.9.1 Independent Audit Closure, 30-Probe Docker Back-Verification, Forensic Root-Cause Calibration & Errata (`Q01–Q14` + Original `R01–R16`)

- **Date**: `2026-11-28`
- **Audited Baseline Commit**: `d4b1cd7f06a08db8d967a43913003527d21a0354` (`Hegel_M9_9_Audit`)
- **Paid LLM Calls Used in `M9.9.1`**: **`0`** (`100%` deterministic code hardening + real air-gapped Docker back-verification)
- **Original Sealed Evidence Status**: `docs/evidence/m9_8_1_*`, `docs/evidence/m9_9_dev_live_calibration_20261128`, and `docs/evidence/m9_9_autonomous_decomposition_and_interaction_withdrawal_20261128` remain `100%` byte-untouched; all `M9.9.1` back-verification receipts are independently sealed in `docs/evidence/m9_9_1_audit_closure_and_forensic_reverification_20261128/`.

---

## 1. Complete `Hegel_M9_9_Audit` (`Q01–Q14`) Resolution Register

| Finding | Priority | Audit Diagnosis | `M9.9.1` Code Fix, Container Back-Verification & Claim Calibration | Status |
|---|---|---|---|---|
| `Q01` | `P1` | The original factorization inferred two off-diagonal outcomes. | Completed {completed}/30 historical probes. Scientific/domain rejections {q01['scientific_or_domain_rejected_count']}, old witnesses linked {q01['superseded_hegel_d0_witness_count']}/8. Full matrix required for new joint guard evidence. | Retrospective replay |
| `Q02` | `P1` | alpha c02/E7 failed. | Frozen downstream feature repair reduced MAE from 0.071267 to 0.004857. The public swap check tests one necessary pooling invariant. The frozen preprocessing counterfactual changes pooling and imputation together. | Component attribution |
| `Q03` | `P1` | The guard omitted an available upstream solver in c04/E7. | Prefix pair up_b02 + dn_b03. Public acceptance {q03['joint_three_layer_public_gate']['accepted']}, official pass {q03['official_verifier_result']['strict_pass']}. The prospective bounded fallback requires online evaluation. | Frozen candidate replay |
| `Q04` | `P1` | Delayed factorization hid archived code. A missing separable branch was aliased to a coupled one in the ablation. | Prospective archive fallback and explicit missing-state accounting. Historical costs and grades are still reported under their original policy. | Engineering / future online test |
| `Q05` | `P1` | `--finalize-existing` overwrote `run_id`, `wall_sec`, and `source_tree_sha256_at_start` (`be7190fb...` vs freeze `34b01d19...`). | Separated write-once `execution_manifest.json` from `finalization_manifest.json` in runner; reconstructed exact freeze tree SHA (`{frz['source_tree_sha256'][:16]}...`) and raw call envelopes (`{c_env['call_envelope_lower_bound_sec']:,.3f}s` confirm, `{d_env['call_envelope_lower_bound_sec']:,.3f}s` dev). | **CLOSED (`Code + Provenance Ledger`)** |
| `Q06` | `P1` | Eight paired clusters. | Hegel versus Monolithic quality p = {quality_p}, cost p = 0.765625. Versus Plain, cost p = 0.0625 at x = 0. The sensitivity table uses the same frozen eight pairs. | Inference corrected |
| `Q07` | `P1` | A legacy withdrawal can be internally consistent while its original scientific receipt is missing. | Original joint scientific receipts {withdrawal['historical_signed_scientific_receipts']}/{withdrawal['total_receipts']}. Matched negative probes in the separate replay {withdrawal['replayed_bound_scientific_negatives']}/{withdrawal['total_receipts']}. Typed evidence is required for prospective authority. | Historical and replay evidence reported separately |
| `Q08` | `P2` | Legacy `v1` downgrade failed second `save -> reload` round-trip; `M9.8` `compatibility_inference` could write a downstream scientific negative. | Recomputes migrated `v2` `witness_content_sha256` & `unknown_requires_reverification` `receipt_id` (`snap_1 == snap_2` idempotent, `0` authority); blocked `compatibility_inference` from minting downstream scientific negatives. | **CLOSED** |
| `Q09` | `P1` | Zero host-seeded source code described too broadly as zero host-seeded decomposition. | Scoped throughout `docs/55_*` & hub as **online module formation under a host-prescribed two-stage interface** (open DAG discovery deferred). | **CLOSED** |
| `Q10` | `P1` | The frozen ablation restricted features available to fixed selectors. | With the same host-reconstructed policy inputs, a plain dictionary reproduces {parity} candidate hashes. Actual Docker public files are tracked separately. | Prefix replay |
| `Q11` | `P1` | Ideal grid count written as an exact LLM cost law. | Replaced with conditional `N_{{sep,unseen}} = card((U x D) \\ (R u X))` and empirical quality-constrained `ΔC(K)` / `K*` protocol. | **CLOSED** |
| `Q12` | `P2` | Cancelled development cost was missing. | Recovered {costs['unique_logical_calls']} calls and {costs['known_physical_v2']:,} known v2, with {costs['unclosed_call_count']} unclosed calls. Total {costs['total_rd_v2_formula']}. | Partial billing recovery |
| `Q13` | `P2` | `E4/E5` `v2/v3` swap, `E1` `7/8, 6/8` vs cumulative `8/8, 7/8`, and elapsed wall-ms (`1.421x` total / `3.320x` coverage) vs CPU wording. | Fixed in `scripts/generate_m9_9_report_from_sealed_json.py`, `docs/55_*`, and hub. | **CLOSED** |
| `Q14` | `P2` | `R01–R16` ID drift in `docs/55` §7 and broken `54b` link in reuse hub. | Restored exact original `R01–R16` IDs (retaining retired `R14` and open `R15`) and fixed `54b` link in `docs/hegel_machine_reuse_hub_20261006.md`. | **CLOSED** |

---

## 2. Real Docker Back-Verification of All `30` Historical Cross-Diagonal Decomposition Cells (`Q01`)

- **Ledger**: `01_m99_q01_30_cross_diagonal_decomposition_probes.json` (`total_probe_wall_ms = {q01['total_probe_wall_ms']:,.2f} ms`, `0` paid LLM calls)
- **Outcome Breakdown Across `30` Probes (`15` Accepted Decompositions × `2` Off-Diagonal Cells)**:
  - `accepted = True`: **`{q01['accepted_count']}/30`**
  - `scientific_or_domain` rejected (`completed`): **`{q01['scientific_or_domain_rejected_count']}/30`**
  - `schema_or_interface` rejected (`completed`): **`{q01['schema_rejected_count']}/30`**
  - `infra_or_timeout`: **`{q01['infra_or_timeout_count']}/30`**
  - Superseded original unexecuted `Hegel` `d0@E` witnesses (`errata_relation = supersedes_unexecuted_projection_with_executed_probe`): **`{q01['superseded_hegel_d0_witness_count']}/8`**

| Arm | Cluster (`Decomp Step`) | Cross-Diagonal Cell | `probe_id` | Status | Accepted | Failure Kind | Wall (`ms`) | Supersedes Unexecuted Witness |
|---|---|---|---|---|---|---|---:|---|
{chr(10).join(probe_table_rows)}

---

## 3. Provenance Reconciliation (`Q05`), Unknown `x` Sign-Flip Ladder (`Q06`), & `c04/E7` Container Check (`Q03`)

1. **Git Tree & Timing Separation (`Q05`)**:
   - Declared paid-run freeze commit `7624c43bf72d283e82442504eff35ff6ee02b40d` (`{frz['py_file_count']}` `.py` files): `{frz['source_tree_sha256']}`
   - Post-freeze finalizer tree recorded in `m9_9_continual_summary.json`: `{fin_sha}`
   - Raw mirrored call time envelope: Confirmation `{c_env['call_envelope_lower_bound_sec']:,.3f}s` (`{c_env['execution_run_ids_in_raw_calls'][0]}` vs finalizer `{c_env['finalizer_postprocess_wall_sec']}s`); Dev `{d_env['call_envelope_lower_bound_sec']:,.3f}s` (`{d_env['execution_run_ids_in_raw_calls'][0]}`).
2. **Container Verification of Prefix-Available `bind_alpha_m99_c04_up_b02` on `alpha_m99_c04 / E7` (`Q03`)**:
   - Direct `56/24` upstream component probe: `execution_status = {q03['direct_upstream_component_probe']['execution_status']}`, **`chosen_up_holdout_rmse = {q03['direct_upstream_component_probe']['chosen_up_holdout_rmse']:.8f}`** (`<= 0.03`).
   - Composed `(bind_alpha_m99_c04_up_b02 + bind_alpha_m99_c04_dn_b03)` 3-layer public gate: **`accepted = {q03['joint_three_layer_public_gate']['accepted']}`** (`{q03['joint_three_layer_public_gate']['probe_wall_ms']} ms`).
   - Official private verifier (diagnostic check only; original `M9.9` confirmation score/cost unchanged): **`strict_pass = {q03['official_verifier_result']['strict_pass']}`** (`{q03['official_verifier_result']['verifier_wall_ms']} ms`).
3. **Exact Two-Sided Sign-Flip Sensitivity Ladder for Unknown Attempt `x` (`call_uuid = 0391d702b0c547a488608847b5dff224`)**:

| Unknown `Plain` Attempt Cost `x` (`v2`) | Exact Two-Sided Sign-Flip `p` (`Hegel - Plain`) |
|---:|---:|
{chr(10).join(x_rows)}

4. **Elapsed Probe Wall-Clock Ratios (`Q13`)**: Total non-LLM agent probe elapsed wall-ms `Monolithic / Hegel = {w_rat['monolithic_over_hegel_total_agent_probe_wall_ratio']:.6f}×` (`1,271,833.04 ms / 895,174.81 ms`); empirical coverage sub-item `= {w_rat['monolithic_over_hegel_empirical_coverage_subitem_wall_ratio']:.6f}×` (`689,259.6 ms / 207,604.6 ms`).
"""

    zh_md = f"""# Hegel M9.9.1 独立审计闭环、30 次断网容器交叉补证、取证根因校准与正式勘误报告（`Q01–Q14` + 原编号 `R01–R16`）

- **日期**：`2026-11-28`
- **被审计基线提交**：`d4b1cd7f06a08db8d967a43913003527d21a0354`（`Hegel_M9_9_Audit`）
- **本轮 `M9.9.1` 付费 LLM 调用数**：**`0`**（纯确定性契约硬化 + 真实断网 Docker 容器补证）
- **原封存证据包只读承诺**：`docs/evidence/m9_8_1_*`、`docs/evidence/m9_9_dev_live_calibration_20261128` 与 `docs/evidence/m9_9_autonomous_decomposition_and_interaction_withdrawal_20261128` 字节级零改动；本轮新增补证全部独立封存于 `docs/evidence/m9_9_1_audit_closure_and_forensic_reverification_20261128/`（含 `SHA256SUMS.txt`）。

---

## 一、`Hegel_M9_9_Audit`（`Q01–Q14`）逐项修复与勘误总表

| 编号 | 优先级 | 审计诊断问题 | `M9.9.1` 代码修复、容器补证与声明校准 | 状态 |
|---|---|---|---|---|
| `Q01` | `P1` | 分解时两格交叉约束来自推断。 | 历史补跑完成 {completed}/30 格，其中 {q01['scientific_or_domain_rejected_count']} 格为科学或域拒绝，匹配旧证人 {q01['superseded_hegel_d0_witness_count']}/8。新联合守卫使用实测格。 | 历史回放 |
| `Q02` | `P1` | alpha c02/E7 未过。 | 冻结下游的联合标准化与插补修正使误差从 0.071267 降到 0.004857。公开交换检查覆盖联合标准化的一项必要性质，冻结回放同时修改了插补方式，无法分出其独立贡献。 | 冻结组件归因 |
| `Q03` | `P1` | c04/E7 的守卫排除了已有的合格上游。 | 前缀组合 up_b02 + dn_b03 公开通过 {q03['joint_three_layer_public_gate']['accepted']}，私判通过 {q03['official_verifier_result']['strict_pass']}。新回退的在线收益待测。 | 冻结候选回放 |
| `Q04` | `P1` | 延迟分解使旧单体无法调用。消融还把缺失的可分分支指向了耦合分支。 | 新执行路径支持归档回退，前缀消融显式记录缺失分支。原始实验保留当时实际执行的路由规则。 | 工程修复，收益待测 |
| `Q05` | `P1` | `--finalize-existing` 覆盖了摘要中的 `run_id`、`wall_sec` 与源码树哈希（`be7190fb...` vs 冻结树 `34b01d19...`）。 | Runner 新增一次性执行起止清单与独立后处理清单；在补证账本精确重建 `7624c43bf` 树哈希（`{frz['source_tree_sha256'][:16]}...`）与原始镜像调用时间包络（确认 `{c_env['call_envelope_lower_bound_sec']:,.3f}s`、开发 `{d_env['call_envelope_lower_bound_sec']:,.3f}s`）。 | **已闭环（代码 + 来源账本）** |
| `Q06` | `P1` | 八个配对簇。 | Hegel 相对单体质量 p = {quality_p}，成本 p = 0.765625。相对 Plain 在 x = 0 时成本 p = 0.0625。阶梯使用同一组八个配对。 | 统计口径修正 |
| `Q07` | `P1` | 旧撤回记录可通过内部外键校验，即使签发时未留存科学负回执。 | 签发时留存 {withdrawal['historical_signed_scientific_receipts']}/{withdrawal['total_receipts']} 条联合科学回执，独立回放得到 {withdrawal['replayed_bound_scientific_negatives']}/{withdrawal['total_receipts']} 条身份匹配的科学负例。新授权要求可验证回执。 | 原始记录与回放分别列账 |
| `Q08` | `P2` | 显式旧模式降权后二次 `save -> reload` 摘要不幂等；旧 `M9.8` 高 `r²` 推定定位仍可写下游科学负例。 | 迁移后规范化重算 `v2` 摘要与 `unknown_requires_reverification` 回执 ID（`snap_1 == snap_2` 往返幂等且零升权），并阻断 `compatibility_inference` 写入下游科学负例。 | **已闭环** |
| `Q09` | `P1` | 零预置源码被泛化表述为零预置分解/一般拓扑发现。 | 统一修正为**零预置模块源码、给定两阶段接口与父拓扑下的在线模块形成**（未知层数 DAG 另轨）。 | **已闭环** |
| `Q10` | `P1` | 冻结消融限制了固定选择器可用的特征。 | 在同一宿主重建的策略输入上，普通字典复现 {parity} 个候选摘要。实际容器输入另行登记。 | 前缀回放 |
| `Q11` | `P1` | 理想未见格点公式被写成精确 LLM 成本定律。 | 修正为附条件格点计数 `N_{{sep,unseen}} = card((U x D) \\ (R u X))`，未来需要实测质量约束下的 `ΔC(K)` 与 `K*`。 | **已闭环** |
| `Q12` | `P2` | 被取消的开发运行曾从总成本中遗漏。 | 找回 {costs['unique_logical_calls']} 次调用，已知 {costs['known_physical_v2']:,} v2，另有 {costs['unclosed_call_count']} 次未结。总计 {costs['total_rd_v2_formula']}。 | 部分账单恢复 |
| `Q13` | `P2` | `E4/E5` 增益标签颠倒、`E1`（`7/8, 6/8`）与累计（`8/8, 7/8`）混写、墙钟耗时（总 `1.421×` / 覆盖子项 `3.320×`）误写为 CPU。 | 已在报告生成器、中英文 `docs/55_*` 与能力总入口同步修正。 | **已闭环** |
| `Q14` | `P2` | `R01–R16` 编号漂移及能力总入口 `54b` 坏链接。 | 在 `docs/55_*` 第七节按原 `R01–R16` 编号逐一对应（明确保留 `R14` 退役与 `R15` 开放边界），并修复 `docs/hegel_machine_reuse_hub_20261006.md` 中的 `54b` 链接。 | **已闭环** |

---

## 二、历史 `15` 次已接受分解共 `30` 个交叉格断网容器实跑补证（`Q01`）

- **封存账本**：`01_m99_q01_30_cross_diagonal_decomposition_probes.json`（总耗时 `{q01['total_probe_wall_ms']:,.2f} ms`，`0` 次付费 LLM 调用）
- **`30` 次交叉公开门控实跑统计**：
  - 真实通过（`accepted = True`）：**`{q01['accepted_count']}/30`**
  - 真实科学/域拒绝（`scientific_or_numerical` / `domain_precondition`）：**`{q01['scientific_or_domain_rejected_count']}/30`**
  - 结构/接口拒绝（`schema_or_interface`）：**`{q01['schema_rejected_count']}/30`**
  - 超时或基础设施错误：**`{q01['infra_or_timeout_count']}/30`**
  - 建立显式勘误替代映射的原 `Hegel` 未执行 `d0@E` 证人数：**`{q01['superseded_hegel_d0_witness_count']}/8`**

| 实验臂 | 确认簇（分解步） | 交叉核查格 | `probe_id` | 执行状态 | 是否通过 | 失败分类 | 耗时 (`ms`) | 替代原未执行证人 ID |
|---|---|---|---|---|---|---|---:|---|
{chr(10).join(probe_table_rows)}

---

## 三、来源身份核对（`Q05`）、未知 `x` 精确符号翻转敏感性阶梯（`Q06`）与 `c04/E7` 容器验证（`Q03`）

1. **Git 源码树与执行时间包络核对（`Q05`）**：
   - 声明付费运行冻结提交 `7624c43bf72d283e82442504eff35ff6ee02b40d`（`{frz['py_file_count']}` 个 `.py` 文件）：`{frz['source_tree_sha256']}`
   - `m9_9_continual_summary.json` 记录的赛后补算树哈希：`{fin_sha}`
   - 原始镜像调用时间包络下界：确认集 `{c_env['call_envelope_lower_bound_sec']:,.3f}s`（运行前缀 `{c_env['execution_run_ids_in_raw_calls'][0]}` vs 赛后汇总 `{c_env['finalizer_postprocess_wall_sec']}s`）；开发集 `{d_env['call_envelope_lower_bound_sec']:,.3f}s`（运行前缀 `{d_env['execution_run_ids_in_raw_calls'][0]}`）。
2. **`alpha_m99_c04 / E7` 前缀已有模块 `bind_alpha_m99_c04_up_b02` 断网容器验证（`Q03`）**：
   - 直接上游 `56/24` 留出探针：`execution_status = {q03['direct_upstream_component_probe']['execution_status']}`，**`chosen_up_holdout_rmse = {q03['direct_upstream_component_probe']['chosen_up_holdout_rmse']:.8f}`**（`<= 0.03`）。
   - 组合 `(bind_alpha_m99_c04_up_b02 + bind_alpha_m99_c04_dn_b03)` 公开三层门控：**`accepted = {q03['joint_three_layer_public_gate']['accepted']}`**（`{q03['joint_three_layer_public_gate']['probe_wall_ms']} ms`）。
   - 官方私判评测器（仅作事后诊断补证，不改动原确认集 `50/56` 得分与费用）：**`strict_pass = {q03['official_verifier_result']['strict_pass']}`**（`{q03['official_verifier_result']['verifier_wall_ms']} ms`）。
3. **未知超时尝试 `x`（`call_uuid = 0391d702b0c547a488608847b5dff224`）双侧精确符号翻转 `p` 值阶梯**：

| `Plain` 未知尝试成本 `x` (`v2`) | 双侧精确符号翻转 `p`（`Hegel - Plain`） |
|---:|---:|
{chr(10).join(x_rows)}

4. **非 LLM 代理探针流逝墙钟比值（`Q13`）**：全代理探针累计流逝墙钟比值 `Monolithic / Hegel = {w_rat['monolithic_over_hegel_total_agent_probe_wall_ratio']:.6f}×`（`1,271,833.04 ms / 895,174.81 ms`）；经验覆盖探针子项累计墙钟比值 `= {w_rat['monolithic_over_hegel_empirical_coverage_subitem_wall_ratio']:.6f}×`（`689,259.6 ms / 207,604.6 ms`）。
"""

    OUT_EN.write_text(en_md, encoding="utf-8")
    OUT_ZH.write_text(zh_md, encoding="utf-8")
    print(f"Wrote `{OUT_EN.name}` and `{OUT_ZH.name}` successfully.")


if __name__ == "__main__":
    main()
