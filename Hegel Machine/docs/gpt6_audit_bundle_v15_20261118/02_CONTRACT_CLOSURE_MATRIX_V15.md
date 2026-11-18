# GPT-6 `v15` 施工与实验合同逐项闭环矩阵

| 合同章节 | 要求摘要 | 实现与核验位置 | 状态 |
| :--- | :--- | :--- | :---: |
| **§1 边界约束** | 零新增 `src/` 模块、零新物理家族、零在线 API 调用、历史 35 份回执 0 字节改动 | 仅修改 4 个既有 `src/` 文件；`git diff docs/evidence/v10..v14` 为空 | ✅ |
| **§2.1 一次性勘误** | 10 项 V14 勘误（232/106、207/25/0、锚点算力/墙钟、D2 测量、结构化采用、失败链叙述等） | `deliverable_1_v14_errata_and_factual_reexport_20261118.json` | ✅ |
| **§2.2 修正事件导出** | 读真实 `autonomous_branch_reconciliation` 与 `scope_not_established`，失败链输出 `unresolved`/`missing_evidence`，表格由 JSON 自动生成 | `run_v14_fair_comparison_and_continuous_object.py` + `05_STANDALONE_VERIFY_V15.py` 内容级检查 | ✅ |
| **§3.1–3.3 真实单版本移除与重建** | 按 `(concept_id, semantic_version)` 清理 list 型 `concepts`/`abstracts`/绑定/缓存，保留双源共享规则与原始经历，允许计费重建 + 3 项控制 | `memory.py::remove_library_object_version` + `branch_reconciliation.py::reconstruct_missing_objects_from_raw_history` | ✅ |
| **§4.1–4.4 统一状态转移与缓存** | `scope_unidentified` 先查模板并标 `instance_scope_unknown`；`K`/`R` 共用 `apply_reconciliation_step_from_evidence`；预算不足 `partial_pending`；证据哈希纳入 `radius_value`/`commitment` | `branch_reconciliation.py` + `premise_commitment_index.py`；8 簇 24 对与前瞻 E1 36 对均为 `0` 分歧 | ✅ |
| **§5.1–5.2 结构化采用链与回放** | 5 阶段 `consumption_events`（含参数/区间/边界变换链）+ 6 个已曝光开发回放 | `unified_research.py` + `deliverable_2_removal_replay_parity_and_exposed_diagnostics_20261118.json` | ✅ |
| **§7.1–7.2 成本剖分与对齐** | 7 桶算力剖分、`H*` 关闭时跳过无消费者源期反统一、成对交叉检查版本缓存、`Ordinary` 与 `K` 局部触发与 API 对齐 | `branch_reconciliation.py` + `unified_research.py` + `run_v15_prospective_e1_e2_studies.py` | ✅ |
| **§8–§10 E1/E2 分开前瞻与停止规则** | 退役 `103/104`，在 `1051000–1056000`（`k=1,3,6` 顺序平衡）分开运行 `E1`（同历史）与 `E2`（零拼接端到端），如实报告追平、取舍与负结果 | `deliverable_3_prospective_e1_...json` + `deliverable_4_prospective_e2_...json` | ✅ |
