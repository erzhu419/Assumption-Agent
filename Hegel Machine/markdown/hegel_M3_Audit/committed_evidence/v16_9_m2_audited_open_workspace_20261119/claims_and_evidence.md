# V16.9 M2.5 Audited Claims & Empirical Evidence (`claims_and_evidence.md`)

## 1. 聚焦红队反例闭环（`focused_audit_recheck.json`）
- **AST 纯数据流规范化与旁注一致性校验**：3 组碰撞探针 `digests_equal` 全部为 `False`，相反符号读出透镜（`+1` vs `-1`）仍稳定收敛至同一 `VERIFIED_PURE_AST` 摘要。
- **确认前缀冻结、诚实 `scoped_unknown` 与收据身份绑定**：`memorizer` 确认 `RMSE = 9.0`；`5 + sin(u)` 返回 `scoped_unknown`（`0` 假确认）；旧目标/旧版本/缺误差/`NaN` 收据全部拒绝。
- **坐标排列不变性与 `program_steps` 真实执行**：`T_M2A1` 逆序特征与原顺序成本及通过率完全一致；删除校准/确认步骤真实改变执行路径与授权结果；`deepcopy(lab)` 拥有独立克隆 RNG。

## 2. 接入 `CodeAgent` 的真实开放作者反例修订链（`open_author_revision_results.json`）
- **链 O1（Haldane 底物抑制 + 旁路分流，带 `scipy.optimize.least_squares` 有状态 `fit`）**：未见逆序反号目标 `T_O1` 上 `Hegel_OpenRevision_Forest` 4/4 通过，复合成本 `53.23` vs 移除新节点 `74.70` vs 同工具普通持续会话 `66.22`。
- **链 O2（欠阻尼振荡阶跃振铃 `1 - exp(-t/tau)*cos(w*t)`）**：未见逆序反号目标 `T_O2` 上 `Hegel_OpenRevision_Forest` 4/4 通过，复合成本 `53.23` vs 移除新节点 `77.45` vs 同工具普通持续会话 `71.22`。
