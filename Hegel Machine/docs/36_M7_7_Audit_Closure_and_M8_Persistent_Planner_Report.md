# Hegel M7.7–M8.1 全链路闭环报告：持久机制图规划、50 局离线回放校准与 16 线程真题切片验证

## 一、核心结论

1. **M7.7 审计控制项与 M8 13 项数学反例全部闭环**（单元测试全绿）。
2. **M8.0 单持久图 `GPT CLI`（`gpt-6.2-sol`）连续发现与迁移微实验**（`docs/evidence/m8_0_variational_planner_gpt_cli_litmus_20261125/`，`46.2s`）：验证 `AncestorReweight`（未见相关任务零样本 RMSE `3.344 -> 0.102`，无关哨兵漂移 `0.0`）、`LocalSplit`、匿名特征实时 `ConceptLift`（少样本未见姐妹任务零额外 LLM 调用 RMSE `0.858 -> 0.140`）、`ChangeMemory` 与异方差噪声 `NoChange`。
3. **M8.1 Part A（50 局历史容器轨迹零泄漏离线校准）**（`docs/evidence/m8_1_offline_replay_and_live_3task_slice_20261125/part_a_offline_calibration/`）：仅用 `Frontier-20` 校准（`0` Holdout 泄漏），`Frontier-20` LOTO 通过率从 `12/20` 提升至 **`15/20`**，盲测 `Holdout-12`（`gpt-6.2-sol`）提升至 **`12/12`（100%）**，盲测跨模型 `gpt-6-luna` 提升至 **`11/12`（91.7%）**。
4. **M8.1 Part B（16 线程并发 3 道真实科学容器题在线验证，墙钟总耗时 `441.2s = 7.35 分钟`）**（`docs/evidence/m8_1_offline_replay_and_live_3task_slice_20261125/part_b_live_3task_16thread_slice/`）：
   - `ho04`：识别 `VERIFIED` -> `CommitPrimary1x`（`10` 轮，`PASS`）；
   - `ho06`：识别 `VERIFIED` -> `CommitPrimary1x`（`13` 轮，`PASS`）；
   - `ho12`：主分支在线失败（`antithesis_1x_pass=False`），M8 实时捕获 `UNKNOWN` 未探针状态并触发 `SpawnDialecticalChallenger` 唤醒 `Thesis`（`thesis_1x_pass=True`）成功逆转（`14` 轮，`PASS`）；
   - **总成绩 `3/3`（100% 通过），轮次倍率仅 `1.1935x`**。
