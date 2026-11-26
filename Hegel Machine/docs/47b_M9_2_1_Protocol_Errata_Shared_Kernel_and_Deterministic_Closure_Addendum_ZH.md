# M9.2.1 协议勘误、共享证书核重评与确定性闭环补遗

- **日期**：2026-11-26
- **分支**：`exp/m6-causal-wormhole-sandbox`
- **M9.2.1 封存证据包**：`docs/evidence/m9_2_1_audit_closure_and_regrade_20261126/`（含 `SHA256SUMS.txt` 与原始 `junit_m9_2_1_and_regressions.xml`；`0` 次付费模型调用）
- **原封不动保留的上游证据包**：`docs/evidence/m9_2_equal_information_on_demand_reuse_20261126/`（`315` 个文件零改动）

---

## 一、`M9.2` 协议偏离与计量字段勘误（对应 `docs/47_*`）

1. **容器启动公共 Fixture 偏离勘误（`m9_2_protocol_deviation_and_funnel_errata.json`）**：
   - `M9.2`（`1883431e5`）生产启动路径未将 `Hegel-M92-EqualInfo-4` 接入 `public_setup_py`，而运行器末尾将 `shared_public_schema_validator_all_arms` 硬编码为 `True`。
   - 实际协议状态：`protocol_deviation_public_setup_missing = True`、`shared_public_schema_validator_all_arms_actual = False`。由于缺失公开 fixture 时 `K`/`E` 可参考库内实现而 `B` 需从零自建校验，原结果反映无预置公开样本条件下的政策表现，不代表完整挂载公共 fixture 协议下的纯增量。
2. **`K/E` 检索与详情接口统一前后的差异边界**：
   - `M9.2` 中 `K` 与 `E` 底层数学目录与源码完全一致（`4/4`），但两套 reader 在离线重放 24 条真实查询时有 `16/24` 排序不同、`9/24` 集合不同。`M9.2.1`（`A3`）抽出唯一 `_SHARED_CATALOG_READER_CORE_PY`，重放全部 24 条真实查询实现 **`0/24` 排序差异、`0/24` 集合差异**（`m9_2_unified_reader_24_query_replay.json`）。
3. **漏斗、首轮 Token、计时窗口与 `eq04` 拒用记录勘误**：
   - `K` 臂 `00_episode_summary.json` 中 `inspected_knowledge_ref_in_bash` 实际为 **`7 / 8`**（运行器误读不存在的 `inspected_knowledge_reference_txt` 而计 `0`）。
   - 单集 `turn1_prompt_tokens` 已统一匹配 `D_TURN_01` 与 `D2_TURN_01`。
   - 原 `student_elapsed_sec` 为 `monotonic` 私判窗口墙钟（`eq03`/`eq04` 含真值生成），`M9.2.1`（`A8`）已在候选子进程内拆分 `candidate_call_cpu_sec`、`candidate_call_wall_sec`、`ipc_and_codec_wall_sec` 与 `verifier_and_gt_wall_sec`。
   - `eq04` 上 `E` 两次查询 `solve_and_certify_l0`；`K` 在 `rep_01` 查询 `fit_ridge_nuisance` 与 `box_ridge_cube_lower_bound`、在 `rep_02` 仅执行 `--search`（全部 `0` 次 `L1` 调用与留存）。

---

## 二、`A1..A9` 确定性闭环与 `eq02` 固定产物重评

- **`A1`（`eq02_shared_kernel_fixed_artifact_regrade_12of12.json`）**：删除 `eq02` 复制的私判与简版公开 schema，统一接入中央 `m8_9_1_box_l0_certificate_kernel`（显式映射 `dual_nu_sign=-1.0`）并强制检验 Stage 2→3 公开交接等式。两类错误高下界 dual 反例与重复半立方体叶反例均被拒绝；原封存 6 份 `eq02` 终稿在两私有实例上 **`12 / 12` 全部通过**（交接残差 `<= 3.55e-15`）。
- **`A2`**：改为由 `task.holdout_public_setup_py` 能力驱动容器初始化，首次模型调用前强制核验 `required_public_assets` 存在性与 SHA-256 并写入 `public_setup_receipt`。
- **`A3`**：`K` 与 `E` 共享纯数据 `_shared_search_catalog` / `_shared_describe_catalog`（`24/24` 真实查询零分歧，裸名/限定名与 JSON schema 一致）。
- **`A4`**：四元组资格强制校验 `expected_protocol_versions` 与不可变容器镜像/依赖版本摘要 `hegel_env_v2:...` 相等，新增宿主独立双次重复性纯净检验，升级在线 `L1` 编译器同步签发完整四元组，并生成干净容器重验的 `pretrained_autogenous_forest_v7_m921.json`。
- **`A5`**：分离候选自述 `agent_claimed_unverified` 与宿主受控签发 `HostControlledProbeReceipt`（关闭 `assert 1 == 1; print('gauge scale residual=0.0')` 伪见证）。
- **`A6`**：在调用 oracle 前完成 `history_dt` 继承与逐行/每轨迹 `initial_memory_states` 字典校验，拒绝未迁移的新增时间列，并在 oracle 调用后异常回滚模型事务的同时永久保留付费观测与决策成本记录。
- **`A7`–`A9`**：每次 attempt 独立响应文件（关闭旧响应串用）、封存 `01b_provider_attempts/` 与 `usage_unknown_attempt_count`、修复 `eq03` 零空间真实同导数孪生与独立阈值分布，并将 `LocalContainerRepair` 统一接入四元组资格与多版本选择器。
