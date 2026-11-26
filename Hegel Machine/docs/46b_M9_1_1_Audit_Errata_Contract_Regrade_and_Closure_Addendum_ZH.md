# M9.1.1 审计勘误、`hd01` 契约敏感性重评与确定性工程闭环增补报告

- **日期**：2026-11-26
- **分支**：`exp/m6-causal-wormhole-sandbox`
- **密封证据包**：`docs/evidence/m9_1_1_audit_closure_and_regrade_20261126/`（`SHA256SUMS.txt` 已校验，`0` 次付费 LLM 调用）
- **保留不可变上游证据包**：`docs/evidence/m9_1_zero_tax_demand_retrieval_20261126/`（`d3ffa0018`，原封未改）

---

## 1. 术语与 Turn-1 Token 记账勘误（`docs/46_*`）

1. **零 Turn-1 技能清单（`[]`）与完整首轮请求 Token（`+127` 输入 Token）严格区分**：
   - 在 `M9.1` 中，`demand_triggered_zero_tax_library` 的首轮 `staged_wormhole_manifest` 为空列表（`[]`，`2` 个 JSON 字符，`init_payload` 与 `bare_no_library` 字节级一致），彻底消除了 `M8.9`/`M9.0a` 按模块递增的首轮技能卡片税（`450–672` 字符）。
   - 但该臂在 `system_prompt` 中包含一段固定 `+425` 字符的休眠库说明（`5. DORMANT ON-DEMAND PRIMITIVE LIBRARY`），因此在全部 `8/8` 对（`Demand` vs `Bare`）上产生了精确恒定的 **`+127` Turn-1 输入 Token 增量**（见 `m9_1_turn1_full_request_accounting_errata.json`）。
   - 自 `M9.1.1` 起，代码与报告统一将 `turn1_staged_manifest_json_chars` 与 `turn1_request_accounting`（完整请求字符数与首轮真实 `input_tokens`）分列记账。
2. **订阅制 CLI Token 代理指标重命名**：
   - 实验运行于官方 OpenAI ChatGPT Subscription Codex CLI（`auth_mode == "chatgpt"`），原 `effective_billed_tokens` 属于分析代理权重而非真实 API 账单扣费。自 `M9.1.1` 起统一报告原始 `(input_tokens, cached_input_tokens, output_tokens)` 三元组，并将加权指标更名为 `token_cost_proxy(cache_weight=0.25, output_weight=1.0)`。

---

## 2. `hd01` 事后契约表示敏感性重评（`18/18` 实例）

- **口径声明**：本重评（`hd01_contract_sensitivity_regrade_18of18.json`）仅对 `M9.1` 已冻结的 6 份不可变 `hd01` 提交产物做**契约表示敏感性检验**，**并非**新的盲测模型评估；原始密封裁决（`6/8, 7/8, 7/8`）完整保留。
- **根因确认**：`hd01` v1 文本未通过公共 schema 校验器强制锁定 `fixed_zero` / `fixed_active` 的 `bool[p]` 布尔掩码表示，6 份提交中有 4 份输出了数学等价的不相交整数下标列表（`list[int]`），在原始验证器 `len(fz) != p` 处被判为 `bad_masks`。
- **归一化重评结果（`normalize_and_validate_box_l0_public_schema(..., allow_index_list_compat=True)`）**：
  - **`18 / 18` 个候选×私有种子实例全部通过**三阶段全部数值与全局最优性证书检验（`aligned_mae = 0.0569..0.0752 <= 0.12`，`potency_rel_l2 = 0.0268..0.0574 <= 0.08`，对偶间隙 `3.71e-8..4.05e-8 <= 1e-6`），三臂归一化通过率均为 **`8 / 8`**：

| 实验臂 | `M9.1` 原始密封通过数 | 契约归一化通过数（`hd01` `bool[p]`） |
| --- | :---: | :---: |
| `bare_no_library` | `6 / 8` | **`8 / 8`** |
| `progressive_upfront_library` | `7 / 8` | **`8 / 8`** |
| `demand_triggered_zero_tax_library` | `7 / 8` | **`8 / 8`** |

---

## 3. 确定性工程与协议闭环（`C1..C6`）

1. **`C1`（`normalize_and_validate_box_l0_public_schema`，`hegel_box_l0_cert_schema_v2`）**：建立零私有真值依赖的公共 `bool[p]` 证书 schema 与二叉超立方体划分完备性校验器，严格拒绝重复下标、越界坐标、`fixed_zero`/`fixed_active` 交叠与非有限数值。
2. **`C2`（`L1` 四元组资格绑定与顺序无关多版本池）**：将 `L1` 资格升级为 `(verified_code_sha256, synthetic_unit_test_sha256, verifier_protocol_version, dependency_env_digest)` 四元组（`hegel_l1_qual_4tuple_v2`）；修改 `synthetic_unit_test_code` 立即使旧收据失效并触发洁净容器重验；移除 `reverify_in_clean_container=False` 下对缺失 `verified_code_sha256` 的静默补章；在洁净容器中重验全部 `6/6` 个 `L1` 原语并生成 `pretrained_autogenous_forest_v6_m911.json`（`sha256 = 7868bad10fbfaabe512fee2df2d75fc378a7407eab1c7e8d831932dfe02020c3`）；修复同名多版本 `L1` 候选池的顺序依赖并记录 `superseded_l1_versions`。
3. **`C3`（同义词/函数级检索、子问题作用域前置条件、事件干净触发与修复目录透传）**：增强 `search_skills` 与 `describe_ref.py`（`"robust regression"` 命中 `cauchy_linear_fit`/`robust_residual_weights`，`"certified lower bound"` 精准返回 `box_ridge_cube_lower_bound`，`"complex Helmholtz preconditioner"` 显式报告前置条件拒绝）；修复软换行拆句并实现按子问题作用域与否定感知的前置条件门控（`vr04` 保持 `0` 误推荐，同时 `complex128 + real float64 IRLS` 混合多阶段任务正确保留实数阶段原语）；收紧按需提示状态机并将 `mathematical_preconditions` 完整透传至 `LocalContainerRepair`。
4. **`C4`（定量反证见证、处理臂不变路由器 `v5`、跨 Python 版本 AST 哈希与多尝试累计记账）**：拒绝 `callable`/`hasattr` 反射伪造与纯关键词 `nogood` 消解，保留真实定量多规范/锚点见证；将 `classify_public_container_verification_state` 与 `build_pre_action_feature_vector` 默认设为处理臂不变（`treatment_invariant_mode=True`），并透明并列报告 `m7_4` 20 折 LOTO（`v5_treatment_invariant`：`13/20` vs `v4_legacy`：`14/20`）；修复 Python 3.10–3.12+ `type_params` AST 哈希漂移；在 `FrontierCodexClient.call_structured` 中跨全部重试与超时尝试累计真实 token/耗时并落盘 `attempts_<seq>.json`。
5. **`C6`（保留分析重绑定/局部遮蔽剔除与时序 `ObservationContext` 生产链路闭环）**：`inspect_final_deliverable_skill_retention` 剔除顶层重绑定/删除与函数体内局部遮蔽符号；`M8VariationalPlanner` 的全部候选 `predict_fn`、`AcquireObservation` 与 `save_checkpoint`/`load_checkpoint` 完整支持含 `past_driver_history` 的时序 `ObservationContext`（`test_m9_1_1_audit_closure_suite.py` `5/5` 全通过）。
