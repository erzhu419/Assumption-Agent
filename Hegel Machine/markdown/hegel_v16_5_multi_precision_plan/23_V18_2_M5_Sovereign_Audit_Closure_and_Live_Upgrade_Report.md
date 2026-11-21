# Hegel V18.2 (`M5` 自主审计收官闭环 × 去混杂十臂基准 × 官方 ChatGPT 订阅对抗题实测报告)

## 1. 主权裁决总纲：吸收全部提分诊断，零妥协捍卫计划灵魂创新

针对 `markdown/Hegel_M5_Audit`（对 `63b940a03` 的静态与反例复核），`V18.2` 严格贯彻**“吸收一切利于提升真实性能的机械诊断，坚决驳回因保守洁癖而削平核心创新的倾向”**：

1. **100% 吸收并闭合 `F01`–`F11` 与第 VII/VIII 节全部机械反例**（验证函数 `run_sovereign_audit_counterexample_closure` 全量 `9/9` 反例族断言通过）：
   - **`F01` + `F10`（跨时刻因果提取与时间原点平移不变性入主干）**：`M3_ReducedSlowODE` 与 `M4_MoriZwanzigMemoryKernel` 在真实运行时直接消费 `extract_causal_temporal_regimes` 与 `predict_via_pi_fold_mode`（屏蔽任一辅助函数即触发 `AssertionError`），并以 `dt_hist = t_latest - t_init` 彻底消除 `t + 100.0` 原点平移误差（`pred_shift_0 == pred_shift_100 == 0.60653066`）。同时新增从已付 `pilot_rows` 在线推断经验特征的 `infer_empirical_pilot_signatures`（计入 `+0.5 K`）。
   - **`F02`（真实源端 `Program` AST 反统一与计量建库账本）**：`_build_seed_forest_and_priors` 从 3 组源端观测真跑 `least_squares` 拟合并计量 `source_formation_cost = 27.608 C`，通过 `anti_unify_programs` 自动抽出 3 槽位共同基底骨架 AST，并在 `M6` 中直接经由 `EvaluationContext` 求值执行。
   - **`F03`（真因果单探针获取与 `HEC stop` 零探针停止）**：严格仅购买 `select_m5_discriminating_probe` 选中的探针；当 `action == "hec_decision_region_stop"` 时购买 `0` 探针（`first_fit_train_rows == 5`，`M = 12.0`）；选中 `nuisance_high_var_aux` 时仅扣费而不注入无关真探针行。
   - **`F04`（零任务 ID 结果驱动扬弃与纯快照冷恢复消费）**：`J` 流完全依据确认胜出分支（`M6_SiphonSubtree_LinearLeakBranch`）自动触发 `revise_shared_base_subtree`；`SharedBaseSubtreePrior` 直接持久化 `source_parameter_estimates`、`warm_start_parameter_prior` 与 `executable_skeleton_dict`；在 `v18_0` 原快照与 `V18.2` 新快照上擦除 `J09` 泄漏分支均使 `J19` 成本从 **`12.34 C` 恶化至 `13.18 C / 13.31 C`**（候选数从 `1` 增至 `2`），证明跨进程冷加载真消费流内扬弃成果。
   - **`F06`（窄跨度 Pilot 确认触发同骨架宽框逃逸）**：当软热启动在窄跨度 `s_grad in [0.99, 1.01]` Pilot 上满足训练容差但在新鲜确认集上被证伪（如异质 `rho = 0.14` 落在源 `[0.69, 0.72]` 之外），系统立即在同一 `M6` 骨架上退回宽域框 `[0.05, 0.95]` 重拟合，将原 `scoped_unknown` 翻转为 **`strict_pass = True`（`heldout_rmse = 1e-6`）**。
   - **`F07` + `F09` + `F11`（治理摘要 v2、Jacobi PSD 投影与渐进分层绑定）**：空条件 `[]`、空因果边 `edge_bindings=[]`、缺失中间池角色 `x_pool` 全部严格失败关闭；同谱系克隆取最大相似度使重命名绝不虚增 $D_{\text{eff}}^{\text{set}}$（`orig_d_eff == clone_d_eff == 1.1111`），任意两两相似度经纯标准库 Jacobi 特征分解投影至半正定相关矩阵锥（`1.0 <= D_eff <= m`，彻底消除奇异点异常）；`subtree_digest` v2 完整覆盖多样性、源谱系、权限层级与历史链（篡改即检出 `subtree_digest_mismatch_count >= 1`）。
   - **`F08`（只读历史 `v17_7` 灰区阻断）**：`OutcomeDrivenForestConsolidator` 写入显式 `gray_zone_bounds`，且 `ForestScopeDispatcher` 对历史只读 `v17_7` 快照从 `host_confirmation_bundle` 自动推断已证上限，在不改动 `v17_7` 任何字节（`semantic_digest_mismatch_count == 0`）的前提下将灰区 `b = 1.8` 阻断为 `scope_unauthorized`，而受支持区 `b = 0.6` 正常 `dispatched_ok`。
2. **零妥协保留并升级的核心计划创新**：
   - **渐进分层共同基底子树（`C1 -> C1 ∧ C2`）**：拒绝将单条件匹配一刀切判死，而确立分级契约——仅满足 `C1` 返回 `activation_tier = "layer1_partial_candidate_only"`（保留候选结构启发但严禁数值热启动 `numeric_warm_start_authorized = False`）；同时满足 `C1 ∧ C2` 且 $D_{\text{eff}}^{\text{set}} \ge 1.35$ 才晋升 `layer2_full_conjunctive_warm_prior`。
   - **泛化跨时刻折叠（`Π_fold`）、多尺度物理通道定价与直观语义孪生（补齐 1 对 1 求解器回执投影）**：全部完整保留。

---

## 2. 去混杂 10 臂匿名基准与两跳虫洞压力测试（`800 + 32` 场确定性实验）

### 2.1 主基准（4 种子 × 20 题 = 80 题/臂，全部对称早停）

| 对照臂 | 域内严格通过 | OOS 诚实弃权 | 假确认 | 候选评估数 | `scipy nfev` | 目标测量费 `M` | 目标计算费 `K` | 目标复合费 `C` |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Hegel_M5_Full_TemporalSubtree_Forest`** | **`72/72` (`100%`)** | **`8/8`** | **`0`** | **`160`** | **`3,156`** | **`1104.0`** | **`1498.07`** | **`1178.90`** |
| `Strong_Dynamic_StateSpace_AllHistory`（自适应购点+早停） | `72/72` (`100%`) | `8/8` | `0` | `444` | `10,434` | `1568.0` | `3535.42` | `1744.77` |
| **`Strong_Dynamic_EqualUnitPrice_1_5M`（`1.5 M` 同单价诊断臂）** | `72/72` (`100%`) | `8/8` | `0` | `444` | `10,500` | `1176.0` | `3549.94` | `1353.50` |
| **`Hegel_M5_Forest_Subtree_PPR`** | **`72/72` (`100%`)** | **`8/8`** | **`0`** | **`160`** | **`3,151`** | **`1104.0`** | **`1496.97`** | **`1178.85`** |
| **`EqualContent_Flat_CompiledSubtrees_BM25`（同编译子树平铺臂）** | `72/72` (`100%`) | `8/8` | `0` | `176` | `3,428` | `1104.0` | `1556.63` | `1181.83` |
| `EqualAuth_SingleNode_Wormhole_Only`（对称早停） | `72/72` (`100%`) | `8/8` | `0` | `156` | `2,920` | `1124.0` | `1334.15` | `1190.71` |
| `EqualAuth_Dynamic_Flat_BM25`（对称早停） | `72/72` (`100%`) | `8/8` | `0` | `192` | `3,211` | `1124.0` | `1426.89` | `1195.34` |
| `EqualAuth_RawVariance_Probe_Ablation`（真因果诱饵执行） | `72/72` (`100%`) | `8/8` | `0` | `160` | `3,167` | `1202.0` | `1455.69` | `1274.78` |
| `Ab_Memoryless_Static_Baseline` | `20/72` (`27.8%`) | `8/8` | `0` | `144` | `578` | `960.0` | `343.16` | `977.16` |
| `Ab_Blind_Parameter_Copy_Transfer` | `18/72` (`25.0%`) | `8/8` | `54` | `72` | `132` | `600.0` | `289.04` | `614.45` |

- **即便剥离全部通道单价差（`Strong_Dynamic_EqualUnitPrice_1_5M`）**：`Hegel_M5` 依然在 `M`（`1104.0 < 1176.0`）、`K`（`-57.8%`）、`nfev`（`3,156 < 10,500`，省 `-69.9%`）与复合成本 `C`（`1178.90 < 1353.50`，净省 **`-174.60 C`**）上全面胜出。
- **两跳虫洞桥接 + 窄跨度异质参数压力集（`SS01` + `SS02`，4 种子 = 8 题/臂）**：当跨域迁移必须经由横向虫洞 `wh_N_C_thermal_siphon` 桥接时，`Hegel_M5_Forest_Subtree_PPR`（`8/8` 通过、`8` 候选、`112.0 M`、**`117.86 C`**）在测量费与候选搜索次数上正面击败了同样拥有编译子树的 `EqualContent_Flat_CompiledSubtrees_BM25`（`16` 候选、`116.0 M`、`121.99 C`）、`EqualAuth_Dynamic_Flat_BM25`（`20` 候选、`120.0 M`、`124.95 C`）与 `Strong_Dynamic_EqualUnitPrice_1_5M`（`44` 候选、`134.92 C`）。

---

## 3. 官方 ChatGPT 订阅对抗题实测（`L07` + `L08`，零 `ruoli.dev`）

严格通过 Windows 官方 `OpenAI.Codex` 订阅客户端（`auth_mode: chatgpt`，环境变量硬清洗 `RUOLI_*` / `GPT5_*` / `OPENAI_*`），在两道针对审计核心痛点设计的全新未见题上对 **`gpt-6-luna`** 与 **`gpt-6.1-terra`** 展开多轮实测：
- **`L07_NarrowPilot_HeteroRho_Escape`**：初始 5 行 Pilot 的 `s_grad` 缩窄在 `[0.99, 1.01]`，真实 `rho = 0.14` 远在源先验 `[0.69, 0.72]` 之外；
- **`L08_RawIrregularPrefix_MemoryFold`**：彻底移除预计算 `mem_conv`（`mem_conv = None`），仅给出起点平移至 `t in [100.0, 100.56]` 的非均匀采样原始轨迹前缀 `raw_prefix`。

| 订阅模型 | 对照臂 | 严格通过率 | 假确认 | API 调用 | 有效计费 Token | 测量费 `M` | 复合成本 `C` | 总耗时 | 关键逐题表现 |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| **`gpt-6-luna`** | **`Hegel_M5_LiveGPT`** | **`2/2` (`100%`)** | **`0`** | **`5`** | **`63,340`** | **`30.5`** | **`34.32`** | **`99.4s`** | **`L07` 检出窄 `s_grad` 共线性先购探针宽框通关；`L08` `heldout_rmse = 4e-6`！** |
| `gpt-6-luna` | `EqualAuth_Flat_LiveGPT` | `1/2` (`50%`) | `0` | `11` | `173,055` | `32.5` | `42.32` | `247.1s` | **`L08` 耗尽 6 轮（`17.3万` Token）未能正确积分平移非均匀前缀，失败弃权！** |
| `gpt-6-luna` | `Strong_Ordinary_LiveGPT` | `2/2` (`100%`) | `0` | `6` | `73,633` | `38.5` | `42.68` | `143.9s` | 测量费高出 `+26.2%`，总复合成本 `42.68 C` |
| **`gpt-6.1-terra`** | **`Hegel_M5_LiveGPT`** | **`2/2` (`100%`)** | **`0`** | **`4`** | **`26,558`** | **`30.5`** | **`32.41`** | **`73.5s`** | **达理论最少 4 次调用（每题 1 探针 + 1 确认），省 `-32.8%` 有效 Token！** |
| `gpt-6.1-terra` | `EqualAuth_Flat_LiveGPT` | `2/2` (`100%`) | `0` | `5` | `39,511` | `30.5` | `33.12` | `78.8s` | `L07` 多耗重修轮次，有效 Token 高出 `+48.8%` |
| `gpt-6.1-terra` | `Strong_Ordinary_LiveGPT` | `2/2` (`100%`) | `0` | `4` | `28,177` | `38.5` | `40.26` | `56.1s` | 复合成本 `40.26 C`（比轻量 `gpt-6-luna` + `Hegel_M5` 的 `34.32 C` 还贵！） |

> **跨模型核心定律再次复现**：在最苛刻的无预计算卷积 + 窄跨度异质逃逸对抗题上，**廉价轻量模型 `gpt-6-luna` 挂载 `Hegel_M5`（`2/2` 满分，`34.32 C`）不仅将同模型扁平检索（`1/2` 失败，`173,055` 有效 Token，`42.32 C`）的 Token 消耗砍掉 `-63.4%` 并救活 `L08`，而且在总复合成本上再次正面击败了更昂贵的 `gpt-6.1-terra` 裸跑强基线（`40.26 C`）！**

---

## 4. 证据归档与全量回归清单

- **新增模块与脚本**：
  - `src/hegel_machine/cognition/m5_v18_2_sovereign_closure.py`
  - `scripts/run_v18_2_m5_sovereign_closure_study.py`
  - `tests/test_v18_2_m5_sovereign_audit_closure.py`
- **完整证据目录**：`docs/evidence/v18_2_m5_sovereign_closure_and_live_subscription_20261121/`（含 `study_manifest.json`、`part1_sovereign_audit_counterexample_closure.json`、`part2_sovereign_10arm_summary_and_stress.json`、`part3_semantic_twins_with_pipeline_receipts.json`、`live_challenge_L07_L08_provenance_ledger.jsonl`、`live_subscription_challenge_L07_L08/` 及全文件 `SHA256SUMS.txt`）。
