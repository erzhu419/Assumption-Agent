# 21 — Hegel V18.0（`M5` 终章收官实测报告）：泛化跨时刻多尺度时间折叠 × 部分共同基底子树先验与少样本不确定性治理

- **对应总体设计合同**：[20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md](./20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md)
- **新增与增量升级核心模块（零推翻既往 `M0`–`M5.0` 代码，全量 `51/51` 测试 `100%` 通过）**：
  - 增量扩展：[src/hegel_machine/cognition/wormhole_forest.py](../../src/hegel_machine/cognition/wormhole_forest.py)、[src/hegel_machine/cognition/m4_4_instance_role_closure_and_online_writeback.py](../../src/hegel_machine/cognition/m4_4_instance_role_closure_and_online_writeback.py)
  - `M5` 终章主引擎：[src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py](../../src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py)
  - 独立复现实验入口：[scripts/run_v18_0_m5_final_temporal_and_subtree_study.py](../../scripts/run_v18_0_m5_final_temporal_and_subtree_study.py)
  - 自动化验收单测：[tests/test_v18_0_m5_temporal_multiscale_and_base_subtree.py](../../tests/test_v18_0_m5_temporal_multiscale_and_base_subtree.py)
  - 静态证据归档包（`6/6` `SHA256SUMS.txt` 校验通过）：[docs/evidence/v18_0_m5_final_temporal_multiscale_and_base_subtree_20261120/](../../docs/evidence/v18_0_m5_final_temporal_multiscale_and_base_subtree_20261120/study_manifest.json)

---

## 一、核心工程兑现概览（零平行重写、100% 数据驱动真求解）

1. **泛化因果时态相位 $(\mathcal{P}_\tau, \mathcal{N}_\tau, \mathcal{F}_{\tau, H})$ 与三形态 `Π_fold` 时间折叠**：
   - 在 `ObservationMap` 与 `ScaleMap` 中向后兼容地扩展时态与时间尺度字段（旧 `M0`–`M5.0` 节点语义摘要保持字节级不变）；
   - 实现 `extract_causal_temporal_regimes()`：支持不规则采样步长 $\Delta t_k$，显式区分 $(\Delta t, \tau_i, w, H)$，在线状态提取强制锁死前缀因果截止 $\mathcal{H}_{\le t_0}$（拦截居中窗口或未来峰值泄漏）；
   - 实现 `predict_via_pi_fold_mode()`：支持 `effective_constant`（稳态常数）、`reduced_slow_ode`（绝热快变量消元）与 `memory_kernel`（Mori–Zwanzig 指数历史卷积核 + 初态余波，严格保住“同现在、异过去 $\implies$ 异将来”）。
2. **一致联合绑定 `SharedBaseSubtreePrior` 与校准版少样本不确定性治理**：
   - 直接复用 `branch_reconciliation.anti_unify_programs` 从源任务程序 AST 抽取带参数槽的共同骨架，结合 `validate_consistent_joint_subtree_binding()` 验证跨共享节点一致的角色、读出与时间尺度绑定（原生支持因果链与 `requires_premise_and` 并列条件组合，并对重复证据 ID 去重）；
   - 实现集合级、与入库顺序严格无关且按 `provenance_lineage_id` 去同源换皮复制的谱多样性 $D_{\text{eff}}^{\text{set}}$；
   - 实现无量纲软先验热启动 + 宽域兜底拟合器 `fit_with_soft_warm_start_and_wide_fallback()`，同时接入仓库现成 [`probing.py:value_forecast_experiments`](../../src/hegel_machine/cognition/probing.py) 任务决策风险价值器、`question_semantics.evaluate_decision_region` 停止准则及 `authorize_abstraction_use` 灰区边界守卫。

---

## 二、Part 1：12 项数学与红队反例控制门（`12/12` 全过）

| 控制编号 | 检验目标 | 实测结果 |
|---|---|---|
| `C01` | 居中窗口与 `> t0` 样本在线泄漏拦截 + 不规则采样步长识别 | **通过**（`future_information_leakage_past_t0` 精准拦截） |
| `C02` | 按 $t/\tau_i$ 归一化的快动态（`10,000x` 时间跨度）准入基底 vs 单次偶然脉冲拒绝 | **通过** |
| `C03` | $\dot{x}=-2x+z,\ \dot{z}=x-2z$ 同现在 $x(t_0)=0$、异初态 $z=\pm 1$ 下 `memory_kernel` vs 静态折叠 | **通过**（静态折叠坍缩为 `0`，记忆核误差 `< 0.008`） |
| `C04` | 短窗不可辨识的慢反馈反转（$T=40$）：短时距目标 `HEC` 立停 vs 长时距目标自动购慢探针 | **通过** |
| `C05` | `s(A,B)=0.9, s(A,C)=0.1, s(B,C)=0.2` 全部 `6` 种入库排列顺序不变性 + 三种换皮克隆去重 | **通过**（6 种排列差值 `0.0`；换皮克隆 `D_eff^set = 1.0`） |
| `C06` | 相容目标软热启动少步收敛 + 域间强异质目标（$\rho=0.12$ 远在源区间外）自动逃逸至宽域兜底 | **通过**（硬截断失败，软热启动+宽兜底 `RMSE < 1e-5`） |
| `C07` | 正交大幅干扰项（`±100`）vs 目标相关小分歧项（`±0.1`）：`probing.py` 风险下降器击败裸方差 | **通过** |
| `C08` | 断连边拼接（`a->b` + `c->d`）拦截 + `C1 AND C2` 重复编码证据不虚增先验票数 | **通过** |
| `C09` | 末端失配源于上游守恒外漏时：精准修订最小受影响子图，同时保持源域原守恒节点版本与状态完好 | **通过** |
| `C10` | 复现 `S02` 灰区反例（`b_sup=1.45, b_ref=2.25`，`b=1.80` 遗漏项 `0.156 > 0.05`）：拦截 `1.80` 并折半收紧至 `1.65` | **通过** |
| `C11` | 调用原生 `anti_unify_programs` 自动提炼含 `event_delta(lag=1)` 的跨源共同基底骨架 | **通过** |
| `C12` | 候选内部完全一致（方差为 `0`）但共同漏掉真机制时：强制目标留出确认拦截并干净退出 | **通过**（假确认 `0`） |

---

## 三、Part 2（轨道一）：纯匿名多随机种子硬核基准（4 种子 × 20 题 = 每臂 80 题，100% 真 `least_squares` / RK4 求解，零训练-确认-留出重叠）

### 1. 主对比矩阵（严格分离“端到端同源历史强基线”与“同证书平权纯组织消融”）
- **粗通道授权完全平权核验**：在绝热粗粒化任务集 `B_adiabatic_coarse_valid` 上，`Hegel_M5_Forest_Subtree_PPR` 与 `EqualAuth_Dynamic_Flat_BM25` 的实测物理测量费均为 **`64.0 M`**（`zero_coarse_auth_confound = True`），彻底消除了 `M5.0` 中的粗通道授权混杂。

| 对照分组 | 实验臂 | 域内严格通过 (`72`) | 域外诚实 `unknown` (`8`) | 实测假确认数 | 评估候选总数 | `scipy` `nfev` 总步数 | 目标测量费 `M` | 目标总复合成本 `C` | 含源形成全周期成本 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **组 I：端到端对决** | **`Hegel_M5_Full_TemporalSubtree_Forest`** | **`72/72` (`100%`)** | **`8/8`** | **`0`** | **`152`** | **`2,952`** | **`1,096.0`** | **`1,164.18`** | **`1,192.18`** |
| 组 I：同权限强动态对手 | `Strong_Dynamic_StateSpace_AllHistory` | `72/72` (`100%`) | `8/8` | `0` | `408` | `9,056` | `1,920.0` | `2,076.62` | `2,076.62` |
| 组 I：专项能力消融 | `Ab_Memoryless_Static_Baseline`（无历史静态） | `20/72` (`27.8%`) | `8/8` | `0` | `140` | `578` | `960.0` | `977.16` | `977.16` |
| 组 I：方差治理消融 | `Ab_Blind_Parameter_Copy_Transfer`（盲搬源参数） | `18/72` (`25.0%`) | `8/8` | **`54`** | `72` | `0` | `600.0` | `614.45` | `642.45` |
| **组 II：同证书纯组织消融** | **`Hegel_M5_Forest_Subtree_PPR`** | **`72/72` (`100%`)** | **`8/8`** | **`0`** | **`152`** | **`2,947`** | **`1,096.0`** | **`1,164.12`** | — |
| 组 II：仅单点虫洞（无基底子树） | `EqualAuth_SingleNode_Wormhole_Only` | `72/72` (`100%`) | `8/8` | `0` | `156` | `3,747` | `1,124.0` | `1,197.64` | — |
| 组 II：平权扁平 BM25 | `EqualAuth_Dynamic_Flat_BM25` | `72/72` (`100%`) | `8/8` | `0` | `192` | `4,087` | `1,124.0` | `1,203.27` | — |
| 组 II：裸方差探针消融 | `EqualAuth_RawVariance_Probe_Ablation` | `72/72` (`100%`) | `8/8` | `0` | `152` | `2,940` | `1,180.0` | `1,243.73` | — |

### 2. 旗舰联合持续流 `J`（`J01`–`J16` 在线时空折叠 × 共同基底演化 + 独立子进程冷恢复续跑 `J17`–`J20`）
- **主流 16 步（`14` 域内 + `2` OOS，全部非消融臂 `14/14` 严格通过、`0` 假确认）**：
  - **`Hegel_M5_Full_TemporalSubtree_Forest`**：**`243.41 C`**（`228.5 M`，`32` 候选，`669` `nfev`）
  - `EqualAuth_SingleNode_Wormhole_Only`：`256.13 C`（`238.5 M`，`34` 候选，`977` `nfev`）
  - `EqualAuth_Dynamic_Flat_BM25`：`257.18 C`（`238.5 M`，`42` 候选，`1,026` `nfev`）
  - `Strong_Dynamic_StateSpace_AllHistory`：`418.40 C`（`384.0 M`，`90` 候选，`2,022` `nfev`）
- **跨进程冷恢复续跑（`J17`–`J20`）**：独立 Python 子进程从可重定位快照恢复后，全部节点、粗边证书与 `shared_base_subtrees`（含 `J09` 在线修订写回的外漏分支 `version=2`）摘要零不一致，`4/4` 续跑任务全部严格通过（`J17` 两层基底热启动仅需 `9` 步 `nfev`，`J18` 记忆核 `heldout_rmse = 0.0034`，`J19` 外漏修订复用 `heldout_rmse = 0.0`，`J20` OOS 诚实 `scoped_unknown`），假确认为 `0`。

---

## 四、Part 3（轨道二）：同引擎直观现实语义孪生展示（零背景可读）

1. **孪生案例 1（跨时刻三层树：毫秒级楞次电磁感应 $\leftrightarrow$ 分钟级勒夏特列化学缓冲 $\leftrightarrow$ 铁磁/黏弹迟滞）**：
   跨 `2,000` 倍物理时间尺度（`5ms` vs `10s`），经各自特征时间 $t/\tau_i$ 归一化后，`过去 -> 现在` 变迁对比幅度精确对齐（`-0.33264` vs `+0.33149`，符号映射由 `ObservationMap` 隔离），第 3 层再按前史记忆卷积区分迟滞。
2. **孪生案例 2（跨尺度时间折叠与短利长反：第 1 个月补贴红利 `+1.07` vs 第 36 个月套利反噬 `-0.717`）**：
   短期窗口下单调增益支与慢反馈反转支同时存活；`probing.py` 任务风险价值器自动忽略大幅无关噪音指标（`short_buzz_metric ±40`），精准选购中期套利探针 `mid_term_gaming_indicator` 锁定长期反转。
3. **孪生案例 3（渐进共同基底子树：长沙型 $\leftrightarrow$ 南昌型内陆省会，`C1 湿冷气候 ∧ C2 毗邻珠三角经济极`）**：
   一致联合角色绑定激活两层并列基底子树（`3` 条关系、`D_eff^set = 3.0`），给出珠三角虹吸比软热启动区间 `[0.658, 0.748]`，在南昌型目标上仅用 **`8` 次 `nfev`** 即收敛，并由价值器避开方言差异等无关高方差项，一针 `Q_local_anchor_industry_retention` 分叉出长沙文娱重工支与南昌航空制造配套支。
