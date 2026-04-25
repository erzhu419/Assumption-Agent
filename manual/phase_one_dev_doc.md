# 阶段一：策略调度器——完整开发文档 (v1)

---

## 🏁 2026-04-23 v16 里程碑回看

**写于**: v16 架构验证完成后。

### 原意 → v16 实际覆盖

原计划：RL 训练（PPO/DPO）一个 learned dispatcher，输入 problem features，输出 strategy ID + confidence。

**v16 实际走的路：**

| 原设计组件 | v16 实际做法 | 状态 |
|---|---|---|
| RL-trained dispatcher | ❌ 未训练任何 RL | **放弃 RL 路径** |
| 问题特征提取器 | `signal_embeddings.npz`: sentence-transformer 嵌入 (multilingual MiniLM) | ✅ 有，但不是 learned feature |
| 策略选择 | 1) **硬编码 domain router** (math/sci → hygiene, else → case+reflect)<br>2) **LLM 语义选择** (v3 wisdom selections: 75 中选 3-5 条 per problem) | ✅ 两层 rule + LLM，非 RL |
| 置信度输出 | ❌ 没有显式 confidence | ❌ 放弃 |
| 策略-条件可视化 | ❌ 没做 | ❌ 放弃 |

### 为什么 RL 路径被放弃

1. **`7e353ed` commit (Phase 1 改造)** 的结论: Polya/Popper orientation restoration 带来 INCONCLUSIVE 信号 → RL learned dispatcher 成本高，预期增益低
2. **Self-Discover (`ours_27`) 对比** 显示: 即使非常好的 strategy library，单次 execute 在 3-flash 上只比 baseline 高 ~4pp (因为 LLM 内部推理已经很强)
3. **v16 的 route + LLM-select 组合 (rule-based)** 已经把 dispatcher 的信号价值榨出来了，**进一步用 RL 训练 unlikely to payoff**

### 实际起作用的 "dispatcher-like" 机制

1. **Domain router** (`phase2_v12_framework.py:MATH_SCI`): 硬编码 if-else
2. **v3 selections** (`phase2_v3_selections.json`): 每问题 Flash 从 75 wisdom 中选 3-5 条，冻结为静态缓存
3. **同域判例检索** (`phase2_v15_exemplar_framework.py:build_same_domain_exemplar`): 运行时 embedding cosine similarity

这三个组合起来 = **"零训练的 dispatcher"**，比原设计简单很多但有效。

### 仍未做到的原意

- ❌ **RL 训练循环**: 无 reward model、无 policy gradient、无经验回放
- ❌ **learned feature extractor**: 用的是 off-the-shelf sentence-transformer
- ❌ **调度器可训练 RL baseline 对比**: 因为从来没训，没法对比
- ❌ **多领域 benchmark 评估论文**: 我们跑的是 100 题 + 50 held-out，远小于原设计的 benchmark_problems.json

### 如果重启这份 dev doc 应该做什么

**两条路**:

1. **接受 v16 的 zero-training 架构为 Phase 1 的实际产物** (推荐)
   - 把原 RL 规划重写为 "rule-based routing + LLM semantic selection"
   - 论文 framing: "when does RL dispatcher worth training?" 答: 不在这个问题域
   
2. **尝试 RL as future work**: v16 仍然有部分域（如 math 33%）表现弱，可能 RL 能在这些痛点上找到更好的 routing
   - 但需要 reward model（目前依赖 LLM judge，不是 scalar reward）

### 启示

**原设计的最大假设错了**: 假设"LLM 不会主动调用哲学方法论"。事实是：
- 3-flash 内部的 CoT thinking 已经做了很多方法论工作（thinking tokens ~1000 个）
- 外部给策略反而稀释它的自主推理空间

**v16 的成功路径是不同的**: 不"选策略给它"，而是"**给它案例让它抽象**"（判例法原则）。

**v16 artifact 索引**:
- Code: `phase one/scripts/validation/phase2_v12_framework.py` (domain router), `phase2_v15_exemplar_framework.py` (same-domain retrieval)
- Data: `signal_embeddings.npz`, `phase2_v3_selections.json`
- Result: `v16_final_results.md`

---

## 0. 文档概述

### 0.1 本阶段在整体架构中的位置

本阶段是整个"递归假设-验证智能体"系统中**从知识到行动**的桥梁。阶段零构建了哲学方法论知识库（语义记忆），本阶段要回答的核心问题是：**面对一个具体问题，应该调用知识库中的哪条策略？**

这对应了 Claude.md 讨论中确定的关键洞察：LLM 已经在预训练中隐式地学到了大量哲学方法论（它读过 Polya 的 *How to Solve It*，读过所有科学哲学教科书），但它**不会主动把这些原则当作操作指令来执行**。本阶段的目标就是构建一个显式的调度层，让 LLM 在面对具体问题时，将已有的知识转化为结构化的策略选择。

**三组件架构：**
- **哲学知识库**（阶段零输出）：存放人类已有的方法论规则，每条规则附带适用条件和历史案例
- **调度器**（本阶段核心）：面对当前问题，从知识库中选择最可能有效的策略。这是 RL 训练的对象
- **执行器**：按照选定策略指导具体的问题求解过程。这是 LLM 本身

后续阶段的依赖关系：阶段二的经验反馈机制需要调度器已经能做出有意义的策略选择（即使不完美），才能从执行结果中提取有价值的反馈信号。阶段三的形式化层需要调度器积累的策略-问题匹配数据来验证范畴论度量的有效性。

### 0.2 本阶段目标

设计并训练一个策略调度器，使 LLM 智能体在多领域任务上能系统性地选择并执行阶段零知识库中的方法论策略，在策略选择准确率和任务完成率上显著优于无调度器的 baseline。

### 0.3 交付物

1. 一个可训练的调度器模块（`dispatcher/`），包含策略选择模型和推理接口
2. 一个多领域任务环境（`task_env/`），支持调度器的训练和评估
3. 一个问题特征提取器（`feature_extractor/`），将自然语言问题描述转化为调度器可用的结构化特征向量
4. 一套评估协议和 baseline 实现（`evaluation/`）
5. 调度器学到的策略-条件映射的可视化分析（`analysis/`）
6. 一篇可投稿的系统论文

### 0.4 时间预算

总计 6-8 个月。架构设计与任务环境构建：6-8 周。调度器训练与调参：8-10 周。多领域评估与消融实验：4-6 周。分析、可视化与论文写作：4 周。

---

## 1. 系统架构

### 1.1 总体架构图

```
                    ┌─────────────────────────────┐
                    │       问题输入 (自然语言)      │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │     问题特征提取器             │
                    │  (Problem Feature Extractor)  │
                    │  输出: 结构化特征向量 φ(P)      │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │          调度器               │
                    │       (Dispatcher)            │
                    │                               │
                    │  输入: φ(P), 知识库快照         │
                    │  输出: 策略 ID + 置信度         │
                    │  训练: RL (PPO/DPO)           │
                    └──────┬───────────┬───────────┘
                           │           │
              ┌────────────▼──┐   ┌───▼────────────┐
              │  知识库检索     │   │  fallback 图    │
              │ (阶段零输出)   │   │ (策略失败时)     │
              └────────────┬──┘   └───┬────────────┘
                           │          │
                    ┌──────▼──────────▼──────────┐
                    │        执行器 (LLM)          │
                    │                              │
                    │  输入: 策略 prompt + 问题      │
                    │  输出: 执行轨迹 + 结果         │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │       结果评估器              │
                    │    (Outcome Evaluator)        │
                    │                               │
                    │  输出: 奖励信号 r              │
                    │  → 反馈给调度器的 RL 训练      │
                    │  → 写入经验日志 (供阶段二)      │
                    └─────────────────────────────┘
```

### 1.2 调度器详细设计

调度器是本阶段唯一的新模块。其他所有组件（知识库、LLM 执行器、任务环境）都使用已有的可靠组件。

**设计原则：调度器不生成策略内容，只做选择。** 它从阶段零知识库的策略列表中选择最匹配当前问题的策略 ID。策略的具体执行步骤（`operational_steps`）和执行 prompt（`strategy_prompts`）全部来自知识库，调度器不修改它们。

#### 1.2.1 输入空间

调度器在每个决策步接收以下信息：

```python
@dataclass
class DispatcherInput:
    # 问题特征向量
    problem_features: ProblemFeatures
    # 知识库当前快照（阶段零的导出格式）
    kb_snapshot: KBSnapshot
    # 历史上下文（本次任务已经尝试过的策略及其结果）
    history: List[StrategyAttempt]
    # 自反思输入（参考 Reflexion 的"语义梯度"）
    # 当同类问题上次策略选择失败时，将归因分析结果注入
    recent_reflection: Optional[str] = None
    # 执行中状态（用于中途策略切换，参考 ReMA 多轮交互）
    mid_execution_context: Optional[MidExecutionContext] = None
    # 跨问题上下文：最近解决的相关问题的策略和结果
    # 真实使用中问题往往是序列出现的（同一项目的子问题），
    # 前一个问题的策略选择结果应该影响后一个问题的决策
    cross_problem_context: Optional[CrossProblemContext] = None

@dataclass
class CrossProblemContext:
    # 最近 N 个已完成问题的摘要（按时间倒序）
    recent_problems: List[ProblemSummary]   # 最多保留 5 个
    # 最近问题中策略的成功/失败统计
    recent_strategy_stats: Dict[str, RecentStats]
    # 是否处于同一项目/会话中
    same_session: bool = True
```

**跨问题上下文的使用方式：**
- 方案 A（轻量级网络）：将 `recent_strategy_stats` 编码为 K 维向量（每条策略最近的成功率），拼接到输入特征中
- 方案 B（LLM 调度器）：将 `recent_problems` 的摘要追加到 prompt 中（"最近 3 个相关问题都用分而治之成功了"）

直觉上：如果最近 3 个同类子问题都用 S02 成功了，第 4 个子问题选 S02 的先验应该更高——调度器不需要每次从零开始评估。

其中 `ProblemFeatures` 的结构为：

```python
@dataclass
class ProblemFeatures:
    # === 文本特征 ===
    problem_embedding: np.ndarray       # 问题描述的语义嵌入 (dim=768 或 1536)
    
    # === 结构特征 (从问题描述中提取) ===
    domain: str                         # 领域标签 (software, math, science, ...)
    component_count: int                # 涉及的组件/变量数量
    coupling_estimate: float            # 组件间耦合度估计 (0.0-1.0)
    has_baseline: bool                  # 是否存在已知可工作的基准状态
    randomness_level: float             # 问题中的随机性程度 (0.0-1.0)
    decomposability: float              # 问题是否可自然分解为子问题 (0.0-1.0)
    constraint_count: int               # 约束条件数量
    information_completeness: float     # 信息完整度 (0.0-1.0)
    reversibility: float                # 操作的可逆性程度 (0.0-1.0)
    
    # === 元特征 ===
    difficulty: str                     # 难度等级 (easy, medium, hard)
    time_pressure: float                # 时间压力 (0.0-1.0)
    failure_cost: float                 # 失败代价 (0.0-1.0)
```

**关键设计决策：** 结构特征不是人工标注的，而是由特征提取器从自然语言问题描述中自动推断的。这保证了系统在部署时不需要人工干预。特征提取器本身用 LLM 实现（见 1.3 节），但它不是 RL 训练的对象——它是一个固定的、预训练好的组件。

#### 1.2.2 动作空间

```python
@dataclass
class DispatcherAction:
    primary_strategy: str               # 策略/组合/特殊行动 ID
    confidence: float                   # 选择置信度 (0.0-1.0)
    backup_strategy: Optional[str]      # 备选策略 ID（当主策略失败时使用）
    execution_hint: Optional[str]       # 对执行器的额外提示（如"关注耦合问题"）
```

**动作空间由三类行动组成：**

| 类型 | ID 范围 | 数量 | 说明 |
|------|--------|------|------|
| 单策略 | S01-S23 | 23 | 阶段零知识库中的元策略（含 S21-S23 元决策策略） |
| 策略组合 | COMP_001-005 | 5 | 阶段零定义的预验证策略序列 |
| 特殊行动 | SPECIAL_* | 1 | 见下文 |
| **总计** | | **29** | |

**特殊行动——SPECIAL_GATHER_INFO（"先收集信息"）：**

有些问题在初始描述中信息不完整（`information_completeness` 很低），此时选择任何具体策略都可能是盲目的。人类面对这种情况会先"调研"——收集更多信息后再决策。

当调度器选择 `SPECIAL_GATHER_INFO` 时，执行器不按任何策略的操作步骤执行，而是执行一组预定义的信息收集操作：
1. 要求 LLM 列出"解决此问题还需要知道什么"
2. 尝试从问题描述中推断隐含的约束和假设
3. 将收集到的信息追加到问题描述中，重新运行特征提取
4. 用更新后的特征重新调用调度器选择策略

**SPECIAL_GATHER_INFO 的奖励：** 如果信息收集后重新选择的策略成功了，`SPECIAL_GATHER_INFO` 获得 0.7 × 最终奖励（成功了但多花了一步，略扣分）。如果信息收集后仍然失败，获得 0.1（至少尝试了，比直接选错策略的 0 好一点）。

**策略组合（COMP_*）的训练集成：**

当调度器选择 COMP_001（如 S06→S01）时，执行流程如下：
1. 按序列中的第一条策略（S06）执行
2. 检查过渡条件是否满足（COMP_001 的 `transition_condition`）
3. 如果满足，切换到序列中的下一条策略（S01）继续执行
4. 如果不满足（S06 失败或过渡条件不成立），视为组合失败

**组合策略的奖励计算：**

```python
def compute_composition_reward(
    comp: StrategyComposition,
    step_outcomes: List[StepOutcome]
) -> float:
    """
    组合策略的奖励 = 最终步成功奖励 × 过渡效率因子。
    """
    if not step_outcomes:
        return 0.0
    
    # 最终步的结果决定主奖励
    final_outcome = step_outcomes[-1]
    base_reward = compute_reward(final_outcome)
    
    # 过渡效率因子：所有过渡是否顺利
    n_transitions = len(step_outcomes) - 1
    smooth_transitions = sum(
        1 for i in range(n_transitions)
        if step_outcomes[i].transition_success
    )
    transition_factor = (
        (smooth_transitions / n_transitions) if n_transitions > 0
        else 1.0
    )
    
    # 组合策略的奖励 = 基础奖励 × 过渡因子 × 0.95
    # 0.95 的轻微折扣：鼓励在单策略能解决时不使用组合（简单优先）
    return base_reward * transition_factor * 0.95
```

**组合策略的一致性评估：** 需要验证执行轨迹是否**按序**遵循了组合中各策略的步骤，且过渡条件被正确触发。用 LLM 裁判做二阶段判断：先判断"执行轨迹是否包含两个明显的阶段"，再分别判断每个阶段是否遵循了对应策略。

**为什么不用连续动作空间（如策略的加权混合）：** 可解释性。离散选择允许我们精确追踪"调度器为什么选了这条策略"，这是阶段二经验反馈的前提。如果调度器输出的是策略的连续混合权重，归因分析会困难得多。

#### 1.2.3 输出与执行流程（含中途策略切换）

**设计动机（参考 ReMA）：** ReMA 的核心创新是多轮交替的 meta-thinking 和 reasoning——meta-thinking agent 可以在执行过程中多次介入。Phase 1 的原始设计是单次决策（选策略 → 执行完 → 看结果），无法在执行中途发现策略选错了并及时切换。ReMA 的实验显示多轮 meta-thinking 在 OOD 任务上有显著优势。

执行流程支持两种模式：

**模式 A：整体执行（默认，适用于简单任务）**

1. 从知识库中检索 `S_k` 的 `strategy_prompts`
2. 将策略 prompt 与原始问题描述拼接，构造执行器的输入
3. 调用 LLM 执行器生成完整执行轨迹
4. 结果评估器评估执行结果，产生奖励信号

**模式 B：分步执行与中途切换（适用于复杂任务，difficulty="hard"）**

```python
def execute_with_mid_switch(
    dispatcher, executor, task, kb_snapshot, max_switches=2
):
    """
    ReMA 风格的多轮执行：执行器每完成一个操作步骤后，
    调度器检查是否需要中途切换策略。
    """
    strategy = dispatcher.select(task.features, kb_snapshot)
    switches = 0
    all_step_results = []
    
    for step in strategy.operational_steps:
        step_result = executor.run_one_step(step, task)
        all_step_results.append(step_result)
        
        # 调度器评估是否需要中途切换
        if switches < max_switches:
            mid_context = MidExecutionContext(
                current_strategy=strategy.id,
                completed_steps=len(all_step_results),
                step_results=all_step_results,
                # 关键信号：当前步骤是否暴露了策略不适用的证据
                distress_signal=step_result.difficulty_score > 0.8
            )
            
            if dispatcher.should_switch(
                task.features, kb_snapshot, mid_context
            ):
                # 生成反思信息（Reflexion 风格语义梯度）
                reflection = generate_switch_reflection(
                    strategy, all_step_results
                )
                # 用反思信息辅助选择新策略
                strategy = dispatcher.select(
                    task.features, kb_snapshot,
                    recent_reflection=reflection,
                    mid_execution_context=mid_context
                )
                switches += 1
                all_step_results = []  # 新策略从头执行
    
    return ExecutionOutcome(
        trajectory=all_step_results,
        switches=switches,
        ...
    )
```

**何时使用模式 B：** 任务难度为 "hard" 或策略组合（COMP_*）类型时自动启用。简单任务仍用模式 A 以节省调度器调用次数。

**中途切换的后续流程（两种模式共用）：**

5. 如果主策略失败且存在 `backup_strategy`，跳转到步骤 1 使用备选策略
6. 如果备选策略也失败，查询知识库的 `fallback_graph`（阶段零 5.3 节），选择 fallback 策略
7. 将完整的执行记录（含中途切换信息）写入经验日志

**最大重试次数：** 3 次（1 次主策略 + 1 次备选 + 1 次 fallback）。模式 B 的中途切换不计入重试次数——它是策略内部的调整，不是策略失败后的重选。

### 1.3 问题特征提取器

特征提取器将自然语言的问题描述转化为调度器可用的结构化特征向量 `ProblemFeatures`。

**实现方式：** 使用一个固定的 LLM（不参与 RL 训练）进行结构化信息提取。具体做法是将问题描述和一个 JSON Schema 模板一起输入 LLM，要求 LLM 填充模板中的各个字段。

```python
FEATURE_EXTRACTION_PROMPT = """
你是一个问题分析专家。给定以下问题描述，请分析其结构特征并填写 JSON 模板。

问题描述：
{problem_description}

请输出以下 JSON（不要添加其他文字）：
{
    "domain": "问题所属领域 (software_engineering / mathematics / science / business / daily_life / engineering)",
    "component_count": "涉及的独立组件或变量数量 (整数)",
    "coupling_estimate": "组件间耦合度 (0.0=完全独立, 1.0=完全耦合)",
    "has_baseline": "是否存在已知可工作的基准状态 (true/false)",
    "randomness_level": "问题中的随机性程度 (0.0=完全确定, 1.0=完全随机)",
    "decomposability": "问题是否可分解为独立子问题 (0.0=不可分, 1.0=完全可分)",
    "constraint_count": "显式和隐式约束条件的数量 (整数)",
    "information_completeness": "给定信息的完整度 (0.0=几乎无信息, 1.0=信息完全)",
    "reversibility": "操作的可逆性 (0.0=完全不可逆, 1.0=完全可逆)",
    "difficulty": "难度 (easy/medium/hard)"
}
"""
```

**校准方法：** 使用阶段零的标注数据集（`benchmark_problems.json`）中人类标注的问题特征作为 ground truth，评估特征提取器在各个字段上的准确性。对于连续值字段，使用 MAE（平均绝对误差）；对于离散值字段，使用准确率。

**可靠性保障：** 
- 每个问题运行 3 次特征提取，取中位数（连续值）或众数（离散值）
- 如果 3 次结果的方差过大（连续值标准差 > 0.3，或离散值 3 次不一致），标记该特征为 "uncertain"，调度器在匹配时降低该特征的权重

### 1.4 结果评估器

结果评估器将执行器的输出转化为标量奖励信号，用于调度器的 RL 训练。

**评估维度：**

| 维度 | 权重 | 计算方式 |
|------|------|---------|
| 任务完成度 | 0.35 | 任务是否被正确解决（二元/连续） |
| 策略一致性 | 0.10 | 执行轨迹是否真正遵循了所选策略的步骤 |
| 效率 | 0.10 | 相对于基准步数的归一化步数 |
| 部分进展 | 0.10 | 即使未完全解决，问题规模/复杂度是否降低了 |
| 步进进度 | 0.20 | 每一步是否让问题更接近解决（MRT/MEL 稠密过程奖励） |
| 置信度校准 | 0.15 | 置信度是否与实际成功率匹配（见下文） |

**奖励函数：**

```python
def compute_reward(
    outcome: ExecutionOutcome,
    dispatcher_action: DispatcherAction
) -> float:
    r_completion = 1.0 if outcome.success else (
        0.5 if outcome.partial_success else 0.0
    )
    r_consistency = compute_strategy_consistency(
        outcome.trajectory, outcome.selected_strategy
    )
    r_efficiency = max(0, 1.0 - outcome.total_steps / outcome.baseline_steps)
    r_progress = compute_progress_score(outcome)
    r_step_progress = compute_step_progress_reward(outcome.trajectory)
    r_calibration = compute_confidence_calibration_reward(
        dispatcher_action.confidence, outcome.success
    )
    
    reward = (0.35 * r_completion +
              0.10 * r_consistency +
              0.10 * r_efficiency +
              0.10 * r_progress +
              0.20 * r_step_progress +
              0.15 * r_calibration)
    
    # 策略选择失败的惩罚：如果用了 fallback 才成功，主策略选择扣分
    if outcome.used_fallback:
        reward *= 0.7
    
    # 中途切换的奖励调整（模式 B）
    if outcome.switches > 0 and outcome.success:
        reward *= 0.9
    
    # 组合策略的特殊奖励（见 1.2.2 节）
    if outcome.is_composition:
        reward = compute_composition_reward(
            outcome.composition, outcome.step_outcomes
        )
    
    return reward


def compute_confidence_calibration_reward(
    confidence: float, success: bool
) -> float:
    """
    置信度校准奖励：惩罚 confidence 与实际结果的偏差。
    
    Phase 4 的策略空白检测依赖调度器的低置信度信号。
    如果 confidence 没有被校准，Phase 4 可能检测到大量
    虚假空白（调度器胡乱给低置信度）或遗漏真实空白（过度自信）。
    
    奖励逻辑：
    - 高置信度 + 成功 → 高奖励（自信且正确）
    - 低置信度 + 失败 → 中等奖励（诚实承认不确定）
    - 高置信度 + 失败 → 低奖励（过度自信）
    - 低置信度 + 成功 → 中等奖励（过度保守但无害）
    """
    actual = 1.0 if success else 0.0
    # Brier score 风格：(confidence - actual)^2 越小越好
    calibration_error = (confidence - actual) ** 2
    # 转化为 [0, 1] 的奖励（误差为 0 → 奖励 1，误差为 1 → 奖励 0）
    return 1.0 - calibration_error


def compute_step_progress_reward(
    trajectory: List[TrajectoryStep]
) -> float:
    """
    MRT/MEL 风格的稠密过程奖励（参考 MRT Section 6 的 progress bonus
    和 MEL 3.4 节的 dense process reward）。
    
    标准 outcome-only RL 的问题是：调度器无法区分"稳步接近正确"
    和"走了完全错误的路"——两者都在最后才得到一个 0/1 信号。
    步进奖励在执行的每一步都提供反馈。
    """
    if len(trajectory) < 2:
        return 0.5  # 无法计算步进进度
    
    progress_scores = []
    for i in range(1, len(trajectory)):
        complexity_before = trajectory[i-1].remaining_complexity
        complexity_after = trajectory[i].remaining_complexity
        
        if complexity_after < complexity_before:
            progress_scores.append(1.0)   # 正进展：问题变简单了
        elif complexity_after == complexity_before:
            progress_scores.append(0.5)   # 无进展
        else:
            progress_scores.append(0.0)   # 负进展：问题变更复杂了
    
    return np.mean(progress_scores) if progress_scores else 0.5
```

**策略一致性的计算：** 这是一个非平凡的问题。我们需要判断 LLM 执行器在执行任务时，是否真正遵循了所选策略的 `operational_steps`。实现方式是用另一个 LLM（裁判模型，不参与训练）对执行轨迹进行分析：

```python
CONSISTENCY_JUDGE_PROMPT = """
以下是策略 "{strategy_name}" 的执行步骤：
{operational_steps}

以下是智能体的实际执行轨迹：
{execution_trajectory}

请判断智能体是否真正遵循了该策略的步骤（0.0-1.0）：
- 1.0：完全按照策略步骤执行
- 0.5：大致遵循但有显著偏离
- 0.0：完全没有遵循策略步骤
"""
```

---

## 2. 训练环境

### 2.1 任务来源

训练环境需要覆盖多个领域，以确保调度器学到的是通用的策略调度能力，而不是某个特定领域的技巧。任务来源有三类：

**A 类：阶段零标注数据集（直接复用）**
- 来源：`benchmark_problems.json`（阶段零交付的 100-150 道标注问题）
- 优势：每道题已有人类标注的最优策略，可直接作为监督信号
- 用途：预训练调度器的初始策略（warm-start），以及最终评估

**B 类：可自动验证的任务集（新构建）**
- 这是训练环境的主体。RL 需要大量的交互数据，100-150 道人类标注题不够
- 要求：任务的执行结果可以被自动评估（不需要人工判断成功/失败）

| 领域 | 任务来源 | 自动评估方式 | 规模 |
|------|---------|------------|------|
| 代码调试 | SWE-bench Lite / HumanEval+ 变体 | 测试用例通过率 | 300-500 题 |
| 数学证明 | miniF2F / MATH 数据集 | 形式验证器 / 答案匹配 | 300-500 题 |
| 逻辑推理 | ARC / LogiQA / 自建推理题 | 答案匹配 | 200-300 题 |
| 科学实验设计 | 自建：给定假设 + 数据，设计实验验证 | 实验设计评分准则（LLM 裁判） | 100-200 题 |
| 工程规划 | 自建：项目分解与排序任务 | 与参考方案的对比评分 | 100-200 题 |
| 商业决策 | 自建：简化的资源分配/风险评估 | 模拟执行结果 | 100-150 题 |

**总规模目标：** 1000-1800 道题，其中 70% 用于训练，15% 用于验证，15% 用于测试。

**C 类：对抗任务集（用于鲁棒性测试）**
- 专门设计的"策略陷阱"任务：表面特征暗示应该用策略 A，但实际上策略 B 才是正确的
- 例如：一个看起来可以分而治之的问题，但子问题之间存在隐藏的强耦合，使得分治法失效
- 规模：50-100 题
- 用途：仅用于评估，不参与训练

### 2.2 任务的策略标注

每道训练任务需要以下标注：

```json
{
  "task_id": "task_sw_001",
  "description": "你正在调试一个 Web 应用...",
  "domain": "software_engineering",
  "difficulty": "medium",
  
  "strategy_annotations": {
    "optimal_strategies": ["S01", "S17"],
    "acceptable_strategies": ["S02", "S14"],
    "explicitly_bad_strategies": ["S09"],
    "annotation_source": "human | heuristic | llm_generated",
    "annotation_confidence": 0.85
  },
  
  "auto_evaluation": {
    "type": "test_cases | answer_match | llm_judge | simulation",
    "evaluation_config": { ... }
  },
  
  "ground_truth_features": {
    "component_count": 5,
    "coupling_estimate": 0.3,
    "has_baseline": true,
    ...
  }
}
```

**标注策略：**

对于 A 类任务，标注由阶段零的人类标注者完成。

对于 B 类任务，由于规模太大无法全部人工标注，采用以下半自动流程：
1. **启发式预标注：** 根据任务的结构特征和知识库中策略的 `applicability_conditions` 进行自动匹配，生成候选最优策略
2. **LLM 辅助标注：** 用一个强 LLM（如 GPT-4/Claude）对每道任务生成策略推荐和理由
3. **人工抽样审核：** 随机抽取 20% 的任务进行人工审核，如果审核一致率低于 80%，回退到更大比例的人工标注
4. **交叉验证：** 对同一道题用启发式和 LLM 分别标注，只有两者一致的标注才被视为高置信度

### 2.3 任务环境接口

所有任务统一实现以下接口，使调度器可以与不同领域的任务无缝交互：

```python
class TaskEnvironment(ABC):
    """所有任务环境的基类"""
    
    @abstractmethod
    def reset(self, task_id: str) -> TaskObservation:
        """重置环境到指定任务的初始状态"""
        pass
    
    @abstractmethod
    def get_problem_description(self) -> str:
        """返回当前任务的自然语言描述"""
        pass
    
    @abstractmethod
    def execute_with_strategy(
        self, 
        strategy_prompt: str,
        llm_executor: LLMExecutor
    ) -> ExecutionOutcome:
        """用给定的策略 prompt 指导 LLM 执行任务"""
        pass
    
    @abstractmethod
    def evaluate(self, outcome: ExecutionOutcome) -> float:
        """评估执行结果，返回 [0, 1] 的分数"""
        pass
    
    @abstractmethod
    def get_ground_truth_strategy(self) -> List[str]:
        """返回该任务的标注最优策略列表"""
        pass


class ExecutionOutcome:
    """执行结果的标准格式"""
    success: bool
    partial_success: bool
    trajectory: List[TrajectoryStep]     # 执行轨迹
    total_steps: int
    wall_clock_seconds: float
    llm_output: str                      # LLM 的原始输出
    evaluation_score: float              # 自动评估分数
    failure_reason: Optional[str]
```

---

## 3. 调度器训练

### 3.1 训练范式选择

调度器的训练分两个阶段：监督预训练（warm-start）和强化学习微调。

#### 3.1.1 阶段 A：监督预训练

**目的：** 为调度器提供一个合理的初始策略，避免 RL 从完全随机的策略开始（冷启动问题）。

**训练数据：** 阶段零的人类标注数据 + B 类任务的高置信度自动标注

**方法：** 标准的多分类学习。输入是 `ProblemFeatures`，输出是策略 ID 的概率分布，用交叉熵损失训练。

```
L_supervised = -Σ_i log p(s_i* | φ(P_i))
```

其中 `s_i*` 是任务 `P_i` 的标注最优策略，`φ(P_i)` 是问题特征向量。

**预期效果：** 预训练后的调度器应能达到 50-60% 的 top-1 策略选择准确率（相对于人类标注的最优策略）。这个准确率不需要很高——它只是 RL 的起点。

#### 3.1.2 阶段 B：RL 微调

**目的：** 通过与任务环境的实际交互，学习人类标注未能覆盖的细微策略选择偏好和条件-策略映射。

**算法选择：** 因架构而异

- **方案 A（轻量级网络）使用 PPO**：网络本身有 value head，PPO 在小型离散动作空间下稳定高效
- **方案 B（LLM 调度器）使用 GRPO**（参考 ReMA/EvolveR/MEL 的共识）：GRPO 不需要学习价值函数，通过组内相对比较估计优势函数，对 LLM 更合适且训练更稳定

**方案 A 的 PPO 训练配置：**

```python
ppo_config = {
    "learning_rate": 1e-4,
    "clip_epsilon": 0.2,
    "value_loss_coef": 0.5,
    "entropy_coef": 0.05,          # 鼓励早期探索
    "entropy_decay": 0.995,         # 随训练逐步减少探索
    "gamma": 0.99,                  # 折扣因子
    "gae_lambda": 0.95,
    "num_epochs_per_update": 4,
    "batch_size": 64,
    "max_episodes": 50000,
    "early_stopping_patience": 2000,
}
```

**方案 B 的 GRPO 训练配置（参考 ReMA 和 EvolveR）：**

```python
grpo_config = {
    "learning_rate": 1e-5,
    "clip_epsilon": 0.2,
    "group_size": 8,               # 每个问题生成 8 条策略选择做组内比较
    "kl_penalty_beta": 0.05,       # KL 散度约束，防止策略漂移过远
    # 无 value_loss_coef — GRPO 不需要 value function
    "entropy_coef": 0.03,
    "batch_size": 32,
    "max_episodes": 50000,
    "early_stopping_patience": 2000,
}
```

**探索机制：** 

初期（前 10000 episodes）使用较高的 entropy bonus（`entropy_coef=0.05`），鼓励调度器尝试知识库中所有策略，包括那些人类标注较少推荐的策略。这可能发现某些策略在特定条件下的意外有效性。

后期（10000+ episodes）逐步降低 entropy bonus，让调度器收敛到稳定的策略选择策略。

**策略坍缩监控（参考 ReMA Figure 4）：**

ReMA 发现 1B 参数的小模型在训练中快速坍缩到最简单的 meta-action（EMPTY），而 8B 模型能根据难度选择不同策略。Phase 1 方案 A 的调度器只有 ~300K 参数，有类似的坍缩风险。训练过程中必须持续监控策略覆盖率：

```python
def monitor_strategy_collapse(
    recent_selections: List[str],
    window: int = 500,
    collapse_threshold: float = 0.6
) -> Optional[CollapseWarning]:
    """
    如果最近 500 次选择中，前 3 策略的占比超过 60%，
    说明调度器正在坍缩到少数"安全"策略。
    """
    from collections import Counter
    counts = Counter(recent_selections[-window:])
    total_strategies_used = len(counts)
    top3_ratio = sum(c for _, c in counts.most_common(3)) / window
    
    if top3_ratio > collapse_threshold:
        return CollapseWarning(
            top3_strategies=[s for s, _ in counts.most_common(3)],
            top3_ratio=top3_ratio,
            total_strategies_used=total_strategies_used,
            remedy="increase entropy_coef by 50% for next 2000 episodes"
        )
    return None
```

每 500 episodes 检查一次。如果检测到坍缩，自动将 `entropy_coef` 提升 50% 并持续 2000 episodes。

**早停检测（参考 Memory-R1 的数据效率）：**

Memory-R1 用仅 152 条训练数据就实现了有效的 RL 训练。如果调度器的动作空间（15-20 策略）足够结构化且有阶段零的标注数据做 warm-start，可能远不需要 50000 episodes。增加早停检测：

```python
early_stopping_config = {
    "check_interval": 1000,          # 每 1000 episodes 检查一次
    "min_episodes": 5000,            # 至少训练 5000 episodes
    "improvement_threshold": 0.01,   # 连续 N 次检查提升 < 1% 则停止
    "patience": 3,                   # 连续 3 次不达标才停止
}
```

**课程学习（Curriculum Learning）：**

训练任务按难度分三批次引入：

| 批次 | Episodes | 任务难度 | 任务特征 |
|------|----------|---------|---------|
| 第一批 | 0 - 15000 | easy | 最优策略明显，几乎只有一个合理选择 |
| 第二批 | 10000 - 35000 | easy + medium | 混合简单和中等难度，有 2-3 个合理策略 |
| 第三批 | 25000 - 50000 | easy + medium + hard | 全难度覆盖，包含需要策略组合的复杂问题 |

批次之间有重叠，确保平滑过渡。

### 3.2 调度器模型架构

调度器本身可以用两种架构实现。本阶段先实现方案 A 作为主要方案，方案 B 作为对比实验。

**方案 A：基于嵌入的轻量级网络**

```
ProblemFeatures (dim=d)
        │
        ▼
   Linear(d, 256) + ReLU
        │
        ▼
   Linear(256, 128) + ReLU + Dropout(0.1)
        │
        ▼
   ┌────┴────┐
   │         │
   ▼         ▼
策略头      价值头
Linear      Linear
(128, K)   (128, 1)
   │         │
   ▼         ▼
Softmax    Scalar
π(s|φ)     V(φ)
```

其中 K = 知识库中策略数量（15-20），d = 问题特征向量维度。

输入特征 `φ(P)` 的构造：
- 问题文本嵌入（768 维，来自预训练 encoder）
- 结构特征（约 10 维，来自 ProblemFeatures 中的数值字段）
- 知识库适用条件匹配分数（K 维，每条策略的 favorable/unfavorable 条件与当前问题的匹配度）

总输入维度 d ≈ 768 + 10 + K ≈ 798

**优势：** 训练快速，可解释性强（可以直接分析隐藏层的激活模式）。
**风险：** 可能无法捕捉策略之间的复杂依赖关系。

**方案 B：基于 LLM 的调度器（ReMA 风格）**

直接用一个小型 LLM（如 Qwen2.5-3B 或 Llama-3.2-3B）作为调度器。输入是问题描述和知识库的自然语言摘要，输出是策略选择的自然语言推理过程 + 策略 ID。

```
System Prompt:
你是一个策略调度专家。你拥有以下 {K} 种问题解决策略：
{strategy_list_with_descriptions}

当面对一个新问题时，你需要：
1. 分析问题的关键特征
2. 将特征与各策略的适用条件进行匹配
3. 选择最适合的策略并说明理由
4. 如果不确定，给出备选策略

User:
{problem_description}
```

用 RL（PPO）对该 LLM 进行微调，奖励信号来自执行结果。

**优势：** 能利用 LLM 的语义理解能力，处理微妙的问题特征；推理过程以自然语言形式输出，可解释性极强。
**风险：** 训练成本远高于方案 A；LLM 可能绕过策略选择直接生成答案（reward hacking）。

**防止 reward hacking 的措施：** 要求调度器输出严格遵循固定格式（JSON），包含策略 ID 和理由，不包含任何问题解答内容。如果输出格式不符，直接给零奖励。

### 3.3 训练成本估算

| 配置 | 方案 A (轻量级网络) | 方案 B (LLM 调度器) |
|------|-------------------|-------------------|
| 调度器参数量 | ~300K | ~3B |
| 动作空间大小 | 29（23 单策略 + 5 组合 + 1 特殊行动） | 同左 |
| 每 episode LLM 调用 | 特征提取 3 次 + 执行器 1 次 = **4 次** | 调度器 1 次 + 执行器 1 次 = 2 次（特征提取集成在调度器 prompt 中） |
| 50K episodes 总成本（model-free） | ~$800-1500 (API) 或 ~72 GPU-hours (本地) | ~$3000-5000 (API) 或 ~200 GPU-hours (本地) |
| 50K episodes 总成本（model-based 90/10） | ~$250-400 (API) | ~$500-800 (API) |
| 训练时间 | 2-4 天 (单 A100) | 1-2 周 (4x A100) |

**成本注释：** 方案 A 的特征提取需要 3 次 LLM 调用（1.3 节的可靠性保障），之前的估算只计了 1 次。Model-based 模式（90% 模拟）大幅降低真实执行成本，但模拟本身的 LLM 调用成本需要额外计入（见 Phase 0.5 的 3.2 节）。

方案 A 是默认选择。方案 B 仅在方案 A 的性能明显不足时启用。

### 3.4 与世界模型的集成

阶段零点五提供的世界模型是调度器 RL 训练的关键基础设施。**调度器的 RL 训练主要在世界模型中进行**（model-based RL），真实环境仅用于校准和最终评估。

**Model-Based RL 训练流程：**

本阶段使用阶段零点五的 `ModelBasedTrainer` 和 `SimulatedTaskEnvironment`。训练中 90% 的 episodes 在世界模型中运行，10% 在真实环境中运行。

```python
# 训练配置
model_based_config = {
    "real_ratio": 0.1,              # 10% 的 episodes 在真实环境
    "calibration_interval": 500,     # 每 500 episodes 校准一次世界模型
    "fallback_to_model_free": {
        "trigger": "kendall_tau < 0.3",  # 世界模型准确性过低时回退
        "recovery_check_interval": 2000  # 回退后每 2000 episodes 重新检查
    }
}
```

**训练循环：**
1. 调度器选择策略 S_k
2. 世界模型预测执行结果（成功概率、可能的失败模式）
3. 如果世界模型预测置信度高（≥ 0.6）：用预测结果作为奖励信号，不做真实执行
4. 如果世界模型预测置信度低（< 0.6）或到了校准轮次：在真实环境中执行，用真实结果同时更新调度器和世界模型
5. 每 500 episodes 做一次校准，检查世界模型的排序准确性

**成本对比：**

| 训练方案 | 真实执行次数 | 真实执行成本 | 世界模型成本 | 总成本 |
|---------|------------|------------|------------|--------|
| 纯 model-free | 50000 | ~$500-1000 | $0 | ~$500-1000 |
| Model-based (90/10) | 5000 | ~$50-100 | ~$135 | ~$185-235 |

Model-based 方案将真实执行成本降低了约 80%，总成本降低约 70%。

**安全机制：**
- 世界模型准确性低于阈值（Kendall's τ < 0.3）时，自动切换到 model-free 模式
- 最终评估（第 5 章）始终在真实环境中进行，不使用世界模型
- 世界模型的预测偏差作为额外指标记录，用于分析调度器行为

**冷启动：** 系统刚启动时世界模型无数据，前 2000 episodes 使用 50/50 的模拟/真实比例，之后随世界模型数据积累逐步提升到 90/10。详见阶段零点五文档 3.3 节。

---

## 4. 评估方案

### 4.1 评估指标

#### 核心指标

| 指标 | 定义 | 目标值 |
|------|------|-------|
| **策略选择准确率 (Top-1)** | 调度器选择的策略 ∈ 任务标注的 optimal_strategies | ≥ 65% |
| **策略选择准确率 (Top-3)** | 调度器的前 3 选择与 optimal + acceptable 策略的重叠 | ≥ 80% |
| **任务完成率** | 使用调度器选择的策略后任务被成功解决的比例 | ≥ baseline + 15% |
| **人类一致率** | 调度器选择与人类专家选择一致的比例 | ≥ 70% |
| **跨领域泛化** | 在训练中未见过的领域上的策略选择准确率 | ≥ 55% |

#### 辅助指标

| 指标 | 定义 | 用途 |
|------|------|------|
| 策略覆盖率 | 知识库中至少被选择过一次的策略比例 | 检测是否有策略被系统性忽略 |
| 策略-领域分布 | 每个领域中各策略被选择的频率分布 | 分析调度器是否学到了领域特异性 |
| Fallback 使用率 | 主策略失败后使用 fallback 的比例 | 反映主策略选择的质量 |
| 平均执行步数 | 在调度器指导下的平均执行步数 | 反映策略选择的效率 |
| 策略切换一致性 | 面对相似问题，调度器是否给出一致的策略选择 | 检测策略是否稳定 |

### 4.2 Baseline 对比

设计以下 baseline 进行对比，以隔离调度器的贡献：

**Baseline 1：无策略 (No Strategy)**
- LLM 直接解题，不提供任何方法论策略 prompt
- 这是最基本的 baseline，用于验证"显式策略指导是否有价值"

**Baseline 2：随机策略 (Random Strategy)**
- 从知识库中随机选择一条策略传递给 LLM
- 用于验证"策略选择是否重要，还是有任何策略都比没有好"

**Baseline 3：规则匹配 (Rule-Based Matching)**
- 基于问题特征和知识库适用条件的硬编码匹配规则
- 不使用 RL，纯粹基于阶段零知识库中记录的 favorable/unfavorable conditions
- 用于验证 "RL 训练是否比规则匹配学到了更好的调度策略"

**Baseline 4：LLM 自选策略 (LLM Self-Select)**
- 向 LLM 展示知识库中所有策略的摘要，让它自己选择最合适的策略
- 不经过 RL 训练，直接利用 LLM 的预训练知识
- 用于验证"RL 训练的调度器是否优于 LLM 的零样本策略选择能力"

**Baseline 5：EvolveR 风格 (Experience-Distilled)**
- 不使用预定义知识库，让 LLM agent 从自身经验中蒸馏出策略
- 直接与 EvolveR 的方法对比
- 用于验证"人类哲学先验初始化是否优于从零蒸馏"

### 4.3 消融实验

| 消融变量 | 实验设计 | 目标 |
|---------|---------|------|
| 知识库大小 | 分别用 5/10/15/20 条策略的知识库训练 | 策略数量的边际效益 |
| 特征类型 | 分别去掉文本嵌入/结构特征/匹配分数 | 哪类特征对策略选择最重要 |
| 训练领域数 | 分别用 1/3/5 个领域的任务训练 | 跨领域训练数据量的影响 |
| 监督预训练 | 有/无 warm-start | 监督预训练的必要性 |
| Fallback 机制 | 有/无 fallback 图 | Fallback 机制的贡献 |
| 策略一致性奖励 | 奖励中包含/不包含策略一致性项 | 执行一致性是否影响学习效果 |

### 4.4 人类专家评估

除了自动指标外，邀请阶段零的标注者（或新招募的领域专家）进行以下人类评估：

**评估任务：** 给定 50 道 held-out 问题，展示调度器的策略选择和理由（如果用方案 B，直接展示推理过程；如果用方案 A，展示特征匹配分数最高的条件），请专家判断：

1. 策略选择是否合理（合理/部分合理/不合理）
2. 如果不合理，专家会选择什么策略
3. 调度器的策略选择理由是否可理解（可理解/模糊/不可理解）

**目标：** 合理率 ≥ 75%，可理解率 ≥ 80%。

---

## 5. 技术实现

### 5.1 项目文件结构

```
assumption_agent/
├── README.md
├── config/
│   ├── dispatcher_config.yaml        # 调度器配置
│   ├── training_config.yaml          # RL 训练配置
│   └── evaluation_config.yaml        # 评估配置
├── kb/                               # 阶段零知识库 (符号链接或 Git submodule)
│   └── ...                           # 阶段零的完整目录结构
├── dispatcher/
│   ├── models/
│   │   ├── embedding_dispatcher.py   # 方案 A：基于嵌入的轻量级网络
│   │   ├── llm_dispatcher.py         # 方案 B：基于 LLM 的调度器
│   │   └── base_dispatcher.py        # 调度器基类
│   ├── feature_extractor.py          # 问题特征提取器
│   ├── outcome_evaluator.py          # 结果评估器
│   ├── strategy_retriever.py         # 从知识库检索策略 prompt
│   └── fallback_manager.py           # Fallback 策略管理
├── executor/
│   ├── llm_executor.py               # LLM 执行器封装
│   ├── strategy_prompt_builder.py    # 策略 prompt 与问题的拼接
│   └── trajectory_recorder.py        # 执行轨迹记录器
├── task_env/
│   ├── base_env.py                   # TaskEnvironment 基类
│   ├── code_debug_env.py             # 代码调试任务环境
│   ├── math_proof_env.py             # 数学证明任务环境
│   ├── logic_reasoning_env.py        # 逻辑推理任务环境
│   ├── science_experiment_env.py     # 科学实验设计任务环境
│   ├── engineering_planning_env.py   # 工程规划任务环境
│   ├── business_decision_env.py      # 商业决策任务环境
│   └── tasks/                        # 任务数据文件
│       ├── code_debug/
│       ├── math_proof/
│       ├── logic_reasoning/
│       ├── science_experiment/
│       ├── engineering_planning/
│       └── business_decision/
├── training/
│   ├── supervised_pretrain.py        # 阶段 A：监督预训练
│   ├── rl_finetune.py                # 阶段 B：RL 微调 (PPO)
│   ├── curriculum.py                 # 课程学习调度
│   ├── reward_shaping.py             # 奖励函数实现
│   └── world_model_integration.py    # 世界模型加速 (可选)
├── evaluation/
│   ├── baselines/
│   │   ├── no_strategy.py            # Baseline 1
│   │   ├── random_strategy.py        # Baseline 2
│   │   ├── rule_matching.py          # Baseline 3
│   │   ├── llm_self_select.py        # Baseline 4
│   │   └── evolver_style.py          # Baseline 5
│   ├── metrics.py                    # 评估指标计算
│   ├── run_evaluation.py             # 评估脚本
│   └── human_eval_protocol.md        # 人类评估协议
├── analysis/
│   ├── strategy_condition_map.py     # 策略-条件映射可视化
│   ├── domain_distribution.py        # 领域分布分析
│   ├── ablation_analysis.py          # 消融实验分析
│   └── figures/                      # 生成的图表
├── experience_writer/
│   ├── log_writer.py                 # 写入阶段零格式的经验日志
│   └── attribution_analyzer.py      # 生成 attribution 字段
├── scripts/
│   ├── prepare_tasks.py              # 任务数据准备
│   ├── annotate_tasks.py             # 半自动策略标注
│   ├── train.py                      # 训练入口
│   ├── evaluate.py                   # 评估入口
│   └── export_for_phase2.py          # 导出给阶段二的数据
└── tests/
    ├── test_dispatcher.py
    ├── test_feature_extractor.py
    ├── test_evaluator.py
    ├── test_env.py
    └── test_integration.py
```

### 5.2 与阶段零的接口（输入接口）

本阶段从阶段零读取以下数据，格式遵循阶段零 5.3 节定义的导出规范：

```python
class KBSnapshot:
    """知识库快照——调度器的只读视图"""
    
    action_space: List[str]
    # 例: ["S01", "S02", ..., "S20"]
    
    strategy_prompts: Dict[str, str]
    # 例: {"S01": "你现在需要使用控制变量法来解决这个问题..."}
    
    applicability_features: Dict[str, ApplicabilityInfo]
    # 例: {"S01": {"favorable": [...], "unfavorable": [...]}}
    
    fallback_graph: Dict[str, FallbackInfo]
    # 例: {"S01": {"if_failed_try": ["S17", "S14"], "reason": [...]}}
    
    strategy_relationships: Dict[str, List[Relationship]]
    # 策略间的 prerequisite/complementary/alternative/subsumption 关系
```

**加载方式：** 在训练和推理开始时，调用阶段零的 `scripts/export_for_agent.py` 生成最新的知识库快照。快照在一次训练 session 中保持不变（不在训练过程中动态更新知识库——那是阶段二的工作）。

### 5.3 与阶段二的接口（输出接口）

本阶段向阶段二输出以下数据：

**A. 经验日志**

每次任务执行的记录，格式严格遵循阶段零 2.4 节的 experience_log schema：

```python
def write_experience_log(
    task: Task,
    strategy_selection: StrategySelection,
    execution_outcome: ExecutionOutcome,
    attribution: Attribution
) -> str:
    """
    将一次执行记录写入 experience_log/executions/ 目录。
    返回生成的 execution_id。
    
    关键：attribution 字段由 attribution_analyzer.py 自动生成，
    它将执行结果与知识库中策略的适用条件进行对比，
    识别哪些条件被满足、哪些被违反、是否发现了新的条件候选。
    """
    record = {
        "execution_id": generate_execution_id(),
        "timestamp": datetime.utcnow().isoformat(),
        "task": task.to_dict(),
        "strategy_selection": {
            "selected_strategy": strategy_selection.primary_strategy,
            "selection_reason": strategy_selection.reason,
            "alternatives_considered": strategy_selection.alternatives,
            "selector_confidence": strategy_selection.confidence
        },
        "execution_trajectory": execution_outcome.trajectory.to_dict(),
        "outcome": {
            "success": execution_outcome.success,
            "partial_success": execution_outcome.partial_success,
            "failure_reason": execution_outcome.failure_reason
        },
        "attribution": attribution.to_dict()
    }
    # 写入 experience_log/executions/YYYY/MM/
    save_to_experience_log(record)
    return record["execution_id"]
```

**B. 调度器的策略偏好数据**

阶段二需要知道调度器在什么条件下偏好什么策略，以便分析调度器是否学到了知识库中未记录的隐式规则。

```python
def export_dispatcher_preferences() -> Dict:
    """
    导出调度器的学习结果供阶段二分析。
    """
    return {
        "strategy_usage_stats": {
            "S01": {
                "total_selections": 1234,
                "success_rate": 0.72,
                "avg_confidence": 0.81,
                "most_common_domains": ["software_engineering", "science"],
                "most_common_features": {
                    "has_baseline": True,
                    "coupling_estimate": "<0.4"
                }
            },
            ...
        },
        "strategy_confusion_matrix": [...],   # 哪些策略经常被互相替换
        "discovered_patterns": [              # 调度器发现的新模式
            {
                "pattern": "当 component_count > 8 且 coupling > 0.6 时，S02 优于 S01",
                "evidence_count": 47,
                "confidence": 0.78
            }
        ]
    }
```

### 5.4 依赖与技术栈

| 组件 | 技术选择 | 理由 |
|------|---------|------|
| RL 框架 | Stable-Baselines3 或 TRL | PPO 实现成熟，与 HuggingFace 生态兼容 |
| LLM 执行器 | OpenAI API / vLLM + 开源模型 | 训练时用便宜的开源模型，评估时加测 GPT-4/Claude |
| 嵌入模型 | sentence-transformers (all-MiniLM-L6-v2 或 BGE-M3) | 问题文本嵌入 |
| 任务评估 | 领域特定工具 (pytest, Lean4, 自建评估器) | 各领域的自动验证 |
| 实验管理 | Weights & Biases | 训练过程跟踪和超参数记录 |
| 数据格式 | JSON (与阶段零一致) | 跨阶段兼容性 |

---

## 6. 经验归因分析（Attribution）

### 6.1 为什么需要归因

经验归因是连接阶段一（调度器）和阶段二（经验反馈）的桥梁。每次任务执行后，系统不仅要记录"选了什么策略、成功了没"，还要分析**为什么成功/失败与策略选择有关**。

如果没有归因分析，阶段二收到的经验日志只是一堆"策略 X 在任务 Y 上成功/失败"的记录，无法区分以下情况：
- 策略选对了，执行也正确 → 该策略的适用条件被验证
- 策略选对了，但执行错误 → 执行器的问题，不应归因于策略
- 策略选错了，正好歪打正着成功了 → 不应增加该策略的置信度
- 策略选错了，失败了 → 需要区分是因为策略本身不适合还是适用条件判断错误

### 6.2 归因分析流程

```python
class AttributionAnalyzer:
    """
    分析执行结果与策略适用条件的关系。
    输出格式遵循阶段零 2.4 节的 attribution schema。
    """
    
    def analyze(
        self,
        task: Task,
        selected_strategy: str,
        problem_features: ProblemFeatures,
        kb_snapshot: KBSnapshot,
        outcome: ExecutionOutcome
    ) -> Attribution:
        
        strategy_conditions = kb_snapshot.applicability_features[selected_strategy]
        
        # 1. 检查哪些 favorable 条件被满足
        matched_favorable = []
        for cond in strategy_conditions["favorable"]:
            if self._check_condition_match(cond, problem_features):
                matched_favorable.append(cond["condition_id"])
        
        # 2. 检查哪些 unfavorable 条件被触发
        violated_unfavorable = []
        for cond in strategy_conditions["unfavorable"]:
            if self._check_condition_match(cond, problem_features):
                violated_unfavorable.append(cond["condition_id"])
        
        # 3. 如果策略失败且没有触发已知的 unfavorable 条件，
        #    可能发现了新的 unfavorable 条件
        newly_discovered = []
        if not outcome.success and not violated_unfavorable:
            candidate = self._infer_new_condition(
                task, problem_features, outcome
            )
            if candidate:
                newly_discovered.append(candidate)
        
        # 4. 如果策略在已知 unfavorable 条件下意外成功，
        #    可能该条件需要被削弱
        surprising_successes = []
        if outcome.success and violated_unfavorable:
            surprising_successes = violated_unfavorable
        
        return Attribution(
            matched_conditions=matched_favorable,
            violated_conditions=violated_unfavorable,
            newly_discovered_condition_candidates=newly_discovered,
            surprising_successes=surprising_successes
        )
```

`_infer_new_condition` 方法使用 LLM 分析失败原因并尝试推断可能的新适用条件。这个推断出的条件是候选状态（`evidence_strength: "weak"`），需要多次独立验证后才能被阶段二正式写入知识库。

### 6.3 对比式策略分析（参考 MEL 的分叉点识别）

**设计动机：** MEL 3.2 节的核心创新是在同一问题上对比正确和错误的推理轨迹，精确定位"推理开始偏离的那一步"（bifurcation point）。Phase 1 的调度器经常会在同一个问题（或同类问题）上先用策略 A 失败、再用策略 B 成功。这种配对数据包含极其丰富的信息——远比两个独立的 (策略, 结果) 对有价值。

当同一问题上出现策略 A 失败、策略 B 成功的情况时，自动生成对比分析：

```python
class StrategyContrastAnalyzer:
    """
    MEL 风格的分叉点分析：
    对比同一问题上失败策略和成功策略的执行轨迹。
    """
    
    def analyze_contrast(
        self,
        problem_id: str,
        failed_record: ExecutionRecord,
        success_record: ExecutionRecord
    ) -> StrategyContrast:
        
        # 1. 找到两个轨迹开始分化的步骤
        bifurcation = self._find_divergence_point(
            failed_record["execution_trajectory"]["steps"],
            success_record["execution_trajectory"]["steps"]
        )
        
        # 2. 用 LLM 分析分叉原因
        analysis = self._llm_analyze_bifurcation(
            failed_record, success_record, bifurcation
        )
        
        return StrategyContrast(
            problem_id=problem_id,
            failed_strategy=failed_record["strategy_selection"]["selected_strategy"],
            success_strategy=success_record["strategy_selection"]["selected_strategy"],
            bifurcation_step=bifurcation.step_num,
            why_failed_diverged=analysis["failure_cause"],
            why_success_worked=analysis["success_cause"],
            key_condition=analysis["discriminating_condition"]
        )
    
    def _find_divergence_point(
        self, failed_steps, success_steps
    ) -> BifurcationPoint:
        """
        找到两个轨迹的语义分叉点：
        前 N 步可能相似（都在分析问题），
        但从某一步开始采取了不同方向。
        """
        # 用 embedding 相似度逐步比对
        for i in range(min(len(failed_steps), len(success_steps))):
            sim = compute_embedding_similarity(
                failed_steps[i]["action"],
                success_steps[i]["action"]
            )
            if sim < 0.5:  # 行动开始显著不同
                return BifurcationPoint(
                    step_num=i,
                    failed_action=failed_steps[i]["action"],
                    success_action=success_steps[i]["action"]
                )
        
        return BifurcationPoint(step_num=0, ...)  # 从一开始就不同

BIFURCATION_ANALYSIS_PROMPT = """
同一个问题上，两种策略产生了不同的结果。

## 失败策略: {failed_strategy_name}
执行轨迹: {failed_trajectory_summary}
失败原因: {failure_reason}

## 成功策略: {success_strategy_name}
执行轨迹: {success_trajectory_summary}

## 分叉点
在第 {bifurcation_step} 步，两个策略开始采取不同行动：
- 失败策略选择了: {failed_action}
- 成功策略选择了: {success_action}

请分析：
1. 成功策略在分叉点做了什么正确的决定？
2. 失败策略在分叉点犯了什么错误？
3. 决定分叉走向的关键问题条件是什么？
   （即：什么条件下应该像成功策略那样做，而非像失败策略那样做）

输出 JSON:
{{"failure_cause": "...", "success_cause": "...",
  "discriminating_condition": "..."}}
"""
```

**对比分析的三重用途：**
1. **调度器的 reflection 输入**：`discriminating_condition` 作为 `recent_reflection` 注入调度器，帮助它在遇到类似问题时做出更好的选择
2. **Phase 2 经验蒸馏的高质量输入**：对比分析比单条经验的信息密度高得多
3. **训练时的额外信号**：对比对可用于构造 DPO 风格的偏好数据——(问题, 成功策略) > (问题, 失败策略)

---

## 7. 实验设计

### 7.1 核心实验：调度器有效性验证

**假设 H1：** 配备哲学策略调度器的 LLM agent 在多领域任务上的综合完成率显著高于所有 baseline。

**实验设计：**
- 在 B 类任务集的测试集（约 200-270 道题）上运行调度器和 5 个 baseline
- 每个方法在每道题上运行 3 次（消除随机性）
- 使用 paired t-test 或 Wilcoxon signed-rank test 检验显著性（p < 0.05）

**报告指标：** 任务完成率、策略选择准确率（top-1 和 top-3）、平均执行步数、fallback 使用率

### 7.2 核心实验：跨领域泛化

**假设 H2：** 调度器在训练时未见过的领域上，策略选择准确率仍显著高于 Baseline 3（规则匹配）和 Baseline 4（LLM 自选）。

**实验设计：**
- Leave-one-domain-out：每次训练时去掉一个领域的所有任务，在该领域上测试
- 重复 6 次（6 个领域各留出一次）
- 报告平均跨领域准确率和最差领域准确率

### 7.3 关键实验：人类先验 vs 从零蒸馏

**假设 H3：** 用人类哲学先验初始化知识库的调度器，比从零蒸馏策略的 EvolveR 风格系统收敛更快且最终性能更高。

**实验设计：**
- 方法 A：本系统（知识库初始化 + RL 调度器）
- 方法 B：EvolveR 风格（无知识库，从 agent 自身经验蒸馏策略）
- 在相同的训练预算（LLM API 调用次数）下比较两者的学习曲线和最终性能
- 特别关注早期阶段（前 5000 episodes）的性能差异——人类先验的优势应该在早期最明显

**这是论文的核心贡献实验。** 如果 H3 成立，直接证明了"人类哲学可以被装载为 AI 的操作系统"这一核心论点。

### 7.4 分析实验：调度器学到了什么

这不是为了验证假设，而是为了深入理解调度器的行为模式。

**分析 1：策略-特征热力图**
- X 轴：问题特征（component_count, coupling_estimate, ...）
- Y 轴：策略 ID
- 颜色：调度器选择该策略的概率
- 目标：可视化调度器学到的"什么条件下用什么策略"

**分析 2：策略混淆分析**
- 哪些策略对经常被互相替换（调度器在它们之间犹豫）
- 与阶段零标注数据中的人类混淆矩阵对比
- 如果调度器和人类的混淆模式一致，说明调度器学到了与人类相似的策略理解

**分析 3：发现的新模式**
- 调度器是否学到了知识库中未记录的策略-条件映射
- 例如：调度器可能发现"当 information_completeness < 0.3 时，贝叶斯更新 (S12) 优于试错法 (S08)"——这是否与直觉一致？
- 这些发现直接输出给阶段二用于知识库更新

### 7.5 补充实验：训练-部署领域鸿沟检测

**问题：** Phase 1 在 benchmark 任务上训练（SWE-bench、MATH、LogiQA 等），但系统最终要处理的是开放式真实问题——Claude.md 中的例子包括"搭建世界模型外推系统"、"在地铁口卖布朗尼"。跨领域泛化实验（7.2 节 leave-one-domain-out）测试的是同一 benchmark 分布内的领域迁移，不是从 benchmark 到真实世界的迁移。

**实验设计：**

构造 20-30 道**非结构化真实问题**（不来自任何 benchmark，由研究者从自身经历中提取）：
- 来源 1：研究者过去遇到的实际调试/设计/决策问题（如"世界模型外推系统用一次性搭建还是增量替换"）
- 来源 2：日常生活中的非学术问题（如"搬家时如何高效打包"）
- 来源 3：跨学科的开放式问题（如"如何判断一个商业点子是否可行"）

对每道题，请 3 名人类专家独立标注最优策略，然后运行调度器。

**指标：**
- 调度器在非结构化问题上的 top-3 命中率（与 benchmark 上的 top-3 命中率对比）
- 如果非结构化问题上的命中率 < benchmark 上的 60%，说明存在显著的训练-部署鸿沟

**应对鸿沟的方案（如果检测到）：**
1. 在 B 类任务集中增加非结构化问题的比例（从 0% 到 20%）
2. 用 LLM 生成更多非结构化风格的训练题（给 benchmark 题"去结构化"改写）
3. 在特征提取中增加"结构化程度"字段——对低结构化问题，调度器应倾向于更灵活的策略（如 S08 试错法、SPECIAL_GATHER_INFO）

---

## 8. 风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| LLM 执行器不遵循策略 prompt | 高 | 高 | 设计约束性更强的 prompt 格式；策略一致性奖励引导执行器遵循策略；必要时用 few-shot 示例 |
| 调度器 reward hacking | 中 | 高 | 方案 A 不存在此问题（输出是离散选择）；方案 B 用严格输出格式约束 |
| 训练任务的策略标注质量不足 | 高 | 高 | 半自动标注 + 人工抽样审核 + 交叉验证；对标注置信度低的任务降低其在训练中的权重 |
| 特征提取器不稳定 | 中 | 中 | 3 次提取取中位数/众数；对高方差特征降权 |
| 策略数量太多导致探索不足 | 低 | 中 | 15-20 策略的动作空间在 RL 中属于较小的，entropy bonus 和课程学习应能覆盖 |
| 跨领域泛化性不足 | 中 | 中 | 增加训练领域数量；在特征中强化领域无关的结构特征 |
| 评估中 LLM 裁判的一致性差 | 中 | 中 | 对 LLM 裁判的评分做校准实验（与人类评分对比）；对一致性差的维度增加人类评估比例 |
| 训练成本超预算 | 中 | 中 | 优先用方案 A（成本低 10 倍）；用课程学习减少困难任务的无效探索 |
| 与阶段零知识库的格式不兼容 | 低 | 高 | 本阶段启动前先跑一遍阶段零的 `export_for_agent.py`，确认输出格式可被本阶段代码正确解析 |
| 阶段零知识库质量不足 | 低 | 高 | 在训练开始前用 Baseline 3（规则匹配）做冒烟测试——如果规则匹配的准确率 < 30%，说明知识库的适用条件描述需要改进 |
| 策略组合（COMP_*）在训练中被忽略 | 中 | 中 | 组合策略的奖励和一致性评估已独立设计（1.2.2 节）；监控组合策略的选择频率，如果 < 5% 则增加定向探索 |
| 调度器置信度不校准 | 高 | 高 | 置信度校准奖励权重 0.15（1.4 节）；训练后输出校准曲线（预测置信度 vs 实际成功率） |
| 训练-部署领域鸿沟 | 中 | 高 | 7.5 节补充实验检测鸿沟程度；增加非结构化问题在训练集中的比例（最终目标 20%） |
| SPECIAL_GATHER_INFO 被过度使用 | 中 | 中 | 信息收集的奖励折扣 0.7 限制滥用；监控使用频率，如果 > 15% 则增加折扣 |

---

## 9. 增量开发计划

本阶段自身也遵循"增量替换"原则。按以下顺序开发，每一步都以前一步的已验证组件为基础。

### Step 1：最小可运行系统（第 1-3 周）

**目标：** 搭建完整的管线（pipeline），端到端跑通但不追求性能。

- 实现 `TaskEnvironment` 基类和一个最简单的任务环境（如 20 道手写逻辑推理题）
- 实现特征提取器（直接用 LLM 调用）
- 实现方案 A 调度器（随机初始化）
- 实现 LLM 执行器（调用 API）
- 实现结果评估器（简单的答案匹配）
- 实现经验日志写入器
- **验证：** 整个管线能跑通，经验日志格式符合阶段零 schema

### Step 2：监督预训练（第 4-5 周）

**目标：** 用阶段零的标注数据预训练调度器。

- 加载阶段零的 `benchmark_problems.json` 和知识库导出
- 实现监督预训练流程
- **验证：** 预训练后 top-1 准确率 ≥ 50%（在标注数据的验证集上）

### Step 3：扩充任务环境（第 6-9 周）

**目标：** 构建 B 类任务集的主体。

- 实现代码调试、数学证明、逻辑推理三个任务环境
- 实现半自动策略标注流程
- 生成 600-800 道标注任务
- **验证：** 各任务环境的自动评估与人工抽查一致率 ≥ 85%

### Step 4：RL 训练（第 10-14 周）

**目标：** 用 PPO 训练调度器。

- 实现 PPO 训练流程（基于 Stable-Baselines3 或 TRL）
- 实现课程学习调度
- 实现奖励函数
- 运行训练，调参
- **验证：** 训练曲线收敛，验证集上 top-1 准确率 ≥ 65%

### Step 5：评估与消融（第 15-18 周）

**目标：** 全面评估，运行所有 baseline 和消融实验。

- 实现 5 个 baseline
- 扩充剩余任务环境（科学实验、工程规划、商业决策）
- 构建 C 类对抗任务集
- 运行所有实验
- 人类专家评估
- **验证：** 核心假设 H1-H3 的统计检验结果

### Step 6：分析与论文（第 19-22 周）

**目标：** 深度分析和论文写作。

- 策略-条件映射可视化
- 导出给阶段二的数据
- 论文写作
- **验证：** 论文完成初稿

---

## 10. 完成标准（Definition of Done）

阶段一在以下所有条件同时满足时视为完成：

1. 调度器在多领域测试集上的 top-1 策略选择准确率 ≥ 65%
2. 配备调度器的 LLM agent 在任务完成率上显著优于 Baseline 1（无策略）至少 15 个百分点（p < 0.05）
3. 调度器优于 Baseline 4（LLM 自选策略），证明 RL 训练的价值（p < 0.05）
4. 用人类哲学先验初始化的调度器在早期训练阶段（前 5000 episodes）显著优于 EvolveR 风格的从零蒸馏（p < 0.05）
5. 跨领域泛化实验中，平均准确率 ≥ 55%
6. 人类专家评估中，策略选择的合理率 ≥ 75%
7. 经验日志格式严格符合阶段零 2.4 节的 schema，且 attribution 字段被正确填充
8. 知识库读取接口与阶段零的 `export_for_agent.py` 输出格式完全兼容
9. 导出给阶段二的策略偏好数据格式已定义，且包含至少一个"调度器发现的新模式"示例
10. 所有代码有单元测试，核心模块（调度器、特征提取器、评估器）的测试覆盖率 ≥ 80%
11. 完成一篇论文初稿，包含核心实验结果（H1-H3）和策略-条件映射分析