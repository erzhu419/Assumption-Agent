# 阶段一：策略调度器——完整开发文档 (v1)

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
```

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
    primary_strategy: str               # 主策略 ID (如 "S01")
    confidence: float                   # 选择置信度 (0.0-1.0)
    backup_strategy: Optional[str]      # 备选策略 ID（当主策略失败时使用）
    execution_hint: Optional[str]       # 对执行器的额外提示（如"关注耦合问题"）
```

动作空间的大小 = 知识库中的策略数量（阶段零目标为 15-20 条）。这是一个离散动作空间，但通过 `confidence` 和 `backup_strategy` 引入了策略组合的可能性。

**为什么不用连续动作空间（如策略的加权混合）：** 可解释性。离散选择允许我们精确追踪"调度器为什么选了这条策略"，这是阶段二经验反馈的前提。如果调度器输出的是策略的连续混合权重，归因分析会困难得多。

#### 1.2.3 输出与执行流程

当调度器选择了策略 `S_k` 后，系统执行以下流程：

1. 从知识库中检索 `S_k` 的 `strategy_prompts`（阶段零 5.3 节定义的导出格式）
2. 将策略 prompt 与原始问题描述拼接，构造执行器的输入
3. 调用 LLM 执行器生成执行轨迹
4. 结果评估器评估执行结果，产生奖励信号
5. 如果主策略失败且存在 `backup_strategy`，跳转到步骤 1 使用备选策略
6. 如果备选策略也失败，查询知识库的 `fallback_graph`（阶段零 5.3 节），选择 fallback 策略
7. 将完整的执行记录写入经验日志（格式遵循阶段零 2.4 节的 schema）

**最大重试次数：** 3 次（1 次主策略 + 1 次备选 + 1 次 fallback）。超过 3 次视为该问题上策略选择失败。

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
| 任务完成度 | 0.50 | 任务是否被正确解决（二元/连续） |
| 策略一致性 | 0.20 | 执行轨迹是否真正遵循了所选策略的步骤 |
| 效率 | 0.15 | 相对于基准步数的归一化步数 |
| 部分进展 | 0.15 | 即使未完全解决，问题规模/复杂度是否降低了 |

**奖励函数：**

```python
def compute_reward(outcome: ExecutionOutcome) -> float:
    r_completion = 1.0 if outcome.success else (
        0.5 if outcome.partial_success else 0.0
    )
    r_consistency = compute_strategy_consistency(
        outcome.trajectory, outcome.selected_strategy
    )
    r_efficiency = max(0, 1.0 - outcome.total_steps / outcome.baseline_steps)
    r_progress = compute_progress_score(outcome)
    
    reward = (0.50 * r_completion +
              0.20 * r_consistency +
              0.15 * r_efficiency +
              0.15 * r_progress)
    
    # 策略选择失败的惩罚：如果用了 fallback 才成功，主策略选择扣分
    if outcome.used_fallback:
        reward *= 0.7
    
    return reward
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

**算法选择：** PPO（Proximal Policy Optimization）

选择 PPO 的理由：
1. 动作空间是离散的且较小（15-20 个策略），PPO 在这种设置下稳定且高效
2. PPO 是 RLHF/RLVR 中的标准算法，与 LLM 生态系统兼容性好
3. 相比 DPO 等离线方法，PPO 支持在线探索，这对发现知识库中未被人类标注充分覆盖的策略组合至关重要

**PPO 训练配置：**

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

**探索机制：** 

初期（前 10000 episodes）使用较高的 entropy bonus（`entropy_coef=0.05`），鼓励调度器尝试知识库中所有策略，包括那些人类标注较少推荐的策略。这可能发现某些策略在特定条件下的意外有效性。

后期（10000+ episodes）逐步降低 entropy bonus，让调度器收敛到稳定的策略选择策略。

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
| 每 episode 推理成本 | 特征提取 1 次 LLM 调用 + 执行器 1 次 LLM 调用 | 调度器 1 次 LLM 调用 + 执行器 1 次 LLM 调用 |
| 50K episodes 总成本 | ~$500-1000 (API) 或 ~48 GPU-hours (本地) | ~$3000-5000 (API) 或 ~200 GPU-hours (本地) |
| 训练时间 | 2-4 天 (单 A100) | 1-2 周 (4x A100) |

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