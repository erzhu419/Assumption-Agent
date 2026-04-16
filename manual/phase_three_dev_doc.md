# 阶段三：形式化对齐层——完整开发文档 (v1)

## 0. 文档概述

### 0.1 本阶段在整体架构中的位置

阶段零给了系统自然语言描述的策略知识库。阶段一让系统学会选择策略。阶段二让知识库从经验中演化。到此为止，一切都在自然语言层面运作——策略的适用条件是自然语言，策略之间的关系是人工标注的语义标签（prerequisite, complementary, alternative, subsumption）。

**本阶段的核心问题：两条看起来不同的策略，是否其实是同一个抽象原则的不同实例？**

自然语言层面无法可靠地回答这个问题。"控制变量法"和"消融实验"表述完全不同，但它们在结构上是同构的——都是固定其他因素、改变一个因素、观察效果。要检测这种深层同构，需要把策略从自然语言提升到数学对象，然后在数学空间中计算距离。

这对应了 Gemini.md 和 Claude.md 讨论中确定的架构：**范畴论提供结构骨架（两个策略在结构上是否等价），信息几何提供度量（两个策略在定量上有多接近）。** 但正如讨论中澄清的，这两个工具不是用来"统一所有假设"的——它们的作用范围被限定在"方法论策略之间的同构检测"这个可操作的问题上。

**本阶段的新模块只有一个：形式化层。** 知识库、调度器、反馈回路全部沿用阶段一二的已验证组件。

### 0.2 本阶段目标

1. 将知识库中的每条策略形式化为数学对象（FinStoch 范畴中的 Markov 核）
2. 在策略的数学表示之间定义并计算距离度量
3. 用形式化距离自动检测策略之间的结构同构
4. 验证形式化度量与实际执行表现之间的一致性

### 0.3 交付物

1. 策略形式化模块（`formalization/`）：将每条策略转化为 Markov 核表示
2. 距离度量模块（`metrics/`）：实现多种策略间距离的计算
3. 同构检测模块（`isomorphism/`）：自动检测并报告策略间的结构等价关系
4. 形式化后的策略表示数据库（`formal_kb/`）
5. 形式化度量与实际性能一致性分析报告
6. 一篇可投稿论文，核心贡献：首次用范畴论+信息几何对方法论策略进行形式化度量

### 0.4 时间预算

总计 6-8 个月。形式化框架设计与数学推导：4-6 周。策略形式化实现：6-8 周。距离度量与同构检测：4-6 周。实验验证与分析：4-6 周。论文写作：4 周。

### 0.5 两种技术路线概述

本阶段的核心挑战是**将自然语言描述的策略转化为数学对象**。这一步有两种实现路线：

| 维度 | 路线 A：Prompting + 数值计算 | 路线 B：Fine-tune 专用模型 |
|------|---------------------------|--------------------------|
| 形式化方法 | LLM prompting 辅助建模 + 人工审核 + 数值拟合 | 训练专用的 NL→数学结构 转换模型 |
| 距离计算 | 纯数学计算（numpy/scipy），不涉及 ML | 同路线 A（距离计算本身是数学问题） |
| 同构检测 | 数学判据 + LLM prompting 辅助解释 | 数学判据 + 训练同构分类器 |
| 成本 | LLM API ~$50-100 + 大量人工审核时间 | LoRA 训练 ~$100-200 + 少量人工审核 |
| 质量瓶颈 | LLM 在数学建模上的零样本能力 | 训练数据的获取（需要人工构造） |
| 适用场景 | 策略数量少（15-20 条），可以逐条人工把关 | 策略数量多（50+），或需要频繁重新形式化 |

**默认选择路线 A**（策略数量为 15-20 条，逐条形式化的人工成本可控）。路线 B 作为扩展方案，在策略数量显著增长或需要自动化形式化时启用。

以下正文以路线 A 为主线展开。路线 B 的差异点在附录 A 中详述。

---

## 1. 数学基础

### 1.1 核心数学对象：Markov 核

**为什么选 Markov 核作为策略的数学表示？**

一条方法论策略（如"控制变量法"）本质上描述的是：给定一个问题状态，应该采取什么行动。如果问题状态是离散的，这就是一个从"状态集"到"行动分布"的映射——即一个 Markov 核（stochastic matrix）。

在 Fritz 等人的 Markov 范畴框架中，Markov 核是态射（morphism），策略之间的关系（同构、包含、互补）对应态射之间的关系。这让我们能用范畴论的工具来分析策略结构。

**形式化定义：**

设 $\mathcal{X}$ 为问题特征空间（离散化后为有限集），$\mathcal{A}$ 为行动空间（策略的操作步骤集合）。一条策略 $S$ 被形式化为 Markov 核：

$$K_S: \mathcal{X} \to \Delta(\mathcal{A})$$

其中 $\Delta(\mathcal{A})$ 是 $\mathcal{A}$ 上的概率分布集合。在离散情况下，$K_S$ 就是一个 $|\mathcal{X}| \times |\mathcal{A}|$ 的随机矩阵（每行和为 1）。

**直观理解：** $K_S(x, a)$ 表示"当问题具有特征 $x$ 时，策略 $S$ 推荐行动 $a$ 的概率"。这里的"概率"不一定是真正的随机性——它也可以理解为"在不同执行实例中，策略在这种特征下实际选择该行动的频率"。

### 1.2 问题特征空间的离散化

要构造 Markov 核，首先需要将阶段一定义的连续问题特征空间 $\text{ProblemFeatures}$ 离散化为有限状态集 $\mathcal{X}$。

**离散化方案：**

选取对策略选择影响最大的特征维度（基于阶段一消融实验的结果），将每个维度分为 2-3 个区间：

```python
FEATURE_DISCRETIZATION = {
    "coupling_estimate": {
        "low": (0.0, 0.3),
        "medium": (0.3, 0.7),
        "high": (0.7, 1.0)
    },
    "decomposability": {
        "low": (0.0, 0.4),
        "high": (0.4, 1.0)
    },
    "has_baseline": {
        "yes": True,
        "no": False
    },
    "information_completeness": {
        "low": (0.0, 0.4),
        "medium": (0.4, 0.7),
        "high": (0.7, 1.0)
    },
    "component_count": {
        "few": (1, 4),
        "moderate": (4, 8),
        "many": (8, float('inf'))
    }
}

# 状态空间大小 = 3 × 2 × 2 × 3 × 3 = 108 个离散状态
```

每个离散状态 $x \in \mathcal{X}$ 是一个特征组合，如 `(coupling=low, decomposability=high, has_baseline=yes, info=medium, components=few)`。

**为什么不用所有特征：** 离散化的状态空间会随维度指数增长。选取 5 个最重要的维度（108 个状态）在可计算性和表达力之间取得平衡。其余特征在后续版本中可以通过扩展维度或使用连续表示来引入。

### 1.3 行动空间的定义

行动空间 $\mathcal{A}$ 不是知识库中具体的操作步骤（那太细了），而是策略执行中的**抽象行动类别**：

```python
ACTION_SPACE = [
    "decompose",           # 分解问题为子问题
    "isolate_variable",    # 隔离单一变量
    "simplify",            # 简化/降维
    "analogize",           # 类比已知问题
    "negate",              # 反向推理/反证
    "test_boundary",       # 测试边界条件
    "enumerate",           # 枚举/穷举
    "estimate_update",     # 估计后更新（贝叶斯式）
    "build_incrementally", # 增量构建
    "compare_cases",       # 对比案例（求同/求异）
    "abstract",            # 抽象化/泛化
    "relax_constraint",    # 松弛约束
    "seek_falsification",  # 寻找反例
    "evaluate_select",     # 评估并选择（满意化/最优化）
    "switch_perspective",  # 切换视角
    "no_action"            # 无特定行动（策略不适用时）
]
# |A| = 16
```

**关键设计：行动空间跨策略共享。** 所有策略的 Markov 核共享同一个行动空间 $\mathcal{A}$，这使得不同策略之间的直接比较成为可能。一条策略的"个性"体现在它对同一问题状态推荐的行动分布不同。

### 1.4 距离度量

在 Markov 核空间上定义以下距离度量：

#### 1.4.1 谱距离（Spectral Distance）

比较两个 Markov 核的特征值集合。

$$d_{\text{spectral}}(K_A, K_B) = d_{\text{Hausdorff}}(\Lambda(K_A), \Lambda(K_B))$$

其中 $\Lambda(K)$ 是 $K$ 的特征值集合，$d_{\text{Hausdorff}}$ 是复平面上的 Hausdorff 距离。

**直觉：** 如果两个策略的 Markov 核具有相似的特征值分布，它们的"动力学行为"相似——它们让问题状态以相似的速率收敛。

#### 1.4.2 Fisher 信息距离

将每个 Markov 核看作统计流形上的一个点，用 Fisher 信息度量计算测地线距离。

对于两个离散分布 $p = K_A(x, \cdot)$ 和 $q = K_B(x, \cdot)$（固定状态 $x$），Fisher-Rao 距离为：

$$d_{\text{FR}}(p, q) = 2 \arccos\left(\sum_a \sqrt{p(a) \cdot q(a)}\right)$$

在所有状态上取平均（加权按状态的经验频率 $\pi(x)$）：

$$d_{\text{Fisher}}(K_A, K_B) = \sum_{x \in \mathcal{X}} \pi(x) \cdot d_{\text{FR}}(K_A(x, \cdot), K_B(x, \cdot))$$

**直觉：** Fisher 信息距离是"最佳统计区分度"——它衡量的是"如果用策略 A 的行动分布代替策略 B 的，会损失多少决策信息"。

#### 1.4.3 Blackwell 序

Fritz 等人在 Markov 范畴框架中证明的 Blackwell-Sherman-Stein 定理给出了统计实验之间的偏序关系。

策略 $A$ Blackwell-支配 策略 $B$（记作 $A \succeq_B B$），当且仅当存在 Markov 核 $M$ 使得 $K_B = M \circ K_A$。直觉上，这意味着从策略 $A$ 的行动分布可以通过"加噪"得到策略 $B$ 的行动分布——$A$ 包含了 $B$ 的所有信息，外加额外的精细区分。

**计算方法：** 在 FinStoch 中，检查 Blackwell 支配关系等价于检查一个线性规划的可行性：

$$\exists M \geq 0,\ M \mathbf{1} = \mathbf{1},\ K_B = M K_A$$

**用途：** Blackwell 序给出策略之间的"信息量"偏序。如果 $A \succeq_B B$，说明策略 $A$ 比 $B$ 更精细——$A$ 能区分更多的问题状态。这对应阶段零知识库中的 `subsumption` 关系（策略 $B$ 是策略 $A$ 的特殊情况）。

#### 1.4.4 Frobenius 范数距离（基准度量）

最简单的矩阵距离，作为复杂度量的基准对比：

$$d_{\text{Frob}}(K_A, K_B) = \|K_A - K_B\|_F = \sqrt{\sum_{x,a} (K_A(x,a) - K_B(x,a))^2}$$

### 1.5 同构的形式化定义

两条策略 $A$ 和 $B$ 在 Markov 范畴意义下是同构的，当且仅当存在可逆的 Markov 核 $\Phi: \mathcal{X} \to \mathcal{X}$（状态空间的置换）和 $\Psi: \mathcal{A} \to \mathcal{A}$（行动空间的置换），使得：

$$K_B(\Phi(x), \Psi(a)) = K_A(x, a) \quad \forall x, a$$

直觉上：$A$ 和 $B$ 在重新标记状态和行动后变成同一个 Markov 核。

**实际操作中的放松：** 严格同构的检测在计算上是困难的（涉及图同构问题）。我们使用以下近似判据：

- **弱同构（practical isomorphism）：** $d_{\text{spectral}}(K_A, K_B) < \epsilon_1$ 且 $d_{\text{Fisher}}(K_A, K_B) < \epsilon_2$
- 阈值 $\epsilon_1, \epsilon_2$ 通过已知同构对（如"控制变量法"和"消融实验"）的距离来校准

---

## 2. 策略形式化流程

### 2.1 从自然语言到 Markov 核

将知识库中的每条策略转化为 Markov 核 $K_S$ 需要估计每个条目 $K_S(x, a)$——"在问题状态 $x$ 下，策略 $S$ 推荐行动 $a$ 的概率"。估计来源有两个：

**来源 1：先验估计（从知识库的适用条件推导）**

知识库中每条策略的 `applicability_conditions` 和 `operational_steps` 可以被转化为 Markov 核的先验结构。

```python
class StrategyFormalizer:
    """
    将自然语言策略描述转化为 Markov 核。
    使用 LLM prompting + 规则函数，不涉及模型训练。
    """
    
    def formalize_from_prior(
        self,
        strategy: Dict,
        feature_space: List[DiscreteState],
        action_space: List[str]
    ) -> np.ndarray:
        """
        从知识库描述构造先验 Markov 核。
        返回 |X| × |A| 的随机矩阵。
        """
        K = np.zeros((len(feature_space), len(action_space)))
        
        for i, state in enumerate(feature_space):
            # 用 LLM 判断：在这个问题状态下，
            # 该策略最推荐的行动是什么
            action_dist = self._llm_estimate_action_distribution(
                strategy, state, action_space
            )
            K[i, :] = action_dist
        
        return K
```

LLM 估计的 prompt：

```python
PRIOR_ESTIMATION_PROMPT = """
你是一个方法论分析专家。

## 策略信息
名称: {strategy_name}
描述: {strategy_description}
操作步骤: {operational_steps}

## 当前问题状态
- 组件间耦合度: {coupling}
- 可分解性: {decomposability}
- 是否有基准: {has_baseline}
- 信息完整度: {information_completeness}
- 组件数量: {component_count}

## 行动列表
{action_list_with_descriptions}

在这种问题状态下，如果严格按照策略 "{strategy_name}" 的指导来行动，
你最可能执行的行动是哪些？

请为每个行动分配一个 0-10 的相关性分数（不需要加总为 10）。
分数越高表示该策略在此状态下越可能推荐该行动。

输出 JSON:
{{
    "action_scores": {{
        "decompose": 分数,
        "isolate_variable": 分数,
        ...
    }},
    "reasoning": "简要说明为什么这些行动得分最高"
}}
"""
```

LLM 输出的原始分数经过 softmax 归一化后成为概率分布，构成 Markov 核的一行。

**来源 2：经验估计（从执行轨迹统计）**

阶段一二积累的执行经验提供了策略在实际使用中的行动分布数据。

```python
def formalize_from_experience(
    self,
    strategy_id: str,
    experience_log: List[ExecutionRecord],
    feature_space: List[DiscreteState],
    action_space: List[str]
) -> np.ndarray:
    """
    从执行经验统计 Markov 核。
    """
    # 筛选使用该策略的执行记录
    records = [r for r in experience_log
               if r["strategy_selection"]["selected_strategy"] == strategy_id]
    
    K = np.zeros((len(feature_space), len(action_space)))
    counts = np.zeros((len(feature_space), len(action_space)))
    
    for record in records:
        # 将问题特征离散化到状态空间
        state_idx = discretize(record["task"]["complexity_features"])
        
        # 从执行轨迹中提取实际采取的行动
        actions = self._extract_actions_from_trajectory(
            record["execution_trajectory"],
            action_space
        )
        for action_idx in actions:
            counts[state_idx, action_idx] += 1
    
    # 归一化 + Laplace 平滑
    for i in range(len(feature_space)):
        row_sum = counts[i, :].sum()
        if row_sum > 0:
            K[i, :] = (counts[i, :] + 0.1) / (row_sum + 0.1 * len(action_space))
        else:
            # 无数据的状态用均匀分布
            K[i, :] = 1.0 / len(action_space)
    
    return K
```

**从执行轨迹中提取行动：** 轨迹是自然语言描述的步骤序列。需要将每一步映射到 `ACTION_SPACE` 中的行动类别。这通过 LLM prompting 实现：

```python
ACTION_EXTRACTION_PROMPT = """
以下是一段问题解决轨迹中的一个步骤：

步骤描述: "{step_description}"

这个步骤最接近以下哪个抽象行动类别？
{action_list_with_descriptions}

选择最匹配的 1-2 个行动类别。输出 JSON:
{{"actions": ["action_1", "action_2"], "reasoning": "..."}}
"""
```

### 2.2 先验与经验的融合

最终的 Markov 核是先验估计和经验估计的加权融合：

$$K_S = \alpha \cdot K_S^{\text{prior}} + (1 - \alpha) \cdot K_S^{\text{exp}}$$

其中权重 $\alpha$ 取决于经验数据的充分性：

```python
def compute_fusion_weight(
    strategy_id: str,
    experience_count: int,
    min_samples_per_state: int = 5,
    total_states: int = 108
) -> float:
    """
    经验数据越充分，经验估计的权重越高。
    """
    # 平均每个状态的样本数
    avg_samples = experience_count / total_states
    
    if avg_samples < 1:
        return 0.9   # 几乎纯先验
    elif avg_samples < min_samples_per_state:
        # 线性过渡
        return 0.9 - 0.7 * (avg_samples / min_samples_per_state)
    else:
        return 0.2   # 主要依赖经验
```

### 2.3 人工审核与校准

每条策略的 Markov 核形式化后，需要经过人工审核。审核不是检查每个数值（108 × 16 = 1728 个条目），而是检查关键的定性模式：

**审核项 1：主导行动一致性**
- 在每个状态下，Markov 核的最高概率行动是否与直觉一致？
- 例如："控制变量法"在 `(coupling=low, has_baseline=yes)` 状态下，`isolate_variable` 应该是最高概率行动

**审核项 2：零概率行动合理性**
- 哪些行动的概率接近零？这些零概率是否合理？
- 例如："反证法"在任何状态下都不应该推荐 `build_incrementally`

**审核项 3：已知关系验证**
- 阶段零标注的 `prerequisite`/`complementary`/`alternative`/`subsumption` 关系是否在形式化度量中得到体现？
- 例如：标注为 `alternative` 的两条策略，其 Markov 核的 Fisher 距离应该较大（它们推荐不同的行动），但谱距离可能较小（它们在相似的问题状态下被激活）

---

## 3. 同构检测

### 3.1 检测流程

```
所有策略对 (S_i, S_j), i < j
        │
        ▼
┌───────────────────┐
│  第一层：快速筛选    │  Frobenius 距离 < 阈值?
│  (Frobenius 范数)  │  O(K²) 计算，毫秒级
└────────┬──────────┘
         │ 候选同构对
         ▼
┌───────────────────┐
│  第二层：谱分析      │  谱距离 < 阈值?
│  (特征值比较)       │  需要特征值分解
└────────┬──────────┘
         │ 高度候选
         ▼
┌───────────────────┐
│  第三层：Fisher 距离 │  Fisher-Rao 距离 < 阈值?
│  (信息几何度量)      │  逐状态计算
└────────┬──────────┘
         │ 强候选
         ▼
┌───────────────────┐
│  第四层：Blackwell 序│  A ≥_B B 或 B ≥_B A?
│  (线性规划)         │  精确但计算量最大
└────────┬──────────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
  同构  包含  独立
```

### 3.2 实现

```python
class IsomorphismDetector:
    """
    检测策略 Markov 核之间的结构等价关系。
    纯数学计算，不涉及 ML 模型。
    """
    
    def __init__(self, formal_kb: Dict[str, np.ndarray]):
        self.formal_kb = formal_kb  # {strategy_id: Markov_kernel}
        self.thresholds = {
            "frobenius_candidate": 0.5,
            "spectral_candidate": 0.3,
            "fisher_strong": 0.2,
            "blackwell_feasibility_tol": 1e-6
        }
    
    def detect_all_isomorphisms(self) -> List[IsomorphismReport]:
        strategies = list(self.formal_kb.keys())
        reports = []
        
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                sid_a, sid_b = strategies[i], strategies[j]
                K_a = self.formal_kb[sid_a]
                K_b = self.formal_kb[sid_b]
                
                report = self._analyze_pair(sid_a, K_a, sid_b, K_b)
                if report.relationship != "independent":
                    reports.append(report)
        
        return reports
    
    def _analyze_pair(
        self, sid_a, K_a, sid_b, K_b
    ) -> IsomorphismReport:
        
        # 层 1：Frobenius 距离
        d_frob = np.linalg.norm(K_a - K_b, 'fro')
        if d_frob > self.thresholds["frobenius_candidate"]:
            return IsomorphismReport(
                strategy_a=sid_a, strategy_b=sid_b,
                relationship="independent",
                distances={"frobenius": d_frob}
            )
        
        # 层 2：谱距离
        eigs_a = np.sort(np.abs(np.linalg.eigvals(K_a)))
        eigs_b = np.sort(np.abs(np.linalg.eigvals(K_b)))
        d_spectral = self._hausdorff_distance(eigs_a, eigs_b)
        
        # 层 3：Fisher 距离
        d_fisher = self._fisher_rao_distance(K_a, K_b)
        
        # 层 4：Blackwell 序
        a_dominates_b = self._check_blackwell(K_a, K_b)
        b_dominates_a = self._check_blackwell(K_b, K_a)
        
        # 判定关系
        if d_spectral < self.thresholds["spectral_candidate"] and \
           d_fisher < self.thresholds["fisher_strong"]:
            relationship = "isomorphic"
        elif a_dominates_b and not b_dominates_a:
            relationship = "a_subsumes_b"
        elif b_dominates_a and not a_dominates_b:
            relationship = "b_subsumes_a"
        elif a_dominates_b and b_dominates_a:
            relationship = "isomorphic"  # 互相支配 = 等价
        elif d_fisher < self.thresholds["fisher_strong"] * 2:
            relationship = "similar"
        else:
            relationship = "independent"
        
        return IsomorphismReport(
            strategy_a=sid_a, strategy_b=sid_b,
            relationship=relationship,
            distances={
                "frobenius": d_frob,
                "spectral": d_spectral,
                "fisher": d_fisher
            },
            blackwell={"a_dom_b": a_dominates_b, "b_dom_a": b_dominates_a}
        )
    
    def _fisher_rao_distance(
        self, K_a: np.ndarray, K_b: np.ndarray,
        state_weights: np.ndarray = None
    ) -> float:
        """
        计算两个 Markov 核的加权 Fisher-Rao 距离。
        """
        n_states = K_a.shape[0]
        if state_weights is None:
            state_weights = np.ones(n_states) / n_states
        
        total = 0.0
        for x in range(n_states):
            p = K_a[x, :] + 1e-10  # 避免 log(0)
            q = K_b[x, :] + 1e-10
            # Bhattacharyya 系数
            bc = np.sum(np.sqrt(p * q))
            bc = np.clip(bc, -1.0, 1.0)
            # Fisher-Rao 距离
            d_fr = 2.0 * np.arccos(bc)
            total += state_weights[x] * d_fr
        
        return total
    
    def _check_blackwell(
        self, K_a: np.ndarray, K_b: np.ndarray
    ) -> bool:
        """
        检查 K_a 是否 Blackwell-支配 K_b。
        即是否存在 Markov 核 M 使得 K_b = K_a @ M。
        通过线性规划求解。
        """
        from scipy.optimize import linprog
        
        n_states, n_actions = K_a.shape
        # M 是 n_actions × n_actions 的随机矩阵
        # 变量: M 的所有 n_actions² 个条目
        n_vars = n_actions * n_actions
        
        # 约束 1: K_b[x, b] = sum_a K_a[x, a] * M[a, b] 对所有 x, b
        A_eq_list = []
        b_eq_list = []
        for x in range(n_states):
            for b in range(n_actions):
                row = np.zeros(n_vars)
                for a in range(n_actions):
                    row[a * n_actions + b] = K_a[x, a]
                A_eq_list.append(row)
                b_eq_list.append(K_b[x, b])
        
        # 约束 2: sum_b M[a, b] = 1 对所有 a（随机矩阵约束）
        for a in range(n_actions):
            row = np.zeros(n_vars)
            for b in range(n_actions):
                row[a * n_actions + b] = 1.0
            A_eq_list.append(row)
            b_eq_list.append(1.0)
        
        A_eq = np.array(A_eq_list)
        b_eq = np.array(b_eq_list)
        
        # 约束 3: M[a, b] >= 0（非负性由 bounds 保证）
        bounds = [(0, 1)] * n_vars
        
        # 目标函数无关紧要（只检查可行性）
        c = np.zeros(n_vars)
        
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                         method='highs')
        
        return result.success
```

### 3.3 LLM 辅助解释

当同构检测器发现一对策略具有数学上的同构/包含关系时，用 LLM 生成人类可理解的解释：

```python
ISOMORPHISM_EXPLANATION_PROMPT = """
数学分析发现以下两条策略之间存在结构关系：

策略 A: {strategy_a_name} - {strategy_a_description}
策略 B: {strategy_b_name} - {strategy_b_description}

数学关系: {relationship_type}
距离度量:
- Frobenius 距离: {d_frob:.4f}
- 谱距离: {d_spectral:.4f}
- Fisher-Rao 距离: {d_fisher:.4f}
- Blackwell 支配: {blackwell_result}

请用自然语言解释这种关系的含义：
1. 为什么这两条策略在数学上被判定为 {relationship_type}？
2. 它们共享了哪些核心逻辑？
3. 如果有差异，差异体现在哪些具体的问题状态下？
4. 这种关系对实际的策略调度有什么启示？

输出 JSON:
{{
    "explanation": "...",
    "shared_core_logic": "...",
    "key_differences": ["...", "..."],
    "dispatch_implications": "..."
}}
"""
```

---

## 4. 知识库增强

### 4.1 形式化结果写回知识库

同构检测的结果需要写回知识库，增强阶段零定义的 `relationships_to_other_strategies` 字段。

```python
def update_kb_with_formal_relationships(
    kb: KnowledgeBase,
    reports: List[IsomorphismReport]
):
    """
    将形式化分析的结果写回知识库。
    遵循阶段零的变更历史机制。
    """
    for report in reports:
        if report.relationship == "independent":
            continue
        
        # 更新策略 A 的关系列表
        strategy_a = kb.load_strategy(report.strategy_a)
        
        new_relationship = {
            "related_strategy": report.strategy_b,
            "relationship_type": _map_to_kb_type(report.relationship),
            "description": report.explanation["explanation"],
            "formal_evidence": {
                "distances": report.distances,
                "blackwell": report.blackwell,
                "detection_method": "markov_category_analysis"
            }
        }
        
        # 检查是否已有该关系（避免重复）
        existing = [r for r in strategy_a["relationships_to_other_strategies"]
                    if r["related_strategy"] == report.strategy_b]
        
        if existing:
            # 更新已有关系，添加形式化证据
            existing[0]["formal_evidence"] = new_relationship["formal_evidence"]
            if existing[0]["relationship_type"] != new_relationship["relationship_type"]:
                # 形式化分析与人工标注不一致——标记冲突
                existing[0]["formal_conflict"] = {
                    "human_labeled": existing[0]["relationship_type"],
                    "formal_detected": new_relationship["relationship_type"],
                    "needs_review": True
                }
        else:
            # 新增关系
            strategy_a["relationships_to_other_strategies"].append(
                new_relationship
            )
        
        kb.save_strategy(strategy_a)
        write_change_history(
            report.strategy_a,
            change_type="relationship_added" if not existing else "relationship_modified",
            author="phase3_formal",
            changes=new_relationship,
            evidence_refs=[]
        )

def _map_to_kb_type(formal_type: str) -> str:
    """将形式化关系类型映射到阶段零的类型枚举"""
    mapping = {
        "isomorphic": "alternative",      # 同构 → 可互相替代
        "a_subsumes_b": "subsumption",    # A 包含 B
        "b_subsumes_a": "subsumption",
        "similar": "complementary"         # 相似但不同构 → 互补
    }
    return mapping.get(formal_type, "complementary")
```

### 4.2 调度器增强

形式化层为阶段一的调度器提供额外信号：

**增强 1：策略 fallback 的优化**

当主策略失败时，调度器目前从知识库的 `fallback_graph` 中选择替代策略。形式化距离可以优化这个选择——在距离最近的策略中选择 fallback，而不是依赖人工标注的 fallback 列表。

```python
def suggest_fallback(
    failed_strategy: str,
    formal_kb: Dict[str, np.ndarray],
    current_state: int
) -> List[str]:
    """
    基于 Fisher 距离在当前状态下推荐 fallback 策略。
    """
    K_failed = formal_kb[failed_strategy]
    
    distances = {}
    for sid, K in formal_kb.items():
        if sid == failed_strategy:
            continue
        # 只计算当前状态下的局部 Fisher 距离
        p = K_failed[current_state, :] + 1e-10
        q = K[current_state, :] + 1e-10
        bc = np.sum(np.sqrt(p * q))
        d = 2.0 * np.arccos(np.clip(bc, -1.0, 1.0))
        distances[sid] = d
    
    # 返回距离中等的策略（太近 = 同构没用，太远 = 完全不同）
    sorted_strategies = sorted(distances.items(), key=lambda x: x[1])
    # 跳过距离最近的（可能是同构策略），选距离适中的
    return [s for s, d in sorted_strategies
            if 0.1 < d < 1.0][:3]
```

**增强 2：新策略的自动归类**

当阶段二发现了新的行为模式（尚未被命名为正式策略）时，形式化层可以检查这个新模式是否与已有策略同构——如果是，就不需要添加新策略，只需扩展已有策略的适用条件。

---

## 5. 技术实现

### 5.1 项目文件结构

```
assumption_agent/
├── ...                              # 阶段一二的所有目录保持不变
├── formalization/                   # 阶段三新增目录
│   ├── core/
│   │   ├── markov_kernel.py         # Markov 核的构造与操作
│   │   ├── feature_discretizer.py   # 特征空间离散化
│   │   ├── action_extractor.py      # 从轨迹提取行动类别
│   │   └── kernel_fusion.py         # 先验与经验的融合
│   ├── metrics/
│   │   ├── frobenius.py             # Frobenius 范数距离
│   │   ├── spectral.py              # 谱距离
│   │   ├── fisher_rao.py            # Fisher-Rao 距离
│   │   ├── blackwell.py             # Blackwell 序（线性规划）
│   │   └── metric_suite.py          # 统一度量接口
│   ├── isomorphism/
│   │   ├── detector.py              # 同构检测器
│   │   ├── explainer.py             # LLM 辅助解释
│   │   └── thresholds.py            # 阈值校准
│   ├── kb_enhancement/
│   │   ├── relationship_updater.py  # 知识库关系更新
│   │   ├── fallback_optimizer.py    # Fallback 优化
│   │   └── new_strategy_classifier.py # 新策略归类
│   ├── formal_kb/                   # 形式化后的策略表示
│   │   ├── kernels/                 # 每条策略的 Markov 核 (.npy)
│   │   ├── metadata.json            # 形式化元数据
│   │   └── distance_matrix.json     # 预计算的距离矩阵
│   ├── prompts.py                   # 所有 LLM prompt 模板
│   └── config.py                    # 配置参数
├── scripts/
│   ├── ...
│   ├── formalize_all.py             # 形式化所有策略
│   ├── compute_distances.py         # 计算距离矩阵
│   ├── detect_isomorphisms.py       # 运行同构检测
│   └── visualize_strategy_space.py  # 策略空间可视化
└── tests/
    ├── ...
    ├── test_markov_kernel.py
    ├── test_metrics.py
    ├── test_blackwell.py
    └── test_isomorphism.py
```

### 5.2 依赖

| 组件 | 技术选择 | 用途 |
|------|---------|------|
| 矩阵计算 | numpy | Markov 核的构造与运算 |
| 特征值分解 | numpy.linalg | 谱距离计算 |
| 线性规划 | scipy.optimize.linprog | Blackwell 序检测 |
| 聚类/可视化 | scikit-learn, matplotlib | 策略空间可视化 |
| LLM 调用 | OpenAI API | 先验估计、行动提取、同构解释 |
| 嵌入模型 | sentence-transformers | 行动类别匹配 |

**不引入新的 ML 框架。** 所有计算用 numpy/scipy 完成，LLM 仅用 prompting。

### 5.3 计算成本

| 操作 | 频率 | 成本 |
|------|------|------|
| 先验估计（LLM） | 每条策略 108 个状态 × 1 次调用 | 20 条策略 × ~$0.10 = ~$2 |
| 行动提取（LLM） | 每条经验轨迹 ~10 步 | 取决于经验数量，~$5-20 |
| 距离矩阵计算 | 20 × 19 / 2 = 190 对 | 纯数值计算，几秒 |
| Blackwell 序检测 | 190 对 × LP 求解 | 每对 ~0.1s，总计 ~20s |
| 同构解释（LLM） | 仅对检测到的同构对 | ~5-10 对 × ~$0.05 = ~$0.5 |
| **总计一次性成本** | | **~$10-25** |

形式化是一次性操作（除非策略集发生变化）。日常运行中，只有"新策略归类"需要增量计算。

### 5.4 与其他阶段的接口

| 方向 | 数据 | 说明 |
|------|------|------|
| 阶段零 → 阶段三 | 知识库 JSON | 策略描述、适用条件、关系标注 |
| 阶段一 → 阶段三 | 执行经验日志 | 用于经验估计 Markov 核 |
| 阶段一 → 阶段三 | 调度器偏好数据 | 用于验证形式化度量与实际偏好的一致性 |
| 阶段二 → 阶段三 | 变更历史 | 识别频繁变更的条件，指导形式化优先级 |
| 阶段三 → 阶段零 | 形式化关系更新 | 写回 `relationships_to_other_strategies` |
| 阶段三 → 阶段一 | 优化后的 fallback 图 | 基于 Fisher 距离的 fallback 推荐 |
| 阶段三 → 阶段四 | 策略拓扑图 | 阶段四用于检测"已知策略空间的空白" |

---

## 6. 实验设计

### 6.1 核心实验：形式化度量与实际性能的一致性

**假设 H1：** 形式化距离小的策略对，在相同任务上的表现差异也小。

**实验设计：**
- 计算所有策略对的 Fisher 距离
- 对同一组任务，分别用每对策略执行，记录任务完成率
- 计算 Fisher 距离与完成率差异的 Spearman 相关系数
- **目标：** 相关系数 ≥ 0.5（正相关，即距离越近、表现越相似）

### 6.2 核心实验：自动检测已知同构

**假设 H2：** 系统能自动检测出人类已知的策略同构对。

**实验设计：**
- 构造 ground truth：请领域专家标注知识库中哪些策略对是"本质上相同的抽象原则"
- 运行同构检测器
- 计算检测结果与 ground truth 的 precision/recall
- **目标：** Precision ≥ 0.80, Recall ≥ 0.60

**预期的已知同构对示例：**
- "控制变量法" (S01) ↔ 部分场景下的"增量构建" (S15)
- "求同法" (S16) ↔ 部分场景下的"类比推理" (S03)
- "先特殊后一般" (S06) ↔ 部分场景下的"降维简化" (S09)

### 6.3 关键实验：发现未标注的结构关系

**假设 H3：** 形式化分析能发现阶段零人工标注中未记录的策略关系。

**实验设计：**
- 比较形式化检测到的关系集合与阶段零标注的关系集合
- 对形式化发现但人工未标注的关系，请专家评审是否合理
- **目标：** 至少发现 2 条被专家确认为合理的新关系

### 6.4 消融实验

| 消融变量 | 实验设计 | 目标 |
|---------|---------|------|
| 距离度量选择 | 分别用 Frobenius/谱距离/Fisher/Blackwell 做同构检测 | 哪种度量的检测效果最好 |
| 特征空间维度 | 分别用 3/4/5/6 个特征维度 | 维度对形式化质量的影响 |
| 先验 vs 经验 | 纯先验/纯经验/融合 的 Markov 核 | 哪种估计方式更可靠 |
| 行动空间粒度 | 8/12/16/20 个行动类别 | 粒度对区分度的影响 |

### 6.5 可视化分析

**策略空间图：** 用 t-SNE 或 UMAP 将 Markov 核投影到 2D，策略之间的空间距离反映 Fisher 距离。用颜色编码策略类别（阶段零的 CAT_A - CAT_F）。

**距离热力图：** 20 × 20 的策略对距离矩阵，直观展示哪些策略接近、哪些远离。

**Blackwell 格图：** 策略之间的偏序关系图（Hasse diagram），展示信息量的层级结构。

---

## 7. 风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| LLM 先验估计不准确 | 高 | 高 | 每条策略 3 次独立估计取中位数；人工审核关键的主导行动；随经验数据积累逐步降低先验权重 |
| 离散化导致信息损失 | 中 | 中 | 从 5 维开始，如果区分度不足逐步增加维度；同时保留连续版本作为辅助验证 |
| 行动类别映射不一致 | 高 | 高 | 对 LLM 的行动提取结果做人工抽样审核；对一致性低的行动类别合并或细化定义 |
| Blackwell 序检测的 LP 数值不稳定 | 中 | 中 | 使用 HiGHS 求解器（scipy 默认）；对边界情况（接近可行/不可行）增大 tolerance |
| 所有策略都被判定为独立（无同构） | 中 | 高 | 放宽阈值做灵敏度分析；检查是否是行动空间粒度太粗导致所有核都趋于均匀分布 |
| 形式化度量与实际性能不相关 | 中 | 高 | 先在合成数据（已知同构的人工策略）上验证度量的有效性；如果在真实数据上不相关，说明 Markov 核表示不够充分，需要更丰富的数学结构 |
| 计算可行性（BorelStoch 太复杂） | 低 | — | 本阶段只在 FinStoch 中工作（离散化后的有限状态）。连续版本留给后续研究 |

---

## 8. 增量开发计划

### Step 1：合成数据验证（第 1-3 周）

**目标：** 在已知 ground truth 的合成策略上验证整个形式化管线。

- 人工构造 5 条"玩具策略"——3 条互相同构（只是标签不同的同一个 Markov 核），2 条独立
- 实现 Markov 核构造、距离计算、同构检测
- **验证：** 在合成数据上，同构检测的 precision = recall = 100%

### Step 2：先验形式化（第 4-7 周）

**目标：** 用 LLM prompting 对知识库中所有策略做先验形式化。

- 实现特征空间离散化
- 实现 LLM 先验估计 prompt
- 对 20 条策略逐一形式化
- 人工审核每条策略的主导行动模式
- **验证：** 审核通过率 ≥ 80%（即 ≥ 16 条策略的形式化结果被审核者认为合理）

### Step 3：经验形式化（第 8-10 周）

**目标：** 从阶段一二的执行经验中统计 Markov 核。

- 实现行动提取 prompt
- 从经验日志中统计每条策略的经验 Markov 核
- 实现先验-经验融合
- **验证：** 融合后的 Markov 核与纯先验的差异分析（哪些条目变化最大？变化是否合理？）

### Step 4：距离与同构（第 11-14 周）

**目标：** 计算完整距离矩阵并运行同构检测。

- 实现所有距离度量（Frobenius、谱、Fisher、Blackwell）
- 校准同构检测阈值
- 运行同构检测
- LLM 生成同构解释
- **验证：** 至少检测到 2 对策略之间的非平凡关系

### Step 5：知识库增强（第 15-17 周）

**目标：** 将形式化结果写回知识库并增强调度器。

- 实现关系更新
- 实现基于 Fisher 距离的 fallback 优化
- 在阶段一的任务集上验证增强后的效果
- **验证：** Fallback 成功率提升 ≥ 5%

### Step 6：实验与论文（第 18-22 周）

**目标：** 运行所有实验，完成论文。

- H1-H3 实验
- 消融实验
- 可视化分析
- 论文写作

---

## 9. 完成标准（Definition of Done）

阶段三在以下所有条件同时满足时视为完成：

1. 知识库中至少 15 条策略被成功形式化为 Markov 核，并通过人工审核
2. 形式化距离与实际性能差异的 Spearman 相关系数 ≥ 0.4
3. 同构检测在已知同构对上的 precision ≥ 0.70
4. 至少发现 2 条被专家确认为合理的、阶段零未标注的新策略关系
5. 形式化结果已写回知识库的 `relationships_to_other_strategies` 字段
6. 基于 Fisher 距离的 fallback 优化在阶段一任务集上有可测量的改进
7. 所有距离度量的实现有单元测试，覆盖已知同构/已知独立的测试用例
8. 策略空间的可视化图谱（t-SNE/UMAP + 距离热力图 + Blackwell 格图）已生成
9. 完成一篇论文初稿

---

## 附录 A：Fine-Tune 路线 B 详述

### A.1 路线 B 的适用场景

当以下条件之一成立时，应考虑从路线 A 切换到路线 B：

1. **策略数量显著增长：** 知识库从 20 条扩展到 50+ 条，逐条 LLM prompting + 人工审核的成本不可接受
2. **LLM 先验估计质量持续不足：** 审核通过率低于 60%，且优化 prompt 和更换模型后仍无改善
3. **需要频繁重新形式化：** 阶段二的知识库更新导致策略内容频繁变动，每次都需要重新估计 Markov 核

### A.2 Fine-tune 目标：NL→Markov 核 转换模型

训练一个专用模型，输入策略的自然语言描述 + 问题状态描述，输出行动分布。

**训练数据构造：**

```python
# 来源 1：路线 A 中人工审核通过的先验估计
# 每条策略 × 108 个状态 = 2160 条训练数据（20 条策略时）
training_data_prior = [
    {
        "input": {
            "strategy_name": "控制变量法",
            "strategy_description": "...",
            "operational_steps": "...",
            "problem_state": {
                "coupling": "low",
                "decomposability": "high",
                "has_baseline": "yes",
                "info_completeness": "medium",
                "component_count": "few"
            }
        },
        "output": {
            "action_distribution": {
                "isolate_variable": 0.45,
                "test_boundary": 0.20,
                "compare_cases": 0.15,
                ...
            }
        },
        "source": "prior_human_reviewed"
    },
    ...
]

# 来源 2：经验估计中的高置信度样本
# 仅使用每个状态-行动对有 ≥10 次观测的数据点
training_data_experience = [...]
```

**模型架构与训练：**

```python
FORMALIZATION_FINETUNE_CONFIG = {
    "base_model": "Qwen2.5-3B-Instruct",
    "method": "LoRA",
    "lora_rank": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "v_proj"],
    
    "training": {
        "num_epochs": 5,
        "learning_rate": 3e-5,
        "batch_size": 8,
        "max_seq_length": 1024,      # 输入较短
        "validation_split": 0.15,
    },
    
    "data": {
        "min_samples": 1000,          # 低于此数量不启动 fine-tune
        "prior_weight": 0.6,          # 先验数据与经验数据的混合比例
        "augmentation": True,         # 通过打乱状态描述顺序做数据增强
    },
    
    "output_format": "json",          # 输出 action_distribution JSON
    "output_validation": "softmax_check",  # 验证输出分布和为 1
}
```

**与路线 A 的性能对比实验：**

| 指标 | 路线 A (Prompting) | 路线 B (Fine-tune) |
|------|-------------------|-------------------|
| 每条策略形式化时间 | ~10 分钟（含 LLM 调用等待） | ~2 秒（本地推理） |
| 人工审核成本 | 每条策略需审核 | 仅在训练阶段审核 |
| 形式化质量 | 取决于 LLM 零样本能力 | 取决于训练数据质量 |
| 增量更新成本 | 改 prompt 即时 | 需重训或增量训练 |
| 一次性成本 | ~$10-25 | ~$50-100（训练） + $10（数据准备） |

### A.3 Fine-tune 目标：同构分类器

在路线 A 中，同构检测是纯数学计算（距离阈值判断）。路线 B 可以训练一个二分类器，直接判断两条策略是否同构。

**训练数据：**
- 正例：人工标注的同构策略对 + 人工构造的合成同构对
- 负例：随机策略对（距离远的自然负例）

**输入特征：**
- 两条策略的 Markov 核拼接
- 预计算的四种距离度量值
- 两条策略的自然语言描述嵌入的差向量

```python
ISOMORPHISM_CLASSIFIER_CONFIG = {
    "architecture": "MLP",               # 简单全连接网络
    "input_features": [
        "kernel_concat",                 # 2 × |X| × |A| = 2 × 108 × 16 = 3456
        "distance_vector",               # 4 维（Frob, spectral, Fisher, Blackwell flag）
        "text_embedding_diff"            # 768 维
    ],
    "total_input_dim": 3456 + 4 + 768,   # = 4228
    "hidden_layers": [512, 128],
    "output": "binary (isomorphic / not)",
    
    "training": {
        "min_positive_samples": 50,       # 需要至少 50 个正例
        "negative_sampling_ratio": 3,     # 正负比 1:3
        "num_epochs": 20,
        "learning_rate": 1e-3,
    }
}
```

**数据不足的问题：** 15-20 条策略只能产生 190 对，其中同构对可能只有 5-10 对。训练数据严重不足。解决方案：

1. **合成数据增强：** 对已知同构对，通过对 Markov 核添加小噪声 + 随机行重排来生成合成正例
2. **跨数据集迁移：** 如果有其他方法论体系（如设计模式、编程范式），可以在那些体系上构造额外的训练数据

**风险评估：** 由于正例数据极其稀缺（~5-10 对真实同构），fine-tune 分类器很可能过拟合。在策略数量 < 50 的情况下，**不推荐使用本方案**。路线 A 的纯数学方法在小规模下更可靠。

### A.4 启用决策

```
策略数量 < 30?
    │
    是 → 使用路线 A（Prompting + 数值计算）
    否 ↓
        
路线 A 审核通过率 < 60%?
    │
    是 → 启用 A.2（NL→Markov 核 转换模型）
    否 → 保持路线 A，但用 A.2 做加速
    
策略数量 > 100 且需要频繁同构检测?
    │
    是 → 考虑 A.3（同构分类器），但需先积累训练数据
    否 → 不启用 A.3
```

### A.5 Fine-tune 方案对项目结构的影响

```
assumption_agent/
├── formalization/
│   ├── ...                          # 现有目录不变
│   └── finetuned_models/            # fine-tune 方案新增
│       ├── kernel_estimator/        # A.2: NL→Markov 核 模型
│       │   ├── prepare_data.py      # 从审核通过的先验估计构造训练数据
│       │   ├── train.py
│       │   ├── config.yaml
│       │   └── checkpoints/
│       └── iso_classifier/          # A.3: 同构分类器
│           ├── prepare_data.py      # 构造正负样本对
│           ├── synthetic_augment.py # 合成数据增强
│           ├── train.py
│           ├── config.yaml
│           └── checkpoints/
```