# 阶段零点五：世界模型——完整开发文档 (v1)

## 0. 文档概述

### 0.1 本阶段在整体架构中的位置

世界模型不是可选的附加模块——**它是整个系统能运转的前提**。

回到 Claude.md 中奇异博士的类比：他需要时间宝石来模拟每条路径的后果。本系统也需要一个地方来**模拟策略执行的结果**，而不是每次都在真实环境里试错。这个地方就是世界模型。

**没有世界模型会发生什么：** 阶段一的调度器选择了"控制变量法"，然后必须在真实环境里跑完整个任务才能知道选对了没有。如果任务是写一个复杂系统的代码，一次执行可能要几个小时、花费几美元的 API 调用。RL 需要成千上万次试错，在真实环境中根本跑不起来。

**有世界模型会发生什么：** 调度器选择了"控制变量法"，先在世界模型里快速模拟这条策略的执行轨迹——预测"如果我按这个策略走，大概会在哪一步遇到什么问题"。如果模拟结果看起来不好，立刻换一条策略重新模拟。只有当模拟结果足够好时，才在真实环境中执行。

**世界模型在系统中的角色：假设的廉价验证器。** 它让递归假设-验证循环的每次迭代从"小时级"降到"秒级"。

### 0.2 对各阶段的影响

| 阶段 | 无世界模型 | 有世界模型 |
|------|-----------|-----------|
| 阶段一 | 调度器在真实环境中训练，每次试错成本高 | 调度器在世界模型中训练（model-based RL），真实环境仅用于校准 |
| 阶段二 | 反馈仅来自真实执行，数据量受限 | 双来源反馈：真实执行（高质量昂贵）+ 世界模型模拟（低质量廉价） |
| 阶段三 | 无直接影响（形式化层作用在策略空间而非状态空间） | 无直接影响 |
| 阶段四 | 新假设必须在真实环境中验证，成本极高 | 新假设首先在世界模型中大量模拟测试，只有通过的才进真实环境 |

### 0.3 本阶段目标

构建或接入一个世界模型，使其能在给定"当前任务状态 + 将要执行的策略"的情况下，预测"执行后的结果"。世界模型不需要完美——它只需要在预测"这条路大概率走不通"这件事上足够可靠。

### 0.4 交付物

1. 世界模型核心模块（`world_model/`）
2. 模拟执行接口——与阶段一任务环境接口兼容
3. 世界模型的准确性评估报告
4. 与阶段一调度器训练的集成方案

### 0.5 时间预算

总计 2-4 个月。世界模型设计与实现：4-6 周。准确性评估与校准：3-4 周。与阶段一的集成测试：2-3 周。

---

## 1. 世界模型的设计

### 1.1 预测目标

世界模型要预测的不是物理世界的状态变化，而是**"在某个策略指导下，LLM 执行器在某类任务上大概会得到什么结果"**。

具体来说，世界模型的输入和输出：

```python
@dataclass
class WorldModelInput:
    # 任务描述和特征
    problem_features: ProblemFeatures      # 阶段一定义的结构化特征
    problem_embedding: np.ndarray          # 问题文本嵌入
    
    # 选定的策略
    strategy_id: str
    strategy_summary: str                  # 策略的摘要描述
    
    # 可选：历史上下文
    previous_attempts: List[AttemptSummary] # 之前在此任务上尝试过的策略及结果

@dataclass
class WorldModelOutput:
    # 预测的执行结果
    predicted_success_probability: float   # 成功概率 (0.0 - 1.0)
    predicted_partial_success: float       # 部分成功概率
    predicted_failure_modes: List[str]     # 可能的失败模式
    predicted_steps_to_complete: int       # 预测的执行步数
    
    # 预测的置信度
    prediction_confidence: float           # 模型对自身预测的置信度
    
    # 预测的策略-条件匹配质量
    condition_match_score: float           # 策略适用条件与问题特征的匹配度
```

### 1.2 两种实现路线

#### 路线 A：基于经验的统计模型（默认）

最简单的世界模型：一个查找表 + 平滑插值。

```python
class StatisticalWorldModel:
    """
    基于历史执行经验的统计世界模型。
    不涉及任何神经网络训练。
    """
    
    def __init__(self, experience_log: List[ExecutionRecord]):
        # 构建 (问题特征, 策略) → 成功率 的查找表
        self.success_table = self._build_table(experience_log)
        # 用 k-NN 做平滑插值
        self.knn_model = self._build_knn(experience_log)
    
    def predict(self, input: WorldModelInput) -> WorldModelOutput:
        # 1. 查找精确匹配
        state_idx = discretize(input.problem_features)
        exact_match = self.success_table.get(
            (state_idx, input.strategy_id)
        )
        
        if exact_match and exact_match.sample_count >= 10:
            # 有足够的历史数据，直接用统计值
            return WorldModelOutput(
                predicted_success_probability=exact_match.success_rate,
                prediction_confidence=min(0.9, exact_match.sample_count / 50),
                condition_match_score=self._compute_match(
                    input.problem_features, input.strategy_id
                ),
                ...
            )
        
        # 2. 没有精确匹配，用 k-NN 插值
        feature_vec = self._encode(input)
        neighbors = self.knn_model.kneighbors(
            feature_vec, n_neighbors=5
        )
        interpolated_rate = np.mean([
            n.success_rate for n in neighbors
        ])
        
        return WorldModelOutput(
            predicted_success_probability=interpolated_rate,
            prediction_confidence=0.3,  # 插值结果置信度较低
            ...
        )
    
    def update(self, record: ExecutionRecord):
        """用新的执行经验增量更新查找表"""
        state_idx = discretize(record["task"]["complexity_features"])
        strategy = record["strategy_selection"]["selected_strategy"]
        success = record["outcome"]["success"]
        
        key = (state_idx, strategy)
        if key not in self.success_table:
            self.success_table[key] = RunningStats()
        self.success_table[key].add(success)
```

**优势：** 零训练成本，完全可解释，增量更新简单。
**劣势：** 无法预测从未见过的 (问题特征, 策略) 组合；无法预测失败模式。

#### 路线 B：基于 LLM 的模拟器

用 LLM 本身作为世界模型——给它任务描述和策略，让它**预测**执行结果而不是真正执行。

```python
class LLMWorldModel:
    """
    用 LLM prompting 模拟策略执行的结果。
    比真实执行快（不需要实际运行代码/求解），
    但比统计模型慢（需要 LLM API 调用）。
    """
    
    def predict(self, input: WorldModelInput) -> WorldModelOutput:
        response = llm_call(
            self.SIMULATION_PROMPT.format(
                problem=input.problem_description,
                strategy=input.strategy_summary,
                features=input.problem_features,
                previous_attempts=input.previous_attempts
            )
        )
        return self._parse_prediction(response)

SIMULATION_PROMPT = """
你是一个经验丰富的问题解决专家。

## 任务
{problem}

## 问题特征
- 组件间耦合度: {coupling}
- 可分解性: {decomposability}
- 是否有基准: {has_baseline}
- 信息完整度: {info_completeness}
- 组件数量: {component_count}

## 选定策略
{strategy}

## 之前的尝试（如有）
{previous_attempts}

请**不要**解决这个任务。而是**预测**如果按照上述策略来解决这个任务，会发生什么。

具体预测：
1. 按这个策略走，大概会在哪一步遇到困难？
2. 最终成功解决问题的概率有多大？(0-100%)
3. 最可能的失败原因是什么？
4. 大概需要多少步才能完成？

输出 JSON:
{{
    "predicted_success_probability": 0.0-1.0,
    "predicted_bottleneck_step": "...",
    "predicted_failure_modes": ["..."],
    "predicted_steps": 整数,
    "reasoning": "..."
}}
"""
```

**优势：** 能预测从未见过的组合；能给出结构化的失败模式预测；利用了 LLM 的广泛"世界知识"。
**劣势：** 每次预测需要 LLM API 调用（~$0.01-0.05）；LLM 的预测可能不准确（模拟推理 vs 真正的逻辑验证）。

#### 路线 C：混合模型（推荐）

将路线 A 和路线 B 结合：

```python
class HybridWorldModel:
    """
    统计模型 + LLM 模拟器的混合。
    有历史数据时用统计模型（快且准），
    无历史数据时退回到 LLM 模拟（慢但有泛化能力）。
    """
    
    def __init__(self):
        self.stat_model = StatisticalWorldModel()
        self.llm_model = LLMWorldModel()
    
    def predict(self, input: WorldModelInput) -> WorldModelOutput:
        stat_pred = self.stat_model.predict(input)
        
        if stat_pred.prediction_confidence >= 0.6:
            # 统计模型有足够数据，直接用
            return stat_pred
        
        elif stat_pred.prediction_confidence >= 0.3:
            # 统计模型有一些数据，与 LLM 预测融合
            llm_pred = self.llm_model.predict(input)
            return self._fuse(stat_pred, llm_pred)
        
        else:
            # 统计模型几乎没数据，用 LLM 模拟
            return self.llm_model.predict(input)
    
    def _fuse(
        self,
        stat_pred: WorldModelOutput,
        llm_pred: WorldModelOutput
    ) -> WorldModelOutput:
        # 按置信度加权融合
        w_stat = stat_pred.prediction_confidence
        w_llm = 1 - w_stat
        
        fused_prob = (
            w_stat * stat_pred.predicted_success_probability +
            w_llm * llm_pred.predicted_success_probability
        )
        
        return WorldModelOutput(
            predicted_success_probability=fused_prob,
            prediction_confidence=max(
                stat_pred.prediction_confidence,
                llm_pred.prediction_confidence
            ),
            predicted_failure_modes=(
                llm_pred.predicted_failure_modes  # LLM 更擅长预测原因
            ),
            ...
        )
```

### 1.3 世界模型的准确性要求

世界模型不需要精确预测成功率——它只需要做到以下两点：

**要求 1：排序准确性。** 对于同一个问题，如果策略 A 实际上比策略 B 更好，世界模型预测的 A 的成功概率也应该比 B 高。用 Kendall's tau 衡量。**目标：τ ≥ 0.5。**

**要求 2：否决准确性。** 如果世界模型预测某个 (问题, 策略) 对的成功概率 < 0.2，那实际成功率也应该很低。用否决精确率（predicted low 且 actual low 的比例）衡量。**目标：否决精确率 ≥ 0.7。**

直觉上：我们不需要世界模型告诉我们"这条路一定能成功"（这太难了），只需要它能可靠地告诉我们"这条路大概率走不通"（排除明显差的选择）。

### 1.4 世界模型的校准

世界模型的预测需要定期与真实执行结果进行校准。

```python
class WorldModelCalibrator:
    """
    用真实执行结果校准世界模型的预测偏差。
    """
    
    def calibrate(
        self,
        world_model: HybridWorldModel,
        real_records: List[ExecutionRecord],
        n_calibration: int = 50
    ) -> CalibrationReport:
        
        predictions = []
        actuals = []
        
        for record in real_records[-n_calibration:]:
            # 用世界模型重新预测（隐去真实结果）
            wm_input = self._record_to_input(record)
            pred = world_model.predict(wm_input)
            predictions.append(pred.predicted_success_probability)
            actuals.append(1.0 if record["outcome"]["success"] else 0.0)
        
        # 排序准确性
        tau, p_value = kendalltau(predictions, actuals)
        
        # 否决精确率
        low_pred = [i for i, p in enumerate(predictions) if p < 0.2]
        if low_pred:
            veto_precision = sum(
                1 for i in low_pred if actuals[i] == 0.0
            ) / len(low_pred)
        else:
            veto_precision = None
        
        # 校准曲线（预测概率 vs 实际频率）
        calibration_bins = self._compute_calibration_curve(
            predictions, actuals, n_bins=5
        )
        
        # 如果偏差过大，调整世界模型的统计表
        if tau < 0.3:
            world_model.stat_model.rebuild(real_records)
        
        return CalibrationReport(
            kendall_tau=tau,
            veto_precision=veto_precision,
            calibration_bins=calibration_bins,
            needs_rebuild=(tau < 0.3)
        )
```

**校准频率：**
- 阶段一训练中：每 500 episodes 用 50 条真实执行做一次校准
- 阶段二运行中：每月一次
- 阶段四验证中：每次启动新假设验证前做一次

---

## 2. 模拟执行接口

### 2.1 与阶段一任务环境的兼容

世界模型必须实现与阶段一 `TaskEnvironment` 兼容的接口，使调度器训练代码无需修改即可在模拟和真实环境之间切换。

```python
class SimulatedTaskEnvironment(TaskEnvironment):
    """
    世界模型驱动的模拟任务环境。
    实现与真实 TaskEnvironment 完全相同的接口。
    """
    
    def __init__(
        self,
        world_model: HybridWorldModel,
        real_env: TaskEnvironment   # 底层的真实环境（用于校准）
    ):
        self.world_model = world_model
        self.real_env = real_env
        self.current_task = None
    
    def reset(self, task_id: str) -> TaskObservation:
        """复用真实环境的任务定义"""
        return self.real_env.reset(task_id)
    
    def get_problem_description(self) -> str:
        return self.real_env.get_problem_description()
    
    def execute_with_strategy(
        self,
        strategy_prompt: str,
        llm_executor: LLMExecutor  # 在模拟模式下不使用
    ) -> ExecutionOutcome:
        """
        不实际执行，而是用世界模型预测结果。
        """
        wm_input = WorldModelInput(
            problem_features=self._extract_features(),
            problem_embedding=self._embed_problem(),
            strategy_id=self._infer_strategy_id(strategy_prompt),
            strategy_summary=strategy_prompt[:200]
        )
        
        wm_output = self.world_model.predict(wm_input)
        
        # 将世界模型的预测转化为 ExecutionOutcome 格式
        simulated_success = (
            random.random() < wm_output.predicted_success_probability
        )
        
        return ExecutionOutcome(
            success=simulated_success,
            partial_success=(
                not simulated_success and
                random.random() < wm_output.predicted_partial_success
            ),
            trajectory=[],   # 模拟不产生真实轨迹
            total_steps=wm_output.predicted_steps_to_complete,
            wall_clock_seconds=0.1,  # 模拟几乎瞬时
            llm_output="[simulated]",
            evaluation_score=wm_output.predicted_success_probability,
            failure_reason=(
                wm_output.predicted_failure_modes[0]
                if wm_output.predicted_failure_modes and not simulated_success
                else None
            ),
            is_simulated=True   # 标记为模拟结果
        )
    
    def evaluate(self, outcome: ExecutionOutcome) -> float:
        return outcome.evaluation_score
```

### 2.2 模拟与真实的切换

阶段一的训练框架需要支持在模拟和真实环境之间切换：

```python
class ModelBasedTrainer:
    """
    Model-based RL 训练器。
    大部分 episodes 在世界模型中运行，
    少量 episodes 在真实环境中运行以校准。
    """
    
    def __init__(
        self,
        dispatcher,
        sim_env: SimulatedTaskEnvironment,
        real_env: TaskEnvironment,
        real_ratio: float = 0.1    # 10% 的 episodes 在真实环境中
    ):
        self.dispatcher = dispatcher
        self.sim_env = sim_env
        self.real_env = real_env
        self.real_ratio = real_ratio
    
    def train_episode(self, task_id: str) -> EpisodeResult:
        use_real = random.random() < self.real_ratio
        env = self.real_env if use_real else self.sim_env
        
        # 标准的 RL episode
        obs = env.reset(task_id)
        features = extract_features(obs)
        action = self.dispatcher.select_strategy(features)
        outcome = env.execute_with_strategy(action.prompt, self.executor)
        reward = env.evaluate(outcome)
        
        # 更新调度器
        self.dispatcher.update(features, action, reward)
        
        # 如果是真实执行，用结果更新世界模型
        if use_real:
            self.sim_env.world_model.update(
                self._to_record(task_id, action, outcome)
            )
        
        return EpisodeResult(
            reward=reward,
            is_real=use_real,
            task_id=task_id
        )
```

---

## 3. 技术实现

### 3.1 项目文件结构

```
assumption_agent/
├── ...                              # 阶段零的所有目录保持不变
├── world_model/                     # 阶段零点五新增
│   ├── models/
│   │   ├── statistical_model.py     # 路线 A: 统计模型
│   │   ├── llm_simulator.py         # 路线 B: LLM 模拟器
│   │   ├── hybrid_model.py          # 路线 C: 混合模型（默认）
│   │   └── base_model.py            # 世界模型基类
│   ├── sim_env.py                   # SimulatedTaskEnvironment
│   ├── calibrator.py                # 世界模型校准
│   ├── model_based_trainer.py       # Model-based RL 训练器
│   ├── prompts.py                   # LLM 模拟 prompt
│   └── config.py
├── scripts/
│   ├── ...
│   ├── build_world_model.py         # 从经验日志构建统计模型
│   ├── calibrate_world_model.py     # 校准
│   └── evaluate_world_model.py      # 准确性评估
└── tests/
    ├── ...
    ├── test_statistical_model.py
    ├── test_llm_simulator.py
    ├── test_hybrid_model.py
    └── test_sim_env.py
```

### 3.2 依赖与成本

| 组件 | 技术 | 成本 |
|------|------|------|
| 统计模型 | numpy, scikit-learn (k-NN) | 零 |
| LLM 模拟器 | OpenAI API (gpt-4o-mini) | ~$0.01/次预测 |
| 混合模型 | 上述两者 | 视 LLM 调用频率而定 |

**阶段一训练中的成本估算：**
- 50000 episodes，其中 90% 用世界模型
- 统计模型覆盖 ~60% 的状态-策略对 → 27000 次免费
- LLM 模拟覆盖 ~30% → 13500 次 × $0.01 = ~$135
- 真实执行 10% → 5000 次（成本与无世界模型方案相同，但总量只有 1/10）
- **总额外成本：~$135，换来的是真实执行量减少 90%**

### 3.3 冷启动问题

系统刚启动时没有执行经验，统计模型为空。冷启动策略：

1. **阶段零的标注数据作为种子：** 用阶段零的 100-150 道标注题和人类标注的"最优策略"作为初始数据——虽然没有真实的执行结果，但可以给出 (问题特征, 策略) → 适用/不适用 的先验分布
2. **纯 LLM 模拟启动：** 前 2000 episodes 完全依赖 LLM 模拟器，同时用 10% 的真实执行逐步填充统计模型
3. **过渡期：** 随着统计模型数据积累，LLM 模拟器的使用比例自动下降

---

## 4. 实验设计

### 4.1 世界模型准确性评估

**实验设计：**
- 收集 200 条真实执行记录
- 对每条记录，用世界模型重新预测（隐去真实结果）
- 计算排序准确性（Kendall's τ）和否决精确率

### 4.2 Model-Based vs Model-Free RL

**假设：** 使用世界模型的 model-based RL 调度器训练在相同的真实执行预算下，比 model-free RL 达到更高的最终性能。

**实验设计：**
- 方法 A：Model-free（所有 episodes 在真实环境中）
- 方法 B：Model-based（90% 模拟 + 10% 真实）
- 固定真实执行总量相同（如 5000 次），方法 B 额外有 45000 次模拟
- 比较两者在测试集上的策略选择准确率

---

## 5. 风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| 世界模型预测严重不准 | 中 | 高 | 频繁校准（每 500 episodes）；如果 τ < 0.3，暂停 model-based 训练回到 model-free |
| LLM 模拟器的"自信偏差" | 高 | 中 | LLM 倾向于给出偏高的成功概率预测；用真实数据校准一个偏差修正项 |
| 统计模型的状态空间稀疏 | 中 | 中 | k-NN 平滑插值处理；数据不足的区域自动退回 LLM 模拟 |
| 模拟训练的策略在真实环境中失效 | 中 | 高 | 10% 的真实执行作为持续校准；在真实环境中做最终评估 |
| 冷启动阶段世界模型完全不可靠 | 高 | 中 | 前 2000 episodes 降低模拟比例到 50%；增加真实执行比例 |

---

## 6. 完成标准（Definition of Done）

1. 混合世界模型已实现并通过单元测试
2. `SimulatedTaskEnvironment` 与阶段一 `TaskEnvironment` 接口完全兼容
3. 在至少 100 条真实执行记录上，排序准确性 Kendall's τ ≥ 0.4
4. 否决精确率 ≥ 0.6
5. Model-based RL 训练框架已实现，可在模拟和真实环境间切换
6. 校准机制已实现并通过测试
7. 冷启动流程已设计并在小规模实验中验证