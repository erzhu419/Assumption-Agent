# 阶段四：新假设生成——完整开发文档 (v1)

---

## 🏁 2026-04-23 v16 里程碑回看

**写于**: v16 架构验证完成后。**这是整个项目的终极目标**——让 AI 在已有框架不够时提出新方法论假设并自我验证。

### 原意 → v16 实际覆盖

**原设计的 4 个核心模块**:
1. 策略空白检测 (gap_detector) — 识别 KB 覆盖不到的问题
2. 候选策略生成 (hypothesis_generator) — LLM 生成新策略候选
3. 形式化一致性检验 — 确保新策略与已有不矛盾
4. 经验验证协议 — 在任务上测试新策略的价值

**v16 对这 4 个模块的覆盖度：**

| 原模块 | v16 对应 | 覆盖度 |
|---|---|---|
| gap_detector | ❌ 没做（v16 无"所有策略都失败"的信号检测） | 0% |
| hypothesis_generator | ✅ **部分**: v13/v16 的 audit 层是 micro-version（提出 1-2 个"草稿没应用的 prior"作为新假设） | ~20% |
| 形式化一致性检验 | ❌ 没做（Phase 3 形式化本身没做） | 0% |
| 经验验证协议 | 🟡 已有评估框架（A/B judge）可支持，但缺**自动回流**机制 | ~30% |

**v16 离"终极目标"还很远**。它实现的只是**单问题内的微型版本**。

### v16 audit-revise 作为 micro-hypothesis-generation

**原设想**:
```
KB 所有策略失败 → LLM 提出新 meta-strategy → 验证 → 纳入 KB
(跨问题 + 跨策略 + 无限递归)
```

**v16 实际**:
```
问题 P → Turn 1 draft → Turn 2 audit: "哪条 prior 没真用？"
→ 识别 1-2 个遗漏 → integrate into revised answer
(单问题 + 已有 prior 范围 + 1 次迭代)
```

这是原目标的**极度简化版**——相同的 recursive pattern，但深度只有 1，宽度只有 1-2 个 blindspot。

### 用户在 Claude.md 中的原愿景 (回顾)

> "我想要的是一个能在已有哲学框架不足时提出新方法论假设并自我验证的 AI 系统。"

**v16 在这条路上的进展位置：**
- ✅ 单问题内的反思-改写 (audit pass)
- 🟡 跨问题的经验提取（v6 的 mine_v5_losses 做了一次）
- ❌ 策略空白检测（不知道什么时候"已有哲学不够"）
- ❌ 生成跨 domain 通用的新 meta-strategy
- ❌ 自动纳入 KB 并在后续使用
- ❌ 递归式无限深度

### 仍未做到的原意（核心 gaps）

1. **Gap detection**: 如何从"多个问题都失败"识别到"缺少某类 strategy"？  
   - 原设计: 调度器的 top-K 置信度都低 + 执行失败 → flag
   - v16 无调度器无 confidence，没法识别

2. **Hypothesis 生成 ≠ 答案生成**: v16 的 audit 生成的是"修订后的答案"，不是新的 meta-strategy  
   - 原设计: 从失败中抽 出"我现在 miss 的是什么 orientation"  
   - v16: 从失败中抽出"本题漏用的是什么 existing orientation"  
   - **缺了一层：从"已有不够"跳到"需要新的"**

3. **Novelty 验证**: 新策略和已有 wisdom library 的 overlap 检测没做

4. **自动回流**: v6 的 141 new triggers 手动做了一次，没有自动触发机制

5. **递归深度**: v16 只有 2-turn (draft + audit)，没有 "audit revised again, audit again..." 的递归循环

### 与用户最新讨论的连接

用户在这轮对话中提过**"递归式的循环提出假设到论证假设，也许本来就该发生在 reasoning/thinking 层"** （见 `world_model_thinking_layer.md`）。

这意味着**Phase 4 的原设计可能方向错了**：
- 原设计: **外部搭建 gap_detector + hypothesis_generator + validator** (工程化架构)
- 更接近目标的可能路径: **让 LLM 的 thinking tokens 自己做递归假设生成** (依赖模型能力)

如果后者正确，Phase 4 的大部分工程模块可能是多余的——真正要做的是：
- 优化 prompt 让 thinking tokens 自己展开 hypothesis loop
- 或等更强 reasoning models (o3/o4/o5) 把这件事内化

### 用户 2026-04-23 的核心 framing: **Phase 4 = Residual Chaos → Paradigm Shift**

用户原话:
> "这和 MC-WM 那个项目最开始的设想一样——总有目前哲学框架无法解释的 **residual chaos**（在 MC-WM 里叫做 aleatoric uncertainty）。把这些 aleatoric uncertainty/chaos 积攒起来，里面可能蕴含一个全新的哲学观点。例如一个科学家算来算去那个量比理论少了，于是他大胆认为理论错了，结果新理论的确更完美解释了实验数据——这就是提出了新的假设并自我论证。"

这**彻底重定义了 Phase 4 的路径**，和原 dev doc 的工程化架构（gap_detector + hypothesis_generator + consistency_checker）**本质不同**。

### Popperian paradigm shift 的机制（人类科学史上验证过上百次）

经典例子:
- **Mercury 近日点进动**: Newton 力学算的 vs 观测的差 43″/century → Einstein GR 诞生
- **Michelson-Morley 实验**: 预期的 ether velocity 测不到 → 相对论
- **黑体辐射紫外灾难**: 经典物理发散 → Planck 量子假设
- **原子光谱离散**: Rutherford 模型预测连续谱 → Bohr 模型
- **Kuhn 的范式转换**: 异常累积到临界点 → 旧范式无法包容 → 新范式诞生

**共同模式**:
```
theory T → predicts X
observe actual outcome Y
residual = Y - X
if residual is RANDOM → T is fine
if residual is SYSTEMATIC → T has missing component
  → propose T' that explains residual + preserves T's previous successes
  → T' 是新哲学假设
```

### 和 MC-WM aleatoric uncertainty 的对偶

MC-WM 的核心区分:
- **Epistemic uncertainty** = 数据不够（可 reducible）
- **Aleatoric uncertainty** = 世界本身的随机性（irreducible）

**关键 insight**（用户点出）:
> "aleatoric" 里往往**不是纯随机**，而是**当前 theory 覆盖不了的 structure**。系统地积累它，就能发现新 theory。

在 assumption-agent 框架下的对应:
- **Epistemic residual** = 现有 wisdom library 没找全的 orientation（wisdom 不够多）
- **Aleatoric residual** = 即使 library 全了，也有一些 "wisdom 解释不了的失败 pattern"（wisdom 的表征形式不够）

**Phase 4 真正的任务**: 不是"当所有 wisdom 都失败时生成新的"，而是**从系统性 residual 中提炼新的哲学视角**。

### Phase 4 v2 设计: Residual-Driven Paradigm Shift Pipeline

**Stage 1 — Residual 收集**:
```python
for problem in all_attempted_problems:
    # 每个问题都有 "wisdom-guided answer" 和 "baseline answer"
    # residual = judge's reasoning for which wisdom failed to help
    # 不是看 win/lose，而是看"wisdom 预测 LLM 会这样答、LLM 实际那样答"的 gap
    residuals[problem] = extract_residual(
        wisdom_selected, draft_answer, baseline_answer, judge_reasoning
    )
```

**Stage 2 — Pattern Detection (关键步骤)**:
对 residuals 做 clustering / embedding:
- 如果分布均匀随机 → epistemic，只需扩 library
- 如果某些 cluster 集中 → **发现了 systematic bias** = aleatoric-structure

具体算法（LLM-native，不需要训小模型）:
```
1. 取过去 200 题中 wisdom 明显没帮上的 50 题
2. 让 GPT-5.4 读全部 50 题 + 各自的 (selected_wisdom, draft_answer, 为什么没帮上)
3. Prompt: "这 50 个失败里，有没有一个共有的 missing orientation，
   现有 75 条 wisdom library 里没覆盖？"
4. 如果 GPT 能提炼出 3-5 条 candidate new wisdom → 进 Stage 3
5. 如果 GPT 说"没有系统 pattern，只是偶发失败" → 跳过，继续收集
```

**Stage 3 — 新 wisdom 验证（关键：self-consistency）**:
```
1. 新 wisdom W_new 加入 library
2. 在没有用过的 held-out 问题上测试:
   - 用 v16 架构（cases + audit）跑一次
   - 比较 with W_new vs without W_new
3. 如果 with W_new 在目标 cluster 上 > +5pp → KEEP
4. 否则 REVERT
```

这**就是"提出新假设 + 自我论证"**。和用户描述的"那个科学家算来算去"完全同构:
- 科学家: 累积观测数据 → 发现 residual → 提新理论 → 新理论解释 residual + 保留旧成功 → KEEP
- Phase 4 v2: 累积 problem-solving failures → 发现 residual cluster → GPT 提新 wisdom → 新 wisdom 在 held-out 上 >+5pp → KEEP

### 和 v16 的 audit 层不同在哪

v16 audit 是**单问题内**的 "Turn 1 draft → Turn 2 audit"。审查范围是 existing wisdom。

Phase 4 residual-driven 是**跨问题**的 "all failures → common missing orientation"。审查范围**超出** existing wisdom。

这两者互补:
- v16 audit = 单次推理的 local refinement
- Phase 4 residual → paradigm shift = 跨推理的 global framework update

### 与原 dev doc 的关系

原 dev doc §0.3 的 4 目标:
1. 策略空白检测 → **被 residual clustering 替代**（更 principled）
2. 候选策略生成 → **被 GPT-5.4 从 residual 归纳替代**
3. 形式化一致性检验 → **被 held-out empirical 验证替代**
4. 经验验证协议 → **保留，且更清晰**（A/B on held-out）
5. 知识库集成 → **保留，标准 wisdom library append**

**原设计的大方向对了，细节机制都可以用 v16 的路径替换**。

### 仍未做到的原意（Phase 4 的真 gap）

即使引入 residual-driven pipeline:

1. ❌ **Residual 收集架构**: 我们目前只有 A/B 判决 (win/lose)，不记录 residual text。需要 judge 的 reasoning 存档 + 可分析
2. ❌ **长期 corpus 积累**: 100 题太少，residual pattern 不够 dense。需要 n=500+ 才能聚类出真 pattern
3. ❌ **"新 wisdom 不与已有矛盾"的自动检测**: 原设计的 consistency checker 没做（虽然 v16 的 diverse exemplar mining 间接在做）
4. ❌ **递归深度**: v16 只做 2-turn。用户原愿景里的"递归"意味着 meta-wisdom（wisdom about wisdom），还没碰

### 启示

**用户这个 framing 是对 Phase 4 的根本性重新定位**。从"工程化 hypothesis generator"到**"residual-driven paradigm shift mimicking 科学史"**。

这也解释了为什么 v16 的 audit 不够——它只做了 local paradigm correction，没做 **across-problem residual accumulation + new paradigm proposal**。

**v16 artifact 索引**:
- 单问题 audit: `phase2_v13_reflect_framework.py`, `phase2_v16_cases_reflect.py`
- 跨问题 residual 分析（需要新建）: 应该做 `phase 4/residual_analyzer.py` 读取 judgments + answers 差异
- MC-WM 参考: 用户另一个项目 `/home/erzhu419/mine_code/MC-WM/` 的 aleatoric uncertainty 处理
- 理论: Popper *Logic of Scientific Discovery* + Kuhn *Structure of Scientific Revolutions*

**Phase 4 v2 MVP 的下一步**（如果要做）:
1. 把所有 existing judgment 文件里 `reasoning` 字段 extract 成 corpus
2. 筛出 v16 明显没帮上的 ~30-50 例
3. GPT-5.4 分析：有没有系统性 missing orientation
4. 如果有 → 提炼 W_76+（新 wisdom）→ 在 held-out 上 A/B 验证
5. PASS → 正式加入 library，Phase 4 闭环完成一次

**v16 artifact 索引**:
- Code: `phase2_v13_reflect_framework.py` (micro-audit), `phase2_v16_cases_reflect.py` (cases + audit)
- 相关讨论: `reflection_wisdom_vs_technique.md`, `world_model_thinking_layer.md`
- 原愿景: `Claude.md`, `Gemini.md` (recursive hypothesis generation 的哲学 / 数学基础)

---

## 0. 文档概述

### 0.1 本阶段在整体架构中的位置

这是整个系统的终极目标，也是整段思考旅程的起点——让智能体在碰到已有哲学框架解决不了的问题时，能**提出与已有框架不冲突的新方法论假设，并自我验证它**。

阶段零到阶段三构建的系统已经能做到：装载人类哲学（阶段零）、选择正确策略（阶段一）、从经验中精化策略边界（阶段二）、在数学空间中检测策略同构（阶段三）。但这一切都发生在**已有策略集合的内部**——系统从未超越人类给定的 15-20 条元策略。

本阶段要突破这个边界。当系统面对一个问题，知识库中所有策略的适用条件匹配分数都很低、且执行后全部失败时，系统必须认识到"我现有的工具箱不够用了"，然后做出人类科学家最核心的创造性行为——提出一个新的方法论假设。

**这是从"利用已有哲学"到"创造新哲学"的跃迁。**

回到 Claude.md 讨论中的那句话：

> 你真正想要的不是一个能度量 E=mc² 和楞次定律距离的系统。你想要的是一个能在已有哲学框架不足时提出新方法论假设并自我验证的 AI 系统。

本阶段就是实现这件事。

### 0.2 前置依赖

本阶段严重依赖前三个阶段的输出：

| 依赖 | 来源 | 用途 |
|------|------|------|
| 策略知识库 | 阶段零+二 | 定义"已知策略空间"的内容 |
| 调度器 | 阶段一 | 检测"所有已有策略都失败"的信号 |
| 反馈管线 | 阶段二 | 验证新策略并将其集成到知识库 |
| 形式化表示 | 阶段三 | 检测新策略与已有策略的关系（同构？冲突？互补？） |

**新模块只有一个：假设生成与一致性检验模块。** 其余全部沿用前三个阶段的已验证组件。

### 0.3 本阶段目标

1. 建立"策略空白检测"机制——识别知识库覆盖不到的问题类型
2. 构建新策略的候选生成管线——从 LLM 的生成能力中提取结构化的策略候选
3. 实现形式化一致性检验——确保新策略与已有策略不矛盾
4. 设计经验验证协议——在任务上测试新策略并评估其价值
5. 完成知识库集成——将通过验证的新策略正式纳入知识库

### 0.4 交付物

1. 策略空白检测模块（`gap_detector/`）
2. 候选策略生成模块（`hypothesis_generator/`）
3. 形式化一致性检验模块（`consistency_checker/`）
4. 经验验证协议与执行框架（`validation/`）
5. 知识库集成流程（复用阶段二的机制并扩展）
6. 一篇终极论文，核心贡献：首个能在已有哲学框架不足时提出并验证新方法论假设的 AI 系统

### 0.5 时间预算

总计 6-10 个月。空白检测与候选生成：6-8 周。一致性检验：4-6 周。经验验证与迭代：8-12 周。分析与论文写作：4-6 周。

### 0.6 技术路线概述

本阶段的核心操作——空白检测、候选生成、一致性检验、经验验证——全部可以通过 LLM prompting + 阶段三的数学工具 + 阶段二的反馈管线实现，**不需要 fine-tune**。

Fine-tune 的潜在价值在于训练一个专门的"假设生成器"，使其比通用 LLM 更擅长提出结构化的方法论候选。该方案在附录 A 中详述。

---

## 1. 策略空白检测

### 1.1 什么是"空白"

策略空白（strategy gap）是指问题特征空间中的一个区域，在该区域内知识库中所有策略的表现都不令人满意。形式化地：

设 $x \in \mathcal{X}$ 为一个离散化的问题状态（阶段三定义），$r(S_k, x)$ 为策略 $S_k$ 在状态 $x$ 下的平均成功率。状态 $x$ 是一个策略空白当且仅当：

$$\max_{k} r(S_k, x) < \theta_{\text{gap}}$$

其中 $\theta_{\text{gap}}$ 是空白阈值（默认 0.4）。

直觉上：如果在某种问题特征下，知识库中最好的策略也只有不到 40% 的成功率，说明现有策略对这类问题力不从心。

### 1.2 空白检测的数据来源

空白检测需要策略在各种问题状态下的成功率数据。来源有**四个**，其中来源 1 是最高优先级：

**来源 1（首要）：阶段二的跨策略失败信号**

Phase 2 新增的 `CrossStrategyFailureDetector`（Phase 2 第 6 节）已经在经验持续积累时**实时运行**，将 ≥3 条策略全部失败且 0 成功的问题特征标记为 `strategy_gap`，写入 `experience_log/distilled/gap_signals/`。Phase 4 的空白检测器**首先读取这些信号**，而非从原始经验日志重新统计。

```python
def load_phase2_gap_signals(self) -> List[StrategyGap]:
    """
    优先读取 Phase 2 已经检测到的策略空白信号。
    这些信号已经做了跨策略因果归因——质量高于从原始日志重新统计。
    """
    gap_signals_dir = "experience_log/distilled/gap_signals/"
    signals = load_all_json(gap_signals_dir)
    
    return [
        StrategyGap(
            state=signal["problem_state"],
            state_idx=discretize(signal["problem_state"]),
            best_existing_strategy=None,   # 全部失败，无最佳
            best_existing_rate=0.0,
            total_samples=signal["total_attempts"],
            failing_strategies=signal["failed_strategies"],
            source="phase2_cross_strategy",  # 标记来源
            # Phase 2 已经标记这是 strategy_gap 而非 condition_gap
            gap_type=GapType.TRUE_GAP
        )
        for signal in signals
        if signal["signal_type"] == "strategy_gap"
    ]
```

**来源 2（补充）：阶段一调度器的执行经验**

对 Phase 2 信号未覆盖的区域，从原始经验日志补充统计。这捕捉的是"最佳策略成功率 < 40% 但不是全部失败"的灰色地带——Phase 2 的跨策略检测器不会标记这些（因为至少有一条策略偶尔成功）。

```python
class GapDetector:
    
    def detect_gaps(
        self,
        experience_log: List[ExecutionRecord],
        feature_space: List[DiscreteState],
        strategies: List[str],
        gap_threshold: float = 0.4,
        min_samples: int = 10
    ) -> List[StrategyGap]:
        
        # 统计每个 (状态, 策略) 对的成功率
        success_counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        # success_counts[state_idx][strategy_id] = [successes, total]
        
        for record in experience_log:
            state_idx = discretize(record["task"]["complexity_features"])
            strategy = record["strategy_selection"]["selected_strategy"]
            total = success_counts[state_idx][strategy]
            total[1] += 1
            if record["outcome"]["success"]:
                total[0] += 1
        
        gaps = []
        for state_idx, state in enumerate(feature_space):
            # 检查该状态下是否有足够的数据
            total_samples = sum(
                counts[1]
                for counts in success_counts[state_idx].values()
            )
            if total_samples < min_samples:
                continue  # 数据不足，无法判断
            
            # 计算最高成功率
            best_rate = 0.0
            best_strategy = None
            for sid in strategies:
                counts = success_counts[state_idx].get(sid, [0, 0])
                if counts[1] > 0:
                    rate = counts[0] / counts[1]
                    if rate > best_rate:
                        best_rate = rate
                        best_strategy = sid
            
            if best_rate < gap_threshold:
                gaps.append(StrategyGap(
                    state=state,
                    state_idx=state_idx,
                    best_existing_strategy=best_strategy,
                    best_existing_rate=best_rate,
                    total_samples=total_samples,
                    failing_strategies=self._get_all_tried(
                        success_counts[state_idx]
                    )
                ))
        
        return gaps
```

**来源 2：阶段三的形式化拓扑**

阶段三的策略空间可视化可能揭示 Markov 核分布的"稀疏区域"——问题状态空间中没有任何策略的 Markov 核给出高概率行动的区域。

```python
def detect_coverage_gaps(
    formal_kb: Dict[str, np.ndarray],
    entropy_threshold: float = 0.9
) -> List[int]:
    """
    找到所有策略的 Markov 核在其上都接近均匀分布的状态。
    均匀分布 = 策略"不知道在这种状态下该做什么" = 空白。
    """
    max_entropy = np.log(formal_kb[list(formal_kb.keys())[0]].shape[1])
    
    uncovered_states = []
    for state_idx in range(formal_kb[list(formal_kb.keys())[0]].shape[0]):
        all_high_entropy = True
        for sid, K in formal_kb.items():
            row = K[state_idx, :]
            entropy = -np.sum(row * np.log(row + 1e-10))
            if entropy / max_entropy < entropy_threshold:
                all_high_entropy = False
                break
        if all_high_entropy:
            uncovered_states.append(state_idx)
    
    return uncovered_states
```

**来源 3：调度器的低置信度信号**

阶段一的调度器在选择策略时输出 `confidence` 值。如果调度器在某类问题上持续给出低置信度（< 0.3），说明它"不确定该用什么策略"——这本身就是空白的信号。

### 1.3 空白的分类

检测到的空白不全都需要新策略。分类如下：

| 空白类型 | 特征 | 处理方式 |
|---------|------|---------|
| **数据空白** | 该状态下的样本量 < 10 | 不生成新策略，收集更多数据 |
| **适用条件空白** | 某条已有策略其实适用，但其适用条件描述不够精确 | 交给阶段二精化条件 |
| **组合空白** | 单独的策略不够用，但策略组合可能有效 | 生成"策略组合"候选而非全新策略 |
| **真正的空白** | 已有策略在结构上确实无法覆盖 | 生成新策略候选（本阶段核心） |

区分这四种空白的方法：

```python
def classify_gap(
    gap: StrategyGap,
    kb_snapshot: KBSnapshot,
    formal_kb: Dict[str, np.ndarray]
) -> GapType:
    
    # 数据空白
    if gap.total_samples < 10:
        return GapType.DATA_GAP
    
    # 适用条件空白：检查是否有策略的 Markov 核在该状态下
    # 给出高概率行动，但调度器没选它
    for sid, K in formal_kb.items():
        max_action_prob = np.max(K[gap.state_idx, :])
        if max_action_prob > 0.3 and sid not in gap.failing_strategies:
            return GapType.CONDITION_GAP
    
    # 组合空白：检查是否有两条策略的 Markov 核在该状态下
    # 互补（一条覆盖某些行动，另一条覆盖其余行动）
    strategy_ids = list(formal_kb.keys())
    for i in range(len(strategy_ids)):
        for j in range(i + 1, len(strategy_ids)):
            K_i = formal_kb[strategy_ids[i]][gap.state_idx, :]
            K_j = formal_kb[strategy_ids[j]][gap.state_idx, :]
            # 如果两者的高概率行动不重叠且合集覆盖多数行动
            top_i = set(np.where(K_i > 0.15)[0])
            top_j = set(np.where(K_j > 0.15)[0])
            if len(top_i & top_j) == 0 and len(top_i | top_j) >= 5:
                return GapType.COMBINATION_GAP
    
    # 排除以上三种后，判定为真正的空白
    return GapType.TRUE_GAP
```

---

## 2. 新策略候选生成

### 2.1 设计理念

候选生成是本阶段最具创造性的环节。它利用 LLM 的生成能力，在理解空白特征的基础上提出新的方法论策略。

**关键约束：** 生成的候选必须遵循阶段零定义的策略 schema——它不是一段自由文本，而是一个结构化的 JSON 对象，包含名称、描述、操作步骤、适用条件等所有必填字段。这确保新策略可以被无缝集成到现有系统中。

### 2.2 生成流程

```
策略空白
(GapType.TRUE_GAP)
        │
        ▼
┌───────────────────┐
│  Step 1: 空白分析   │  理解为什么已有策略失败
│  (Gap Analysis)    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Step 2: 候选生成   │  LLM 提出 3-5 个候选策略
│  (Candidate Gen)   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Step 3: 结构化     │  将自然语言候选转为知识库 schema
│  (Structuring)     │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Step 4: 去重       │  检查是否与已有策略同构
│  (Deduplication)   │  → 调用阶段三的同构检测
└────────┬──────────┘
         │ 通过去重的候选
         ▼
    进入一致性检验（第 3 节）
```

### 2.3 Step 1：空白分析

在生成新策略之前，先深入分析为什么已有策略在该空白区域失败。

```python
GAP_ANALYSIS_PROMPT = """
你是一个方法论研究专家。

## 背景
以下问题特征组合是一个"策略空白"——现有知识库中所有策略在此类问题上的成功率都低于 {gap_threshold}%。

## 问题特征
- 组件间耦合度: {coupling}
- 可分解性: {decomposability}
- 是否有基准: {has_baseline}
- 信息完整度: {information_completeness}
- 组件数量: {component_count}

## 已尝试过的策略及其失败原因
{failing_strategies_with_reasons}

## 分析任务
请回答：
1. 这些策略为什么都失败了？它们共同的盲点是什么？
2. 这种问题特征组合的核心困难是什么？
3. 理想的解决策略应该具备什么特性？（不需要给出具体策略，只描述特性）

输出 JSON:
{{
    "common_blindspot": "所有失败策略共同的盲点描述",
    "core_difficulty": "这类问题的核心困难",
    "desired_properties": [
        "新策略应具备的特性 1",
        "新策略应具备的特性 2",
        ...
    ],
    "reasoning": "分析推理过程"
}}
"""
```

### 2.4 Step 2：候选生成

基于空白分析的结果，让 LLM 提出候选新策略。

```python
CANDIDATE_GENERATION_PROMPT = """
你是一个方法论创新专家。你的任务是提出一个全新的问题解决策略。

## 空白分析
核心困难: {core_difficulty}
已有策略的共同盲点: {common_blindspot}
新策略应具备的特性: {desired_properties}

## 已有策略列表（新策略不能和这些重复）
{existing_strategy_summaries}

## 要求
提出 3 个候选。候选可以是**全新策略**或**已有策略的新组合**（标明类型）。

每个候选需要：
1. 有一个简洁的名称
2. 有一句话描述
3. 如果是新策略：有 5-8 个操作步骤，**每步标注遇到困难时应如何处理**（递归调用其他策略或放弃）
4. 如果是新组合：指定策略序列和过渡条件
5. 不是已有策略的简单重组——必须包含至少一个"新思路"
6. 至少引用一个人类知识体系中的灵感来源
7. 提供 3-4 条知识三元组（subject, relation, object），捕捉策略的核心逻辑骨架

## 灵感参考
以下是一些可能与当前空白相关的思维模式（仅供参考）：
- 对偶性思维：从问题的对立面入手
- 层次分解：不是分解为子问题，而是分解为不同抽象层次
- 约束翻转：不是在约束下找解，而是修改约束
- 随机扰动：引入随机性来跳出局部最优
- 元认知：对自己的推理过程进行推理
- 类比迁移：从远程领域借用结构
- 退步法：故意让问题变得更难，从更一般的问题中找到线索

输出 JSON:
{{
    "candidates": [
        {{
            "type": "new_strategy 或 new_composition",
            "name_zh": "策略名称（中文）",
            "name_en": "Strategy Name (English)",
            "one_sentence": "一句话描述",
            "inspiration_source": "灵感来源",
            "key_novelty": "相比已有策略，新在哪里",
            "knowledge_triples": [
                {{"subject": "...", "relation": "...", "object": "..."}},
                ...
            ],
            // 如果 type == "new_strategy":
            "operational_steps": [
                {{
                    "step": 1,
                    "action": "步骤描述",
                    "on_difficulty": "遇困难时的处理（递归调用哪条策略，或 null）"
                }},
                ...
            ],
            // 如果 type == "new_composition":
            "sequence": ["S_XX", "S_YY"],
            "transition_condition": "何时从第一条策略切换到第二条",
            "hypothesized_favorable_conditions": [
                "适用条件 1",
                "适用条件 2"
            ],
            "hypothesized_unfavorable_conditions": [
                "不适用条件 1"
            ]
        }},
        ...
    ]
}}
"""
```

**为什么让 LLM 生成 3 个而非 1 个：** 多样性。LLM 的第一个回答往往是最"安全"的——最接近训练数据中的已知模式。要求 3 个候选迫使 LLM 探索更远的可能性空间。

**灵感参考的作用：** 提供一组抽象的思维方向，帮助 LLM 跳出"在已有策略的表面上做变形"的惯性。这些灵感来源本身就是元策略——"关于如何发明新策略的策略"。

### 2.5 Step 3：结构化

将 LLM 的候选输出转化为完全符合阶段零 2.1 节策略 schema 的 JSON 对象。

```python
STRUCTURING_PROMPT = """
请将以下候选策略转化为标准的知识库格式。

## 候选策略
名称: {candidate_name}
描述: {candidate_description}
操作步骤: {candidate_steps}
假设的适用条件: {candidate_conditions}
灵感来源: {inspiration_source}

## 目标格式
请输出完整的策略 JSON，遵循以下格式：
- id: 使用 "S_NEW_{number}" 格式
- 所有字段都必须填写
- source_references 填写灵感来源
- confidence 统一设为 0.5（初始假设）
- status 统一设为 "under_review"
- historical_cases 中至少构造 1 个假想的成功案例和 1 个假想的失败案例

{strategy_schema_template}
"""
```

### 2.6 Step 4：去重

调用阶段三的同构检测器，检查候选策略是否与已有策略在结构上等价。

```python
def deduplicate_candidate(
    candidate_kernel: np.ndarray,
    candidate_seg: StrategyExecutionGraph,
    formal_kb: Dict[str, np.ndarray],
    seg_graphs: Dict[str, StrategyExecutionGraph],
    iso_detector: IsomorphismDetector
) -> DeduplicationResult:
    """
    双表示去重：同时检查 Markov 核和策略执行图。
    Phase 3 新增了三种关系类型，去重需要全部覆盖：
    - isomorphic: 两种表示都近 → 真重复
    - surface_similar_only: 核近但 SEG 远 → 不算重复（结构不同）
    - hidden_isomorphic: 核远但 SEG 近 → 隐藏重复（最容易遗漏）
    """
    for sid, K_existing in formal_kb.items():
        seg_existing = seg_graphs.get(sid)
        report = iso_detector._analyze_pair(
            "candidate", candidate_kernel,
            sid, K_existing,
            seg_a=candidate_seg,
            seg_b=seg_existing
        )
        if report.relationship in ("isomorphic", "a_subsumes_b",
                                    "hidden_isomorphic"):
            return DeduplicationResult(
                is_duplicate=True,
                duplicate_of=sid,
                relationship=report.relationship,
                distances=report.distances,
                # hidden_isomorphic 特别标注：
                # 候选看起来新颖（Markov 核远），但执行逻辑和已有策略相同
                is_hidden_duplicate=(
                    report.relationship == "hidden_isomorphic"
                )
            )
    
    return DeduplicationResult(is_duplicate=False)
```

去重的意义：防止系统"发明"一个其实已经存在的策略（只是换了个名字和表述）。`hidden_isomorphic` 检测尤其重要——LLM 生成的候选可能在表面描述上看起来全新，但其执行逻辑（SEG 图结构）与已有策略相同。如果候选与已有策略同构，不添加新策略，而是将空白区域的经验反馈给阶段二，让阶段二精化已有策略的适用条件。

---

## 3. 形式化一致性检验

### 3.1 什么是"一致性"

一个新策略与已有知识库一致，意味着：

1. **无直接矛盾：** 新策略的 favorable 条件不与任何已有策略的 unfavorable 条件在同一场景下发生逻辑冲突
2. **无行动冲突：** 在同一问题状态下，新策略推荐的行动不与一个公认有效的已有策略的推荐行动直接矛盾（除非新策略是后者的 alternative）
3. **拓扑可容纳：** 新策略的 Markov 核在策略空间中占据一个此前空白的位置，而非挤入一个已有策略密集的区域

### 3.2 一致性检验流程

```python
class ConsistencyChecker:
    """
    检验新策略候选与已有知识库的一致性。
    使用阶段三的形式化工具 + LLM prompting，不涉及模型训练。
    """
    
    def check(
        self,
        candidate: Dict,           # 候选策略（阶段零 schema 格式）
        candidate_kernel: np.ndarray,  # 候选策略的 Markov 核
        kb: KnowledgeBase,
        formal_kb: Dict[str, np.ndarray]
    ) -> ConsistencyReport:
        
        issues = []
        
        # 检验 1：条件矛盾检测
        condition_conflicts = self._check_condition_conflicts(
            candidate, kb
        )
        issues.extend(condition_conflicts)
        
        # 检验 2：行动冲突检测
        action_conflicts = self._check_action_conflicts(
            candidate_kernel, formal_kb
        )
        issues.extend(action_conflicts)
        
        # 检验 3：拓扑位置检查
        topology_check = self._check_topology(
            candidate_kernel, formal_kb
        )
        issues.extend(topology_check)
        
        # 检验 4：LLM 语义一致性审核
        semantic_check = self._llm_semantic_review(
            candidate, kb
        )
        issues.extend(semantic_check)
        
        # 汇总
        critical = [i for i in issues if i.severity == "critical"]
        warnings = [i for i in issues if i.severity == "warning"]
        
        if critical:
            verdict = "rejected"
        elif len(warnings) > 3:
            verdict = "needs_revision"
        else:
            verdict = "passed"
        
        return ConsistencyReport(
            verdict=verdict,
            issues=issues,
            critical_count=len(critical),
            warning_count=len(warnings)
        )
    
    def _check_condition_conflicts(
        self, candidate, kb
    ) -> List[ConsistencyIssue]:
        """
        检查候选策略的 favorable 条件是否与某个已有策略的
        unfavorable 条件在语义上高度相似。
        现在感知 Phase 2 的 stability_tier：与 foundational 条件
        冲突是 critical，与 tentative 条件冲突只是 info。
        """
        issues = []
        for cond in candidate["applicability_conditions"]["favorable"]:
            for sid in kb.get_all_strategy_ids():
                strategy = kb.load_strategy(sid)
                for unfav in strategy["applicability_conditions"]["unfavorable"]:
                    sim = compute_embedding_similarity(
                        cond["condition"], unfav["condition"]
                    )
                    if sim > 0.8:
                        # 根据冲突条件的稳定性分层决定严重程度
                        tier = unfav.get("stability_tier", "empirical")
                        if tier == "foundational":
                            severity = "critical"
                            desc_suffix = (
                                "此冲突涉及文献核心规则（foundational），"
                                "候选策略必须修改或放弃"
                            )
                        elif tier == "empirical":
                            severity = "warning"
                            desc_suffix = (
                                "此冲突涉及经验验证规则，"
                                "可能是合理的 alternative 关系"
                            )
                        else:  # tentative
                            severity = "info"
                            desc_suffix = (
                                "此冲突涉及待验证规则（tentative），"
                                "冲突方可能是错误的"
                            )
                        
                        issues.append(ConsistencyIssue(
                            type="condition_conflict",
                            severity=severity,
                            description=(
                                f"候选 favorable '{cond['condition']}' "
                                f"与 {sid} 的 unfavorable "
                                f"'{unfav['condition']}' 相似 "
                                f"(sim={sim:.2f}, tier={tier})。"
                                f"{desc_suffix}"
                            ),
                            related_strategy=sid
                        ))
        return issues
    
    def _check_action_conflicts(
        self, candidate_kernel, formal_kb
    ) -> List[ConsistencyIssue]:
        """
        检查在某个问题状态下，候选策略推荐的主行动
        是否与一个高成功率的已有策略推荐的主行动直接对立。
        """
        issues = []
        n_states = candidate_kernel.shape[0]
        
        for state_idx in range(n_states):
            cand_top = np.argmax(candidate_kernel[state_idx, :])
            
            for sid, K in formal_kb.items():
                exist_top = np.argmax(K[state_idx, :])
                # 检查是否是"对立行动"
                if self._are_opposing_actions(cand_top, exist_top):
                    exist_prob = K[state_idx, exist_top]
                    cand_prob = candidate_kernel[state_idx, cand_top]
                    if exist_prob > 0.4 and cand_prob > 0.4:
                        issues.append(ConsistencyIssue(
                            type="action_conflict",
                            severity="warning",
                            description=(
                                f"在状态 {state_idx}，候选推荐 "
                                f"'{ACTION_SPACE[cand_top]}' "
                                f"(p={cand_prob:.2f})，"
                                f"但 {sid} 强烈推荐对立行动 "
                                f"'{ACTION_SPACE[exist_top]}' "
                                f"(p={exist_prob:.2f})"
                            ),
                            related_strategy=sid
                        ))
        return issues
    
    def _check_topology(
        self, candidate_kernel, formal_kb
    ) -> List[ConsistencyIssue]:
        """
        检查候选策略在策略空间中的位置。
        如果它距离所有已有策略都非常近，可能是冗余的。
        如果它距离所有已有策略都非常远，可能是"幻觉策略"。
        """
        issues = []
        distances = []
        
        for sid, K in formal_kb.items():
            d = fisher_rao_distance(candidate_kernel, K)
            distances.append((sid, d))
        
        min_dist = min(d for _, d in distances)
        max_dist = max(d for _, d in distances)
        mean_dist = np.mean([d for _, d in distances])
        
        if min_dist < 0.05:
            nearest = min(distances, key=lambda x: x[1])
            issues.append(ConsistencyIssue(
                type="too_close",
                severity="warning",
                description=(
                    f"候选策略与 {nearest[0]} 的 Fisher 距离仅 "
                    f"{nearest[1]:.4f}，可能是冗余的"
                )
            ))
        
        if mean_dist > 2.0:
            issues.append(ConsistencyIssue(
                type="too_far",
                severity="warning",
                description=(
                    f"候选策略与所有已有策略的平均 Fisher 距离 "
                    f"为 {mean_dist:.2f}，远高于正常范围。"
                    f"可能是 LLM 幻觉生成的无意义策略"
                )
            ))
        
        return issues
```

### 3.3 LLM 语义一致性审核

数学检验可能遗漏语义层面的矛盾。用 LLM 做最终的语义审核：

```python
SEMANTIC_CONSISTENCY_PROMPT = """
你是一个哲学方法论专家。请审核以下候选新策略是否与已有的方法论体系一致。

## 候选策略
名称: {candidate_name}
描述: {candidate_description}
操作步骤: {candidate_steps}
假设的适用条件: {candidate_conditions}

## 已有策略体系摘要
{existing_strategies_summary}

## 审核标准
1. 候选策略的逻辑是否自洽（步骤之间不矛盾）？
2. 候选策略是否与某个已有策略在本质上相同（换了表述但核心一样）？
3. 候选策略是否包含任何明显不合理的假设？
4. 候选策略的适用条件是否合理（不过度宽泛也不过度狭窄）？

输出 JSON:
{{
    "self_consistent": true/false,
    "potential_duplicate_of": "策略 ID 或 null",
    "unreasonable_assumptions": ["..."],
    "condition_quality": "too_broad / reasonable / too_narrow",
    "overall_verdict": "accept / revise / reject",
    "revision_suggestions": ["..."],
    "reasoning": "..."
}}
"""
```

---

## 4. 经验验证

### 4.1 验证协议

通过一致性检验的候选策略进入经验验证阶段。验证的核心问题是：**这个候选策略在其声称适用的问题类型上，是否真的比已有策略表现更好？**

验证分四个阶段，前两个阶段利用阶段零点五的世界模型做廉价预筛，后两个阶段在真实环境中做严格验证。

```
候选策略 S_new
        │
        ▼
┌───────────────────┐
│  阶段 0: 世界模型    │  在世界模型中大量模拟
│  预筛 (WM Screen)  │  200+ 次模拟，零真实执行成本
└────────┬──────────┘
         │ 模拟成功率 > 25%
         ▼
┌───────────────────┐
│  阶段 1: 小规模验证 │  在 20-30 道精选任务上真实测试
│  (Pilot Test)      │  快速失败检测
└────────┬──────────┘
         │ 通过 (成功率 > 30%)
         ▼
┌───────────────────┐
│  阶段 2: 对比验证   │  与最佳已有策略 head-to-head
│  (Comparative)     │  在 50-100 道任务上
└────────┬──────────┘
         │ 优于 baseline (p < 0.05)
         ▼
┌───────────────────┐
│  阶段 3: 泛化验证   │  在未见过的领域/任务上测试
│  (Generalization)  │  确认不是过拟合
└────────┬──────────┘
         │ 通过
         ▼
    正式纳入知识库
```

### 4.2 阶段 0：世界模型预筛（WM Screening）

**目的：** 在不消耗任何真实执行成本的情况下，快速淘汰明显无效的候选策略。这是阶段零点五世界模型在阶段四中最重要的应用。

```python
def world_model_screening(
    candidate: Dict,
    gap: StrategyGap,
    world_model: HybridWorldModel,
    n_simulations: int = 200
) -> WMScreeningResult:
    """
    用世界模型大量模拟候选策略在空白区域的表现。
    成本：200 次模拟 × ~$0.01 = ~$2（如果走 LLM 模拟器）
    对比：200 次真实执行 × ~$0.10 = ~$20
    """
    # 从空白区域的特征组合中采样问题
    sampled_features = sample_features_from_gap(gap, n_simulations)
    
    predictions = []
    for features in sampled_features:
        wm_input = WorldModelInput(
            problem_features=features,
            strategy_id="candidate",
            strategy_summary=candidate["description"]["one_sentence"]
        )
        pred = world_model.predict(wm_input)
        predictions.append(pred)
    
    # 统计模拟成功率
    sim_success_rate = np.mean([
        p.predicted_success_probability for p in predictions
    ])
    
    # 与最佳已有策略的模拟对比
    baseline_predictions = []
    for features in sampled_features:
        wm_input = WorldModelInput(
            problem_features=features,
            strategy_id=gap.best_existing_strategy,
            strategy_summary=""
        )
        pred = world_model.predict(wm_input)
        baseline_predictions.append(pred)
    
    baseline_rate = np.mean([
        p.predicted_success_probability for p in baseline_predictions
    ])
    
    # 候选的模拟成功率应高于已有策略的模拟成功率
    sim_improvement = sim_success_rate - baseline_rate
    
    passed = sim_success_rate > 0.25 and sim_improvement > 0.0
    
    return WMScreeningResult(
        sim_success_rate=sim_success_rate,
        baseline_sim_rate=baseline_rate,
        sim_improvement=sim_improvement,
        passed=passed,
        n_simulations=n_simulations,
        # 记录世界模型预测的主要失败模式，供后续分析
        predicted_failure_modes=Counter(
            mode
            for p in predictions
            for mode in p.predicted_failure_modes
        ).most_common(5)
    )
```

**预筛阈值 25% 的理由：** 世界模型本身有预测偏差（通常偏乐观），所以模拟成功率 25% 大约对应真实成功率 15-20%。只有高于这个下限的候选才值得花真实执行的成本去验证。

**世界模型预筛的价值：** 如果候选生成阶段产生了 10 个候选，其中 7 个在世界模型预筛中被淘汰，那么后续的真实执行验证只需要在 3 个候选上进行——节省了 70% 的验证成本。

**世界模型预测偏差的补偿：** 如果世界模型的校准报告显示它对新策略有系统性乐观偏差（这很可能，因为新策略不在世界模型的训练分布内），预筛阈值应该上调。具体调整幅度基于阶段零点五的校准数据。

### 4.3 阶段 1：小规模验证（Pilot Test）

**目的：** 在真实环境中快速淘汰世界模型预筛的假阳性——那些在模拟中看起来可行但实际执行中失败的候选。

```python
def pilot_test(
    candidate: Dict,
    gap: StrategyGap,
    task_env: TaskEnvironment,
    llm_executor: LLMExecutor,
    n_tasks: int = 25
) -> PilotResult:
    """
    在策略空白区域的任务上快速测试候选策略。
    """
    # 从空白区域的特征组合中选取任务
    tasks = task_env.sample_tasks_by_features(
        gap.state, n_tasks
    )
    
    successes = 0
    trajectories = []
    
    for task in tasks:
        # 用候选策略的 operational_steps 构造 prompt
        strategy_prompt = build_strategy_prompt(candidate)
        outcome = task_env.execute_with_strategy(
            strategy_prompt, llm_executor
        )
        if outcome.success:
            successes += 1
        trajectories.append(outcome)
    
    success_rate = successes / n_tasks
    
    return PilotResult(
        success_rate=success_rate,
        passed=success_rate > 0.30,
        trajectories=trajectories,
        n_tasks=n_tasks
    )
```

**淘汰阈值 30% 的理由：** 空白区域的定义是"最好的已有策略成功率 < 40%"。如果候选策略连 30% 都达不到，它不太可能在更大规模的测试中超越已有策略。

### 4.4 阶段 2：对比验证（Comparative Test）

```python
def comparative_test(
    candidate: Dict,
    gap: StrategyGap,
    task_env: TaskEnvironment,
    llm_executor: LLMExecutor,
    kb: KnowledgeBase,
    n_tasks: int = 80
) -> ComparativeResult:
    """
    将候选策略与该空白区域的最佳已有策略进行对比。
    """
    tasks = task_env.sample_tasks_by_features(gap.state, n_tasks)
    
    candidate_results = []
    baseline_results = []
    
    for task in tasks:
        # 候选策略执行
        cand_prompt = build_strategy_prompt(candidate)
        cand_outcome = task_env.execute_with_strategy(
            cand_prompt, llm_executor
        )
        candidate_results.append(cand_outcome.success)
        
        # 最佳已有策略执行（同一任务）
        baseline_strategy = kb.load_strategy(gap.best_existing_strategy)
        base_prompt = build_strategy_prompt(baseline_strategy)
        base_outcome = task_env.execute_with_strategy(
            base_prompt, llm_executor
        )
        baseline_results.append(base_outcome.success)
    
    # 配对统计检验
    from scipy.stats import mcnemar
    
    # 构造 2x2 列联表
    both_success = sum(c and b for c, b in zip(candidate_results, baseline_results))
    cand_only = sum(c and not b for c, b in zip(candidate_results, baseline_results))
    base_only = sum(not c and b for c, b in zip(candidate_results, baseline_results))
    both_fail = sum(not c and not b for c, b in zip(candidate_results, baseline_results))
    
    table = [[both_success, base_only], [cand_only, both_fail]]
    stat_result = mcnemar(table, exact=True)
    
    cand_rate = sum(candidate_results) / n_tasks
    base_rate = sum(baseline_results) / n_tasks
    
    return ComparativeResult(
        candidate_rate=cand_rate,
        baseline_rate=base_rate,
        improvement=cand_rate - base_rate,
        p_value=stat_result.pvalue,
        passed=(cand_rate > base_rate and stat_result.pvalue < 0.05),
        n_tasks=n_tasks
    )
```

### 4.5 阶段 3：泛化验证

对比验证仅在策略空白区域的任务上进行。泛化验证检查候选策略是否在非空白区域也有价值——或者至少不会造成危害。

```python
def generalization_test(
    candidate: Dict,
    task_env: TaskEnvironment,
    llm_executor: LLMExecutor,
    n_tasks_per_domain: int = 20
) -> GeneralizationResult:
    """
    在多个领域上测试候选策略，检查泛化能力。
    """
    results_by_domain = {}
    
    for domain in task_env.get_all_domains():
        tasks = task_env.sample_tasks_by_domain(domain, n_tasks_per_domain)
        successes = sum(
            1 for task in tasks
            if task_env.execute_with_strategy(
                build_strategy_prompt(candidate),
                llm_executor
            ).success
        )
        results_by_domain[domain] = successes / n_tasks_per_domain
    
    # 候选策略在非目标领域上不应该有灾难性差表现
    min_rate = min(results_by_domain.values())
    
    return GeneralizationResult(
        domain_rates=results_by_domain,
        min_domain_rate=min_rate,
        passed=min_rate >= 0.15,  # 即使在最差领域也不应低于 15%
    )
```

### 4.6 候选策略的迭代修正

如果候选策略未通过验证但表现出部分潜力（例如 pilot 成功率在 20-30% 之间），不直接丢弃，而是尝试修正。

```python
STRATEGY_REVISION_PROMPT = """
候选策略 "{candidate_name}" 在验证中表现不佳。

## 验证结果
成功率: {success_rate}%
典型的失败案例:
{failure_examples}

## 分析
策略的操作步骤中，步骤 {step_number} 最常关联到失败:
{step_analysis}

## 修正任务
请修改策略的操作步骤，使其更好地应对验证中暴露的问题。
要求：
1. 保留策略的核心思路（不要改成一个完全不同的策略）
2. 只修改与失败关联度最高的步骤
3. 添加对失败条件的显式处理

输出修改后的 operational_steps（JSON 数组）。
"""
```

最多允许 2 轮修正。如果 2 轮修正后仍未通过 pilot test，该候选被正式丢弃。

---

## 5. 知识库集成

### 5.1 集成流程

通过全部三阶段验证的候选策略按以下流程集成到知识库中。

```python
def integrate_new_strategy(
    candidate: Dict,
    candidate_kernel: np.ndarray,
    validation_results: ValidationResults,
    kb: KnowledgeBase,
    formal_kb: Dict[str, np.ndarray],
    iso_detector: IsomorphismDetector
):
    """
    将验证通过的新策略集成到知识库。
    """
    # 1. 分配正式 ID
    new_id = kb.allocate_next_id()  # 如 "S21"
    candidate["id"] = new_id
    
    # 2. 更新元数据
    candidate["metadata"].update({
        "version": "1.0",
        "created": datetime.utcnow().isoformat(),
        "confidence": "low",    # 新策略初始置信度为 low
        "completeness": "low",
        "source": "phase4_generated",
        "validation_results": {
            "pilot_success_rate": validation_results.pilot.success_rate,
            "comparative_improvement": validation_results.comparative.improvement,
            "comparative_p_value": validation_results.comparative.p_value,
            "generalization_min_rate": validation_results.generalization.min_domain_rate
        },
        "total_experience_records": validation_results.total_tasks,
        "successful_applications": validation_results.total_successes,
        "failed_applications": validation_results.total_failures
    })
    
    # 3. 适用条件状态更新
    for cond in (candidate["applicability_conditions"]["favorable"] +
                 candidate["applicability_conditions"]["unfavorable"]):
        cond["status"] = "active"
        cond["source"] = "phase4_hypothesized"
        # 置信度基于验证结果调整
        cond["confidence"] = min(0.7, validation_results.pilot.success_rate)
    
    # 4. 计算与所有已有策略的关系
    relationships = []
    for sid, K in formal_kb.items():
        report = iso_detector._analyze_pair(
            new_id, candidate_kernel, sid, K
        )
        if report.relationship != "independent":
            relationships.append({
                "related_strategy": sid,
                "relationship_type": _map_to_kb_type(report.relationship),
                "description": f"形式化分析检测到的关系",
                "formal_evidence": report.distances
            })
    candidate["relationships_to_other_strategies"] = relationships
    
    # 5. 保存策略文件
    kb.save_strategy(candidate)
    
    # 6. 更新形式化知识库
    formal_kb[new_id] = candidate_kernel
    save_kernel(new_id, candidate_kernel)
    
    # 7. 写入变更历史
    write_change_history(
        new_id,
        change_type="initial_creation",
        author="phase4_auto",
        changes="新策略由阶段四自动生成并通过验证",
        evidence_refs=[
            r.execution_id
            for r in validation_results.all_records
        ]
    )
    
    # 8. 更新调度器的动作空间
    # 在下次调度器加载知识库时自动生效
    
    # 9. 通知阶段二：新策略已添加，进入经验监控
    # 阶段二的健康监控将跟踪新策略的后续表现
```

### 5.2 新策略的"试用期"——与 Phase 2 生命周期管理统一

新策略在集成后处于"试用期"，其管理**统一纳入 Phase 2 的策略生命周期管理框架**（Phase 2 第 5 节），而非独立的试用期系统：

- 新策略的 `stability_tier` 初始化为 `tentative`（Phase 2 的 1.3 节）
- Phase 2 的健康监控对 `tentative` 策略自动使用更严格的阈值（成功率下降 15% 即触发警告）
- 在 `tentative` 阶段，调度器选择新策略时必须附带一个 `backup_strategy`
- 升级路径：`tentative` → `empirical`（需要 30+ 条经验且成功率 > 60%，见 Phase 2 的 STABILITY_TIERS 定义）
- 退役路径：如果 50+ 次使用中成功率 < 20%，由 Phase 2 的 `check_strategy_retirement()` 自动建议退役

**Phase 4 不再维护独立的试用期逻辑。** 所有策略（无论来源）的生命周期由 Phase 2 统一管理。Phase 4 的职责在策略通过验证并写入知识库后结束——后续的监控、升级、退役全部由 Phase 2 接管。

---

## 6. 数据驱动的策略发现——补充 LLM 生成

### 6.1 为什么需要数据驱动

Claude.md 讨论中的一个关键洞察：LLM 读过所有哲学教科书——它"发明"的策略很可能只是对已读内容的重组，表述变了但本质没变。真正新颖的策略应该来自**执行数据中的模式发现**——发现"在某类问题上，执行器倾向于采取一种没有名字的行为模式，且这个模式的成功率高于所有已命名策略"。

这是数据挖掘，不是文本生成。

### 6.2 执行轨迹聚类——发现无名策略

```python
class DataDrivenStrategyDiscoverer:
    """
    从执行轨迹中发现尚未被命名的有效行为模式。
    补充 LLM 生成的候选——LLM 重组已有知识，
    数据驱动发现真正新颖的模式。
    """
    
    def discover(
        self,
        experience_log: List[ExecutionRecord],
        action_space: List[str],
        min_cluster_size: int = 10,
        min_success_rate: float = 0.5
    ) -> List[DiscoveredPattern]:
        
        # 1. 提取所有成功执行的行动序列
        success_sequences = []
        for record in experience_log:
            if record["outcome"]["success"]:
                actions = extract_action_sequence(
                    record["execution_trajectory"],
                    action_space
                )
                success_sequences.append({
                    "actions": actions,
                    "features": record["task"]["complexity_features"],
                    "strategy_used": record["strategy_selection"]["selected_strategy"]
                })
        
        # 2. 对行动序列做聚类（用编辑距离）
        from sklearn.cluster import DBSCAN
        distance_matrix = compute_edit_distance_matrix(
            [s["actions"] for s in success_sequences]
        )
        clusters = DBSCAN(
            eps=0.3, min_samples=min_cluster_size,
            metric="precomputed"
        ).fit(distance_matrix)
        
        # 3. 找出"不属于任何已有策略"的成功聚类
        patterns = []
        for cluster_id in set(clusters.labels_):
            if cluster_id == -1:
                continue  # 噪声
            
            members = [
                success_sequences[i]
                for i in range(len(success_sequences))
                if clusters.labels_[i] == cluster_id
            ]
            
            # 检查这个聚类是否与某条已有策略高度重叠
            strategy_distribution = Counter(
                m["strategy_used"] for m in members
            )
            dominant_strategy = strategy_distribution.most_common(1)[0]
            dominant_ratio = dominant_strategy[1] / len(members)
            
            if dominant_ratio < 0.5:
                # 这个成功模式不属于任何单一已有策略——
                # 可能是一种尚未被命名的新策略！
                
                # 提取该聚类的典型行动序列
                typical_sequence = find_medoid(
                    [m["actions"] for m in members],
                    distance_matrix
                )
                
                # 提取该聚类的共同问题特征
                common_features = find_common_features(
                    [m["features"] for m in members]
                )
                
                patterns.append(DiscoveredPattern(
                    cluster_size=len(members),
                    success_rate=len(members) / count_all_attempts(
                        experience_log, common_features
                    ),
                    typical_action_sequence=typical_sequence,
                    common_problem_features=common_features,
                    strategy_distribution=dict(strategy_distribution),
                    # 用 LLM 给这个模式命名和描述
                    name=None,  # 待 LLM 命名
                    description=None  # 待 LLM 描述
                ))
        
        return [p for p in patterns if p.success_rate >= min_success_rate]
    
    def name_discovered_pattern(
        self, pattern: DiscoveredPattern
    ) -> NamedPattern:
        """
        用 LLM 为数据驱动发现的无名模式命名和结构化。
        注意：LLM 的角色是命名和描述，不是发明——
        模式本身来自数据，LLM 只是给它一个名字。
        """
        response = llm_call(NAME_PATTERN_PROMPT.format(
            typical_sequence=pattern.typical_action_sequence,
            common_features=pattern.common_problem_features,
            success_rate=pattern.success_rate,
            strategy_distribution=pattern.strategy_distribution
        ))
        return NamedPattern(
            pattern=pattern,
            name=response["name"],
            description=response["description"],
            operational_steps=response["operational_steps"],
            # 数据驱动发现的策略标记来源
            source="data_driven_discovery"
        )
```

**与 LLM 生成的关系：** 数据驱动发现和 LLM 生成是互补的两条路径。两者的候选都进入同一个去重→一致性检验→验证管线。但它们在性质上不同：

| 维度 | LLM 生成 | 数据驱动发现 |
|------|---------|------------|
| 创新来源 | 重组 LLM 训练数据中的已有方法论 | 从执行数据中发现无名的新行为模式 |
| 新颖性 | 可能是旧酒新瓶（换表述） | 真正来自实践的新模式（如果存在的话） |
| 风险 | 候选看起来新但实质重复 | 候选可能只是噪声不是真正的模式 |
| 需要的数据量 | 少（LLM 零样本） | 多（需要 500+ 成功执行记录才能聚类） |

**启动条件：** 数据驱动发现只在经验日志积累了 500+ 条成功执行记录后才启动。在此之前，Phase 4 完全依赖 LLM 生成。

---

## 7. 技术实现

### 6.1 项目文件结构

```
assumption_agent/
├── ...                              # 阶段一二三的所有目录保持不变
├── hypothesis/                      # 阶段四新增目录
│   ├── gap_detector/
│   │   ├── detector.py              # 策略空白检测
│   │   ├── gap_classifier.py        # 空白分类
│   │   └── coverage_analyzer.py     # 基于形式化拓扑的覆盖分析
│   ├── generator/
│   │   ├── gap_analyzer.py          # 空白分析
│   │   ├── candidate_generator.py   # 候选策略生成
│   │   ├── structurer.py            # 结构化为知识库 schema
│   │   ├── deduplicator.py          # 去重（调用阶段三）
│   │   └── prompts.py               # 所有 LLM prompt 模板
│   ├── consistency/
│   │   ├── checker.py               # 一致性检验主模块
│   │   ├── condition_conflict.py    # 条件矛盾检测
│   │   ├── action_conflict.py       # 行动冲突检测
│   │   ├── topology_check.py        # 拓扑位置检查
│   │   └── prompts.py
│   ├── validation/
│   │   ├── pilot_test.py            # 小规模验证
│   │   ├── comparative_test.py      # 对比验证
│   │   ├── generalization_test.py   # 泛化验证
│   │   ├── revision.py              # 候选修正
│   │   └── protocol.py              # 验证协议（串联三阶段）
│   ├── integration/
│   │   ├── integrator.py            # 知识库集成
│   │   └── probation.py             # 试用期管理
│   ├── pipeline.py                  # 完整管线
│   └── config.py
├── scripts/
│   ├── ...
│   ├── detect_gaps.py               # 运行空白检测
│   ├── generate_candidates.py       # 生成候选策略
│   ├── validate_candidate.py        # 验证候选策略
│   └── integrate_strategy.py        # 集成新策略
└── tests/
    ├── ...
    ├── test_gap_detector.py
    ├── test_generator.py
    ├── test_consistency.py
    ├── test_validation.py
    └── test_integration.py
```

### 6.2 LLM 调用成本估算

| 操作 | 频率 | 每次 token 消耗 | 模型 | 估算成本 |
|------|------|----------------|------|---------|
| 空白分析 | 每个空白 1 次 | ~2K in + 500 out | gpt-4o | ~$0.10 |
| 候选生成 | 每个空白 1 次 | ~3K in + 2K out | gpt-4o | ~$0.20 |
| 结构化 | 每个候选 1 次 | ~2K in + 3K out | gpt-4o | ~$0.20 |
| 语义一致性审核 | 每个候选 1 次 | ~4K in + 1K out | gpt-4o | ~$0.20 |
| 候选修正 | ~30% 候选需修正 | ~2K in + 1K out | gpt-4o | ~$0.10 |
| **每个空白的总成本** | | | | **~$1-3** |
| **假设发现 5-10 个空白** | | | | **~$10-30** |

验证阶段的主要成本是 LLM 执行器的调用（与阶段一相同），不是本阶段的额外成本。

### 6.3 与其他阶段的接口

| 方向 | 数据 | 说明 |
|------|------|------|
| 阶段一 → 四 | 低置信度信号 | 调度器在哪些问题上"不确定" |
| 阶段一 → 四 | 经验日志 | 空白检测的成功率统计 |
| 阶段二 → 四 | 变更历史 | 频繁失败的策略可能暗示空白 |
| 阶段三 → 四 | 形式化拓扑 | 覆盖分析 + 去重 + 一致性检验 |
| 阶段四 → 零 | 新策略 JSON | 写入 `kb/strategies/` |
| 阶段四 → 三 | 新策略的 Markov 核 | 写入 `formal_kb/kernels/` |
| 阶段四 → 二 | 新策略的经验记录 | 验证阶段的执行记录写入经验日志 |
| 阶段四 → 一 | 扩展的动作空间 | 调度器下次加载时自动获取 |

---

## 8. 实验设计

### 7.1 核心实验：系统能否发现有价值的新策略

**假设 H1：** 系统生成的新策略中，至少有一条在其目标空白区域上的表现显著优于所有已有策略。

**实验设计：**
- 运行完整管线：空白检测 → 候选生成 → 一致性检验 → 三阶段验证
- 记录每个候选的全流程结果
- 对通过验证的新策略，报告其在空白区域上相对于最佳已有策略的改进幅度

### 7.2 核心实验：新策略的人类评审

**假设 H2：** 系统生成的新策略中，至少有一条被人类专家确认为"有价值的、此前未被显式提出的方法论洞察"。

**实验设计：**
- 将通过验证的所有新策略提交给 5 名领域专家（覆盖 CS、数学、自然科学、工程、哲学）
- 每名专家独立评审，评估维度：
  - 新颖性（0-5）：此策略在方法论文献中是否前所未见？
  - 合理性（0-5）：此策略的逻辑是否自洽且有说服力？
  - 实用性（0-5）：如果你知道此策略，是否会在实际问题中使用它？
  - 表述清晰度（0-5）：操作步骤是否足够清晰到可以执行？
- **目标：** 至少一条新策略的平均新颖性 ≥ 3.0 且合理性 ≥ 3.5

### 7.3 关键实验：新策略集成后的系统性能

**假设 H3：** 将新策略集成到知识库后，调度器在全量测试集上的综合任务完成率提升 ≥ 3%。

**实验设计：**
- 对比条件 A：仅含阶段零原始策略的知识库（KB v1.0）
- 对比条件 B：经过阶段二演化的知识库（KB v1.x）
- 对比条件 C：含阶段四新策略的知识库（KB v2.0）
- 在全量测试集上运行调度器（调度器可能需要用 KB v2.0 重新训练或微调）
- **注意：** 新策略主要在空白区域有贡献，在非空白区域应保持不变。因此全量提升 3% 对应空白区域的大幅提升

### 7.4 分析实验

**分析 1：候选淘汰漏斗**
- 记录每个环节的淘汰率：去重淘汰了多少？一致性检验淘汰了多少？Pilot 淘汰了多少？
- 分析哪个环节的淘汰率最高——揭示 LLM 在策略生成上的主要弱点

**分析 2：生成多样性**
- 对同一个空白，LLM 生成的 3 个候选之间的 Fisher 距离分布
- 多样性是否足够？是否存在"模式坍缩"（3 个候选其实很相似）

**分析 3：新策略的来源分析**
- 新策略的灵感来源分布（科学、哲学、工程、日常生活等）
- LLM 是否倾向于从某些领域借鉴？这种倾向是否合理？

---

## 9. 风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| 没有检测到真正的空白 | 中 | 高 | 降低空白阈值（从 0.4 到 0.3）；增加任务多样性；检查是否是任务集覆盖不足而非策略不足 |
| LLM 生成的候选全部是已有策略的变形 | 高 | 高 | 去重环节过滤；prompt 中显式要求"新思路"并惩罚与已有策略的重复；提供更抽象的灵感来源 |
| 候选策略通过一致性检验但经验验证全部失败 | 高 | 中 | 增加候选修正轮次；分析失败模式寻找系统性问题；如果反复失败说明 LLM 在方法论创新上的能力有限——这本身是一个有价值的研究发现 |
| 新策略在空白区域有效但污染非空白区域 | 中 | 中 | 泛化验证环节检查非空白区域不退化；试用期机制限制新策略的影响范围 |
| 验证成本过高（每个候选需要 100+ LLM 调用） | 中 | 中 | Pilot test 快速淘汰无效候选；对 pilot 成功率 < 20% 的候选直接淘汰 |
| 人类专家对新策略的评审结果不一致 | 中 | 中 | 增加专家人数；对评审不一致的策略做深入讨论会 |
| 形式化一致性检验阈值设置不当 | 中 | 中 | 先在已知的策略关系上校准阈值；灵敏度分析 |
| 整个阶段的核心前提（LLM 能发明新方法论）不成立 | 低 | 极高 | 即使失败也是有价值的结论——它精确标定了当前 AI 系统在方法论创新上的能力边界；论文可以转向分析失败模式 |

---

## 10. 增量开发计划

### Step 1：空白检测（第 1-3 周）

**目标：** 从阶段一二的执行经验中检测策略空白。

- 实现基于经验的空白检测器
- 实现基于阶段三形式化拓扑的覆盖分析
- 实现空白分类器
- **验证：** 在现有数据上识别出至少 3 个 TRUE_GAP 类型的空白

### Step 2：候选生成（第 4-7 周）

**目标：** 为每个空白生成候选新策略。

- 实现空白分析 prompt
- 实现候选生成 prompt
- 实现结构化和去重
- **验证：** 对 3 个空白各生成 3 个候选，去重后剩余 ≥ 5 个独特候选

### Step 3：一致性检验（第 8-10 周）

**目标：** 对所有候选进行一致性检验。

- 实现四层一致性检验（条件矛盾、行动冲突、拓扑位置、语义审核）
- **验证：** 至少 3 个候选通过一致性检验

### Step 4：经验验证（第 11-16 周）

**目标：** 对通过一致性检验的候选进行三阶段验证。

- 实现 pilot test / comparative test / generalization test
- 实现候选修正流程
- **验证：** 至少 1 个候选通过全部三阶段验证

### Step 5：集成与试用期（第 17-19 周）

**目标：** 将验证通过的新策略集成到知识库。

- 实现集成流程
- 实现试用期管理
- 用更新后的知识库重新评估调度器性能
- **验证：** 新策略集成后，空白区域的成功率提升可测量

### Step 6：评估与论文（第 20-24 周）

**目标：** 运行所有实验，完成论文。

- H1-H3 实验
- 人类专家评审
- 候选淘汰漏斗分析
- 论文写作

---

## 11. 完成标准（Definition of Done）

阶段四在以下所有条件同时满足时视为完成：

1. 系统在阶段一二的执行经验中检测到至少 3 个真正的策略空白
2. 系统为每个空白生成了至少 2 个通过去重的独特候选策略
3. 至少 1 个候选策略通过了完整的三阶段验证（pilot + comparative + generalization）
4. 通过验证的新策略在其目标空白区域上的成功率显著优于最佳已有策略（p < 0.05）
5. 至少 1 个新策略被人类专家确认为"有价值的、此前未被显式提出的方法论洞察"（新颖性 ≥ 3.0 且合理性 ≥ 3.5）
6. 新策略已正式集成到知识库，并通过 schema 验证
7. 新策略的形式化 Markov 核已加入 formal_kb，与已有策略的距离和关系已计算
8. 新策略进入试用期后，在 100 次使用中成功率保持 ≥ 40%（或做出降级/移除决策）
9. 完整的候选淘汰漏斗数据已记录（记录每个环节淘汰了多少候选及其原因）
10. 完成一篇论文初稿

---

## 附录 A：Fine-Tune 替代方案

### A.1 为什么默认不 fine-tune

本阶段的核心操作在默认方案中全部通过 prompting + 数学计算实现：

| 操作 | 默认实现方式 | 不 fine-tune 的理由 |
|------|------------|-------------------|
| 空白检测 | 规则函数（经验统计 + 阈值） | 纯数值计算，不涉及 ML |
| 候选生成 | LLM prompting | 策略发明是开放式创造任务，通用 LLM 的多样性是优势 |
| 一致性检验 | 阶段三数学工具 + LLM prompting | 数学部分无需 ML，语义审核用通用 LLM 足够 |
| 经验验证 | 阶段一的执行框架 | 复用已有组件 |

### A.2 场景一：假设生成器的专用训练

**触发条件：** LLM 生成的候选策略在去重环节中 > 70% 被淘汰为已有策略的变形（LLM 无法跳出已有模式），或在一致性检验中 > 80% 被淘汰为逻辑不自洽。

**方案：** 训练一个专用的假设生成器，使其比通用 LLM 更擅长生成结构化、一致、新颖的策略候选。

**训练数据构造：**

这是最大的挑战——"好的新策略"的训练数据从何而来？

```python
HYPOTHESIS_GENERATOR_FINETUNE_CONFIG = {
    "base_model": "Qwen2.5-7B-Instruct",
    "method": "LoRA",
    "lora_rank": 16,
    
    "training_data_sources": {
        # 来源 1: 阶段零知识库中的已有策略
        # 模拟"从空白到策略"的过程
        "existing_strategies": {
            "method": "leave_one_out",
            # 每次去掉一条策略，用其余策略定义"已有知识库"
            # 训练目标是从空白特征中"重新发明"被去掉的策略
            "n_samples": 20,  # 每条策略 1 个样本
        },
        
        # 来源 2: 跨领域方法论迁移
        # 从设计模式、编程范式等领域提取方法论
        # 训练目标是将其翻译为本系统的策略 schema
        "cross_domain": {
            "sources": [
                "software_design_patterns",    # GoF 设计模式
                "cognitive_biases_debiasing",   # 认知偏误的纠偏策略
                "military_strategy",            # 军事战略原则
                "negotiation_tactics",          # 谈判策略
            ],
            "n_samples_per_source": 10,
        },
        
        # 来源 3: 系统自身成功的候选（bootstrapping）
        # 在 prompting 方案运行一段时间后，
        # 收集通过验证的候选作为正例
        "validated_candidates": {
            "min_samples": 5,  # 至少 5 条才启动
            "quality_filter": "passed_all_3_stages",
        },
    },
}
```

**Leave-one-out 训练的详细设计：**

```python
def generate_leave_one_out_data(kb: KnowledgeBase) -> List[Dict]:
    """
    对知识库中的每条策略，构造一个 (空白描述 → 策略) 的训练样本。
    """
    training_data = []
    all_strategies = kb.get_all_strategies()
    
    for target in all_strategies:
        # 构造"假装 target 不存在"的场景
        remaining = [s for s in all_strategies if s["id"] != target["id"]]
        
        # 找到 target 策略最擅长的问题状态
        best_states = find_dominant_states(target, remaining)
        
        for state in best_states:
            training_data.append({
                "input": {
                    "existing_strategies": [
                        summarize_strategy(s) for s in remaining
                    ],
                    "gap_description": describe_state(state),
                    "failing_strategies": find_failing_in_state(
                        remaining, state
                    ),
                    "desired_properties": infer_properties(target)
                },
                "output": {
                    "generated_strategy": format_as_candidate(target)
                }
            })
    
    return training_data
```

**风险：** Leave-one-out 数据只有 20 条（每条策略 1 条），远不够训练。需要配合跨领域数据和数据增强。即便如此，fine-tune 方案在"发明新策略"这种开放式任务上的价值不确定——可能反而导致模式坍缩（所有候选都像训练数据中的策略）。

### A.3 场景二：用 RL 优化生成质量

**触发条件：** 候选生成的淘汰漏斗中，一致性检验和经验验证的通过率都很低（< 20%），但分析发现失败模式有规律可循（如"总是在步骤 3 出现逻辑跳跃"）。

**方案：** 将候选生成建模为 RL 问题——策略是"如何生成策略"，奖励来自验证结果。

```python
GENERATOR_RL_CONFIG = {
    "base_model": "hypothesis_generator (A.2 的输出)",
    "rl_algorithm": "PPO",
    
    "reward_design": {
        "passed_dedup": 0.2,            # 通过去重
        "passed_consistency": 0.3,       # 通过一致性检验
        "passed_pilot": 0.5,             # 通过 pilot test
        "passed_comparative": 0.8,       # 通过对比验证
        "passed_generalization": 1.0,    # 通过全部验证
        "rejected_at_dedup": -0.1,       # 去重失败
        "rejected_inconsistent": -0.2,   # 一致性失败
    },
    
    "training": {
        "episodes": 500,                 # 每个 episode = 生成 1 个候选并走完验证
        "max_revisions_per_episode": 2,
    },
    
    # 关键约束: 每个 episode 的成本约 $1-3
    # 500 episodes ≈ $500-1500
    "budget_cap": 2000,
}
```

**风险评估：** 这个方案的成本极高（$500-1500），且 RL 在如此稀疏的奖励信号（每个 episode 耗时数小时才得到一个标量奖励）下很可能无法收敛。**不推荐在本阶段启用。** 仅在系统已稳定运行 6+ 个月、积累了大量候选生成-验证数据之后考虑。

### A.4 启用决策

```
prompting 方案运行 3+ 个月
            │
            ▼
候选去重淘汰率 > 70%?
    │
    是 → 启用 A.2（假设生成器 fine-tune）
    否 ↓

一致性/验证通过率 < 20% 且失败模式有规律?
    │
    是 → 先尝试优化 prompt，如仍无效 → 考虑 A.3（RL 优化）
    否 → 保持 prompting 方案
```

### A.5 Fine-tune 方案对项目结构的影响

```
assumption_agent/
├── hypothesis/
│   ├── ...                          # 现有目录不变
│   └── finetuned_models/
│       ├── hypothesis_generator/    # A.2: 假设生成器
│       │   ├── prepare_loo_data.py  # leave-one-out 数据准备
│       │   ├── prepare_xdomain.py   # 跨领域数据准备
│       │   ├── train.py
│       │   └── config.yaml
│       └── generator_rl/           # A.3: RL 优化
│           ├── reward_model.py      # 奖励函数
│           ├── train_rl.py
│           └── config.yaml
```