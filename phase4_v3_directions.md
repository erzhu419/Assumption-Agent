# Phase 4 v3 方向图 — 从"静态 scaffold"走向"自主成长 agent"

写于 2026-04-23，v20 实验跑完（等结果），v19b +14pp 是当前最高。**核心问题**: 静态架构已经 squeeze 得差不多了，第二篇 ("recursive self-hypothesizing") 需要的是**闭环驱动的 library 演化**。

---

## 基线：我们已经 possess 的开环组件

```
┌─────────────────────────────────────────────────┐
│  开环组件 (都单独 work)                            │
├─────────────────────────────────────────────────┤
│  ✅ Residual 识别 (residual_analyzer.py)         │
│  ✅ 新 wisdom 提案 (propose_new_wisdoms_mode_b)   │
│  ✅ Novelty check (embedding)                    │
│  ✅ Exemplar 挖掘 (build_diverse_exemplars)      │
│  ✅ A/B judge (cached_framework.run_judge)        │
│  ✅ v19b 动态 reframing (Turn 0 rewrite)         │
└─────────────────────────────────────────────────┘
         │ 缺失
         ▼
┌─────────────────────────────────────────────────┐
│  闭环 orchestration (必须补)                       │
│    把上面 6 个组件连成自主 loop                     │
└─────────────────────────────────────────────────┘
```

---

## 方向 1 — Minimal Closed Loop (~1 周)

**目标**: 最简版本，证明"agent 能自己改善自己"这件事真能发生

### 架构
```
每 20 问题 = 1 个 learning round:
  1. v16/v19b 解 20 个新问题 → 存 (problem, draft, final, judge_reason)
  2. 自动 residual_analyzer 扫描：有没有 ≥5 个失败在同一 cluster_tag？
     - 若没有 → 跳过这轮（纯 exploit）
     - 若有 → 触发 Step 3 (explore)
  3. Mode B mining 从那些 residual 提 1-2 个 candidate wisdom
  4. Novelty check 过滤
  5. A/B: 对下一轮 20 题并行跑 (with W_new) vs (without)
  6. 若 +5pp → append 到 library，否则 discard
  7. 记录 library 版本号 + 此次加/减的 wisdom
```

### Deliverable
- `autonomous_loop.py` 跑 200-300 个问题
- `library_evolution.json` 显示: V75 → V76 → V77 → ... 每次加的 wisdom 和当时的 gain
- 一条 performance curve: 每 round 结束的 held-out wr vs round 编号

### 论文核心 claim
> "Without human intervention, the system grew its wisdom library from 75 → K entries over N problems, each addition passing held-out A/B validation. The library's held-out win rate rose monotonically from 54% to Y%."

### 难度 / 风险
- 难度: 中等（组件都有，只缺 orchestration）
- 风险 1: LLM judge 的 noise 可能 mask +5pp 信号，需 larger held-out (至少 30 题/次)
- 风险 2: 可能 N 个 round 都没触发 Step 3 (一直 exploit)。需要 explore bonus 机制
- 风险 3: 新加的 wisdom 可能在 round K 有效，round K+10 变成噪声 (distribution shift)。需要 revert 机制

---

## 方向 2 — Reframing Distillation (~2 周)

**目标**: 从 v19b 每题的 Turn 0 rewrite 里**反向蒸馏**出 stable wisdom

### 关键 insight
Mode B 从**失败**里找新 orientation → 容易歪。
v19b 的 Turn 0 rewrite **是成功模式的痕迹** → 它展示了 LLM 判定"这题真正该怎么看"。大量 rewriting 里**重复出现的 reframing pattern 就是 stable orientation**。

### 架构
```
1. 跑 v19b on 500 问题 → 500 条 {problem, rewritten_problem, what_changed}
2. 对 500 条 what_changed 做 sentence-transformer encoding → HDBSCAN 聚类
3. 每个 stable cluster (n≥8, silhouette>0.3) →
   GPT-5.4: "这 8 条 rewriting 共同体现了什么 orientation？用 wisdom schema 形式化"
4. Novelty check vs 现有 library
5. 每条 candidate 做 30 题 A/B 验证
6. 保留 gain 的
```

### Deliverable
- 500 问题的 v19b rewrite corpus
- cluster tree 可视化
- 5-10 条 data-driven wisdom
- Held-out 验证后的 keep/revert ledger

### 和方向 1 的关系
- 方向 1 是**从 failure 驱动**（residual → new wisdom）
- 方向 2 是**从 success 驱动**（rewriting pattern → new wisdom）
- 两者可以融合: loop 里两种信号都触发 candidate generation

### 难度 / 风险
- 难度: 中等
- 风险: v19b rewriting quality 方差大，clustering 可能找到的是 surface pattern 而非 deep orientation
- 优点: **这是 Mode B 之外完全新的 signal source** —— 其他论文没做过这个

---

## 方向 3 — Meta-Reasoning Layer (~1-2 月)

**目标**: 让 agent **对自己的推理模式有 understanding**

### 架构
```
每 50 题 → 触发 self-audit:
  Prompt: "过去 50 题里:
    1. 哪些 pattern 你反复出错？
    2. 哪些 wisdom 你激活但没真正 deliver 结果？(over-used)
    3. 哪些情境你漏激活？(under-used)
    输出 JSON"

Self-audit 输出驱动 3 个后续动作:
  - failure_patterns → Mode B mining
  - over_used → 评估是否 refine signal/unpacked (Mode A')
  - under_used → 评估 selection 层是否需要调整
```

### Deliverable
- 系统写出的 self-audit report (paper 里可以直接引用原文)
- Self-audit → intervention → performance 的 causal chain
- 对比: 有 self-audit vs 无 self-audit 的 learning speed

### 论文 claim 升级
> "The system demonstrates rudimentary **metacognitive awareness** — it can correctly identify its own failure modes and propose corrections that improve performance."

这 phrasing 直接进**顶会 (NeurIPS / ICLR) 的 Agent track**。

### 难度 / 风险
- 难度: 高。Self-audit 的 faithfulness 要验证 (LLM 说"我在 X 上弱"，是真的还是 plausible 谎言？)
- 风险: LLM 对自己的 introspection 众所周知 unreliable。可能产出好听但无用的 self-analysis
- Mitigation: 把 self-audit 输出和 ground-truth failure pattern (外部数据分析出的) 做对比，看一致性

---

## 方向 4 — Long-Horizon Research Task (~3-6 月)

**目标**: 换一个**真正需要递归推理**的 benchmark

### Why
现有 100 题都是**单次 advisory**，没有真正的 multi-step assumption-experiment loop。即使我们做完方向 1-3，演示的也只是"每题做得更好"，不是"递归假设验证"。

原 Claude.md "奇异博士时间宝石"设想的是 **multi-round decision**：
- Round 1: 提 3 个 hypothesis
- Round 2: 根据实验结果，保留最有可能的，refine 或 discard 其他
- Round 3: 基于更新后的 belief 设计下一步
- ...

### 候选 benchmark
1. **科研 workflow 仿真**: GPT-5.4 模拟"实验结果"作为 feedback，系统 navigate 假设树
2. **围棋/国际象棋 + 解释**: 不是下棋，是**解释为什么这步好**，然后 ground-truth 是之后几步的发展
3. **案例法模拟**: 给一段 case facts，要求**分阶段**产出"法律框架 → 预测 judge 会怎么看 → 调整 argument"

### Deliverable
- 新 benchmark (~50 题，每题 3-5 步)
- Agent 在 multi-round 下的表现 vs 传统 RAG/CoT
- "assumption 修正轨迹" 可视化

### 难度 / 风险
- 难度: 最高。建 benchmark 是主要成本
- 风险: 最大 — 可能这个 benchmark 和 v16/v19b 架构 mismatch（我们的架构没做过真 multi-step）
- 优点: **这才是 Claude.md 原设想的那篇 paper**

---

## 我的建议 (第二篇最快路径)

**方向 1 + 方向 2 组合，~3-4 周**:

- Week 1: 方向 1 的 autonomous_loop.py infrastructure + 5 rounds demo
- Week 2: 方向 2 的 reframing distillation + A/B 验证
- Week 3: 两个 signal source 整合 (failure + success) 进同一 loop
- Week 4: 论文初稿

这是**真正的"递归自我假设" paper, 而不依赖换 benchmark**。

---

---

# 🔥 脑洞 section — 高温度方向 (不保证 implementable)

以下方向都是**更激进、更 speculative**。每条都标注 "疯狂度" (1=可做, 5=纯想)。

---

## 脑洞 A — Anti-wisdom Library (疯狂度 2)

给每条 wisdom 配一条 **anti-wisdom** (显式的"什么情况下 NOT 这样想")。

比如 W025 "范式不只给答案，还规定何为问题" 的 anti-wisdom:
> "当问题已经给定精确输入输出且评价标准唯一时，别再质疑范式。"

**为什么有意思**: 人类专家之所以强，不是知道多少 rule，而是知道**什么时候 override rule**。Anti-wisdom 是 rule 的自我限制。

**实验设计**: 对每条 wisdom 用 GPT-5.4 自动生成 anti-wisdom。injection 时两者一起给 LLM。测试是否减少 "wisdom 过度激活" 的问题。

---

## 脑洞 B — Wisdom-as-Executable-Code (疯狂度 3)

把每条 wisdom 从 natural language → **可执行的 decision procedure**:

```python
def W025(problem_text):
    # 检测 paradigm-level signals
    has_stakeholder = any(k in problem_text for k in
        ["stakeholder", "监管", "审计", "投入", "合规"])
    has_metric_debate = "什么算" in problem_text or "评判" in problem_text
    has_regulatory = "compliance" in problem_text.lower()

    if has_stakeholder + has_regulatory + has_metric_debate >= 2:
        return {"fire": True, "weight": 0.9, "hint": "paradigm-level"}
    return {"fire": False}
```

**为什么有意思**: 让 wisdom activation 从"LLM 主观判断"变成**可验证、可调试、可量化**的程序。

**缺点**: 失去 LLM 的 semantic flexibility。可能只对部分 wisdom 适用 (过度 procedural 的)。

---

## 脑洞 C — Bayesian Wisdom Weights (疯狂度 3)

每条 wisdom 有**先验激活概率 + 后验更新**。每次 wisdom 被激活且 outcome 有反馈:
- Outcome 好 → 提高 prior
- Outcome 差 → 降低 prior
- 长期下 library 自己 renormalize 成"加权 activation 频率 ∝ 实际 gain"

**和 v3 selections 的区别**: 现在 selection 是二元 (选或不选)。Bayesian 版本是**连续 confidence**，LLM 看到的不是"这条 wisdom 可能适用"，而是"这条 wisdom 有 72% 历史 gain 在类似问题上"。

---

## 脑洞 D — Adversarial Wisdom Generation (疯狂度 3)

两个 agent 对抗:
- **Proposer**: 提新 wisdom
- **Critic**: 试图证明这个 wisdom 和已有 library 冲突或无用
- 只有通过 Critic 的 wisdom 才进 library

**和单 LLM Mode B 的区别**: Self-hypothesis 天然 biased。对抗版本逼 Proposer 准备 counter-example。

**借鉴**: GAN, debate, multi-agent alignment

---

## 脑洞 E — Library Pruning via Darwinian Forgetting (疯狂度 2)

现有 system 只加 wisdom 不减。但人类 expert 会**遗忘过时的 rule**。

每 M 问题，标记"过去 50 题完全没被激活的 wisdom"为 deprecated:
- 不立刻删除，放入 quarantine 区
- 再 50 问题后如仍不激活 → 真删
- 激活过但 outcome 差的 wisdom 也走这个淘汰通道

**为什么有意思**: library 不只是增长，是**演化**。有生有死。论文 framing: "Wisdom Library as Evolving Species"。

---

## 脑洞 F — Agent 给自己出题 (疯狂度 4)

当 Phase 4 detect 到"这条 wisdom 无法确定是否有用"时，**agent 自己生成测试题**:

```
Proposed: W_new
Hypothesized activation domain: business + regulatory
Agent auto-generates 20 synthetic problems in that domain
Runs A/B on synthetic problems
若 +5pp → commit, else drop
```

**为什么疯**: agent 既是学生又是出卷人。但也是**自我改善 loop 最极致的版本**。

**风险**: 生成题可能过 tailored，不 represent real distribution。

---

## 脑洞 G — Cross-LLM Wisdom Transfer (疯狂度 4)

用 Opus / GPT-5.4 solve 那些 3-flash 解不了的 residual，然后**从 Opus 的解法里蒸馏 new wisdom 回馈给 3-flash library**。

```
3-flash fails on P (hard math)
→ Opus solves P with reasoning trace R
→ GPT-5.4 从 R 里提取 "Opus 用了什么 orientation 3-flash 没用"
→ 作为 wisdom 加入 3-flash library
→ 下次遇到类似 P，3-flash 可用这条 wisdom
```

**为什么有意思**: **跨模型 knowledge distillation at wisdom-level**, not weight-level. 论文非常 attractive.

**实际难点**: Opus API 贵, reasoning trace 需要结构化.

---

## 脑洞 H — Wisdom 之间的 relation graph (疯狂度 3)

现有 library 是 flat list。**给 wisdom 之间加关系**:
- W_a **implies** W_b (用 W_a 就隐含要考虑 W_b)
- W_c **contradicts** W_d (两者不能同时应用)
- W_e **refines** W_f (W_e 是 W_f 在特定条件下的专化)

Agent 激活 W_a 时，自动也考虑 W_b。检测到 W_c 和 W_d 同时被选时，强制 pick one。

**论文 framing**: "Wisdom Knowledge Graph" — 每条 wisdom 不是孤立条目，是 interconnected network.

---

## 脑洞 I — 用 Diffusion 生成 wisdom? (疯狂度 5, 纯想)

Wisdom 是高维概念 embedding。已有 75 条 wisdom 是 75 个 embedding points。

**脑洞**: 训一个小 diffusion model 在 wisdom embedding 空间。残余驱动 = conditional diffusion sampling from "current library's uncovered regions"。

**为什么疯**: 可能是过度 novel for novelty's sake. 但**如果真 work**, 就是 full autonomous library growth via generative model.

---

## 脑洞 J — 从用户 interaction 学 (疯狂度 2)

你 (user) 每次和 agent 对话都在**隐式反馈**。把对话 log 本身作为 wisdom 来源:
- 你说 "试试 X" → 该问题 domain 下 "尝试 X" 成为 candidate orientation
- 你说 "不对吧" → 对应 wisdom 降权
- 你说 "好，保留" → 加权

**和方向 1/2 的区别**: 不是纯 automated loop，是 **RLHF-at-wisdom-level**。你是 noisy oracle。

**为什么有意思**: 如果我们把当前这个对话的所有 message 作为 training data（你 53 次提醒我方向 / 36 次 reject 我的建议），就能从里面蒸馏出 metacognitive 规则。

---

## 脑洞 K — 让 wisdom library 变成 plot (疯狂度 4)

把 wisdom library 可视化到**可玩的 UI**:
- 每条 wisdom 是一个 star
- 激活过一次 → star 亮一点
- 关系 → 连线
- 过时了 → 变暗

用户可以**人工拖动 wisdom 位置、标注关系**，这些人工信号 feed 回 system。

**为什么疯**: human-in-the-loop 但 at knowledge-architecture level. 不是人标签数据，是人参与 library 架构演化.

---

## 脑洞 L — Agent 写自己的 prompt template (疯狂度 4)

现在 v16/v19 的 prompt template 是**我们写的**。终极形态: **让 agent 自己写**:

```
Every 100 problems:
  Agent analyzes its own outputs + judge reasoning
  Produces: "if I rewrite my EXECUTE prompt like so, I'd do better"
  A/B test: new template vs old
  若 +5pp → 替换 template
```

**这是 meta-meta**: agent 不只是改 wisdom library, 还改 agent architecture。真正 recursive self-improvement.

**风险**: prompt 自我改写容易 spiral 到诡异模式。需要强 safety gate。

---

# 🎯 脑洞总 ranking

按**论文新颖度 × 可实现性**给个推荐度:

| 脑洞 | 实现 | 新颖 | 结合度 | 推荐 |
|---|---|---|---|---|
| A (Anti-wisdom) | 易 | 中 | 高 | ⭐⭐⭐ |
| E (Darwinian pruning) | 易 | 中 | 高 | ⭐⭐⭐⭐ |
| G (Cross-LLM distillation) | 中 | **高** | 高 | ⭐⭐⭐⭐⭐ |
| H (Relation graph) | 易-中 | 中-高 | 高 | ⭐⭐⭐⭐ |
| B (Wisdom-as-code) | 中 | 中 | 低 | ⭐⭐ |
| C (Bayesian weights) | 中 | 中 | 中 | ⭐⭐⭐ |
| D (Adversarial) | 中-难 | 高 | 中 | ⭐⭐⭐⭐ |
| F (Self-generated tests) | 中 | 高 | 低 | ⭐⭐⭐ |
| J (RLHF-at-wisdom) | 难 | 中 | 高 | ⭐⭐ |
| K (UI plot) | 难 | 中 | 低 | ⭐ |
| L (Self-written prompt) | 难 | 很高 | 高 | ⭐⭐⭐⭐ |
| I (Diffusion wisdom) | 很难 | 很高 | 低 | ⭐⭐ |

---

## 如果我是你（纯个人 take）

**第二篇 paper 方向 1 + 方向 2 + 脑洞 E + 脑洞 G**:
1. 方向 1: 闭环骨架
2. 方向 2: 成功模式蒸馏
3. 脑洞 E: library 有进有出 (不只膨胀)
4. 脑洞 G: 跨 LLM wisdom transfer

四个元素叠加的 paper:
> "An autonomous AI agent that grows its methodological library from both
> failure-driven residuals and success-driven reframing patterns, with
> Darwinian pruning of stale wisdoms and cross-model knowledge distillation
> from stronger generators — achieving monotonic performance gain over N
> problems without human curation."

这是**真正博士级 paper**——可能还不够顶会 best paper，但肯定够一篇扎实的 NeurIPS 主会/ ICLR / ACL。

---

**等 v20 结果出来再决定启动顺序**。
