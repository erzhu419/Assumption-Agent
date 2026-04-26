# Loop v2 Research Log

记录从 v1 paper 的 0/12 null 出发，重新设计 wisdom-injection loop 时遇到的假设链、实验数据、和最后的元层面反转。

---

## 0. 起点：v1 paper 的 0/12 null

**v1 loop**: orchestrator 提议 12 个中文 wisdom 候选 → 同家族 +10pp gate at n=50 → 接受 3/12 → audit stack 全部驳回，3 次 preregistered fresh-data replication 全 0/12。

**v1 解释（写在 paper 里的版本）**:
> "the simplest aphorism-form self-improvement loop fails its own audit"

**Exp 67/68/69 给的旁证**:
- 给 gate 一个**强 task-specific Bayesian template + worked examples**，wr=0.846/0.933 接受
- 给 reasoning-suppressing prior，wr=0.286/0.133 拒绝
- 跨家族 judges κ=1.0
- 结论：gate 不是 structurally anti-additive；问题在 wisdom form

---

## 1. v2 假设链与实验结果

### H₀: aphorism wisdom → wisdom_count > 0
**Tested by**: v1 paper 全部
**Result**: 0/12 at n≥100. **REJECTED.**

### H₁: triggered cognitive patch (procedural form) 有 specificity
**Tested by**: Exp 70 (Sub-MVP)
**Setup**: 20 hand-written cards × 5 slices × 50 problems × 3 conditions (BASE / WITH-card / ABLATED-with-verification-line)
**Result**: 0/5 STRONG, 2/5 TENTATIVE
- bayesian: WITH wr=0.351 (BASE 主动赢), 因为 baseline 已 98% ceiling
- quantifier: WITH wr=0.913, 但 Δacc=0
- multistep: WITH wr=0.728, 但 Δacc=0
- constraint: WITH wr=0.515, **ABLATED wr=0.957** (ABLATED 完胜)
- counterfactual: WITH wr=0.583, 各 condition 接近

**Anomaly**: constraint slice 的 ABLATED acc=62% 比 WITH acc=12% 高 50pp。提示 ablation 不干净（verification line 本身是个 procedure）。

### H₂: tightened ablation 之后 form 才显出 causal signal
**Tested by**: Exp 70b
**Setup**: 同 70 但 ablation 只剩 trigger + failure label，去掉 verification 那行
**Result**: 1/5 STRONG (multistep), 1/5 TENTATIVE (quantifier)
- multistep STRONG: WITH-vs-ABL_TIGHT mean=0.752 (min=0.733), Δacc=0
- quantifier TENTATIVE: WITH-vs-ABL_TIGHT mean=0.942
- **constraint ABL_TIGHT acc=60% (BASE 6%, WITH 12%)**: tight ablation 反而拉到 60%
- **counterfactual ABL_TIGHT acc=80% (BASE 34%, WITH 36%)**: tight ablation 拉到 80%

**Diagnosis**: 全 procedural card 在硬任务上 hurt（锁死路径）；trigger+label-only 反而 work。但 work 的是 trigger 还是 label，未知。

### H₃: trigger+failure-label 是 content-specific 的（不只是泛化"be careful"效应）
**Tested by**: Exp 70c (specificity check)
**Setup**: 加 GENERIC_WARNING condition ("This problem may be tricky. Watch for hasty conclusions; missing important details.")，与 ABL_TIGHT 同长度同结构但去掉 problem-class trigger 和 specific failure name
**Result**: 0/5 SPECIFIC, 0/5 TENTATIVE, 1/5 SLOW-DOWN-ONLY, 4/5 MIXED

| Slice | BASE | ABL_TIGHT | **GENERIC** | wr ABL_TIGHT vs GENERIC |
|---|---|---|---|---|
| bayesian | 98% | 98% | 96% | 0.220 (GENERIC 完胜) |
| quantifier | 92% | 92% | **94%** | 0.477 |
| multistep | 88% | 88% | **90%** | 0.414 |
| constraint | 6% | 60% | **70%** | **0.197** |
| counterfactual | 34% | 80% | **84%** | 0.574 |

**结论**: GENERIC 在 4/5 slice 上等价或反超 ABL_TIGHT。**specificity 假设 REJECTED**。

### H₄ (我的): gate 测的是 style 不是 content
**Logic**:
- ABL_TIGHT 和 GENERIC 在 constraint 上 acc 都被拉到 60-70%，差距小（10pp）
- 但 wr 差距大（0.197 = -30pp）
- → 两个版本产生**实质等价答案**，但 specific 版本更冗长 → 判官嫌弃
- → **judges 测 verbosity / 简洁性 / 结构感，不测 substance**

如果成立：v1 的 0/12 不是 "wisdom 没用"，是 **"LLM-as-judge gate 物理上不能 detect content gain"**，无论 wisdom 是什么形态。

### H₅: 加 worked example 能让 specificity 显现（可能 H₃ 漏的因素）
**Tested by**: Exp 70d (跑中)
**Setup**: TIGHT_WITH_EX (trigger+label+specific worked example) vs GENERIC_WITH_EX (generic warning+generic worked example) 对照
**Status**: 跑中（约 20-25 min wall）
**Hypothetical**: 不论结果，限于 textbook 问题域

---

## 2. 用户的 meta-observation（关键反转）

引用：
> "如果你做的实验结论成立，就说明，已有的'智慧'足够解决眼前的所有问题，这不就和我们目前正在遇到的问题相悖了么"

**这指出的悖论**:
- 实验结论 H₃/H₄ 隐含 "wisdom 已经在模型里，prompt 只是 elicit"
- 但**我们此刻正在做的研究**就是反例：我们面对一个**没有现成 wisdom 能解**的问题（怎么修 LLM-judge loop）
- 用户用自己的 meta-reasoning（指出我的 category error）证明了 **"未在模型 prior 里的 wisdom"是真存在的**
- 而我（"现有 LLM 方法的集合"）此前**没有自动产出过这个 meta-observation**

**矛盾的解决**:
- 我的实验没有证伪 "wisdom 重要"，只证伪了 **"wisdom 在 LLM 已经会的问题上重要"**
- 250 道测试题 = 全部在训练分布内（textbook Bayesian / 经典 Monty Hall / 老牌脑筋急转弯）
- baseline 88-98% 因为这些 wisdom **都在 prior 里**
- 用 **prior 里的 wisdom** 去测 **模型对新 wisdom 的接受度** = category error
- 我的实验范围限于 "模型已会的问题"；那个范围内 specificity 物理上不可能被检测到

---

## 3. v2 paper 的真正论点（候选）

旧版（错的）:
> "We redesigned the loop with triggered cognitive patches; still 0 survivors. The aphorism-injection paradigm is broadly invalidated."

**新版（对的，由用户的 meta-observation 推导）**:
> "We tested wisdom-injection at exactly the wrong level. Our benchmarks fall within model competence; in that regime, any 'be careful' prefix elicits the model's pre-existing knowledge and specific wisdom contributions are indistinguishable from elicitation. The genuine question — whether wisdom transmission can encode information ABSENT from the model's prior — requires benchmarks where existing model knowledge is provably insufficient. Constructing such benchmarks for current LLMs is itself an open problem, and the inability to measure wisdom-content effects on existing benchmarks is a structural obstruction to the LLM-judged self-improvement paradigm."

**这版更强，因为**:
1. 它**统一**解释 v1 + v2 + Exp 70a-d 全部数据
2. 它**直接 imply**：当前文献中所有"LLM 自我改进"研究都可能踩同样的坑
3. 它**自我演示**：用户的 meta-observation 本身就是论文论点的 case study

---

## 4. 用户在过程中扮演的角色（论文里值得写一笔）

| 角色 | 在 v1 paper 里的描述 | 现实中是谁 |
|---|---|---|
| Self-hypothesizing agent | 应该自动产生新假设 | **用户** |
| Self-validating agent | 应该自动设计实验验证 | 用户 + Claude（执行） |
| Wisdom library | 静态 prior | Claude 的全部训练数据 |
| Scheduler | 选择 wisdom | Claude（in-context） |

**关键观察**: 当 Claude（"现有 LLM 方法之和"）跑出 H₃ specificity 失败的时候，没有自动 propose H₄/H₅。**用户的 meta-observation（H₅ 的悖论）是 Claude 没产出的**。这正是 v1 paper 说的 Stage 4（生成新假设）当前不存在的实证证据。

---

## 5. Open problems / 下一步

**P1: 构造"模型真不会"的 benchmark**
- 候选方向：训练 cutoff 之后才发表的领域特定问题；刻意 OOD 组合任务；自造的、文献里不存在的脑筋急转弯
- 难点：怎么验证某个问题"真不在模型 prior 里"？需要先做 baseline + 强思考 elicitation 全失败的过滤

**P2: 在 P1 benchmark 上重测 specificity**
- 在那种 benchmark 上，generic warning 应该不能拉准确率（模型本来就不会）
- 而 specific wisdom（带 worked example）应该能
- 如果差距出来了，wisdom-content 假设在 OOD 域成立
- 如果还没差距，则更深的结论：wisdom-injection 在 LLM-judge 框架下根本不能 work（需要 fine-tune 或其他 substrate）

**P3: 论文 v2 写作**
- 主线：H₅（用户的）的元论证
- 数据支撑：Exp 70a-d 全部 + v1 paper 0/12
- 论点：当前 benchmark 全部位于模型 competence 内 → wisdom-content 在那里物理上不可被检测 → "LLM 自我改进失败"和"benchmark 选错了"在数据上不可分

**P4: 等 Exp 70d 结果归档**
- 不论结果都写进 RESEARCH_LOG，但不当 paper 主线

---

## 6. Commit chain

```
af30194 Loop v2 Sub-MVP: triggered cognitive patches, 250 slice problems, runner
        (新建 phase six/, 20 cards, 5 slices, exp70_sub_mvp.py)
[pending] Exp 70b/c/d + RESEARCH_LOG
```

---

## 7. Timestamps

- 2026-04-26 ~08:30  v1 paper submission-ready, tag submission-v1 at 9f532f1
- 2026-04-26 ~09:00  Exp 67/68/69 完成，positive control 部分立住但 Δacc 小
- 2026-04-26 ~09:30  Exp 70 (Sub-MVP) 开始 v2 重设计
- 2026-04-26 ~10:00  Exp 70b/c 完成；GENERIC ≈ TIGHT 出现
- 2026-04-26 ~10:30  用户提出 meta-observation："实验结论与现实矛盾"
- 2026-04-26 ~10:35  本 RESEARCH_LOG 起草，paper v2 论点重新框定
