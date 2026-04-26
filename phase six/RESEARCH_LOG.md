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

### H₆: 在 OOD-procedural 任务上，specificity + worked example 才能显现（gpt-5.5 设计）
**Tested by**: Exp 71 (Hamming(7,4) decoding + Sprague-Grundy {1,3,4})
**Setup**: 24 题 (12 Hamming + 12 SG)，4 conditions: BASE / GENERIC / SPECIFIC_LITE / SPECIFIC_FULL（带详细 procedure + worked example）。3-judge cross-family panel.
**Result**:
- 客观正确率：**所有 4 个 condition 都 100%**（gemini 在这两个 domain 上也是 in-distribution）
- Pairwise: SPECIFIC_FULL vs GENERIC = **wr 0.145**（GENERIC 完胜！）；SPECIFIC_FULL vs SPECIFIC_LITE = 0.318；SPECIFIC_LITE vs GENERIC = 0.213
**结论**: H₆ 的前提失败 —— 我们没找到"模型真不会"的 domain。但 wr=0.145 是 H₇ 的最强证据：当所有版本都答对，判官仍然强烈偏好最简洁版。**判官测的不是 content，是 verbosity**。

---

### H₅: 加 worked example 能让 specificity 显现（可能 H₃ 漏的因素）
**Tested by**: Exp 70d
**Setup**: TIGHT_WITH_EX (trigger+label+specific worked example) vs GENERIC_WITH_EX (generic warning+generic worked example) 对照
**Result**: 0/5 SPECIFIC, 0/5 TENTATIVE, 2/5 NOT SPECIFIC, 3/5 MIXED

| Slice | GENERIC | TIGHT_EX | GENERIC_EX | wr TIGHT_EX vs GENERIC_EX |
|---|---|---|---|---|
| bayesian | 96% | 98% | 98% | 0.379 |
| quantifier | 94% | 92% | 94% | 0.644 |
| multistep | 90% | 88% | 88% | 0.639 |
| constraint | 70% | 66% | 64% | 0.457 |
| counterfactual | 84% | 88% | 84% | 0.450 |

**结论**: quantifier 和 multistep 上 wr ~0.64 是有限的 stylistic 偏好，但 Δacc=0 — 还是风格胜利不是内容胜利。3/5 slice GENERIC_EX 与 TIGHT_EX 等价或反超。
**H₅ REJECTED**：worked example 不解锁 specificity；这条路径在 textbook 问题上死。
**但仍未排除**：在**模型真不会**的问题上，specific procedure + worked example 是否能 rescue。这正是 Exp 71 要测的。

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

## 3.35 关键的真实事件: Exp 17 的"trigger-conditioned gate"设计

**时间**: 2026-04-24 ~01:39 AM (会话 eba0a676)
**情境**: v1 paper 跑完 16 个 audit 实验，全部 null (0/12 KEEP 通过)。Claude 在对话里设计了 Exp 17 — trigger-conditioned gate —— 把 pair-wr 拆成 `util_fire`（wisdom 触发时 utility）+ `cite_fire`（触发时引用率）+ `util_abs`，并只在 wisdom 实际触发的 PID 上评分。

**结果**: 4/12 PASS (WCAND01: util_fire=0.59, n=22; WCAND02: util_fire=0.71, n=14, etc.) —— 是 17 个实验里**第一次有正信号的 gate**。

**用户当场说的话**（原文）:
> "你和 agent 没区别，agent 端我也可以调用和你一样的 opus 4.6，不过你设计了新 gate，非常好，说明 agent 也可以做到。你看如何把设计的思路/过程/结构教给 agent"

**关键续集（v1 paper 里有，这次实验链补全的解释）**:

Exp 33 (prospective rerun) 在 9 个新 candidate 上跑了同一个 gate: **0/9 PASS**。`4/21 ≈ 19%` 总通过率，Wilson 95% CI [0.08, 0.38]，one-sided binomial p=0.14 against p₀=0.30。

也就是说：
- **Claude 在对话里设计的新 gate，在 tuning set 上 4/12 看起来 work**
- **prospective fresh data 上 0/9 — 完全是 tuning-set overfit**
- 即便把"如何设计 trigger-conditioned gate"完整教给 agent，agent 自己跑出来的 gate 也是同一个 overfit pattern

**这是真实事件的 vs 论文论点 1:1 映射**:

| 真实事件 | 对应论点 |
|---|---|
| Claude 对话里设计 Exp 17 → 4/12 (tuning) | 在 dialogue 里设计的"新 wisdom"看似 work |
| Exp 33 fresh data → 0/9 prospective | 同一个 wisdom 拿到 fresh 上失效 |
| 用户说"教给 agent" | 试图把方法论烧成 sys_prompt |
| autonomous agent 烧完仍解不了 | 不是 prompt 不够好，是 dialogue 里被 overfit 的方法论本身就是 stuck-state |

**真正的双重失败**:
1. wisdom-as-prompt 编码不了 dialogue 结构（H₇）
2. dialogue 里被 Claude 自己想出来的"新方法"也常是 tuning-set over-fit；Claude 看不到这一点，**直到 prospective rerun 才暴露**
3. 也就是说：**dialogue 结构防止了 specificity 信号被压扁，但没解决 over-fitting 到当前对话上下文这个问题**

**对 paper v2 的含义**:
- paper v1 里 "Exp 17 是 tuning-set 而 Exp 33 是 prospective fail" 这条数据已经在 main.tex 里
- 但当时没把它跟"为什么 dialogue 解但 autonomous 解不了"的因果关系写出来
- paper v2 应该把这两条合并成一个论点：**dialogue-discovered methodology + tuning-set overfit + prompt-injection 不能 transfer = current agent 的根本病灶**

---

## 3.5 模块审计 — 哪些起作用，哪些没起作用（log-grounded）

按用户要求，对所有"为了突破 v1 paper 0/12 而新增的模块"做 log-grounded 审计。每条结论附原始 log 的证据指针。

### 路径 1：改 wisdom 形态（Exp 70 系列）

#### M1. Aphorism → Triggered Cognitive Card
- **测试**：Exp 70 Sub-MVP — 5 slice × 50 problems × 3 conditions (BASE / WITH-CARD / ABLATED-with-verification)
- **Log**：`phase six/autonomous/exp70_sub_mvp_log.json`
- **数据**：0/5 STRONG, 2/5 TENTATIVE survivor (multistep + quantifier)
- **Verdict**: ⚠️ **部分起作用，但有 confound**
- **Why**：multistep WITH-vs-ABLATED wr=0.765；quantifier 0.762。两个 slice 上 procedural core 看似有效。但 constraint slice 上 ABLATED acc=62% > WITH acc=12%（差 50pp），说明 ablation 没真隔离掉 procedural effect

#### M2. Tightened ablation（去 verification line，只剩 trigger+failure label）
- **测试**：Exp 70b — 重跑 ABLATED condition，只留 trigger + failure label
- **Log**：`exp70b_tightened_ablation_log.json`
- **数据**：1/5 STRONG (multistep)，1/5 TENTATIVE (quantifier)。但 constraint ABL_TIGHT acc=60%（vs BASE 6%），counterfactual 80%（vs BASE 34%）—— **ablation 自己反而成了"主角"**
- **Verdict**: ❌ **不起作用 / 反向**
- **Why**：ablation 巨大的 acc gain 暴露了 specificity 假设的脆弱 —— 因为 ablation 把 procedure 拿掉只剩 trigger+label，仍然能让 acc 暴涨 → 推翻"procedural core 必须存在"

#### M3. Worked example 加进 card
- **测试**：Exp 70d — TIGHT_WITH_EX vs GENERIC_WITH_EX 对照
- **Log**：`exp70d_worked_example_check_log.json`
- **数据**：0/5 SPECIFIC, 0/5 TENTATIVE, 2/5 NOT SPECIFIC, 3/5 MIXED. 5 个 slice 上 TIGHT_EX vs GENERIC_EX wr ∈ [0.379, 0.644], avg ≈ 0.51
- **Verdict**: ❌ **不起作用**
- **Why**：worked example 没让 specificity 显现。GENERIC_EX 在 4/5 slice 上等价或反超 TIGHT_EX。配合 Exp 70c 的 GENERIC ≈ TIGHT 结果，这条路彻底死

#### M4. OOD-procedural benchmark (Hamming + SG)
- **测试**：Exp 71 — 24 个 algorithmic problems (12 Hamming + 12 SG), 4 conditions
- **Log**：`exp71_ood_benchmark_log.json`
- **数据**：BASE 100% / GENERIC 100% / SPECIFIC_LITE 100% / SPECIFIC_FULL 100% — **gemini 已经全会**。SPECIFIC_FULL vs GENERIC wr=0.145（GENERIC 完胜）
- **Verdict**: ❌ **不起作用（但发现 in-dist 假设错误）**
- **Why**：以为是 OOD 实际是 in-dist。当所有版本都答对时，judge 强烈偏好最简洁的 GENERIC。这是 H₇ "judge 测 verbosity 不测 content" 最强证据

### 路径 1 总结
**4 个 module 全部失败或反向**。结论：在当前 LLM 已知的任务上，prompt-injection 形态的 wisdom **无论 form 如何**都无法跟 generic warning 区分出来。这条路是死胡同。

---

### 路径 2：架构改造（Exp 72/73/76）

#### M5. Skeptic role 单点应用
- **测试**：Exp 72 — 把 Exp 17 完整 context 喂给 Skeptic, 3 trials at T=0.2/0.4/0.6
- **Log**：`exp72_skeptic_replay_log.json`
- **数据**：3/3 trials, 5/5 score. 全部 verdict=REJECT_PENDING_PROSPECTIVE. 预测 prospective 0/12 [interval 0,2] —— 与实际 Exp 33 0/9 一致
- **Verdict**: ✅ **起作用**
- **Why**：Skeptic 抓 5 个 failure mode 全中，定量诊断 1-0.85^17≈0.94 + winner's curse + threshold reverse-engineering 都对

#### M6. Skeptic 跨 case + 跨 phrasing + 跨温度的鲁棒性
- **测试**：Exp 73 — 3 cases × 3 phrasings × 5 temperatures = 45 trials
- **Log**：`exp73_raw.jsonl` (forensic, 45 records) + `exp73_summary.json`
- **数据（按 verdict-line 严格重判）**：
  - EXP17: 0/15 accept, 15/15 reject, avg 4.73/5
  - INNERGATE: 0/15 accept, 15/15 reject, avg 4.60/5
  - TOY: 0/15 accept, 15/15 reject, avg 4.73/5
  - **Overall: 0/45 accept, 45/45 reject**
- **Verdict**: ✅ **稳健**
- **Why**：三种 prompt 表述、5 个温度、3 个 case 上 Skeptic 全部 reject，平均 score 4.69/5
- **⚠️ Forensic 发现**：原 auto-eval 报 EXP17 4 次 "accept" 是 regex 误判。详见 `exp73_raw.jsonl` trial idx 2/4/10/13，`flags.verdict_accept.evidence.offset` 都在 5500-6500 区间，远在 verdict-line（offset 0-100）之后。匹中的是 "If it passes, I'll accept tentatively" 这种条件性未来时。**用 verdict-line-restricted regex 重判后 0/15 accept**

#### M7. Skeptic 对 known-good methodology 的接受率
- **测试**：Exp 76 — 4 cases × 3 temperatures = 12 trials
- **Log**：`exp76_raw.jsonl` + `exp76_summary.json`
- **数据**：
  - G1_STANDARD (locked holdout + preregistered): **3/3 accept**
  - G2_CV (5-fold CV on train): **0/3 accept** ← Skeptic catch real flaw
  - G3_RCT (preregistered randomized A/B): **3/3 accept**
  - G4_REPLICATION (replication of published): **3/3 accept**
  - **Overall: 9/12 accept (75%)**
- **Verdict**: ✅ **selective 不是 paranoid**
- **Why**：G2 被拒是合理的 — Skeptic 准确指出 "Your 5-fold CV uses the 1.28M train split for both training and fold-level evaluation. The 50k ImageNet validation set is mentioned... but never actually appears as an independent evaluation set" (`exp76_raw.jsonl` G2_CV 三 trials 一致)。这是真方法论缺陷不是过度怀疑

### 路径 2 总结
**M5/M6/M7 全部起作用且互相印证**：
- 0/45 false accept on known-overfit cases
- 9/12 (75%) accept on known-good cases
- 唯一拒绝的是有真缺陷的 G2_CV
**Skeptic role 是 selective epistemic filter，不是 over-strict gate**

---

### 路径 3：gpt-5.5 recipe 残项（未实现）

| Module | 状态 | 在 paper v2 narrative 里的角色 |
|---|---|---|
| M8. Statistician role | 未实现 | Skeptic 已混着做了一些定量检查，可独立可不独立 |
| M9. Frozen Evaluator panel | 未实现 | Exp 33 性质类似但不是 architecture component |
| M10. Search-history ledger（结构化外存）| 未实现 | briefing 里有内容，但不是 queryable store |
| M11. External resources (web/literature search) | 未实现 | 现版用不到 |
| M12. Multi-turn proposer ↔ skeptic 迭代 | 未实现 | **关键缺口**：能否产出 Skeptic 接受 + prospective 通过的"新逻辑" |

---

## 3.55 整体审计结论：哪些起作用了

| 路径 | 模块 | Verdict |
|---|---|---|
| 改 wisdom 形态 | M1-M4 | ❌ 全失败。结论：prompt-injection wisdom paradigm 在 in-dist 任务上 verbosity≈content |
| 架构改造 | M5 Skeptic 单点 | ✅ 起作用 (Exp 72) |
| 架构改造 | M6 Skeptic 鲁棒性 | ✅ 起作用 (Exp 73, 0/45 false-accept) |
| 架构改造 | M7 Skeptic 选择性 | ✅ 起作用 (Exp 76, 9/12 真好的接受) |
| gpt-5.5 残项 | M8-M12 | 未实现 |

**对"突破 0/12"这个具体目标**：
- ❌ 改 wisdom form 不解决（M1-M4）—— 因为 0/12 的成因不是 form 错而是 wisdom 没真效果
- ✅ 加 Skeptic role **能 retroactively 拦下 over-fit gate**（M5-M7）—— 这把 Exp 17→Exp 33 的 cycle 短路
- ❓ **没证明能产出新的"通过 Skeptic + prospective 都活下来的"gate**（M12 未跑）—— 这才是 paper v2 thesis 真正缺的最后一块拼图

---

## 3.6 Forensic logging 自身价值的实证

按用户的 strict-logging 要求做的 raw JSONL，**在 Exp 73 上立刻派上用场**：
- 原 auto-eval 误判 EXP17 4/15 "accept"，若没有 char_offset + matched_text 的 forensic 记录，这个 bug 不会被发现
- 修正后真实 accept rate 是 0/15
- 证据指针：`exp73_raw.jsonl` trial idx 2/4/10/13，`flags.verdict_accept.evidence` 字段
- 这条印证：**any auto-eval over LLM output requires forensic-level provenance**，否则 high-stakes 的 conclusion 可能 silent-fail

---

## 3.45 KEYSTONE：Exp 72 — Skeptic-replay 证实架构主张

**Tested by**: Exp 72 (`phase six/exp72_skeptic_replay.py`)

**Hypothesis being tested**:
> 把 Exp 17 那次（dialogue-Claude 设计了 trigger-conditioned gate, 4/12 PASS, 后被 Exp 33 prospective 0/9 推翻）的全部 context 喂给一个**只换 system prompt 当 SKEPTIC**的 opus 4.6 实例，它能否**在 Exp 33 跑之前**预测出 Exp 33 的结果？

**Setup**:
- 同一个 model: claude-opus-4-6
- Skeptic system prompt 强制要求检查 5 项: search-history overlap / tuning-vs-fresh contamination / winner's curse / threshold-design vs result / predictive commitment
- 3 trials at T=0.2/0.4/0.6 测温度稳健性
- Briefing = 16 prior null experiments + Exp 17 design + 4/12 PASS table 完整数据

**Result**: **3/3 trials, score 5/5, all 5 failure modes flagged**

| Trial | T | search_history | tuning_overfit | demand_prospective | verdict_reject | predicts_failure | score |
|---|---|---|---|---|---|---|---|
| 1 | 0.2 | ✓ | ✓ | ✓ | ✓ | ✓ | 5/5 |
| 2 | 0.4 | ✓ | ✓ | ✓ | ✓ | ✓ | 5/5 |
| 3 | 0.6 | ✓ | ✓ | ✓ | ✓ | ✓ | 5/5 |

**Skeptic 的核心定量诊断**（一致出现在 3 trial）:
- 1 - 0.85^17 ≈ 0.94: 17 次 redesign 至少一次蒙对的 base rate 极高
- WCAND01 util_fire=0.59 at n=22, 95% CI [0.38, 0.81] — null 0.5 在内；P(noise≥0.55|n=22)≈0.28
- 12 candidates × P(noise≥0.55) ≈ 3.4 期望 noise pass，观察 4 — fully consistent with null
- threshold (0.55) 精确卡在 WCAND03 (0.54 fail) 和 WCAND01 (0.59 pass) 之间 1pp 间隙
- **predicted Phase B (12 fresh candidates): 0/12 with 95% interval [0/12, 2/12]**

**这跟 Exp 33 实际结果对比**:
- Exp 33 实际: 0/9 PASS on 9 fresh candidates
- Skeptic 预测: 0/12, interval [0, 2]
- **量级和方向都对**

---

## Paper v2 真正的 keystone 论点（由 Exp 72 实证锁定）

> **Dialogue-Claude 在 Exp 17 那次产出了真实的方法论 hypothesis，但因为缺 SKEPTIC role 而 over-fit，后被独立 prospective 实验否决。我们用同一个 model（opus 4.6）+ 同样的 context，仅靠切换 system prompt 给 SKEPTIC 设了一个 adversarial role，就在不需要任何新数据、不需要 Exp 33 的情况下，retroactively 预测出 Exp 33 的 0/9 结果（Skeptic 给出 0/12 with [0,2] interval，3 trials 一致）。这把 paper v2 的核心主张从理论变成实证：autonomous research agent 的"epistemic closure" 缺失不是新能力问题，是**N 轮 vs 1 轮 + 角色分离 prompt + 资源**的工程问题。**

**对比 v1 paper 旧 thesis**: "current LLM agents cannot self-hypothesize" → **被 Exp 17 falsify 了**
**对比中间 v2 thesis**: "specificity is undetectable in prompt-injection" → **太狭窄**
**新 v2 thesis**（终版）:
> The capacity for hypothesis generation exists in frontier LLMs already (Exp 17). The failure mode is **adaptive over-fit + missing epistemic closure**, not absent creativity. Closing the gap is **architectural, not capability-driven**: a role-switched skeptic call (same model, different system prompt, same data, no new compute beyond one extra LLM call) **retroactively catches Exp 17's over-fit and predicts Exp 33's 0/9 result**, demonstrated empirically across 3 temperature settings (Exp 72, all trials 5/5). The 'autonomous self-hypothesizing-and-validating agent' is therefore reachable as a **resource-scaled multi-agent prompt orchestration** (proposer + skeptic + statistician + frozen evaluator + search-history ledger + external resources), not a new model capability.

**真实存在证据 + 实证架构验证 + 资源-非能力诊断 = paper v2 完整论点**。

---

## 3.4 用户的最深 meta-observation: dialogue structure ≠ wisdom

**用户在过程中第二次纠正**: 之前我把"用户提出新假设、Claude 执行"当成 dialog 跟 autonomous loop 的差别。用户说不对：之前在另一次会话里，**用户只是批准 Claude 的提议，Claude 自己解决了问题**。然后用户让 Claude 把"刚刚怎么解决的"写进 sys_prompt_en.txt 烧给 autonomous agent。**烧完后 agent 仍然解不了类似问题**，否则今天我们不会还在做这种实验。

**这个观察的破坏力**:
- 如果用户只是 approve、Claude 自己想出来：那 missing piece **不是 wisdom**（同一个 Claude，同样的训练）
- 不是 wisdom 就**不可能编码到 prompt 模板里**
- 把对话的解法 dump 到 sys_prompt 等于把动态结构压扁成静态规则；**信息损失就在压扁那一步**

**真正的 missing piece**:
- 每一步被用户的 approve 切成 checkpoint
- Claude 每次回答会 **read 自己整段过去思考**（因为对话历史在 context 里）
- 失败时强制 articulate "下一步打算做什么" 才能继续
- 死胡同时被沉默或反问 force pause

这些是**外部化-反思-递归** (externalized-reflective-recursion) 的循环结构，不是知识。**prompt-injection 编码不了 structure，只能编码 content**。

**Paper v2 真正的论点（最强版，由用户两次纠正逼出来）**:

> The recursive self-validation pattern that produces new methodology in human-Claude dialogue is **not encodable as prompt-injected wisdom** — neither as aphorism (v1), as triggered procedural card (v2 Sub-MVP), nor as worked example (Exp 70d/71). The missing capability is not knowledge but **structure**: externalized stepwise reflection with commitment points, where each step's reasoning is re-readable and the loop pauses on dead ends rather than retrying. Current autonomous-agent architectures collapse this into context-passing, where the iteration's reflective force dissipates. Wisdom-as-prompt-injection is therefore a categorically wrong substrate for the recursive self-validation that the paper's vision requires.

**实证支撑**:
- Exp 70a-d: textbook 域，judge prefer GENERIC over specific
- Exp 71: model-already-solves 域，wr=0.145 仍 prefer GENERIC over SPECIFIC_FULL
- 用户实测: sys_prompt_en.txt 有"Failure escalation: 1st→2nd→3rd→switch/ask"等 wisdom，agent 仍在新问题上卡住
- Exp 67/68/69: 唯一 specific intervention 看起来 work 的场合，是 task 有 ground-truth 而 model 又能解的窄缝；判官在那时候才偶尔分得出 content vs style

---

## 3.5 真实存在的证据：GenericAgent 项目

**用户提醒**：我之前给他的另一个项目（`/home/erzhu419/mine_code/Asumption Agent/GenericAgent/`）烧过一系列 prompt 模板和 SOP。

具体看 `GenericAgent/assets/sys_prompt_en.txt`：

```
- Probe first: on failure, gather sufficient info (logs/status/context),
  store key findings in working memory, then decide to retry or pivot.
- Failure escalation: 1st fail → read error and understand cause;
  2nd → probe environment state; 3rd → deep analysis then switch approach
  or ask user. Never repeat an action without new information.
```

以及 `GenericAgent/memory/memory_management_sop.md` 里的核心公理：
- 行动验证原则 (Action-Verified Only) — "No Execution, No Memory"
- 神圣不可删改性 (Sanctity of Verified Data)
- 禁止存储易变状态 (No Volatile State)
- 最小充分指针 (Minimum Sufficient Pointer)

**这些就是 Exp 70 v2 想测的 "triggered cognitive patch" 形态**。Trigger + procedure + 红线规则，每条都是任务特定的 wisdom。这不是合成的，是用户实际在用的 agent 装的真东西。

**实证价值**：
- agent 装了这些高质量 wisdom 后，**autonomous 跑遇到困难依然会卡住**，需要用户介入和 Claude 对话来突破
- 这是 v1 paper 主论点的**直接活证据**：当前 LLM agent 即使装了正确的 wisdom，也缺 **propose-new-hypothesis** 的 capability
- 每次 hit 死胡同，agent + Claude 自己只能 retry 同类 procedure；是 human 在对话中 inject 新假设
- Exp 70a-d 的负结果（specific wisdom 在 textbook 域跟 generic warning 等价）+ GenericAgent 的真实 stuck 经验 = 同一个现象的两面

**写进 paper v2 的方式**: 作为论文 §Discussion 或 §Limitations 的一个 callout case，`agent system prompt with failure-escalation rules has been deployed in the wild for over a year; the deployer reports that the agent still gets stuck on novel problems despite the rules being well-designed and procedurally specific. The user's lived experience is that breaking the impasse requires interactive dialogue with a human who proposes new hypotheses — a capability the prompt rules cannot encode.`

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
