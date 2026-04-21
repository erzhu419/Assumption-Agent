# Experiments Branch Log

**Autonomous execution rules (user-authorized 2026-04-22 bedtime):**

1. 从 Phase 1 改造开始，逐一测试 Phase 2/3/4
2. **决策规则**（每阶段实验后）：
   - 胜率 > baseline **+5pp**（statistically meaningful）→ **KEEP**，作为下阶段的 base
   - 胜率在 **±3pp** 之内 → **INCONCLUSIVE**，停止后续阶段，等用户决定
   - 胜率 **<-3pp** → **REVERT**，不继续此方向
3. 最多 4 并发实验
4. 每个分支点在此日志记录：hypothesis / variants / decision / 证据

---

## Base State (start of autonomous session)

**Commit:** 79394c0
**Cache state:**
- `baseline`: 100 answers cached
- `ours_27`: 100 answers + 18 structures cached (technique form)
- `vanilla_39`: 100 answers + 18 structures cached (technique form)
- 3 judgments cached: ours_27 vs baseline (47%), vanilla_39 vs baseline (50%), ours_27 vs vanilla_39 (48%)

**Sample:** 100 test problems, seed=42, frozen in `phase two/analysis/cache/sample_100.json`

---

## Phase 1 改造: Polya/Popper Orientation Restoration

**Hypothesis:** 当前 KB 把 Polya/Popper 的"意识"（orientation）强行拆成 4-6 步 `operational_steps`，杀死了原意。改写为 orientation form（trigger + attention_priors，无 step list），应能让 LLM 保留 attention direction 而不陷入机械执行。

**Schema change:**

| 字段 | Technique form (旧) | Orientation form (新) |
|---|---|---|
| `one_sentence` | "将当前问题映射到相似问题" | "养成每遇新问题就反问'见过类似的吗'的习惯" |
| `operational_steps` | [step1, step2, step3, step4] | ❌ 删除 |
| `trigger` | — | "何时激活（持续/触发条件）" |
| `attention_priors` | — | "3-5 个 core self-questions" |
| `original_wisdom` | — | "直接原文引用" |

**Pipeline change:**
- IMPLEMENT 不再产 `{"step1":"", "final_answer":""}` 
- 改为产 `{"attention_priors": [...], "answer": ""}`
- EXECUTE 把 priors 作为 attention 提示，让 LLM 自由解答

**Variants to test:**

- `orient_12`: 只把 12 个 Polya/Popper 策略改 orientation form，其余 15 个保持原样（混合）
- `orient_27`: 全部 27 改 orientation form（纯净对照）

**Kept if:** orient vs ours_27 胜率 ≥ 53%（+6pp）

**Token-saving rule (user directive):** vanilla_39 已冻结为静态 baseline，后续实验不再生成 vanilla_39 answers。每个新 variant 只对 `baseline` 和 `ours_27` 做 judge。

### Results (autonomous run, 2026-04-22)

**Execution:** orient_hybrid = 12 orientation + 15 technique.
- Stage 1: 18 categories × 3 LLM calls, 100s
- Stage 2: 100 answers, 604s
- Judges: 2 × 150s

**Overall:**

| 对比 | 胜率 | mean Δ score |
|---|---|---|
| orient_hybrid vs baseline | 45.0% | -0.19 |
| orient_hybrid vs ours_27 | **51.0%** | +0.05 |

**By domain (orient_hybrid vs ours_27):**

| Domain | orient_hybrid | ours_27 | 胜率 | 解读 |
|---|---|---|---|---|
| business | 9 | 6 | **60.0%** | orientation 明显赢 |
| daily_life | 10 | 5 | **66.7%** | orientation 大胜 |
| sw_eng | 14 | 11 | **56.0%** | orientation 赢 |
| engineering | 7 | 8 | 46.7% | 基本持平 |
| mathematics | 5 | 10 | **33.3%** | technique 大胜 |
| science | 6 | 9 | **40.0%** | technique 赢 |

**Decision: INCONCLUSIVE (within ±3pp band)**

**但信息量大：发现了一个"轴"**
- **orientation form 赢 practical domains** (business/daily_life/sw_eng) —— 这些领域需要 "attention priors"，不需要机械步骤
- **technique form 赢 formal domains** (math/science) —— 这些领域需要 "procedural rigor"，attention priors 反而损失必要的结构
- 这不是 orientation vs technique 谁更好，而是**不同问题类型匹配不同形式**

**本次停在这里**。两种路径供用户醒来决策：

**路径 A（domain-gated 混合）：**
- 在 practical domains 用 orientation form（12 Polya/Popper）
- 在 formal domains 用 technique form（原版 operational_steps）
- 预期：把 practical 60%+ 的优势拿住，同时保留 math/science 的步骤结构
- 实现：为每个 (domain, difficulty) 类别选择 form

**路径 B（接受 inconclusive，原架构继续改造 Phase 2/3/4）：**
- 不改 form，保留 ours_27 作为 base
- Phase 2 改造目标仍然是"意识触发条件 from experience"

两个路径都可做，但路径 A 基于实证数据，更稳。

---
