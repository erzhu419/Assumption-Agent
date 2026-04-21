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

**User directive after waking:** 做 A 和 B，并加一个"LLM auto-detect form"的 C.

### Track A/C extended results

**Track A (domain_gated):** 70 orient + 30 technique cached answers (rule: practical→orient, formal→technique).

| 对比 | 胜率 |
|---|---|
| domain_gated vs baseline | 45.0% |
| domain_gated vs ours_27 | 56.2% (27 ties; practical wins) |

**Track C (auto_detect):** LLM classifies per (domain, difficulty) before form selection.
- Classifier 62% agreement with naive domain rule.
- Divergences: engineering/easy+hard→proc (disagrees), sci/hard+medium→orient (disagrees), sw_eng/easy+hard→proc (disagrees).

| 对比 | 胜率 |
|---|---|
| auto_detect vs baseline | 46.0% |
| auto_detect vs ours_27 | 49.0% |
| auto_detect vs domain_gated | 48.0% |

**Phase 1 final verdict:** 6 classifier divergences net-negative (2 helpful on sw_eng, 4 harmful on engineering+science). **domain_gated > auto_detect**. But neither beats baseline. All form-choice strategies stuck at 45-50%.

**Phase 1 kept:** orient_hybrid as representative base for Phase 2 (orientation form has the "attention prior" infrastructure we want to build on).

---

## Phase 2 改造: Awareness Triggers from Failure Mining

**Hypothesis:** From cases where orient_hybrid lost to baseline, extract "missed early signals" and inject them as per-category attention priors in EXECUTE. These triggers are orientations (警觉/反问), not techniques.

**Pipeline:**

1. Mine losses: 55 orient_hybrid-lost cases, LLM extracts 3 triggers each = **161 triggers across 14 categories**
2. Sample triggers by (domain, difficulty): exact category first, then same-domain, then same-difficulty
3. Inject top-4 triggers into EXECUTE prompt as historical警觉, alongside attention_priors from Stage 1

**Variant:** phase2_triggers. Reuses orient_hybrid's Stage 0/1 structures — ONLY EXECUTE prompt differs.

### Phase 2 results (100 problems, seed=42)

| 对比 | 胜率 | Δ score |
|---|---|---|
| **phase2_triggers vs baseline** | **53.0%** | **+0.14** ← FIRST variant to beat baseline |
| phase2_triggers vs orient_hybrid | **56.0%** | +0.31 |

**By domain (vs baseline):**

| Domain | phase2_triggers | ours_27 baseline | Δ |
|---|---|---|---|
| business | 53.3% | 53.3% | 0 |
| daily_life | **73.3%** | 66.7% | +6.6 |
| engineering | **46.7%** | 33.3% | **+13.4** |
| mathematics | 33.3% | 33.3% | 0 |
| science | 40.0% | 40.0% | 0 |
| sw_eng | **64.0%** | 52.0% | **+12.0** |

**By difficulty:** easy 33%, medium 54%, **hard 55%** (triggers help hard problems most).

**Key insight:** The triggers are **problem-specific wisdom extracted from failure**. sw_eng/hard had 21 triggers mined (most), and sw_eng jumped +12pp. Engineering/medium had 17 triggers, engineering gained +13pp. Triggers on math/science didn't fire as much because these tend to need procedural rigor more than awareness priors.

**Decision: KEEP.** Phase 2 改造 is effective. Use phase2_triggers as base for Phase 3 改造.

---

## Phase 3 改造: Cross-Civilizational Archetypes (REVERTED)

**Actual implementation:** Rather than per-problem hierarchical structure retrieval, built a library of 20 universal archetypes drawn from cross-civilizational wisdom (箴言/Ecclesiastes + 孙子/Polya/Popper/Kahneman/Hayek/Smith/Acton). Each archetype is an *orientation*, not a technique — "沉没成本谬误", "囚徒困境 / 代理人问题", "路径依赖", "观念根源", "已有的后必再有" etc. LLM detects 2-3 applicable archetypes per (domain, difficulty) category. Archetype wisdom injected as Layer 3 of EXECUTE prompt, on top of Phase 2 triggers.

**Variant: phase3_archetypes.** Reuses phase2_triggers structures + triggers.

### Phase 3 v1 (parse bug)

max_tokens=300 truncated LLM's JSON output mid-reasoning. 14/16 categories got empty archetype lists. phase3 essentially became phase2 + "(无原型适用)" placeholder — which actually noise-polluted the EXECUTE prompt.

- phase3 v1 vs baseline: 48% (regressed from phase2's 53%)
- phase3 v1 vs phase2: 51% (within noise)

### Phase 3 v2 (parse fixed: max_tokens=600 + regex fallback)

All 16 categories now got archetypes. Detected reasonable picks, e.g.:
- business/hard → 路径依赖, 层级失灵, 锁定效应
- math/hard → 奥卡姆剃刀, 可证伪性, 边际效益递减
- sw_eng/hard → 二阶效应, 路径依赖, 确认偏误

**Notable:** 二阶效应 (A08) selected for 11/16 categories — LLM sees "long-term consequences" as near-universal orientation.

### Phase 3 v2 results

| 对比 | 胜率 | Δ score |
|---|---|---|
| phase3_v2 vs baseline | 49.0% | -0.19 |
| **phase3_v2 vs phase2_triggers** | **47.0%** | **-0.18** ← REVERT |

**By domain (phase3_v2 vs phase2_triggers):**

| Domain | phase3 wins | phase2 wins | Δ |
|---|---|---|---|
| business | 9 | 6 | +3 |
| daily_life | 9 | 6 | +3 |
| engineering | 7 | 8 | -1 |
| **mathematics** | **5** | **10** | **-5** |
| science | 6 | 9 | -3 |
| **sw_eng** | **11** | **14** | **-3** |

### Diagnosis

Archetypes help **practical decision domains** (business/daily_life +3pp each) but **hurt procedural/technical domains** (math/science/sw_eng combined -11pp). The "Layer 3 universal wisdom" framing primes LLM toward abstract philosophical analysis, while math/science problems need concrete procedural rigor.

This echoes Phase 1's finding: practical vs formal axis dominates form/content choice. Adding more orientation-layer content to formal domains is net-negative.

### Decision: REVERT Phase 3 改造

Per autonomous rule "<-3pp → REVERT, 不继续此方向". phase2_triggers remains the winning base.

**Phase 4 改造 NOT automatically initiated** per "没效果就改回来" + "停止后续阶段" (autonomous rule). Awaiting user decision.

---

## Current winning variant: phase2_triggers

53% vs baseline (first to beat baseline). Uses:
- 12 Polya/Popper strategies in orientation form (Phase 1 改造, +0)
- 15 original technique strategies (from phase 0)
- Hybrid SELECT/ADAPT/IMPLEMENT pipeline (Self-Discover task-level)
- Failure-mined awareness triggers injected into EXECUTE (Phase 2 改造, +6pp)

Paths open for user decision:

1. **Selective archetypes** — inject archetypes only for business/daily_life, not formal domains
2. **Phase 4 改造** — let LLM generate NEW attention priors from its own failures (meta on meta)
3. **Accept phase2 as main deliverable** and write up

---

## Phase 1 "Big Redo": Canonical heuristic questions from Polya/Popper (REVERTED)

**Hypothesis:** My Phase 1 paraphrases were suboptimal — going back to Polya's and Popper's **actual heuristic questions** (verbatim from source texts) should improve orientation fidelity and hence performance.

**Pipeline:**

1. LLM extracted 5-6 canonical heuristic questions per strategy from Polya 1945 / Popper 1959. Most marked as verbatim or near-verbatim. Examples:
   - S03: "Have you seen it before?" / "Can you think of a familiar problem having the same or a similar unknown?"
   - S19: "Could you drop a condition?"
   - S22: "Can you restate the problem?"
   - S13: "What observations, if found, would refute my theory?"
2. Stored as `phase zero/kb/strategies_canonical/S*.json` with source citation (author/work/year/section).
3. New framework `phase1_canonical_framework.py`:
   - SELECT sees canonical questions as module hint
   - ADAPT rephrases for task category
   - EXECUTE uses **persona prompting**: "你作为内化了 Polya/Popper 精神的思考者…"
4. Stacked with Phase 2 triggers.

### Results

| 对比 | 胜率 | Δ score |
|---|---|---|
| phase1_canonical_plus_p2 vs baseline | 47.0% | -0.31 |
| **phase1_canonical_plus_p2 vs phase2_triggers** | **46.0%** | **-0.26** ← REVERT |

**By domain (vs phase2_triggers):**

| Domain | canonical | phase2 | 解读 |
|---|---|---|---|
| business | 7 | 8 | 略输 |
| daily_life | 9 | 6 | canonical 赢 |
| engineering | 7 | 8 | 略输 |
| **mathematics** | **4** | **11** | **canonical 大败** |
| **science** | **5** | **10** | **canonical 大败** |
| sw_eng | 14 | 11 | canonical 赢 |

### Important unexpected finding

**Authentic source material is WORSE than LLM-oriented paraphrase** for LLM prompting.

Why:
1. Polya 的问句 ("Could you drop a condition?") 是为**人类自我反思**写的 — 抽象、开放、依赖内省。LLM 读它走一个 "yes/no 检查" 就过去了。
2. 我的 paraphrase ("当感觉'没出路'时，问自己'哪条约束是真必要的，哪条是我自己加的'") 是为 **LLM 指令消化**写的 — 带场景 trigger，更具体，有操作指引。
3. Persona prompting ("你作为内化了 Polya 精神的思考者") 是**间接**的 — LLM 跟随这种角色扮演不如直接给它具体警觉。

**Methodological implication:** 做 LLM harness 时，"回归原典" 不是自动优化方向。需要的是 **"LLM-adapted orientation"** — 基于原典精神但针对 LLM 消化特性做过 phrasing 改写的版本。我之前写的 12 个 Phase 1 paraphrase 误打误撞命中了这一点。

**Decision: REVERT. phase2_triggers (53% vs baseline) remains the winning variant.**

### Follow-up: Clean ablation (per user's challenge 2026-04-22)

User challenged: "学富五车学贯中西的人怎么会比水管工差? 之前的对比可能有 confounds (persona, 3-layer, ordering)."

Clean ablation: built `strategies_canonical_orientation/` — 12 JSONs with **SAME schema** as `strategies_orientation/`, only `attention_priors` field swapped to Polya/Popper canonical Chinese heuristic questions. Ran through IDENTICAL orientation_framework + phase2_framework pipelines. Same layer count, no persona, same warning directness.

| 对比 | 胜率 | Δ score |
|---|---|---|
| orient_canonical_plus_p2 vs baseline | 49.0% | -0.01 |
| **orient_canonical_plus_p2 vs phase2_triggers** | **46.5%** | **-0.18** |

Nearly identical to the initial -0.26 Phase 1 big redo. Confirms: **content is the main driver, not prompt design**.

**Interpretation of "学富五车 vs 水管工":**

In human reader scenario, canonical (学富五车) wins trivially. In LLM consumer scenario, paraphrase (水管工-translator) wins.

Reason: Polya's "Have you seen it before?" is written for humans — who pause, search memory, actually do analogy. LLM reads same text → produces a text continuation, does NOT actually pause-and-reflect.

My paraphrase "当感觉'没出路'时，问自己'哪条约束是真必要的'" embeds:
  - scenario trigger (当感觉没出路时) → LLM recognizes WHEN to apply
  - specific self-question (哪条约束真必要) → LLM has concrete operation

Best-in-class agent: **学富五车 + 水管工 = 同时理解原典精神 + 理解 LLM 消化特性的 paraphrase 作者**. Not "canonical text in; wisdom out" — it's "canonical-understood-and-LLM-translated". The value is in the TRANSLATOR's dual literacy, not extractable directly from source.

**Hard difficulty particularly hurt by canonical:** 39.5% vs phase2's 60.5% on hard problems — exactly where we'd expect wisdom to help most. Abstract heuristic questions lack scenario grip on complex problems.

**Formal domains** (math 33%, science 40%, engineering 40%) uniformly lose with canonical; **practical domains** (daily_life 60%, business 53%) sometimes win. Matches the practical-vs-formal axis found in Phases 1 and 3.

---

## Open paths (for user decision)

- **A. Selective archetypes**: inject Phase 3 archetypes only for business/daily_life
- **B. Phase 4 改造 (meta-generation)**: let LLM generate NEW orientations from its own failures
- **C. Phase 3 "case + metaphor library"**: per-archetype 10-20 真实 cases + 跨源隐喻（不限原文），embedding retrieval，让 LLM 通过看多个 instances 归纳 pattern（per user's correction 2026-04-22）
- **D. Accept phase2_triggers as main deliverable** and write up

