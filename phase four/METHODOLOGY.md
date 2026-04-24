# 方法论：从提出假设到自我验证（带 trigger-conditions）

> **v2 升级说明**：v1 只列了 8 条 principle，缺每条的 **trigger-condition**
> （什么时候该用、什么时候不该用、怎么识别"现在"是那个时刻）。
> 正是本项目 Exp 15 → Exp 22 发现的同一个 gap ——
> **principle 不自我应用**，必须配 conditioning 才能落到每一步。
>
> v1 的错误本身是个 case-in-point：在 wisdom library 层发现 conditioning 不够，
> 到 methodology library 层我仍然只列 scalar rule。这次修正。

---

## 0. 元结构：为什么单列 principle 不够

看这个递归：

| 层级 | 知识单位 | 单列 principle 的失败 | 加 trigger 后 |
|---|---|---|---|
| **对象层** | wisdom (如 W078) | 12/12 PASS (Exp 21c, 无 discrimination) | Exp 17 SHOULD_FIRE partition → 4/12 PASS |
| **架构层** | orthogonal decomposition | agent gate 12/12 PASS | 加 conditioning → 可 discriminate |
| **方法层** | "from least to most" | 原则记住了，每一步不知道是否适用 | 加 applies-when + canonical case + trigger signal |

三层**同构**：任何可复用知识单位 X 都需要配 trigger-condition C_X。
C_X 不能从 X 自身 derive，必须从**正负 case 对比**中 distill 出来。

本文档 v2 版本每条 pattern 都带 **5 个字段**：
- **Applies WHEN** — 该动手的条件
- **Does NOT apply WHEN** — 该放手的条件
- **Positive case** — 本 session 里用对了的具体例子
- **Negative case** — 用错或 overapply 的例子
- **Trigger signal** — 实际当下能察觉到 "现在该用这个" 的外显信号

---

## 1. Pattern: Read before write

**Applies WHEN**
- 要写代码读/写某个已存在的 cache / meta / answer JSON
- Claim 依赖于某具体数据结构
- 整合到已有 pipeline（复用现成 script）

**Does NOT apply WHEN**
- 纯 greenfield，没有先前 artifact 可参考
- 纯算法实现，无 I/O side effect
- Refactor 已有 tested 代码（Read 只会 slow）

**Positive case**
Exp 14/15/17 我每次都先跑:
```bash
python -c "import json; d=json.loads(open('_valp_v20_ext_WCAND01_meta.json').read());
          print(list(d[next(iter(d))].keys()))"
```
→ 知道 schema: `frame, critical_reframe, anti_patterns, evaluation_criteria, rewritten_problem, what_changed`。所有后续 component 函数都用对字段名。

**Negative case**
Exp 21 agent 没做 Read，写 `rec["prompt"]` 假设字段叫 prompt，实际叫 problem。
静默 exit 0 但 n_prompts=0，花了 3 轮 correction 才发现。

**Trigger signal**
每次要 `open(some_cached_file).read()` 前 → 停。先 grep/cat 看真实 schema。

---

## 2. Pattern: Cross-artifact distillation (最重要，liftable 到任意层级)

**Applies WHEN**
- 有两个（或多个）做同样任务的 artifact
- 能判定哪个 strong 哪个 weak（有外部 metric）
- 结构差异可被 LLM 识别并命名

**Does NOT apply WHEN**
- 只有 1 个 artifact（没对比基准）
- 不知道哪个更好（没 strong/weak 划分）
- 差异在连续参数空间（无结构性 move 可蒸馏）

**Positive case**
- 对象层 (原 `cross_llm_distiller.py`): 3-flash 答卷 vs GPT-5.4 答卷 → 蒸馏 orientation
- 架构层 (Exp 22 Phase A): Gate A (12/12) vs Gate B (4/12) → agent 独立命名 **"condition-stratified measurement"**
- 方法层 (本 doc v1 → v2): scalar principle list vs conditioned list → 蒸馏出本 gap

**Negative case**
如果给 agent 只看 Gate A 自己的 failure，没给 Gate B 的 success 对照，
它无法 distill missing move —— Exp 20 Phase 3 就是这种：没有"怎样算好"的对照，只输出 related design (sim=0.28)。对比 input 才让 Exp 22 Phase A 命中。

**Trigger signal**
"我手上有 2+ 个做同样事的东西且知道哪个好" → 立刻套 distill recipe：
1. 列两边结构差异
2. 假设去掉每条差异，strong 会不会降级
3. 找"单独移除就会塌"的那一条
4. 抽象它为可复用 principle

---

## 3. Pattern: Orthogonal falsification

**Applies WHEN**
- 刚拿到一个 positive signal 且它 load-bearing
- 项目里有 ≥ 2 个**数据流互不相交**的测量源
- Compute budget 容许额外 1-3 个独立测量

**Does NOT apply WHEN**
- 信号已从 redundant 渠道 confirm 过
- 唯一数据源（无法 diversify）
- 紧急决策不能等
- 每个 intermediate computation 都 check 会 compute 爆炸

**Positive case**
W076 pair-wr=0.64 →
- Exp 1 cross-family judge: 0.40 (FLIP)
- Exp 5 side-shuffle: 0.41 (FLIP)
- Exp 8 n=100 extension: 0.52 (跌穿)
- Exp 9 embedding faithfulness: ≈ 0
- Exp 13 LLM faithfulness: 5% YES
6 个 orthogonal 维度都 FAIL → 确诊 false positive

**Negative case**
如果对 agent 每次 intermediate 推理都要 4 family judge × 3 side-shuffle，compute 炸。Orthogonal falsification 应 reserve 给**commit-level decision**, 不是每个 inner step。

**Trigger signal**
"我要 commit 一个 decision，基于一个 positive signal" → 至少跑 1 个 data-stream-disjoint 的再测一次。

---

## 4. Pattern: Switch role when module fails (MC-WM)

**Applies WHEN**
- 某 module 在角色 X 上 ≥ 2 次失败
- 同一 module 在其他场景 work（不是本质坏）
- 能设计出新 module 接手 X

**Does NOT apply WHEN**
- Module 本质坏（输入 → 输出 map 就错）
- 没有替代 module 可接手 X
- 失败只是参数问题，调参就好（误用 switch）

**Positive case**
- pair-wr 作 "gate"：反复 fail → 让它作 "component of 4" (Exp 15)
- MC-WM SINDy 作 "corrector"：OOD fail → 作 "detector"
- single-family judge 作 "gate"：family bias → 作 "1/3 voter" in majority (Exp 7)

**Negative case**
pair-wr 如果让它作 "prior"、"normalizer"、"confidence regularizer" 等花式角色都不 work ——
问题在于它本身是**混合多源噪声**的量。Switch role 不能救 "量本身是 noise" 的情况。

**Trigger signal**
"这个 module 在当前角色 ≥ 2 次 fail 了" → 问："它在别的角色上是否有结构性价值"。若有，switch；若没有，扔掉。

---

## 5. Pattern: Agent self-rebuttal as prediction

**Applies WHEN**
- Agent 刚 commit 一个 hypothesis
- 任何独立 validation 都**还没跑**
- Budget 容许 1 个额外 LLM call

**Does NOT apply WHEN**
- Validation 已 done（rebuttal 变事后合理化）
- Agent 无 introspection 能力（小模型）
- 时间 critical 不能等 1 min extra call

**Positive case**
Exp 2 在 Exp 1 cross-judge 跑之前，agent 对 W076/W077/W078 提 3 条反驳，
#1 均指向 judge-preference confound。Exp 1 确认 3/3 flip。agent 的 pre-committed 反驳 = 可验证的 prediction。

**Negative case**
若 rebuttal 每条都是泛泛 "further_test / small sample / judge noise"（无 specific mechanism / falsifier）→ 不可证伪，不是 prediction。需在 prompt 里强制**specific mechanism + specific falsifier experiment**。

**Trigger signal**
"刚 commit 一个 KEEP 决定" → 立刻跑 self-rebuttal，记下 #1 confound。后续 validation 看是否命中。

---

## 6. Pattern: Bisection through bug layers

**Applies WHEN**
- Generated code 失败 (runtime error / silent fail)
- 代码大小适中 (< 500 行)
- 能从 stderr / stdout 定位 specific line

**Does NOT apply WHEN**
- 代码跨多文件 + 深调用栈，bisection 成本高于重写
- 没有可 Read 的 generated artifact
- LLM correction 明显能一次修好（simple typo etc.）

**Positive case**
Exp 21 attempt 2 `returncode=0 无输出` → Read code → 发现 3 层叠加 bug：
1. 输出路径错 (`generated/` not `autonomous/`)
2. tokenize 不切 CJK
3. 字段名 `prompt` 应为 `problem`
4. 2 个 evaluator 类被 truncated

每层单独修，不盲目 re-prompt rewrite。

**Negative case**
如果 bug 在一个 100 行的 single function 且 traceback 直指 1 行 → 直接让 LLM 修那行即可，不用 bisection overhead。

**Trigger signal**
"LLM code returncode=0 但不产出期望 output" 或 "traceback 指向多处" → 停止 re-prompt 循环，Read 代码。

---

## 7. Pattern: Null results are findings

**Applies WHEN**
- 测量结果 trivial (all-PASS / all-FAIL / 无 discrimination)
- 期望的是有差异的分布
- 此类 null 可能暴露结构性问题

**Does NOT apply WHEN**
- 有 mixed 结果，只是单个 outlier
- 期望答案本来就是 null（e.g. ablation 期望 "nothing changes"）
- 过度 over-claim 每个 null 都是 finding

**Positive case**
- Exp 21c 12/12 PASS → 停下来命名"orthogonal 不够，需要 conditioning"
- Exp 10 W076/W077/W078 n=100 都跌穿 → 确认 Exp 3 meta-wisdom falsifier

**Negative case**
如果每次 ablation 的 "no effect" 都拿来当 finding → paper 塞满 over-claim，失去 signal-to-noise。

**Trigger signal**
"结果明显缺 discrimination / 所有样本同一 verdict" → 不要调 threshold 让它好看，先诊断 why。

---

## 8. Pattern: Least-to-most construction（user insight）

**Applies WHEN**
- 有 ≥ 1 个现成 repo 和目标任务 ≥ 70% 相似
- 新增贡献 ≤ 1 module 规模
- 能 articulate "我要替换/添加的是哪一块"
- 基线 repo 自己 pass baseline tests

**Does NOT apply WHEN**
- 核心算法前无古人，必须 from scratch
- 所有候选 repo 架构 fundamentally 不同（改一块带动全身）
- 说不清要改什么（需求自身模糊）
- 基线 repo 自己 broken

**Positive case**
- 本 session 大量复用：`validate_parallel.py` 基础设施直接继承；`cached_framework.judge_pair` 直接用；`exp21_data_api.py` 写了个 thin adapter 让 agent 代码少造轮子
- 给 agent 写 gate 时给 `exp21_data_api` 而不是让它从零读 file paths

**Negative case**
- 初始 `auto_recurse.py` 是从 ARCHITECTURE.md 规划的 4 组件**scratch 写**，如果先找 CBR / skill-library 类似 repo 作 base 会快不少
- 如果把 "least-to-most" 误用到"需要新核心算法"场景（e.g. 新 distill algorithm），会被 constrained 到现成架构，扼杀创新

**Trigger signal**
"我要实现 {feature}" 时立刻问：
1. 是否知道 ≥ 3 个 repo 做了**结构上**相似的事？
2. 如果是，选最近的 1 个；明确 "我要换哪一块"；其他保留
3. 如果答案是 "no existing repo"，换去 research path，不硬套

---

## 9. Pattern 本身的 trigger：选 pattern 的 meta-trigger

以上 8 个 pattern 之间**也需要**一个 meta-routing：什么情况下该 reach for 哪个？

快速索引：

| 当下外显信号 | 该 reach for |
|---|---|
| 要开写新代码碰缓存文件 | Pattern 1 Read before write |
| 面前有 2+ 个 artifact 且知道谁强谁弱 | Pattern 2 Cross-artifact distill |
| 刚拿到一个 load-bearing positive signal | Pattern 3 Orthogonal falsification |
| 某 module 在 X 角色 ≥ 2 次 fail | Pattern 4 Switch role |
| 刚 commit 一个 KEEP 且还没跑 validation | Pattern 5 Self-rebuttal as prediction |
| LLM code 失败且 traceback 指向多处 | Pattern 6 Bisection through bugs |
| 结果 trivial / no discrimination | Pattern 7 Null as finding |
| 要实现 feature 且知道 ≥ 3 相似 repo | Pattern 8 Least-to-most |

这 8 个 trigger 是**互斥**的：每条针对一个特定 external signal。**选错 pattern 的根因通常是 misread signal**（把 "代码报错" 当 "要换整个方法"）。

---

## 10. 诚实说：v2 仍不是终点

v2 给每条 pattern 配了 trigger-condition，但：

1. **trigger-condition 本身也是 wisdom** —— 它们也是从 case 里归纳出来的，可能遗漏边界情况
2. **case 库还小** —— 每条只有 1-2 个 positive + 1-2 个 negative。真正 robust 的 trigger boundary 需要 10+ cases
3. **meta-trigger (第 9 节) 手写的** —— 严格讲应该像 Exp 20 那样让 agent 自己提，这一步**没做**

下一步（v3 如果做）：
- 每条 pattern 积累更多 (positive, negative) case
- 对 case 做 cluster → trigger-condition emerge empirically (success_distiller 手法)
- meta-trigger 也让 agent 提一次 → cross-check 和手写版差距（Exp 20 手法）

此时 METHODOLOGY 本身就是一个**按自己规矩长大的 wisdom library**，每次 session 后 append 新 case。

---

## 附录 A — 本 session 每一步 reach for 了哪个 pattern

| Step | External signal | Pattern used |
|---|---|---|
| 初次读 phase4 docs | 要整合已有 pipeline | 1 (Read) |
| success_distiller + validate_parallel 跑 | 有现成 infrastructure | 8 (Least-to-most) |
| 3 KEEPs 出来 user 问"确定 work 吗" | 刚 commit 且未独立 validation | 5 (Self-rebuttal) |
| Exp 1 cross-judge | positive signal load-bearing | 3 (Orthogonal) |
| Exp 7 majority-3 看起来 work 了 | 新 signal，需继续 falsification | 3 (Orthogonal again) |
| Exp 15 trigger-conditioned | pair-wr 作 gate fails 多次 | 4 (Switch role) |
| Exp 20 agent 独立设计 | 2 个 gate (weak/strong) 可对比 | 2 (Cross-artifact) |
| Exp 21 agent code 失败 | LLM code traceback 多处 | 6 (Bisection) |
| Exp 21c 12/12 PASS | trivial discrimination | 7 (Null as finding) |
| Exp 22 agent 诊断 missing move | Gate A vs Gate B 现成对比 | 2 (Cross-artifact at arch layer) |
| 本 v2 重写 METHODOLOGY | v1 scalar list 像 Exp 21c 失败 | 7 + 2 (null → distill) |

每条 step 都能溯源到**哪个 external signal 触发了哪个 pattern**。这才是 "道理 → 每一步对齐" 的实际形态。

---

## 附录 B — 一句话总结

> **Principle 只是一半。另一半是 trigger-condition。没有 trigger，principle 不自我应用。**
>
> METHODOLOGY.md v1 只写了一半，等于在 methodology library 层犯了 wisdom library 层早已诊断过的同一个错。v2 补上另一半。
