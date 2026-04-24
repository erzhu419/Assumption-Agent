# 方法论：从提出假设到自我验证

> 本文档记录从 auto_recurse 初始闭环到 Exp 22 L5 self-recovery 的完整方法论。
> 用户在 session 中主要作 direction-setting（选方向、审 framing），execution 全部由 Claude 走。
> 本文诚实记录 Claude 实际走过的**步骤、参考物、失败-修复模式**，而不是事后合理化。

---

## 0. 核心元原则（事后提炼）

所有其他动作都服务于这几条：

1. **Read before write** — 在写任何新 exp 代码前，必先 grep/Read 现有 meta、answer、log、md 文件，确认数据 schema 和已有工具。**80% 的实现 bug 是 data-contract 假设错**。
2. **Orthogonal falsification first** — 任何 positive signal 默认是噪声，直到用**数据流不同源**的独立测量再证实一次。
3. **Cross_llm_distiller pattern 可在任意抽象层级复用** — 这是 project 最基础的推理原语，能作用在 answer、gate、architecture 三个层级。
4. **MC-WM 切角色原则** — 一个 module 在角色 X 上反复失败，不要加固 X，让它做 Y，把 X 交给别人。
5. **Agent 自己的怀疑是 design input** — 在跑 validation 前先让 agent 提反驳，把反驳当作**必须被证伪或证实的 prediction**。
6. **Null results 是发现** — 12/12 PASS 不是"成功"，是"gate 不 discriminate"的信息。

---

## 1. Session 的完整走向

```
初始愿景(user)         agent 能自己提假设 + 自我验证
    ↓
Phase 1 闭环           success_distiller + validate_parallel → 3 KEEPs (W076/77/78)
    ↓
Phase 2 falsification  cross-judge (Claude) → 0/3 STABLE
                       side-shuffle → 0/3 STABLE
                       cross-domain → 多数 FAIL
                       auto_recurse → 3-cycle fixed point at 0 KEEP
    ↓
Phase 3 orthogonal 诊断   embedding-Stage-A: 对齐 ≈ 0
                          LLM-faithfulness: strict YES ≤ 10%
    ↓
Phase 4 architectural     L1 trigger-conditioned gate: 4/12 PASS ← 第一次 discriminate
                          L2 trigger-indexed library
                          L3 citation-aware solver: 95% prospective cite
    ↓
Phase 5 agent self-design agent 独立产 gate 设计 (sim=0.28 + 3 novel axes)
                          agent 自写 460 行 gate code
                          填补 → 12/12 PASS (undiscriminating)
    ↓
Phase 6 agent self-recovery  agent 读自己失败 + researcher 成功
                             独立命名 "condition-stratified measurement"
```

---

## 2. 实际用到的 7 个方法模式

### Pattern 1 — Read before write

**每个新 exp 开始前都做的事**：

```bash
# 1. grep 项目里有没有做过类似的
grep -rn "关键词" phase*/scripts

# 2. 找相关 json schema
python -c "
import json
d = json.loads(open('some_file.json').read())
print(type(d).__name__, list(d.keys())[:3] if isinstance(d, dict) else len(d))
# sample first entry
"

# 3. 找文件 mtime + 是否被 gitignore
ls -la some_cache_dir/ | head
```

**为什么重要**：Exp 21 agent 写 gate code 失败的核心原因就是**不做 Read before write** —— 它假设字段叫 `prompt` 而实际叫 `problem`。我自己每次都先 Read 所以几乎没犯过同类 bug。

**参考物**：
- `manual/phase_four_dev_doc.md` — 原 3-stage validation 计划
- `phase4_v3_directions.md` — 4 directions + 12 脑洞
- `phase two/analysis/cache/answers/*_meta.json` — Turn-0 JSON schema
- `MEMORY.md` / `project_mcwm_architecture_pivot.md` — MC-WM "switch role" 原则
- 各次实验的 traceback 和 stdout tail

---

### Pattern 2 — Cross-artifact distillation (复用 cross_llm_distiller)

**这是整个项目最核心的 primitive，可作用在三个层级**：

| 层级 | weak artifact | strong artifact | distilled thing |
|---|---|---|---|
| Answer (原版) | 3-flash 答卷 | GPT-5.4 答卷 | "orientation W 没用到的" |
| Architecture (Exp 22) | Gate A (12/12 PASS) | Gate B (4/12 PASS) | "condition-stratified measurement" |
| Method (可拓展) | 单一判官 gate | 三家族 majority gate | "triangulation" 作为元原则 |

**Recipe**：
```
1. 取两个做同样任务的产物，一个 strong 一个 weak
2. 用同一评价标准确认哪个好
3. 询问: "strong 做到的哪些事 weak 没做到"
4. 把差异形式化成可复用 principle
5. 把 principle 写回 prompt，让 weak agent 自己试着整合
```

**参考物**：`phase four/cross_llm_distiller.py` — 原版实现，直接改 prompt scope 就能提升层级。

---

### Pattern 3 — Orthogonal falsification

**规则**：任何 positive signal 必须被来自**完全不同数据流**的测量再证实一次。

**实际应用**：
```
pair-wr on holdout_50 → W076 wr=0.64 (positive?)
├─ Exp 1:  cross-family judge on SAME pairs       → claude_wr=0.40 (FLIP)
├─ Exp 5:  side-shuffle (fresh PYTHONHASHSEED)    → gemini re-roll=0.41 (FLIP)
├─ Exp 8:  extend to fresh n=50 problems         → wr_combined=0.57 (gray)
├─ Exp 9:  embedding-space faithfulness alignment → ≈ 0 (no signal)
├─ Exp 13: LLM-judged strict faithfulness        → 5% YES
└─ Exp 15: trigger-conditioned utility           → conditional wr=0.60 on n=26
```

每个都从**disjoint data stream** 测一次。6 个独立维度都同意，才算"真的"。

**为什么 orthogonal 关键**：如果所有测量都从 pair-wr 派生（比如多加几家族 judge 投票），它们共享 judge-preference / position-bias / seed noise，一起错。必须从 embedding、meta、answer-text 这些**没碰 pair verdict 的源**再测一次。

---

### Pattern 4 — Switch role (MC-WM 原则)

**规则**：一个 module 在角色 X 上反复失败时，不要继续调参让它做 X，让它做 Y，把 X 交给新 module。

**实际应用轨迹**：

```
pair-wr 作 gate                    → 失败 (noise dominated)
    ↓ 不加严阈值 (那是继续做 X)
pair-wr 作 component of 4          → 成立，其他 3 components 从 disjoint source 补齐

single judge family 作 gate        → 失败 (family bias)
    ↓ 不扩样本 (那是继续做 X)
single judge 作 1/3 voter         → 成立，加 2 个独立家族作 majority-3

gate 作 all-problems aggregator    → 失败 (conflates fire/no-fire)
    ↓ 不改 formula
gate 作 per-subset conditional     → 成立，先 trigger-label 再聚合
```

**参考物**：`MEMORY.md` 的 `project_mcwm_architecture_pivot.md`:
> SINDy correction fails OOD (-837%) → **SINDy as OOD detector** (not corrector); QΔ Residual Bellman penalizes Q-targets

完全同构的 move，只是映射到 wisdom library 场景。

---

### Pattern 5 — Agent self-rebuttal as prediction

**规则**：让 agent 在跑 validation 之前先提出 **"如果这次 KEEP 其实是假阳性，原因可能是什么"**。把这些预测记录下来。后续 validation 结果要么证实要么证伪这些预测。

**实际应用**：
- Exp 2 agent 预测 W076/W077/W078 的 #1 failure mode 是 "judge-preference alignment"
- Exp 1 cross-judge 运行 → 3/3 flip，#1 confound 正是 judge preference
- **预测命中**：agent 在**没看到 Exp 1 结果时**就说对了

**为什么有用**：比起事后合理化，pre-committed skepticism 让 agent 的诊断能力有 falsifiability。且 agent 往往能列出**人类会忽略的 confound**（比如 library-interaction effect）。

---

### Pattern 6 — Bisection through bug layers

**规则**：当 agent 生成的代码失败，不要直接 re-prompt 要求 rewrite。先 Read 实际代码，识别**具体哪一层**的 bug，一层层修。

**实际应用（Exp 21 debug）**：
```
attempt 0   SyntaxError line 1       → code 开头带 ```python 没清理
attempt 1   returncode 0, no output  → 代码 exit 0 但没写输出文件
实际 Read 代码 →
    发现 1: output path 写到了 generated/ 不是 autonomous/  (wrapper 加 adapter 搜索更多目录)
    发现 2: tokenize 不切 CJK (sed patch)
    发现 3: 字段名 prompt vs problem (sed patch 5 处)
    发现 4: SubstantiveContentDelta 类只有 def 头没 body (truncated)
    发现 5: 后两个 evaluator 根本没生成 (truncated)
```

**每一层的修复方式不同**：wrapper 改、sed replace、手动补写。盲目 re-prompt 会让 LLM 从头重写，可能换一套新 bug。

**参考物**：每次 attempt 的 `stderr tail` + 手动 `head`/`tail`/`grep` 读生成的 .py。

---

### Pattern 7 — Null results 认定为发现

**规则**：不 discriminate、所有候选全过、cross-judge 全 flip 这类结果**本身就是发现**。要停下来命名这个发现，不要继续调参直到"看起来对"。

**实际应用**：
- Exp 17 trigger-conditioned gate: 4/12 PASS → 停下来，这是第一次 discriminate，commit
- Exp 21c agent-designed gate: 12/12 PASS → 停下来，这是 **"orthogonal decomposition 不够，还需要 conditioning"** 的发现
- Exp 10 W076/W077/W078 在 n=100 都跌穿 → 停下来，这证实 Exp 3 meta-wisdom 的 falsifier

**为什么有用**：chasing higher PASS 容易掉进"tune-until-green"。把 null result 当 finding 让 paper 每一步有硬 claim，不靠数字好看。

---

## 3. 具体参考过的文件清单

**项目 md**：
- `phase4_v3_directions.md` — Direction 1-4 + 12 脑洞 roadmap
- `phase four/ARCHITECTURE.md` — 4-组件闭环设计
- `manual/phase_four_dev_doc.md` — 原 3-stage 验证计划

**MEMORY 文件**：
- `project_mcwm_architecture_pivot.md` — SINDy 作 detector（不做 corrector）→ 直接启发 pair-wr 作 component（不做 gate）
- `feedback_mcwm_strategy.md` — "module-by-module validation; rebuild after 2 fails"
- `feedback_mvp_gap_criterion.md` — ">10% solid, <5% 可疑" 量级判断

**已有代码**：
- `phase four/cross_llm_distiller.py` — **最重要**，pattern 2 的原型
- `phase four/validate_parallel.py` — 并行 v20 + cached judge infrastructure
- `phase one/scripts/validation/cached_framework.py` — judge_pair + content cache
- `phase one/scripts/validation/phase2_v20_framework.py` — Turn-0 meta schema
- `phase zero/scripts/llm_client.py`, `gpt5_client.py`, `claude_proxy_client.py` — 三家族客户端

**数据**：
- `phase two/analysis/cache/answers/*_meta.json` — Turn-0 JSON (frame/critical_reframe/anti_patterns/what_changed)
- `phase two/analysis/cache/answers/*_answers.json` — final answer text
- `phase two/analysis/cache/judgments/*.json` — pair verdicts per pid (domain + difficulty)
- `phase two/analysis/cache/sample_holdout_50.json` — 50 test problems (domain + difficulty labels)
- `phase four/autonomous/success_distilled_candidates.json`, `cross_llm_candidates.json` — 12 candidate wisdoms 的元数据

**每次失败 log**：
- `phase four/logs/exp{N}_*.log` — 每个 exp 的 stdout/stderr tail，用于判断下一步
- 每次 subprocess traceback — 决定哪个 bug 先修

---

## 4. 从本 session 学到的可转写模板

### Template A — Cross-artifact distill prompt (Pattern 2 instantiation)

```
== 任务 ==
你要对两个做 {same_task} 的产物做 cross-artifact distillation。

== Weak artifact ==
{weak_spec}
它在 {metric} 上得 {weak_score}。

== Strong artifact ==
{strong_spec}
它在 {metric} 上得 {strong_score}。

== 你的推理步骤（按顺序）==
1. 列出两个 artifact **结构上**的所有差异（不只是参数）
2. 对每条差异，问：如果去掉这条差异，strong 还会 strong 吗？
3. 找出**单独去掉会让 strong 降到 weak** 的那一条 → 它是 strong 的 active ingredient
4. 把 active ingredient 抽象为 general principle（不限此任务）

== 输出 JSON ==
{{...schema...}}
```

### Template B — Self-rebuttal before validation (Pattern 5 instantiation)

```
你刚刚 commit 了 {thing}，证据是 {evidence}。
在任何独立 validation 跑之前，列出 3 条**最强的反驳**：如果这个 commit 其实是假阳性，可能的原因是什么？

每条反驳说明：
- alt_hypothesis:
- mechanism:   为什么这个机制能制造出 {evidence} 的假象
- falsifier:   什么独立实验能明确证伪这条反驳
- severity:    high / medium / low

最后给 overall verdict: accept / further_test / reject
```

### Template C — Orthogonal measurement enumeration (Pattern 3 instantiation)

```
当前你只用了 {data_source_used} 来测量 {quantity}。
枚举项目里其他**没被 gate 使用**的 data sources，对每个问：
- 它能独立回答什么 validity 问题？
- 它和当前 data source 在噪声层是否 disjoint？

从最 disjoint 的至少 3 个里，每个设计一个测量，构造 {quantity} 的 3+ 维 orthogonal replacement。
```

---

## 5. 诚实说"我没做到"的

- **Agent 的 gate code 需要多轮修** — 不是 1-shot，实际是 3 次 attempt + 手工填 3 个 evaluator。agent-only 一次性实现 **没 demonstrate 出来**。
- **n_sample 仍然偏小** — 12 candidates × 50 pids，统计功率对 +10pp 阈值勉强够。
- **Conditioning 这一步 agent 最后 discover 了（Exp 22 Phase A）**，但是是在我喂"cross_llm_distiller 方法论"prompt 之后。**不是 zero-shot**。只能说 "agent 具备这个认知，需要正确的 teaching prompt 激活"。
- **trigger label 本身由 Claude 给**，有单家族 bias 风险，尚未 audit。

---

## 6. 一句话总结方法论

> **先读，再怀疑，测量用不相交的流，失败当数据，已有 primitive 升维用，role 能换就换，每次只修一层 bug，agent 自己的怀疑当 prediction**。

这 8 条组合起来就是本 session 从"agent 随口提 W076"走到"agent 独立命名 condition-stratified measurement"的所有操作。每一条**都已在项目代码里实例化**，后续 session 复用成本 ≈ 0。
