# Exp 82 v2 — Complete Summary (从重设计到破口)

**完成日期**：2026-05-01
**状态**：Path 1（cross-judge）+ Path 2（gating composition）+ C2（LLM-classifier proposer）全部跑完
**核心结论**：**v1 的 0/12 在 v2 router + 结构干预架构下被破开 7 个组合**（cross-judge consistency 保留）

---

## 1. 起点 — v1 paper 的 0/12 null

v1 paper 测的是 12 个候选 wisdom（W069–W078 + WCROSSL01）在 50 道 hold-out 题上的 prompt-injection 效果。Gate 要求：

- EXT 答案在 pair-wr 上 ≥ 0.60
- 跨 judge 家族一致
- 比 GENERIC "be careful" 警告显著更好（specificity）
- Trigger fit-rate 合理

**12 个候选全部失败**——任何一个都过不了 gate 的全栏标准。这是 v1 的核心 negative result。

之前的复盘（Exp 70a-d、71、72/73/76/77、78-81、Exp 68）都试图破这个 0/12 但没成功：
- 单次 prompt-injection 的 "格言"边际信息 ≈ 0（modern LLM 训练数据里全是方法学文本）
- 多数 solver 已在 hold-out 上 ceiling，没有 headroom
- copy_semantics 在 deepseek 上的 +29pp 是 8 BASE-OK 试验的小样本巧合
- Exp 68 在 Bayesian 反直觉题上的 +6.7% 是 14/15 ceiling 撞顶

---

## 2. Pivot — 从 "格言级 wisdom" 到 typed Hypothesis（MC-WM 镜像）

### 2.1 启发来源

用户原话："**MC-WM 能用混叠特征的方式找到新假设，我们能不能也用这个方式找到？**"

我去读了 MC-WM 的真实代码（`mc_wm/self_audit/hypothesis.py`、`diagnosis.py`、`constraint_system.py`），发现 MC-WM 实际的工作流是：

```python
@dataclass
class Hypothesis:
    claim: str               # 一句英文/中文描述
    kind: str                # "feature" | "constraint" | "hp_change"
    expr: str                # 真正的表达式/谓词/HP 值（机器可执行）
    expected_metric: str     # 比如 "val_mse"
    expected_min_delta: float = 0.005
    evidence: dict           # A/B counterfactual 测出的实测 Δ
    decision: str            # accepted | rejected | deferred
    failure_reason: str      # redundant | insignificant | destabilising | unsafe
```

每个 Hypothesis 由 **诊断（DiagnosisBattery）→ 提议（LLM Role）→ 测（counterfactual A/B）→ accept/reject → 单调增长库** 这个流水线产生。

**关键洞察**：MC-WM 的"新假设"不是哲理 / 格言 —— 是 `sin(s[1])`、`z >= 0` 这种**机器可执行的谓词或表达式**。

### 2.2 用户决策

> "feature / constraint / decomposition / verification / hp_change 这几个也许需要在初始阶段就通过消融的方式找最合适的。先保留+扩展试试，不行再完全推翻 v1"

最终方案：**保留 v1 的 12 wisdoms + 50 holdout 题作为 seed**，在它们上面扩展出 5 种 kind 的 typed Hypothesis。

### 2.3 五种 kind 的设计

| kind | 注入位置 | 谓词形态 | 测什么 |
|---|---|---|---|
| `feature` | trigger 路由器（调 solver 前）| 给定 problem 返回 0/1 的检测器 | 触发率 vs 子类 |
| `constraint` | solver 输出后置检查 | 给定 (problem, answer) 返回 pass/fail | 强制 retry 直到过 → 终轮 correctness |
| `decomposition` | solver 输入 prompt 模板 | "分 N 步：①…②…③…" 强结构 | EXT vs BASE correctness |
| `verification` | solver 答完后追加 verify 步 | "用具体例子代入验证答案"指令 | EXT vs BASE correctness |
| `hp_change` | solver 调用参数 | `{temperature, max_tokens, top_p}` 改写 | EXT vs BASE correctness |

---

## 3. 实施 — 代码与产物

### 3.1 新建文件（`phase six/exp82/`）

| 文件 | 作用 | 产物 |
|---|---|---|
| `hypothesis.py` | typed Hypothesis dataclass + persistence + `accept_decide()` | `hypotheses.jsonl` |
| `proposers.py` | 5 种 kind 的 LLM-templated proposer | (生成 Hypothesis 时调用) |
| `evaluators.py` | 5 种 kind 的 A/B injection 机制 + forensic logging | `forensic.jsonl` |
| `generate_gold.py` | 用 claude-opus 给 50 题打 gold 答案 | `gold_answers.json` |
| `ablation_runner.py` | 主 runner：propose → eval → aggregate | `ablation_summary.json` |
| `regrade.py` | cross-judge lenient 重打分（gemini + haiku）| `cross_judge_summary.json` |
| `regrade_strict.py` | cross-judge strict rubric 重打分（3 judges）| `cross_judge_strict_summary.json` |
| `feature_eval_v2.py` | feature kind selectivity rubric（替代 correctness Δ）| `feature_eval_v2_summary.json` |
| `feature_classify.py` | LLM-classifier proposer（替代 keyword regex）| `feature_classify_summary.json` |
| `composition_analysis.py` | 后置 gating 分析（feature × structure）| `composition_summary.json` |
| `build_matrix.py` | 从 v1 cache 重建 verdict matrix（早期 SINDy 方案，后保留为 baseline）| `verdict_matrix.json` |
| `features.py` | 从 verdict matrix 抽 problem/wisdom feature（早期 SINDy 方案）| `features.json` |

### 3.2 关键基础设施

- **Forensic logging**：每次 LLM 调用记录完整 prompt + response + 字符级位置 + retries + judge model（用户原话："漏一个标点找你"）
- **Cached BASE answers**：复用 v1 的 `_valp_v20p1_base_answers.json`，BASE 边不重跑
- **3 model families**：gpt-5.4-mini（OpenAI）+ gemini-3-flash（Google）+ claude-haiku-4-5（Anthropic）通过 ruoli.dev proxy
- **Gold answers**：用 claude-opus-4-6 给 50 题写参考答（每题 ~2000 tokens，~$2.5 一次性成本）
- **Resume support**：所有 LLM-heavy 脚本支持断点续跑

### 3.3 模型路由修复

`phase zero/scripts/model_router.py` 之前只读 legacy `CLAUDE_PROXY_API_KEY` 等老变量名，更新为优先读 `RUOLI_*` env vars（live 在 `~/.api_keys`），fallback 老命名。这是基础设施修，影响所有 phase。

---

## 4. 实验时间线

### Phase 0 — Seed 选择

用 exp17 trigger labels（`phase four/autonomous/exp17_trigger_labels.json`）按 SHOULD_FIRE 数排序：

| cid | aphorism | SHOULD_FIRE | NO_FIRE |
|---|---|---|---|
| WCAND10 | 没有调查，就没有发言权 | 26 | 6 |
| WCAND09 | 不谋全局者，不足谋一域 | 25 | 5 |
| WCAND01 | 上工治未病 | 22 | 11 |
| WCROSSL01 | 是骡子是马，拉出来遛遛 | 21 | 5 |
| WCAND02 | 别高效解决一个被看错的问题 | 14 | 8 |
| WCAND03 | 凡事预则立，不预则废 | 13 | 11 |
| WCAND04 | 急则治其标，缓则治其本 | 11 | 15 |
| WCAND08 | 想理解行为，先看激励 | 9 | 21 |
| WCAND05 | 凡益之道，与时偕行 | 5 | 24 |
| WCAND06 | 覆水难收 | 3 | 34 |
| WCAND07 | 亲兄弟，明算账 | 2 | 38 |
| WCAND11 | 不谋全局者 (dup) | 2 | 39 |

**Small-3 seeds**：WCAND10, WCAND09, WCAND01（trigger-fit-rate top 3）
**Full ablation seeds**：9 wisdoms 通过 MIN_TRIGGER_PIDS=5 筛选（跳过 WCAND06/07/11）

### Phase 1 — Gold 答案生成

50 题 × claude-opus-4-6 × 2000 max_tokens × 0.2 temperature = 50 个参考答案。整套 ~7 分钟。
**注意**：首次跑遇到 `.env` 里 `CLAUDE_PROXY_API_KEY` 过期 → 401 全失败 → 修 `model_router.py` 后通。

### Phase 2 — Small-3 Ablation（3 wisdoms × 5 kinds = 15 hypotheses）

用 gpt-5.4-mini 同时作 solver 和 judge。每 hypothesis 测 15 trigger + 5 outside = 20 cells × 3 conditions (BASE/EXT/GENERIC) = 60 LLM 调用，加 grading = ~120 calls/hyp。总计 ~1500 solver+judge calls，~$3-5。

**结果**：

| kind | n | accept | trig Δ | out Δ | EXT-GEN |
|---|---|---|---|---|---|
| feature | 3 | 0/3 | +0% | +0% | -33% |
| **constraint** | 3 | **3/3** | **+31%** | +7% | +4% |
| decomposition | 3 | 2/3 | +18% | +0% | -4% |
| verification | 3 | 2/3 | +7% | +7% | -20% |
| **hp_change** | 3 | **3/3** | **+36%** | **+27%** | +4% |

**Specificity-passing winners**（trig +Δ, out ≈ 0, beats GENERIC）：
- WCAND10/constraint：trig +40%, out +0%, vs GEN +13%
- WCAND09/decomposition：trig +33%, out +0%, vs GEN +20%
- WCAND01/constraint：trig +33%, out +0%, vs GEN ±0%

**初步结论**：似乎破了 0/12——但 solver 和 judge 同模型有 bias 风险。

### Phase 3 — A: Cross-Judge Lenient Regrade（gemini + haiku）

用 `regrade.py` 重打分 15 个 hypothesis × 3 conditions × 20 pids × 2 新 judge = 1800 grade calls，~25 分钟。

**结果**：

| judge | BASE | EXT | GEN |
|---|---|---|---|
| gpt_mini (orig) | 58.7% | 76.9% | 86.7% |
| **gemini (lenient)** | **99.6%** | 95.6% | 97.3% | ← 几乎全 pass |
| **claude_haiku (lenient)** | **98.7%** | 96.0% | 97.8% | ← 几乎全 pass |

**Gemini/Haiku 是 judge ceiling**——它们用 lenient 标准把 99% 答案都判对，无 headroom 测 Δ。0/15 跨 judge 共识——但这是 judge 太宽松，不是 EXT 没改进。

### Phase 4 — A2: Cross-Judge Strict Rubric Regrade（3 judges）

改 grade prompt 让 judge 先从 gold 抽 3-5 checkpoint，再判 candidate 覆盖。`regrade_strict.py` 跑 1800 × 3/2 ≈ 2700 calls，~50 分钟。

**结果（small-3 数据）**：

| judge | BASE | EXT | GEN |
|---|---|---|---|
| gpt_mini_strict | 40.4% | 56.4% | **71.1%** |
| **gemini_strict** | **1.8%** | 1.3% | 2.2% | ← strict 下崩溃（small-3 子集巧合）|
| claude_haiku_strict | 30.7% | 47.1% | **72.9%** |

**Strict 下 GENERIC 一致最强**——EXT 比 BASE 好但**输给"小心"警告**。
2-of-3 majority 下 5/15 trig +Δ，但其中只有 **WCAND09/verification + WCAND09/hp_change** EXT > GEN。

### Phase 5 — B: Full Ablation（9 wisdoms × 5 kinds = 45 hypotheses）

`SEED_CIDS="WCAND01,...,WCROSSL01"` 跑 `ablation_runner.py`。9 usable wisdoms × 5 = 45 hypothesis，每个 evaluated on 15 trig + 5 outside。约 **5 小时 wall-time**，~4500 solver+judge calls。

**Per-kind aggregate（gpt_mini judge，未 cross-judge）**：

| kind | n | accept | trig Δ | out Δ | EXT-GEN |
|---|---|---|---|---|---|
| feature | 9 | 0/9 | +0% | +0% | -33% |
| **constraint** | 9 | **9/9** | +30% | +20% | -4% |
| decomposition | 9 | 6/9 | +15% | +2% | -15% |
| verification | 9 | 7/9 | +19% | +7% | -13% |
| **hp_change** | 9 | 7/9 | +26% | +11% | -3% |

`constraint 9/9` 高 accept 但 outside Δ +20%（confound——结构化要求帮所有题）。
`decomposition trig +15 / out +2` 是 specificity 最干净的 kind，但输给 GEN -15%。

### Phase 6 — B2: Full Cross-Judge Strict Regrade

`regrade_strict.py` 跑 45 hyp × 3 judges × 3 conditions × 20 pids = 8100 grade calls，~3.5 小时。

**Strict pass rates 全档（n=560）**：

| judge | BASE | EXT | GEN |
|---|---|---|---|
| gpt_mini_strict | 43.2% | 56.4% | 70.2% |
| gemini_strict | 37.5% | 39.1% | 45.9% | ← 在大样本下不再崩溃 |
| claude_haiku_strict | 55.2% | 62.9% | 78.9% |

3 judge 都看到 BASE < EXT < GEN。Gemini-strict 在 n=560 下 37.5% pass，证明 small-3 的 1.8% 是子集采样巧合。

**严格 cross-judge winner**（trig Δ ≥ +5pp 在 3 judges 全部）：

| | wisdom/kind | trig (gpt/gem/haiku) | spec3 | >GEN3 | all3 |
|---|---|---|---|---|---|
| ⭐⭐⭐ | **WCAND03/hp_change** | +23/+23/+38 | ✓ | ✓ | ✓ |
| ⭐⭐ | WCAND04/verification | +27/0/+27 | ✓ | ✓ | (gem=0) |

**Path 1（uniform）最终结论**：1/45 严格过全三栏（WCAND03/hp_change，但 hp_change 是 temperature=0 通用收益）。
非 confound 的清洁候选：WCAND04/verification 但 gemini Δ=0% 不过 all3。

### Phase 7 — C: Feature Evaluator 重设计

发现 feature kind 在 0/3 + 0/9 全失败的原因：原 evaluator 把 ext_answer 设为 base_answer（feature 不修改 solver 输出），导致 Δ_ext_base = 0 → 永远 rejected。

**Fix**：feature 是 **trigger router**，不是 solver-modifier。正确指标是 **selectivity**：

```
selectivity_lift = fire_rate(SHOULD_FIRE) - fire_rate(NO_FIRE)
accept iff: selectivity_lift ≥ 0.30 AND fire_rate(NO_FIRE) ≤ 0.30 AND fire_rate(SHOULD_FIRE) ≥ 0.50
```

`feature_eval_v2.py` 跑（**0 LLM 调用**，纯本地 regex/keyword 匹配）：

| feature/wisdom | fire SHOULD | fire NO | lift | F1 |
|---|---|---|---|---|
| WCAND10 | 0/26 (0%) | 0/6 (0%) | 0.00 | 0.00 |
| WCAND09 | 3/25 (12%) | 1/5 (20%) | -0.08 | 0.19 |
| WCAND01 | 1/22 (5%) | 1/11 (9%) | -0.05 | 0.08 |

**0/3 accepted under v2 selectivity bar**。但揭示：**真正失败的是 proposer**——LLM 用 keyword regex 没法捕捉抽象的"问题应该 fire"信号。WCAND09 甚至**反向**（NO_FIRE 命中率高于 SHOULD_FIRE）。

### Phase 8 — C2: LLM-as-Classifier Feature Proposer

`feature_classify.py` 用 3 cheap-tier LLM 各自做分类器：给定 (wisdom, problem) 输出 fired 0/1。9 wisdoms × 50 pids × 3 classifiers = 1281 LLM calls，~30 分钟。

**3-classifier 共识（≥2 of 3 同意）**：

| wisdom | fire SHOULD | fire NO | lift | F1 |
|---|---|---|---|---|
| WCAND01 | 86% | 0% | +0.86 | 0.79 |
| WCAND02 | 93% | 0% | +0.93 | 0.76 |
| WCAND03 | 92% | 0% | +0.92 | 0.83 |
| WCAND04 | 91% | 0% | +0.91 | 0.67 |
| WCAND05 | 100% | 8% | +0.92 | 0.50 |
| WCAND08 | 89% | 0% | +0.89 | 0.76 |
| WCAND09 | 80% | 0% | +0.80 | 0.75 |
| WCAND10 | 77% | 0% | +0.77 | 0.78 |
| WCROSSL01 | 76% | 0% | +0.76 | 0.80 |

**9/9 wisdoms 全过 v2 selectivity bar**——LLM-classifier 跨 3 个家族独立判定一致，本质上**重现了 exp17 的人工 SHOULD_FIRE/NO_FIRE 标注**。

**从 0/3（regex）→ 9/9（LLM-classifier）**——proposer 选择是真正的瓶颈。

### Phase 9 — Path 2: Composition Gating（feature × structure）

关键观察：C2 给出可靠的 router → A2 给出 uniform EXT 跨 judge 的 per-pid 0/1 grades。Compose post-hoc，**0 新 LLM 调用**：

```
gated_ext_correct(cid, kind, pid, judge) =
    ext_correct(cid, kind, pid, judge)   if feature_consensus_fire(cid, pid)
    base_correct(cid, kind, pid, judge)  else
```

`composition_analysis.py` 跑：

**Per-kind aggregate（9 wisdoms × 3 judges）**：

| kind | uniform trig | gated trig | uniform out | **gated out** | uniform spec | **gated spec** |
|---|---|---|---|---|---|---|
| constraint | +8.1% | +6.0% | +6.7% | **+0%** | +1.4% | **+6.0%** |
| decomposition | -1.4% | -2.5% | -0.7% | **+0%** | -0.6% | -2.5% |
| verification | +7.1% | +4.2% | +3.0% | **+0%** | +4.1% | +4.2% |
| **hp_change** | +19.7% | +16.8% | +11.1% | **+0%** | +8.5% | **+16.8%** |

Gating 的两个固定结构性效应：
1. **outside Δ → 0**（by-construction：router 在 NO_FIRE 上 ~0% 触发，全 fallback BASE）
2. **trigger Δ × 0.77-0.92**（router recall 76-93%，剩余 fallback BASE）

净效果：**specificity 被架构强制保证**，所有 36 (kind × wisdom) 通过 outside ≤ +5%。

**7 个跨 judge winner（gated trig Δ ≥ +5% 跨 3 judges + gated out ≤ +5%）**：

| | wisdom/kind | gated trig (gpt/gem/hk) | aphorism |
|---|---|---|---|
| ⭐ | **WCAND01/constraint** | +27% / +7% / +40% | 上工治未病 |
| ⭐ | **WCAND01/decomposition** | +7% / +13% / +27% | 上工治未病 |
| ⭐ | **WCAND01/verification** | +7% / +13% / +20% | 上工治未病 |
| | WCAND01/hp_change | +20% / +20% / +7% | 上工治未病 |
| | WCAND02/hp_change | +36% / +7% / +21% | 别高效解决一个被看错的问题 |
| | WCAND03/constraint | +23% / +8% / +15% | 凡事预则立，不预则废 |
| | WCAND03/hp_change | +23% / +23% / +38% | 凡事预则立 |

**WCAND01（上工治未病）4/4 kinds 全过**——是最稳定的"破口"种子。WCAND03 2/4。

非 hp_change（非通用 temperature 收益）的清洁 winner：
- WCAND01/constraint, WCAND01/decomposition, WCAND01/verification
- WCAND03/constraint

---

## 5. v1 vs v2 — 终极对比

| 标准 | v1（uniform prompt-inject）| v2（uniform） | **v2（router gated）** |
|---|---|---|---|
| trig Δ ≥ +5pp 跨 3 judge | 0/12 | 1/45 | **7/36** |
| 同时 outside ≤ +5pp（specificity） | 0/12 | 1/45 | **7/36** |
| feature kind selectivity F1 ≥ 0.5 | n/a | 0/9 (regex) | **9/9 (LLM-classifier)** |

**0/12 在 v2 router + 结构干预下被 7 个组合破开**。

架构：
```
problem
   │
   ▼
[LLM-classifier router (3-classifier consensus)]
   │
   ├── 不触发 → BASE 直接答（保持 outside ≈ 0）
   │
   └── 触发 → 注入 (constraint | decomposition | verification | hp_change)
           │
           ▼
        solver call → EXT 答
```

这是 MC-WM "诊断 → 针对性介入" 范式在 LLM 问题求解上的实例化。

---

## 6. Caveats — 老实承认的限制

### 6.1 仍然没解决的问题

1. **EXT 输给 GENERIC**（"be careful"）在 strict rubric 下 - GEN 70-79% > EXT 56-62%
   - 即 wisdom-shaped EXT 比 BASE 好，但单独打不过通用警告
   - 7 个 gated winner **没测过 vs gated-GENERIC**（如果 GENERIC 也用 router 自适应注入，可能也产出 spec3）
   - **未解决的关键问题**：router 的好处来自架构本身，还是真的来自 wisdom 的内容？

2. **hp_change confound**
   - 多个 hp_change winner 实质是 "temperature=0" 通用收益
   - 4/7 gated winner 是 hp_change

3. **judge ceiling / floor 风险**
   - lenient rubric：gemini/haiku 99% BASE pass，无 headroom
   - strict rubric：gemini 在 small-3 上 1.8%（floor），在全档上 37.5%（恢复）
   - rubric 形状显著影响结论——cross-judge 共识不是 robust 的判定

4. **小样本**
   - trigger 15 / outside 5 per cell → 单 cell Δ 的 95% CI 很宽（约 ±20pp）
   - 跨 9 wisdoms 聚合改善但仍不达 paper-grade

### 6.2 实验设计偏差

5. **Gold 由 claude-opus 写**，judge 由 cheap-tier 打分 → 评分可能偏好 opus 风格
6. **Solver 是 gpt-5.4-mini**（cheap），可能过强或过弱
7. **Trigger labels 来自 exp17 LLM 标注**，不是人工 ground truth——LLM-classifier 跨家族一致只是证明"LLM 之间一致"，不是"真正对"

### 6.3 未验证的步骤

8. **Composition 假设 gating 是硬决策**：classifier 0/1 输出 → 用 ext 或 base。没测软 gating（用 classifier 置信度加权）
9. **Outside subset n=5 太小**——specificity 的检验力弱
10. **未测过非 WCAND01/WCAND03 wisdom 在 router 加持下是否也能贡献**——可能只是这两条特别契合 router 架构

---

## 7. 文件清单

### 7.1 代码（`phase six/exp82/`）

```
hypothesis.py             # Hypothesis dataclass + accept_decide()
proposers.py              # 5 种 kind 的 LLM proposer
evaluators.py             # 5 种 kind 的 A/B injection + forensic log
generate_gold.py          # 50 题 gold 答案生成（claude-opus）
ablation_runner.py        # propose → eval → aggregate
regrade.py                # lenient cross-judge re-grade
regrade_strict.py         # strict (checkpoint-based) cross-judge re-grade
feature_eval_v2.py        # feature selectivity rubric（regex 检测器）
feature_classify.py       # LLM-classifier feature proposer
composition_analysis.py   # post-hoc feature × structure gating
build_matrix.py           # v1 verdict matrix 重建（SINDy 早期路线，保留为 baseline）
features.py               # v1 cache 抽取 problem/wisdom feature（早期 SINDy 路线）
```

### 7.2 数据产物（`phase six/exp82/`）

```
gold_answers.json                       # 50 道题的 claude-opus 参考答
verdict_matrix.json                     # v1 (W, P, J, verdict) 矩阵 928 行
features.json                           # SINDy 早期路线的特征矩阵

# Small-3 ablation（archived）
hypotheses_small3.jsonl                 # 15 hypothesis 的 typed 记录
forensic_small3.jsonl                   # small-3 的所有 LLM 调用 forensic
ablation_summary_small3.json
ablation_small3.log

# Full ablation
hypotheses.jsonl                        # 45 hypothesis 的 typed 记录
forensic.jsonl                          # full ablation 所有 LLM 调用 forensic
ablation_summary.json                   # per-kind aggregate（gpt_mini judge）
ablation_full.log

# Cross-judge re-grade
cross_judge_summary.json                # lenient: gemini + haiku
cross_judge_strict_summary.json         # strict: 3 judges, checkpoint-based
regrade_forensic.jsonl
regrade_strict_forensic.jsonl
regrade.log
regrade_strict.log
regrade_strict_full.log

# Feature redesign
feature_eval_v2_summary.json            # regex 检测器 selectivity
feature_classify_summary.json           # LLM-classifier 3-clf consensus selectivity
feature_classify_log.jsonl
feature_classify.log

# Composition gating
composition_summary.json                # 7 个 gated winner + per-kind aggregate
```

### 7.3 修改的代码

- `phase zero/scripts/model_router.py` — 更新 env var 解析以兼容 `RUOLI_*` 命名 + legacy fallback

---

## 8. 决策点 — 接下去做什么

### 8.1 立即可做（无 LLM 调用）

**问题 A**：7 个 gated winner 中，多少在测过 vs gated-GENERIC 后仍保留？
- 实施：扩展 `composition_analysis.py` 计算 gated EXT vs gated GENERIC（GENERIC 也用 router 自适应注入）
- 成本：纯 post-hoc，0 LLM 调用
- 价值：分离"router 架构通用收益" vs "wisdom 内容特异收益"

### 8.2 中等成本（~$5-10）

**问题 B**：WCAND01/WCAND03 的破口在 stronger solver（如 gpt-5.4 expensive tier）上是否保留？
- 实施：用 gpt-5.4（expensive）作 solver 重跑 winner hypothesis（4 个非 hp_change）
- 成本：4 hyp × 20 pids × 4 conditions × $0.04 ≈ $13

**问题 C**：能不能找出 WCAND01 之外其他 wisdom 也能产 winner？
- 实施：跑剩下 3 个 wisdom（WCAND06/07/11，被 MIN_TRIGGER_PIDS=5 跳过）用更小 trigger n
- 价值：检验是不是只有 WCAND01 这一条 wisdom 特别契合 router 架构

### 8.3 大成本 / 大改动

**问题 D**：跨 problem domain 重现（v1 hold-out 是混合域，可能 WCAND01 在某些域上特别强）
- 实施：按 problem 的 domain 子集（business/engineering/software/math/daily_life/science）分桶 aggregate
- 价值：找出 wisdom × kind × domain 的稀疏 winner table

**问题 E**：写 paper v2
- 用户已明确指示："实验有效果之前，再也不要改论文了"
- 现在 v2 (gated) 有 7 个 winner——**够格触发 paper v2 的开始**
- 但 caveat（vs gated-GENERIC 没测、hp_change confound、小 n）需要先解决

---

## 9. 用户对话中的关键 directive 与决策

- **"实验有效果之前，再也不要改论文了"** —— v2 keystone 出来前不动 paper
- **"漏一个标点找你"** —— forensic logging 必须完整
- **"投 api，花时间，继续，破 0/12 是 holy grail"** —— path B（不计成本）已激活
- **"feature / constraint / decomposition / verification / hp_change 通过消融找最合适的"** —— ablation 优先于猜
- **"先保留+扩展试试，不行再完全推翻 v1"** —— v2 是 v1 的扩展，不是替代
- **"A 最先 B 之后 C 最后，但是确保 3 个都要做"** —— 全流程不抄近路

---

## 10. Final Gate — gated-EXT vs gated-GENERIC（决定性结果）

`composition_vs_generic.py` 跑后置分析：

```
gated_ext_correct(pid) = ext_correct if feature_fires else base_correct
gated_gen_correct(pid) = gen_correct if feature_fires else base_correct
```

如果 wisdom 内容是真信号，gated_ext > gated_gen 应跨 judge 保留。

### Path 2 的 7 个 winner 在 vs gated-GEN 下的存活

| wisdom/kind | gpt/gem/hk Δ(gated E - gated G) | all3 ≥ 0? |
|---|---|---|
| WCAND01/constraint | -13% / +13% / +13% | ❌ |
| WCAND01/decomposition | -7% / -13% / -7% | ❌ |
| WCAND01/verification | -13% / +13% / -13% | ❌ |
| WCAND01/hp_change | 0% / -7% / +7% | ❌ |
| WCAND02/hp_change | 0% / -14% / -14% | ❌ |
| WCAND03/constraint | -23% / -8% / -15% | ❌ |
| **WCAND03/hp_change** | 0% / 0% / +8% | ✓（仍 HP confound）|

**1/7 存活——且 hp_change 是温度通用收益**。

### Per-kind aggregate（9 wisdoms × 3 judges）

| kind | gated E-B | gated G-B | **gated E-G** |
|---|---|---|---|
| constraint | +6.0% | +14.3% | **-8.3%** |
| decomposition | -2.5% | +16.6% | **-19.1%** |
| verification | +4.2% | +20.0% | **-15.9%** |
| hp_change | +16.8% | +18.6% | **-1.8%** |

**4/4 kind 下 gated GENERIC ≥ gated EXT**——router 架构给所有提示干预都加 buff，但通用"小心"的 buff 反而更大。

### 全 36 hypothesis 在 gated E-G ≥ 0 跨 3 judge 下：

- 6/36 通过（其中 4/6 是 hp_change）
- 非 hp_change 通过者：WCAND04/verification + WCAND09/decomposition（都是 min=0 即勉强不输）

### 真正成立的 v2 claim

> **"LLM-as-classifier 路由架构对所有形式的提示干预都提升 correctness on trigger 子集且 by-construction 保证 outside 不退化。但具体 wisdom 内容的边际贡献不显著——4/4 kind 上 gated-GENERIC 平均 ≥ gated-wisdom-EXT。"**

这是 paper-grade null result：**不是 "wisdom 破 0/12"，是 "router 架构破 0/12，wisdom 内容是 noise"**。

---

## 11. 终极一句话总结

> v1 的 0/12 prompt-injection null **不是被 wisdom 内容破开的，是被 "LLM-as-classifier router + 触发时注入任意形式干预" 这个架构破开的**。在同一 router 加持下，通用 "be careful" 警告在 4/4 kind 上的边际收益**等于或大于**具体 wisdom 衍生的 (constraint/decomposition/verification) 干预。仅 1/36 hypothesis 在最严标准（gated E ≥ gated G 跨 3 judges, ≥ +5pp）下存活——WCAND03/hp_change，但本质是 "temperature=0" 通用 HP 收益。

**核心发现**：
1. ✅ LLM-as-classifier router 跨 3 个模型家族独立判定一致（9/9 wisdoms 高 selectivity）——这是真的架构发现
2. ✅ Router 触发时注入 prompt 改造 > 无干预（trig correctness 显著提升）
3. ❌ Wisdom 具体内容 vs 通用警告：等价或更差（4/4 kind 上 gated-GEN ≥ gated-EXT）
4. ⚠ 仅剩 1 个 hp_change winner（temperature=0 通用收益，不属"wisdom"范畴）

**v2 paper 的真实卖点应是**：
- "我们识别到一个非平凡的架构 contribution（LLM-classifier router）"
- "但 wisdom-as-prompt-content 在该架构下仍是 noise"
- "这是 v1 0/12 的根因诊断 + 部分解决"——比 "破 0/12 by wisdom" 更诚实、更可发表
