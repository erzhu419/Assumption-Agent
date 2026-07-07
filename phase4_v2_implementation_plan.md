# Phase 4 v2 实施方案: Residual-Driven Paradigm Shift

写于 2026-04-23，在 v16 架构稳定后、准备做 Phase 4 真正突破之前。

## 核心原则: **保守优先，激进补位**

科学史的通用模式:
1. 理论 T 有 residual → 先**修补 T**（refine parameters / add perturbations）
2. 修补失败 → 才**提新理论 T'**（新本体 / 新 foundation）

我们的系统也按这个 cascade 走。不要一上来就"提新 wisdom"——大部分 residual 其实是"旧 wisdom 用错了"或"旧 wisdom 表述不精确"。

---

## 架构总览

```
┌────────────────────────────────────────────────────────┐
│  Data Pipeline (持续收集，后台运行)                      │
│  ─ 每次 v16 gen 保存：problem + wisdoms + draft +       │
│    revised + judgment + judge_reasoning                │
└────────────────────┬───────────────────────────────────┘
                     │ N ≥ 200 累积后触发
                     ▼
┌────────────────────────────────────────────────────────┐
│  Residual Detection                                    │
│  ─ 筛选：v16 vs baseline_long 输 + judge 有明确原因      │
│  ─ 聚类：GPT-5.4 看 ~50 loss samples → 找系统性 pattern  │
└────────────────────┬───────────────────────────────────┘
                     │ 如果发现 systematic cluster
                     ▼
┌────────────────────────────────────────────────────────┐
│  Mode A: Refine Existing Wisdom (保守)                 │
│  ─ 对 cluster 最相关的 existing W, refine signal /      │
│    unpacked / exemplars                                │
│  ─ Held-out 20 题验证: 目标 cluster 上 ≥ +5pp           │
│  ─ Regression 20 题验证: 其他域 ≤ -2pp                  │
└──────┬────────┬────────────────────────────────────────┘
       │PASS    │FAIL
       ▼        ▼
   Commit    Mode B (激进)
             │
             ▼
┌────────────────────────────────────────────────────────┐
│  Mode B: Propose New Wisdom                            │
│  ─ GPT-5.4: 从 cluster 中提炼新 orientation            │
│  ─ Novelty check: embedding sim 与已有 75 条 wisdom    │
│    < 0.65 → 确认非重复                                  │
│  ─ 生成 3 个跨域 exemplars (同 v15 流程)                │
│  ─ Held-out 验证: 同上 (cluster +5pp, regression -2pp) │
└──────┬────────┬────────────────────────────────────────┘
       │PASS    │FAIL
       ▼        ▼
   Commit    Escalate:
   new W     标记 "impossible cluster"
             可能是 LLM 能力天花板
```

---

## 1. Data Pipeline — Residual Collection

### 1.1 当前数据问题

现有 `phase two/analysis/cache/judgments/*_vs_*.json` 保存的是:
```json
{
  "pid": {
    "winner": "baseline_long",
    "score_a": 6, "score_b": 8,
    "reasoning": "B 的答案更具操作性，给出了具体步骤...",
    ...
  }
}
```

**已经有 `reasoning` 字段**——好消息，不需要改生成管线。

### 1.2 新增: 结构化 residual 记录

每题 residual = **v16 缺什么 orientation 才会输**。需要新字段（GPT-5.4 分析 judgment 后回填）:

```json
{
  "pid": {
    ...existing...,
    "residual_analysis": {
      "what_v16_missed": "没有识别到 X 类约束的隐含 tension",
      "nearest_existing_wisdom": "W042",
      "wisdom_applicability": 0.3,   // 0-1, 现有 wisdom 能覆盖这 residual 的程度
      "proposed_refinement": "W042 的 signal 字段应扩展到包含 Y 情境",
      "cluster_tag": "implicit_tradeoff_blindness"
    }
  }
}
```

### 1.3 脚本: `phase four/residual_analyzer.py` (要写)

```python
for pid in judgments:
    if judgment[pid]["winner"] == opponent:  # v16 输了
        prompt = """
        Problem: {problem}
        v16 answer: {v16_answer}
        Winning answer: {winner_answer}
        Judge said: {reasoning}

        v16 used wisdoms: {list of wisdom_ids + aphorisms}

        Your task:
        1. What orientation did v16 miss that the winner had?
        2. Is it covered by existing wisdom (which one)? or novel?
        3. If existing but misapplied, how should its signal/unpacked be refined?
        4. Label with a short cluster_tag (5-10 字)
        """
        → GPT-5.4 analysis
        → append to residual_analysis field
```

**成本**: 200 losses × GPT-5.4 ≈ $3-5，~15min。

### 1.4 聚类

```python
all_clusters = defaultdict(list)
for pid, r in residuals.items():
    all_clusters[r["cluster_tag"]].append(pid)

# 只处理 |cluster| ≥ 5 的（少于 5 个可能是噪声）
systematic_clusters = {k: v for k, v in all_clusters.items() if len(v) ≥ 5}
```

---

## 2. Mode A — Refine Existing Wisdom

### 2.1 触发条件

`systematic_clusters` 中每个 cluster 的 `nearest_existing_wisdom` 众数 W*，且 `wisdom_applicability` 平均 > 0.4。

意思: LLM 觉得**已有 W\* 应该能处理这类问题，但没用好**。

### 2.2 Refinement 生成

```python
for cluster_tag, cluster_pids in systematic_clusters.items():
    W_star = mode(residuals[pid]["nearest_existing_wisdom"] for pid in cluster_pids)
    current_signal = wisdom_library[W_star]["signal"]
    current_unpacked = wisdom_library[W_star]["unpacked_for_llm"]
    sample_residuals = [residuals[p]["proposed_refinement"] for p in cluster_pids[:10]]

    prompt = """
    Current wisdom W*:
      signal: {current_signal}
      unpacked: {current_unpacked}

    V16 failed on 5+ problems in cluster "{cluster_tag}".
    Reason: {sample_residuals}

    Refine signal and unpacked so it covers this cluster while preserving existing coverage.
    Don't add domain-specific terms. Must stay abstract.
    """
    → GPT-5.4 produces refined W_star
```

### 2.3 验证协议

**Target test** (cluster 内 20 问题):
- 10 从 cluster_pids 取（已收集的 seed=42 数据）
- 10 从 held-out seed=7 里找同 cluster_tag 的
- 跑 v16 with refined wisdom vs v16 with original
- 要求: +5pp 以上

**Regression test** (非 cluster 随机 20 问题):
- 从 seed=42 中非 cluster_pids 里随机取 20
- 跑 v16 with refined vs original
- 要求: -2pp 以内（不能变差太多）

**决策**:
- PASS → commit 到 `wisdom_library.json`（更新 W_star）
- FAIL → 进入 Mode B

---

## 3. Mode B — Propose New Wisdom (激进)

### 3.1 触发条件

Mode A fail **或** `wisdom_applicability` 平均 < 0.3（现有 wisdom 连"有点相关"都算不上）。

这种情况表明: **现有 library 里压根没这类 orientation**。

### 3.2 新 wisdom 生成

```python
for cluster_tag in mode_b_clusters:
    sample_residuals = [residuals[p] for p in cluster_tag_pids[:10]]

    prompt = """
    10 problems share a failure pattern: {cluster_tag}

    Here are their missed orientations (what v16 failed to apply):
    {what_v16_missed for each}

    Existing wisdom library (75 conditions) does NOT cover this pattern well.

    Propose a NEW wisdom entry with:
      - aphorism: ≤35 中文字符 (not a rewording of existing wisdom)
      - source: real attribution to a thinker/text, or "untraceable 民间谚语"
      - signal: when this orientation fires (15-30 字)
      - unpacked_for_llm: 60-120 字 scenario+self-question form
      - cross_domain_examples: 2 examples in very different domains
      - abstraction_check: confirmed no domain-specific tokens

    The aphorism/source should feel like it COULD have existed in human wisdom
    corpus (Bible/Confucius/Kant/Russell/folk sayings). Don't invent sci-fi 哲学家.
    """
    → GPT-5.4 produces W_new candidate
```

### 3.3 Novelty Check (防止就是重复)

```python
existing_embs = sentence_transformer.encode([w["unpacked_for_llm"] for w in library])
new_emb = sentence_transformer.encode([W_new["unpacked_for_llm"]])

max_sim = max(cosine(new_emb, e) for e in existing_embs)

if max_sim > 0.85:
    # 几乎是重复，reject，让 GPT 重试
    retry with "avoid similarity to {max_sim_wisdom}"
elif max_sim > 0.65:
    # 模糊地带，标记需要人类审核
    flag_for_human_review(W_new, max_sim_wisdom)
else:
    # 确认 novel
    proceed
```

### 3.4 Diverse Exemplar 生成

同 `build_diverse_exemplars_v15.py` 流程：对 W_new 在 sample_100 中挑 3 个 cross-domain 判例。

### 3.5 验证协议

同 Mode A 的 target + regression，但 target 更严格:
- 必须在 cluster 内 +8pp（而不是 +5pp）—— 因为新 wisdom 是"大胆假设"，需要更强证据

### 3.6 决策与入库

- PASS → append W_76+ 到 `wisdom_library_v3.json`，mined exemplars 也入 `wisdom_diverse_exemplars.json`
- FAIL → 标记 cluster 为 `impossible_via_wisdom`
  - 可能是 LLM 能力天花板
  - 或者这个 cluster 本身是 ill-defined，residual 只是噪声

---

## 4. MVP 实验计划

### 4.1 数据准备（立即可做）

现有已有的数据可直接用:
- `judgments/phase2_v16_vs_baseline_long.json`: v16 输的问题（`baseline_long=14` 题）
- `judgments/phase2_v16_vs_ours_27.json`: v16 输给 Self-Discover 的问题（`ours_27=39` 题）
- `judgments/phase2_v16_vs_phase2_v13_reflect.json`: v16 输 v13-reflect 的（38 题）

**合并去重后可能有 50-60 个 v16 residual 问题**——够做一轮 Mode A 实验。

### 4.2 步骤

1. **Week 1**: 写 `residual_analyzer.py`，跑 GPT-5.4 分析 ~50 residuals
2. **Week 1**: 人工查看 cluster_tag 质量（GPT-5.4 聚类的靠谱程度）
3. **Week 2**: Mode A 实施——选 top-3 clusters，refine 相关 wisdoms
4. **Week 2**: Held-out 验证（20 + 20 协议）
5. **Week 3**: 如果 Mode A PASS → 更新 library + 跑完整 v16 回归测试
6. **Week 3**: 如果 Mode A FAIL → 上 Mode B（提新 wisdom），重复验证

**成本**:
- GPT-5.4 分析: ~50 × $0.05 ≈ $3
- v16 重跑 held-out: ~50 × 16s = 13min per variant
- 判分 held-out: ~10min
- 总 API 成本估计 < $20 一整轮

### 4.3 什么情况叫成功

- 至少 1 个 cluster 找到 Mode A refinement，pass 验证 → **Phase 4 闭环一次**
- 如果 Mode A 全 fail，Mode B 有 1 个 pass → **Phase 4 激进路径成立**
- 如果两者都 fail → **当前 LLM 能力 + 判例方法 的组合天花板**

---

## 5. 和 MC-WM 的类比（架构对偶）

| MC-WM | Phase 4 v2 |
|---|---|
| Physics-based model (先验) | Wisdom library (先验) |
| Model-predicted trajectory | v16 draft answer |
| Real trajectory | Baseline_long / Self-Discover answer |
| Residual (aleatoric) | Judge reasoning + wisdom_applicability |
| Random residual → irreducible | No systematic cluster → noise, skip |
| Systematic residual → missing model term | Systematic cluster → missing wisdom |
| New model term → verify on held-out | New wisdom → verify on held-out |

**关键同构**: 两者都在做 **"learn from what your current framework can't explain"**。MC-WM 在连续状态空间，Phase 4 v2 在离散 wisdom 空间。

---

## 6. 风险与权衡

### 6.1 主要风险

1. **Judge reasoning 可能不够 specific**
   mitigation: GPT-5.4 分析时要求强制给出 "what_v16_missed" 字段，不允许 "just worse" 这种空话

2. **聚类可能都是小集群 (size < 5)**
   mitigation: 如果 N=200 residuals 还没 systematic cluster，扩到 N=500 再分析

3. **Mode B 提的新 wisdom 其实只是 rewording**
   mitigation: 多层 novelty check (embedding + 人工 spot-check + diverse exemplar 的可辨识度)

4. **Target +5pp / Regression -2pp 阈值设死太严或太松**
   mitigation: 先跑 3 轮看 empirical distribution，再 calibrate

### 6.2 权衡

**为什么不做完整的 recursive hypothesis chain**:
原 Claude.md 愿景是"递归式自我验证"（无限深度）。v2 只做 1-level 递归（提 1 个新 wisdom → 验证）。
理由: 多 level 递归会放大错误传播，在 v2 里先做 single-step 闭环验证工程可行性。

**为什么不 fine-tune LLM**:
保持原 Phase 2 约束: 所有学习在 KB 中，不碰权重。
Fine-tune 是 Phase 5+ 的故事。

---

## 7. 立即可做的第一步

**最小起步任务（0.5 天）**:

1. 写 `phase 4/residual_analyzer.py` (约 80 行 Python)
2. 从已有 3 个 v16 judgment 文件里 extract v16 输了的问题 ID list
3. 跑 GPT-5.4 对每个做 residual_analysis 分析
4. Dump 成 `phase four/residuals/v16_residuals.json`

然后**人肉 review** 5 分钟，看 cluster_tag 质量如何再决定继续 Mode A 还是调整方法。

**这 0.5 天之后我们就知道 Phase 4 v2 能不能跑得动**。
