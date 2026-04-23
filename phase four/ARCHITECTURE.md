# Phase 4 v3 — Autonomous Wisdom Library Evolution

统合 4 组件为一个闭环 agent。

## 组件职责

### 1 — Failure-Driven Candidate Generator (`failure_generator.py`)
- 监听 v20 的 loss/tie 事件
- 累积 residual clusters → GPT-5.4 提新 wisdom
- 输入: judgments + answers 最近 20 问题
- 输出: 0-2 条 candidate wisdom

### 2 — Success-Driven Candidate Generator (`success_distiller.py`)
- 监听 v20 的 Turn 0 rewriting
- 聚类最近 N 题的 `what_changed` 字段
- 稳定聚类 (≥8 items) 用 GPT-5.4 提炼为 wisdom
- 输入: v20_meta files 最近 100 问题
- 输出: 0-3 条 candidate wisdom

### E — Darwinian Pruner (`pruner.py`)
- 每 50 题扫描一次
- 若某 wisdom 在过去 50 题**未被激活** → 标记 deprecated
- 若 deprecated 再 50 题仍未激活 → 从 library 移除
- 若某 wisdom 激活但关联问题**始终输** → 降低 signal 权重

### G — Cross-LLM Distiller (`cross_llm_distiller.py`)
- 针对 v20 **严重输** 的问题（判官给 B 8+ 分差距）
- 用 Opus/GPT-5.4 重解（API 允许的话）
- GPT-5.4 从 Opus 解中蒸馏 "3-flash 没用到的 orientation"
- 输出: 0-1 条 cross-model candidate wisdom

### Orchestrator (`autonomous_loop.py`)
- 推 20 问题 batch → 跑 v20 → 收集 signal
- Trigger 4 个 generator
- 合并 candidates → Novelty check → Held-out A/B
- KEEP/REVERT decision
- Library version bump
- Log to `library_evolution.json`

## 数据流

```
┌──────────────────────────────────────────────────┐
│  Problem Queue (从 test split 持续拉 20 问题)      │
└───────────────────────┬──────────────────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  v20 架构 solve             │
        │  → meta/drafts/answers       │
        └─────┬────────────┬───────────┘
              │            │
    [success] │            │ [failure/tie]
              ▼            ▼
        ┌──────────┐  ┌──────────┐
        │ Success  │  │ Failure  │
        │ Distill  │  │ Generator│
        │ (方向 2) │  │ (方向 1) │
        └────┬─────┘  └────┬─────┘
             │             │
             │             ▼
             │   ┌──────────────────┐
             │   │ Cross-LLM        │
             │   │ Distill (脑洞 G) │
             │   │ (hard residuals) │
             │   └────┬─────────────┘
             │        │
             └────────┴────┐
                           ▼
                  ┌────────────────┐
                  │  Candidate     │
                  │  aggregator    │
                  │  (novelty +    │
                  │   conflict)    │
                  └───┬────────────┘
                      ▼
                  ┌──────────┐
                  │  A/B     │
                  │  Test    │
                  └────┬─────┘
                       │
                       ▼ keep
              ┌────────────────┐
              │  Library Y+1  │
              └──┬────────────┘
                 ▼
         ┌──────────────┐
         │ Pruner       │
         │ (每 50 轮)   │
         │ (脑洞 E)     │
         └──┬───────────┘
            ▼
         Library Y+2 (或 Y+1 after prune)
```

## 关键数据结构

### `library_evolution.json`
```json
[
  {
    "round": 1,
    "timestamp": "2026-04-23T...",
    "library_version": "v20.0",
    "problems_run": [pid1, pid2, ...],
    "held_out_wr_before": 0.62,
    "held_out_wr_after": 0.62,
    "actions": []
  },
  {
    "round": 2,
    "library_version": "v20.1",
    "actions": [
      {
        "type": "add",
        "wisdom_id": "W080",
        "source": "success_distiller",
        "candidate_cluster": "...",
        "held_out_a_b": {"with": 0.65, "without": 0.58},
        "gain": 0.07,
        "committed": true
      }
    ],
    "held_out_wr_after": 0.65
  },
  ...
]
```

### `wisdom_registry.json` — version-aware library
```json
{
  "version": "v20.3",
  "wisdoms": [
    {
      "id": "W001",
      "status": "active" | "deprecated" | "removed",
      "created_at": "2026-04-17",
      "last_activated": "2026-04-23",
      "activation_count": 42,
      "contribution_gain": 0.23,  # 历史激活后 avg held-out gain
      "source": "original" | "failure_driven" | "success_distilled" | "cross_llm",
      ...
    }
  ]
}
```

## 论文 claim (第二篇核心)

> "We demonstrate the first autonomous closed-loop LLM agent that grows its
> methodological wisdom library from BOTH failure-driven residuals AND
> success-driven reframing patterns, with Darwinian pruning of stale wisdoms
> and cross-model knowledge distillation from stronger generators. Over N=300
> problems, the library evolved from 75 → K entries, and held-out performance
> rose monotonically from v20-base by +Mpp, without any human curation."

---

## Implementation 顺序

**Week 1** (MVP + 骨架):
- `wisdom_registry.py` — versioned library data structure
- `autonomous_loop.py` — 主 orchestrator skeleton
- `failure_generator.py` — 方向 1 (Mode B 升级版)
- 先跑 10-20 round，无 success/pruner/cross-LLM，只做 failure-driven

**Week 2** (加 success + prune):
- `success_distiller.py` — 方向 2
- `pruner.py` — 脑洞 E
- 再跑 30 rounds

**Week 3** (加 cross-LLM):
- `cross_llm_distiller.py` — 脑洞 G
- 跑 50 rounds 完整 agent
- 收集 `library_evolution.json`

**Week 4** (写 paper):
- Performance curve 可视化
- Library evolution 叙事
- Paper 初稿
