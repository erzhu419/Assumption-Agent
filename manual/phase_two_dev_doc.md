# 阶段二：经验反馈与规则演化——完整开发文档 (v1)

## 0. 文档概述

### 0.1 本阶段在整体架构中的位置

阶段零构建了知识库（静态的人类先验），阶段一构建了调度器（选择策略的能力）。本阶段要解决的问题是：**知识库中的策略适用条件是人类从文献中总结的，它们在实际执行中是否准确？有没有遗漏的条件？有没有需要修正的边界？**

这是整个系统从"装载人类哲学"到"超越人类哲学"的关键转折点。阶段零和阶段一合在一起，本质上还是在利用已有的人类知识。本阶段开始，系统将从自身的执行经验中发现人类文献未曾显式描述的规律——比如"控制变量法在组件耦合度超过 0.6 时开始失效"这种定量边界条件。

**关键设计约束：本阶段不修改调度器的模型权重，也不 fine-tune 任何 LLM。** 所有"学习"都发生在知识库的 JSON 结构中——新增条件、调整置信度、修改适用边界。LLM 仅作为分析工具（通过 prompting）使用，而非训练对象。这与 Letta 提出的"Continual Learning in Token Space"思路一致：学习应该发生在上下文（知识库）中，而非权重中。

这样设计的好处：
1. **零训练成本：** 不需要 GPU 集群来 fine-tune 模型
2. **完全可审计：** 每次知识库更新都有明确的证据链和变更日志，可以被人类审核
3. **即时生效：** 知识库更新后，调度器下次读取时立即使用新规则，不需要重新训练
4. **可回滚：** 如果更新引入了错误规则，可以用阶段零定义的回滚机制恢复

### 0.2 本阶段目标

构建一个经验反馈闭环，使知识库能从调度器的执行经验中持续演化——修正已有策略的适用条件边界，发现策略之间的新组合模式，识别知识库中未记录的隐式规则。

### 0.3 交付物

1. 经验评估器模块（`experience_evaluator/`）：判断每条经验是否值得记录
2. 经验蒸馏器模块（`experience_distiller/`）：从原始轨迹中提取知识库更新候选
3. 知识整合器模块（`knowledge_integrator/`）：将更新候选应用到知识库，包含冲突检测和质量门控
4. 一套完整的更新流程自动化脚本
5. 更新前后的知识库质量对比报告
6. 一篇可投稿的论文，核心贡献：系统通过经验自动发现了已有哲学策略的适用边界条件

### 0.4 时间预算

总计 4-6 个月。经验评估与蒸馏模块开发：4-6 周。知识整合与质量门控：3-4 周。大规模经验收集与更新迭代：6-8 周。分析与论文写作：4 周。

### 0.5 关于不 fine-tune LLM 的设计决策

本阶段完全不涉及 LLM 的参数更新。以下是三个核心模块的实现方式：

| 模块 | 实现方式 | 为什么不需要 fine-tune |
|------|---------|---------------------|
| 经验评估器 | LLM prompting + 规则函数 | 判断经验的信息量是一个分析任务，通用 LLM 的零样本能力足够 |
| 经验蒸馏器 | LLM prompting（对比分析） | 从轨迹中提取规则是一个推理任务，与 MEL 论文的方法论一致 |
| 知识整合器 | 规则函数 + LLM prompting（冲突检测） | 条件合并和置信度更新是确定性计算；冲突检测用 LLM 的语义理解 |

如果未来发现 prompting 的分析质量不足（比如在经验蒸馏的精确度上达不到要求），可以考虑对一个小型 LLM 做轻量级 fine-tune（如 LoRA），但这不是本阶段的默认方案。

---

## 1. 系统架构

### 1.1 总体数据流

```
阶段一的经验日志
(experience_log/executions/)
        │
        ▼
┌───────────────────┐
│   经验评估器        │     "这条经验值得记录吗？"
│  (Experience       │
│   Evaluator)       │     输入: 原始执行记录
│                    │     输出: 信息量评分 + 是否保留
└────────┬──────────┘
         │ 仅保留高信息量经验
         ▼
┌───────────────────┐
│   经验蒸馏器        │     "这条经验对知识库有什么启示？"
│  (Experience       │
│   Distiller)       │     输入: 筛选后的经验 + 知识库当前状态
│                    │     输出: 更新候选 (UpdateCandidate)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  更新候选队列       │     pending_review/ 目录
│  (Update Queue)    │     等待质量门控
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   知识整合器        │     "这个更新安全吗？和已有规则冲突吗？"
│  (Knowledge        │
│   Integrator)      │     输入: 更新候选 + 知识库当前状态
│                    │     输出: 应用/拒绝/人工审核
└────────┬──────────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
  应用  拒绝  人工审核
   │    │      │
   ▼    ▼      ▼
  kb/  rejected/ pending_human/
  更新  归档     等待
```

### 1.2 三个模块与阶段零 schema 的关系

本阶段的三个模块直接操作阶段零定义的数据结构：

- **经验评估器**读取 `experience_log/executions/` 中的记录（阶段零 2.4 节 schema）
- **经验蒸馏器**生成更新候选，写入 `experience_log/distilled/pending_review/`（阶段零 5.1 节目录结构）
- **知识整合器**修改 `kb/strategies/S*.json` 中的适用条件，并写入 `change_history/S*.jsonl`（阶段零 2.5 节 schema）

不引入任何新的数据格式——完全复用阶段零预留的演化接口。

---

## 2. 经验评估器

### 2.1 设计理念

不是所有经验都值得记录。核心的指导原则来自 Claude.md 讨论中引用的实证研究：

> 直接把所有经验都加入记忆的做法表现出平坦或下降的成功率，说明有缺陷的记录在不断积累。相比之下，严格的选择性添加策略随时间持续改善。

经验评估器的任务是**守门**：只有高信息量的经验才进入蒸馏流程。

### 2.2 信息量评分

一条经验的信息量取决于它对知识库的"意外程度"。如果一切都按知识库预测的那样发生了，这条经验的信息量为零——它只是确认了已知规则。只有当结果偏离预测时，才包含新信息。

```python
class ExperienceEvaluator:
    """
    评估单条执行经验的信息量。
    纯规则函数 + LLM prompting，不涉及任何模型训练。
    """
    
    def evaluate(
        self,
        record: ExecutionRecord,
        kb_snapshot: KBSnapshot
    ) -> EvaluationResult:
        
        strategy = record["strategy_selection"]["selected_strategy"]
        outcome = record["outcome"]
        attribution = record["attribution"]
        
        # === 规则评分（确定性计算） ===
        score = 0.0
        tags = []
        
        # 情况 1：策略在 favorable 条件下失败
        # → 高信息量：可能发现了新的 unfavorable 条件
        if (not outcome["success"] and
            len(attribution["matched_conditions"]) > 0 and
            len(attribution["violated_conditions"]) == 0):
            score += 0.8
            tags.append("unexpected_failure")
        
        # 情况 2：策略在 unfavorable 条件下成功
        # → 高信息量：可能该 unfavorable 条件需要被削弱
        if (outcome["success"] and
            len(attribution.get("surprising_successes", [])) > 0):
            score += 0.7
            tags.append("surprising_success")
        
        # 情况 3：发现了新的条件候选
        if len(attribution.get("newly_discovered_condition_candidates", [])) > 0:
            score += 0.6
            tags.append("new_condition_candidate")
        
        # 情况 4：策略成功且完全符合预期
        # → 低信息量，但仍然增加一点（用于置信度微调）
        if (outcome["success"] and
            len(attribution["matched_conditions"]) > 0 and
            len(attribution["violated_conditions"]) == 0 and
            len(attribution.get("surprising_successes", [])) == 0):
            score += 0.1
            tags.append("expected_success")
        
        # 情况 5：策略失败且完全符合预期（在已知的 unfavorable 条件下）
        # → 低信息量
        if (not outcome["success"] and
            len(attribution["violated_conditions"]) > 0):
            score += 0.1
            tags.append("expected_failure")
        
        # === LLM 辅助评分（对边界情况的补充判断）===
        if 0.3 < score < 0.7:
            # 处于模糊地带，用 LLM 做二次判断
            llm_score = self._llm_assess_novelty(record, kb_snapshot)
            score = 0.6 * score + 0.4 * llm_score
        
        # 信息量阈值
        keep = score >= 0.3
        
        return EvaluationResult(
            execution_id=record["execution_id"],
            information_score=score,
            tags=tags,
            keep=keep,
            reason=self._generate_reason(tags, score)
        )
```

### 2.3 LLM 辅助评估 Prompt

对于信息量评分处于模糊地带（0.3-0.7）的经验，使用 LLM 进行补充评估：

```python
LLM_NOVELTY_ASSESSMENT_PROMPT = """
你是一个知识库维护专家。以下是一条执行经验记录和当前知识库中相关策略的适用条件。

## 执行记录摘要
- 任务: {task_description}
- 选用策略: {strategy_name}
- 结果: {outcome_summary}
- 失败原因（如有）: {failure_reason}

## 该策略当前的适用条件
有利条件: {favorable_conditions}
不利条件: {unfavorable_conditions}

## 问题
这条经验是否包含知识库当前未记录的新信息？请从以下维度评估：

1. 这次执行的结果是否在知识库已有条件下可以被预测到？(是/否)
2. 如果不能被预测，原因是什么？是否暗示了一个新的适用/不适用条件？
3. 这个潜在的新条件是否足够通用（在其他任务中也可能适用）？

输出 JSON:
{
    "predictable_from_kb": true/false,
    "novelty_score": 0.0-1.0,
    "potential_new_condition": "条件描述（如有）" 或 null,
    "generalizability": "high/medium/low",
    "reasoning": "简短理由"
}
"""
```

### 2.4 评估器的运行模式

经验评估器有两种运行模式：

**在线模式（Online）：** 每当阶段一的调度器完成一次任务执行并写入经验日志后，立即触发评估。适用于实时运行的系统。

**批量模式（Batch）：** 定期（如每天或每积累 100 条经验后）批量评估所有未处理的经验记录。适用于离线分析。

默认使用批量模式，因为 LLM 辅助评估有 API 调用成本，批量处理可以利用 batch API 降低成本。

---

## 3. 经验蒸馏器

### 3.1 设计理念

经验蒸馏器是本阶段最核心的模块。它的任务是把原始的执行轨迹**压缩**为对知识库的结构化更新建议。

类比人类认知科学中的概念：
- 原始经验 = 情景记忆（episodic memory）
- 蒸馏后的规则更新 = 语义记忆（semantic memory）的增量修改
- 蒸馏过程 = 认知科学中的"睡眠整合"（memory consolidation）

**关键约束：蒸馏器不直接修改知识库。** 它只生成更新候选（`UpdateCandidate`），写入待审核队列。实际应用由知识整合器负责。

### 3.2 更新候选 Schema

```json
{
  "candidate_id": "upd_20260401_001",
  "timestamp": "2026-04-01T10:30:00Z",
  "update_type": "condition_added | condition_modified | confidence_adjusted | condition_moved | relationship_added",
  "target_strategy": "S01",
  
  "proposed_change": {
    "type": "condition_added",
    "placement": "unfavorable",
    "new_condition": {
      "condition_id": "S01_U_NEW_001",
      "condition": "组件之间存在通过共享状态（如全局变量、共享缓存）的隐性耦合",
      "source": "experience",
      "confidence": 0.65,
      "supporting_cases": ["exec_20260301_142301_task42", "exec_20260315_091122_task57"],
      "contradicting_cases": [],
      "status": "under_review",
      "locked": false
    }
  },
  
  "evidence": {
    "supporting_executions": [
      {
        "execution_id": "exec_20260301_142301_task42",
        "relevance": "策略在此任务上失败，失败原因是组件 C 和 F 之间的共享状态导致逐一测试无法复现 bug",
        "outcome": "failure"
      },
      {
        "execution_id": "exec_20260315_091122_task57",
        "relevance": "同类任务，共享缓存导致组件测试结果不可复现",
        "outcome": "failure"
      }
    ],
    "total_supporting": 2,
    "total_contradicting": 0,
    "evidence_strength": "medium"
  },
  
  "distiller_reasoning": "两次独立的失败案例都涉及组件间的隐性耦合（共享状态），且知识库中当前只有'因素之间存在强耦合'这一笼统条件。新条件对耦合类型做了细分——即使组件接口看起来是独立的，共享状态也会导致控制变量法失效。",
  
  "auto_apply_eligible": false,
  "reason_not_auto": "evidence_strength < strong (仅 2 条支持证据)"
}
```

### 3.3 蒸馏流程

蒸馏器对每批筛选后的经验执行以下分析，全部通过 LLM prompting 实现：

#### Step 1：单条经验分析

对每条高信息量经验，提取其对知识库的潜在启示。

```python
SINGLE_EXPERIENCE_ANALYSIS_PROMPT = """
你是一个方法论知识库的维护专家。

## 当前知识库中策略 {strategy_name} 的信息
描述: {strategy_description}
有利条件: {favorable_conditions}
不利条件: {unfavorable_conditions}
已知失败模式: {failure_modes}

## 执行经验
任务描述: {task_description}
任务特征: {problem_features}
执行结果: {outcome}
失败原因（如有）: {failure_reason}
执行轨迹摘要: {trajectory_summary}

## 归因分析（来自阶段一）
匹配的有利条件: {matched_conditions}
触发的不利条件: {violated_conditions}
新发现的条件候选: {newly_discovered}

## 分析任务
请回答以下问题：

1. 这次执行的结果是否暴露了知识库中**未记录**的适用条件？
   - 如果是，请用一句话描述这个新条件
   - 这个条件应该放在 favorable 还是 unfavorable 中？
   - 你对这个判断的置信度是多少？(0.0-1.0)

2. 这次执行是否表明某个**已有条件**的描述需要修改？
   - 如果是，请指出哪个条件（condition_id），以及如何修改
   
3. 这次执行是否表明某个已有条件的**置信度**需要调整？
   - 如果是，请指出哪个条件，以及应该上调还是下调

4. 这次执行是否揭示了该策略与其他策略之间**新的关系**？
   - 例如：在此场景下，策略 X 失败后换成策略 Y 成功了

输出 JSON:
{
    "new_conditions": [
        {
            "condition_text": "...",
            "placement": "favorable | unfavorable",
            "confidence": 0.0-1.0,
            "reasoning": "..."
        }
    ],
    "modified_conditions": [
        {
            "condition_id": "...",
            "original_text": "...",
            "proposed_text": "...",
            "reasoning": "..."
        }
    ],
    "confidence_adjustments": [
        {
            "condition_id": "...",
            "direction": "increase | decrease",
            "magnitude": 0.0-0.2,
            "reasoning": "..."
        }
    ],
    "new_relationships": [
        {
            "related_strategy": "...",
            "relationship_type": "...",
            "description": "..."
        }
    ]
}
"""
```

#### Step 2：跨经验聚合

当多条独立的经验指向同一个结论时，证据强度显著增加。蒸馏器定期（每积累 50 条经验后）对所有经验的分析结果进行聚合。

```python
class ExperienceDistiller:
    
    def aggregate_insights(
        self,
        individual_analyses: List[SingleAnalysis],
        kb_snapshot: KBSnapshot
    ) -> List[UpdateCandidate]:
        """
        将多条独立经验的分析结果聚合为更新候选。
        
        聚合逻辑（规则函数，不需要 LLM）：
        1. 将所有 new_conditions 按语义相似度聚类
        2. 同一簇内的条件合并为一条，置信度取加权平均
        3. 被 ≥3 条独立经验支持的条件升级为 evidence_strength="strong"
        4. 对 confidence_adjustments 取平均方向和幅度
        """
        
        # 1. 聚类：用嵌入相似度将语义相近的新条件归为一组
        condition_clusters = self._cluster_conditions(
            [c for a in individual_analyses for c in a.new_conditions],
            similarity_threshold=0.8
        )
        
        candidates = []
        for cluster in condition_clusters:
            if len(cluster.members) >= 2:
                # 至少 2 条经验支持才生成候选
                candidate = self._merge_cluster_to_candidate(
                    cluster, kb_snapshot
                )
                candidates.append(candidate)
        
        # 2. 聚合置信度调整
        confidence_updates = self._aggregate_confidence_adjustments(
            individual_analyses
        )
        candidates.extend(confidence_updates)
        
        return candidates
    
    def _cluster_conditions(
        self,
        conditions: List[NewCondition],
        similarity_threshold: float
    ) -> List[ConditionCluster]:
        """
        用 embedding 相似度聚类语义相近的条件。
        不需要 fine-tune——直接用预训练的 sentence-transformer。
        """
        embeddings = self.encoder.encode(
            [c.condition_text for c in conditions]
        )
        # 层次聚类
        clusters = AgglomerativeClustering(
            distance_threshold=1 - similarity_threshold,
            metric="cosine",
            linkage="average"
        ).fit(embeddings)
        return self._group_by_labels(conditions, clusters.labels_)
```

#### Step 3：LLM 辅助条件合并

当一个簇内有多条语义相近但表述不同的条件时，用 LLM 将它们合并为一条精炼的表述。

```python
CONDITION_MERGE_PROMPT = """
以下是从多次独立执行经验中提取的、语义相近的适用条件描述：

{conditions_list}

请将它们合并为一条精炼、准确的条件描述。要求：
1. 保留所有条件中共同的核心含义
2. 去掉只在个别经验中出现的特殊细节
3. 表述要足够通用，能适用于不同领域
4. 长度不超过 50 字

输出：
{
    "merged_condition": "...",
    "covers_all_originals": true/false,
    "lost_nuances": ["如果有某条原始条件的含义在合并中丢失，列出来"]
}
"""
```

### 3.4 蒸馏质量控制

为了防止蒸馏器产生低质量的更新候选，设置以下质量门控：

**门控 1：最少证据数量**
- `evidence_strength = "weak"`：1 条经验支持
- `evidence_strength = "medium"`：2-4 条经验支持
- `evidence_strength = "strong"`：≥ 5 条经验支持
- 只有 `medium` 和 `strong` 的候选才进入知识整合器

**门控 2：交叉验证**
- 对每个更新候选，用 LLM 从知识库中检索可能冲突的已有条件
- 如果存在直接矛盾（如新条件说"X 有利"但已有条件说"X 不利"），标记为冲突，不自动应用

**门控 3：可泛化性检查**
- 如果支持一个新条件的所有经验都来自同一个领域，标记为"领域特异性"
- 领域特异性的条件需要额外标注 `domain_specific: true`，在其他领域不自动匹配

---

## 4. 知识整合器

### 4.1 设计理念

知识整合器是知识库的"看门人"。它接收蒸馏器的更新候选，决定是否应用到知识库中。

**核心原则：宁可遗漏，不可污染。** 一条错误的规则进入知识库的代价远高于一条正确的规则被延迟纳入。因此整合器的默认倾向是保守的——不确定的更新一律进入人工审核队列。

### 4.2 自动应用条件

一个更新候选可以被自动应用（不需要人工审核）当且仅当满足以下全部条件：

```python
def can_auto_apply(candidate: UpdateCandidate, kb: KBSnapshot) -> bool:
    """判断更新候选是否可以被自动应用"""
    
    # 条件 1：证据强度 ≥ strong
    if candidate.evidence.evidence_strength != "strong":
        return False
    
    # 条件 2：没有冲突的已有条件
    if has_conflicting_conditions(candidate, kb):
        return False
    
    # 条件 3：不修改 locked 条件
    if candidate.update_type in ("condition_modified", "confidence_adjusted"):
        target_cond = get_condition(kb, candidate.target_condition_id)
        if target_cond and target_cond.get("locked", False):
            return False
    
    # 条件 4：置信度调整幅度不超过 0.15
    if candidate.update_type == "confidence_adjusted":
        if abs(candidate.proposed_change["magnitude"]) > 0.15:
            return False
    
    # 条件 5：新条件的置信度不超过 0.85（自动添加的条件不应有过高初始置信度）
    if candidate.update_type == "condition_added":
        if candidate.proposed_change["new_condition"]["confidence"] > 0.85:
            return False
    
    return True
```

### 4.3 更新应用流程

```python
class KnowledgeIntegrator:
    
    def process_candidate(
        self,
        candidate: UpdateCandidate,
        kb: KnowledgeBase
    ) -> IntegrationResult:
        
        # 1. 冲突检测
        conflicts = self._detect_conflicts(candidate, kb)
        if conflicts:
            return IntegrationResult(
                action="human_review",
                reason=f"与已有条件冲突: {conflicts}",
                destination="pending_human/"
            )
        
        # 2. 自动应用检查
        if can_auto_apply(candidate, kb):
            # 应用更新
            self._apply_update(candidate, kb)
            # 写入变更历史
            self._write_change_history(candidate, kb)
            return IntegrationResult(
                action="applied",
                destination="approved/"
            )
        
        # 3. 不满足自动应用条件
        return IntegrationResult(
            action="human_review",
            reason=self._explain_why_not_auto(candidate),
            destination="pending_human/"
        )
    
    def _apply_update(
        self,
        candidate: UpdateCandidate,
        kb: KnowledgeBase
    ):
        """
        将更新应用到知识库。
        直接修改 kb/strategies/S*.json 文件。
        """
        strategy = kb.load_strategy(candidate.target_strategy)
        
        if candidate.update_type == "condition_added":
            placement = candidate.proposed_change["placement"]
            new_cond = candidate.proposed_change["new_condition"]
            strategy["applicability_conditions"][placement].append(new_cond)
            
        elif candidate.update_type == "confidence_adjusted":
            cond_id = candidate.proposed_change["condition_id"]
            direction = candidate.proposed_change["direction"]
            magnitude = candidate.proposed_change["magnitude"]
            cond = find_condition(strategy, cond_id)
            if direction == "increase":
                cond["confidence"] = min(1.0, cond["confidence"] + magnitude)
            else:
                cond["confidence"] = max(0.0, cond["confidence"] - magnitude)
            cond["version"] += 1
            cond["last_updated"] = datetime.utcnow().isoformat()
            
        elif candidate.update_type == "condition_modified":
            cond_id = candidate.proposed_change["condition_id"]
            cond = find_condition(strategy, cond_id)
            cond["condition"] = candidate.proposed_change["proposed_text"]
            cond["version"] += 1
            cond["last_updated"] = datetime.utcnow().isoformat()
        
        elif candidate.update_type == "condition_moved":
            # 条件从 favorable 移到 unfavorable 或反之
            cond_id = candidate.proposed_change["condition_id"]
            from_list = candidate.proposed_change["from"]
            to_list = candidate.proposed_change["to"]
            cond = find_and_remove(strategy, from_list, cond_id)
            strategy["applicability_conditions"][to_list].append(cond)
            cond["version"] += 1
            cond["last_updated"] = datetime.utcnow().isoformat()
            
        elif candidate.update_type == "relationship_added":
            strategy["relationships_to_other_strategies"].append(
                candidate.proposed_change["new_relationship"]
            )
        
        # 更新元数据
        strategy["metadata"]["version"] = str(
            float(strategy["metadata"]["version"]) + 0.1
        )
        strategy["metadata"]["last_updated"] = datetime.utcnow().isoformat()
        
        # 更新经验计数
        for exec_ref in candidate.evidence["supporting_executions"]:
            if exec_ref["outcome"] == "success":
                strategy["metadata"]["successful_applications"] += 1
            else:
                strategy["metadata"]["failed_applications"] += 1
        strategy["metadata"]["total_experience_records"] += len(
            candidate.evidence["supporting_executions"]
        )
        
        # 保存
        kb.save_strategy(strategy)
    
    def _write_change_history(
        self,
        candidate: UpdateCandidate,
        kb: KnowledgeBase
    ):
        """
        写入变更历史（阶段零 2.5 节的 JSONL 格式）。
        """
        history_entry = {
            "change_id": generate_change_id(),
            "timestamp": datetime.utcnow().isoformat(),
            "type": candidate.update_type,
            "author": "phase2_auto",
            "changes": candidate.proposed_change,
            "evidence_refs": [
                e["execution_id"]
                for e in candidate.evidence["supporting_executions"]
            ],
            "previous_version": get_current_version(kb, candidate.target_strategy),
            "new_version": get_current_version(kb, candidate.target_strategy) + 1
        }
        append_to_jsonl(
            f"change_history/{candidate.target_strategy}.jsonl",
            history_entry
        )
```

### 4.4 冲突检测

冲突检测分两层：规则层（快速）和语义层（精确）。

**规则层冲突检测（确定性）：**

```python
def detect_rule_conflicts(
    candidate: UpdateCandidate,
    kb: KBSnapshot
) -> List[Conflict]:
    conflicts = []
    
    if candidate.update_type == "condition_added":
        new_text = candidate.proposed_change["new_condition"]["condition"]
        placement = candidate.proposed_change["placement"]
        opposite = "unfavorable" if placement == "favorable" else "favorable"
        
        # 检查是否和对立列表中的条件语义相近
        for existing in kb.get_conditions(candidate.target_strategy, opposite):
            sim = compute_embedding_similarity(new_text, existing["condition"])
            if sim > 0.85:
                conflicts.append(Conflict(
                    type="contradictory",
                    existing_condition=existing["condition_id"],
                    similarity=sim,
                    description=f"新条件放在 {placement}，但与 {opposite} "
                                f"中的 {existing['condition_id']} 语义高度相似"
                ))
    
    return conflicts
```

**语义层冲突检测（LLM prompting）：**

当规则层检测到可能的冲突（相似度在 0.6-0.85 之间）时，调用 LLM 进行精确判断：

```python
CONFLICT_DETECTION_PROMPT = """
以下是知识库中策略 {strategy_name} 的两个适用条件：

已有条件（{existing_placement}）: {existing_condition}
新增候选条件（{new_placement}）: {new_condition}

请判断这两个条件之间的关系：
1. "contradictory": 两者直接矛盾（不能同时为真）
2. "overlapping": 两者部分重叠但不矛盾（新条件是已有条件的细化）
3. "independent": 两者描述不同维度，互不影响

如果是 overlapping，建议是否应该将已有条件拆分或细化。

输出 JSON:
{
    "relationship": "contradictory | overlapping | independent",
    "reasoning": "...",
    "suggested_resolution": "..."
}
"""
```

### 4.5 健康监控与自动回滚

知识整合器持续监控更新后策略的表现，以检测有害更新。

```python
class KBHealthMonitor:
    """
    监控知识库更新后策略的成功率变化。
    如果更新后某策略的成功率显著下降，触发自动回滚。
    """
    
    def check_strategy_health(
        self,
        strategy_id: str,
        window_size: int = 50
    ) -> HealthStatus:
        
        recent_records = get_recent_executions(
            strategy_id, limit=window_size
        )
        
        if len(recent_records) < window_size:
            return HealthStatus(status="insufficient_data")
        
        # 找到最近一次知识库更新的时间点
        last_update = get_last_update_timestamp(strategy_id)
        
        # 分为更新前和更新后两组
        before = [r for r in recent_records if r["timestamp"] < last_update]
        after = [r for r in recent_records if r["timestamp"] >= last_update]
        
        if len(before) < 10 or len(after) < 10:
            return HealthStatus(status="insufficient_data")
        
        success_rate_before = sum(
            1 for r in before if r["outcome"]["success"]
        ) / len(before)
        success_rate_after = sum(
            1 for r in after if r["outcome"]["success"]
        ) / len(after)
        
        drop = success_rate_before - success_rate_after
        
        if drop > 0.20:
            # 成功率下降超过 20%，触发回滚警告
            return HealthStatus(
                status="degraded",
                drop=drop,
                recommendation="rollback",
                message=f"策略 {strategy_id} 成功率从 "
                        f"{success_rate_before:.0%} 降至 "
                        f"{success_rate_after:.0%}，建议回滚"
            )
        elif drop > 0.10:
            return HealthStatus(
                status="warning",
                drop=drop,
                recommendation="monitor"
            )
        else:
            return HealthStatus(status="healthy")
```

回滚操作调用阶段零的 `kb_rollback.py`（阶段零 2.6 节），不需要重新实现。

---

## 5. "睡眠整合"机制

### 5.1 设计理念

Claude.md 讨论中引用了一个关键概念：

> 交替进行收集（清醒）和压缩/抽象（睡眠）阶段，使记忆增强型智能体能够层级式地增长技能库，同时限制上下文过拟合和灾难性遗忘。

本系统的"清醒"阶段是日常的经验收集和即时更新。"睡眠"阶段是定期的全局整合——不是处理单条经验，而是审视知识库的整体状态。

### 5.2 整合触发条件

睡眠整合在以下任一条件满足时触发：

- 自上次整合以来，知识库被更新超过 20 次
- 自上次整合以来，经过了 30 天
- 任一策略的成功率出现 > 10% 的变化
- 人工触发

### 5.3 整合任务

每次睡眠整合执行以下操作（全部通过 LLM prompting + 规则函数实现）：

**任务 1：条件去重与合并**

```python
def consolidate_conditions(strategy: Dict) -> List[MergeProposal]:
    """
    检查一条策略的所有适用条件，合并语义重复的条件。
    """
    all_conditions = (
        strategy["applicability_conditions"]["favorable"] +
        strategy["applicability_conditions"]["unfavorable"]
    )
    
    # 用嵌入相似度找到高度相似的条件对
    embeddings = encoder.encode([c["condition"] for c in all_conditions])
    merge_proposals = []
    
    for i in range(len(all_conditions)):
        for j in range(i + 1, len(all_conditions)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > 0.85:
                # 用 LLM 确认是否应该合并
                should_merge = llm_confirm_merge(
                    all_conditions[i], all_conditions[j]
                )
                if should_merge:
                    merge_proposals.append(MergeProposal(
                        condition_a=all_conditions[i]["condition_id"],
                        condition_b=all_conditions[j]["condition_id"],
                        similarity=sim
                    ))
    
    return merge_proposals
```

**任务 2：低价值条件清理**

```python
def identify_low_value_conditions(strategy: Dict) -> List[str]:
    """
    识别长期无证据支持的条件。
    """
    low_value = []
    for placement in ["favorable", "unfavorable"]:
        for cond in strategy["applicability_conditions"][placement]:
            if (cond["source"] == "experience" and
                len(cond["supporting_cases"]) == 0 and
                cond["version"] == 1 and
                days_since(cond["last_updated"]) > 90):
                low_value.append(cond["condition_id"])
    return low_value
```

**任务 3：跨策略模式识别**

检查是否有多个策略共享相似的条件模式，可能暗示策略之间存在未记录的关系。

```python
CROSS_STRATEGY_PATTERN_PROMPT = """
以下是知识库中所有策略的适用条件摘要：

{all_strategies_conditions}

请识别以下模式：
1. 哪些策略共享非常相似的 favorable 条件？（可能暗示 alternative 关系）
2. 哪些策略的 favorable 条件是另一个策略的 unfavorable 条件？（可能暗示 complementary 关系）
3. 是否有条件出现在多个策略中但表述不一致？（需要统一）

输出 JSON:
{
    "potential_alternatives": [...],
    "potential_complements": [...],
    "inconsistent_conditions": [...]
}
"""
```

**任务 4：生成整合报告**

每次睡眠整合后生成一份人类可读的报告，记录所做的变更和原因。报告写入 `docs/consolidation_reports/`。

---

## 6. 技术实现

### 6.1 项目文件结构

```
assumption_agent/                    # 在阶段一的项目基础上扩展
├── ...                              # 阶段一的所有目录保持不变
├── feedback/                        # 阶段二新增目录
│   ├── experience_evaluator/
│   │   ├── evaluator.py             # 经验评估器
│   │   ├── novelty_scorer.py        # 信息量评分
│   │   └── prompts.py               # LLM 评估 prompts
│   ├── experience_distiller/
│   │   ├── distiller.py             # 经验蒸馏器
│   │   ├── single_analysis.py       # 单条经验分析
│   │   ├── aggregator.py            # 跨经验聚合
│   │   ├── condition_clusterer.py   # 条件聚类
│   │   └── prompts.py               # LLM 蒸馏 prompts
│   ├── knowledge_integrator/
│   │   ├── integrator.py            # 知识整合器
│   │   ├── conflict_detector.py     # 冲突检测
│   │   ├── auto_apply.py            # 自动应用逻辑
│   │   ├── health_monitor.py        # 健康监控
│   │   └── prompts.py               # LLM 冲突检测 prompts
│   ├── consolidation/
│   │   ├── sleep_integrator.py      # 睡眠整合
│   │   ├── condition_dedup.py       # 条件去重
│   │   ├── cross_strategy.py        # 跨策略模式识别
│   │   └── report_generator.py      # 整合报告生成
│   ├── pipeline.py                  # 完整反馈管线（评估→蒸馏→整合）
│   └── config.py                    # 阈值和配置参数
├── scripts/
│   ├── ...                          # 阶段一的脚本保持不变
│   ├── run_feedback_pipeline.py     # 运行反馈管线
│   ├── run_consolidation.py         # 运行睡眠整合
│   ├── review_pending.py            # 人工审核待处理更新
│   └── generate_evolution_report.py # 生成知识库演化报告
└── tests/
    ├── ...                          # 阶段一的测试保持不变
    ├── test_evaluator.py
    ├── test_distiller.py
    ├── test_integrator.py
    ├── test_consolidation.py
    └── test_feedback_pipeline.py
```

### 6.2 配置参数

```python
# feedback/config.py

FEEDBACK_CONFIG = {
    # === 经验评估器 ===
    "evaluator": {
        "information_threshold": 0.3,       # 低于此分数的经验被丢弃
        "llm_assessment_range": (0.3, 0.7), # 在此范围内调用 LLM 辅助
        "llm_weight": 0.4,                  # LLM 评分的权重
    },
    
    # === 经验蒸馏器 ===
    "distiller": {
        "min_evidence_for_candidate": 2,    # 最少支持经验数
        "clustering_similarity": 0.8,        # 条件聚类相似度阈值
        "batch_size": 50,                    # 每批次处理的经验数
    },
    
    # === 知识整合器 ===
    "integrator": {
        "auto_apply_min_evidence": 5,        # 自动应用的最少证据数
        "auto_apply_max_confidence_delta": 0.15,  # 自动置信度调整上限
        "auto_apply_max_initial_confidence": 0.85, # 新条件的最大初始置信度
        "conflict_similarity_threshold": 0.85,     # 冲突检测相似度阈值
        "conflict_llm_range": (0.6, 0.85),         # 调用 LLM 的相似度范围
    },
    
    # === 健康监控 ===
    "health_monitor": {
        "window_size": 50,                   # 健康检查的经验窗口大小
        "degradation_threshold": 0.20,       # 触发回滚的成功率下降幅度
        "warning_threshold": 0.10,           # 触发警告的成功率下降幅度
        "min_samples_for_check": 10,         # 更新前后各需要的最少样本数
    },
    
    # === 睡眠整合 ===
    "consolidation": {
        "trigger_update_count": 20,          # 更新次数触发阈值
        "trigger_days": 30,                  # 天数触发阈值
        "dedup_similarity": 0.85,            # 条件去重相似度阈值
        "low_value_days": 90,                # 低价值条件的天数阈值
    },
    
    # === LLM 配置 ===
    "llm": {
        "model": "gpt-4o-mini",              # 默认用便宜的模型
        "temperature": 0.1,                  # 低温度确保分析的确定性
        "max_retries": 3,
        "fallback_model": "gpt-4o",          # 当便宜模型输出质量不足时
    },
}
```

### 6.3 LLM 调用成本估算

本阶段所有 LLM 使用都是 prompting，没有 fine-tuning。以下是成本估算：

| 操作 | 频率 | 每次 token 消耗 | 模型 | 月成本估算 |
|------|------|----------------|------|-----------|
| 经验评估（LLM辅助） | ~30% 的经验触发 | ~1K input + 200 output | gpt-4o-mini | ~$5 |
| 单条经验分析 | 每条保留的经验 | ~2K input + 500 output | gpt-4o-mini | ~$15 |
| 条件合并 | 每次聚合 | ~1K input + 200 output | gpt-4o-mini | ~$2 |
| 冲突检测（LLM层） | 每个候选 | ~1K input + 300 output | gpt-4o | ~$5 |
| 睡眠整合 | 每月 1-2 次 | ~5K input + 1K output | gpt-4o | ~$3 |
| **总计** | | | | **~$30/月** |

**对比：** 如果用 fine-tuning 方案，单次 LoRA 训练约 $50-200，且每次知识库大幅更新后可能需要重新训练。Prompting 方案在成本上有显著优势。

### 6.4 与阶段零的接口

本阶段直接操作阶段零的以下文件和目录：

| 操作 | 目标 | 模式 |
|------|------|------|
| 读取 | `experience_log/executions/*.json` | 只读 |
| 写入 | `experience_log/distilled/pending_review/` | 写入更新候选 |
| 移动 | `pending_review/ → approved/ 或 rejected/` | 审核完成后移动 |
| 修改 | `kb/strategies/S*.json` | 应用更新时修改 |
| 追加 | `change_history/S*.jsonl` | 记录变更历史 |
| 调用 | `scripts/kb_rollback.py` | 回滚时调用 |
| 调用 | `scripts/validate_kb.py` | 每次更新后验证 schema |

**每次更新后必须验证：** 调用阶段零的 `validate_kb.py`，确保更新后的 JSON 仍然符合 schema。如果验证失败，自动回滚本次更新。

### 6.5 与阶段一的接口

| 方向 | 数据 | 说明 |
|------|------|------|
| 阶段一 → 阶段二 | 经验日志 | 阶段一写入，阶段二读取并分析 |
| 阶段一 → 阶段二 | 调度器偏好数据 | 阶段一导出，阶段二用于比对分析 |
| 阶段二 → 阶段一 | 更新后的知识库 | 阶段二修改知识库，阶段一下次加载时自动获取新版本 |

**无需修改阶段一的任何代码。** 阶段一的调度器在每次启动时调用 `export_for_agent.py` 加载知识库快照。知识库被阶段二更新后，阶段一下次启动时自然读到新版本。

### 6.6 与阶段零点五（世界模型）的接口——双来源反馈

阶段二的经验反馈有**两个来源**，来自阶段零点五的世界模型集成：

**来源 1：真实执行反馈（高质量，昂贵）**
- 阶段一调度器在真实环境中执行任务产生的经验记录
- 特征：结果可信，轨迹真实，但每条记录的获取成本高（LLM API 调用 + 执行时间）
- 占比：约 10%（阶段一 model-based RL 训练中的真实执行部分）

**来源 2：世界模型模拟反馈（低质量，廉价）**
- 阶段一调度器在世界模型中模拟执行产生的预测记录
- 特征：结果是预测值而非真实值，无真实执行轨迹，但获取成本极低（毫秒级）
- 占比：约 90%

**双来源的差异化处理：**

```python
class DualSourceEvaluator(ExperienceEvaluator):
    """
    扩展经验评估器，区分真实和模拟经验的处理方式。
    """
    
    def evaluate(self, record, kb_snapshot) -> EvaluationResult:
        is_simulated = record.get("is_simulated", False)
        
        if not is_simulated:
            # 真实经验：走标准评估流程（第 2 节）
            return super().evaluate(record, kb_snapshot)
        
        else:
            # 模拟经验：降低权重，提高筛选门槛
            base_result = super().evaluate(record, kb_snapshot)
            
            # 模拟经验的信息量评分打折
            base_result.information_score *= 0.5
            
            # 模拟经验不能单独触发知识库更新——
            # 必须有 ≥1 条真实经验佐证同一结论
            base_result.requires_real_corroboration = True
            
            # 模拟经验的淘汰阈值更高
            base_result.keep = base_result.information_score >= 0.5
            
            return base_result
```

**蒸馏器中的双来源聚合规则：**

- 一个更新候选的 `evidence_strength` 计算中，1 条真实经验 = 3 条模拟经验
- `evidence_strength = "strong"` 的条件变为：≥ 5 条真实经验，或 ≥ 2 条真实 + ≥ 9 条模拟
- 纯模拟经验（0 条真实佐证）永远不能达到 `"strong"`，最高只能 `"medium"`
- 自动应用条件（第 4.2 节）额外要求：至少 2 条真实经验支持

**世界模型预测偏差的利用：**

当世界模型的预测与真实执行结果不一致时，这本身是一条高价值信息：

```python
def detect_model_reality_gap(
    simulated_record: ExecutionRecord,
    real_record: ExecutionRecord
) -> Optional[ModelRealityGap]:
    """
    检测同一 (问题, 策略) 对上模拟与真实结果的偏差。
    偏差暗示系统对这类任务的"世界理解"有缺陷。
    """
    sim_success = simulated_record["outcome"]["success"]
    real_success = real_record["outcome"]["success"]
    
    if sim_success != real_success:
        return ModelRealityGap(
            task_features=real_record["task"]["complexity_features"],
            strategy=real_record["strategy_selection"]["selected_strategy"],
            model_predicted=sim_success,
            reality=real_success,
            gap_type=(
                "false_positive" if sim_success and not real_success
                else "false_negative"
            ),
            # false_positive 更危险：世界模型说能成功但实际失败
            # 可能暗示一个知识库中未记录的 unfavorable 条件
            severity="high" if sim_success and not real_success else "medium"
        )
    return None
```

`false_positive` 类型的偏差（模型乐观但现实失败）直接作为高信息量经验进入蒸馏器——它可能揭示了一个知识库中未记录的 unfavorable 条件，且这个条件恰好是世界模型也没有学到的。

### 6.7 与阶段三的接口

本阶段为阶段三（形式化层）预留以下数据：

- 知识库的完整变更历史（`change_history/*.jsonl`）：阶段三可以分析哪些条件经常被修改，作为形式化优先级的依据
- 经验蒸馏器发现的跨策略模式：阶段三可以用范畴论验证这些模式是否对应真正的结构同构
- 策略的成功率统计：阶段三可以将信息几何距离与实际性能差异做相关性分析

---

## 7. 实验设计

### 7.1 核心实验：知识库演化的有效性

**假设 H1：** 经过经验反馈演化后的知识库，在调度器的任务完成率上显著优于演化前的知识库。

**实验设计：**
- 用阶段一训练好的调度器作为经验来源，收集 N 条执行经验
- 运行完整的反馈管线，更新知识库
- 在相同的测试集上比较更新前后的调度器性能
- 注意：调度器的模型权重不变，变化的只是知识库

**对比条件：**
- KB v1.0（阶段零人类标注的原始知识库）
- KB v1.x（经过 N 条经验更新后的知识库）
- KB v1.x-unfiltered（不经过评估器筛选，所有经验直接蒸馏更新的知识库——验证质量门控的必要性）

### 7.2 核心实验：质量门控的必要性

**假设 H2：** 有质量门控的知识库演化优于无门控的暴力更新。

**实验设计：**
- 方法 A：完整管线（评估器 + 蒸馏器 + 整合器的全部质量门控）
- 方法 B：去掉评估器，所有经验直接进入蒸馏
- 方法 C：去掉整合器的自动应用条件，所有候选直接应用
- 方法 D：去掉所有门控，原始经验直接更新知识库

**预期结果：** 方法 A > B > C > D。特别是方法 D 应该表现出"平坦或下降的成功率"（与文献中的实证结果一致）。

### 7.3 关键实验：发现人类未记录的边界条件

**假设 H3：** 系统能自动发现人类哲学文献中未曾显式描述的策略适用边界条件。

**实验设计：**
- 在大量任务执行后，收集所有被系统发现的新条件
- 邀请领域专家（来自阶段零的标注者或新招募的专家）评审这些条件
- 评审标准：
  - 合理性：该条件在逻辑上是否成立？
  - 新颖性：该条件是否在常见方法论教科书中未被提及？
  - 实用性：如果将该条件告知人类问题解决者，是否有助于他们做出更好的策略选择？

**这是论文的核心贡献实验。** 如果系统能发现至少 3 条被专家确认为"合理且新颖"的边界条件，就证明了系统具有超越人类先验的知识发现能力。

### 7.4 分析实验：知识库演化轨迹

**目标：** 可视化知识库随经验积累的演化过程。

**分析内容：**
- 置信度随时间的变化曲线（每条策略的每个条件）
- 新增条件的时间分布（知识库在什么阶段增长最快）
- 策略关系图的演化（是否发现了新的策略间关系）
- 回滚事件的分析（有害更新的特征是什么）

---

## 8. 风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| LLM prompting 的蒸馏质量不足 | 中 | 高 | 对蒸馏器的输出做人工抽样审核（每 50 个候选审核 10 个）；如果质量 < 70%，考虑换更强的 LLM 或加入少样本示例 |
| 经验数据不足以触发有意义的更新 | 中 | 中 | 确保阶段一的调度器在足够多的任务上运行；如果经验不足，可以用阶段一的 B 类任务集做额外的经验收集 |
| 自动更新引入有害规则 | 高 | 高 | 严格的自动应用条件（5 条证据 + 无冲突 + 不改 locked）；健康监控 + 自动回滚；重要条件标记为 locked |
| 条件爆炸（每条策略积累过多条件） | 中 | 中 | 睡眠整合的去重机制；对每条策略设置条件数量上限（如 favorable 最多 10 条，unfavorable 最多 10 条） |
| 蒸馏器的 LLM 产生幻觉条件 | 高 | 中 | 最少 2 条独立经验支持才生成候选；交叉验证（不同 LLM 调用是否给出一致结论） |
| 嵌入相似度不准导致聚类错误 | 中 | 中 | 使用经过验证的 sentence-transformer 模型；对关键操作（冲突检测）加入 LLM 语义确认层 |
| 知识库更新后 schema 验证失败 | 低 | 高 | 每次更新后立即调用 validate_kb.py；失败则自动回滚 |
| prompting 方案在某些分析任务上完全失败 | 低 | 高 | 预留 LoRA 微调作为后备方案，但仅在 prompting 经过充分调试后仍不达标时启用 |

---

## 9. 增量开发计划

### Step 1：Mock 管线验证（第 1-2 周）

**目标：** 用阶段零可更新性测试中的模拟数据，端到端跑通反馈管线。

- 复用阶段零 4.3 节的 10-20 条模拟经验记录
- 实现经验评估器（规则部分，暂不接 LLM）
- 实现最简版蒸馏器（直接从 attribution 字段提取候选，不调用 LLM）
- 实现知识整合器（自动应用 + 变更历史写入）
- **验证：** 阶段零的 6 种更新场景全部按预期运行

### Step 2：接入 LLM prompting（第 3-4 周）

**目标：** 为三个模块接入 LLM 分析能力。

- 实现所有 LLM prompt 模板
- 实现 LLM 辅助评估、单条经验分析、冲突检测
- 用 10 条真实经验（来自阶段一的试运行）测试 LLM 输出质量
- **验证：** LLM 输出的 JSON 格式正确率 ≥ 95%；人工审核分析质量 ≥ 70%

### Step 3：跨经验聚合（第 5-6 周）

**目标：** 实现条件聚类和跨经验聚合。

- 实现 embedding 聚类
- 实现 LLM 辅助条件合并
- 实现更新候选的完整生成流程
- **验证：** 给定 50 条模拟经验，聚合后生成 5-10 个更新候选，人工审核合理率 ≥ 60%

### Step 4：大规模经验收集与更新（第 7-12 周）

**目标：** 在真实规模上运行反馈管线。

- 用阶段一的调度器在 B 类任务集上运行，收集 500+ 条经验
- 运行完整管线，更新知识库
- 运行健康监控
- **验证：** 至少 3 条更新被自动应用且未触发回滚

### Step 5：睡眠整合（第 13-14 周）

**目标：** 实现并运行睡眠整合机制。

- 实现条件去重、低价值清理、跨策略模式识别
- 在 Step 4 的更新结果上运行一次完整整合
- 生成整合报告
- **验证：** 整合后知识库的 schema 验证通过；条件数量没有无限增长

### Step 6：评估与论文（第 15-18 周）

**目标：** 运行所有实验，写论文。

- 运行 H1-H3 实验
- 知识库演化轨迹可视化
- 人类专家评审系统发现的新条件
- 论文写作

---

## 10. 完成标准（Definition of Done）

阶段二在以下所有条件同时满足时视为完成：

1. 完整的反馈管线（评估→蒸馏→整合）端到端可运行
2. 经过经验反馈更新后的知识库在调度器的任务完成率上优于原始知识库至少 5 个百分点
3. 有质量门控的更新优于无门控的暴力更新（H2 实验验证）
4. 系统自动发现了至少 3 条被人类专家确认为"合理且新颖"的边界条件
5. 所有知识库更新都有完整的变更历史记录，可追溯到具体的经验证据
6. 回滚机制经过验证：人工注入一条有害更新后，健康监控能在 50 条经验内检测到并建议回滚
7. 睡眠整合机制已运行至少一次，生成了可读的整合报告
8. 所有知识库更新后 schema 验证通过
9. LLM prompting 的总月成本不超过 $50（验证不 fine-tune 方案的经济可行性）
10. 完成一篇论文初稿，核心贡献为"系统通过经验发现人类未记录的策略边界条件"

---

## 附录 A：Fine-Tune 替代方案

默认方案（正文所述）完全基于 prompting + 规则函数，不涉及任何模型参数更新。但在三个具体场景下，fine-tuning 可以带来额外收益。以下方案作为**可选升级路径**，仅在 prompting 方案经过充分验证后仍不达标时启用。

### A.1 场景一：蒸馏器专用模型（最可能需要）

**触发条件：** 经验蒸馏器的 prompting 输出质量在人工审核中持续低于 60%（即超过 40% 的更新候选被专家判定为不合理），且更换更强的 LLM（如 gpt-4o）和优化 prompt 后仍无改善。

**问题本质：** 经验蒸馏是本阶段最难的分析任务——它要求 LLM 同时理解执行轨迹的语义、知识库的现有条件结构、以及两者之间的因果关系。通用 LLM 可能在这种高度结构化的多步推理上存在系统性偏差。

**Fine-tune 方案：**

用 prompting 管线积累的人工审核数据来训练一个专用蒸馏模型。

```
训练数据构造:
- 输入: (执行记录, 知识库当前状态)
- 输出: 人工审核通过的 UpdateCandidate
- 来源: prompting 管线运行一段时间后，pending_human/ 和 approved/ 中积累的数据
- 所需数据量: 200-500 条审核过的 (输入, 输出) 对
```

```python
# 蒸馏器 fine-tune 配置
DISTILLER_FINETUNE_CONFIG = {
    "base_model": "Qwen2.5-7B-Instruct",    # 小型开源模型
    "method": "LoRA",
    "lora_rank": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "batch_size": 4,
    "max_seq_length": 4096,
    
    # 训练数据
    "min_training_samples": 200,              # 低于此数量不启动 fine-tune
    "validation_split": 0.15,
    "quality_filter": "only_approved",        # 只用人工审核通过的数据
}
```

**与 prompting 方案的关系：** 不是替换而是补充。Fine-tune 后的专用模型替换蒸馏器中的 LLM 调用，但整个管线（评估器→蒸馏器→整合器）的结构不变，质量门控不变，变更历史不变。

**成本对比：**

| 维度 | Prompting 方案 | Fine-tune 方案 |
|------|---------------|---------------|
| 一次性成本 | $0 | ~$50-100（LoRA 训练，4-8 GPU-hours） |
| 月运行成本 | ~$15（蒸馏部分） | ~$3（本地推理，无 API 调用） |
| 数据要求 | 无 | 200+ 条审核过的训练数据 |
| 迭代成本 | 改 prompt 即时生效 | 每次重训需要 1-2 小时 |
| 5 个月后累计成本 | ~$75 | ~$65-115 |
| 质量天花板 | 受限于通用 LLM 的零样本能力 | 可通过更多数据持续提升 |

**结论：** 如果系统计划长期运行（> 6 个月），fine-tune 方案在总成本上开始持平甚至更低，且质量天花板更高。但前 2-3 个月不应启用——需要先用 prompting 方案积累足够的高质量训练数据。

### A.2 场景二：调度器的协同更新（MEL 风格）

**触发条件：** 知识库经过大量更新后，阶段一的调度器（模型权重固定）无法充分利用新增的条件——表现为新增条件的匹配率很低，调度器仍然依赖旧的特征模式做决策。

**问题本质：** 默认方案假设调度器能通过阶段零的导出接口自动感知知识库变化。但如果调度器是方案 A（轻量级网络），它的输入特征中"知识库适用条件匹配分数"是基于嵌入相似度计算的，新增的条件需要调度器重新学习如何利用这些匹配信号。

**Fine-tune 方案：**

参考 MEL（Meta-Experience Learning）的思路，将知识库更新产生的"元经验"内化到调度器的参数中。

```python
# 调度器协同更新配置
DISPATCHER_COUPDATE_CONFIG = {
    "trigger": "kb_version_delta >= 5",       # 知识库版本差 ≥ 5 时触发
    "method": "supervised_finetune",           # 不是 RL，是监督微调
    
    # 训练数据: 知识库更新后在验证集上重新标注的 (问题, 最优策略) 对
    # 用更新后的知识库的规则匹配结果作为伪标签
    "pseudo_label_source": "rule_matching_with_updated_kb",
    "human_review_ratio": 0.1,                 # 10% 的伪标签做人工审核
    
    # 微调配置
    "finetune_epochs": 2,                      # 轻量微调，避免过拟合
    "finetune_lr": 5e-5,                       # 较小学习率
    "keep_old_data_ratio": 0.5,                # 50% 旧数据混合防止灾难性遗忘
}
```

**执行流程：**

1. 知识库版本从 v1.0 演化到 v1.5（经过 5 次以上更新）
2. 用更新后的知识库在验证集上重新运行规则匹配，生成新的伪标签
3. 将伪标签与旧的训练数据混合，对调度器做 2 轮监督微调
4. 在测试集上验证微调后的调度器是否优于未微调的版本
5. 如果优于，则采用微调后的版本；否则保持原版

**与默认方案的关系：** 默认方案中，调度器权重在阶段二期间完全冻结。本方案允许在知识库发生重大演化后，用轻量级监督微调使调度器适应新的知识库状态。这不是 RL 重训（那样成本太高），而是用规则匹配的结果做伪标签的监督微调。

**关键约束：** 调度器微调后必须在全部测试集上回归测试，确保在已有任务上不出现性能退化（灾难性遗忘）。如果任何领域的性能下降超过 5%，回滚微调。

### A.3 场景三：经验评估器的奖励模型化

**触发条件：** 经验评估器的规则函数 + LLM 辅助评分在边界情况下（信息量评分在 0.3-0.5 的灰色地带）表现不稳定，导致大量低价值经验进入蒸馏流程或大量高价值经验被误丢弃。

**问题本质：** 判断一条经验是否"有信息量"本质上是一个偏好判断——类似于 RLHF 中的奖励建模。随着系统运行，人工审核者会积累大量"这条经验应该保留/丢弃"的偏好数据，可以训练一个专用的评估模型。

**Fine-tune 方案：**

```python
# 经验评估器奖励模型配置
EVALUATOR_RM_CONFIG = {
    "base_model": "all-MiniLM-L6-v2",        # 轻量级 encoder
    "architecture": "encoder + linear head",   # 不需要生成能力
    "output": "scalar (0.0-1.0)",
    
    # 训练数据: 人工审核过的 (经验记录, 保留/丢弃) 对
    "min_training_samples": 300,
    "label_source": "human_review",
    
    # 用对比学习增强
    "contrastive_pairs": True,                 # (高信息量经验, 低信息量经验) 对
    
    # 训练配置
    "learning_rate": 1e-4,
    "num_epochs": 5,
    "batch_size": 16,
}
```

**好处：** 推理速度极快（无需 LLM API 调用），可在毫秒级完成评估；且随着审核数据增加，精度持续提升。

**与默认方案的关系：** 替换评估器中的 LLM 辅助评估调用。规则评分部分保持不变（它处理的是确定性的高信息量/低信息量情况），fine-tune 模型只处理灰色地带（规则评分在 0.3-0.7 之间的经验）。

### A.4 Fine-Tune 方案的启用决策树

```
                   prompting 方案运行 2+ 个月
                            │
                            ▼
              ┌─── 蒸馏质量 < 60%? ─── 是 ──→ 启用 A.1（蒸馏器专用模型）
              │         否
              │         │
              ▼         ▼
    评估器灰色地带    知识库版本
    误判率 > 30%?    差 ≥ 5?
        │               │
    是  │           是  │
        ▼               ▼
   启用 A.3          启用 A.2
   (评估器 RM)     (调度器协同更新)
```

**原则：先 prompting，后 fine-tune。** Fine-tune 的每一个触发条件都要求 prompting 方案已经运行了足够长的时间，积累了足够的数据和失败证据。不允许在没有实证的情况下"预防性"启用 fine-tune。

### A.5 Fine-Tune 方案对项目结构的影响

如果启用 fine-tune 方案，需要在项目中新增以下目录：

```
assumption_agent/
├── feedback/
│   ├── ...                              # 现有目录不变
│   └── finetuned_models/                # fine-tune 方案新增
│       ├── distiller_model/             # A.1: 蒸馏器专用模型
│       │   ├── train_distiller.py
│       │   ├── config.yaml
│       │   └── checkpoints/
│       ├── evaluator_rm/                # A.3: 评估器奖励模型
│       │   ├── train_evaluator_rm.py
│       │   ├── config.yaml
│       │   └── checkpoints/
│       └── dispatcher_coupdate/         # A.2: 调度器协同更新
│           ├── generate_pseudo_labels.py
│           ├── coupdate.py
│           └── regression_test.py
```

### A.6 Fine-Tune 方案的实验设计

如果启用了任一 fine-tune 方案，需要额外进行以下对比实验：

**实验 H4（对应 A.1）：** Fine-tune 蒸馏器 vs prompting 蒸馏器，在更新候选的人工审核合理率上的差异。

**实验 H5（对应 A.2）：** 调度器协同更新 vs 调度器冻结，在知识库大幅更新后的任务完成率差异。特别关注新增条件的利用率。

**实验 H6（对应 A.3）：** 奖励模型评估器 vs LLM 辅助评估器，在经验筛选的精确率/召回率上的差异。

这些实验的结果应作为论文的补充材料或消融分析呈现，核心论文仍以 prompting 方案为主线。