"""
Mining v2 — A+B combined, mined by GPT-5.4.

A: Aphorism-style — each trigger ≤ 35 chars, punchy, memorable (not template).
B: Cluster-then-condense — cluster losses semantically, condense each cluster
   into ONE aphorism capturing the shared pattern.

Output: trigger_library_v2.json with ~10-15 high-quality universal triggers
(not category-keyed — all triggers shown to all problems).
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

from llm_client import parse_json_from_llm  # noqa: E402
from gpt5_client import GPT5Client  # noqa: E402

CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
OUT_PATH = CACHE / "trigger_library_v2.json"


CONDENSE_PROMPT = """你是方法论警句提炼者。下面是同一类失败模式的 N 个具体案例。你的任务是提炼一条**严格跨领域通用**的警句式 trigger。

## 失败案例（N 条，同一 failure pattern）

{cases_block}

## 硬性要求（违反任何一条，trigger 无效）

1. **长度 ≤ 35 中文字符**，警句化、可作为 meme 记忆
2. **跨领域通用 —— 最严格约束**：
   - trigger 文本里**不能出现任何 domain-specific 术语**：
     - 禁词：代码、debug、调试、CI、客户、用户、员工、产品、订单、投资、股票、算法、证明、实验、分子、细胞、家庭、孩子、父母、同事...
     - 一旦出现这些词，意味着 trigger 还没被提炼到"通用"层
   - 合法的抽象词汇：结论、假设、约束、前提、决策、征兆、模式、权衡、反例、权威、习惯、自尊、安全感、变化速率...
3. **有 insight compression**：读完后读者应该觉得"啊"，不是"是的我知道"
4. **必须有可察觉的情境信号** —— 能在读完问题一瞬间辨认出"应该激活这条"
5. **禁用 meta-reflection 模板句**：
   - 禁写：是否充分识别、是否充分考虑、是否充分评估、是否已重述、是否已清晰定义...
   - 合法：直接陈述 pattern，或"X 出现时 Y 值得重审"式的 conditional statement

## 参考优秀警句风格（不要复用，只为让你懂风格）

- "能解释一切的理论解释不了任何事"（Popper）
- "已有的后必再有"（传道书 1:9）
- "温水煮青蛙"（成语）
- "一叶知秋"（成语）
- "让你舒服的答案通常是错的"（实用智慧）
- "紧迫感是决策最糟的顾问"（经验）

## 提炼流程

1. 从 N 个失败案例中**抽象出共同的结构性模式**（不是共同的 domain）
2. 用**完全通用的抽象词汇**表述这个 pattern
3. 压缩成 ≤35 字的警句
4. 自检：把这句话给没看过任何案例的人，他能理解并在完全不同的领域里用上吗？如果不能，重写。

## 同时给出：

- **核心 insight**：一句话说清楚这个 pattern 是什么（不受 35 字限制）
- **领域 A 示例**：一个具体领域的适用场景
- **领域 B 示例**：**完全不同**的领域的适用场景（如果 A 是 sw_eng，B 必须是 daily_life / business / math / science / engineering 其中之一）
- **abstraction check**：明确声明 trigger 中没有 domain 特异术语

## 输出 JSON（不要代码块）

{{
  "trigger_aphorism": "≤35 字，完全通用的警句",
  "core_insight": "一句话 pattern 说明",
  "cross_domain_example_a": "领域 A 中的适用场景",
  "cross_domain_example_a_domain": "daily_life / business / sw_eng / ...",
  "cross_domain_example_b": "领域 B（与 A 完全不同）中的适用场景",
  "cross_domain_example_b_domain": "与 A 不同的领域",
  "abstraction_check": "我的 trigger 中没有 domain 特异术语。用了哪些抽象词汇？[列出]"
}}
"""

# Post-filter: reject triggers containing these domain-specific tokens
DOMAIN_SPECIFIC_TOKENS = [
    "代码", "debug", "调试", "CI", "CD", "bug", "Bug", "服务器",
    "客户", "用户", "员工", "产品", "订单", "投资", "股票", "金融",
    "算法", "证明", "定理", "公式", "实验", "分子", "细胞", "基因",
    "家庭", "孩子", "父母", "配偶", "同事",
    "商业", "商务", "project", "项目",  # mild domain flavor
    "代码库", "数据库",
]


def _embed_cases(cases: List[Dict]) -> np.ndarray:
    """Use our cached sentence-transformer embeddings if available; else fresh."""
    from pathlib import Path as _P
    emb_cache_path = PROJECT / "cache" / "embeddings.npz"
    if emb_cache_path.exists():
        data = np.load(emb_cache_path, allow_pickle=True)
        ids = data["ids"]
        embs = data["embeddings"]
        id_to_emb = dict(zip(ids, embs))
        out = []
        for c in cases:
            pid = c["problem_id"]
            if pid in id_to_emb:
                out.append(id_to_emb[pid])
            else:
                out.append(np.zeros(384, dtype=np.float32))
        arr = np.stack(out)
        print(f"  [embed] using cached embeddings (dim={arr.shape[1]}, n={arr.shape[0]})")
        return arr

    # Fallback: load sentence-transformer fresh
    from sentence_transformers import SentenceTransformer
    print("  [embed] loading sentence-transformer (slow)...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    texts = [c["problem"][:400] for c in cases]
    return model.encode(texts, batch_size=16)


def cluster_cases(cases: List[Dict], n_clusters: int = 7) -> List[List[int]]:
    """Agglomerative clustering on case embeddings.
    Returns list of [case_indices] per cluster."""
    emb = _embed_cases(cases)
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    )
    labels = clustering.fit_predict(emb)
    clusters: List[List[int]] = [[] for _ in range(n_clusters)]
    for i, lbl in enumerate(labels):
        clusters[int(lbl)].append(i)
    # Sort by cluster size descending
    clusters.sort(key=len, reverse=True)
    return clusters


def build_cases_block(cluster_cases_: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(cluster_cases_):
        parts.append(
            f"### 案例 {i+1}\n"
            f"**领域/难度**: {c['domain']} / {c['difficulty']}\n"
            f"**问题**: {c['problem'][:400]}\n"
            f"**我们的答案（输了）**: {c['our_answer'][:350]}\n"
            f"**对手的答案（赢了）**: {c['baseline_answer'][:350]}\n"
            f"**评委理由**: {c['judge_reasoning']}\n"
        )
    return "\n".join(parts)


def main():
    judgments = json.loads((CACHE / "judgments" / "orient_hybrid_vs_baseline.json").read_text(encoding="utf-8"))
    our_ans = json.loads((CACHE / "answers" / "orient_hybrid_answers.json").read_text(encoding="utf-8"))
    base_ans = json.loads((CACHE / "answers" / "baseline_answers.json").read_text(encoding="utf-8"))
    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))
    prob_by_id = {p["problem_id"]: p for p in sample}

    losses = []
    for pid, j in judgments.items():
        if j.get("winner") != "baseline":
            continue
        if pid not in our_ans or pid not in base_ans or pid not in prob_by_id:
            continue
        losses.append({
            "problem_id": pid,
            "domain": j.get("domain", "?"),
            "difficulty": j.get("difficulty", "?"),
            "problem": prob_by_id[pid].get("description", ""),
            "our_answer": our_ans[pid],
            "baseline_answer": base_ans[pid],
            "judge_reasoning": j.get("reasoning", ""),
        })

    print(f"Loaded {len(losses)} loss cases")
    domains_present = sorted(set(c["domain"] for c in losses))

    # Cluster
    n_clusters = 7
    clusters = cluster_cases(losses, n_clusters=n_clusters)
    print(f"\nClustering into {n_clusters} groups by embedding similarity:")
    for i, idx_list in enumerate(clusters):
        doms = [losses[j]["domain"] for j in idx_list]
        print(f"  cluster {i}: {len(idx_list)} cases  domains={dict((d, doms.count(d)) for d in set(doms))}")

    # Condense each cluster via GPT-5.4
    client = GPT5Client()
    results = []
    t0 = time.time()
    for ci, idx_list in enumerate(clusters):
        if len(idx_list) < 2:
            print(f"  [skip cluster {ci}] only {len(idx_list)} cases")
            continue
        cluster_cases_ = [losses[i] for i in idx_list[:8]]  # cap 8 cases per cluster
        cases_block = build_cases_block(cluster_cases_)
        prompt = CONDENSE_PROMPT.format(
            cases_block=cases_block,
            domains_available=", ".join(domains_present),
        )
        try:
            resp = client.generate(prompt, max_tokens=2000, temperature=0.4)
            parsed = parse_json_from_llm(resp["text"])
        except Exception as e:
            print(f"  [cluster {ci}] error: {e}")
            continue

        trigger = parsed.get("trigger_aphorism", "").strip()
        if not trigger or len(trigger) > 60:
            print(f"  [cluster {ci}] rejected length: {trigger!r}")
            continue

        # Post-filter: check for domain-specific tokens
        violations = [tok for tok in DOMAIN_SPECIFIC_TOKENS if tok in trigger]
        if violations:
            print(f"  [cluster {ci}] rejected domain-specific: {trigger!r} contains {violations}")
            continue

        # Post-filter: check for banned meta-reflection templates
        banned_phrases = ["是否充分识别", "是否充分考虑", "是否充分评估",
                          "是否已重述", "是否已清晰", "是否已经识别"]
        banned_found = [p for p in banned_phrases if p in trigger]
        if banned_found:
            print(f"  [cluster {ci}] rejected template: {trigger!r} contains {banned_found}")
            continue

        # Check that example A and B are in different domains
        dom_a = (parsed.get("cross_domain_example_a_domain", "") or "").lower().strip()
        dom_b = (parsed.get("cross_domain_example_b_domain", "") or "").lower().strip()
        if dom_a and dom_b and dom_a == dom_b:
            print(f"  [cluster {ci}] rejected same-domain examples: {dom_a} == {dom_b}")
            continue

        results.append({
            "cluster_id": ci,
            "n_cases": len(idx_list),
            "trigger": trigger,
            "core_insight": parsed.get("core_insight", ""),
            "example_a": parsed.get("cross_domain_example_a", ""),
            "example_a_domain": dom_a,
            "example_b": parsed.get("cross_domain_example_b", ""),
            "example_b_domain": dom_b,
            "abstraction_check": parsed.get("abstraction_check", ""),
            "source_problem_ids": [losses[i]["problem_id"] for i in idx_list],
        })
        print(f"\n  [cluster {ci}] ({len(idx_list)} cases) → {trigger}")
        print(f"    insight: {parsed.get('core_insight','')[:120]}")
        print(f"    A ({dom_a}): {parsed.get('cross_domain_example_a','')[:80]}")
        print(f"    B ({dom_b}): {parsed.get('cross_domain_example_b','')[:80]}")

    # Save
    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n\nSaved {len(results)} v2 triggers to {OUT_PATH.name}")
    print(f"Total time: {time.time()-t0:.0f}s")

    # Summary
    print(f"\n=== Summary ===")
    for r in results:
        print(f"  [{r['n_cases']} cases]  {r['trigger']}")


if __name__ == "__main__":
    main()
