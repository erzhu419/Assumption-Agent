"""Phase 4 v3 Direction 2: success-driven wisdom distillation from v19b/v20 rewritings.

Every v19b/v20 Turn 0 produces (rewritten_problem, what_changed) + frame info.
These capture the LLM's own judgment of "what's the true reframe for this
problem". Accumulating many such rewritings reveals STABLE orientation
patterns — candidates for new wisdom.

Pipeline:
  1. Collect all available rewriting signals (v20 meta + v19b rewrites + loop meta)
  2. Embed `what_changed` + `critical_reframe` (dual-signal)
  3. Agglomerative cluster (Ward linkage, distance threshold)
  4. For each cluster with ≥N items: GPT-5.4 proposes wisdom formalization
  5. Novelty check vs existing library
  6. Output candidates

Note: unlike failure_generator which ran per-round, success_distiller is
batch-mined (not in-loop). Candidates then go through same A/B validation
in autonomous_loop.
"""

import json
import sys
import time
import glob
from pathlib import Path
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
from llm_client import parse_json_from_llm
from gpt5_client import GPT5Client


CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
WISDOM_REG = AUTO_DIR / "wisdom_registry.json"
OUT_CANDIDATES = AUTO_DIR / "success_distilled_candidates.json"
OUT_CLUSTERS = AUTO_DIR / "success_distill_clusters.json"


MIN_CLUSTER_SIZE = 6        # at least 6 rewritings to consider a pattern
DISTANCE_THRESHOLD = 0.4    # agglomerative merge threshold on normalized embeddings
NOVELTY_MAX_SIM = 0.78      # max cosine sim to existing wisdom; below = novel


DISTILL_PROMPT = """你是 wisdom library 设计师。下面是 v19b/v20 架构在 {n} 个问题上 Turn 0 产生的 **同一 cluster** 的 reframing signals——LLM 自己判定"这类问题真正该怎么看"。

这不是一次性现象，是**反复出现的 reframing pattern**——很可能反映一个**现有 library 没明确 articulate 的 orientation**。

## 现有 wisdom library（请确认这条 cluster 不是简单重复）
{wisdom_brief}

## Cluster 内的 reframings
{cluster_signals}

## 你的任务
判断这批 reframings 是否共享一个 **coherent orientation**。若是：

1. 给 orientation 起个名字 + aphorism (≤35 中文字符)
2. 标 source (真实作者+作品，或 "民间谚语"，**不要编造**)
3. 提炼 signal (什么情境激活，15-30 字)
4. unpacked_for_llm (60-120 字 scenario+self-question)
5. 两个 cross_domain_examples (不同 domain)

## 输出 JSON（不要代码块）
若 cluster 不 coherent 或和现有 library 重复：`{{"proposal": null, "reason": "..."}}`

若 coherent 且 novel:
{{"proposal": {{
  "aphorism": "≤35 字",
  "source": "真实作者+作品 / 民间谚语",
  "signal": "15-30 字",
  "unpacked_for_llm": "60-120 字",
  "cross_domain_examples": [
    {{"domain": "...", "scenario": "30-50 字"}},
    {{"domain": "不同域", "scenario": "..."}}
  ],
  "rationale": "为什么这是 coherent + novel (50-80 字)",
  "cluster_signal_count": {n}
}}}}
"""


def collect_rewritings():
    """Collect (problem_pid, what_changed, critical_reframe, source_file) tuples."""
    signals = []

    # 1. v20 meta files (main runs + autonomous loop)
    meta_files = glob.glob(str(CACHE / "answers" / "*_meta.json"))
    for f in meta_files:
        try:
            m = json.loads(Path(f).read_text(encoding="utf-8"))
            for pid, data in m.items():
                wc = data.get("what_changed", "").strip()
                cr = data.get("critical_reframe", "").strip()
                if wc and cr and len(wc) > 10:
                    signals.append({
                        "pid": pid,
                        "what_changed": wc,
                        "critical_reframe": cr,
                        "frame": data.get("frame", "?"),
                        "source": Path(f).stem,
                    })
        except Exception as e:
            print(f"  [skip {Path(f).name}] {e}")

    # 2. v19b rewrites
    rw_path = CACHE / "answers" / "phase2_v19b_rewrites.json"
    if rw_path.exists():
        rw = json.loads(rw_path.read_text(encoding="utf-8"))
        for pid, data in rw.items():
            wc = data.get("what_changed", "").strip()
            rp = data.get("rewritten_problem", "").strip()
            if wc and len(wc) > 10:
                signals.append({
                    "pid": pid,
                    "what_changed": wc,
                    "critical_reframe": rp[:200],  # v19b's 'rewritten' serves as reframe
                    "frame": "?",
                    "source": "v19b_rewrites",
                })

    # Deduplicate by (pid, source)
    seen = set()
    unique = []
    for s in signals:
        key = (s["pid"], s["source"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)

    return unique


def cluster_signals(signals, model):
    """Cluster by what_changed + critical_reframe combined embedding."""
    texts = [
        f"{s['what_changed']} | {s['critical_reframe']}"
        for s in signals
    ]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # Agglomerative clustering with cosine distance threshold
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DISTANCE_THRESHOLD,
        metric="cosine",
        linkage="average",
    )
    labels = clusterer.fit_predict(embs)

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(i)

    return clusters, labels


def main():
    print("=== success_distiller ===")

    # Collect signals
    signals = collect_rewritings()
    print(f"Collected {len(signals)} rewriting signals")
    by_src = Counter(s["source"] for s in signals)
    for src, n in by_src.most_common():
        print(f"  {src}: {n}")

    if len(signals) < 20:
        print("Too few signals to cluster")
        return

    # Cluster
    print(f"\nLoading sentence-transformer...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print(f"Clustering (distance_threshold={DISTANCE_THRESHOLD})...")
    clusters, labels = cluster_signals(signals, model)
    print(f"  → {len(clusters)} clusters total")

    # Filter by size
    big_clusters = {k: v for k, v in clusters.items() if len(v) >= MIN_CLUSTER_SIZE}
    print(f"  → {len(big_clusters)} clusters with ≥{MIN_CLUSTER_SIZE} items")

    # Save cluster breakdown
    cluster_report = {
        "n_signals": len(signals),
        "n_clusters": len(clusters),
        "big_clusters": {
            str(k): {
                "size": len(v),
                "sample_signals": [signals[i] for i in v[:3]],
            }
            for k, v in sorted(big_clusters.items(), key=lambda x: -len(x[1]))
        },
    }
    OUT_CLUSTERS.write_text(json.dumps(cluster_report, ensure_ascii=False, indent=2))
    print(f"  Cluster report → {OUT_CLUSTERS.name}")

    # Load existing wisdom library from registry
    registry = json.loads(WISDOM_REG.read_text(encoding="utf-8"))
    active = [w for w in registry["wisdoms"] if w.get("status") == "active"]
    wisdom_brief = "\n".join(
        f"[{w['id']}] {w['aphorism']} — {w.get('signal','')[:45]}"
        for w in active
    )

    existing_texts = [w["unpacked_for_llm"] for w in active]
    existing_embs = model.encode(existing_texts, normalize_embeddings=True,
                                  show_progress_bar=False)

    # GPT-5.4 distill each big cluster
    client = GPT5Client()
    candidates = []
    print(f"\nDistilling {len(big_clusters)} clusters via GPT-5.4...")

    for cluster_id, indices in sorted(big_clusters.items(), key=lambda x: -len(x[1])):
        n = len(indices)
        sigs_in_cluster = [signals[i] for i in indices]
        # Show up to 10 signals to GPT-5.4
        sample_signals = sigs_in_cluster[:10]
        cluster_signals_text = "\n".join(
            f"[{s['pid']}] what_changed: {s['what_changed']}\n  critical_reframe: {s['critical_reframe'][:150]}"
            for s in sample_signals
        )

        try:
            prompt = DISTILL_PROMPT.format(
                n=n, wisdom_brief=wisdom_brief, cluster_signals=cluster_signals_text,
            )
            resp = client.generate(prompt, max_tokens=1200, temperature=0.35)
            parsed = parse_json_from_llm(resp["text"])
            proposal = parsed.get("proposal")
            if not proposal:
                reason = parsed.get("reason", "no proposal")
                print(f"  [cluster {cluster_id} n={n}] rejected: {reason[:80]}")
                continue

            # Novelty check
            unp = proposal.get("unpacked_for_llm", "")
            if not unp:
                continue
            new_emb = model.encode([unp], normalize_embeddings=True)[0]
            max_sim = float(np.max(existing_embs @ new_emb))
            proposal["novelty_sim"] = max_sim
            proposal["_cluster_id"] = int(cluster_id)
            proposal["_cluster_size"] = n
            proposal["_source"] = "success_distilled"
            proposal["covers_batch_pids"] = [s["pid"] for s in sigs_in_cluster[:5]]

            if max_sim > NOVELTY_MAX_SIM:
                print(f"  [cluster {cluster_id} n={n}] '{proposal.get('aphorism','?')}' "
                      f"REJECT novelty sim={max_sim:.2f}")
                continue

            print(f"  [cluster {cluster_id} n={n}] '{proposal.get('aphorism','?')}' "
                  f"KEEP sim={max_sim:.2f}")
            candidates.append(proposal)
        except Exception as e:
            print(f"  [cluster {cluster_id}] error: {e}")

    OUT_CANDIDATES.write_text(json.dumps(candidates, ensure_ascii=False, indent=2))
    print(f"\n=== Summary ===")
    print(f"  Signals: {len(signals)}")
    print(f"  Clusters ≥{MIN_CLUSTER_SIZE}: {len(big_clusters)}")
    print(f"  Novel candidates: {len(candidates)}")
    print(f"  Saved to {OUT_CANDIDATES.name}")

    if candidates:
        print(f"\n=== Proposed wisdoms ===")
        for c in candidates:
            print(f"  • {c.get('aphorism','?')} ({c.get('source','?')}) "
                  f"cluster n={c.get('_cluster_size','?')}, sim={c.get('novelty_sim',0):.2f}")


if __name__ == "__main__":
    main()
