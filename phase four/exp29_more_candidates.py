"""Exp 29 — Scale candidates: distill from smaller clusters (size 3-5).

Original success_distiller required cluster size ≥ 6 → produced 12 candidates.
Of 44 total clusters, 32 have size 3-5 (unexplored). Here we relax and
ask GPT-5.4 (expensive tier — distillation is a reasoning task) to
propose wisdom from these smaller clusters. Novelty-filtered as before.

Partially addresses reviewer objection #1 (N too small).
"""

import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase four"))

from sentence_transformers import SentenceTransformer
from model_router import expensive
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
CLUSTERS_PATH = AUTO_DIR / "success_distill_clusters.json"
OUT_CANDIDATES = AUTO_DIR / "exp29_extra_candidates.json"

MIN_SIZE_INCLUSIVE = 3
MAX_SIZE_EXCLUSIVE = 6  # include cluster sizes 3, 4, 5 (previously excluded)
NOVELTY_MAX_SIM = 0.78
PARALLEL = 4


DISTILL_PROMPT = """你是 wisdom library 设计师。下面是 v19b/v20 在 {n} 个问题上 Turn 0 产生的**同一个 cluster** 的 reframing signals。

这是**小 cluster** (size {n})，之前的 distill 用 ≥6 阈值过滤掉了。我们现在放宽检查它。如果不 coherent 就 reject。

## 现有 wisdom library（确认 cluster 不是重复）
{wisdom_brief}

## Cluster 内的 reframings
{cluster_signals}

## 你的任务
判断这批 reframings 是否共享 coherent orientation 且 novel。若是:

1. 起名 + aphorism (≤35 字)
2. source (真实作者/作品 or 民间谚语，不要编造)
3. signal (15-30 字触发情境)
4. unpacked_for_llm (60-120 字 scenario+self-question)
5. 两个 cross_domain_examples

## 输出 JSON (不要代码块)
若不 coherent / 重复: {{"proposal": null, "reason": "..."}}
若 coherent + novel:
{{"proposal": {{
  "aphorism": "≤35 字",
  "source": "真实作者+作品 / 民间谚语",
  "signal": "15-30 字",
  "unpacked_for_llm": "60-120 字",
  "cross_domain_examples": [
    {{"domain": "...", "scenario": "30-50 字"}},
    {{"domain": "不同域", "scenario": "..."}}
  ],
  "rationale": "为什么 coherent + novel (50-80 字)",
  "cluster_signal_count": {n}
}}}}
"""


def cache_load(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return default


def main():
    clusters = cache_load(CLUSTERS_PATH)
    if not clusters:
        print(f"{CLUSTERS_PATH.name} not found"); return

    # We need to re-collect signals + cluster labels. The stored clusters
    # file only has big_clusters (size ≥ 6). Recollect from source.
    # Import success_distiller helpers.
    sys.path.insert(0, str(PROJECT / "phase four"))
    from success_distiller import collect_rewritings, cluster_signals as do_cluster

    signals = collect_rewritings()
    print(f"Collected {len(signals)} signals; re-clustering...")

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    clusters_dict, labels = do_cluster(signals, model)
    print(f"{len(clusters_dict)} total clusters")

    # Filter to medium-size clusters
    mid_clusters = {k: idx for k, idx in clusters_dict.items()
                     if MIN_SIZE_INCLUSIVE <= len(idx) < MAX_SIZE_EXCLUSIVE}
    print(f"{len(mid_clusters)} clusters with size {MIN_SIZE_INCLUSIVE}-"
          f"{MAX_SIZE_EXCLUSIVE - 1} (previously skipped)")

    # Existing library for novelty check
    lib = json.loads((CACHE / "wisdom_library.json").read_text(encoding="utf-8"))
    # Plus existing candidates (avoid duplicating)
    existing = cache_load(AUTO_DIR / "success_distilled_candidates.json", default=[]) + \
               cache_load(AUTO_DIR / "cross_llm_candidates.json", default=[])
    all_existing = lib + existing
    existing_texts = [w.get("unpacked_for_llm", "") for w in all_existing
                       if w.get("unpacked_for_llm")]
    existing_embs = model.encode(existing_texts, normalize_embeddings=True,
                                  show_progress_bar=False)

    wisdom_brief = "\n".join(
        f"[{w.get('id','?')}] {w.get('aphorism','?')} — {w.get('signal','')[:40]}"
        for w in lib[:30]
    )

    gpt = expensive("gpt5")  # GPT-5.4 for distillation reasoning
    print(f"Distiller: {gpt.model}\n")

    candidates = []
    def task(cluster_id, indices):
        n = len(indices)
        sigs = [signals[i] for i in indices]
        cluster_text = "\n".join(
            f"[{s['pid']}] what_changed: {s['what_changed']}\n  "
            f"critical_reframe: {s['critical_reframe'][:150]}"
            for s in sigs
        )
        prompt = DISTILL_PROMPT.format(
            n=n, wisdom_brief=wisdom_brief, cluster_signals=cluster_text,
        )
        try:
            resp = gpt.generate(prompt, max_tokens=1200, temperature=0.35)
            parsed = parse_json_from_llm(resp["text"])
            proposal = parsed.get("proposal")
            if not proposal:
                return cluster_id, None, parsed.get("reason", "no proposal")
            # Novelty check
            unp = proposal.get("unpacked_for_llm", "")
            if not unp:
                return cluster_id, None, "empty unpacked"
            new_emb = model.encode([unp], normalize_embeddings=True)[0]
            max_sim = float(np.max(existing_embs @ new_emb))
            if max_sim > NOVELTY_MAX_SIM:
                return cluster_id, None, f"novelty_sim={max_sim:.2f} > {NOVELTY_MAX_SIM}"
            proposal["novelty_sim"] = max_sim
            proposal["_cluster_id"] = int(cluster_id)
            proposal["_cluster_size"] = n
            proposal["_source"] = "success_distilled_mid"
            return cluster_id, proposal, f"novel sim={max_sim:.2f}"
        except Exception as e:
            return cluster_id, None, f"err: {e}"

    print(f"Distilling {len(mid_clusters)} clusters...")
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(task, cid, idx) for cid, idx in mid_clusters.items()]
        for f in as_completed(futs):
            cid, prop, note = f.result()
            done += 1
            if prop:
                candidates.append(prop)
                print(f"  [cluster {cid} n={prop['_cluster_size']}] ✓ "
                      f"'{prop['aphorism']}' ({note})")
            else:
                print(f"  [cluster {cid}] ✗ {note[:60]}")
            if done % 10 == 0:
                print(f"  -- {done}/{len(mid_clusters)} ({time.time()-t0:.0f}s) --")

    # Dedup by aphorism across what we produced
    seen = set(); uniq = []
    for c in candidates:
        a = c["aphorism"]
        if a in seen: continue
        seen.add(a); uniq.append(c)

    OUT_CANDIDATES.write_text(json.dumps(uniq, ensure_ascii=False, indent=2))
    print(f"\n{len(uniq)} novel candidates from small clusters, saved → "
          f"{OUT_CANDIDATES.name}")

    # Total candidates now available
    total = len(existing) + len(uniq)
    print(f"\nTotal candidates available: {len(existing)} (previous) + "
          f"{len(uniq)} (new) = {total}")


if __name__ == "__main__":
    main()
