"""
Phase 2 v7a — signal-level retrieval (replace category-key with embedding top-K).

Motivation: v6 showed category-key retrieval is too coarse — new mined triggers
get distributed by domain/difficulty, but the LLM doesn't know which actually
match the CURRENT problem's signal. Embedding retrieval picks by semantic
similarity to the problem itself.

Pool:
  - 301 triggers (trigger_library_v6.json, flat)
  - 75 wisdom entries (wisdom_library.json)

Per problem:
  - Stage-1 priors (as before, category-based; these are generic per-category)
  - Top-4 triggers by cosine similarity to problem description
  - Top-3 wisdom by cosine similarity

Model: paraphrase-multilingual-MiniLM-L12-v2 (pre-computed via build_signal_embeddings.py)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg  # noqa: E402
from llm_client import create_client  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from cached_framework import _generate_with_retry


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"
WISDOM_PATH = CACHE / "wisdom_library.json"
EMB_PATH = CACHE / "signal_embeddings.npz"


EXECUTE_V7A = """# 你要解决下面的问题。

## 思维背景 1：本类别的通用 attention priors
{priors_block}

## 思维背景 2：与本题信号最匹配的警觉（embedding-retrieved）
{triggers_block}

## 思维背景 3：与本题信号最匹配的跨文明 wisdom（embedding-retrieved）
{wisdom_block}

## 问题
{problem}

## 要求
- **不要**用 Step 1/2 格式
- 带这三层觉知直接答题；引用 aphorism 仅当真 fire
- 语言精炼，不超过 500 字

开始：
"""


def cache_load(p: Path):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def cache_save(p: Path, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def format_priors(priors: List[str]) -> str:
    return "\n".join(f"  {i+1}. {p}" for i, p in enumerate(priors)) or "  (无)"


def format_triggers(trs: List[str]) -> str:
    return "\n".join(f"  - {t}" for t in trs) or "  (无)"


def format_wisdom(entries: List[Dict]) -> str:
    if not entries:
        return "  (无匹配)"
    return "\n".join(
        f"  • 【{e['id']} — {e['source']}】《{e['aphorism']}》\n"
        f"      {e['unpacked_for_llm']}"
        for e in entries)


def retrieve_top_k(query_vec: np.ndarray, pool_vecs: np.ndarray, k: int) -> List[int]:
    sims = pool_vecs @ query_vec
    return np.argsort(-sims)[:k].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_v7a")
    ap.add_argument("--base", default="orient_hybrid")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k-triggers", type=int, default=4)
    ap.add_argument("--k-wisdom", type=int, default=3)
    args = ap.parse_args()

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    retrievals_path = CACHE / "phase2_v7a_retrievals.json"
    answers = cache_load(answers_path)
    retrievals = cache_load(retrievals_path)

    if not struct_path.exists():
        base_struct = cache_load(STRUCTURES_DIR / f"{args.base}_structures.json")
        if base_struct:
            cache_save(struct_path, base_struct)
    structures = cache_load(struct_path)

    # Load embeddings
    emb = np.load(EMB_PATH, allow_pickle=True)
    trig_emb = emb["trigger_emb"]
    trig_texts = emb["trigger_texts"].tolist()
    wis_emb = emb["wisdom_emb"]
    wis_ids = emb["wisdom_ids"].tolist()
    prob_emb = emb["problem_emb"]
    prob_ids_emb = emb["problem_ids"].tolist()
    prob_idx_map = {pid: i for i, pid in enumerate(prob_ids_emb)}

    wisdom = json.loads(WISDOM_PATH.read_text(encoding="utf-8"))
    wis_by_id = {e["id"]: e for e in wisdom}

    print(f"  triggers pool: {len(trig_texts)} | wisdom pool: {len(wis_ids)}")
    print(f"  retrieving top-{args.k_triggers} triggers + top-{args.k_wisdom} wisdom per problem")

    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))[: args.n]

    client = create_client()
    t0 = time.time()
    new_count = 0
    hit_count = 0

    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in answers:
            hit_count += 1
            continue
        dom = p.get("domain", "?")
        diff = p.get("difficulty", "?")
        key = f"{dom}__{diff}"
        struct = structures.get(key)
        if struct is None:
            print(f"  [skip] {pid}: no structure for {key}")
            continue

        priors = struct.get("attention_priors", [])
        priors_block = format_priors(priors)

        # Embedding retrieval
        if pid not in prob_idx_map:
            print(f"  [skip] {pid}: no embedding")
            continue
        qv = prob_emb[prob_idx_map[pid]]
        top_t = retrieve_top_k(qv, trig_emb, args.k_triggers)
        top_w = retrieve_top_k(qv, wis_emb, args.k_wisdom)
        sel_triggers = [trig_texts[j] for j in top_t]
        sel_wisdom = [wis_by_id[wis_ids[j]] for j in top_w if wis_ids[j] in wis_by_id]

        retrievals[pid] = {
            "triggers": sel_triggers,
            "wisdom_ids": [wis_ids[j] for j in top_w],
        }

        prompt = EXECUTE_V7A.format(
            priors_block=priors_block,
            triggers_block=format_triggers(sel_triggers),
            wisdom_block=format_wisdom(sel_wisdom),
            problem=p.get("description", ""),
        )

        try:
            resp = _generate_with_retry(client, prompt, max_tokens=900, temperature=0.3)
            answers[pid] = resp["text"].strip()
        except Exception as e:
            print(f"  [error] {pid}: {e}")
            continue

        new_count += 1
        if new_count % 10 == 0:
            cache_save(answers_path, answers)
            cache_save(retrievals_path, retrievals)
            print(f"  [{args.variant}] {i+1}/{len(sample)} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    cache_save(retrievals_path, retrievals)
    print(f"\n  [{args.variant}] done: {new_count} new, {hit_count} cached ({time.time()-t0:.0f}s)")

    # Stats: which triggers/wisdom got picked most
    from collections import Counter
    t_c = Counter()
    w_c = Counter()
    for r in retrievals.values():
        for t in r.get("triggers", []):
            t_c[t[:40]] += 1
        for wid in r.get("wisdom_ids", []):
            w_c[wid] += 1
    print("\n  Top 10 most-retrieved triggers:")
    for t, n in t_c.most_common(10):
        print(f"    {n:>3}x  {t}")
    print("\n  Top 10 most-retrieved wisdom:")
    for wid, n in w_c.most_common(10):
        if wid in wis_by_id:
            print(f"    {n:>3}x  [{wid}] {wis_by_id[wid]['aphorism']}")


if __name__ == "__main__":
    main()
