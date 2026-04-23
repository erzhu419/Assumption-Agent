"""Phase 4 v3 Direction 1: failure-driven candidate wisdom generation.

Input: recent v20 round's (problem, draft, final, judge) for failed problems
       + registry snapshot
Output: 0-2 candidate wisdom dicts (not yet validated)

Differences from Mode B (phase4/propose_new_wisdoms_mode_b.py):
  - Takes a BATCH of recent failures (not all 61 historic residuals)
  - Aware of CURRENT registry state (doesn't re-propose deprecated wisdoms)
  - Conservative: 0-2 candidates per batch (let loop iterate, don't dump)
"""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
from llm_client import parse_json_from_llm
from gpt5_client import GPT5Client


PROPOSE_PROMPT = """你是 wisdom library 的 curator。下面 N 个问题上 v20 架构失败了。你的任务：判断这些失败是否共享一个**现有 library 没覆盖的 orientation**。保守：只有明确 systematic pattern 才提，否则空 list。

## 现有 library（active wisdoms）
{wisdom_brief}

## 最近失败的 batch
{failures_text}

## 判断
1. 这批失败共享某个 pattern 吗？(n≥3)
2. 现有 library 真的不覆盖吗？
3. 若覆盖，是不是 signal 不够明显导致 selection 没选中？（这种情况别提新 wisdom）
4. 若真的缺，提 0-2 条（不要更多）

## 输出 JSON（不要代码块）
{{"candidates": [
  {{
    "rationale": "为什么这条是 new，现有哪条最近但仍不覆盖 (40-80 字)",
    "covers_batch_pids": ["pid..."],  // ≥3 个
    "aphorism": "≤35 中文字符",
    "source": "真实作者+作品 / 民间谚语",
    "signal": "激活条件 (15-30 字)",
    "unpacked_for_llm": "60-120 字 scenario+self-question",
    "cross_domain_examples": [
      {{"domain": "...", "scenario": "30-50 字"}},
      {{"domain": "不同域", "scenario": "30-50 字"}}
    ]
  }}
]}}

**若没有 systematic pattern 或覆盖度足够**：返回 `{{"candidates": []}}`.
"""


def generate_candidates(
    registry: dict,
    failure_batch: list,  # [{pid, problem, v20_answer, opponent_answer, judge_reason}, ...]
    sample: list,
) -> list:
    """Return 0-2 candidate wisdoms (not validated)."""
    if len(failure_batch) < 3:
        return []

    active = [w for w in registry["wisdoms"] if w.get("status") == "active"]
    wisdom_brief = "\n".join(
        f"[{w['id']}] {w['aphorism']} — {w.get('signal','')[:45]}"
        for w in active
    )

    pid_to_info = {p["problem_id"]: p for p in sample}
    failures_lines = []
    for f in failure_batch:
        pid = f["pid"]
        info = pid_to_info.get(pid, {})
        failures_lines.append(
            f"[{pid} / {info.get('domain','?')}] "
            f"problem: {info.get('description','')[:150]}\n"
            f"  v20 answer summary: {f['v20_answer'][:200]}\n"
            f"  opponent answer summary: {f['opponent_answer'][:200]}\n"
            f"  judge: {f['judge_reason'][:150]}\n"
        )
    failures_text = "\n".join(failures_lines)

    client = GPT5Client()
    prompt = PROPOSE_PROMPT.format(
        wisdom_brief=wisdom_brief,
        failures_text=failures_text,
    )

    try:
        resp = client.generate(prompt, max_tokens=2500, temperature=0.35)
        parsed = parse_json_from_llm(resp["text"])
        candidates = parsed.get("candidates", [])
    except Exception as e:
        print(f"  [failure_generator] error: {e}")
        return []

    # Novelty check
    if not candidates:
        return []
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    active_texts = [w["unpacked_for_llm"] for w in active]
    active_embs = model.encode(active_texts, normalize_embeddings=True)

    kept = []
    for c in candidates:
        unp = c.get("unpacked_for_llm", "")
        if not unp or len(unp) < 30:
            continue
        if len(c.get("covers_batch_pids", [])) < 3:
            continue
        new_emb = model.encode([unp], normalize_embeddings=True)[0]
        max_sim = float(np.max(active_embs @ new_emb))
        if max_sim > 0.78:
            print(f"  [reject novelty] '{c.get('aphorism','?')}' sim={max_sim:.2f}")
            continue
        c["novelty_sim"] = max_sim
        c["_source"] = "failure_driven"
        kept.append(c)

    return kept[:2]  # cap at 2 per batch


if __name__ == "__main__":
    # Smoke test on existing residuals
    from wisdom_registry import load_or_init_registry
    r = load_or_init_registry()
    print(f"Registry: {len(r['wisdoms'])} wisdoms")

    # Simulate a failure batch from existing v16 losses
    CACHE = PROJECT / "phase two" / "analysis" / "cache"
    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))
    v20_ans = json.loads((CACHE / "answers" / "phase2_v20_answers.json").read_text(encoding="utf-8"))
    v16_ans = json.loads((CACHE / "answers" / "phase2_v16_answers.json").read_text(encoding="utf-8"))
    judgments = json.loads((CACHE / "judgments" / "phase2_v20_vs_phase2_v16.json").read_text(encoding="utf-8"))

    # Take 5 problems where v20 lost to v16
    batch = []
    for pid, j in judgments.items():
        if j.get("winner") != "phase2_v20":
            batch.append({
                "pid": pid,
                "v20_answer": v20_ans.get(pid, "")[:400],
                "opponent_answer": v16_ans.get(pid, "")[:400],
                "judge_reason": j.get("reasoning", "")[:200],
            })
        if len(batch) >= 5:
            break

    print(f"\nTesting on {len(batch)} v20 failures vs v16 (sanity check)")
    candidates = generate_candidates(r, batch, sample)
    print(f"Candidates returned: {len(candidates)}")
    for c in candidates:
        print(f"  - {c.get('aphorism','?')} ({c.get('source','?')}) sim={c.get('novelty_sim',0):.2f}")
