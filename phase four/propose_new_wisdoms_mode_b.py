"""
Phase 4 v2 Mode B: propose NEW wisdoms from v16 residuals.

Not relying on GPT-5.4's earlier wisdom_applicability scores (too optimistic).
Instead: take all 61 what_v16_missed descriptions, feed them to GPT-5.4 for
cross-cluster abstraction, and ask for 2-4 genuinely NEW orientations that:
  1. Capture a pattern across multiple residuals (not a one-off)
  2. Are NOT subsumed by any of the 75 existing wisdoms
  3. Follow the standard wisdom schema (aphorism + source + signal + unpacked_for_llm + cross_domain_examples)

Then:
  - Embedding novelty check (max cosine sim to existing wisdoms < 0.75)
  - If pass: add to wisdom_library_v18.json
  - If fail: re-request with "avoid X" constraint
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from llm_client import parse_json_from_llm
from gpt5_client import GPT5Client


CACHE = PROJECT / "phase two" / "analysis" / "cache"
WISDOM_ORIG = CACHE / "wisdom_library.json"
WISDOM_OUT = CACHE / "wisdom_library_v18.json"
RESIDUALS = PROJECT / "phase four" / "residuals" / "v16_residuals.json"
NEW_WISDOMS_OUT = PROJECT / "phase four" / "residuals" / "mode_b_new_wisdoms.json"


PROPOSE_PROMPT = """你是 wisdom library 的资深设计师。你要从 {n} 个 v16 架构的 residual 失败里，提炼 **2-4 条真正 NEW 的 orientation**（现有 75 条 library 不覆盖的）。

## 背景
v16 架构在这 {n} 个问题上失败了，分析已经显示：
- 绝大部分 residual 都能"套上"某条现有 wisdom（W025 最多 14 次，W031 7 次等）
- **但实际跑 v16 with aggressive W025 push (v16_sel_v2) 总分基本没变**
- 这说明：**现有 wisdom 的文本其实没真正覆盖这些 residual——GPT 只是在打标签**

你的任务：**跳出现有 75 条的概念空间**，从下面 61 条 what_v16_missed 里抽出现有 wisdom 完全没说清的 orientation。

## 现有 75 条 wisdom library（简要 — 别只是改写它们）
{wisdom_brief}

## 61 条 v16 residual 失败描述
{residuals_text}

## 提炼规则
1. 提 2-4 条，不是越多越好
2. 每条必须对应 ≥ 5 个 residual（不是一次性现象）
3. **不能是现有 wisdom 的改写** — 你要提的必须是现有 library 文本里没有的 orientation
4. 可以是"操作层"（ops-level）或"元认知层"（meta-cognitive），不限
5. source 优先用真实人物/著作（Kahneman/Drucker/Polya 等）；若不合适，写"民间谚语"或"未详"，**不要编造**

## 输出 JSON（不要代码块）
{{"proposals": [
  {{
    "rationale": "为什么这条是 new，现有 library 为什么不覆盖 (40-80 字)",
    "covers_residual_pids": ["pid1", "pid2", ...],  // 至少 5 个
    "aphorism": "≤35 中文字符的警句",
    "source": "真实来源（人名+作品），或 '民间谚语'",
    "signal": "激活条件 (15-30 字)",
    "unpacked_for_llm": "60-120 字 scenario+self-question 形式",
    "cross_domain_examples": [
      {{"domain": "...", "scenario": "30-50 字"}},
      {{"domain": "不同领域", "scenario": "30-50 字"}}
    ]
  }},
  ... (2-4 条)
]}}
"""


def main():
    # Load data
    orig_lib = json.loads(WISDOM_ORIG.read_text(encoding="utf-8"))
    wisdom_brief = "\n".join(
        f"[{e['id']}] {e['aphorism']} — {e.get('signal', '')[:50]}"
        for e in orig_lib
    )

    residuals = json.loads(RESIDUALS.read_text(encoding="utf-8"))
    residuals_text_lines = []
    for pid, r in residuals.items():
        residuals_text_lines.append(
            f"[{pid} / {r.get('domain')}] missed: {r.get('what_v16_missed', '')[:160]}"
        )
    residuals_text = "\n".join(residuals_text_lines)

    print(f"61 residuals loaded, {len(orig_lib)} existing wisdoms")

    # Propose new wisdoms
    client = GPT5Client()
    prompt = PROPOSE_PROMPT.format(
        n=len(residuals),
        wisdom_brief=wisdom_brief,
        residuals_text=residuals_text,
    )
    t0 = time.time()
    resp = client.generate(prompt, max_tokens=3000, temperature=0.4)
    print(f"\nGPT-5.4 proposal ({time.time()-t0:.0f}s)")

    try:
        parsed = parse_json_from_llm(resp["text"])
        proposals = parsed.get("proposals", [])
    except Exception as e:
        print(f"Parse error: {e}")
        print("Raw text:")
        print(resp["text"][:1000])
        return

    print(f"\n{len(proposals)} raw proposals")
    for i, p in enumerate(proposals):
        print(f"\n  [{i+1}] aphorism: {p.get('aphorism', '?')}")
        print(f"      source: {p.get('source', '?')}")
        print(f"      covers: {len(p.get('covers_residual_pids', []))} residuals")
        print(f"      rationale: {p.get('rationale', '')[:100]}")

    # Novelty check via embedding
    print("\n=== Novelty check ===")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    existing_texts = [e["unpacked_for_llm"] for e in orig_lib]
    existing_embs = model.encode(existing_texts, normalize_embeddings=True)

    kept = []
    for i, p in enumerate(proposals):
        unp = p.get("unpacked_for_llm", "")
        if not unp:
            print(f"  [{i+1}] REJECT: no unpacked_for_llm")
            continue
        new_emb = model.encode([unp], normalize_embeddings=True)[0]
        sims = existing_embs @ new_emb
        max_sim = float(np.max(sims))
        max_idx = int(np.argmax(sims))
        max_w = orig_lib[max_idx]
        if max_sim > 0.85:
            print(f"  [{i+1}] REJECT: max_sim={max_sim:.3f} to [{max_w['id']}] {max_w['aphorism']}")
        elif max_sim > 0.70:
            print(f"  [{i+1}] MARGINAL: max_sim={max_sim:.3f} to [{max_w['id']}] {max_w['aphorism']}")
            kept.append(p)  # Keep marginal for now
        else:
            print(f"  [{i+1}] PASS: max_sim={max_sim:.3f} (closest [{max_w['id']}])")
            kept.append(p)

    print(f"\n=== Kept {len(kept)}/{len(proposals)} proposals ===")

    # Assign IDs and save
    for i, p in enumerate(kept):
        p["id"] = f"W{76+i:03d}"
        p["cluster"] = "mode_b_mined"
        p["abstraction_check"] = "Mode B 产出，人工审核"

    NEW_WISDOMS_OUT.write_text(json.dumps(kept, ensure_ascii=False, indent=2))
    print(f"Saved to {NEW_WISDOMS_OUT.name}")

    # Build v18 library = orig + new wisdoms
    v18 = list(orig_lib) + kept
    WISDOM_OUT.write_text(json.dumps(v18, ensure_ascii=False, indent=2))
    print(f"Built {WISDOM_OUT.name}: {len(v18)} total entries")


if __name__ == "__main__":
    main()
