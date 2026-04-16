"""
Phase 0: Build the philosophical methodology knowledge base.
Expands each strategy seed into a full JSON schema using LLM API.
Supports both Gemini and Claude (configure in .env).

Usage:
    python build_kb.py                    # Build all 27 strategies
    python build_kb.py --strategy S01     # Build only S01
    python build_kb.py --dry-run          # Print prompts without calling API
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent))
from strategy_seeds import STRATEGY_SEEDS, COMPOSITION_SEEDS, CATEGORY_DEFINITIONS
from llm_client import create_client, parse_json_from_llm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
KB_DIR = Path(__file__).parent.parent / "kb"
STRATEGY_DIR = KB_DIR / "strategies"
COMPOSITION_DIR = KB_DIR / "compositions"
CHANGE_HISTORY_DIR = Path(__file__).parent.parent / "change_history"

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

EXPAND_STRATEGY_PROMPT = """你是一个哲学方法论和认知科学专家。请根据以下策略种子信息，生成一个完整的策略知识库条目。

## 策略种子
- ID: {id}
- 名称: {name_zh} ({name_en})
- 一句话描述: {one_sentence}
- 类别: {category}
- 文献来源: {sources}

## 输出要求
请输出一个严格的 JSON 对象，包含以下所有字段。注意：
1. operational_steps 必须是结构化对象数组（不是简单字符串），每步包含 step, action, on_difficulty 字段
2. on_difficulty 的值是 null（该步不太可能遇到困难）或一个字符串描述遇困难时应如何递归处理
3. applicability_conditions 的 favorable 至少 3 条，unfavorable 至少 2 条
4. historical_cases 的 successes 至少 3 个（跨至少 2 个不同领域），failures 至少 2 个
5. knowledge_triples 3-4 条，捕捉策略的核心逻辑骨架
6. 所有文本用中文

```json
{{
  "id": "{id}",
  "name": {{
    "zh": "{name_zh}",
    "en": "{name_en}"
  }},
  "aliases": ["别名1", "别名2"],
  "category": "{category}",
  "source_references": [
    {{
      "author": "作者",
      "work": "著作名",
      "year": 年份,
      "chapter": "相关章节（如果知道）",
      "relevance": "与本策略的关系"
    }}
  ],
  "description": {{
    "one_sentence": "{one_sentence}",
    "detailed": "详细描述（100-200字）",
    "intuitive_analogy": "一个直觉类比"
  }},
  "operational_steps": [
    {{
      "step": 1,
      "action": "步骤描述",
      "on_difficulty": null 或 "递归调用 S_XX（策略名）..."
    }}
  ],
  "applicability_conditions": {{
    "favorable": [
      {{
        "condition_id": "{id}_F_001",
        "condition": "适用条件描述",
        "source": "literature",
        "source_ref": "来源引用",
        "confidence": 0.90,
        "supporting_cases": [],
        "contradicting_cases": [],
        "last_updated": "{today}",
        "version": 1,
        "status": "active",
        "locked": false,
        "stability_tier": "foundational"
      }}
    ],
    "unfavorable": [
      {{
        "condition_id": "{id}_U_001",
        "condition": "不适用条件描述",
        "source": "literature",
        "source_ref": "来源引用",
        "confidence": 0.85,
        "supporting_cases": [],
        "contradicting_cases": [],
        "last_updated": "{today}",
        "version": 1,
        "status": "active",
        "locked": false,
        "stability_tier": "foundational"
      }}
    ],
    "failure_modes": [
      {{
        "mode_id": "{id}_FM_001",
        "description": "失败模式描述",
        "source": "literature",
        "confidence": 0.80,
        "observed_cases": []
      }}
    ]
  }},
  "historical_cases": {{
    "successes": [
      {{
        "case_id": "{id}_SUC_001",
        "domain": "领域",
        "case": "案例名称",
        "description": "案例描述（50-100字）",
        "why_this_strategy_worked": "为什么策略在此案例中有效",
        "demonstrates_conditions": ["{id}_F_001"]
      }}
    ],
    "failures": [
      {{
        "case_id": "{id}_FAIL_001",
        "domain": "领域",
        "case": "案例名称",
        "description": "案例描述",
        "why_this_strategy_failed": "为什么策略在此案例中失败",
        "demonstrates_conditions": ["{id}_U_001"]
      }}
    ]
  }},
  "relationships_to_other_strategies": [
    {{
      "related_strategy": "S_XX",
      "relationship_type": "prerequisite|complementary|alternative|subsumption",
      "description": "关系描述"
    }}
  ],
  "knowledge_triples": [
    {{"subject": "...", "relation": "...", "object": "..."}}
  ],
  "formalization_hints": {{
    "mathematical_structure": "数学结构描述",
    "category_theory_analogue": "范畴论类比",
    "information_geometry_analogue": "信息几何类比",
    "connection_to_known_algorithms": ["算法1", "算法2"],
    "markov_kernel_prior_hint": {{
      "dominant_actions_by_state": {{
        "示例状态条件": ["行动1", "行动2"]
      }},
      "note": "供阶段三先验估计使用"
    }}
  }},
  "metadata": {{
    "version": "1.0",
    "created": "{today}",
    "last_updated": "{today}",
    "update_history_ref": "change_history/{id}.jsonl",
    "confidence": "high",
    "completeness": "medium",
    "needs_review": [],
    "total_experience_records": 0,
    "successful_applications": 0,
    "failed_applications": 0,
    "effectiveness_score": 0.5
  }}
}}
```

请直接输出 JSON，不要添加 markdown 代码块标记或其他文字。"""


# ---------------------------------------------------------------------------
# Build one strategy
# ---------------------------------------------------------------------------

def build_strategy(seed: dict, client, dry_run: bool = False) -> dict:
    """Expand a strategy seed into full schema using LLM API."""
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = EXPAND_STRATEGY_PROMPT.format(
        id=seed["id"],
        name_zh=seed["name_zh"],
        name_en=seed["name_en"],
        one_sentence=seed["one_sentence"],
        category=seed["category"],
        sources=", ".join(seed["sources"]),
        today=today,
    )

    if dry_run:
        print(f"[DRY RUN] {seed['id']}: {seed['name_zh']}")
        print(f"  Prompt length: {len(prompt)} chars")
        return None

    print(f"Building {seed['id']}: {seed['name_zh']}...", end=" ", flush=True)
    t0 = time.time()

    # Try with increasing max_tokens if truncated
    for max_tok in [12000, 16384]:
        response = client.generate(prompt, max_tokens=max_tok, temperature=0.3)
        try:
            strategy = parse_json_from_llm(response["text"])
            elapsed = time.time() - t0
            print(f"OK ({elapsed:.1f}s, {response['input_tokens']}+{response['output_tokens']} tokens)")
            return strategy
        except (json.JSONDecodeError, ValueError) as e:
            if max_tok < 16384:
                print(f"truncated ({response['output_tokens']} tokens), retrying with more...", end=" ", flush=True)
                time.sleep(1)
                continue
            else:
                print(f"FAILED (JSON parse error: {e})")
                err_path = STRATEGY_DIR / f"{seed['id']}_raw_error.txt"
                err_path.write_text(response["text"], encoding="utf-8")
                print(f"  Raw output saved to {err_path}")
                return None


# ---------------------------------------------------------------------------
# Build compositions
# ---------------------------------------------------------------------------

def build_compositions():
    """Write composition seed files (no API needed — static data)."""
    COMPOSITION_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    for comp in COMPOSITION_SEEDS:
        full = {
            "composition_id": comp["id"],
            "name": {
                "zh": comp["name_zh"],
                "en": comp["name_en"],
            },
            "sequence": comp["sequence"],
            "transition_condition": comp["transition_condition"],
            "applicable_when": "",  # to be filled by experience
            "source": "literature",
            "source_ref": "",
            "historical_cases": [],
            "metadata": {
                "effectiveness_score": 0.5,
                "total_uses": 0,
                "successful_uses": 0,
                "created": today,
            },
        }
        path = COMPOSITION_DIR / f"{comp['id']}.json"
        path.write_text(json.dumps(full, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Build categories and relationships
# ---------------------------------------------------------------------------

def build_categories():
    path = KB_DIR / "categories.json"
    path.write_text(
        json.dumps(CATEGORY_DEFINITIONS, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Write change history seed
# ---------------------------------------------------------------------------

def write_change_history(strategy_id: str):
    CHANGE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = CHANGE_HISTORY_DIR / f"{strategy_id}.jsonl"
    entry = {
        "change_id": "chg_001",
        "timestamp": datetime.now().isoformat(),
        "type": "initial_creation",
        "author": "human",
        "changes": "初始版本由 build_kb.py 生成",
        "evidence_refs": [],
        "previous_version": None,
        "new_version": 1,
    }
    path.write_text(json.dumps(entry, ensure_ascii=False) + "\n",
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Phase 0 KB")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Build only this strategy ID (e.g. S01)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling API")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip strategies that already have JSON files")
    args = parser.parse_args()

    STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        client = create_client()
    else:
        client = None

    # Filter seeds
    seeds = STRATEGY_SEEDS
    if args.strategy:
        seeds = [s for s in seeds if s["id"] == args.strategy]
        if not seeds:
            print(f"Strategy {args.strategy} not found in seeds")
            sys.exit(1)

    # Build strategies
    built = 0
    failed = 0
    for seed in seeds:
        out_path = STRATEGY_DIR / f"{seed['id']}_{seed['name_en'].lower().replace(' ', '_').replace('/', '_')}.json"

        if args.skip_existing and out_path.exists():
            print(f"Skipping {seed['id']} (already exists)")
            continue

        strategy = build_strategy(seed, client, dry_run=args.dry_run)

        if strategy is not None:
            out_path.write_text(
                json.dumps(strategy, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            write_change_history(seed["id"])
            built += 1
        else:
            if not args.dry_run:
                failed += 1

        # Rate limiting: 1 second between API calls
        if not args.dry_run and client:
            time.sleep(1)

    # Build compositions and categories (no API needed)
    build_compositions()
    build_categories()

    print(f"\nDone. Built: {built}, Failed: {failed}, Total seeds: {len(seeds)}")
    if failed > 0:
        print("Re-run with --skip-existing to retry failed strategies.")


if __name__ == "__main__":
    main()
