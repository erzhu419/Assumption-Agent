"""
Phase 0: Data augmentation — expand 163 gold problems to ~1600.
For each seed problem, generate N variations that:
- Keep the same optimal strategy (label preserved)
- Change the domain/context/specifics (diversity)
- Vary difficulty slightly

Usage:
    python augment_problems.py                    # 10x augmentation (~1630)
    python augment_problems.py --multiplier 5     # 5x (~815)
    python augment_problems.py --domain software  # One domain only
    python augment_problems.py --dry-run
"""

import json
import sys
import time
import random
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import create_client, parse_json_from_llm

PROBLEMS_DIR = Path(__file__).parent.parent / "benchmark" / "problems"
OUT_DIR = Path(__file__).parent.parent / "benchmark" / "problems_augmented"

AUGMENT_PROMPT = """你是一个问题设计专家。基于以下种子问题，生成 {n} 个变体。

## 种子问题
领域: {domain}
描述: {description}
最优策略: {optimal}
理由: {reasoning}

## 变体要求
1. 每个变体的最优策略必须和种子问题**完全相同**: {optimal}
2. 变体必须改变具体场景（不同的人物/公司/技术/情境）
3. 至少 2 个变体要换到不同的子领域（但仍在 {domain} 大类下）
4. 描述长度 100-250 字
5. 不要简单地改几个词——要创造一个全新的具体场景

输出 JSON 数组（不要代码块标记），每个元素:
{{"description": "新问题描述", "difficulty": "easy/medium/hard"}}"""


def load_gold_problems():
    """Load all gold problems."""
    all_problems = {}
    for f in sorted(PROBLEMS_DIR.glob("*.json")):
        if "error" in f.name or "gemini" in f.name:
            continue
        domain = f.stem
        problems = json.loads(f.read_text(encoding="utf-8"))
        all_problems[domain] = problems
    return all_problems


def augment_one(seed, n, client):
    """Generate n variations of one seed problem."""
    prompt = AUGMENT_PROMPT.format(
        n=n,
        domain=seed["domain"],
        description=seed["description"],
        optimal=", ".join(seed["reference_answer"]["optimal_strategies"]),
        reasoning=seed["reference_answer"].get("reasoning", ""),
    )

    try:
        response = client.generate(prompt, max_tokens=4096, temperature=0.8)
        variants = parse_json_from_llm(response["text"])
        if not isinstance(variants, list):
            return []

        # Assign IDs and copy labels from seed
        result = []
        for i, v in enumerate(variants):
            if not isinstance(v, dict) or "description" not in v:
                continue
            result.append({
                "problem_id": f"{seed['problem_id']}_aug{i+1:02d}",
                "domain": seed["domain"],
                "description": v["description"],
                "difficulty": v.get("difficulty", seed.get("difficulty", "medium")),
                "reference_answer": seed["reference_answer"],  # Same label
                "coverage_tags": seed.get("coverage_tags", []),
                "augmented_from": seed["problem_id"],
            })
        return result
    except Exception as e:
        print(f"    Error: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiplier", type=int, default=10)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Variants per API call (lower = more calls but less truncation)")
    args = parser.parse_args()

    gold = load_gold_problems()
    total_seeds = sum(len(ps) for ps in gold.values())
    print(f"Gold problems: {total_seeds}")
    print(f"Target: ~{total_seeds * args.multiplier} augmented problems")

    if not args.dry_run:
        client = create_client()
    else:
        client = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grand_total = 0

    for domain, problems in gold.items():
        if args.domain and domain != args.domain:
            continue

        print(f"\n=== {domain} ({len(problems)} seeds) ===")
        all_augmented = list(problems)  # Start with originals

        for i, seed in enumerate(problems):
            if args.dry_run:
                print(f"  [{i+1}/{len(problems)}] {seed['problem_id']}: would generate {args.multiplier} variants")
                continue

            remaining = args.multiplier
            while remaining > 0:
                batch = min(args.batch_size, remaining)
                print(f"  [{i+1}/{len(problems)}] {seed['problem_id']}: generating {batch}...", end=" ", flush=True)
                variants = augment_one(seed, batch, client)
                if variants:
                    all_augmented.extend(variants)
                    remaining -= len(variants)
                    print(f"got {len(variants)}")
                else:
                    remaining -= batch  # Skip on failure
                    print("failed, skipping")
                time.sleep(0.5)

        # Renumber all IDs sequentially
        for j, p in enumerate(all_augmented):
            p["problem_id"] = f"{domain}_{j+1:04d}"

        # Save
        out_path = OUT_DIR / f"{domain}.json"
        out_path.write_text(json.dumps(all_augmented, ensure_ascii=False, indent=2),
                           encoding="utf-8")
        print(f"  Wrote {len(all_augmented)} problems → {out_path}")
        grand_total += len(all_augmented)

    print(f"\nTotal augmented: {grand_total}")

    # Coverage check
    if not args.dry_run:
        print("\n=== Augmented Coverage ===")
        strategy_counts = Counter()
        for domain_file in OUT_DIR.glob("*.json"):
            for p in json.loads(domain_file.read_text(encoding="utf-8")):
                for s in p.get("reference_answer", {}).get("optimal_strategies", []):
                    strategy_counts[s] += 1
        for sid in sorted(strategy_counts.keys()):
            print(f"  {sid}: {strategy_counts[sid]}")


if __name__ == "__main__":
    main()
