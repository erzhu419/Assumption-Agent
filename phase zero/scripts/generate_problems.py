"""
Phase 0: Generate benchmark problems for strategy-problem matching annotation.
Uses Claude API to generate 150-200 problems across 6 domains.

Usage:
    python generate_problems.py                        # Generate all
    python generate_problems.py --domain software      # One domain only
    python generate_problems.py --count 5 --domain math  # 5 math problems
    python generate_problems.py --dry-run
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from strategy_seeds import STRATEGY_SEEDS
from llm_client import create_client, parse_json_from_llm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROBLEMS_DIR = Path(__file__).parent.parent / "benchmark" / "problems"

DOMAINS = {
    "software_engineering": {
        "name": "软件工程",
        "types": ["调试", "系统设计", "性能优化", "架构选择", "代码重构", "部署故障排除"],
        "target_count": 30,
    },
    "mathematics": {
        "name": "数学/逻辑",
        "types": ["证明策略选择", "问题建模", "估算", "组合优化", "方程求解", "逻辑推理"],
        "target_count": 28,
    },
    "science": {
        "name": "科学实验",
        "types": ["实验设计", "假说检验", "结果解释", "数据异常分析", "因果推断"],
        "target_count": 28,
    },
    "business": {
        "name": "商业决策",
        "types": ["市场进入策略", "资源分配", "风险评估", "竞争分析", "产品迭代"],
        "target_count": 25,
    },
    "daily_life": {
        "name": "日常问题解决",
        "types": ["计划制定", "冲突解决", "学习策略", "搬家/装修", "时间管理"],
        "target_count": 25,
    },
    "engineering": {
        "name": "工程设计",
        "types": ["原型迭代", "故障排除", "需求分析", "系统集成", "安全评估"],
        "target_count": 28,
    },
}

# Strategy list for reference in prompt
STRATEGY_LIST = "\n".join(
    f"  {s['id']}: {s['name_zh']} — {s['one_sentence']}"
    for s in STRATEGY_SEEDS
)

GENERATE_PROBLEM_PROMPT = """你是一个跨学科问题设计专家。请生成 {count} 道用于测试方法论策略选择能力的问题。

## 要求
- 领域: {domain_name}
- 问题类型覆盖: {problem_types}
- 难度分布: 约 30% 简单、50% 中等、20% 复杂
- 每道问题 150-300 字描述
- 简单 = 最优策略几乎只有一个；中等 = 2-3 个合理选择；复杂 = 需要策略组合或存在策略冲突

## 可选策略列表（供参考答案使用）
{strategy_list}

## 特别要求
- 至少 1 道题的最优策略是 S21/S22/S23（元决策策略——问题无解或应该放弃的场景）
- 问题描述中不要提及任何策略名称——标注者需要自己判断
- 每道题要有足够的上下文让标注者能做出合理判断

## 输出格式
输出一个 JSON 数组，每个元素：
```json
{{
    "problem_id": "{domain_prefix}_001",
    "domain": "{domain_key}",
    "description": "问题描述（150-300字）",
    "difficulty": "easy|medium|hard",
    "reference_answer": {{
        "optimal_strategies": ["S_XX", "S_YY"],
        "acceptable_strategies": ["S_ZZ"],
        "explicitly_bad_strategies": ["S_WW"],
        "reasoning": "为什么这些策略最优（2-3句话）"
    }},
    "coverage_tags": ["S_XX", "S_YY"],
    "problem_type": "问题类型"
}}
```

请直接输出 JSON 数组，不要添加 markdown 代码块标记。"""




# ---------------------------------------------------------------------------
# Generate problems for one domain
# ---------------------------------------------------------------------------

def generate_domain_problems(
    domain_key: str,
    domain_info: dict,
    count: int,
    client,
    dry_run: bool = False,
) -> list:
    prompt = GENERATE_PROBLEM_PROMPT.format(
        count=count,
        domain_name=domain_info["name"],
        domain_key=domain_key,
        domain_prefix=domain_key[:2].upper(),
        problem_types=", ".join(domain_info["types"]),
        strategy_list=STRATEGY_LIST,
    )

    if dry_run:
        print(f"[DRY RUN] {domain_key}: {count} problems")
        return []

    print(f"Generating {count} problems for {domain_key}...", end=" ", flush=True)
    t0 = time.time()

    response = client.generate(prompt, max_tokens=8192, temperature=0.7)

    try:
        problems = parse_json_from_llm(response["text"])
    except (json.JSONDecodeError, ValueError) as e:
        print(f"FAILED (JSON parse: {e})")
        err_path = PROBLEMS_DIR / f"{domain_key}_raw_error.txt"
        err_path.write_text(response["text"], encoding="utf-8")
        return []

    elapsed = time.time() - t0
    print(f"OK ({len(problems)} problems, {elapsed:.1f}s, "
          f"{response['input_tokens']}+{response['output_tokens']} tokens)")

    for i, p in enumerate(problems):
        p["problem_id"] = f"{domain_key}_{i+1:03d}"

    return problems


# ---------------------------------------------------------------------------
# Coverage matrix check
# ---------------------------------------------------------------------------

def check_coverage_matrix(all_problems: list):
    """Check strategy × domain coverage after generation."""
    matrix = {}
    for p in all_problems:
        domain = p["domain"]
        for sid in p["reference_answer"]["optimal_strategies"]:
            key = (sid, domain)
            matrix[key] = matrix.get(key, 0) + 1

    all_strategies = sorted(set(s["id"] for s in STRATEGY_SEEDS))
    all_domains = sorted(DOMAINS.keys())

    print("\n=== Strategy × Domain Coverage Matrix ===")
    # Header
    header = f"{'':>6}" + "".join(f"{d[:6]:>8}" for d in all_domains) + "  Total"
    print(header)

    zero_cells = []
    for sid in all_strategies:
        row = f"{sid:>6}"
        total = 0
        for d in all_domains:
            count = matrix.get((sid, d), 0)
            total += count
            cell = f"{count:>8}"
            row += cell
            if count == 0:
                zero_cells.append((sid, d))
        row += f"  {total:>5}"
        print(row)

    print(f"\nTotal problems: {len(all_problems)}")
    print(f"Zero-coverage cells: {len(zero_cells)} / {len(all_strategies) * len(all_domains)}")
    if zero_cells:
        print("Uncovered (strategy, domain) pairs:")
        for sid, d in zero_cells[:20]:
            sname = next(s["name_zh"] for s in STRATEGY_SEEDS if s["id"] == sid)
            print(f"  {sid} ({sname}) × {d}")
        if len(zero_cells) > 20:
            print(f"  ... and {len(zero_cells) - 20} more")

    return zero_cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--count", type=int, default=None,
                        help="Override problem count per domain")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    PROBLEMS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        client = create_client()
    else:
        client = None

    domains = DOMAINS
    if args.domain:
        if args.domain not in domains:
            print(f"Unknown domain: {args.domain}")
            print(f"Available: {', '.join(domains.keys())}")
            sys.exit(1)
        domains = {args.domain: domains[args.domain]}

    all_problems = []

    # Load existing problems
    for f in PROBLEMS_DIR.glob("*.json"):
        if f.stem.endswith("_raw_error"):
            continue
        existing = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            all_problems.extend(existing)

    for domain_key, domain_info in domains.items():
        out_path = PROBLEMS_DIR / f"{domain_key}.json"

        if args.skip_existing and out_path.exists():
            print(f"Skipping {domain_key} (already exists)")
            continue

        count = args.count or domain_info["target_count"]
        problems = generate_domain_problems(
            domain_key, domain_info, count, client, args.dry_run
        )

        if problems:
            out_path.write_text(
                json.dumps(problems, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            all_problems.extend(problems)
            print(f"  Wrote {out_path}")

        if not args.dry_run:
            time.sleep(2)  # Rate limiting between domains

    # Coverage check
    if all_problems and not args.dry_run:
        check_coverage_matrix(all_problems)


if __name__ == "__main__":
    main()
