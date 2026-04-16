"""
Phase 0: Annotate benchmark problems with optimal strategy selections.
Uses multiple independent Claude API calls to simulate inter-annotator agreement.

Each problem gets N independent annotations (default N=5).
Each annotation uses a different system prompt persona to encourage diversity.
Agreement is measured via Fleiss' Kappa.

Usage:
    python annotate_problems.py                  # Annotate all problems
    python annotate_problems.py --annotators 3   # Use 3 annotators
    python annotate_problems.py --domain software
    python annotate_problems.py --dry-run
"""

import json
import os
import sys
import time
import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from strategy_seeds import STRATEGY_SEEDS
from llm_client import create_client, parse_json_from_llm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROBLEMS_DIR = Path(__file__).parent.parent / "benchmark" / "problems"
ANNOTATIONS_DIR = Path(__file__).parent.parent / "benchmark" / "annotations"
ANALYSIS_DIR = Path(__file__).parent.parent / "benchmark" / "analysis"

# ---------------------------------------------------------------------------
# Annotator personas (diversity = better inter-annotator agreement analysis)
# ---------------------------------------------------------------------------

ANNOTATOR_PERSONAS = [
    {
        "id": "annotator_cs",
        "background": "你是一位有 15 年经验的计算机科学研究者，擅长算法设计和系统调试。",
        "bias_note": "你可能偏好分而治之和增量构建等工程化策略。"
    },
    {
        "id": "annotator_math",
        "background": "你是一位数学家，研究领域是数理逻辑和组合数学。",
        "bias_note": "你可能偏好反证法、类比推理等演绎策略。"
    },
    {
        "id": "annotator_science",
        "background": "你是一位实验物理学家，在粒子物理实验室工作了 10 年。",
        "bias_note": "你可能偏好控制变量法、证伪优先等实证策略。"
    },
    {
        "id": "annotator_business",
        "background": "你是一位连续创业者，成功创办过 3 家公司。",
        "bias_note": "你可能偏好满意化、贝叶斯更新等决策策略。"
    },
    {
        "id": "annotator_philosophy",
        "background": "你是一位科学哲学研究者，专注于方法论和认识论。",
        "bias_note": "你可能从更抽象的层面看待问题，偏好元决策策略。"
    },
    {
        "id": "annotator_engineer",
        "background": "你是一位有 20 年经验的系统工程师，擅长大型系统的故障排除。",
        "bias_note": "你可能偏好边界条件分析和增量构建等实践策略。"
    },
]

STRATEGY_LIST = "\n".join(
    f"  {s['id']}: {s['name_zh']} — {s['one_sentence']}"
    for s in STRATEGY_SEEDS
)

ANNOTATION_PROMPT = """## 你的背景
{background}

## 任务
阅读以下问题，从策略列表中选择最适合解决这个问题的 1-3 条策略。

## 问题描述
{problem_description}

## 可选策略列表
{strategy_list}

## 注意
{bias_note}
请尽量克服你的专业偏好，从问题本身出发选择最合适的策略。
如果你认为这个问题应该被放弃或重新定义（而非直接解决），请选择 S21/S22/S23。

## 输出格式
请输出 JSON（不要添加代码块标记）：
{{
    "selected_strategies": ["S_XX", "S_YY"],
    "confidence": "high|medium|low",
    "reasoning": "选择理由（1-2句话）"
}}"""


# ---------------------------------------------------------------------------
# Annotate one problem with one persona
# ---------------------------------------------------------------------------

def annotate_single(
    problem: dict,
    persona: dict,
    client,
) -> dict | None:
    prompt = ANNOTATION_PROMPT.format(
        background=persona["background"],
        problem_description=problem["description"],
        strategy_list=STRATEGY_LIST,
        bias_note=persona["bias_note"],
    )

    try:
        response = client.generate(prompt, max_tokens=512, temperature=0.5)
        result = parse_json_from_llm(response["text"])
        result["annotator_id"] = persona["id"]
        result["problem_id"] = problem["problem_id"]
        return result
    except Exception as e:
        print(f"    Error ({persona['id']}): {e}")
        return None


# ---------------------------------------------------------------------------
# Compute Fleiss' Kappa
# ---------------------------------------------------------------------------

def fleiss_kappa(annotations_by_problem: dict, all_strategies: list) -> float:
    """
    Compute Fleiss' Kappa for multi-annotator agreement.
    annotations_by_problem: {problem_id: [list of selected top-1 strategy IDs]}
    """
    n_subjects = len(annotations_by_problem)
    n_categories = len(all_strategies)
    strategy_to_idx = {s: i for i, s in enumerate(all_strategies)}

    # Build the n_subjects × n_categories matrix
    matrix = []
    n_raters = None
    for pid, annotations in annotations_by_problem.items():
        if n_raters is None:
            n_raters = len(annotations)
        row = [0] * n_categories
        for a in annotations:
            if a in strategy_to_idx:
                row[strategy_to_idx[a]] += 1
        matrix.append(row)

    if not matrix or n_raters is None or n_raters < 2:
        return 0.0

    N = n_subjects
    n = n_raters
    k = n_categories

    # P_i for each subject
    P_i = []
    for row in matrix:
        s = sum(r * (r - 1) for r in row)
        P_i.append(s / (n * (n - 1)) if n > 1 else 0)

    P_bar = sum(P_i) / N if N > 0 else 0

    # P_e
    p_j = []
    for j in range(k):
        col_sum = sum(matrix[i][j] for i in range(N))
        p_j.append(col_sum / (N * n))
    P_e = sum(p ** 2 for p in p_j)

    if P_e >= 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e) if (1 - P_e) != 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotators", type=int, default=5,
                        help="Number of independent annotators per problem")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load problems
    all_problems = []
    for f in sorted(PROBLEMS_DIR.glob("*.json")):
        if f.stem.endswith("_raw_error"):
            continue
        problems = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(problems, list):
            all_problems.extend(problems)

    if args.domain:
        all_problems = [p for p in all_problems if p["domain"] == args.domain]

    if not all_problems:
        print("No problems found. Run generate_problems.py first.")
        sys.exit(1)

    print(f"Found {len(all_problems)} problems to annotate")
    print(f"Using {args.annotators} annotators per problem")

    # Select annotator personas
    personas = ANNOTATOR_PERSONAS[:args.annotators]
    if args.annotators > len(ANNOTATOR_PERSONAS):
        print(f"Warning: only {len(ANNOTATOR_PERSONAS)} personas available")
        personas = ANNOTATOR_PERSONAS

    if not args.dry_run:
        client = create_client()
        total_calls = len(all_problems) * len(personas)
        print(f"Estimated: {total_calls} API calls")
    else:
        client = None

    # Annotate
    all_annotations = {}
    for i, problem in enumerate(all_problems):
        pid = problem["problem_id"]

        # Check if already annotated
        ann_path = ANNOTATIONS_DIR / f"{pid}.json"
        if args.skip_existing and ann_path.exists():
            existing = json.loads(ann_path.read_text(encoding="utf-8"))
            all_annotations[pid] = existing
            continue

        if args.dry_run:
            print(f"[DRY RUN] {pid}: {problem['description'][:60]}...")
            continue

        print(f"[{i+1}/{len(all_problems)}] Annotating {pid}...", flush=True)
        annotations = []
        for persona in personas:
            result = annotate_single(problem, persona, client)
            if result:
                annotations.append(result)
            time.sleep(0.5)  # Rate limiting

        if annotations:
            record = {
                "problem_id": pid,
                "domain": problem["domain"],
                "difficulty": problem["difficulty"],
                "annotations": annotations,
                "reference_answer": problem.get("reference_answer"),
            }
            ann_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            all_annotations[pid] = record
            print(f"  {len(annotations)} annotations saved")

    # Compute agreement
    if not args.dry_run and all_annotations:
        print("\n=== Inter-Annotator Agreement ===")
        all_strategy_ids = sorted(s["id"] for s in STRATEGY_SEEDS)

        # Extract top-1 strategy per annotator per problem
        annotations_by_problem = {}
        for pid, record in all_annotations.items():
            top1s = []
            for ann in record["annotations"]:
                if ann["selected_strategies"]:
                    top1s.append(ann["selected_strategies"][0])
            if top1s:
                annotations_by_problem[pid] = top1s

        kappa = fleiss_kappa(annotations_by_problem, all_strategy_ids)
        print(f"Fleiss' Kappa (top-1): {kappa:.3f}")
        if kappa >= 0.6:
            print("  ✓ Substantial agreement (target: ≥ 0.5)")
        elif kappa >= 0.4:
            print("  ~ Moderate agreement")
        else:
            print("  ✗ Poor agreement — strategy definitions may need refinement")

        # Top-3 hit rate vs reference
        hits = 0
        total = 0
        for pid, record in all_annotations.items():
            ref = record.get("reference_answer", {})
            ref_strategies = set(
                ref.get("optimal_strategies", []) +
                ref.get("acceptable_strategies", [])
            )
            if not ref_strategies:
                continue
            for ann in record["annotations"]:
                selected = set(ann["selected_strategies"][:3])
                if selected & ref_strategies:
                    hits += 1
                total += 1

        if total > 0:
            hit_rate = hits / total
            print(f"Top-3 hit rate vs reference: {hit_rate:.1%} (target: ≥ 65%)")

        # Save analysis
        analysis = {
            "fleiss_kappa": kappa,
            "top3_hit_rate": hits / total if total > 0 else None,
            "n_problems": len(annotations_by_problem),
            "n_annotators": len(personas),
        }
        analysis_path = ANALYSIS_DIR / "agreement_report.json"
        analysis_path.write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nAnalysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
