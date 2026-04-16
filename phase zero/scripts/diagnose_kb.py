"""
Phase 0: Systematic quality diagnosis of generated knowledge base.
Checks for issues that automated validation (validate_kb.py) doesn't catch.

Usage:
    python diagnose_kb.py
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from strategy_seeds import STRATEGY_SEEDS

KB_DIR = Path(__file__).parent.parent / "kb"
STRATEGY_DIR = KB_DIR / "strategies"
PROBLEMS_DIR = Path(__file__).parent.parent / "benchmark" / "problems"
ANNOTATIONS_DIR = Path(__file__).parent.parent / "benchmark" / "annotations"

VALID_STRATEGY_IDS = {s["id"] for s in STRATEGY_SEEDS}
VALID_COMP_IDS = {"COMP_001", "COMP_002", "COMP_003", "COMP_004", "COMP_005"}

# =========================================================================
# Diagnosis functions
# =========================================================================

def diagnose_strategy_references():
    """Check if on_difficulty and relationship fields reference valid strategy IDs."""
    print("\n=== 1. 策略交叉引用检查 ===")
    issues = []

    for path in sorted(STRATEGY_DIR.glob("S*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        sid = d["id"]

        # Check on_difficulty references
        for step in d.get("operational_steps", []):
            if not isinstance(step, dict):
                continue
            od = step.get("on_difficulty")
            if od and isinstance(od, str):
                # Extract S_XX references
                import re
                refs = re.findall(r'\bS[_]?\d{2}\b', od)
                for ref in refs:
                    normalized = ref.replace("_", "")
                    if normalized not in VALID_STRATEGY_IDS:
                        issues.append((sid, f"步骤{step.get('step','?')}",
                                      f"on_difficulty 引用不存在的策略 '{ref}'"))

        # Check relationship references
        for rel in d.get("relationships_to_other_strategies", []):
            ref = rel.get("related_strategy", "")
            if ref and ref not in VALID_STRATEGY_IDS and ref not in VALID_COMP_IDS:
                issues.append((sid, "relationships",
                              f"关系引用不存在的策略 '{ref}'"))

    if issues:
        print(f"  发现 {len(issues)} 个无效引用:")
        for sid, location, msg in issues[:20]:
            print(f"    {sid} [{location}]: {msg}")
        if len(issues) > 20:
            print(f"    ... 还有 {len(issues)-20} 个")
    else:
        print("  ✓ 所有交叉引用有效")

    return issues


def diagnose_case_domain_diversity():
    """Check if historical cases span multiple domains."""
    print("\n=== 2. 案例领域多样性检查 ===")
    issues = []

    for path in sorted(STRATEGY_DIR.glob("S*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        sid = d["id"]

        success_domains = set()
        for case in d.get("historical_cases", {}).get("successes", []):
            success_domains.add(case.get("domain", "unknown"))

        if len(success_domains) < 2:
            issues.append(f"{sid}: 成功案例仅覆盖 {len(success_domains)} 个领域 "
                         f"({success_domains})，要求 ≥ 2")

    if issues:
        print(f"  发现 {len(issues)} 个领域覆盖不足:")
        for msg in issues:
            print(f"    {msg}")
    else:
        print("  ✓ 所有策略的成功案例覆盖 ≥ 2 个领域")

    return issues


def diagnose_condition_quality():
    """Check condition confidence and stability_tier consistency."""
    print("\n=== 3. 适用条件质量检查 ===")
    issues = []

    for path in sorted(STRATEGY_DIR.glob("S*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        sid = d["id"]
        ac = d.get("applicability_conditions", {})

        for placement in ["favorable", "unfavorable"]:
            for cond in ac.get(placement, []):
                # Check confidence range
                conf = cond.get("confidence", 0)
                if conf < 0.5 or conf > 1.0:
                    issues.append(f"{sid} {cond.get('condition_id','?')}: "
                                 f"confidence={conf} (应在 0.5-1.0)")

                # Check stability_tier
                tier = cond.get("stability_tier", "MISSING")
                if tier not in ("foundational", "empirical", "tentative", "MISSING"):
                    issues.append(f"{sid} {cond.get('condition_id','?')}: "
                                 f"无效的 stability_tier='{tier}'")

                # Check condition text is not too short
                text = cond.get("condition", "")
                if len(text) < 10:
                    issues.append(f"{sid} {cond.get('condition_id','?')}: "
                                 f"条件描述过短 ({len(text)} 字)")

    if issues:
        print(f"  发现 {len(issues)} 个问题:")
        for msg in issues[:15]:
            print(f"    {msg}")
        if len(issues) > 15:
            print(f"    ... 还有 {len(issues)-15} 个")
    else:
        print("  ✓ 所有条件质量合格")

    return issues


def diagnose_knowledge_triples():
    """Check knowledge triple quality."""
    print("\n=== 4. 知识三元组检查 ===")
    issues = []

    for path in sorted(STRATEGY_DIR.glob("S*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        sid = d["id"]
        triples = d.get("knowledge_triples", [])

        if len(triples) < 2:
            issues.append(f"{sid}: 仅 {len(triples)} 条三元组 (要求 ≥ 2)")
            continue

        for i, t in enumerate(triples):
            if not all(k in t for k in ("subject", "relation", "object")):
                issues.append(f"{sid} triple[{i}]: 缺少 subject/relation/object")
            elif any(len(t.get(k, "")) < 2 for k in ("subject", "relation", "object")):
                issues.append(f"{sid} triple[{i}]: 某字段过短")

    if issues:
        print(f"  发现 {len(issues)} 个问题:")
        for msg in issues:
            print(f"    {msg}")
    else:
        print("  ✓ 所有知识三元组合格")

    return issues


def diagnose_annotation_agreement_by_domain():
    """Check annotation agreement broken down by domain and difficulty."""
    print("\n=== 5. 标注一致性分领域/难度分析 ===")

    domain_stats = defaultdict(lambda: {"total": 0, "top1_agree": 0})
    difficulty_stats = defaultdict(lambda: {"total": 0, "top1_agree": 0})

    for path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        ann = json.loads(path.read_text(encoding="utf-8"))
        domain = ann.get("domain", "unknown")
        difficulty = ann.get("difficulty", "unknown")
        annotations = ann.get("annotations", [])

        if len(annotations) < 2:
            continue

        # Check if all annotators agree on top-1
        top1s = [a["selected_strategies"][0] for a in annotations
                 if a.get("selected_strategies")]
        if not top1s:
            continue

        most_common = Counter(top1s).most_common(1)[0]
        agreement_ratio = most_common[1] / len(top1s)

        domain_stats[domain]["total"] += 1
        difficulty_stats[difficulty]["total"] += 1

        if agreement_ratio >= 0.6:  # Majority agrees
            domain_stats[domain]["top1_agree"] += 1
            difficulty_stats[difficulty]["top1_agree"] += 1

    print("  按领域:")
    for domain, stats in sorted(domain_stats.items()):
        rate = stats["top1_agree"] / stats["total"] if stats["total"] > 0 else 0
        print(f"    {domain:25s}: {rate:.0%} 多数一致 ({stats['top1_agree']}/{stats['total']})")

    print("  按难度:")
    for diff, stats in sorted(difficulty_stats.items()):
        rate = stats["top1_agree"] / stats["total"] if stats["total"] > 0 else 0
        print(f"    {diff:10s}: {rate:.0%} 多数一致 ({stats['top1_agree']}/{stats['total']})")


def diagnose_annotation_vs_reference():
    """Check where annotations systematically disagree with reference answers."""
    print("\n=== 6. 标注 vs 参考答案的系统性偏差 ===")

    strategy_confusion = defaultdict(lambda: defaultdict(int))
    # strategy_confusion[reference][annotator_choice] = count

    for path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        ann = json.loads(path.read_text(encoding="utf-8"))
        ref = ann.get("reference_answer", {})
        ref_strategies = set(ref.get("optimal_strategies", []))

        if not ref_strategies:
            continue

        for a in ann.get("annotations", []):
            top1 = a["selected_strategies"][0] if a.get("selected_strategies") else None
            if top1 and top1 not in ref_strategies:
                ref_str = ",".join(sorted(ref_strategies))
                strategy_confusion[ref_str][top1] += 1

    # Find most common disagreements
    disagreements = []
    for ref_str, choices in strategy_confusion.items():
        for choice, count in choices.items():
            disagreements.append((count, ref_str, choice))

    disagreements.sort(reverse=True)

    if disagreements:
        print(f"  Top-10 标注偏离参考答案的模式:")
        for count, ref, choice in disagreements[:10]:
            print(f"    参考={ref} → 标注选了 {choice} ({count} 次)")
    else:
        print("  ✓ 无系统性偏差")


def diagnose_problem_strategy_coverage():
    """Check the strategy×domain coverage matrix gaps."""
    print("\n=== 7. 策略×领域覆盖矩阵 ===")

    coverage = defaultdict(lambda: defaultdict(int))

    for path in sorted(PROBLEMS_DIR.glob("*.json")):
        if "error" in path.name:
            continue
        problems = json.loads(path.read_text(encoding="utf-8"))
        for p in problems:
            domain = p.get("domain", "unknown")
            for sid in p.get("reference_answer", {}).get("optimal_strategies", []):
                coverage[sid][domain] += 1

    all_domains = sorted(set(d for s in coverage.values() for d in s.keys()))
    zero_cells = 0
    total_cells = len(VALID_STRATEGY_IDS) * len(all_domains)

    low_coverage = []
    for sid in sorted(VALID_STRATEGY_IDS):
        domains_covered = sum(1 for d in all_domains if coverage[sid].get(d, 0) > 0)
        total = sum(coverage[sid].values())
        if domains_covered < 3:
            low_coverage.append(f"{sid}: 仅覆盖 {domains_covered} 个领域 (总 {total} 题)")
        for d in all_domains:
            if coverage[sid].get(d, 0) == 0:
                zero_cells += 1

    print(f"  零覆盖格子: {zero_cells}/{total_cells} ({zero_cells/total_cells:.0%})")
    if low_coverage:
        print(f"  覆盖不足的策略 (< 3 个领域):")
        for msg in low_coverage:
            print(f"    {msg}")


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("Phase 0 知识库质量诊断报告")
    print("=" * 60)

    all_issues = []

    all_issues.extend(diagnose_strategy_references())
    all_issues.extend(diagnose_case_domain_diversity())
    all_issues.extend(diagnose_condition_quality())
    all_issues.extend(diagnose_knowledge_triples())
    diagnose_annotation_agreement_by_domain()
    diagnose_annotation_vs_reference()
    diagnose_problem_strategy_coverage()

    print("\n" + "=" * 60)
    print(f"总计发现 {len(all_issues)} 个需要关注的问题")
    if all_issues:
        print("建议: 优先修复策略交叉引用问题（影响 Phase 1 的执行流程）")
    else:
        print("Phase 0 知识库质量良好！")


if __name__ == "__main__":
    main()
