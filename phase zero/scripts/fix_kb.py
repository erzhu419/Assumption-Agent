"""
Phase 0: Fix all diagnosed issues in the knowledge base.
1. Fix invalid strategy ID references (99 issues)
2. Fix invalid stability_tier values (4 issues)
3. Fix short knowledge triple fields (3 issues)

Usage:
    python fix_kb.py          # Fix all issues
    python fix_kb.py --dry-run  # Show what would be fixed
"""

import json
import re
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from strategy_seeds import STRATEGY_SEEDS

KB_DIR = Path(__file__).parent.parent / "kb"
STRATEGY_DIR = KB_DIR / "strategies"

VALID_IDS = {s["id"] for s in STRATEGY_SEEDS}

# =========================================================================
# Mapping table: invalid ID patterns → correct S01-S27 IDs
# =========================================================================

# Based on semantic meaning of the invalid references
REFERENCE_FIX_MAP = {
    # Exact matches
    "S_EXP_DESIGN": "S01",       # experimental design ≈ controlled variable
    "S_OBS": "S16",              # observation ≈ method of agreement
    "S_01": "S01",               # underscore variant
    "S_02": "S02",
    "S_03": "S03",
    "S_04": "S04",
    "S_05": "S05",
    "S_06": "S06",
    "S_07": "S07",
    "S_08": "S08",
    "S_09": "S09",
    "S_10": "S10",
    "S_11": "S11",
    "S_12": "S12",
    "S_13": "S13",
    "S_14": "S14",
    "S_15": "S15",
    "S01_ConceptClarification": "S18",  # concept clarification ≈ abstraction
}

# Pattern-based fixes for relationship references
RELATIONSHIP_KEYWORD_MAP = {
    "递归": "S02",           # recursion → divide and conquer
    "动态规划": "S02",       # dynamic programming → divide and conquer
    "抽象": "S18",           # abstraction
    "自底向上": "S15",       # bottom-up → incremental building
    "问题分解": "S02",       # problem decomposition → divide and conquer
    "DirectProof": "S04",    # direct proof → proof methods (close to S04)
    "ConstructiveProof": "S15",  # constructive proof → incremental
    "ModusTollens": "S04",   # modus tollens → proof by contradiction
    "假设生成": "S08",       # hypothesis generation → trial and error
    "溯因推理": "S03",       # abductive reasoning → analogical reasoning
    "可证伪性": "S13",       # falsifiability → falsification first
    "模型选择": "S05",       # model selection → Occam's razor
    "AIC": "S05",
    "BIC": "S05",
    "正向推理": "S07",       # forward reasoning (alternative to backward)
    "手段-目的": "S07",      # means-ends analysis → backward reasoning
    "规划": "S15",           # planning → incremental building
    "归纳": "S06",           # induction → specialization before generalization
    "演绎": "S04",           # deduction → proof by contradiction
    "统计": "S12",           # statistics → Bayesian updating
    "系统思维": "S25",       # systems thinking → emergence detection
    "反馈": "S12",           # feedback → Bayesian updating
    "博弈": "S27",           # game theory → incentive structure
    "激励": "S27",           # incentive → incentive structure
    "路径": "S26",           # path → path dependency
    "瓶颈": "S24",           # bottleneck → critical node
    "网络": "S24",           # network → critical node
}

# Valid stability tiers
VALID_TIERS = {"foundational", "empirical", "tentative"}
TIER_FIX_MAP = {
    "contextual": "empirical",
    "practical": "empirical",
    "derived": "empirical",
    "theoretical": "foundational",
}


def fix_relationship_ref(ref: str) -> str:
    """Fix an invalid strategy reference in relationships."""
    # Direct mapping
    if ref in REFERENCE_FIX_MAP:
        return REFERENCE_FIX_MAP[ref]

    # Already valid
    if ref in VALID_IDS:
        return ref

    # Keyword-based mapping
    for keyword, target in RELATIONSHIP_KEYWORD_MAP.items():
        if keyword in ref:
            return target

    # S_XX pattern → try to extract number
    m = re.match(r'S[_]?(\d{2})', ref)
    if m:
        candidate = f"S{m.group(1)}"
        if candidate in VALID_IDS:
            return candidate

    # Default: remove (will be filtered out)
    return None


def fix_on_difficulty(text: str) -> str:
    """Fix strategy references inside on_difficulty text."""
    if not text or not isinstance(text, str):
        return text

    def replace_ref(match):
        ref = match.group(0)
        # Normalize S_XX to SXX
        normalized = ref.replace("S_", "S")
        if normalized in VALID_IDS:
            return normalized
        # Try fix map
        fixed = REFERENCE_FIX_MAP.get(ref)
        if fixed:
            return fixed
        # Return as-is if can't fix (it's embedded in text, not critical)
        return ref

    # Fix S_XX patterns
    fixed = re.sub(r'\bS[_]?\d{2}\b', replace_ref, text)
    return fixed


def fix_strategy_file(path: Path, dry_run: bool = False) -> int:
    """Fix all issues in one strategy file. Returns number of fixes."""
    d = json.loads(path.read_text(encoding="utf-8"))
    sid = d["id"]
    fixes = 0

    # --- Fix 1: Relationship references ---
    new_rels = []
    for rel in d.get("relationships_to_other_strategies", []):
        ref = rel.get("related_strategy", "")
        if ref in VALID_IDS:
            new_rels.append(rel)
            continue

        fixed_ref = fix_relationship_ref(ref)
        if fixed_ref and fixed_ref != sid:  # Don't self-reference
            old_ref = ref
            rel["related_strategy"] = fixed_ref
            new_rels.append(rel)
            fixes += 1
            if dry_run:
                print(f"  {sid} relationship: '{old_ref}' → '{fixed_ref}'")
        else:
            fixes += 1  # Removing invalid reference
            if dry_run:
                print(f"  {sid} relationship: removing invalid '{ref}'")

    # Deduplicate relationships by target
    seen_targets = set()
    deduped_rels = []
    for rel in new_rels:
        target = rel["related_strategy"]
        if target not in seen_targets:
            seen_targets.add(target)
            deduped_rels.append(rel)

    d["relationships_to_other_strategies"] = deduped_rels

    # --- Fix 2: on_difficulty references ---
    for step in d.get("operational_steps", []):
        if isinstance(step, dict) and step.get("on_difficulty"):
            old = step["on_difficulty"]
            fixed = fix_on_difficulty(old)
            if fixed != old:
                step["on_difficulty"] = fixed
                fixes += 1

    # --- Fix 3: stability_tier ---
    for placement in ["favorable", "unfavorable"]:
        for cond in d.get("applicability_conditions", {}).get(placement, []):
            tier = cond.get("stability_tier", "")
            if tier and tier not in VALID_TIERS:
                new_tier = TIER_FIX_MAP.get(tier, "empirical")
                if dry_run:
                    print(f"  {sid} {cond.get('condition_id','?')}: "
                          f"stability_tier '{tier}' → '{new_tier}'")
                cond["stability_tier"] = new_tier
                fixes += 1
            elif not tier:
                cond["stability_tier"] = "foundational"
                fixes += 1

    # --- Fix 4: Short knowledge triple fields ---
    for i, t in enumerate(d.get("knowledge_triples", [])):
        for field in ("subject", "relation", "object"):
            val = t.get(field, "")
            if len(val) < 2:
                # Pad with strategy name context
                if field == "subject":
                    t[field] = d["name"]["zh"]
                elif field == "relation":
                    t[field] = "相关于"
                elif field == "object":
                    t[field] = d["description"]["one_sentence"][:20]
                fixes += 1
                if dry_run:
                    print(f"  {sid} triple[{i}].{field}: padded from '{val}'")

    if not dry_run and fixes > 0:
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    return fixes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total_fixes = 0
    files_fixed = 0

    for path in sorted(STRATEGY_DIR.glob("S*.json")):
        fixes = fix_strategy_file(path, dry_run=args.dry_run)
        if fixes > 0:
            total_fixes += fixes
            files_fixed += 1
            if not args.dry_run:
                print(f"  Fixed {path.name}: {fixes} fixes")

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Total: {total_fixes} fixes in {files_fixed} files")

    if not args.dry_run and total_fixes > 0:
        print("\nRe-running diagnose to verify...")
        import subprocess
        subprocess.run([sys.executable, "diagnose_kb.py"], cwd=str(Path(__file__).parent))


if __name__ == "__main__":
    main()
