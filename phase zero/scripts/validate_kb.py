"""
Phase 0: Validate knowledge base JSON files against the expected schema.

Usage:
    python validate_kb.py          # Validate all strategies
    python validate_kb.py --fix    # Auto-fix missing optional fields
"""

import json
import sys
from pathlib import Path

KB_DIR = Path(__file__).parent.parent / "kb"
STRATEGY_DIR = KB_DIR / "strategies"

REQUIRED_TOP_FIELDS = [
    "id", "name", "description", "operational_steps",
    "applicability_conditions", "historical_cases",
    "relationships_to_other_strategies", "metadata"
]

REQUIRED_CONDITION_FIELDS = [
    "condition_id", "condition", "source", "confidence",
    "version", "status"
]


def validate_strategy(path: Path) -> list[str]:
    """Validate a single strategy JSON file. Returns list of errors."""
    errors = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    # Top-level fields
    for field in REQUIRED_TOP_FIELDS:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # ID format
    sid = data.get("id", "")
    if not sid.startswith("S") or not sid[1:].isdigit():
        errors.append(f"Invalid ID format: {sid} (expected S01-S23)")

    # Name
    name = data.get("name", {})
    if not name.get("zh") or not name.get("en"):
        errors.append("name must have both 'zh' and 'en'")

    # Operational steps (recursive format)
    steps = data.get("operational_steps", [])
    if len(steps) < 3:
        errors.append(f"operational_steps has only {len(steps)} steps (min 3)")
    for i, step in enumerate(steps):
        if isinstance(step, str):
            errors.append(f"operational_steps[{i}] is a plain string — "
                         "must be structured object with step/action/on_difficulty")
        elif isinstance(step, dict):
            if "action" not in step:
                errors.append(f"operational_steps[{i}] missing 'action'")

    # Applicability conditions
    ac = data.get("applicability_conditions", {})
    fav = ac.get("favorable", [])
    unfav = ac.get("unfavorable", [])
    if len(fav) < 2:
        errors.append(f"favorable conditions: {len(fav)} (min 2)")
    if len(unfav) < 1:
        errors.append(f"unfavorable conditions: {len(unfav)} (min 1)")
    for cond in fav + unfav:
        for field in REQUIRED_CONDITION_FIELDS:
            if field not in cond:
                errors.append(f"Condition {cond.get('condition_id', '?')} "
                             f"missing '{field}'")

    # Historical cases
    hc = data.get("historical_cases", {})
    succ = hc.get("successes", [])
    fail = hc.get("failures", [])
    if len(succ) < 2:
        errors.append(f"successes: {len(succ)} (min 2)")
    if len(fail) < 1:
        errors.append(f"failures: {len(fail)} (min 1)")

    # Knowledge triples
    kt = data.get("knowledge_triples", [])
    if len(kt) < 2:
        errors.append(f"knowledge_triples: {len(kt)} (min 2)")

    # Metadata
    meta = data.get("metadata", {})
    if "effectiveness_score" not in meta:
        errors.append("metadata missing 'effectiveness_score'")
    if "update_history_ref" not in meta:
        errors.append("metadata missing 'update_history_ref'")

    return errors


def main():
    if not STRATEGY_DIR.exists():
        print(f"Strategy directory not found: {STRATEGY_DIR}")
        print("Run build_kb.py first.")
        sys.exit(1)

    files = sorted(STRATEGY_DIR.glob("S*.json"))
    if not files:
        print("No strategy files found.")
        sys.exit(1)

    total_errors = 0
    for path in files:
        errors = validate_strategy(path)
        if errors:
            print(f"\n✗ {path.name} ({len(errors)} errors):")
            for e in errors:
                print(f"  - {e}")
            total_errors += len(errors)
        else:
            print(f"✓ {path.name}")

    print(f"\n{'='*40}")
    print(f"Files checked: {len(files)}")
    print(f"Total errors: {total_errors}")
    if total_errors == 0:
        print("All validations passed!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
