"""Compute and print a KB diff between pre-snapshot and current state."""

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import _config as cfg


def load_conditions(kb_dir: Path) -> dict:
    """Return {strategy_id: {placement: [conditions]}}."""
    out = {}
    for f in sorted(kb_dir.glob("S*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        sid = d["id"]
        out[sid] = {
            "favorable": d.get("applicability_conditions", {}).get("favorable", []),
            "unfavorable": d.get("applicability_conditions", {}).get("unfavorable", []),
        }
    return out


def main():
    pre_dir = cfg.PROJECT_ROOT / "analysis" / "kb_snapshot_pre"
    if not pre_dir.exists():
        print(f"No pre-snapshot at {pre_dir}")
        return

    pre = load_conditions(pre_dir)
    post = load_conditions(cfg.KB_DIR)

    added: list = []
    for sid in sorted(post.keys()):
        pre_ids = {c["condition_id"] for p in pre.get(sid, {}).values() for c in p}
        for placement, conds in post[sid].items():
            for c in conds:
                if c["condition_id"] not in pre_ids:
                    added.append((sid, placement, c))

    print(f"Total new conditions across KB: {len(added)}")
    print()

    by_strategy: dict = defaultdict(list)
    for sid, placement, c in added:
        by_strategy[sid].append((placement, c))

    for sid in sorted(by_strategy.keys()):
        print(f"=== {sid} ({len(by_strategy[sid])} new) ===")
        for placement, c in by_strategy[sid]:
            n_evidence = len(c.get("supporting_cases", []))
            print(f"  [{placement}] confidence={c['confidence']:.2f}, "
                  f"tier={c.get('stability_tier', '?')}, evidence={n_evidence}")
            print(f"      text: {c['condition']}")
            print(f"      id:   {c['condition_id']}")
        print()

    # Distribution summary
    from collections import Counter
    by_placement = Counter(p for _, p, _ in added)
    by_tier = Counter(c.get("stability_tier", "?") for _, _, c in added)
    print("Summary:")
    print(f"  by placement: {dict(by_placement)}")
    print(f"  by tier:      {dict(by_tier)}")
    print(f"  strategies affected: {len(by_strategy)}/27")

    out = cfg.PROJECT_ROOT / "analysis" / "kb_diff_report.json"
    out.write_text(json.dumps({
        "new_conditions_total": len(added),
        "by_strategy_count": {s: len(v) for s, v in by_strategy.items()},
        "by_placement": dict(by_placement),
        "by_tier": dict(by_tier),
        "additions": [
            {"strategy": s, "placement": p, **c}
            for s, p, c in added
        ],
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
