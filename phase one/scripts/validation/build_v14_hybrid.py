"""
v14-hybrid: domain-routed merge of v13-reflect (soft domains) + ours_27 (math/sci).

No regeneration needed — both answer sets already exist on 3-flash.
Just copy per-pid answer based on domain.

Routing:
  - math / science  -> ours_27 (Self-Discover JSON structure)
  - everything else -> v13-reflect (2-turn audit+revise)

Outputs: phase2_v14_hybrid_answers.json (100 entries, mixed source).
"""

import json
from pathlib import Path
from collections import Counter

CACHE = Path("/home/erzhu419/mine_code/Asumption Agent/phase two/analysis/cache")
V13_REFLECT = CACHE / "answers/phase2_v13_reflect_answers.json"
OURS_27 = CACHE / "answers/ours_27_answers.json"
SAMPLE = CACHE / "sample_100.json"
OUT = CACHE / "answers/phase2_v14_hybrid_answers.json"

MATH_SCI = {"mathematics", "science"}


def main():
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    v13 = json.loads(V13_REFLECT.read_text(encoding="utf-8"))
    ours = json.loads(OURS_27.read_text(encoding="utf-8"))

    hybrid = {}
    source_stats = Counter()
    missing = []
    for p in sample:
        pid = p["problem_id"]
        dom = p.get("domain", "?")
        if dom in MATH_SCI:
            if pid in ours:
                hybrid[pid] = ours[pid]
                source_stats["ours_27"] += 1
            else:
                missing.append(pid)
        else:
            if pid in v13:
                hybrid[pid] = v13[pid]
                source_stats["v13_reflect"] += 1
            else:
                missing.append(pid)

    OUT.write_text(json.dumps(hybrid, ensure_ascii=False, indent=2))
    print(f"Built v14-hybrid: {len(hybrid)} answers")
    print(f"  source breakdown: {dict(source_stats)}")
    if missing:
        print(f"  MISSING {len(missing)}: {missing[:5]}...")

    # Length audit
    lens = [len(a) for a in hybrid.values()]
    print(f"\n  avg length: {sum(lens)/len(lens):.0f}")
    by_src_lens = {
        "ours_27_math_sci": [len(hybrid[p["problem_id"]])
                              for p in sample if p.get("domain") in MATH_SCI
                              and p["problem_id"] in hybrid],
        "v13_reflect_other": [len(hybrid[p["problem_id"]])
                               for p in sample if p.get("domain") not in MATH_SCI
                               and p["problem_id"] in hybrid],
    }
    for k, v in by_src_lens.items():
        if v:
            print(f"    {k}: avg={sum(v)/len(v):.0f} (n={len(v)})")


if __name__ == "__main__":
    main()
