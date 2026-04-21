"""
Track A: domain-gated variant — combine existing cached answers by domain rule.

Practical domains (business/daily_life/sw_eng) + engineering → use orient_hybrid answer
Formal domains (mathematics/science) → use ours_27 answer

This requires ZERO new LLM calls. Output: new variant 'domain_gated' as cache.
"""

import json
from pathlib import Path

PROJECT = Path(__file__).parent.parent.parent
CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
SAMPLE = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))

orient_ans = json.loads((CACHE / "answers" / "orient_hybrid_answers.json").read_text(encoding="utf-8"))
technique_ans = json.loads((CACHE / "answers" / "ours_27_answers.json").read_text(encoding="utf-8"))

USE_ORIENT = {"business", "daily_life", "software_engineering", "engineering"}
USE_TECHNIQUE = {"mathematics", "science"}

combined = {}
stats = {"orient": 0, "technique": 0, "missing": 0}
for p in SAMPLE:
    pid = p["problem_id"]
    dom = p.get("domain", "")
    if dom in USE_ORIENT and pid in orient_ans:
        combined[pid] = orient_ans[pid]
        stats["orient"] += 1
    elif dom in USE_TECHNIQUE and pid in technique_ans:
        combined[pid] = technique_ans[pid]
        stats["technique"] += 1
    else:
        stats["missing"] += 1

out = CACHE / "answers" / "domain_gated_answers.json"
out.write_text(json.dumps(combined, ensure_ascii=False, indent=2))
print(f"domain_gated composition: {stats}")
print(f"Wrote {len(combined)} answers to {out.name}")
