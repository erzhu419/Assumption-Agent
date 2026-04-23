"""Re-run wisdom selection with full v18 library (79 entries including W076-79).

Uses v2 SELECT_PROMPT (aggressive paradigm push) since that's what works.
"""

import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase one" / "scripts" / "validation"))
sys.path.insert(0, str(PROJECT / "phase four"))

from llm_client import create_client, parse_json_from_llm
from cached_framework import _generate_with_retry
from run_selection_v2 import SELECT_PROMPT_V2, build_brief_library, select_entries


CACHE = PROJECT / "phase two" / "analysis" / "cache"
WISDOM_V18 = CACHE / "wisdom_library_v18.json"
SAMPLE = CACHE / "sample_100.json"
OUT_PATH = CACHE / "phase2_v3_selections_v18.json"


def main():
    library = json.loads(WISDOM_V18.read_text(encoding="utf-8"))
    print(f"Library: {len(library)} wisdoms (incl W076-79)")

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    out = {}
    if OUT_PATH.exists():
        try:
            out = json.loads(OUT_PATH.read_text(encoding="utf-8"))
            print(f"Resuming: {len(out)} cached")
        except Exception:
            out = {}

    client = create_client()
    t0 = time.time()
    new_count = 0
    errors = 0

    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in out:
            continue
        try:
            sel = select_entries(client, p.get("description", ""), library)
            out[pid] = sel
            new_count += 1
        except Exception as e:
            print(f"  [err {pid}] {e}")
            errors += 1

        if new_count % 10 == 0:
            OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
            print(f"  [{pid}] {new_count}/{len(sample)} done, {time.time()-t0:.0f}s")

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nDone: {len(out)}/{len(sample)}, errors={errors}, {time.time()-t0:.0f}s")

    # Count how often new W076-79 get picked
    from collections import Counter
    picks = Counter()
    for ids in out.values():
        for wid in ids[:5]:
            picks[wid] += 1
    print("\nNew wisdom activation (top-5):")
    for wid in ["W076", "W077", "W078", "W079"]:
        print(f"  {wid}: {picks.get(wid, 0)} / {len(out)} problems")


if __name__ == "__main__":
    main()
