"""Mine diverse exemplars for W076-W079 (Mode B new wisdoms), merge into existing file."""

import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase one" / "scripts" / "validation"))

from llm_client import parse_json_from_llm
from gpt5_client import GPT5Client
# Reuse the same prompt template + extraction logic
from build_diverse_exemplars_v15 import SELECT_PROMPT


CACHE = PROJECT / "phase two" / "analysis" / "cache"
WISDOM_V18 = CACHE / "wisdom_library_v18.json"
SAMPLE = CACHE / "sample_100.json"
V13_REFLECT = CACHE / "answers/phase2_v13_reflect_answers.json"
OURS_27 = CACHE / "answers/ours_27_answers.json"
EXEMPLARS_OUT = CACHE / "wisdom_diverse_exemplars.json"

MATH_SCI = {"mathematics", "science"}
TARGET_IDS = {"W076", "W077", "W078", "W079"}


def main():
    wisdom = json.loads(WISDOM_V18.read_text(encoding="utf-8"))
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    v13 = json.loads(V13_REFLECT.read_text(encoding="utf-8"))
    ours = json.loads(OURS_27.read_text(encoding="utf-8"))

    pid_to_info = {p["problem_id"]: p for p in sample}
    problems_brief = "\n".join(
        f"[{p['problem_id']}] [{p.get('domain','?')}/{p.get('difficulty','?')}] {p.get('description','')[:90]}"
        for p in sample
    )

    # Load existing exemplars and only process new ones
    existing = {}
    if EXEMPLARS_OUT.exists():
        try:
            existing = json.loads(EXEMPLARS_OUT.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    print(f"Existing exemplars: {len(existing)} wisdoms")

    client = GPT5Client()
    t0 = time.time()

    for w in wisdom:
        wid = w["id"]
        if wid not in TARGET_IDS:
            continue
        if wid in existing:
            print(f"  [{wid}] already has exemplars, skip")
            continue

        prompt = SELECT_PROMPT.format(
            aphorism=w["aphorism"],
            source=w.get("source", "?"),
            signal=w.get("signal", "?"),
            unpacked=w.get("unpacked_for_llm", "?"),
            problems_brief=problems_brief,
        )
        for attempt in range(3):
            try:
                resp = client.generate(prompt, max_tokens=800, temperature=0.4)
                parsed = parse_json_from_llm(resp["text"])
                selected = parsed.get("selected", [])
                if len(selected) != 3:
                    raise ValueError(f"got {len(selected)}")
                valid = []
                for item in selected:
                    pid = item.get("pid", "").strip()
                    if pid not in pid_to_info:
                        continue
                    info = pid_to_info[pid]
                    dom = info.get("domain", "?")
                    ans_src = ours.get(pid) if dom in MATH_SCI else v13.get(pid)
                    ans_src = ans_src or v13.get(pid) or ours.get(pid) or ""
                    valid.append({
                        "pid": pid,
                        "domain": dom,
                        "difficulty": info.get("difficulty", "?"),
                        "problem_sketch": info.get("description", "")[:350],
                        "why_applies": item.get("why_applies", "").strip(),
                        "answer_snippet": ans_src[:700] if ans_src else "",
                        "answer_source": "ours_27" if dom in MATH_SCI else "v13_reflect",
                    })
                if len(valid) == 3:
                    existing[wid] = valid
                    print(f"  [{wid}] ✓ {[e['domain'] for e in valid]}")
                    break
                raise ValueError(f"only {len(valid)} valid")
            except Exception as e:
                if attempt == 2:
                    print(f"  [{wid}] FAIL: {e}")
                    break
                time.sleep(3)

    EXEMPLARS_OUT.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(existing)} exemplars total ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
