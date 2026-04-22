"""
P0-2: truncate v12c answers to match v11 length (pairwise, per problem),
ending at nearest sentence boundary. Then judge v12c_trunc vs v11 to test
whether v12c's +8pp is reasoning or length artifact.

Truncation rule per problem:
  - target = min(len(v11_ans), len(v12c_ans))
  - take v12c_ans[:target], then back up to nearest sentence end
    (。 ！ ？ newline) so we don't cut mid-sentence
  - if that leaves too little (<50% of target), keep hard truncation
"""

import json
import re
from pathlib import Path

CACHE = Path("/home/erzhu419/mine_code/Asumption Agent/phase two/analysis/cache")
V11 = CACHE / "answers/phase2_v11_answers.json"
V12C = CACHE / "answers/phase2_v12c_answers.json"
OUT = CACHE / "answers/phase2_v12c_trunc_answers.json"


SENTENCE_END = re.compile(r'[。！？\n](?=\S|$)')


def truncate_at_sentence(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    clipped = text[:max_len]
    # find last sentence-ending punctuation
    matches = list(SENTENCE_END.finditer(clipped))
    if matches and matches[-1].end() >= max_len * 0.5:
        return clipped[:matches[-1].end()].rstrip()
    return clipped.rstrip()


def main():
    v11 = json.loads(V11.read_text(encoding="utf-8"))
    v12c = json.loads(V12C.read_text(encoding="utf-8"))

    out = {}
    stats = []
    for pid, orig in v12c.items():
        v11_ans = v11.get(pid)
        if not v11_ans:
            continue
        target = len(v11_ans)
        truncated = truncate_at_sentence(orig, target)
        out[pid] = truncated
        stats.append({
            "pid": pid,
            "v11_len": len(v11_ans),
            "v12c_orig_len": len(orig),
            "v12c_trunc_len": len(truncated),
        })

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    # Report
    print(f"Truncated {len(out)} answers; saved to {OUT.name}")
    orig_lens = [s["v12c_orig_len"] for s in stats]
    trunc_lens = [s["v12c_trunc_len"] for s in stats]
    v11_lens = [s["v11_len"] for s in stats]
    print(f"\n  v11 avg len:         {sum(v11_lens)/len(v11_lens):.0f}")
    print(f"  v12c orig avg len:   {sum(orig_lens)/len(orig_lens):.0f}")
    print(f"  v12c trunc avg len:  {sum(trunc_lens)/len(trunc_lens):.0f}")

    # How much truncated per domain
    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))
    dom_map = {p["problem_id"]: p["domain"] for p in sample}
    from collections import defaultdict
    by_dom = defaultdict(list)
    for s in stats:
        d = dom_map.get(s["pid"], "?")
        by_dom[d].append((s["v11_len"], s["v12c_orig_len"], s["v12c_trunc_len"]))

    print("\n  Per domain (v11 / v12c_orig / v12c_trunc):")
    for d, triples in sorted(by_dom.items()):
        vs = [t[0] for t in triples]
        os_ = [t[1] for t in triples]
        ts = [t[2] for t in triples]
        print(f"    {d:<22} {sum(vs)/len(vs):>5.0f} / {sum(os_)/len(os_):>5.0f} / {sum(ts)/len(ts):>5.0f}")


if __name__ == "__main__":
    main()
