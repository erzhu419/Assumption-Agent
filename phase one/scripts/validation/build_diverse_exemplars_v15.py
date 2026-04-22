"""
v15 prep: for each wisdom entry, GPT-5.4 selects 3 MAXIMALLY DIVERSE exemplars
(problem + high-quality answer) from sample_100 that illustrate the wisdom.

Diversity is the key design: don't pick 3 similar cases, pick cases from
VERY DIFFERENT domains / problem types / surface features, all illustrating
the same underlying principle.

Analogy: law school teaches contract interpretation via diverse precedents
(construction, securities, marriage) — the cross-domain spread forces the
student to abstract the invariant.

Output: wisdom_diverse_exemplars.json
  {
    "W012": [
      {"pid": "...", "domain": "...", "problem_sketch": "...",
       "why_this_wisdom_applies": "...", "answer_snippet": "..."},
      {...},
      {...}
    ],
    ...
  }
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "phase zero" / "scripts"))
from llm_client import parse_json_from_llm
from gpt5_client import GPT5Client


CACHE = Path("/home/erzhu419/mine_code/Asumption Agent/phase two/analysis/cache")
WISDOM = CACHE / "wisdom_library.json"
SAMPLE = CACHE / "sample_100.json"
V13_REFLECT = CACHE / "answers/phase2_v13_reflect_answers.json"
OURS_27 = CACHE / "answers/ours_27_answers.json"
OUT = CACHE / "wisdom_diverse_exemplars.json"

MATH_SCI = {"mathematics", "science"}


SELECT_PROMPT = """你是 LLM scaffold 的判例集(case library)设计师。任务：为下面这条 wisdom 从 100 个问题候选中挑选 **3 个跨域最远、各自独立**的判例，让 LLM 能从差异中抽象出不变原则。

## wisdom 条目
- aphorism: "{aphorism}"
- source: {source}
- signal: {signal}
- unpacked: {unpacked}

## 候选 problems (100 个，按 pid 列出)
{problems_brief}

## 已有的每题"最佳解答"参考
(如果该题被选中，我会从下面注入相应答案作为"判例内容")

## 任务
从 100 个候选中选 **正好 3 个** pid, 满足：

1. **本条 wisdom 真的能 fire 在该问题上**（不是勉强扯上）
2. **3 个选择跨域最远**：
   - 避免 3 个都是 business / daily_life
   - 最好跨 2-3 个不同 domain
   - 避免 problem 结构高度相似
3. **diversity of mechanism**：即使 domain 相同，问题触发 wisdom 的机制也应该有差别

## 输出 JSON（不要代码块）
{{"selected": [
  {{"pid": "xxx", "why_applies": "为什么这个 wisdom 适用本题 (20-40 字)"}},
  {{"pid": "yyy", "why_applies": "..."}},
  {{"pid": "zzz", "why_applies": "..."}}
]}}
"""


def main():
    wisdom = json.loads(WISDOM.read_text(encoding="utf-8"))
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    v13 = json.loads(V13_REFLECT.read_text(encoding="utf-8"))
    ours = json.loads(OURS_27.read_text(encoding="utf-8"))

    print(f"  wisdom: {len(wisdom)} entries")
    print(f"  sample: {len(sample)} problems")

    # Build problems_brief (one line each, compact)
    pid_to_info = {p["problem_id"]: p for p in sample}
    problems_brief_lines = []
    for p in sample:
        pid = p["problem_id"]
        desc = p.get("description", "")[:90]
        dom = p.get("domain", "?")
        diff = p.get("difficulty", "?")
        problems_brief_lines.append(f"[{pid}] [{dom}/{diff}] {desc}")
    problems_brief = "\n".join(problems_brief_lines)

    out = {}
    if OUT.exists():
        try:
            out = json.loads(OUT.read_text(encoding="utf-8"))
            print(f"  resuming: {len(out)} entries already done")
        except Exception:
            out = {}

    client = GPT5Client()
    t0 = time.time()
    new_count = 0

    for w in wisdom:
        wid = w["id"]
        if wid in out:
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
                    raise ValueError(f"got {len(selected)} selections, want 3")
                # Validate pids
                valid = []
                for item in selected:
                    pid = item.get("pid", "").strip()
                    if pid in pid_to_info:
                        why = item.get("why_applies", "").strip()
                        info = pid_to_info[pid]
                        dom = info.get("domain", "?")
                        ans_src = ours.get(pid) if dom in MATH_SCI else v13.get(pid)
                        ans_src = ans_src or v13.get(pid) or ours.get(pid) or ""
                        valid.append({
                            "pid": pid,
                            "domain": dom,
                            "difficulty": info.get("difficulty", "?"),
                            "problem_sketch": info.get("description", "")[:350],
                            "why_applies": why,
                            "answer_snippet": ans_src[:700] if ans_src else "",
                            "answer_source": "ours_27" if dom in MATH_SCI else "v13_reflect",
                        })
                if len(valid) == 3:
                    out[wid] = valid
                    new_count += 1
                    break
                else:
                    raise ValueError(f"only {len(valid)} valid pids")
            except Exception as e:
                if attempt == 2:
                    print(f"  [fail {wid}] {e}")
                    break
                time.sleep(3)

        if new_count % 5 == 0 and new_count > 0:
            OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
            print(f"  [{wid}] {new_count}/{len(wisdom)} done, {time.time()-t0:.0f}s")

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n=== Summary ===")
    print(f"  Mined {len(out)}/{len(wisdom)} wisdom entries with 3 exemplars each")
    print(f"  Total time: {time.time()-t0:.0f}s")

    # Diversity check
    from collections import Counter
    domain_spread = []
    for wid, exs in out.items():
        doms = sorted(set(e["domain"] for e in exs))
        domain_spread.append((wid, len(doms), doms))
    # How many wisdoms got 3 different domains?
    full_div = sum(1 for _, n, _ in domain_spread if n == 3)
    two_div = sum(1 for _, n, _ in domain_spread if n == 2)
    one_div = sum(1 for _, n, _ in domain_spread if n == 1)
    print(f"\n  Domain diversity: 3-domain={full_div}, 2-domain={two_div}, 1-domain={one_div}")

    print("\n  First 5 sample entries:")
    for w in wisdom[:5]:
        wid = w["id"]
        if wid not in out:
            continue
        print(f"\n  [{wid}] {w['aphorism']}")
        for e in out[wid]:
            print(f"    - [{e['pid']}/{e['domain']}] {e['why_applies']}")


if __name__ == "__main__":
    main()
