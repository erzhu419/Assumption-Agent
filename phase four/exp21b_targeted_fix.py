"""Exp 21b — Targeted field-name fix on agent's existing gate code.

Exp 21 revealed that 2/4 components had data-contract bugs (agent used
'prompt' and 'reframe' instead of the documented 'problem' and
'critical_reframe' / 'base_what_changed' / 'ext_what_changed').

Instead of re-doing Phase 4 from scratch, we show the agent the exact
diagnosis and ask for a surgical patch. This tests whether the gap
is engineering (LLM can fix with a pointer) or cognitive (LLM genuinely
misunderstood the data).
"""

import json
import subprocess
import sys
import time
from pathlib import Path
import re

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from claude_proxy_client import ClaudeProxyClient
from gpt5_client import GPT5Client

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
GEN_DIR = PROJECT / "phase four" / "generated"


FIX_PROMPT = """你之前写的 gate 代码在 2/4 components 上 silently 失败（exit 0 但 n_prompts=0 或 min=-999 sentinel）。原因是**field name 用错了**，没有触发 KeyError（因为你用了 `.get(..., "")` 回退到空串）。

== 具体 bug 诊断 ==

API 里 `per_pid_records(cid)` 返回的每个 dict 包含这些字段（见 exp21_data_api.py）：
  pid, problem, domain, difficulty,
  base_what_changed, ext_what_changed,
  base_anti_patterns, ext_anti_patterns,
  base_answer, ext_answer

**你的代码用了这些不存在的字段**:
  rec["prompt"]          → 应该用 rec["problem"]
  rec["prompt_text"]     → 应该用 rec["problem"]
  rec["question"]        → 应该用 rec["problem"]
  rec["critical_reframe"] (从 meta 里找) → 应该用 rec["ext_what_changed"] 或 rec["base_what_changed"]
  rec["reframe"]         → 同上

这导致:
  reframe_depth          n_prompts=0 (从来没找到 reframe)
  wisdom_problem_alignment  min_alignment_z=-999 sentinel

== 下面是你自己写的代码 (完整) ==

```python
{code}
```

== 任务 ==

做**最小修改**：只改那些用错字段名的行，让 2/4 失败的 component 也能正确读数据跑完。不要 rewrite 整个文件。不要改成功的 2 个 component。

评估函数的正确 intent:
- reframe_depth:   比较 `problem` 和 `ext_what_changed` 的语义距离（ext 相对 problem 的 reframe 深度）
- wisdom_problem_alignment: 用 wisdom.unpacked (来自 candidate_info(cid)) 对每个 pid 的 `problem` 做 embedding cosine

== 输出 ==

完整修复后的 .py 代码，markdown code fence 包裹。"""


def extract_code(text):
    m = re.search(r"```python\s*\n(.*?)(?:\n```|\Z)", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)(?:\n```|\Z)", text, re.DOTALL)
    if m: return m.group(1).strip()
    return text.strip()


def main():
    # Try Claude first; fall back to GPT-5.4 on rate-limit
    try:
        client = ClaudeProxyClient()
        # Quick sanity call
        client.generate("ping", max_tokens=5, temperature=0.0)
    except Exception as e:
        print(f"  [Claude unavailable: {str(e)[:60]}] falling back to GPT-5.4")
        client = GPT5Client()
    orig_code_path = GEN_DIR / "exp21_agent_gate.py"
    fixed_code_path = GEN_DIR / "exp21_agent_gate_fixed.py"

    if not orig_code_path.exists():
        print("Need phase four/generated/exp21_agent_gate.py first"); return
    orig_code = orig_code_path.read_text(encoding="utf-8")
    print(f"Loaded original: {len(orig_code)} chars\n")

    print(f"Asking {client.model} for targeted field-name fix...")
    t0 = time.time()
    prompt = FIX_PROMPT.format(code=orig_code)
    r = client.generate(prompt, max_tokens=12000, temperature=0.1)
    fixed = extract_code(r["text"])
    print(f"  Got {len(fixed)} chars in {time.time()-t0:.0f}s")
    # Sanity: does it have an entry point?
    if 'if __name__' not in fixed and 'main()' not in fixed[-500:]:
        print(f"  [WARN] code likely truncated — no entry point detected")
    fixed_code_path.write_text(fixed, encoding="utf-8")
    print(f"  Saved → {fixed_code_path.relative_to(PROJECT)}\n")

    # Execute
    print("Executing fixed code...")
    t0 = time.time()
    res = subprocess.run([sys.executable, "-u", str(fixed_code_path)],
                         capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    print(f"  returncode={res.returncode}  elapsed={dt:.0f}s")
    if res.returncode != 0:
        print(f"  stderr: {res.stderr[-800:]}")
        return
    print(f"  stdout tail:\n{res.stdout[-1200:]}")

    # Find the output verdicts
    recent = None
    for p in GEN_DIR.glob("*.json"):
        if p.stat().st_mtime > t0:
            recent = p; break
    if not recent:
        print("No new JSON produced"); return
    print(f"\n  Agent wrote → {recent.relative_to(PROJECT)}")

    data = json.loads(recent.read_text(encoding="utf-8"))
    # Compare with Exp 17
    e17 = json.loads((AUTO_DIR / "exp17_trigger_conditioned_log.json").read_text())[-1]["results"]
    e17_pass = {r["cid"] for r in e17 if r.get("gate_pass")}

    # Normalize — handle both dict-keyed and list schemas
    if isinstance(data, dict):
        agent_pass = {cid for cid, row in data.items()
                       if row.get("gate_verdict") == "PASS" or
                          row.get("overall_pass") is True}
        component_stats = {cid: row.get("components", {}) for cid, row in data.items()}
    else:
        agent_pass = {r.get("cid") or r.get("candidate_id") for r in data
                       if r.get("gate_verdict") == "PASS" or
                          r.get("overall_pass") is True or
                          r.get("gate_pass") is True}
        component_stats = {(r.get("cid") or r.get("candidate_id")): r.get("components", {})
                            for r in data}

    all_cids = set(component_stats.keys()) | e17_pass
    both_pass = agent_pass & e17_pass
    only_agent = agent_pass - e17_pass
    only_e17 = e17_pass - agent_pass
    both_fail = all_cids - agent_pass - e17_pass
    agreement = (len(both_pass) + len(both_fail)) / max(len(all_cids), 1)

    print(f"\n{'='*70}\nComparison after targeted fix\n{'='*70}")
    print(f"  Agent gate PASS: {len(agent_pass)}/12  {sorted(agent_pass)}")
    print(f"  Exp 17 gate PASS: {len(e17_pass)}/12  {sorted(e17_pass)}")
    print(f"  Agreement rate: {agreement:.2f}")
    print(f"  Both PASS:      {sorted(both_pass)}")
    print(f"  Only agent PASS: {sorted(only_agent)}")
    print(f"  Only Exp 17 PASS: {sorted(only_e17)}")

    # Per-component stats now
    print(f"\n  Per-candidate gate + components:")
    for cid in sorted(component_stats.keys()):
        comps = component_stats[cid]
        passed = {}
        for n, c in comps.items():
            if isinstance(c, dict):
                passed[n] = c.get("passed")
        print(f"    {cid}: {passed}")

    out_log = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent_pass": sorted(agent_pass),
        "e17_pass": sorted(e17_pass),
        "both_pass": sorted(both_pass),
        "only_agent_pass": sorted(only_agent),
        "only_e17_pass": sorted(only_e17),
        "agreement_rate": agreement,
        "agent_verdicts_path": str(recent),
    }
    (AUTO_DIR / "exp21b_fixed_comparison.json").write_text(
        json.dumps(out_log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
