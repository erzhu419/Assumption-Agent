"""Exp 21 — Phase 4-6 of L4 self-evolution: agent actually writes
executable gate code, it runs, and we compare verdicts.

Builds on Exp 20's Phase 3 design output. Three new phases:

  Phase 4 (Generate):     Claude writes a complete .py implementing its
                          Phase-3 spec.  Uses explicit API inventory.
  Phase 5 (Execute):      subprocess runs the file.  If it crashes or
                          outputs malformed JSON, feed the traceback
                          back to Claude for correction (≤ 2 retries).
  Phase 6 (Compare):      load the agent-gate's verdicts on 12 candidates,
                          compare to Exp 17 trigger-conditioned gate.

Success metric: (i) Phase 4 code runs without human edits,
(ii) it produces a machine-readable verdict per candidate,
(iii) verdict overlap with Exp 17 is non-trivial (≠ 0 and ≠ 12).
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from claude_proxy_client import ClaudeProxyClient
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
GEN_DIR = PROJECT / "phase four" / "generated"
GEN_DIR.mkdir(exist_ok=True)
OUT_LOG = AUTO_DIR / "exp21_meta_execute_log.json"


PHASE4_PROMPT = """你要写一个完整的可执行 Python 文件，实现下面这个 gate 规格。

== 上次你设计的 gate 规格 (Phase 3) ==
{design}

== 已经有一个干净的 data API（你**只能**用这个，不要读其他任何文件）==

```python
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "phase four"))
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from exp21_data_api import list_candidates, candidate_info, per_pid_records

# Usage:
cids = list_candidates()                     # ['WCAND01', ..., 'WCROSSL01']  (12 items)

info = candidate_info("WCAND05")
# info = {{"cid", "aphorism", "wid", "unpacked", "signal", "source"}}

rows = per_pid_records("WCAND05")
# list of dicts, each: {{
#   "pid": "business_0015",
#   "problem": "...",             # 问题原文
#   "domain": "business",
#   "difficulty": "medium",
#   "base_what_changed": "...",   # base solver's Turn-0 reframe
#   "ext_what_changed": "...",    # ext solver's Turn-0 reframe (with this wisdom)
#   "base_anti_patterns": [...],  # list of strings
#   "ext_anti_patterns": [...],
#   "base_answer": "...",         # base solver final answer
#   "ext_answer": "...",          # ext solver final answer (with wisdom)
# }}
# 每个 cid 大约有 35-50 行。
```

== 可用的其他模块 ==

```python
import numpy as np
from sentence_transformers import SentenceTransformer
# m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# m.encode(list_of_strings, normalize_embeddings=True) -> ndarray shape (n, 384)

# 若某 component 需要 LLM judge:
from claude_proxy_client import ClaudeProxyClient
from llm_client import parse_json_from_llm
# c = ClaudeProxyClient()
# r = c.generate(prompt, max_tokens=400, temperature=0.0)
# v = parse_json_from_llm(r["text"])
```

== 输出要求 ==

一个完整 .py 文件。必须：
1. 用上面的 data API，不要自己读文件；
2. 对每个 cid 计算你的 4 个 components（reframe_depth, substantive_content_delta, wisdom_problem_alignment, antipattern_avoidance）；
3. 每个 component 按你 Phase 3 规格里定的 threshold 判 PASS/FAIL；
4. 最后**必须**写入 `AUTO = {project_path} / "phase four" / "autonomous"` 目录下的 `exp21_agent_gate_verdicts.json` 文件（完整绝对路径已帮你算好）；
5. 输出 schema：list，每项 `{{"cid", "component_scores": {{name: float}}, "per_component_pass": {{name: bool}}, "overall_pass": bool}}`。

== 写入文件的标准代码 ==

```python
from pathlib import Path
AUTO = Path(__file__).resolve().parent.parent / "phase four" / "autonomous"
output_path = AUTO / "exp21_agent_gate_verdicts.json"
import json
output_path.write_text(json.dumps(verdicts, ensure_ascii=False, indent=2))
print(f"Wrote {{len(verdicts)}} verdicts to {{output_path}}")
```

== 绝对必要 ==

- 不要 import sentence_transformers 失败就用 random 回退——如果 import 失败应该让脚本失败；
- 不要引入 pair-wr 的新计算（整个 gate 的意义是绕开它）；
- 数据行数 (`len(rows)`) 可能是 30-50；阈值设计要考虑这个；
- ≤ 300 行。

== 输出 ==

只输出一份 markdown code fence 包裹的 Python 代码：

```python
# ... 完整代码 ...
```"""


CORRECTION_PROMPT = """你上一轮写的代码执行失败。下面是**原始任务**（务必遵守）+ 你上一次的代码 + 错误。

== 原始任务（不要偏离）==
{original_task}

== 你上一次提交的代码 ==
```python
{prev_code}
```

== 执行错误 ==
```
{traceback}
```

修正后的完整 .py（严格只输出 markdown code fence 包裹的代码，不要任何其他文本，不要 CSV，不要虚构数据源，只能用 API 清单里列出的数据）："""


# API inventory to show the agent
API_INVENTORY = """from pathlib import Path
PROJECT = Path(__file__).parent.parent
CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO = PROJECT / "phase four" / "autonomous"

# sentence transformer: pip already installed
from sentence_transformers import SentenceTransformer

# LLM judge if needed
import sys
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
from claude_proxy_client import ClaudeProxyClient  # .generate(prompt, max_tokens, temperature) -> {"text":...}
from llm_client import parse_json_from_llm  # robust JSON parse from LLM output

# Basic cache I/O
import json
def cache_load(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return default
"""


def extract_code(text):
    """Robustly extract a Python block. Handles: ```python fences, ``` fences,
    leading narration text, missing closing fence."""
    # Try full ```python ... ``` fence
    m = re.search(r"```python\s*\n(.*?)(?:\n```|\Z)", text, re.DOTALL)
    if m: return m.group(1).strip()
    # Try bare ``` ... ``` fence
    m = re.search(r"```\s*\n(.*?)(?:\n```|\Z)", text, re.DOTALL)
    if m: return m.group(1).strip()
    # Strip any stray fence markers in raw text
    cleaned = re.sub(r"```(?:python)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)
    return cleaned.strip()


def write_candidates_fixture():
    """Create a minimal candidates fixture file for the agent's generated code."""
    fixture = [
        {"cid": "WCAND01", "aphorism": "上工治未病，不治已病",
         "base_meta_stem": "_valp_v20p1_base"},
        {"cid": "WCAND02", "aphorism": "别高效解决一个被看错的问题",
         "base_meta_stem": "_valp_v20p1_base"},
        {"cid": "WCAND03", "aphorism": "凡事预则立，不预则废",
         "base_meta_stem": "_valp_v20p1_base"},
        {"cid": "WCAND04", "aphorism": "急则治其标，缓则治其本",
         "base_meta_stem": "_valp_v20p1_base"},
        {"cid": "WCAND05", "committed_id": "W076",
         "aphorism": "凡益之道，与时偕行",
         "base_meta_stem": "_valp_v20p1_base"},
        {"cid": "WCAND06", "aphorism": "覆水难收，向前算账",
         "base_meta_stem": "_valp_v20p1_base"},
        {"cid": "WCAND07", "aphorism": "亲兄弟，明算账",
         "base_meta_stem": "_valp_v20p1_base"},
        {"cid": "WCAND08", "aphorism": "想理解行为，先看激励",
         "base_meta_stem": "_valp_v20p1_base"},
        {"cid": "WCAND09", "aphorism": "不谋全局者，不足谋一域",
         "base_meta_stem": "_valp_v20p1_base"},
        {"cid": "WCAND10", "committed_id": "W077",
         "aphorism": "没有调查，就没有发言权",
         "base_meta_stem": "_valp_v20p1_base"},
        {"cid": "WCAND11", "aphorism": "若不是品牌，你就只是商品。",
         "base_meta_stem": "_valp_v20p1_base"},
        {"cid": "WCROSSL01", "committed_id": "W078",
         "aphorism": "是骡子是马，拉出来遛遛",
         "base_meta_stem": "_valp_v20_base"},
    ]
    (AUTO_DIR / "exp20_candidates_fixture.json").write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2))
    return fixture


def try_execute(script_path, timeout=600):
    """Run the generated script. Return (success, stdout, stderr)."""
    try:
        r = subprocess.run([sys.executable, "-u", str(script_path)],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT"
    except Exception as e:
        return False, "", f"EXEC_ERROR: {e}"


def compare_verdicts(agent_verdicts, exp17_verdicts):
    """Overlap matrix between agent's and Exp 17's PASS/FAIL per cid."""
    agent_pass = {v["cid"] for v in agent_verdicts if v.get("overall_pass")}
    ex17_pass = {r["cid"] for r in exp17_verdicts if r.get("gate_pass")}
    all_cids = {v["cid"] for v in agent_verdicts} | {r["cid"] for r in exp17_verdicts}

    both_pass   = agent_pass & ex17_pass
    only_agent  = agent_pass - ex17_pass
    only_ex17   = ex17_pass  - agent_pass
    both_fail   = all_cids - agent_pass - ex17_pass

    return {
        "both_pass": sorted(both_pass),
        "only_agent_pass": sorted(only_agent),
        "only_ex17_pass": sorted(only_ex17),
        "both_fail": sorted(both_fail),
        "agent_pass_count": len(agent_pass),
        "ex17_pass_count": len(ex17_pass),
        "agreement_rate": (len(both_pass) + len(both_fail)) / max(len(all_cids), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-retries", type=int, default=2)
    args = ap.parse_args()

    # Load Exp 20 design spec (Phase 3 output)
    exp20_log = json.loads((AUTO_DIR / "exp20_meta_gate_designer_log.json").read_text())
    design = exp20_log[-1]["phases"]["3_design"]["output"]
    print(f"Loaded Exp 20 design: {design.get('gate_name', '?')}")
    print(f"Components: {len(design.get('components', []))}")

    # Load Exp 17 verdicts for comparison
    exp17_log = json.loads((AUTO_DIR / "exp17_trigger_conditioned_log.json").read_text())
    exp17_verdicts = exp17_log[-1]["results"]

    # Create candidates fixture
    write_candidates_fixture()
    print(f"Wrote candidates fixture.\n")

    claude = ClaudeProxyClient()
    print(f"Code generator: {claude.model}\n")

    log = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "design": design, "attempts": []}

    # ========== Phase 4: generate code ==========
    print("=== Phase 4: Generate code ===")
    prompt = PHASE4_PROMPT.format(
        design=json.dumps(design, ensure_ascii=False, indent=2),
        project_path=str(PROJECT),
    )
    t0 = time.time()
    try:
        r = claude.generate(prompt, max_tokens=4500, temperature=0.1)
        code = extract_code(r["text"])
    except Exception as e:
        print(f"  [FAIL] code generation: {e}")
        return
    print(f"  Generated {len(code)} chars in {time.time()-t0:.0f}s")

    script_path = GEN_DIR / "exp21_agent_gate.py"
    script_path.write_text(code, encoding="utf-8")
    print(f"  Saved → {script_path.relative_to(PROJECT)}\n")
    log["attempts"].append({"attempt": 0, "code_chars": len(code)})

    # ========== Phase 5: execute with retry ==========
    print("=== Phase 5: Execute ===")
    success = False
    for attempt in range(args.max_retries + 1):
        t0 = time.time()
        ok, stdout, stderr = try_execute(script_path)
        dt = time.time() - t0
        verdicts_path = AUTO_DIR / "exp21_agent_gate_verdicts.json"
        # Adapter: look for any recently-modified json that looks like verdicts
        # in either AUTO_DIR or generated/
        def _find_verdicts():
            candidates = []
            for d in (AUTO_DIR, GEN_DIR):
                for p in d.glob("*.json"):
                    if p.stat().st_mtime > t0:
                        candidates.append(p)
            for p in sorted(candidates, key=lambda q: q.stat().st_mtime, reverse=True):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        first = data[0]
                        # heuristics: has some id field + some pass field
                        if any(k in first for k in ("cid", "candidate_id", "id")):
                            return p, data
                except Exception:
                    continue
            return None, None
        found_path, found_data = _find_verdicts()
        produced = found_path is not None
        if produced and found_path != verdicts_path:
            # Normalize schema + move
            norm = []
            for row in found_data:
                cid = row.get("cid") or row.get("candidate_id") or row.get("id")
                ovr = row.get("overall_pass")
                if ovr is None:
                    ovr = row.get("gate_pass")
                norm.append({
                    "cid": cid,
                    "component_scores": row.get("component_scores", row.get("scores", {})),
                    "per_component_pass": row.get("per_component_pass", row.get("components_pass", {})),
                    "overall_pass": ovr,
                })
            verdicts_path.write_text(json.dumps(norm, ensure_ascii=False, indent=2))
            print(f"  [adapter] normalized {found_path.name} → {verdicts_path.name}")
        full_ok = ok and produced
        attempt_log = {"attempt": attempt, "returncode_ok": ok,
                        "produced_verdicts": produced, "full_ok": full_ok,
                        "elapsed_s": int(dt),
                        "stdout_tail": stdout[-800:],
                        "stderr_tail": stderr[-800:]}
        log["attempts"].append(attempt_log)
        if full_ok:
            print(f"  [PASS] attempt {attempt} ({dt:.0f}s) — verdicts produced")
            print(f"  stdout tail:\n{stdout[-600:]}")
            success = True
            break
        if ok and not produced:
            print(f"  [FAIL] attempt {attempt}: exit 0 but no verdicts file written")
            stderr = f"Script exited cleanly but did not write the expected "\
                     f"output file {verdicts_path.name}. Re-read the original "\
                     f"task and produce the required JSON file.\n\n{stdout[-500:]}"
        print(f"  [FAIL] attempt {attempt} ({dt:.0f}s)")
        print(f"  stderr tail: {stderr[-400:]}")
        if attempt == args.max_retries:
            break
        # Ask for correction
        print(f"  [correction request to Claude]")
        prev_code = script_path.read_text(encoding="utf-8")[:3500]
        corr_prompt = CORRECTION_PROMPT.format(
            original_task=prompt[:2500],
            prev_code=prev_code,
            traceback=stderr[-1500:],
        )
        try:
            r = claude.generate(corr_prompt, max_tokens=4500, temperature=0.1)
            new_code = extract_code(r["text"])
        except Exception as e:
            print(f"  [correction err] {e}"); break
        script_path.write_text(new_code, encoding="utf-8")
        print(f"  Rewrote {len(new_code)} chars")

    if not success:
        log["success"] = False
        OUT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2))
        print(f"\n[ABORT] agent-written code did not execute after "
              f"{args.max_retries+1} attempts"); return

    # ========== Phase 6: compare verdicts ==========
    print(f"\n=== Phase 6: Compare verdicts ===")
    verdicts_path = AUTO_DIR / "exp21_agent_gate_verdicts.json"
    if not verdicts_path.exists():
        print(f"  [FAIL] agent-gate did not produce {verdicts_path.name}"); return
    agent_verdicts = json.loads(verdicts_path.read_text(encoding="utf-8"))

    comparison = compare_verdicts(agent_verdicts, exp17_verdicts)
    print(f"  Agent gate PASS: {comparison['agent_pass_count']}/12")
    print(f"  Exp 17 gate PASS: {comparison['ex17_pass_count']}/12")
    print(f"  Agreement rate (PASS=PASS AND FAIL=FAIL): "
          f"{comparison['agreement_rate']:.2f}")
    print(f"  Both PASS:        {comparison['both_pass']}")
    print(f"  Only agent PASS:  {comparison['only_agent_pass']}")
    print(f"  Only Exp 17 PASS: {comparison['only_ex17_pass']}")
    print(f"  Both FAIL:        {comparison['both_fail']}")

    log["success"] = True
    log["agent_verdicts"] = agent_verdicts
    log["comparison"] = comparison
    OUT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
