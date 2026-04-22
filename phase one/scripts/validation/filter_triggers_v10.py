"""
v10: filter 161 triggers down to "verified" subset via GPT-5.4 rating.

Paper §2.3.3 "No Execution, No Memory": only strictly verified information
retained. Our pragmatic proxy for "verified": strong-model rating on actual
usefulness for LLM reasoning.

For each category, GPT-5.4 sees all triggers and picks top 40% that would
MOST improve an LLM's reasoning on that category's problems. Reasons:
  - category-local selection (not global; each category has its own gaps)
  - top 40% heuristic: aggressive enough to reduce noise, gentle enough to
    avoid losing legitimate coverage

Writes trigger_library_v10_verified.json.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "phase zero" / "scripts"))
from llm_client import parse_json_from_llm
from gpt5_client import GPT5Client


CACHE = Path("/home/erzhu419/mine_code/Asumption Agent/phase two/analysis/cache")
IN_PATH = CACHE / "trigger_library.json"
OUT_PATH = CACHE / "trigger_library_v10_verified.json"


FILTER_PROMPT = """你是 LLM scaffold 质量评估专家。下面这 {n} 条 trigger 都是 "场景+自问" 形式，是为 {cat} 类问题准备的警觉清单。

## 任务
从这 {n} 条中挑出 **最能真正改善 LLM 推理质量的 {k} 条**。
判断标准（严格按此排序）：
1. **信号强度**：这条 trigger 描述的情境是否真的在该类问题中高频出现？
2. **可激活性**：LLM 读到后能立即在脑中 "fire"，而不是要靠刻意记忆？
3. **差异化价值**：如果不看这条 trigger，LLM 会做错或做浅；看了它会做得更深或避坑？
4. **不 overlap**：挑出的 {k} 条信号点互相之间不应大量重复

## 要淘汰的（不要选）
- 泛泛的 meta 反思（"是否充分考虑"、"是否遗漏"这类）
- 过度细节/过度 instance-specific（"当 X 系统的 Y 组件出现 Z 症状时"这种）
- 重复覆盖同一信号点的多条变体
- 警觉无用、LLM 天然会考虑的点

## 输入（共 {n} 条）
{triggers_json}

## 输出
JSON（不要代码块）：
{{"keep_indices": [0, 3, 7, ...]}}   // 0-indexed，长度 = {k}
"""


def filter_category(client, cat, triggers, keep_ratio=0.40, max_retries=3):
    k = max(3, int(round(len(triggers) * keep_ratio)))
    for attempt in range(max_retries):
        try:
            # Enumerate with indices
            indexed = "\n".join(f"[{i}] {t}" for i, t in enumerate(triggers))
            prompt = FILTER_PROMPT.format(
                n=len(triggers), k=k, cat=cat,
                triggers_json=indexed,
            )
            resp = client.generate(prompt, max_tokens=500, temperature=0.2)
            parsed = parse_json_from_llm(resp["text"])
            idx = parsed.get("keep_indices", [])
            # Validate indices
            idx = [i for i in idx if isinstance(i, int) and 0 <= i < len(triggers)]
            idx = list(dict.fromkeys(idx))  # dedup preserve order
            if abs(len(idx) - k) > 2:
                raise ValueError(f"got {len(idx)} indices, want {k}")
            return [triggers[i] for i in idx]
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(5)


def main():
    orig = json.loads(IN_PATH.read_text(encoding="utf-8"))
    total = sum(len(v) for v in orig.values())
    print(f"Input: {total} triggers / {len(orig)} categories")

    client = GPT5Client()
    out = {}
    t0 = time.time()
    kept = 0

    for cat, triggers in orig.items():
        if not triggers:
            out[cat] = []
            continue
        if len(triggers) <= 3:
            out[cat] = list(triggers)
            kept += len(triggers)
            print(f"[{cat}] {len(triggers)} -> kept all (too few to filter)")
            continue
        print(f"[{cat}] {len(triggers)} -> filtering")
        verified = filter_category(client, cat, triggers)
        out[cat] = verified
        kept += len(verified)
        print(f"  kept {len(verified)}/{len(triggers)}")

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n=== Summary ===")
    print(f"  {total} -> {kept} ({100*kept/total:.0f}%) in {time.time()-t0:.0f}s")
    print(f"  Saved to {OUT_PATH.name}")


if __name__ == "__main__":
    main()
