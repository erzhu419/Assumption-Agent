"""
LLM-as-Translator: compile Polya/Popper canonical questions into LLM-consumable
orientations (scenario trigger + specific self-question).

Zero-shot: don't show LLM my own paraphrases as examples (would contaminate
experiment by inviting imitation).

Two independent features we require in each translated orientation:

  (A) SCENARIO TRIGGER — explicit WHEN condition.
      Good: "当感觉无路可走时...", "每次给出结论后 3 秒...", "如果发现 X 时..."
      Bad: open question without WHEN.

  (B) SPECIFIC SELF-QUESTION — directional, bounded.
      Good: "问自己：哪条约束是真必要的？" (bounded — picks ONE concrete element)
      Bad: "你能否检验假设?" (open yes/no, no direction)

Output: phase zero/kb/strategies_translated/S*.json with attention_priors filled
with LLM-translated paraphrases, SAME schema as strategies_orientation/.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import create_client, parse_json_from_llm

CANONICAL_DIR = Path("/home/erzhu419/mine_code/Asumption Agent/phase zero/kb/strategies_canonical")
ORIENT_DIR = Path("/home/erzhu419/mine_code/Asumption Agent/phase zero/kb/strategies_orientation")
OUT_DIR = Path("/home/erzhu419/mine_code/Asumption Agent/phase zero/kb/strategies_translated")
OUT_DIR.mkdir(parents=True, exist_ok=True)


TRANSLATE_PROMPT = """你是 LLM-prompt 翻译专家。

## 问题背景

{author} 在 {work} ({year}) 写的原典启发式问句是为**人类读者**设计的。人读后会
停下来反思 10 秒，翻动记忆，真的做类比或自检。

但 LLM（如 Gemini、GPT）不会这样"停下来反思"。LLM 读完问句只会**直接继续生成**，
对开放、抽象、反思性的提问会 superficially 处理（"yes 我见过 / no 没见过"）
然后略过。

## 你的任务

把下面这条策略的**原典问句**（给人类的）**编译为 LLM 消费形式**。每条编译后
的 orientation 必须**同时具备两个特征**：

**(A) 场景触发 (Scenario Trigger)** —— 显式 WHEN 条件
- 好: "当感觉陷入僵局时...", "每次开始新题前 3 秒...", "如果发现自己反复失败..."
- 坏: 无时间/条件约束的开放提问

**(B) 具体自问 (Specific Self-Question)** —— 定向、有边界、可操作
- 好: "问自己：**哪条约束**是真必要的？" (定向：追问一个具体维度)
- 好: "问自己：**什么观察**能推翻我这个结论？" (定向：追问具体反例)
- 坏: "你能否检验假设?" (开放 yes/no, 无方向)
- 坏: "你见过这个问题吗?" (Polya 原句直译, 对 LLM 无 trigger)

## 要求

1. **保留原典精神** —— Polya 的类比感、Popper 的证伪态度、特定策略的核心意识
2. **每条 30-60 字**，完整自问单元，不是步骤清单
3. **生成 3-5 条** LLM-consumable paraphrase
4. **不是直译**，是**编译** —— 原问句的精神要穿透到你的编译版里

## 当前策略

**ID**: {sid}
**中文名**: {name_zh}
**英文名**: {name_en}
**一句话描述**: {one_sentence}

## 原典问句（为人类读者，作为你翻译的素材，不要直接复用）

{canonical_questions_zh}

## 作为参考，其英文原话
{canonical_questions_en}

---

现在，**独立生成**你的 LLM 编译版本。不要复述原典文字，要**重造**成可以让
LLM"在读问题时自动浮现的觉知"。

输出 JSON (不要代码块):
{{
  "translations": [
    "场景触发 + 具体自问 的完整一行 paraphrase",
    ...
  ],
  "translation_notes": "说明你从原典里抓住了什么精神"
}}
"""


def translate_for_strategy(client, canonical_path: Path, orient_template_path: Path):
    """Generate translated paraphrases and save as new JSON in orient schema."""
    d_canon = json.loads(canonical_path.read_text(encoding="utf-8"))
    d_orient = json.loads(orient_template_path.read_text(encoding="utf-8"))

    sid = d_canon["id"]
    src = d_canon.get("source", {})
    qs_zh = d_canon.get("heuristic_questions_zh", [])
    qs_en = d_canon.get("heuristic_questions_en", [])

    prompt = TRANSLATE_PROMPT.format(
        author=src.get("author", "?"),
        work=src.get("work", "?"),
        year=src.get("year", "?"),
        sid=sid,
        name_zh=d_canon["name"]["zh"],
        name_en=d_canon["name"].get("en", ""),
        one_sentence=d_canon["description"]["one_sentence"],
        canonical_questions_zh="\n".join(f"  - {q}" for q in qs_zh),
        canonical_questions_en="\n".join(f"  - {q}" for q in qs_en),
    )

    for attempt in range(4):
        try:
            resp = client.generate(prompt, max_tokens=900, temperature=0.3)
            parsed = parse_json_from_llm(resp["text"])
            translations = parsed.get("translations", [])
            if not isinstance(translations, list) or not translations:
                raise ValueError("empty translations")
            break
        except Exception as e:
            if attempt == 3:
                raise
            print(f"    retry {attempt+1}: {e}")
            time.sleep(5 * (2 ** attempt))

    # Build new JSON using strategies_orientation/ schema
    new_d = {
        "id": sid,
        "name": d_canon["name"],
        "aliases": d_canon.get("aliases", []),
        "category": d_canon.get("category", ""),
        "form": "orientation",
        "source_references": d_canon.get("original_source_references", []),
        "description": {
            "one_sentence": d_orient["description"]["one_sentence"],
            "background": d_canon["description"].get("background", ""),
        },
        "trigger": "保持这些编译后的觉知在读问题时浮现，不是执行清单",
        "attention_priors": translations,  # <-- LLM-translated
        "original_wisdom": [
            {"source": f"{src.get('author','?')}, {src.get('work','?')} ({src.get('year','?')})",
             "text": q_en}
            for q_en in qs_en
        ],
        "applicability_conditions": d_canon.get("applicability_conditions", {}),
        "metadata": {
            **(d_canon.get("metadata", {})),
            "content_source": "llm_translated_from_canonical",
            "translator": "gemini-2.5-flash",
            "translation_notes": parsed.get("translation_notes", ""),
            "schema_version": "orient_v1",
        },
    }

    out_path = OUT_DIR / canonical_path.name
    out_path.write_text(json.dumps(new_d, ensure_ascii=False, indent=2))
    return out_path, translations, parsed.get("translation_notes", "")


def main():
    client = create_client()
    canonical_files = sorted(CANONICAL_DIR.glob("S*.json"))
    orient_files = {f.name: f for f in ORIENT_DIR.glob("S*.json")}

    print(f"Translating {len(canonical_files)} canonical strategies via LLM...")
    t0 = time.time()
    summaries = []
    for f in canonical_files:
        if f.name not in orient_files:
            print(f"  [skip] {f.name}: no orient template")
            continue
        out_path, trans, notes = translate_for_strategy(
            client, f, orient_files[f.name])
        d = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"\n  ✓ {d['id']} ({d['name']['zh']}): {len(trans)} translations")
        print(f"    notes: {notes[:120]}")
        for i, t in enumerate(trans[:3]):
            print(f"    {i+1}. {t}")
        summaries.append({"sid": d["id"], "name": d["name"]["zh"],
                         "translations": trans[:3]})

    print(f"\nDone in {time.time()-t0:.0f}s. Wrote {len(summaries)} files to {OUT_DIR.name}")
    return summaries


if __name__ == "__main__":
    main()
