"""
Phase 2 v3: build Wisdom Library from cross-civilizational non-fiction sources.

Use GPT-5.4 to generate 80-100 wisdom entries, each with:
  - aphorism: original or near-original phrasing (≤35 chars, attributed)
  - source: author + work (Russell, Popper, 传道书, 孙子, etc.)
  - signal: when/what pattern triggers this
  - unpacked_for_llm: pre-unpacked scenario+self-question form (40-80 chars)
  - cross_domain_examples: 2 examples in completely different domains

Generate in BATCHES of 10 entries per call (across multiple calls to get
diverse sources). Each call seeds with a different source cluster.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import parse_json_from_llm
from gpt5_client import GPT5Client

OUT_PATH = Path("/home/erzhu419/mine_code/Asumption Agent/phase two/analysis/cache/wisdom_library.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# Source clusters — each batch asks GPT-5.4 to mine from a specific tradition.
# This forces diverse corpora coverage (not all Popper or all 孙子).

SOURCE_CLUSTERS = [
    {
        "name": "biblical_wisdom",
        "description": "圣经智慧文学（传道书、箴言、约伯记、诗篇）中的跨时代洞察",
        "examples_to_include": [
            "已有的后必再有（传道书 1:9）",
            "寻找的就必寻见（马太福音 7:7）",
            "保守你心，胜过保守一切（箴言 4:23）",
        ],
        "target_count": 8,
    },
    {
        "name": "chinese_classics",
        "description": "中国古典（孙子兵法、道德经、论语、庄子、易经、韩非子）的 orientation-level 警句",
        "examples_to_include": [
            "知彼知己，百战不殆（孙子）",
            "祸兮福之所倚（道德经）",
            "己所不欲，勿施于人（论语）",
        ],
        "target_count": 10,
    },
    {
        "name": "popper_russell_wittgenstein",
        "description": "20 世纪哲学（Popper、Russell、Wittgenstein、Kuhn）的 meta-reasoning 警句",
        "examples_to_include": [
            "能解释一切的理论解释不了任何事（Popper）",
            "一切悲剧来自该坚持时不坚持、不该坚持时坚持了（Russell）",
            "凡是可说的都能说清楚，凡不能说的必须保持沉默（Wittgenstein）",
        ],
        "target_count": 10,
    },
    {
        "name": "einstein_feynman_science",
        "description": "科学家（Einstein、Feynman、Poincaré、Polya）的方法论警句",
        "examples_to_include": [
            "凡事应尽可能简单，但不能过度简单（Einstein）",
            "如果你不能简单解释，那你就没真懂（Feynman）",
            "发现始于直觉，证明始于逻辑（Poincaré）",
        ],
        "target_count": 8,
    },
    {
        "name": "economists_hayek_smith",
        "description": "经济学家和社会思想家（Hayek、Smith、Friedman、Acton）的结构性洞察",
        "examples_to_include": [
            "揭示观念的根源是最令人恼怒的发现（Acton）",
            "看不见的手（Adam Smith）",
            "通往地狱的路由善意铺就",
        ],
        "target_count": 8,
    },
    {
        "name": "psychology_kahneman_bias",
        "description": "认知科学与行为经济（Kahneman、Tversky、Ariely、芒格）的 bias 警觉",
        "examples_to_include": [
            "我们高估眼前，低估远方（双曲贴现）",
            "让你舒服的答案通常是错的（confirmation bias）",
            "事后看起来理所当然，事前你完全猜不到（hindsight bias）",
        ],
        "target_count": 10,
    },
    {
        "name": "wisdom_aphorisms_folk",
        "description": "民间成语、谚语、跨文化警句（温水煮青蛙、一叶知秋、刻舟求剑、温故知新、塔木德、亚洲谚语）",
        "examples_to_include": [
            "温水煮青蛙（渐进危机盲区）",
            "一叶知秋（微信号推大局）",
            "刻舟求剑（过时的地图）",
        ],
        "target_count": 10,
    },
    {
        "name": "strategic_military_sunzi_clausewitz",
        "description": "战略思想（孙子、Clausewitz、Liddell Hart、Machiavelli）的 orientation",
        "examples_to_include": [
            "不战而屈人之兵（孙子）",
            "战争是政治的延续（Clausewitz）",
            "宁愿被惧怕，不要被爱（Machiavelli）",
        ],
        "target_count": 8,
    },
    {
        "name": "polya_heuristics",
        "description": "Polya《How to Solve It》的原典 heuristic questions（但必须 pre-unpack 为 scenario+self-question 形式）",
        "examples_to_include": [
            "类似问题你见过吗？",
            "能否放宽一个条件？",
            "逆向从目标推原因",
        ],
        "target_count": 6,
    },
    {
        "name": "miscellaneous_high_insight",
        "description": "其他高 insight 密度的 orientation（Taleb、Ryan Holiday、Jordan Peterson、Nassim 等）",
        "examples_to_include": [
            "黑天鹅：意想不到的极端事件（Taleb）",
            "障碍即道路（Holiday）",
            "秩序与混沌的张力是生命的本质（Peterson）",
        ],
        "target_count": 8,
    },
]


BATCH_PROMPT = """你是人类智慧提炼专家。请从 **{source_name}** 这一源头，生成 {count} 条 orientation-level wisdom entries。

## 源头
{description}

## 一定要包括的经典（或它们的变体）
{examples_to_include}

## 每条 entry 的要求

**格式约束（严格）：**

1. `aphorism`：原文或忠实改写，**≤35 中文字符**，有警句感、可记忆
2. `source`：具体作者+作品（如 "Russell, 《哲学问题》" 或 "传道书 1:9"）
3. `signal`：一句话说明**什么情境/问题结构/征兆**会激活这条（让 LLM 一眼就能识别"该用它了"）
4. `unpacked_for_llm`：**这是关键字段！**把 aphorism 展开成 LLM 可直接消化的 scenario+self-question 形式，60-120 字。
   - 好例子：`"当你考虑放弃或继续时，先问两个问题：(1) 我之前的坚持来源是真 commitment 还是 sunk cost？(2) 这次放弃是理智回撤还是因为短期不舒服？"`
   - 不允许的形式：直接复述 aphorism；开放提问如"你思考过吗"；模板化 meta-reflection
5. `cross_domain_examples`: **两个完全不同领域**（从 business / daily_life / engineering / mathematics / science / software_engineering 里选两个不同的）的适用场景，每个 30-60 字
6. `abstraction_check`：**trigger 文本本身不能含 domain-specific 术语**（不能出现"代码、用户、投资、分子、实验、家庭、孩子"等）

## 严格避免的问题

- 不要重复同源头已经著名的警句（比如生成 Popper 第 5 条重复的证伪）
- 不要生成"是否充分识别/是否充分考虑"这种模板化 meta-reflection
- 不要生成虚构的、你编造的"据说是 X 说的"（如不确定出处就直接写"民间谚语"或"未详"）
- aphorism 必须是**断言 / 悖论 / 反问**形式，不是中性描述

## 输出 JSON（不要代码块）

{{
  "entries": [
    {{
      "aphorism": "...",
      "source": "作者，作品",
      "signal": "...",
      "unpacked_for_llm": "...",
      "cross_domain_examples": [
        {{"domain": "...", "scenario": "..."}},
        {{"domain": "不同 domain", "scenario": "..."}}
      ],
      "abstraction_check": "trigger 用了哪些抽象词汇，没有 domain 术语"
    }},
    ... （{count} 条）
  ]
}}
"""


DOMAIN_SPECIFIC_TOKENS = [
    "代码", "debug", "调试", "CI", "CD", "bug", "Bug", "服务器",
    "客户", "用户", "员工", "产品", "订单", "投资", "股票", "金融",
    "算法", "证明", "定理", "公式", "实验", "分子", "细胞", "基因",
    "家庭", "孩子", "父母", "配偶", "同事",
    "代码库", "数据库",
]


def validate_entry(e: dict) -> (bool, str):
    """Return (is_valid, rejection_reason)."""
    aph = e.get("aphorism", "").strip()
    unp = e.get("unpacked_for_llm", "").strip()
    if not aph or len(aph) > 60:
        return False, f"aphorism length: {len(aph)}"
    if not unp or len(unp) < 40:
        return False, f"unpacked too short: {len(unp)}"
    # Check domain tokens in aphorism (not unpacked — unpacked can mention domains within examples)
    violations = [tok for tok in DOMAIN_SPECIFIC_TOKENS if tok in aph]
    if violations:
        return False, f"aphorism contains domain tokens: {violations}"
    # Check example domains differ
    exs = e.get("cross_domain_examples", [])
    if len(exs) < 2:
        return False, "need 2 cross-domain examples"
    d1 = (exs[0].get("domain", "") if isinstance(exs[0], dict) else "").lower().strip()
    d2 = (exs[1].get("domain", "") if isinstance(exs[1], dict) else "").lower().strip()
    if d1 and d2 and d1 == d2:
        return False, f"examples same domain: {d1}"
    return True, ""


def main():
    client = GPT5Client()
    all_entries = []
    t0 = time.time()

    for cluster in SOURCE_CLUSTERS:
        print(f"\n[{cluster['name']}] targeting {cluster['target_count']} entries")
        examples_text = "\n".join(f"  - {ex}" for ex in cluster["examples_to_include"])
        prompt = BATCH_PROMPT.format(
            source_name=cluster["name"],
            description=cluster["description"],
            examples_to_include=examples_text,
            count=cluster["target_count"],
        )
        try:
            resp = client.generate(prompt, max_tokens=5000, temperature=0.4)
            parsed = parse_json_from_llm(resp["text"])
            entries = parsed.get("entries", [])
        except Exception as e:
            print(f"  [error] {e}")
            continue

        kept = 0
        for entry in entries:
            ok, reason = validate_entry(entry)
            if not ok:
                print(f"  [reject] {entry.get('aphorism','?')[:30]} — {reason}")
                continue
            entry["cluster"] = cluster["name"]
            entry["id"] = f"W{len(all_entries)+1:03d}"
            all_entries.append(entry)
            kept += 1
        print(f"  [ok] kept {kept}/{len(entries)}")

    # Save
    OUT_PATH.write_text(json.dumps(all_entries, ensure_ascii=False, indent=2))
    print(f"\n\nWisdom Library: {len(all_entries)} entries")
    print(f"Saved to {OUT_PATH.name}  ({time.time()-t0:.0f}s)")

    # Summary
    print("\n=== By source cluster ===")
    from collections import Counter
    c = Counter(e.get("cluster", "?") for e in all_entries)
    for k, v in c.most_common():
        print(f"  {k}: {v}")

    print("\n=== First 10 sample aphorisms ===")
    for e in all_entries[:10]:
        print(f"  [{e['id']}] {e['aphorism']} ({e['source']})")


if __name__ == "__main__":
    main()
