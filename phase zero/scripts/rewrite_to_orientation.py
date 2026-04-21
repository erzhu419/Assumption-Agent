"""
Rewrite 12 Polya/Popper-derived strategies as orientation form.

Key schema differences:
  - REMOVE: operational_steps (technique checklist)
  - ADD: trigger (when the awareness activates — usually "continuously")
  - ADD: attention_priors (3-5 self-questions that direct attention, NOT execution)
  - ADD: original_wisdom (direct quote from source author)
  - KEEP: id, name, category, source_references, applicability_conditions
  - REPURPOSE: description.detailed → background (marked as context, not instructions)
"""

import json
from pathlib import Path

KB_DIR = Path("/home/erzhu419/mine_code/Asumption Agent/phase zero/kb/strategies")
OUT_DIR = Path("/home/erzhu419/mine_code/Asumption Agent/phase zero/kb/strategies_orientation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


ORIENTATION_DATA = {
    "S02": {  # 分而治之 - Polya
        "one_sentence_orient": "训练自己在面对复杂问题时自动反问'它的自然裂缝在哪里'",
        "trigger": "每当感觉问题'太大'、思绪被淹没时启动 — 不是先动手，先问分解",
        "attention_priors": [
            "这个问题天然的'接缝'在哪里？沿着哪条线切它，子问题才真正独立？",
            "如果我只解决一半，剩下的一半会不会自动浮现？",
            "我倾向于用'大力出奇迹'处理这题吗？那是信号 —— 该退一步找接缝",
        ],
        "original_wisdom": [
            {"source": "Polya 1945 'How to Solve It'",
             "text": "If you can't solve a problem, then there is an easier problem you can solve: find it. Could you solve a part of the problem?"},
        ],
    },
    "S03": {  # 类比推理 - Polya
        "one_sentence_orient": "养成每遇新问题就反问'我见过类似的吗'的习惯",
        "trigger": "持续启动 — 每次接触新问题时，而不是一个'用完即丢'的工具",
        "attention_priors": [
            "这个问题的抽象骨架是什么？剥掉具体内容后剩下什么？",
            "我在完全不同的领域里见过同样的骨架吗？（数学-物理，医学-经济，生物-工程？）",
            "如果见过类似的，那边的解为什么在这里可能失效？",
        ],
        "original_wisdom": [
            {"source": "Polya 1945 'How to Solve It'",
             "text": "Have you seen a related problem? Can you think of a familiar problem having the same unknown or a similar unknown?"},
        ],
    },
    "S04": {  # 反证法 - Polya
        "one_sentence_orient": "保持对'结论如果错会怎样'的敏感 — 对立面往往藏着最有用的信息",
        "trigger": "当一个结论'看起来显然对'或'大家都这么说'时，刻意启动反向检验",
        "attention_priors": [
            "如果我当前的假设是错的，世界会长什么样？我能观察到什么？",
            "我的论证链条里，哪一环最脆弱？如果它断了，整个结论还成立吗？",
            "我有没有在寻找支持证据，而不是寻找反驳证据？",
        ],
        "original_wisdom": [
            {"source": "Polya 1945 'How to Solve It'",
             "text": "If you cannot solve the proposed problem, try to assume the contrary and see where it leads."},
        ],
    },
    "S06": {  # 先特殊后一般 - Polya
        "one_sentence_orient": "在抽象推理前，先找一个最小具体的 instance 走一遍 — 直觉走不通的地方往往就是真正的难点",
        "trigger": "每次试图直接写通用解法时，先暂停，问'我能手算 n=2 的情况吗'",
        "attention_priors": [
            "最简单的特例是什么？我能完整解掉它吗？",
            "在特例里我发现了什么模式？这个模式会迁移到一般情况吗？",
            "如果一般情况的方法在特例上不适用，那这个方法本身可能就错了",
        ],
        "original_wisdom": [
            {"source": "Polya 1945 'How to Solve It'",
             "text": "Consider special cases. Examples, extreme cases, and limiting cases often throw unexpected light on the general problem."},
        ],
    },
    "S08": {  # 试错法 - Popper
        "one_sentence_orient": "把猜测当作探测器而非答案 — 错的猜测比没猜更有信息",
        "trigger": "信息不足、解析方法失效时启动；不是穷举，是有目的地'猜-学'",
        "attention_priors": [
            "我的下一个猜测是为了获得什么信息？它失败会教我什么？",
            "我在'重复同样的错'吗？如果是，我学到了什么却还没用上？",
            "一个好的猜测应该具有区分力 —— 它的对错能把假设空间切半",
        ],
        "original_wisdom": [
            {"source": "Popper 1959 'The Logic of Scientific Discovery'",
             "text": "All life is problem solving. Bold conjectures, rigorously tested — the mistakes teach us more than the successes."},
        ],
    },
    "S09": {  # 降维/简化 - Polya
        "one_sentence_orient": "保持'我是不是在关心不关键的细节'的警觉 — 复杂性往往是方向错误的症状",
        "trigger": "当感觉被细节淹没、进展缓慢时启动，反问'哪些我可以忽略'",
        "attention_priors": [
            "这个问题的本质核心是什么？去掉什么后核心还在？",
            "我在花精力的细节，如果都错了，结论会变吗？如果不会，为什么要花精力？",
            "有没有一个'愚蠢简单'的版本我还没问自己能不能解？",
        ],
        "original_wisdom": [
            {"source": "Polya 1945 'How to Solve It'",
             "text": "Could you imagine a more accessible related problem? A more general problem? A more special problem? An analogous problem?"},
        ],
    },
    "S10": {  # 对称性 - Polya
        "one_sentence_orient": "感知问题中的'不变量' — 它们往往是解的骨架",
        "trigger": "感觉搜索空间巨大或情况数暴涨时，先问'这里有什么是不变的'",
        "attention_priors": [
            "如果我把问题做 X 变换（交换、翻转、缩放），它本质变了吗？",
            "不变的量是什么？它必然出现在解里",
            "我把对称性'用掉了'吗？还是被对称的表象骗了？",
        ],
        "original_wisdom": [
            {"source": "Polya 1945 'How to Solve It'",
             "text": "Are there essential elements you have not used yet? Look for symmetries, invariants, conserved quantities."},
        ],
    },
    "S13": {  # 证伪优先 - Popper
        "one_sentence_orient": "对自己的答案保持持续警觉，主动寻找能推翻自己的证据",
        "trigger": "生成任何答案之前和之后都启动 — 不是某一步，是贯穿全程的气质",
        "attention_priors": [
            "什么观察能证明我错？如果什么都不能证明我错，我的答案没有信息量",
            "我是在寻找支持证据还是反驳证据？哪一种我更愿意找？",
            "我的答案有没有让我'舒服'的地方？那个舒服可能是 confirmation bias",
            "一个能解释一切的理论，解释不了任何东西",
        ],
        "original_wisdom": [
            {"source": "Popper 1959 'The Logic of Scientific Discovery'",
             "text": "Science must seek not verification but refutation. A theory is scientific only if it is falsifiable."},
            {"source": "Popper 1963 'Conjectures and Refutations'",
             "text": "A theory which is not refutable by any conceivable event is non-scientific. Irrefutability is not a virtue but a vice."},
        ],
    },
    "S14": {  # 边界条件 - Polya
        "one_sentence_orient": "对极端情况保持本能警觉 — 边界是假设最容易破产的地方",
        "trigger": "提出任何规则/答案后，自动问'在极端情况下它还成立吗'",
        "attention_priors": [
            "当输入 → 0、→ ∞、→ 空、→ 重复时，我的答案还成立吗？",
            "我的答案隐含'中间情况'的假设吗？如果遇到极端值我没考虑过的会怎样？",
            "边界上的失败往往暴露了中间的潜在失败 —— 别轻易把边界当'特例'",
        ],
        "original_wisdom": [
            {"source": "Polya 1945 'How to Solve It'",
             "text": "Consider limiting cases and extreme cases. What happens when a parameter takes its largest or smallest possible value?"},
        ],
    },
    "S18": {  # 抽象化/泛化 - Polya
        "one_sentence_orient": "追问'这个问题的骨架是什么' — 表面不同的问题可能同根",
        "trigger": "每次开始解题时，先花一秒跳到抽象层看骨架，再回到细节",
        "attention_priors": [
            "如果我把具体数字/名字/领域全换掉，剩下什么？",
            "这个骨架有名字吗？它属于哪一类问题？",
            "比原题更一般的版本反而可能更容易解 —— 因为细节干扰减少了",
        ],
        "original_wisdom": [
            {"source": "Polya 1945 'How to Solve It'",
             "text": "The more general problem may be easier to solve. This is the inventor's paradox."},
        ],
    },
    "S19": {  # 约束松弛 - Polya
        "one_sentence_orient": "当感觉'没出路'时，问自己'哪条约束是真必要的，哪条是我自己加的'",
        "trigger": "遇到 impossible 或 stuck 时，先检视约束结构，而非硬碰硬",
        "attention_priors": [
            "如果暂时放松约束 X，问题会变成什么样？能解吗？松弛解离真正解多远？",
            "哪些约束是问题硬约束，哪些是我的习惯/偏见带入的软约束？",
            "最后我当然要满足原约束，但先找个放松的可行解能给我方向",
        ],
        "original_wisdom": [
            {"source": "Polya 1945 'How to Solve It'",
             "text": "Keep only part of the condition. How far is the unknown then determined? In other words: drop one condition and see what happens."},
        ],
    },
    "S22": {  # 问题重构 - Polya
        "one_sentence_orient": "当解不出时，怀疑问题本身 — 不是答案错，而是'问得不对'",
        "trigger": "反复尝试都失败、所有策略都撞墙时启动 — 改变框架而非改变答案",
        "attention_priors": [
            "我假设了这个问题'应该'怎么解。这个假设是谁告诉我的？",
            "如果我把'问题'重新定义，原来的难点还存在吗？",
            "真正的问题可能在问题的问法里，而不在答案里",
        ],
        "original_wisdom": [
            {"source": "Polya 1945 'How to Solve It'",
             "text": "Restate the problem. Can you state the problem in your own words? Could you imagine a more accessible related problem?"},
            {"source": "Einstein (attributed)",
             "text": "If I had an hour to solve a problem, I'd spend 55 minutes thinking about the problem and 5 minutes thinking about solutions."},
        ],
    },
}


def rewrite(source_path: Path, out_path: Path, orient: dict):
    d = json.loads(source_path.read_text(encoding="utf-8"))

    # Preserve: id, name, aliases, category, source_references, applicability_conditions,
    #          relationships, historical_cases, failure_modes, metadata, version
    # Transform: description, remove operational_steps, add trigger/attention_priors/original_wisdom
    new_d = {
        "id": d["id"],
        "name": d["name"],
        "aliases": d.get("aliases", []),
        "category": d.get("category", ""),
        "form": "orientation",  # MARKER: this strategy is orientation-form
        "source_references": d.get("source_references", []),
        "description": {
            "one_sentence": orient["one_sentence_orient"],
            "background": d["description"].get("detailed", "")[:400],  # keep as context
            "intuitive_analogy": d["description"].get("intuitive_analogy", ""),
        },
        "trigger": orient["trigger"],
        "attention_priors": orient["attention_priors"],
        "original_wisdom": orient["original_wisdom"],
        "applicability_conditions": d.get("applicability_conditions", {}),
        "relationships_to_other_strategies": d.get("relationships_to_other_strategies", []),
        "historical_cases": d.get("historical_cases", []),
        "failure_modes": d.get("failure_modes", []),
        "metadata": {
            **(d.get("metadata", {})),
            "rewritten_at": "2026-04-22",
            "rewritten_to": "orientation_form",
        },
        "version": d.get("version", 1),
    }
    out_path.write_text(json.dumps(new_d, ensure_ascii=False, indent=2))
    return new_d


def main():
    written = []
    for f in sorted(KB_DIR.glob("S*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        sid = d["id"]
        if sid in ORIENTATION_DATA:
            out_path = OUT_DIR / f.name
            rewrite(f, out_path, ORIENTATION_DATA[sid])
            written.append(sid)
            print(f"  ✓ {sid} ({d['name']['zh']}): rewritten to {out_path.name}")
    print(f"\nWrote {len(written)} orientation files to {OUT_DIR}")
    print(f"Covered: {', '.join(written)}")


if __name__ == "__main__":
    main()
