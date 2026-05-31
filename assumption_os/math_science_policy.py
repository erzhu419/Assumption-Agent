"""Math/science bypass prompts with intent-aware routing."""

from __future__ import annotations


MATH_FORMAL_PROMPT = """你是一位严谨的问题解决者。针对下面的数学问题，给出清晰、完整的解答。

## 计算自检（脑中执行）
- 审题：变量范围、约束、定义都明确了吗？
- 过程：每步代数/逻辑变换是可逆或等价的吗？
- 自检：答案代入原始条件是否自洽？极端 case 下是否合理？

## 要求
- 如果题目给出了明确命题、表达式或待求量，直接证明/推导/计算
- 不要用泛泛的方法论替代数学步骤
- 结构清晰，每步说明理由
- 不超过 650 字

## 问题
{problem}
"""


MATH_RESEARCH_BRIDGE_PROMPT = """你是一位懂数学研究策略的严谨数学家。下面的问题不是要你证明一个已完整给出的命题，而是要判断研究方向、统一视角或证明/反例策略。

## 关键要求
- 不要停在“题目未给出具体公式，所以无法证明”；可以简短说明不确定性，但重点必须是给出最可能的数学桥梁和下一步研究路径
- 若问题是在找统一视角：明确点名可能的理论桥梁、典型定理/工具、对象如何互相转译
- 若问题是在证明还是找反例之间决策：给出有限时间的证伪计划、反例搜索对象、如何把失败反例转成引理
- 保持数学严谨，但回答要具体、可执行、有研究判断
- 不超过 700 字

## 问题
{problem}
"""


SCIENCE_MECHANISM_PROMPT = """你是一位严谨的问题解决者。针对下面的科学问题，给出清晰、结构化的解答。

## 科学自检（脑中执行）
- 量纲与单位：每个量的单位都对得上吗？
- 机制 vs 描述：答案给出的是机制解释，还是只描述了现象？
- 可证伪性：如果假设错了，会产生什么不同观察？

## 要求
- 直接给机制分析和结论
- 每一论断都有推理支撑
- 不超过 650 字

## 问题
{problem}
"""


SCIENCE_DECISION_PROMPT = """你是一位严谨但务实的科研决策顾问。下面的问题包含科学判断，也包含发表、毕业、资源排队、协作或项目风险约束。

## 关键要求
- 先判断当前科学主线是否已经闭环，哪些新实验是必要验证，哪些只是高收益后续探索
- 明确时间、设备、发表优先权、毕业/项目风险和沟通对象
- 给出可执行行动方案：立即做什么、论文/报告怎么写、后续实验如何安排、如何和导师/合作者沟通
- 保留科学可证伪性：说明额外实验若成立/不成立分别意味着什么
- 不超过 700 字

## 问题
{problem}
"""


MATH_RESEARCH_CUES = (
    "数学家",
    "研究生",
    "导师",
    "论文",
    "直觉",
    "统一",
    "联系",
    "框架",
    "视角",
    "证明",
    "反例",
    "定理可能",
    "投入",
    "草稿",
)

SCIENCE_DECISION_CUES = (
    "博士",
    "毕业",
    "合同",
    "投稿",
    "论文",
    "导师",
    "合作者",
    "排队",
    "设备",
    "项目",
    "沟通",
    "发表",
    "申请",
    "到期",
)


def route_math_science_problem(domain: str, problem: str) -> str:
    """Return a stable bypass route for math/science prompts."""

    if domain == "mathematics":
        return "math_research_bridge" if _contains_any(problem, MATH_RESEARCH_CUES) else "math_formal"
    if domain == "science":
        return "science_decision" if _contains_any(problem, SCIENCE_DECISION_CUES) else "science_mechanism"
    return "none"


def format_math_science_prompt(domain: str, problem: str) -> tuple[str, int, str]:
    route = route_math_science_problem(domain, problem)
    if route == "math_research_bridge":
        return MATH_RESEARCH_BRIDGE_PROMPT.format(problem=problem), 1200, route
    if route == "math_formal":
        return MATH_FORMAL_PROMPT.format(problem=problem), 1200, route
    if route == "science_decision":
        return SCIENCE_DECISION_PROMPT.format(problem=problem), 1100, route
    if route == "science_mechanism":
        return SCIENCE_MECHANISM_PROMPT.format(problem=problem), 1000, route
    raise ValueError(f"unsupported math/science domain: {domain}")


def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(cue in text for cue in cues)
