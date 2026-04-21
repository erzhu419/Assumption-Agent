"""
Baseline harness: direct solve via single LLM call.

This is the starting point. All proposer-generated harnesses compete against
it, and their win rate over baseline is the reward signal.
"""


BASELINE_PROMPT = """你是一位严谨的问题解决者。针对下面给出的问题，给出一个清晰、结构化的解决方案。

要求：
1. 先简要重述问题核心
2. 给出你的分析和推理步骤
3. 给出最终建议/解答
4. 语言精炼，不超过 400 字

## 问题
{problem}
"""


def solve(problem: str, ctx) -> str:
    return ctx.generate(
        BASELINE_PROMPT.format(problem=problem),
        max_tokens=800, temperature=0.3,
    )
