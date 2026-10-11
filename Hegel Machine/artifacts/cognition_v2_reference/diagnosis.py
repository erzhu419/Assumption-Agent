"""Competing research questions inferred from observations and residuals."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Sequence

from .evaluation import predict
from .models import Observation, ResearchQuestion


def diagnose(program, observations: Sequence[Observation], fallback: float,
             risk_quantile: float, *, max_depth: int = 2) -> ResearchQuestion:
    residuals = [row.target - predict(program, row, fallback) for row in observations]
    bias = mean(residuals)
    spread = pstdev(residuals) if len(residuals) > 1 else 0.0
    mse = mean(value * value for value in residuals)
    evidence = {"mean_residual": bias, "residual_spread": spread,
                "squared_residual": mse, "program_size": float(program.size)}
    questions = []
    if mse > 0.002:
        questions.append(ResearchQuestion(
            "Can refitting the current observations explain this error?",
            evidence, ("refit",)))
        children = ()
        if max_depth > 1:
            children = (
                ResearchQuestion("Does history separate otherwise similar current inputs?",
                                 {"history_rows": float(sum(bool(r.history) for r in observations))},
                                 ("representation",)),
                ResearchQuestion("Can observable predicates be combined into a better relation?",
                                 {"observed_rows": float(len(observations))},
                                 ("relation", "composition")),
            )
        questions.append(ResearchQuestion(
            "Do the observation representation and relation need a joint change?",
            evidence, ("representation", "relation", "composition"), children))
    if risk_quantile != 0.5 and (spread > 0.01 or abs(bias) > 0.01):
        questions.append(ResearchQuestion(
            "Does the residual distribution require a different uncertainty model?",
            {"residual_spread": spread, "fixed_task_quantile": risk_quantile},
            ("robustification",)))
    if program.size > 3:
        questions.append(ResearchQuestion(
            "Can a smaller program preserve useful predictions and reduce repeated work?",
            {"program_size": float(program.size)}, ("idealization",)))
    return ResearchQuestion(
        "Which experiment or executable revision is worth the remaining budget?",
        evidence, tuple(dict.fromkeys(op for question in questions for op in question.operations)),
        tuple(questions) if max_depth > 0 else ())


def research_steps(question: ResearchQuestion, *, max_depth: int = 2):
    """Resolve bounded subquestions; synthesis results feed the parent comparison."""
    seen = set()

    def visit(node, depth):
        if depth < max_depth:
            for child in node.children:
                yield from visit(child, depth + 1)
        for operation in node.operations:
            if operation not in seen:
                seen.add(operation)
                yield operation, node.question

    yield from visit(question, 0)

