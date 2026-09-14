"""Run ordinary Python and Hegel research against the same public apparatus."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

from .code_agent import CodeAgent, CodeAgentConfig, CodexCodeProvider, evaluate_python_model
from .evaluation import loss
from .language import Program
from .measurement_lab import make_lab
from .measurement_research import MeasurementResearchAgent
from .models import ResearchConfig


def run_code_comparison(workspace, *, tasks=(0, 1, 2), config=None, code_config=None,
                        measurement_budget=512, backend="process", provider_factory=None,
                        code_results=(), hegel_results=(), on_result=None):
    """Compare arms; supplied completed results avoid repeating expensive runs.

    Both arms receive identical controls, optional instruments, observation and
    measurement caps. Hegel charges search candidate-rows; the Python arm charges
    scored candidate-rows and separately records arbitrary Python execution time
    and model tokens. Neither arm receives current evaluation targets.
    """
    config = config or ResearchConfig(observation_budget=128, compute_budget=100000, max_rounds=3, seed=313)
    code_config = code_config or CodeAgentConfig(
        observation_budget=config.observation_budget, measurement_budget=measurement_budget,
        compute_budget=config.compute_budget, max_model_calls=12, token_budget=120000)
    if (code_config.observation_budget, code_config.measurement_budget, code_config.compute_budget) != (
            config.observation_budget, measurement_budget, config.compute_budget):
        raise ValueError("comparison arms need the same observation, measurement and candidate-row caps")
    previous_code = {row["task_index"]: row for row in code_results}
    previous_hegel = {row["task_index"]: row for row in hegel_results}
    report = {"experiment": "ordinary_code_and_hegel_comparison", "config": asdict(config),
              "code_config": asdict(code_config), "backend": backend,
              "resource_contract": {
                  "observation_budget": config.observation_budget, "measurement_budget": measurement_budget,
                  "candidate_row_budget": config.compute_budget,
                  "python_execution_limit_seconds": code_config.execution_seconds,
                  "model_calls_limit": code_config.max_model_calls, "model_token_limit": code_config.token_budget,
                  "model_token_limit_kind": "reported usage between calls; the last call can exceed the threshold",
                  "computation": "Hegel search candidate-rows; Python scored candidate-rows plus separately measured subprocess wall time and LLM tokens",
                  "history": "Both arms receive the same past features and past targets; current evaluation targets are withheld."},
              "results": []}
    for task_index in tasks:
        seed = config.seed + task_index * 10007
        task_workspace = Path(workspace) / f"task_{task_index}"
        if task_index in previous_code:
            code = deepcopy(previous_code[task_index])
        else:
            provider = provider_factory(task_workspace) if provider_factory else CodexCodeProvider(task_workspace)
            with make_lab(task_index, seed, backend=backend) as laboratory:
                code = CodeAgent(provider, task_workspace, code_config).run(laboratory)
                code.update(task_index=task_index, seed=seed)
                if code["model"]:
                    rows = laboratory.evaluation_rows(code["selected_channels"], seed=seed + 1000003,
                                                      length=64, trajectories=4)
                    evaluation = evaluate_python_model(task_workspace, code["model"], rows)
                    evaluation.pop("predictions", None)
                    code["evaluation"] = {**evaluation, "rows": len(rows), "compute_cost": len(rows)}
        tokens = code["costs"]["reported_tokens"]
        code["token_budget_exceeded"] = tokens > code["config"]["token_budget"] if tokens is not None else None
        if task_index in previous_hegel:
            hegel = deepcopy(previous_hegel[task_index])
        else:
            with make_lab(task_index, seed, backend=backend) as laboratory:
                hegel = MeasurementResearchAgent(replace(config, seed=seed), measurement_budget=measurement_budget).run(laboratory)
                rows = laboratory.evaluation_rows(hegel["selected_channels"], seed=seed + 1000003, length=64, trajectories=4)
                program = Program.from_dict(hegel["episode"]["program"])
                hegel.update(task_index=task_index, seed=seed,
                             evaluation={"mse": loss(program, rows, hegel["episode"]["fallback"], .5),
                                         "rows": len(rows), "compute_cost": len(rows)})
        report["results"].append({"task_index": task_index, "seed": seed, "code_agent": code, "hegel": hegel})
        if on_result is not None:
            on_result(deepcopy(report))
    return report
