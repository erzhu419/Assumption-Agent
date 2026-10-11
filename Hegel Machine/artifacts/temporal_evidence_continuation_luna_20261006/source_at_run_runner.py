#!/usr/bin/env python3
"""Common-LLM development pilot with shared, individually selected fit operators."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.online_inquiry_llm import ARMS, InquiryLLMConfig
from hegel_machine.cognition.temporal_operator_agent import OPERATOR_INTERFACE, TemporalOperatorAgent
from scripts.run_online_inquiry_llm import run_study, verify_codex_subscription
from scripts.run_temporal_inquiry_llm import laboratory as plain_laboratory, protocol as base_protocol


def laboratory(task):
    lab = plain_laboratory(task)
    lab.task_interface += "\n" + OPERATOR_INTERFACE
    return lab


def protocol(*, cases=2, arms=ARMS, apparatus_seed=87106007):
    limits = InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
        max_model_calls=24, token_budget=600000, execution_seconds=120., compute_budget=100000)
    plan = base_protocol(cases=cases, arms=arms, config=limits, model="gpt-6-luna")
    for index, chain in enumerate(plan["cases"]):
        seed = apparatus_seed + 1009 * (index // 2)
        chain["chain_id"] = f"temporal_operators_dev_{index:02d}"
        for stage, task in enumerate([chain["source"], *chain["consumers"]]):
            task["lab"].update(seed=seed, apparatus_seed=seed)
            task["evaluation_seed"] = seed + 1000003 + 1009 * stage
    plan.update(study="same_llm_temporal_shared_operators_development",
        question="Can the same LLM select and reuse shared representation and instrument research operations?",
        operator_interface=OPERATOR_INTERFACE,
        shared_operators=["history", "instrument", "response", "disagreement", "assess_instrument"],
        operator_authorship="Developer-authored bounded fit and planning methods available equally to all arms.",
        evidence_continuation="The agent may explicitly compose shared operators in a query's after_observation "
            "list, including in its authored select_actions program. Newly acquired laboratory record indices "
            "are bound at execution; no continuation is supplied by the researcher.",
        decision_ownership="The LLM selects operations, purchases measurements, chooses a returned predictor, "
            "writes any retained procedure and submits it for fresh confirmation. No operator runs a complete research policy.",
        terminal_order="All arms and cases stop training before terminal labels are generated.",
        compute_scope="Host operator work and candidate scoring consume compute units; all operator and ordinary Python "
            "execution consumes the common wall-time allowance. These units are not hardware FLOPs.",
        limitations="Development pilot within fixed operator families. Success is operator-assisted LLM research, "
            "not invention of the fitting algorithms. Only source memory transfers; target-learned methods are not reused. "
            "Previous raw-Python LLM cohorts remain separate.")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=87106007, help="First paired apparatus seed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = protocol(cases=args.cases, arms=args.arms, apparatus_seed=args.seed)
    plan["provider"]["auth_mode"] = verify_codex_subscription()
    def provider(path):
        return CodexCodeProvider(path, model=plan["model"], reasoning_effort="medium",
                                 timeout_seconds=plan["provider"]["timeout_seconds"])
    summary = run_study(plan, args.output, provider_factory=provider, lab_factory=laboratory,
                        workers=args.workers, agent_class=TemporalOperatorAgent)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
