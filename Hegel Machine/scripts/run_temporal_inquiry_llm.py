#!/usr/bin/env python3
"""Development comparison on short-scope hypotheses and longer trajectories.

Uses the existing ChatGPT subscription and shared Python-agent loop. First run
--arms code_react --cases 2 to assess the task before a three-arm campaign.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.online_inquiry_llm import ARMS, InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from scripts.run_online_inquiry_llm import run_study, verify_codex_subscription


def laboratory(task):
    return TemporalInquiryLab(**{key: value for key, value in task.items()
        if key not in {"evaluation_seed", "task_id", "confirmation_seed"}})


def protocol(*, cases=2, arms=ARMS, config=None, model="gpt-6-luna"):
    if not 1 <= cases <= 6:
        raise ValueError("this development pilot supports one to six cases")
    if not arms or set(arms) - set(ARMS) or len(set(arms)) != len(arms):
        raise ValueError("select distinct supported knowledge conditions")
    chains = []
    for index in range(cases):
        # Adjacent cases share all physical parameters and short observations.
        # The only difference is an unobserved continuation past the source scope.
        seed = 80105007 + 1009 * (index // 2)
        mode = ("finite2", "recurrent")[index % 2]
        def task(stage):
            return {"task_id": f"temporal_{index:02d}_{stage}",
                "lab": {"task_index": stage, "seed": seed, "apparatus_seed": seed,
                        "instrument_mode": mode},
                "evaluation_seed": seed + 1000003 + 1009 * stage}
        chains.append({"chain_id": f"temporal_dev_{index:02d}", "source": task(0),
            "consumers": [{**task(1), "condition": "same_physics_longer_horizon"}]})
    limits = config or InquiryLLMConfig(observation_budget=288, confirmation_rows=48,
        max_model_calls=24, token_budget=600000, execution_seconds=120.)
    return {"study": "temporal_scope_expansion_development", "profile": "development",
        "model": model, "arms": list(arms), "cases": chains, "limits": asdict(limits),
        "provider": {"backend": "codex_cli", "auth_mode": "ChatGPT_subscription",
                     "reasoning_effort": "medium", "timeout_seconds": 180},
        "evaluation": {"length": 256, "mse_threshold": 1e-10,
            "prediction_inputs": "current and previous controls only, reset per trajectory"},
        "memory_conditions": {"code_react": "task-local Python, no inherited knowledge",
            "textual_memory": "source-authored textual lesson",
            "hegel": "source-authored executable hypotheses and research programs"},
        "question": "Can an agent test and revise the temporal scope of a learned response?",
        "source_scope": "At most three steps; both physical families agree on every legal source probe.",
        "target_scope": "At most 24 steps; same apparatus physics as source.",
        "source_freeze": "Only source knowledge transfers; no target outcomes cross cases or arms.",
        "primary_outcome": "fresh-confirmed control-history prediction on independent trajectories",
        "costs": "All trajectory rows and optional meter costs recorded; 48 observation rows reserved by the agent for fresh confirmation.",
        "interpretation": "Two paired worlds are development cases, not independent population samples. Neither auxiliary measurements nor recursive questions are compulsory. Direct Python system identification is a valid solution.",
        "limitations": "Finite synthetic apparatus; no evidence yet for comparative benefit, recursive causation or reuse of a target-learned method."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    plan = protocol(cases=args.cases, arms=args.arms)
    plan["provider"]["auth_mode"] = verify_codex_subscription()
    def provider(path):
        return CodexCodeProvider(path, model=plan["model"], reasoning_effort="medium",
                                 timeout_seconds=plan["provider"]["timeout_seconds"])
    summary = run_study(plan, args.output, provider_factory=provider,
                        lab_factory=laboratory, workers=args.workers)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
