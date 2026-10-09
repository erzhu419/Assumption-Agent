#!/usr/bin/env python3
"""Practice and revise a method with paid feedback, then freeze it for transfer."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import load_training_experience, method_memory_pair
from hegel_machine.cognition.temporal_method_practice import MethodPracticeAgent
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_temporal_method_learning import add_costs, protocol as transfer_protocol, run_transfer_phase
from scripts.run_temporal_operator_llm import laboratory


def protocol(practice_seed=99306007, transfer_seed=99406007):
    if practice_seed == transfer_seed:
        raise ValueError("practice and transfer must use different apparatuses")
    plan = transfer_protocol(transfer_seed)
    plan.pop("consolidation_limits")
    plan.update(study="llm_temporal_method_practice_and_transfer",
        practice_task={"task_index": 2, "seed": practice_seed, "apparatus_seed": practice_seed,
                       "instrument_mode": "recurrent", "confirmation_seed": practice_seed + 7919},
        practice_limits=asdict(InquiryLLMConfig(observation_budget=768, confirmation_rows=48,
            max_model_calls=32, token_budget=1000000, execution_seconds=120., compute_budget=100000)),
        learning="Begin with the earlier LLM-authored acquisition-only program. Execute it on a training "
            "apparatus, return actual paid outcomes, revise it, and execute the revised procedure. "
            "Practice allows 768 observation rows to accommodate repeated method revision; transfer keeps "
            "the same 240-row allowance in both conditions. No host-written sampling or fitting policy is supplied.",
        support_diagnostics="Response fitting also reports the consumed instrument's residuals on current "
            "already-paid meter readings, with scoring charged. These are consistency diagnostics, possibly "
            "including fitting evidence, not fresh confirmation. They do not mutate the instrument, "
            "remove candidates, acquire evidence or choose a repair; all knowledge conditions receive them.",
        publication="A revised current method must actually export the submitted predictor through its "
            "own post-acquisition steps; that predictor must pass ordinary fresh confirmation. "
            "This tests a practiced method, not an independently validated transferable procedure.",
        transfer="Only the practiced method and common verbal lesson carry over. Restore the original "
            "pre-practice predictive snapshot in both conditions; do not transfer practice-fitted instruments "
            "or predictors. Four independent transfer tasks follow method freezing.",
        cost_scope="Include original learning, seed-method consolidation and paid practice as common learning "
            "costs once per condition, separately from transfer.",
        limitations="One training apparatus and one paired transfer apparatus in fixed operator families. "
            "The model is explicitly tasked with method practice; meaningful program revision and independent "
            "transfer benefit require the actual execution traces and comparisons.")
    return plan


def run_practice_study(experience, seed_method, output, *, provider_factory,
                       practice_seed=99306007, transfer_seed=99406007, workers=2, auth_mode=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("use a new output directory; preserve earlier results")
    plan = protocol(practice_seed, transfer_seed)
    if auth_mode:
        plan["provider"]["auth_mode"] = auth_mode
    plan.update(training_experience=str(experience) if isinstance(experience, (str, Path)) else "provided_training_record",
                seed_method_source=str(seed_method) if isinstance(seed_method, (str, Path)) else "provided_seed_method")
    write_json(output / "protocol.json", plan)
    raw = json.loads(Path(experience).read_text()) if isinstance(experience, (str, Path)) else experience
    seed = json.loads(Path(seed_method).read_text()) if isinstance(seed_method, (str, Path)) else seed_method
    paid = load_training_experience(raw)
    initial_memory, _ = method_memory_pair(paid["outgoing_knowledge"], seed)
    learner_path = output / "practice"
    agent = MethodPracticeAgent(provider_factory(learner_path), learner_path, knowledge=initial_memory,
        config=InquiryLLMConfig(**plan["practice_limits"]),
        confirmation_seed=plan["practice_task"]["confirmation_seed"])
    learned = agent.run(laboratory(plan["practice_task"]),
        on_checkpoint=lambda data: write_json(learner_path / "active.json", data))
    write_json(learner_path / "result.json", learned)
    prior_costs = add_costs(task["costs"] for task in raw.get("tasks", [raw]))
    common = {"prior": prior_costs, "seed_method_learning": seed["costs"], "practice": learned["costs"],
              "combined": add_costs([prior_costs, seed["costs"], learned["costs"]])}
    # The practiced numerical models stay in the practice workspace. Both
    # consumers start with the exact same original predictive knowledge.
    return run_transfer_phase(plan, output, paid["outgoing_knowledge"], learned, common,
                              provider_factory=provider_factory, workers=workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--seed-method", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--practice-seed", type=int, default=99306007)
    parser.add_argument("--transfer-seed", type=int, default=99406007)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    auth_mode = verify_codex_subscription()
    def provider(path):
        return CodexCodeProvider(path, model="gpt-6-luna", reasoning_effort="medium", timeout_seconds=180)
    print(json.dumps(run_practice_study(args.experience, args.seed_method, args.output,
        provider_factory=provider, practice_seed=args.practice_seed, transfer_seed=args.transfer_seed,
        workers=args.workers, auth_mode=auth_mode), indent=2))


if __name__ == "__main__":
    main()
