#!/usr/bin/env python3
"""Practice evidence-dependent procedure branches, then freeze for transfer."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_conditional_method import ConditionalMethodPracticeAgent
from hegel_machine.cognition.temporal_method_learning import load_training_experience, method_memory_pair
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_temporal_method_learning import add_costs, run_transfer_phase
from scripts.run_temporal_method_practice import protocol as practice_protocol
from scripts.run_temporal_operator_llm import laboratory


def protocol(practice_seed=99706007, transfer_seed=99806007):
    plan = practice_protocol(practice_seed, transfer_seed)
    plan.update(study="llm_temporal_conditional_method_practice_and_transfer",
        learning="Begin with the supplied LLM-authored practiced method and execute it on a new training "
            "apparatus. Revise the procedure using paid observations and operator feedback. The final source "
            "must execute repeatedly and make different experiment or analysis decisions as its evidence or "
            "operator feedback changes. An edit between invocations does not establish conditional behavior "
            "of one fixed procedure. No sampling, fitting or repair strategy is supplied by the host.",
        program_state="Before invocation, research_state.json exposes current paid operator results, limits "
            "and costs. Each operator entry has the same flat arguments/operator/outcome fields as query "
            "continuation feedback, plus its chronological index. The inherited memory remains a snapshot. "
            "A procedure may return empty actions and "
            "a nonempty continuation to analyze existing evidence without buying more rows; those operations "
            "still consume compute and execution time.",
        publication="The repeated final-source procedure must demonstrate substantive evidence-dependent "
            "experiment or operator choices and actually export the submitted predictor through its own "
            "continuation. That predictor must pass ordinary fresh confirmation. Source edits, observation "
            "index substitutions and repeated unchanged analysis alone do not establish this behavior.",
        transfer="Freeze the final conditional procedure before four independent transfers. Restore the "
            "original donor predictive snapshot in both conditions; practice-fitted instruments, predictors, "
            "operator state, auxiliary files and observations do not transfer. Both conditions receive the "
            "same learned verbal lesson. Only the final procedure and its manifest entry are retained or deleted.",
        cost_scope="Take common_learning_costs.combined from the seed study summary as the complete prior "
            "learning lineage, including original donor learning, consolidation and earlier paid practice. "
            "Add only this new conditional practice once. Prior transfer-evaluation work does not become "
            "new learning input or get added to that lineage. Count the resulting common cost once per "
            "retained/deleted condition, separately from its new transfer costs.",
        limitations="Development study of one further training apparatus and one paired transfer apparatus "
            "within fixed operator families. The LLM is explicitly tasked with conditional method practice. "
            "Execution traces establish which branches actually ran; independent retained/deleted comparison "
            "is needed to assess the procedure's added transfer value.")
    return plan


def _load(source):
    return json.loads(Path(source).read_text()) if isinstance(source, (str, Path)) else source


def _source_label(source, label):
    return str(source) if isinstance(source, (str, Path)) else label


def _save_sources(output):
    paths = ["scripts/run_temporal_conditional_method.py", "scripts/run_temporal_method_learning.py",
             "src/hegel_machine/cognition/checkpoint_storage.py",
             "scripts/run_temporal_method_practice.py",
             "scripts/run_temporal_operator_llm.py",
             "src/hegel_machine/cognition/temporal_conditional_method.py",
             "src/hegel_machine/cognition/temporal_method_practice.py",
             "src/hegel_machine/cognition/temporal_method_learning.py",
             "src/hegel_machine/cognition/temporal_operator_agent.py",
             "src/hegel_machine/cognition/online_inquiry_llm.py"]
    for name in paths:
        destination = output / "source_at_run" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)


def run_conditional_study(experience, seed_method, seed_study_summary, output, *, provider_factory,
                          practice_seed=99706007, transfer_seed=99806007, workers=2, auth_mode=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("use a new output directory; preserve earlier results")
    raw, seed, seed_summary = _load(experience), _load(seed_method), _load(seed_study_summary)
    paid = load_training_experience(raw)
    prior_learning = seed_summary["common_learning_costs"]["combined"]
    initial_memory, _ = method_memory_pair(paid["outgoing_knowledge"], seed)
    plan = protocol(practice_seed, transfer_seed)
    if auth_mode:
        plan["provider"]["auth_mode"] = auth_mode
    plan.update(training_experience=_source_label(experience, "provided_training_record"),
                seed_method_source=_source_label(seed_method, "provided_seed_method"),
                seed_study_summary=_source_label(seed_study_summary, "provided_seed_study_summary"))
    write_json(output / "protocol.json", plan)
    _save_sources(output)
    learner_path = output / "practice"
    agent = ConditionalMethodPracticeAgent(provider_factory(learner_path), learner_path,
        knowledge=initial_memory, config=InquiryLLMConfig(**plan["practice_limits"]),
        confirmation_seed=plan["practice_task"]["confirmation_seed"])
    learned = agent.run(laboratory(plan["practice_task"]),
        on_checkpoint=lambda data: write_json(learner_path / "active.json", data))
    write_json(learner_path / "result.json", learned)
    common = {"seed_study_learning": prior_learning, "conditional_practice": learned["costs"],
              "combined": add_costs([prior_learning, learned["costs"]])}
    return run_transfer_phase(plan, output, paid["outgoing_knowledge"], learned, common,
                              provider_factory=provider_factory, workers=workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--seed-method", type=Path, required=True)
    parser.add_argument("--seed-study-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--practice-seed", type=int, default=99706007)
    parser.add_argument("--transfer-seed", type=int, default=99806007)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    auth_mode = verify_codex_subscription()

    def provider(path):
        return CodexCodeProvider(path, model="gpt-6-luna", reasoning_effort="medium", timeout_seconds=180)

    print(json.dumps(run_conditional_study(args.experience, args.seed_method, args.seed_study_summary,
        args.output, provider_factory=provider, practice_seed=args.practice_seed,
        transfer_seed=args.transfer_seed, workers=args.workers, auth_mode=auth_mode), indent=2))


if __name__ == "__main__":
    main()
