"""Commands for executable cognition research and matched-resource comparisons."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .experiments import compare, run_research, run_sequence
from .memory import KnowledgeLibrary
from .models import ResearchConfig


def add_parser(subparsers):
    parser = subparsers.add_parser("cognition", help="Run discovery, revision and cross-task knowledge experiments")
    commands = parser.add_subparsers(dest="cognition_command", required=True)
    for name in ("run", "compare", "research"):
        descriptions = {"run": "Run a task sequence with persistent knowledge",
                        "compare": "Compare discovery, planning, memory and observation strategies",
                        "research": "Run paired multi-seed comparisons and summarize uncertainty"}
        command = commands.add_parser(name, help=descriptions[name])
        command.add_argument("--output", type=Path)
        command.add_argument("--config", type=Path)
        command.add_argument("--seed", type=int)
        command.add_argument("--observation-budget", type=int)
        command.add_argument("--compute-budget", type=int)
        command.add_argument("--rounds", type=int)
        command.add_argument("--partition", choices=("development", "confirmation"))
        command.add_argument("--llm-calls-per-task", type=int)
        command.add_argument("--llm-token-budget", type=int)
        command.add_argument("--llm-timeout-seconds", type=int)
        if name == "run":
            command.add_argument("--memory", type=Path, help="Load and save an executable concept library")
            command.add_argument("--proposer", choices=("search", "codex", "hybrid"))
        else:
            command.add_argument("--include-llm", action="store_true", help="Also run the configured Codex model on every episode")
            command.add_argument("--systems", nargs="+", help="Limit the comparison to named systems")
        if name == "compare":
            command.add_argument("--concept-id", help="Specific learned concept to remove before transfer")
        if name == "research":
            command.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 19])
    return parser


def dispatch(args):
    values = asdict(ResearchConfig())
    if args.config:
        values.update(json.loads(args.config.read_text(encoding="utf-8")))
    for field, value in (("seed", args.seed), ("observation_budget", args.observation_budget),
                         ("compute_budget", args.compute_budget), ("max_rounds", args.rounds),
                         ("partition", args.partition), ("proposer", getattr(args, "proposer", None)),
                         ("llm_calls_per_task", args.llm_calls_per_task),
                         ("llm_token_budget", args.llm_token_budget),
                         ("llm_timeout_seconds", args.llm_timeout_seconds)):
        if value is not None:
            values[field] = value
    config = ResearchConfig(**values)
    if args.cognition_command == "compare":
        report = compare(config, include_llm=args.include_llm, concept_id=args.concept_id,
                         systems=tuple(args.systems) if args.systems else None)
        summary = {key: value for key, value in report.items() if key != "runs"}
    elif args.cognition_command == "research":
        report = run_research(config, args.seeds, config.partition, include_llm=args.include_llm,
                              systems=tuple(args.systems) if args.systems else None)
        summary = {key: value for key, value in report.items() if key != "seed_results"}
    else:
        library = None
        if args.memory and args.memory.exists():
            library = KnowledgeLibrary.from_dict(json.loads(args.memory.read_text(encoding="utf-8")))
        report = run_sequence(config, library=library)
        if args.memory:
            args.memory.parent.mkdir(parents=True, exist_ok=True)
            args.memory.write_text(json.dumps(report["knowledge"], indent=2) + "\n", encoding="utf-8")
        summary = {key: value for key, value in report.items() if key not in {"episodes", "knowledge"}}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary_path = args.output.with_name(args.output.stem + "_summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["output"] = str(args.output.resolve())
        summary["summary_output"] = str(summary_path.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hegel-machine")
    add_parser(parser.add_subparsers(dest="command", required=True))
    return dispatch(parser.parse_args(argv))
