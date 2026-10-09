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
    historical = commands.add_parser("historical", help="Use current implementations of the 181 historical versions")
    historical.add_argument("--version", help="Historical version, for example 0.93 or 0.181.0")
    historical.add_argument("--group", help="Capability group from the historical catalogue")
    historical.add_argument("--query", help="Find a problem or implementation using whitespace-separated terms")
    historical.add_argument("--compact", action="store_true", help="Show concise history entries with current reuse routes")
    historical.add_argument("--execute", nargs=argparse.REMAINDER,
                            help="Run the selected capability's native command and its task arguments")
    growth = commands.add_parser("growth", help="Run joint explanation, measurement and research-method growth")
    growth.add_argument("--checkpoint", type=Path, required=True)
    growth.add_argument("--tasks", nargs="+", type=int, choices=range(8), default=list(range(8)))
    growth.add_argument("--seed", type=int, default=8601)
    growth.add_argument("--observation-budget", type=int, default=384)
    growth.add_argument("--compute-budget", type=int, default=200000)
    growth.add_argument("--measurement-budget", type=int, default=1024)
    growth.add_argument("--rounds", type=int, default=6)
    for name in ("run", "compare", "research", "session"):
        descriptions = {"run": "Run a task sequence with persistent knowledge",
                        "compare": "Compare discovery, planning, memory and observation strategies",
                        "research": "Run paired multi-seed comparisons and summarize uncertainty",
                        "session": "Continue a whole learning session with shared resources and unresolved-task revisits"}
        command = commands.add_parser(name, help=descriptions[name])
        command.add_argument("--output", type=Path)
        command.add_argument("--config", type=Path)
        command.add_argument("--seed", type=int)
        command.add_argument("--observation-budget", type=int)
        command.add_argument("--compute-budget", type=int)
        command.add_argument("--rounds", type=int)
        command.add_argument("--partition", choices=("development", "confirmation", "forward", "continual", "stateful"))
        command.add_argument("--joint-search", choices=("beam", "greedy"))
        command.add_argument("--experiment-policy", choices=("task_value", "disagreement"))
        command.add_argument("--selection-policy", choices=("prospective", "retrospective"))
        command.add_argument("--llm-calls-per-task", type=int)
        command.add_argument("--llm-token-budget", type=int)
        command.add_argument("--llm-timeout-seconds", type=int)
        if name in {"run", "session"}:
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
        if name == "session":
            command.add_argument("--checkpoint", type=Path, required=True, help="Save the session, or continue it if this file exists")
            command.add_argument("--partitions", nargs="+", choices=("development", "confirmation", "forward", "continual", "stateful"),
                                 help="Ordered task sources for the complete session; defaults to --partition")
            command.add_argument("--total-observation-budget", type=int)
            command.add_argument("--total-compute-budget", type=int)
            command.add_argument("--max-visits", type=int, help="Maximum visits per task, initially 2")
            command.add_argument("--steps", type=int, help="Pause after this many further task visits")
    lab = commands.add_parser("lab", help="Learn useful instruments and continuous process models across tasks")
    lab.add_argument("--checkpoint", type=Path, required=True, help="Save completed tasks, active research and the shared concept library")
    lab.add_argument("--resume", action="store_true", help="Continue the saved task sequence without repeating completed tasks")
    lab.add_argument("--backend", choices=("process", "local"), help="Apparatus execution backend; defaults to process")
    lab.add_argument("--tasks", nargs="+", type=int, choices=(0, 1, 2, 3), help="Ordered apparatus tasks; defaults to 0 1 2 3")
    lab.add_argument("--strategy", choices=("learned", "controls_only", "all"), help="Instrument selection strategy; defaults to learned")
    lab.add_argument("--steps", type=int, help="Pause after this many further completed tasks")
    lab.add_argument("--research-steps", type=int, help="Pause an apparatus task after this many instrument or research steps")
    lab.add_argument("--observation-budget", type=int, help="Per-task observation rows; defaults to 128")
    lab.add_argument("--compute-budget", type=int, help="Per-task candidate-row computation units; defaults to 100000")
    lab.add_argument("--measurement-budget", type=int, help="Per-task instrument units; defaults to 512")
    lab.add_argument("--rounds", type=int, help="Research rounds; defaults to 3")
    lab.add_argument("--seed", type=int, help="Research seed; defaults to 313")
    lab.add_argument("--evaluation-length", type=int, help="Independent evaluation trajectory length; defaults to 64")
    lab.add_argument("--evaluation-trajectories", type=int, help="Independent evaluation trajectories; defaults to 4")
    lab.add_argument("--output", type=Path, help="Also save the full report and a compact summary")
    structure = commands.add_parser("structure", help="Discover and test object, event, scale and conservation models")
    structure.add_argument("--readout", choices=("total", "focus", "window", "event"), default="focus")
    structure.add_argument("--variant", choices=("exchange", "sink", "anchored", "nonlocal"), default="exchange")
    structure.add_argument("--seed", type=int, default=71)
    structure.add_argument("--output", type=Path, required=True)
    structure.add_argument("--memory", type=Path)
    mechanism = commands.add_parser("mechanism", help="Discover projections, modules, dynamics, graphs and response regimes")
    mechanism.add_argument("--family", choices=("low_rank", "modules", "dynamics", "topology", "order"), default="low_rank")
    mechanism.add_argument("--variant", default="regular")
    mechanism.add_argument("--seed", type=int, default=71)
    mechanism.add_argument("--output", type=Path, required=True)
    mechanism.add_argument("--memory", type=Path)
    simulator = commands.add_parser("simulator", help="Revise an experiment simulator from measured prediction errors")
    simulator.add_argument("--checkpoint", type=Path, required=True)
    simulator.add_argument("--steps", type=int)
    simulator.add_argument("--memory", type=Path)
    prior = commands.add_parser("prior", help="Propose, test and recursively revise executable world priors")
    prior.add_argument("--checkpoint", type=Path, required=True)
    prior.add_argument("--request", type=Path, help="Initial source and research config; resume uses the saved request")
    prior.add_argument("--memory", type=Path, help="Import or export the shared executable knowledge library")
    prior.add_argument("--steps", type=int, help="Pause after this many further research tasks and their requested deployments")
    prior.add_argument("--research-steps", type=int, help="Pause after this many hypothesis or experiment steps per task")
    prior.add_argument("--evaluate", action="store_true", help="Measure frozen programs after the sequence finishes")
    prior.add_argument("--optimize", action="store_true", help="Deploy each SC-KG task with paid data and retain revisions before researching the next task")
    prior.add_argument("--optimizer-config", type=Path, help="SC-KG settings including total budget N; saved for continuation")
    prior.add_argument("--optimizer-steps", type=int, help="Pause after this many further optimizer tasks")
    frontier = commands.add_parser("frontier", help="Continue an integrated research lifecycle with shared knowledge")
    frontier.add_argument("--checkpoint", type=Path, required=True)
    frontier.add_argument("--steps", type=int, help="Pause after this many further research stages")
    frontier.add_argument("--seed", type=int, default=503)
    frontier.add_argument("--code-comparison", type=Path, help="Attach an actual completed ordinary-code comparison report")
    agenda = commands.add_parser("agenda", help="Choose dynamically among task, instrument and object research questions")
    agenda.add_argument("--checkpoint", type=Path, required=True)
    agenda.add_argument("--request", type=Path, help="JSON shared limits, new questions and delayed experiments")
    agenda.add_argument("--steps", type=int, help="Pause after this many actual research steps")
    agenda_demo = commands.add_parser("agenda-demo", help="Run a mixed research agenda through three real CLI processes")
    agenda_demo.add_argument("--checkpoint", type=Path, required=True)
    agenda_demo.add_argument("--seed", type=int, default=71)
    experiment = commands.add_parser("experiment", help="Submit or collect delayed apparatus feedback for a saved research session")
    experiment.add_argument("--checkpoint", type=Path, required=True)
    experiment.add_argument("--request", type=Path, help="JSON task index, probe, optional source, channels and delay_seconds")
    experiment.add_argument("--directory", type=Path, help="Directory in which the apparatus delivers results")
    code = commands.add_parser("code-compare", help="Run a complete ordinary Python agent and the cognitive agent on the same apparatus")
    code.add_argument("--output", type=Path, required=True)
    code.add_argument("--workspace", type=Path, required=True)
    code.add_argument("--tasks", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2])
    code.add_argument("--seed", type=int, default=313)
    code.add_argument("--max-calls", type=int, default=12)
    code.add_argument("--token-budget", type=int, default=120000)
    code.add_argument("--code-results", type=Path, help="Use recorded code-agent runs to avoid repeating model calls")
    code.add_argument("--hegel-results", type=Path, help="Use recorded matched cognitive-agent runs")
    code_session = commands.add_parser("code-session", help="Continue a shared-knowledge ordinary Python and cognitive agent comparison")
    code_session.add_argument("--checkpoint", type=Path, required=True)
    code_session.add_argument("--workspace", type=Path, required=True)
    code_session.add_argument("--request", type=Path, help="JSON initial tasks, resource limits, agent configuration or new_tasks")
    code_session.add_argument("--steps", type=int, help="Pause after this many further paired task visits")
    code_session.add_argument("--evaluate", action="store_true", help="Evaluate after both agents have finished learning")
    unified = commands.add_parser("unified", help="Choose instruments, history and representations on a common unknown-task stream")
    unified.add_argument("--checkpoint", type=Path, required=True)
    unified.add_argument("--request", type=Path, help="Initial tasks, research configuration and measurement budget")
    unified.add_argument("--steps", type=int, help="Pause after this many further tasks")
    unified.add_argument("--memory", type=Path, help="Load or export the shared executable knowledge library")
    unified.add_argument("--evaluate", action="store_true", help="Evaluate frozen programs after the learning stream finishes")
    return parser


def dispatch(args):
    if args.cognition_command == "historical":
        from .historical_capabilities import dispatch_historical
        return dispatch_historical(args)
    if args.cognition_command == "growth":
        from dataclasses import replace
        from .growth_lab import GrowthLab
        from .research_growth import GrowthResearcher
        state = (json.loads(args.checkpoint.read_text(encoding="utf-8"))
                 if args.checkpoint.exists() else None)
        config = ResearchConfig(observation_budget=args.observation_budget,
                                compute_budget=args.compute_budget, max_rounds=args.rounds,
                                seed=args.seed)
        researcher = GrowthResearcher(config, state=state,
                                      measurement_budget=args.measurement_budget)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        for stage in args.tasks:
            researcher.config = replace(config, seed=args.seed + stage * 10007)
            report = researcher.run(GrowthLab(seed=args.seed, stage=stage))
            args.checkpoint.write_text(json.dumps(researcher.to_dict(), indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"stage": stage, **{key: report[key] for key in
                ("status", "expression", "costs", "selected_channels", "uncertainty")}}))
        return 0
    if args.cognition_command in {"structure", "mechanism", "simulator", "prior", "frontier", "agenda", "agenda-demo", "experiment", "code-compare", "code-session", "unified"}:
        return _dispatch_extended(args)
    if args.cognition_command == "lab":
        return _dispatch_lab(args)
    values = asdict(ResearchConfig())
    session_state = None
    if args.cognition_command == "session" and args.checkpoint.exists():
        session_state = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        values.update(session_state["config"])
    if args.config:
        values.update(json.loads(args.config.read_text(encoding="utf-8")))
    for field, value in (("seed", args.seed), ("observation_budget", args.observation_budget),
                         ("compute_budget", args.compute_budget), ("max_rounds", args.rounds),
                         ("partition", args.partition), ("proposer", getattr(args, "proposer", None)),
                         ("experiment_policy", args.experiment_policy),
                         ("selection_policy", args.selection_policy),
                         ("joint_search", args.joint_search),
                         ("llm_calls_per_task", args.llm_calls_per_task),
                         ("llm_token_budget", args.llm_token_budget),
                         ("llm_timeout_seconds", args.llm_timeout_seconds)):
        if value is not None:
            values[field] = value
    config = ResearchConfig(**values)
    if args.cognition_command == "session":
        from .environments import make_suite
        from .session import ResearchSession
        partitions = args.partitions or (session_state.get("partitions") if session_state else None) or [config.partition]
        environments = [lambda index=index, partition=partition: make_suite(config.seed, partition=partition)[index]
                        for partition in partitions for index in range(len(make_suite(config.seed, partition=partition)))]
        count = len(environments)
        previous = session_state["limits"] if session_state else {}
        visits = args.max_visits if args.max_visits is not None else previous.get("max_visits", 2)
        observation_budget = (args.total_observation_budget if args.total_observation_budget is not None
                              else previous.get("observation_budget", count * config.observation_budget * visits))
        compute_budget = (args.total_compute_budget if args.total_compute_budget is not None
                          else previous.get("compute_budget", count * config.compute_budget * visits))
        session = ResearchSession(config, environments, observation_budget=observation_budget,
                                  compute_budget=compute_budget, max_visits=visits, state=session_state)
        session.state["partitions"] = partitions
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

        def save_session(state):
            args.checkpoint.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        report = session.run(max_steps=args.steps, on_checkpoint=save_session)
        save_session(report)
        summary = {key: value for key, value in report.items()
                   if key not in {"episodes", "knowledge", "tasks", "scheduler_events", "workspaces"}}
        summary["checkpoint"] = str(args.checkpoint.resolve())
    elif args.cognition_command == "compare":
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


def _dispatch_lab(args):
    from .measurement_session import MeasurementSession, session_summary

    state = None
    if args.resume:
        state = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    elif args.checkpoint.exists():
        raise ValueError("measurement checkpoint already exists; use --resume to continue it")
    values = asdict(ResearchConfig(observation_budget=128, compute_budget=100000, max_rounds=3, seed=313))
    if state is not None:
        values.update(state["config"])
    for name, value in (("seed", args.seed), ("observation_budget", args.observation_budget),
                        ("compute_budget", args.compute_budget), ("max_rounds", args.rounds)):
        if value is not None:
            values[name] = value
    session = MeasurementSession(
        ResearchConfig(**values), tasks=args.tasks, backend=args.backend, strategy=args.strategy,
        measurement_budget=args.measurement_budget, evaluation_length=args.evaluation_length,
        evaluation_trajectories=args.evaluation_trajectories, state=state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    def save(state):
        args.checkpoint.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = session.run(max_steps=args.steps, research_steps=args.research_steps, on_checkpoint=save)
    save(report)
    summary = session_summary(report)
    summary["checkpoint"] = str(args.checkpoint.resolve())
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


def _save_json(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dispatch_extended(args):
    if args.cognition_command == "unified":
        from .unified_session import UnifiedResearchSession, unified_summary
        state = json.loads(args.checkpoint.read_text(encoding="utf-8")) if args.checkpoint.exists() else None
        request = json.loads(args.request.read_text(encoding="utf-8")) if args.request else {}
        if state is not None and request:
            raise ValueError("continue the saved sequence without --request")
        if "config" in request:
            request["config"] = ResearchConfig(**request["config"])
        if state is None and args.memory and args.memory.exists():
            request["library"] = KnowledgeLibrary.from_dict(json.loads(args.memory.read_text(encoding="utf-8")))
        session = UnifiedResearchSession(state=state, **request)
        report = session.run(max_tasks=args.steps, on_checkpoint=lambda payload: _save_json(args.checkpoint, payload))
        _save_json(args.checkpoint, report)
        if args.memory:
            _save_json(args.memory, report["knowledge"])
        if args.evaluate:
            report = session.evaluate()
            _save_json(args.checkpoint, report)
        summary = unified_summary(report)
        _save_json(args.checkpoint.with_name(args.checkpoint.stem + "_summary.json"), summary)
    elif args.cognition_command == "experiment":
        from .delayed_feedback import DelayedFeedback
        from .environments import make_suite
        from .models import Probe
        from .session import ResearchSession
        saved = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        config = ResearchConfig(**saved["config"])
        sources = [environment for partition in saved.get("partitions", [config.partition])
                   for environment in make_suite(config.seed, partition)]
        feedback = (None if saved.get("delayed_feedback") else DelayedFeedback(
            args.directory or args.checkpoint.with_suffix("").with_name(args.checkpoint.stem + "_experiments"),
            observation_budget=saved["limits"]["observation_budget"]))
        session = ResearchSession(config, sources, state=saved, feedback=feedback, **saved["limits"])
        if args.request:
            request = json.loads(args.request.read_text(encoding="utf-8"))
            probe = request["probe"]
            session.submit_experiment(request["task_index"], Probe(probe["probe_id"], tuple(probe["sequence"])),
                source=request.get("source"), channels=tuple(request.get("channels", ())),
                delay_seconds=request.get("delay_seconds", 0.))
        received = session.poll_experiments()
        report = session.to_dict()
        _save_json(args.checkpoint, report)
        summary = {"checkpoint": str(args.checkpoint.resolve()), "received": received,
                   "pending_experiments": report["pending_experiments"], "remaining": report["remaining"]}
    elif args.cognition_command == "agenda-demo":
        from .agenda_demo import run_agenda_demo
        summary = run_agenda_demo(args.checkpoint, seed=args.seed)
    elif args.cognition_command == "agenda":
        from .agenda import ResearchAgenda, agenda_summary
        from .models import Probe
        state = json.loads(args.checkpoint.read_text(encoding="utf-8")) if args.checkpoint.exists() else None
        request = json.loads(args.request.read_text(encoding="utf-8")) if args.request else {}
        agenda = ResearchAgenda(directory=args.checkpoint.parent / (args.checkpoint.stem + "_work"),
                                state=state, **(request.get("limits", {}) if state is None else {}))
        for item in request.get("questions", []):
            item = dict(item)
            kind, identifier = item.pop("kind"), item.pop("question_id")
            config = ResearchConfig(**item.pop("config")) if item.get("config") else None
            {"task": agenda.add_task, "measurement": agenda.add_measurement,
             "structure": agenda.add_structure, "mechanism": agenda.add_mechanism}[kind](identifier, config=config, **item)
        for item in request.get("dependencies", []):
            agenda.set_dependencies(item["question_id"], item["depends_on"])
        for item in request.get("experiments", []):
            item = dict(item)
            identifier, probe = item.pop("question_id"), item.pop("probe")
            agenda.submit_experiment(identifier, Probe(probe["probe_id"], tuple(probe["sequence"])), **item)
        report = agenda.run(max_steps=args.steps, on_checkpoint=lambda state: _save_json(args.checkpoint, state))
        _save_json(args.checkpoint, report)
        summary = agenda_summary(report)
        _save_json(args.checkpoint.with_name(args.checkpoint.stem + "_summary.json"), summary)
    elif args.cognition_command == "prior":
        from .meta_prior_session import MetaPriorSession, meta_prior_summary
        state = json.loads(args.checkpoint.read_text(encoding="utf-8")) if args.checkpoint.exists() else None
        request = json.loads(args.request.read_text(encoding="utf-8")) if args.request else {}
        if state and request:
            raise ValueError("continue the saved prior session without --request; use a new checkpoint for a new source")
        if "config" in request:
            request["config"] = ResearchConfig(**request["config"])
        library = KnowledgeLibrary.from_dict(json.loads(args.memory.read_text(encoding="utf-8"))) if not state and args.memory and args.memory.exists() else None
        session = MetaPriorSession(state=state, library=library, **request)
        if args.optimize or "optimizer_config" in session.state:
            config = json.loads(args.optimizer_config.read_text(encoding="utf-8")) if args.optimizer_config else None
            report = session.run_deploy(config, max_tasks=args.steps, research_steps=args.research_steps,
                max_optimizer_tasks=args.optimizer_steps,
                on_checkpoint=lambda payload: _save_json(args.checkpoint, payload))
        else:
            report = session.run(max_tasks=args.steps, research_steps=args.research_steps,
                on_checkpoint=lambda payload: _save_json(args.checkpoint, payload))
        _save_json(args.checkpoint, report)
        if args.evaluate:
            report = session.evaluate()
            _save_json(args.checkpoint, report)
        if args.memory:
            _save_json(args.memory, session.library.to_dict())
        summary = meta_prior_summary(report)
        _save_json(args.checkpoint.with_name(args.checkpoint.stem + "_summary.json"), summary)
    elif args.cognition_command == "frontier":
        from .frontier import FrontierSession, frontier_summary
        state = json.loads(args.checkpoint.read_text(encoding="utf-8")) if args.checkpoint.exists() else None
        session = FrontierSession(seed=args.seed, state=state,
                                  directory=args.checkpoint.parent / (args.checkpoint.stem + "_experiments"))
        if args.code_comparison:
            session.state["ordinary_code_comparison"] = json.loads(args.code_comparison.read_text(encoding="utf-8"))
        report = session.run(max_stages=args.steps, on_checkpoint=lambda state: _save_json(args.checkpoint, state))
        _save_json(args.checkpoint, report)
        summary = frontier_summary(report)
        _save_json(args.checkpoint.with_name(args.checkpoint.stem + "_summary.json"), summary)
    elif args.cognition_command in {"structure", "mechanism"}:
        library = KnowledgeLibrary.from_dict(json.loads(args.memory.read_text(encoding="utf-8"))) if args.memory and args.memory.exists() else KnowledgeLibrary()
        if args.cognition_command == "mechanism":
            from .mechanism_lab import MechanismLab
            from .mechanism_discovery import run_mechanism_research
            report = run_mechanism_research(MechanismLab(family=args.family, variant=args.variant, seed=args.seed), library=library)
        else:
            from .structure_lab import StructureLab
            from .structural_discovery import run_structural_research
            report = run_structural_research(StructureLab(readout=args.readout, variant=args.variant, seed=args.seed), library=library)
        _save_json(args.output, report)
        if args.memory:
            _save_json(args.memory, library.to_dict())
        summary = {key: value for key, value in report.items() if key not in {"episode", "discovery", "batches", "knowledge"}}
    elif args.cognition_command == "simulator":
        from .simulator_research import run_simulator_research
        state = json.loads(args.checkpoint.read_text(encoding="utf-8")) if args.checkpoint.exists() else None
        library = KnowledgeLibrary.from_dict(json.loads(args.memory.read_text(encoding="utf-8"))) if args.memory and args.memory.exists() else KnowledgeLibrary()
        report = run_simulator_research(library=library, state=state, max_steps=args.steps)
        _save_json(args.checkpoint, report)
        if args.memory:
            _save_json(args.memory, library.to_dict())
        summary = {key: value for key, value in report.items() if key not in {"observations", "simulator_state", "knowledge"}}
    elif args.cognition_command == "code-session":
        from .code_agent import CodeAgentConfig
        from .code_session import CodeComparisonSession
        state = json.loads(args.checkpoint.read_text(encoding="utf-8")) if args.checkpoint.exists() else None
        request = json.loads(args.request.read_text(encoding="utf-8")) if args.request else {}
        new_tasks = request.pop("new_tasks", [])
        if "config" in request:
            request["config"] = CodeAgentConfig(**request["config"])
        if "hegel_config" in request:
            request["hegel_config"] = ResearchConfig(**request["hegel_config"])
        session = CodeComparisonSession(args.workspace, state=state, **request)
        for task in new_tasks:
            session.add_task(**task)
        report = session.run(max_tasks=args.steps, on_checkpoint=lambda payload: _save_json(args.checkpoint, payload))
        if args.evaluate:
            session.evaluate()
            report = session.to_dict()
        _save_json(args.checkpoint, report)
        summary = {key: value for key, value in report.items() if key in {"status", "summary", "costs", "totals", "remaining", "comparison"}}
    else:
        from .code_agent import CodeAgentConfig
        from .code_comparison import run_code_comparison
        code_config = CodeAgentConfig(max_model_calls=args.max_calls, token_budget=args.token_budget)
        def results(path):
            return json.loads(path.read_text(encoding="utf-8"))["results"] if path else ()
        report = run_code_comparison(args.workspace, tasks=tuple(args.tasks),
            config=ResearchConfig(observation_budget=128, compute_budget=100000, max_rounds=3, seed=args.seed),
            code_config=code_config, code_results=results(args.code_results), hegel_results=results(args.hegel_results),
            on_result=lambda report: _save_json(args.output, report))
        _save_json(args.output, report)
        summary = {key: value for key, value in report.items() if key not in {"results", "pairs"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
