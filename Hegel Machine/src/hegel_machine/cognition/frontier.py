"""A continuing research lifecycle spanning theory, experiments and observation."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .evaluation import loss
from .language import Program
from .memory import KnowledgeLibrary
from .models import Probe, ResearchConfig


STAGES = ("continued_tasks", "adaptive_measurements", "experiment_transfer", "method_allocation", "local_mechanisms", "total_exchange", "local_exchange",
          "window_events", "change_events", "leaking_system", "broken_symmetry", "distant_effects",
          "simulator_revision", "low_rank_mechanisms", "independent_mechanisms", "stable_dynamics", "graph_topology",
          "broken_low_rank", "coupled_mechanisms", "unstable_dynamics", "changed_topology",
          "ordered_response", "changed_response", "meta_prior_research", "delayed_experiment")

MECHANISM_STAGES = {
    "low_rank_mechanisms": ("low_rank", "regular"), "independent_mechanisms": ("modules", "regular"),
    "stable_dynamics": ("dynamics", "regular"), "graph_topology": ("topology", "regular"),
    "broken_low_rank": ("low_rank", "broken"), "coupled_mechanisms": ("modules", "broken"),
    "unstable_dynamics": ("dynamics", "broken"), "changed_topology": ("topology", "broken"),
    "ordered_response": ("order", "regular"), "changed_response": ("order", "broken"),
}


class FrontierSession:
    """Run each research stage once, carrying executable knowledge into the next.

    Stages describe available environments and research questions. Their
    explanations, methods, measurements and structural hypotheses are learned
    by the corresponding agents from paid observations.
    """

    def __init__(self, *, seed=503, directory, state=None):
        self.directory = Path(directory)
        self.state = deepcopy(state) if state else {
            "experiment": "integrated_cognitive_research_lifecycle", "seed": seed,
            "status": "ready", "stages": {}, "ordinary_code_comparison": None}
        self.seed = self.state["seed"]
        self.library = KnowledgeLibrary.from_dict(self.state["knowledge"]) if state else KnowledgeLibrary()

    def run(self, *, max_stages=None, on_checkpoint=None):
        completed = 0
        for stage in STAGES:
            previous = self.state["stages"].get(stage)
            if previous and not (previous.get("status") == "waiting_for_feedback" or previous.get("active_task")):
                continue
            if max_stages is not None and completed >= max_stages:
                self.state["status"] = "paused"
                break
            if stage == "continued_tasks":
                result = self._continued_tasks()
            elif stage == "adaptive_measurements":
                from .measurement_session import MeasurementSession
                saved = {**previous, "knowledge": self.library.to_dict()} if previous else None
                session = MeasurementSession(ResearchConfig(observation_budget=128, compute_budget=100000,
                    max_rounds=3, seed=self.seed), backend="process", library=self.library, state=saved)

                def measurement_checkpoint(checkpoint):
                    self.library = session.library
                    self.state["stages"][stage] = {key: value for key, value in checkpoint.items() if key != "knowledge"}
                    self.state["status"] = "paused"
                    if on_checkpoint:
                        on_checkpoint(self.to_dict())

                result = session.run(on_checkpoint=measurement_checkpoint)
                self.library = session.library
                result.pop("knowledge")
            elif stage == "meta_prior_research":
                from .meta_prior_session import MetaPriorSession
                saved = {**previous, "knowledge": self.library.to_dict()} if previous else None
                session = MetaPriorSession(library=self.library, state=saved,
                    source={"kind": "scalar", "profile": "nested_readout", "seed": self.seed})

                def prior_checkpoint(checkpoint):
                    self.library = session.library
                    self.state["stages"][stage] = {key: value for key, value in checkpoint.items() if key != "knowledge"}
                    self.state["status"] = "paused"
                    if on_checkpoint:
                        on_checkpoint(self.to_dict())

                result = session.run(on_checkpoint=prior_checkpoint)
                if result["status"] == "completed":
                    result = session.evaluate()
                self.library = session.library
                result.pop("knowledge")
            elif stage == "experiment_transfer":
                from .experiment_research import run_experiment_transfer
                result = run_experiment_transfer(library=self.library)
            elif stage == "method_allocation":
                from .research_allocation_demo import run_method_allocation
                result = run_method_allocation(library=self.library)
            elif stage == "local_mechanisms":
                from .evolution_research import run_evolution_research
                result = run_evolution_research(library=self.library)
            elif stage == "simulator_revision":
                from .simulator_research import run_simulator_research
                result = run_simulator_research(library=self.library, state=previous)
            elif stage in MECHANISM_STAGES:
                from .mechanism_lab import MechanismLab
                from .mechanism_discovery import run_mechanism_research
                family, variant = MECHANISM_STAGES[stage]
                result = run_mechanism_research(MechanismLab(family=family, variant=variant, seed=self.seed),
                                               library=self.library)
            elif stage == "delayed_experiment":
                result = self._delayed_experiment(previous)
            else:
                result = self._structure(stage)
            self.state["stages"][stage] = result
            completed += 1
            if result.get("status") == "waiting_for_feedback" or result.get("active_task"):
                self.state["status"] = result["status"]
                if on_checkpoint:
                    on_checkpoint(self.to_dict())
                break
            self.state["status"] = "running"
            if on_checkpoint:
                on_checkpoint(self.to_dict())
        else:
            self.state["status"] = "completed"
        return self.to_dict()

    def _continued_tasks(self):
        from .environments import make_suite
        from .session import ResearchSession
        config = ResearchConfig(seed=self.seed, partition="continual")
        environments = make_suite(self.seed, "continual")
        session = ResearchSession(config, environments, observation_budget=len(environments) * 144,
                                  compute_budget=len(environments) * 36000, max_visits=2, library=self.library)
        report = session.run()
        self.library = session.library
        evaluation = []
        for index, environment in enumerate(environments):
            episodes = [row for row in report["episodes"] if row["task_index"] == index]
            if not episodes:
                continue
            latest = episodes[-1]
            rows = environment.evaluation_rows()
            evaluation.append({"task_index": index, "rows": len(rows), "risk_quantile": environment.spec.risk_quantile,
                "loss": loss(Program.from_dict(latest["program"]), rows, latest["fallback"], environment.spec.risk_quantile),
                "compute_cost": len(rows)})
        # The common library is saved once at the lifecycle boundary.
        report.pop("knowledge")
        report["evaluation"] = evaluation
        return report

    def _structure(self, stage):
        from .structure_lab import StructureLab
        from .structural_discovery import run_structural_research
        readout, variant = {
            "total_exchange": ("total", "exchange"), "local_exchange": ("focus", "exchange"),
            "window_events": ("window", "exchange"), "change_events": ("event", "exchange"),
            "leaking_system": ("total", "sink"), "broken_symmetry": ("focus", "anchored"),
            "distant_effects": ("focus", "nonlocal")}[stage]
        lab = StructureLab(readout=readout, variant=variant, seed=71)
        report = run_structural_research(lab, library=self.library)
        rows = lab.evaluation_rows(objects=report["selected_objects"], scale=report["selected_scale"],
                                   seed=self.seed + 1000003, length=64)
        report["evaluation"] = {"rows": len(rows), "compute_cost": len(rows),
            "mse": loss(Program.from_dict(report["episode"]["program"]), rows, report["episode"]["fallback"], .5)}
        report.update(status="completed", readout=readout, variant=variant)
        return report

    def _delayed_experiment(self, previous):
        from .delayed_feedback import DelayedFeedback
        from .environments import make_suite
        from .session import ResearchSession
        config = ResearchConfig(seed=17, observation_budget=32, compute_budget=8000, max_rounds=2)
        environments = make_suite(17)[:2]
        options = dict(observation_budget=100, compute_budget=24000, max_visits=1)
        if previous:
            saved = deepcopy(previous)
            saved["knowledge"] = self.library.to_dict()
            session = ResearchSession(config, environments, state=saved, **options)
        else:
            session = ResearchSession(config, environments, library=self.library,
                feedback=DelayedFeedback(self.directory, observation_budget=100), **options)
            probe = Probe("delayed_public_controls", tuple({"x": x, "z": z}
                          for x, z in ((-.8, .4), (.2, -.6), (.9, .1), (-.1, .9))))
            session.submit_experiment(0, probe, delay_seconds=.3)
        result = session.run()
        self.library = session.library
        result.pop("knowledge")
        return result

    def to_dict(self):
        return {**deepcopy(self.state), "knowledge": self.library.to_dict(),
                "pending_stages": [stage for stage in STAGES if stage not in self.state["stages"]
                                   or self.state["stages"][stage].get("status") == "waiting_for_feedback"
                                   or self.state["stages"][stage].get("active_task")]}


def frontier_summary(report):
    stages, knowledge = report["stages"], report["knowledge"]
    sequence = stages.get("continued_tasks", {})
    actions = [action for episode in sequence.get("episodes", []) for action in episode["actions"]]
    methods = knowledge.get("methods", {}).get("methods", [])
    experiments = knowledge.get("experiments", [])
    comparison = report.get("ordinary_code_comparison")
    code_results = []
    if comparison and comparison.get("experiment") == "persistent_code_and_hegel_comparison":
        evaluation = {row["task_position"]: row for row in comparison.get("evaluation") or []}
        for code, hegel in zip(comparison["code"]["results"], comparison["hegel"]["results"]):
            measured = evaluation.get(code["task_position"], {})
            code_results.append({"task_position": code["task_position"], "task_index": code["task_index"],
                "code_status": code["status"], "code_evaluation": measured.get("code"),
                "code_costs": code["costs"], "hegel_evaluation": measured.get("hegel"), "hegel_costs": hegel["costs"]})
    elif comparison:
        code_results = [{"task_index": item["task_index"],
            "code_status": item["code_agent"]["status"], "code_evaluation": item["code_agent"].get("evaluation"),
            "code_costs": item["code_agent"]["costs"], "hegel_evaluation": item["hegel"]["evaluation"],
            "hegel_costs": item["hegel"]["costs"]} for item in comparison.get("results", [])]
    return {"status": report["status"], "completed_stages": len(stages) - int(bool(report["pending_stages"] and
                report["pending_stages"][0] in stages)), "pending_stages": report["pending_stages"],
            "concepts": len(knowledge.get("concepts", [])),
            "methods": {"learned": len(methods), "prospective_trials": sum(item["trials"] for item in methods),
                "confirmed_trials": sum(item["confirmations"] for item in methods),
                "actual_sequence_choices": sum(bool(action["details"].get("learned_method")) for action in actions),
                "revisions": knowledge.get("methods", {}).get("revisions", [])},
            "experiments": {"learned": len(experiments), "reused": sum(item["uses"] for item in experiments),
                "actual_recalled_queries": sum(action["action"] == "observe" and
                    action["details"].get("probe_id", "").startswith("recalled_") for action in actions)},
            "task_allocation": [{"task_index": event["task_index"], "method": event["method_opportunity"]}
                                for event in sequence.get("scheduler_events", []) if event.get("method_opportunity")],
            "sequence": {key: sequence.get(key) for key in ("visited_tasks", "completed_visits", "adequate_tasks",
                          "unresolved_tasks", "total_observation_cost", "total_compute_cost", "evaluation")},
            "research_opportunities": knowledge.get("research_opportunities", {}),
            "distinguishing_questions": {
                "retained": sum(action["action"] == "retain_research_question" for action in actions),
                "executed": sum(action["action"] == "execute_research_question" for action in actions)},
            "decision_research": {
                "questions_retained": sum(action["action"] == "retain_decision_question" for action in actions),
                "questions_executed": sum(action["action"] == "execute_decision_question" for action in actions),
                "plans_adopted": sum(action["action"] == "execute_decision_question" and
                    bool(action["details"].get("adopted")) for action in actions),
                "history_experiments": sum(action["action"] == "plan_experiment" and
                    action["details"].get("policy") == "decision_history_contrast" for action in actions),
                "short_history_experiments": sum(action["action"] == "plan_decision_probe_package" and
                    bool(action["details"].get("selected")) for action in actions),
                "additional_comparisons": sum(action["action"] in {
                    "extend_decision_question", "extend_decision_comparison"} for action in actions),
                "unresolved_comparisons": sum(bool(workspace.get("decision_frontier", {}).get("evidence_states"))
                    for workspace in sequence.get("workspaces", {}).values()),
                "complete_plans_retained": sum(action["action"] == "retain_decision_plan" for action in actions)},
            "adaptive_measurements": {"status": stages.get("adaptive_measurements", {}).get("status"),
                "tasks": [{"task_id": row["task_id"], "channels": row["selected_channels"],
                           "costs": row["costs"], "evaluation": row["evaluation"],
                           "measurement_changes": row.get("measurement_events", []),
                           "measurement_question": row.get("measurement_question", {})}
                          for row in stages.get("adaptive_measurements", {}).get("results", [])]},
            "meta_prior_research": {key: stages.get("meta_prior_research", {}).get(key)
                for key in ("status", "costs", "evaluation")},
            "meta_priors": knowledge.get("meta_priors", []),
            "structure": {name: {key: stage.get(key) for key in ("readout", "variant", "selected_scale",
                            "measurement_reconfiguration", "prior_assessments", "costs", "evaluation")}
                          for name, stage in stages.items() if "selected_scale" in stage},
            "local_mechanisms": stages.get("local_mechanisms", {}).get("summary"),
            "simulator_revision": {key: stages.get("simulator_revision", {}).get(key) for key in
                ("status", "phase", "costs", "selected_probe_changed", "revisions")},
            "mechanism_discovery": {name: {key: stages[name].get(key) for key in
                ("status", "costs", "discovery", "prior_assessments")}
                for name in MECHANISM_STAGES if name in stages},
            "experiment_transfer": stages.get("experiment_transfer", {}).get("summary"),
            "method_allocation": stages.get("method_allocation", {}).get("summary"),
            "delayed_experiment": {key: stages.get("delayed_experiment", {}).get(key) for key in
                ("status", "completed_visits", "total_observation_cost", "total_compute_cost", "total_experiment_cost", "total_measurement_cost")},
            "ordinary_code_comparison_provenance": comparison.get("provenance") if comparison else None,
            "ordinary_code_sequence": comparison.get("summary") if comparison else None,
            "ordinary_code_comparison": code_results}
