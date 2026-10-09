"""Call the existing unified researcher from the public paid inquiry interface.

The bridge changes the laboratory protocol, not the research algorithm. Native
multi-task sessions remain responsible for portfolios and other laboratory
families; this entry operates only on the currently supplied public apparatus.
"""

from copy import deepcopy
from dataclasses import asdict
import json

from .language import Program
from .measurement_lab import MeasurementBatch, MeasurementChannel
from .memory import KnowledgeLibrary
from .models import Observation, ResearchConfig, TaskSpec
from .program_experiment_methods import ProgramExperimentMethods
from .unified_research import UnifiedResearchAgent


LEGACY_KNOWLEDGE_PATH = "shared/legacy_knowledge.json"
METHODS_PATH = "shared/instrument_methods.json"


class InquiryResearchLab:
    """Translate actual inquiry purchases, without opening simulator internals."""

    def __init__(self, agent, laboratory):
        self.agent, self.laboratory = agent, laboratory
        self.public = laboratory.public_spec()
        if self.public.get("action_kinds") != ["trajectory"]:
            raise ValueError("legacy_research requires the public trajectory interface")
        self.channels = tuple(MeasurementChannel(**item) for item in self.public["channels"])
        self.spec = self.spec_for(())

    def spec_for(self, channels=()):
        names = tuple(channels)
        if not set(names) <= {channel.name for channel in self.channels}:
            raise ValueError("unknown public measurement channel")
        return TaskSpec("inquiry_" + self.agent.workspace.name,
            "Predict the current apparatus from public controls and paid measurements.",
            (*self.public["controls"], *names), (), (),
            feature_bounds={name: tuple(bounds)
                for name, bounds in self.public["feature_bounds"].items()},
            max_probe_length=self.public["max_probe_length"])

    @staticmethod
    def measured_rows(record):
        readings = {item["observation_index"]: item["values"]
                    for item in record.get("measurements", [])}
        rows, history = [], []
        for index, payload in zip(record["row_indices"], record["rows"]):
            features = {**payload["features"], **readings.get(index, {})}
            rows.append(Observation(features, payload["target"], tuple(deepcopy(history))))
            history.append({**features, "__target__": payload["target"]})
        return rows

    def acquire(self, probe, channels=()):
        action = {"kind": "trajectory", "sequence": list(deepcopy(probe.sequence)),
                  "channels": list(channels)}
        result = self.agent._query({"actions": [action]}, self.laboratory)
        if result.get("error"):
            raise ValueError(result["error"])
        record = result["observations"][0]
        return MeasurementBatch(self.measured_rows(record), record["observation_cost"],
                                record["measurement_cost"], tuple(channels))


def _paid_question(agent, adapter):
    """Initialize unresolved research using the already purchased trajectories.

    Zero is an explicit unevaluated baseline, not a learned or confirmed claim.
    The native continuation will compare it and research alternatives using the
    purchased labels before asking this adapter for further observations.
    """
    if not agent.observations:
        return None
    batches = []
    for index, record in enumerate(agent.observations):
        batches.append({"probe": {"probe_id": f"paid_inquiry_{index}",
                                  "sequence": deepcopy(record["action"]["sequence"])},
                        "channels": list(record["action"].get("channels", [])),
                        "observations": [asdict(row) for row in adapter.measured_rows(record)],
                        "transitions": []})
    return {"task_id": adapter.spec.task_id, "status": "unresolved",
            "program": Program("const", value=0.).to_dict(), "expression": "0", "fallback": 0.,
            "selected_kind": "unassessed_paid_response_baseline", "selected_channels": [],
            "validation_loss": None, "confirmation_loss": None,
            "actions": [{"action": "import_paid_observations",
                         "record_indices": list(range(len(agent.observations)))}],
            "batches": batches, "transfer": [], "concepts_added": [], "used_sources": [],
            "costs": {"observation_cost": sum(len(record["rows"]) for record in agent.observations),
                      "measurement_cost": sum(record["measurement_cost"] for record in agent.observations),
                      "compute_cost": 0, "queries": len(batches)}}


def run_legacy_research(agent, arguments, laboratory):
    """Execute the legacy loop, save its complete knowledge, and offer its model.

    observation_budget/compute_budget are additional allowances within the
    current agent's remaining caps. Fresh Agent confirmation remains reserved.
    The legacy report's internal confirmation never sets the Agent's model.
    """
    adapter = InquiryResearchLab(agent, laboratory)
    reserve = agent.config.confirmation_rows
    available_rows = max(0, agent.config.observation_budget - agent.costs["observation_cost"] - reserve)
    rows = min(available_rows, int(arguments.get("observation_budget", available_rows)))
    # finish scores all paid rows once before buying its fresh confirmation.
    available_compute = max(0, agent._remaining_compute() - len(agent.rows) - rows - reserve)
    compute = min(available_compute, int(arguments.get("compute_budget", available_compute)))
    if rows < 0 or compute < 1:
        raise ValueError("legacy_research requires nonnegative observation and positive compute allowance")
    previous = _paid_question(agent, adapter)
    if not previous and not rows:
        raise ValueError("legacy_research needs paid evidence or an observation allowance")
    prior_costs = previous["costs"] if previous else {
        "observation_cost": 0, "measurement_cost": 0, "compute_cost": 0, "queries": 0}
    knowledge_path = arguments.get("knowledge", LEGACY_KNOWLEDGE_PATH)
    source = agent.workspace / knowledge_path
    library = KnowledgeLibrary.from_dict(json.loads(source.read_text())) if source.exists() else KnowledgeLibrary()
    method_file = agent.workspace / METHODS_PATH
    if method_file.exists():
        library.program_experiment_methods = ProgramExperimentMethods(json.loads(method_file.read_text()))
    before_concepts = [item.concept_id for item in library.concepts]
    config = ResearchConfig(observation_budget=prior_costs["observation_cost"] + rows,
        compute_budget=compute, minimum_improvement=float(arguments.get("tolerance", 1e-7)),
        max_rounds=int(arguments.get("rounds", 4)), max_history=int(arguments.get("max_history", 4)),
        seed=int(arguments.get("seed", 71)), active_probes=True)
    measurement_allowance = int(arguments.get("measurement_budget",
        rows * sum(channel.cost_per_row for channel in adapter.channels)))
    if measurement_allowance < 0:
        raise ValueError("measurement_budget must be nonnegative")
    researcher = UnifiedResearchAgent(config, library=library,
        measurement_budget=prior_costs["measurement_cost"] + measurement_allowance)
    before = deepcopy(agent.costs)
    report = researcher.run(adapter, previous=previous)
    agent._charge(report["costs"]["compute_cost"])
    report_path = arguments.get("report", "legacy_research/report.json")
    agent._save(report_path, json.dumps(report, indent=2))
    agent._save(knowledge_path, json.dumps(library.to_dict(), indent=2))
    agent._save(METHODS_PATH, json.dumps(library.program_experiment_methods.to_dict(), indent=2))
    program = Program.from_dict(report["program"])
    model_path, export_reason = None, None
    # Use the current predictor interface's existing exporter. Native programs
    # requiring measurements or past targets stay usable in their native lab,
    # and cannot silently acquire either at prediction time here.
    from .temporal_operator_agent import export_temporal_program
    try:
        code = export_temporal_program(program)
    except ValueError as error:
        export_reason = str(error)
    else:
        model_path = arguments.get("path", "shared/legacy_model.py")
        agent._save(model_path, code)
    return {"legacy_status": report["status"], "path": model_path,
            "export_reason": export_reason, "report_path": report_path,
            "knowledge_path": knowledge_path, "selected_channels": report["selected_channels"],
            "selected_kind": report["selected_kind"], "expression": report["expression"],
            "legacy_confirmation_loss": report["confirmation_loss"],
            "requires_agent_confirmation": True,
            "available_concepts": before_concepts, "used_sources": report.get("used_sources", []),
            "concepts_added": report.get("concepts_added", []),
            "paid_rows_reused": prior_costs["observation_cost"],
            "new_costs": {key: agent.costs[key] - before[key]
                          for key in ("observation_cost", "measurement_cost", "compute_cost")},
            "actions": [action["action"] for action in report["actions"]]}
