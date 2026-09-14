"""Allocate an unfinished question using a method learned in actual later work."""

from dataclasses import replace

from .controller import CognitiveAgent
from .environments import make_suite
from .memory import KnowledgeLibrary
from .models import Observation, Probe, ResearchConfig
from .session import ResearchSession


class _ResearchInstrument:
    """Present existing public experiments with this instrument's channel names."""

    def __init__(self, environment, name):
        self.environment = environment
        self.names = dict(zip(environment.spec.feature_names, ("allocation_input", "allocation_disturbance")))
        self.inverse = {value: key for key, value in self.names.items()}
        spec = environment.spec
        self.spec = replace(spec, task_id=name, feature_names=tuple(self.names.values()),
            initial_probes=tuple(self._probe(probe, self.names) for probe in spec.initial_probes),
            available_probes=tuple(self._probe(probe, self.names) for probe in spec.available_probes),
            feature_bounds={self.names[key]: value for key, value in spec.feature_bounds.items()})

    @staticmethod
    def _probe(probe, names):
        return Probe(probe.probe_id, tuple({names.get(key, key): value for key, value in row.items()}
                                         for row in probe.sequence))

    def observe(self, probe):
        rows = self.environment.observe(self._probe(probe, self.inverse))
        return [Observation({self.names.get(key, key): value for key, value in row.features.items()}, row.target,
                tuple({self.names.get(key, key): value for key, value in point.items()} for point in row.history))
                for row in rows]


def run_method_allocation(library=None):
    """A bounded source episode changes which real session question runs next.

    Methods are learned exclusively by CognitiveAgent's later prediction
    feedback. The scheduler compares their measured costs and returns with an
    unvisited question; the selected visit executes the requested next step.
    """
    library = library if library is not None else KnowledgeLibrary()
    # Each visit must be able to execute the learned representation step,
    # whose measured source cost can exceed the former 6000-unit visit cap.
    config = ResearchConfig(seed=7, partition="continual", compute_budget=9000,
        observation_budget=48, max_rounds=1, use_abstractions=False)
    suite = make_suite(7, "continual")
    unfinished = _ResearchInstrument(suite[0], "allocation:unfinished")
    unvisited = _ResearchInstrument(suite[9], "allocation:unvisited")
    environments = [unfinished, unvisited]
    session = ResearchSession(config, environments, observation_budget=160,
                              compute_budget=60000, max_visits=3, library=library)
    initial = session.run(max_steps=1)
    earlier_loss = initial["tasks"][0]["validation_loss"]
    source = _ResearchInstrument(make_suite(7, "continual")[2], "allocation:method_source")
    source_config = replace(config, compute_budget=16000, observation_budget=64, max_rounds=4)
    learned = CognitiveAgent(source_config, library).run(source)
    source_episodes = [learned]
    if learned.validation_loss is None or learned.validation_loss > config.minimum_improvement:
        learned = CognitiveAgent(replace(source_config, seed=source_config.seed + 1000003), library).run(
            source, research_state=learned.research_state)
        source_episodes.append(learned)
    # Carry actual observations, methods, and pending opportunities through the
    # same checkpoint API used by ordinary interrupted research sessions.
    before = session.to_dict()
    restored = ResearchSession(config, environments, observation_budget=160,
        compute_budget=60000, max_visits=3, state=before)
    completed = restored.run(max_steps=1)
    # A method may have several dependent steps. Continue the scheduler's own
    # selected unfinished sequence while its quoted opportunity still wins.
    for _ in range(2):
        pending = restored._pending()
        if (not pending or pending[0]["task_index"] != 0 or not pending[0].get("next_method")
                or not completed["tasks"][0]["unresolved"]):
            break
        previous_visits = completed["completed_visits"]
        completed = restored.run(max_steps=1)
        if completed["completed_visits"] == previous_visits:
            break
    library.__dict__.update(restored.library.__dict__)
    selected_episode = completed["episodes"][1]
    episode = completed["episodes"][completed["tasks"][selected_episode["task_index"]]["latest_episode"]]
    visits = [visit for visit in completed["episodes"][1:] if visit["task_index"] == selected_episode["task_index"]]
    execution = [action["details"] for visit in visits for action in visit["actions"] if action["action"] == "execute_method_step"]
    feedback = [action["details"] for visit in visits for action in visit["actions"] if action["action"] == "action_feedback"]
    trace = completed["workspaces"][str(episode["task_index"])].get("method_trace", [])
    measured_steps = [step for step in trace if step.get("method_id")]
    summary = {"earlier_question_unresolved": initial["tasks"][0]["unresolved"],
        "earlier_validation_loss": earlier_loss,
        "source_validation_loss": learned.validation_loss,
        "source_compute_cost": sum(result.compute_cost for result in source_episodes),
        "source_observation_cost": sum(result.observation_cost for result in source_episodes),
        "source_compute_budget": 2*source_config.compute_budget, "source_observation_budget": 2*source_config.observation_budget,
        "source_visits": len(source_episodes), "source_visit_losses": [result.validation_loss for result in source_episodes],
        "selected_task_index": selected_episode["task_index"], "selection_reason": selected_episode["scheduler"]["reason"],
        "method_opportunity": selected_episode["scheduler"].get("method_opportunity"),
        "method_steps_executed": execution, "measured_method_returns": measured_steps,
        "continued_validation_loss": episode["validation_loss"],
        "unvisited_question_still_pending": completed["tasks"][1]["visits"] == 0,
        "session_compute_cost": completed["total_compute_cost"],
        "session_observation_cost": completed["total_observation_cost"],
        "session_compute_budget": 60000, "session_observation_budget": 160}
    return {"schema": "hegel_machine_method_allocation_v1", "summary": summary,
            "source_episode": learned.to_dict(), "source_episodes": [result.to_dict() for result in source_episodes], "checkpoint": completed,
            "actual_feedback": feedback, "methods": library.methods.to_dict()}
