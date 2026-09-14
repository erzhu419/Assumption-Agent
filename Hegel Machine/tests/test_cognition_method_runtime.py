"""Method steps learned from outcomes must direct a later actual task."""

import json

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig


def test_a_learned_sequence_executes_on_another_task_and_receives_new_feedback():
    library = KnowledgeLibrary()
    config = ResearchConfig(seed=503, partition="continual")
    environments = make_suite(503, "continual")
    source = CognitiveAgent(config, library).run(environments[0])
    learned = next(method for method in library.methods.methods
                   if method["steps"] == ["refit", "representation"])
    assert learned["source_task"] == source.task_id
    assert any(step.get("produced_method_id") == learned["method_id"] and step["gain"] > 0
               for step in source.research_state["method_trace"])
    # Isolate method transfer from concept transfer. In the unrestricted
    # seed-503 sequence, existing concepts and ordinary plans solve the later
    # squared-loss tasks without selecting a learned method. This fixture
    # carries the actually learned procedures, with no transferred predictors.
    methods_only = KnowledgeLibrary()
    methods_only.methods = type(library.methods)(library.methods.to_dict())
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(methods_only.to_dict())))
    result = CognitiveAgent(config, restored).run(environments[4])
    choices = [action.details["learned_method"] for action in result.actions
               if action.details.get("learned_method")]
    assert choices
    identifier = learned["method_id"]
    method = next(item for item in restored.methods.methods if item["method_id"] == identifier)
    assert method["source_task"] != result.task_id
    assert method["trials"] == 2
    assert any(step.get("method_id") == identifier and step["gain"] > 0
               for step in result.research_state["method_trace"])
    selected_steps = [choice["step"] for choice in choices if choice["method_id"] == identifier]
    assert selected_steps == [0, 1]
    measured = [step for step in result.research_state["method_trace"] if step.get("method_id") == identifier]
    assert [step["action"] for step in measured] == ["refit", "representation"]
    assert all(step["execution_complete"] and step["compute_cost"] > 0 for step in measured)
    assert result.validation_loss < config.minimum_improvement
    # A completed sequence cannot sell itself as an unperformed opportunity.
    isolated = type(restored.methods)({"methods": [method]})
    assert isolated.opportunity(method["context"], method["condition"], task_id=result.task_id,
        observation_remaining=1000, compute_remaining=1000000,
        trace=result.research_state["method_trace"]) is None
