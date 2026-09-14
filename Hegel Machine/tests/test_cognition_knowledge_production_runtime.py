"""Knowledge acquired by research returns feedback to its producing method."""

import json

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig


def test_renamed_task_returns_actual_knowledge_value_to_its_producing_method_after_reload():
    source, target = [make_suite(1237, 'confirmation')[index] for index in (1, 6)]
    config = ResearchConfig(seed=1237)
    library = KnowledgeLibrary()
    learned = CognitiveAgent(config, library).run(source)
    links = [link for action in learned.actions if action.action == 'link_knowledge_method'
             for link in action.details['production_methods']]
    assert links
    assert any(step.get('produced_method_id') for step in learned.research_state['method_trace'])
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    transferred = CognitiveAgent(config, restored).run(target)
    returns = [record for action in transferred.actions if action.action == 'knowledge_return'
               for record in action.details['method_returns']]
    produced = {link['method_id'] for link in links}
    useful = [record for record in returns if record['method_id'] in produced and record['retained_gain'] > 0]
    assert useful
    assert all(record['downstream_task'] == target.spec.task_id for record in useful)
    assert any(record['comparisons'] for record in useful)
    assert transferred.observation_cost <= config.observation_budget
    assert transferred.compute_cost <= config.compute_budget
