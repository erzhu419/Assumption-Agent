"""A partial five-object binding is not a completed focal representation."""

from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from hegel_machine.cognition.common_object_research import CommonObjectResearchAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, SearchBudget, TaskSpec
from hegel_machine.cognition.partial_object_knowledge import partial_object_schema
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def inspect_structure_route(workspace, model, organization='pr3'):
    """Inspect preparation and dispatch only; neither branch performs science."""
    saved = deepcopy(model)
    fields = tuple(model.get('raw_sensor_features') or [group[role]
        for group in model['groups'] for role in ('q', 'u', 'x', 'f') if role in group])
    context = {'object_model': model}
    if model.get('schema'):
        context['object_schema'] = json.dumps(model['schema'])
    spec = TaskSpec('saved_partial_roles', 'Paid role representation', fields, (), (), context=context)
    agent = CommonObjectResearchAgent(workspace=workspace, organization=organization)
    captured, actions, budget = {}, [], SearchBudget(0)

    def prepared(_agent, binding_spec, training, validation, _budget, _actions, **kwargs):
        captured['binding_schema'] = json.loads(binding_spec.context['object_schema'])
        incumbent = {'candidate': Candidate('paid_affine', Program('const', value=0.), 'affine'),
            'kind': 'affine', 'training_loss': 1., 'validation_loss': 1.}
        return {'spec': binding_spec, 'training': [], 'comparison': [],
            'transitions': [], 'parent': None, 'assessed': [incumbent], 'uncompared': [],
            'comparison_complete': True, 'terminal': False, 'prepare_compute_cost': 0,
            'fallback': 0., 'retained_on_current': None, 'usable_parent': False}

    def partial(current, *args, **kwargs):
        captured['route'] = 'partial_object_explanations'
        captured['research_context'] = current.context
        return []

    def complete(current, *args, **kwargs):
        captured['route'] = 'complete_structural_search'
        captured['research_context'] = current.context
        return []

    with patch.object(UnifiedResearchAgent, 'prepare_research_decision', prepared), \
            patch.object(agent, '_object_explanations', partial), \
            patch('hegel_machine.cognition.unified_structure.propose_structures', complete):
        frame = agent.prepare_research_decision(spec, [], [], budget, actions)
        agent._execute_prepared_research_segment(frame, 'structural_research',
            budget, actions, search_limit=0)
    assert frame['spec'] is spec
    assert model == saved
    assert budget.spent == 0
    return captured


@pytest.mark.parametrize('organization', ['archive', 'fixed_pr2', 'pr3'])
@pytest.mark.parametrize('complete', [False, True])
def test_five_object_partial_binding_keeps_old_partial_research_route(tmp_path, organization, complete):
    roles = ('q', 'u', 'x', 'f') if complete else ('q', 'u')
    model = {'status': 'identified' if complete else 'partial', 'complete': complete,
        'groups': [{'object_id': f'induced_{index}',
            **{role: f'measured_{role}_{index}' for role in roles},
            'controls': {'q': f'quantity_{index}', 'u': f'drive_{index}'}}
            for index in range(5)],
        'focus_control': 'selector', 'unresolved': [] if complete else
            [f'induced_{index}:{role}' for index in range(5) for role in ('x', 'f')]}
    model['schema'] = partial_object_schema(model) if complete else None
    result = inspect_structure_route(tmp_path, model, organization)
    assert len(result['binding_schema']['objects']) == 5
    assert all(set(fields) == set(roles) for fields in result['binding_schema']['fields'].values())
    assert result['research_context']['object_model']['unresolved'] == model['unresolved']
    if complete:
        assert result['route'] == 'complete_structural_search'
        assert json.loads(result['research_context']['object_schema']) == model['schema']
    else:
        assert result['route'] == 'partial_object_explanations'
        assert 'object_schema' not in result['research_context']
