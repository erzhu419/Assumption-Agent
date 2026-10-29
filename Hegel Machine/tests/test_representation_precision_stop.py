"""A fully checked state solution closes search before unrelated composition."""

from collections import Counter

import pytest

from hegel_machine.cognition.representation_inquiry import load_native_representation_result
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_native_discrete_state_route import scene
from test_native_representation_routing import PublicOnlyLab


def test_actual_discrete_solution_stops_later_composition_and_still_compares_parents(tmp_path, monkeypatch):
    agent, arguments = scene(tmp_path)
    parent_source = (agent.workspace / 'old.py').read_bytes()

    def unnecessary_composition(*args, **kwargs):
        pytest.fail('a paid exact state solution must not trigger unrelated composition')

    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', unnecessary_composition)
    assessed = Counter()
    assess = UnifiedResearchAgent._assess

    def count_assess(self, claims, *args, **kwargs):
        for claim in claims:
            assessed[claim['candidate'].program] += 1
        return assess(self, claims, *args, **kwargs)

    monkeypatch.setattr(UnifiedResearchAgent, '_assess', count_assess)
    returned = agent._execute('run', arguments, PublicOnlyLab())
    result = load_native_representation_result(returned['native_representation_ref'], agent.workspace)
    assert result['selected']['training_mse'] == result['selected']['validation_mse'] == 0
    assert [entry['direction'] for entry in result['search_directions']] == ['discrete_state']
    assert result['search_stop']['completed_direction'] == 'discrete_state'
    assert result['search_stop']['unexecuted_directions'] == ['retained_composition']
    assert not result['search_stop']['independent_confirmation']
    assert result['comparison_completed'] and result['all_parent_candidates_retained']
    assert {entry['id'] for entry in result['retained_comparison']} == {'old'}
    assert all(count == 1 for count in assessed.values())
    assert (agent.workspace / 'old.py').read_bytes() == parent_source
    assert agent.model is None and not returned['parent_revision_completed']


def test_training_precision_without_validation_support_continues_composition(tmp_path, monkeypatch):
    agent, arguments = scene(tmp_path, validation_flip=True)
    continued = []

    def composition(*args, **kwargs):
        continued.append(True)
        return []

    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', composition)
    returned = agent._execute('run', arguments, PublicOnlyLab())
    result = load_native_representation_result(returned['native_representation_ref'], agent.workspace)
    assert continued == [True]
    assert [entry['direction'] for entry in result['search_directions']] == [
        'discrete_state', 'retained_composition']
    assert 'search_stop' not in result
    assert result['comparison_completed']
    assert any(entry['training_mse'] == 0 and entry['validation_mse'] > 0
               for entry in result['candidates'])
