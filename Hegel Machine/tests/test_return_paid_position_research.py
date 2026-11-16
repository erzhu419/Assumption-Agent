"""The paid-return entry consumes role evidence without rerunning fitted candidates."""
from types import SimpleNamespace

import pytest

from scripts.return_paid_position_research import consume_saved_owner_fits
from hegel_machine.cognition.common_object_research import CommonObjectResearchAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget
from hegel_machine.cognition.object_continuation import encode_entry


def fixture(tmp_path):
    agent = CommonObjectResearchAgent(workspace=tmp_path, organization='pr3')
    parent = {'candidate': Candidate('totals', Program('const', value=0.), 'retained'),
              'training_loss': .01, 'validation_loss': .1, 'kind': 'partial_object'}
    new = [{'candidate': Candidate(str(i), Program('const', value=i+1.), 'paid_fit'),
            'training_loss': .01, 'validation_loss': score, 'kind': 'structural',
            'working_role_revision': {'schema_index': i}} for i, score in enumerate((.2, .05))]
    frame = dict(assessed=[parent], comparison_complete=True, native_full_pool_comparison=False,
        uncompared=[], fallback=0., spec=SimpleNamespace(risk_quantile=.5),
        comparison=[Observation({'x': 1.}, 0.)], independent_validation=True)
    model = {'position_candidates': {'x': {'quantities': ['a']}, 'y': {'quantities': ['b']}}}
    assignments = {'schemas': [{'fields': {'a': {'q': 'a', 'x': x}, 'b': {'q': 'b', 'x': y}}}
                               for x,y in [('x','y'), ('y','x')]]}
    saved = {'parent_working_loss': .1, 'owner_response_fits': [
        {'schema_index': i, 'candidate_id': str(i), 'validation_loss': item['validation_loss']}
        for i, item in enumerate(new)]}
    return agent, frame, model, [encode_entry(x) for x in new], assignments, saved


def test_paid_fits_are_filtered_by_measured_domain_without_rescoring(tmp_path, monkeypatch):
    import hegel_machine.cognition.unified_research as unified
    inputs = fixture(tmp_path)
    def no_new_prediction(*args, **kwargs):
        raise AssertionError('The original working scores must be reused')
    monkeypatch.setattr(unified, 'loss', no_new_prediction)
    budget, actions = SearchBudget(4), []
    result, dispositions = consume_saved_owner_fits(*inputs, budget, actions)
    assert [d['status'] for d in dispositions] == [
        'compatible_with_measured_position', 'excluded_by_measured_position']
    assert result['selected'] is inputs[1]['assessed'][0]
    assert len(result['new_claims']) == 1 and result['new_claims'][0]['candidate'].candidate_id == '0'
    assert result['execution_complete'] and not result['native_full_pool_comparison']
    assert result['comparison_compute_cost'] == 0 and budget.spent == 4
    # The rejected structural explanation and its superior scalar score remain
    # in the paid artifact. A total-only incumbent has no such binding claim.
    assert len(inputs[3]) == 2 and inputs[3][1]['validation_loss'] == .05


def test_another_parent_or_candidate_score_is_not_reused(tmp_path):
    inputs = fixture(tmp_path)
    inputs[-1]['parent_working_loss'] = .12
    with pytest.raises(ValueError, match='original parent'):
        consume_saved_owner_fits(*inputs, SearchBudget(20), [])
    inputs[-1]['parent_working_loss'] = .1
    inputs[-1]['owner_response_fits'][1]['schema_index'] = 0
    with pytest.raises(ValueError, match='same candidate'):
        consume_saved_owner_fits(*inputs, SearchBudget(20), [])
