"""Current native alternatives must not replay success or invent response terms."""

from copy import deepcopy
from types import SimpleNamespace

from hegel_machine.cognition.object_method_research import publish_object_response_options


def scene(loss=1., *, attempted=False):
    parent = {'validation_loss': loss, 'structure': {'fits': [{
        'response_terms': [{'op': 'feature', 'name': 'q'}]}]}}
    comparison = {'full_comparison': True, 'selected': parent, 'claims': [parent],
        'status': 'no_improvement' if attempted else 'compared',
        'action': {'observation_rows': 3,
            'action': 'revise_paid_structure' if attempted else 'compare_structure_revision',
            'search_compute': 17 if attempted else 0}}
    arguments = dict(compute_remaining=100000, tolerance=1e-7,
        data_ref={'file': 'paid.json', 'json_path': ['rows']},
        parent_ref={'file': 'comparison.json', 'json_path': ['selected']},
        pool_ref={'file': 'comparison.json', 'json_path': ['claims']},
        comparison_ref={'file': 'comparison.json', 'json_path': []})
    return SimpleNamespace(context={'object_schema': {'objects': ['o']}}), [1, 2, 3], comparison, arguments


def publish(prepared):
    spec, rows, comparison, arguments = prepared
    return publish_object_response_options(spec, rows, comparison, **arguments)


def test_explained_parent_or_incomplete_comparison_has_no_research_options():
    assert publish(scene(1e-21, attempted=True))['status'] == 'already_explained'
    unresolved = scene()
    unresolved[2]['full_comparison'] = False
    assert publish(unresolved)['reason'] == 'current_complete_paid_comparison_required'
    unresolved[2]['full_comparison'] = True
    unresolved[2]['action']['observation_rows'] = 2
    assert publish(unresolved)['options'] == []


def test_unexplained_parent_uses_ordinary_segment_without_claiming_a_new_method(monkeypatch):
    from hegel_machine.cognition import unified_structure
    def no_search(*args, **kwargs):
        raise AssertionError('Publishing must not perform fitting or prediction')
    monkeypatch.setattr(unified_structure, 'propose_structures', no_search)
    monkeypatch.setattr(unified_structure, 'propose_response_revisions', no_search)
    current = scene()
    before = deepcopy(current[2])
    published = publish(current)
    assert published['default_operation'] == 'structural_research'
    assert published['method_selection_identified'] is False
    option, = published['options']
    assert option['execution']['method'] == '_revise_paid_structure'
    assert option['execution']['research_action'] == 'structural_research'
    assert option['execution']['native_proposer'] == 'propose_structures'
    assert option['content_status'] == 'existing_native_segment_not_learned'
    assert option['complete_quote']['new_comparison_reserve'] == 48
    assert option['complete_quote']['compute_cost'] == 90048
    assert current[2] == before


def test_completed_unexplained_relation_opens_existing_response_segment_once():
    current = scene(attempted=True)
    option, = publish(current)['options']
    assert option['operation'] == 'response_basis_revision'
    assert option['execution']['method'] == '_revise_paid_structure'
    assert option['execution']['research_action'] == 'response_basis_revision'
    assert option['execution']['native_proposer'] == 'propose_response_revisions'
    assert option['complete_quote']['compute_cost'] == 32024
    assert option['complete_quote']['independent_confirmation_included'] is False
    current[3]['completed_segments'] = [{'operation': option['operation'],
        'binding': option['binding'], 'execution_complete': True}]
    assert publish(current)['options'] == []


def test_completed_response_is_not_relabelled_as_a_completed_relation():
    current = scene(attempted=True)
    current[2]['action']['research_action'] = 'response_basis_revision'
    current[3]['completed_segments'] = [{'operation': 'structural_research',
        'binding': {key: current[3][key] for key in ('data_ref', 'parent_ref', 'pool_ref')},
        'execution_complete': True}]
    result = publish(current)
    assert result['options'] == []
    assert result['default_operation'] is None


def test_response_terms_and_funded_full_comparison_are_required():
    current = scene(attempted=True)
    current[2]['selected']['structure']['fits'][0]['response_terms'] = []
    assert publish(current)['options'] == []
    current = scene(attempted=True)
    current[3]['compute_remaining'] = 24
    result = publish(current)
    assert result['options'] == []
    assert result['excluded'][0]['reason'] == 'new_candidate_comparison_not_funded'
