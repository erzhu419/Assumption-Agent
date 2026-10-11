"""Old paid exclusions must survive bounded recomparison without excluding new laws."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import world_response_consequences as consequences
from hegel_machine.cognition.language import Program
from test_world_response_consequences import (
    LINEAR, QUADRATIC, assess, fixture, response, selected_result,
)


SAVED = json.loads((Path(__file__).parent / "fixtures" /
                    "response_family_retention_20261010.json").read_text())


def current_with_paid_exclusions(name):
    old = deepcopy(SAVED['common'])
    retained = consequences.reuse_paid_family_exclusions(old['state'], old['estimators'])
    assert retained['signal_gap_interaction_5']['status'] == 'UNSAT_VERIFIED'
    current = deepcopy(SAVED[name])
    current['state']['family_history_checks'] = old['state']['family_history_checks']
    return current


def test_saved_after3_restores_full_refutation_and_existing_research_obligation():
    case = current_with_paid_exclusions('after_3')
    result = consequences.assess_response_consequences(
        case['state'], case['estimators'], selected_result([]))
    research = result['family_research']
    assert all(f['status'] == 'UNSAT_VERIFIED' for f in research['families'])
    assert result['status'] == 'family_research_needed'
    assert research['selected_role'] == 'shared_premise_research'
    assert research['research_obligation']['reason'] == 'declared_shared_response_families_refuted'


def test_saved_after1_keeps_compatible_quadratic_and_after2_keeps_new_cubic():
    for name, viable in [('after_1', 'grammar_signal_1'), ('after_2', 'signal_cubic_5')]:
        case = current_with_paid_exclusions(name)
        retained = consequences.reuse_paid_family_exclusions(case['state'], case['estimators'])
        assert viable not in retained
        result = consequences.assess_response_consequences(
            case['state'], case['estimators'], selected_result([]))
        families = {f['id']: f['status'] for f in result['family_research']['families']}
        assert families['signal_gap_interaction_5'] == 'UNSAT_VERIFIED'
        assert families[viable] == 'SAT_WITNESS'


@pytest.mark.parametrize('changed', ['measurement', 'basis', 'responses', 'hypothetical', 'scope'])
def test_changed_scientific_scope_cannot_reuse_old_exclusion(changed):
    case = current_with_paid_exclusions('after_3')
    state = case['state']
    family = 'signal_gap_interaction_5'
    # Keep only the historical proof being exercised; the current comparison
    # must not provide a second independently valid exclusion for this test.
    proof = state['family_history_checks'][family]
    state['family_history_checks'] = {family: proof}
    state['response_research'] = {}
    if changed == 'measurement':
        state['research_table']['targets'][-1]['measurements']['signal']['value'] += 1
    elif changed == 'basis':
        next(e for e in case['estimators'] if e['id'] == family)['basis'] = [
            Program('feature', name='signal').to_dict()]
    elif changed == 'responses':
        missing = tuple(proof['responses'][0])
        state['response_settings'] = [r for r in state['response_settings']
            if (r['crystal'], r['frequency'], r['activated']) != missing]
    elif changed == 'hypothetical':
        proof['queries'] = [response(5, 7906., True)]
    else:
        proof['scope'] = 'fixed_member_point_prediction'
    retained = consequences.reuse_paid_family_exclusions(state, case['estimators'])
    assert family not in retained


def test_fresh_paid_exclusion_survives_next_query_allowance_without_reusing_sat(monkeypatch):
    state = fixture()
    state['response_settings'].extend([response(1, 11., False), response(3, 14., True)])
    first = assess(state, [LINEAR, QUADRATIC])
    statuses = {f['id']: f['status'] for f in first['family_research']['families']}
    assert statuses == {'linear': 'UNSAT_VERIFIED', 'quadratic': 'SAT_WITNESS'}
    assert 'linear' in state['family_history_checks']
    assert 'quadratic' not in state['family_history_checks']
    calls = []

    def unresolved(features, responses, family, **kwargs):
        calls.append(family['id'])
        return {'status': 'UNKNOWN', 'compute_cost': 0, 'lp_calls': 0}

    monkeypatch.setattr(consequences, 'query_response_family', unresolved)
    result = assess(state, [LINEAR, QUADRATIC])
    statuses = {f['id']: f['status'] for f in result['family_research']['families']}
    assert statuses == {'linear': 'UNSAT_VERIFIED', 'quadratic': 'UNKNOWN'}
    assert calls and set(calls) == {'quadratic'}


def test_conditional_support_exclusion_does_not_refute_enclosing_family():
    state = fixture()
    state['response_settings'].extend([response(1, 11., False), response(3, 14., True)])
    state['family_history_checks'] = {}
    # This is the shallow state used for conditional simpler supports.
    assumed = {**state, 'response_research': {'family_research': {}}}
    assert assess(assumed, [LINEAR])['family_research']['families'][0]['status'] == 'UNSAT_VERIFIED'
    assert 'linear' in state['family_history_checks']
    result = assess(state, [QUADRATIC])
    assert result['family_research']['families'][0]['status'] == 'SAT_WITNESS'
