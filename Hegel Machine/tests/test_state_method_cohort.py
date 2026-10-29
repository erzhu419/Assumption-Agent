"""One frozen method body accumulates actual returns from distinct paid sources."""

from copy import deepcopy
from pathlib import Path

import pytest

from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint
from hegel_machine.cognition.finite_native_author import export_finite_native_author
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.research_programs import current_option_selection_plan
from scripts import run_state_method_pairing as study
from test_state_method_pairing import paid_fixture


def source_fixture(directory, index):
    paid, case, values = paid_fixture(directory)
    if not index:
        return paid, case, values
    # Another public source and parent names: the same policy must consume
    # this invocation's real binding, rather than replay source zero's action.
    values['case']['seed'] = 22
    values['programs'] = {f'second_{name}': program
                          for name, program in values['programs'].items()}
    sequence = [.3, .4, .8, .3, .1]
    history, state, rows = [], 0., []
    for value in sequence:
        state = float(value >= (.2 if state else .6))
        rows.append({'features': {'u': value}, 'history': deepcopy(history), 'target': state})
        history.append({'u': value})
    values['records'][0]['action']['sequence'] = [{'u': value} for value in sequence]
    values['records'][0]['rows'] = rows
    study.save(directory / 'inputs.json', values)
    contract = study.read(paid)
    for parent in contract['parents']:
        parent['candidate_id'] = 'second_' + parent['candidate_id']
        parent['program_ref']['json_path'][-1] = parent['candidate_id']
    contract['diagnostic_parent_id'] = 'second_old'
    study.save(directory / 'paid.json', contract)
    return paid, case, values


def frozen_pair(directory, paid, case):
    prepared = study.prepare(paid, directory / 'proposal', case_ref=case,
        evaluation_seed=700, research_compute_cap=10000, compute_cap=30000,
        evaluation_rows=20, approval_ref=paid)
    selection = study.read(prepared['static_selection_ref'])
    option = next(item for item in selection['view']['options']
                  if item['id'] == 'native_representation:whole_state')
    source = directory / 'ordinary_source.py'
    source.write_text(export_finite_native_author(), encoding='utf-8')
    refs = [prepared['static_selection_ref']]
    pair = {'ordinary_source_ref': {'file': str(source)}, 'cells': {
        '00': {'id': 'cohort_P0', 'selection_plan': {'ordinary_method_path': 'ordinary.py'},
               'source_refs': refs},
        '10': {'id': 'cohort_P1', 'selection_plan': current_option_selection_plan(
            option, source_refs=refs), 'source_refs': refs}}}
    return study.save(directory / 'frozen_pair.json', pair)


def run_source(directory, paid, case, values, bundle, prior, monkeypatch):
    registration = study.prepare(paid, directory / 'prepared', case_ref=case,
        evaluation_seed=701 + values['case']['seed'], research_compute_cap=10000,
        compute_cap=30000, evaluation_rows=20, approval_ref=paid,
        method_pair_ref=bundle, prior_pair_refs=prior)
    output = directory / 'run'

    class Evaluator:
        def __init__(self, **_):
            manifest = read_json_checkpoint(output / 'frozen_outputs.json')
            assert set(manifest['models']) == {'00', '10'}
            assert not manifest['evaluation_labels_opened']

        def evaluation_rows(self, *, length, seed):
            assert length == 20 and seed == 701 + values['case']['seed']
            return [Observation(row['features'], row['target'], tuple(row['history']))
                    for record in values['records'] for row in record['rows']]

    monkeypatch.setattr(study, 'AnonymousHistoryLab', Evaluator)
    result = study.run(registration, output)
    outcome = read_json_checkpoint(output / '10/outcome.json')
    return result, outcome, output


@pytest.mark.parametrize('alter_prior_body', [False, True])
def test_same_frozen_body_accumulates_sources_but_rejects_a_different_body(
        tmp_path, monkeypatch, alter_prior_body):
    first = source_fixture(tmp_path / 'source0', 0)
    bundle = frozen_pair(tmp_path, first[0], first[1])
    result0, outcome0, output0 = run_source(
        tmp_path / 'study0', *first, bundle, [], monkeypatch)
    retained = {name: (output0 / name).read_bytes() for name in (
        'summary.json', 'pair_00_10.json', '10/outcome.json', '10/workspace/training.json')}
    prior = result0['pair_refs']
    if alter_prior_body:
        # Keep the original files intact. This different frozen body must not
        # receive the actual measured method's identity or its support.
        artifact = study.read(outcome0['method_artifact_ref'])
        artifact['selection_plan'] = {'ordinary_method_path': 'ordinary.py'}
        artifact_ref = study.save(tmp_path / 'other_method.json', artifact)
        other = {**outcome0, 'method_artifact_ref': artifact_ref}
        outcome_ref = study.save(tmp_path / 'other_outcome.json', other)
        pair = study.read(prior[0])
        pair['challenger_ref'] = outcome_ref
        prior = [study.save(tmp_path / 'other_pair.json', pair)]
    second = source_fixture(tmp_path / 'source1', 1)
    result1, outcome1, output1 = run_source(
        tmp_path / 'study1', *second, bundle, prior, monkeypatch)
    assert outcome0['execution_complete'] and outcome1['execution_complete']
    assert outcome0['method_ref'] == outcome1['method_ref'] == {'id': 'cohort_P1', 'version': 1}
    artifact0, artifact1 = (study.read(outcome['method_artifact_ref'])
                            for outcome in (outcome0, outcome1))
    for key in ('id', 'version', 'execution', 'selection_plan', 'ordinary_source_ref'):
        assert artifact0[key] == artifact1[key]
    for index, name in enumerate(('00', '10')):
        path = f'workspace/shared/cohort_P{index}.py'
        assert (output0 / name / path).read_bytes() == (output1 / name / path).read_bytes()
    assert outcome0['actual_action_content'] != outcome1['actual_action_content']
    assert outcome0['data_ref'] != outcome1['data_ref']
    actions = [item['arguments']['native_representation_plan']
               for item in outcome1['actual_action_content']
               if item.get('arguments', {}).get('native_representation_plan')]
    assert actions and all(item['search_directions'] == ['discrete_state'] for item in actions)
    assert all('second_old' in item['parent_theory_ids'] for item in actions)
    feedback = result1['ordinary_use']
    assert feedback['consumed_pair_refs'] == [*prior, *result1['pair_refs']]
    assert not feedback['general_guard_supported']
    route = study.read(feedback['research_route_ref'])['method_action_research']
    if alter_prior_body:
        assert route['reason'] == 'paired_body_content_mismatch'
        assert route['installed_candidate'] is None
        assert 'paired_selection' not in route
    else:
        assert route['paired_selection']['paired_invocations'] == 2
        assert len(route['paired_selection']['preferences']) == 2
        assert route['paired_selection']['pair_refs'] == feedback['consumed_pair_refs']
    for name, original in retained.items():
        assert (output0 / name).read_bytes() == original


def test_changed_ordinary_source_cannot_reuse_the_frozen_cohort(tmp_path):
    paid, case, _ = source_fixture(tmp_path / 'source', 0)
    bundle = frozen_pair(tmp_path, paid, case)
    source = Path(study.read(bundle)['ordinary_source_ref']['file'])
    source.write_text(source.read_text() + '\n# Different ordinary source.\n')
    with pytest.raises(ValueError, match='original ordinary source'):
        study.prepare(paid, tmp_path / 'rejected', case_ref=case,
            evaluation_seed=800, research_compute_cap=10000, compute_cap=30000,
            evaluation_rows=20, approval_ref=paid, method_pair_ref=bundle)
