"""Common reuses the paid transition index, not an earlier role decision."""

from copy import deepcopy
from dataclasses import replace
import json

from hegel_machine.cognition.common_object_research import CommonObjectDependencyMethods
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.object_dependency_methods import propose_partial_role_experiment
from test_v8_partial_role_experiment import scene


def consumer(tmp_path):
    spec, model, records, sources = scene()
    common = CommonObjectDependencyMethods(tmp_path, sources,
        budget=SearchBudget(100000), actions=[]).common
    common.config = replace(common.config, observation_budget=24)
    common._save('spec.json', json.dumps({'feature_bounds': spec.feature_bounds,
        'max_probe_length': spec.max_probe_length}))
    common._save('sources.json', json.dumps(sources))
    common._save('selected.json', json.dumps({'fa': 0., 'fb': 1.}))
    return common, spec, model, records, sources


def invoke(common, model, records, *, previous=None, source='records.json', allowance=10000):
    common._save('model.json', json.dumps(model))
    common._save(source, json.dumps(records))
    arguments = {'operator': 'partial_role_experiment', 'allocation': {'compute': allowance},
        'spec_ref': {'file': 'spec.json'}, 'model_ref': {'file': 'model.json'},
        'records_ref': {'file': source}, 'sources_ref': {'file': 'sources.json'},
        'selected_values_ref': {'file': 'selected.json'}}
    if previous is not None:
        common._save('previous.json', json.dumps(previous))
        arguments['previous_ref'] = {'file': 'previous.json'}
    return common._execute('run', arguments, None)


def decisions(result):
    return json.loads(json.dumps({key: value for key, value in result.items()
        if key not in {'compute_cost', 'operator'}}))


def test_actual_pair_and_changed_parent_use_identical_decisions_with_only_new_row_charges(tmp_path):
    common, spec, model, records, sources = consumer(tmp_path)
    original = invoke(common, model, records)
    units = sum(len(row['controls']) + 1 for row in records)
    assert original['compute_cost'] == units
    previous = original
    for serial, action in enumerate(original['actions']):
        records.append({'controls': action['sequence'][-1],
            'before': {'qa': 1., 'qb': 1.}, 'after': {'qa': 2., 'qb': 1.},
            'target': 2. + serial, 'history': action['sequence'][:-1],
            'position': len(action['sequence']) - 1, 'probe_id': f'paid:{serial}'})
        actual = invoke(common, model, records, previous=previous)
        expected = propose_partial_role_experiment(spec, model, records,
            sources=sources, selected_values={'fa': 0., 'fb': 1.},
            compute_limit=10000, remaining_rows=common.config.observation_budget,
            previous=previous)
        assert decisions(actual) == decisions(expected)
        assert actual['compute_cost'] == len(records[-1]['controls']) + 1
        previous = actual
    assert actual['completed_relation']['status'] == 'discriminating'
    changed = deepcopy(model)
    changed['focus_candidates']['fa']['quantities'] = ['qa']
    updated = invoke(common, changed, records, previous=previous)
    expected = propose_partial_role_experiment(spec, changed, records,
        sources=sources, selected_values={'fa': 0., 'fb': 1.},
        compute_limit=10000, remaining_rows=common.config.observation_budget,
        previous=previous)
    assert decisions(updated) == decisions(expected)
    assert updated['compute_cost'] == 0
    assert common.costs['compute_cost'] == sum(len(row['controls']) + 1 for row in records)


def test_changed_record_source_or_corrected_prefix_rebuilds_the_actual_index(tmp_path):
    common, spec, model, records, sources = consumer(tmp_path)
    first = invoke(common, model, records)
    assert invoke(common, model, records)['compute_cost'] == 0
    assert invoke(common, model, records, source='other_episode.json')['compute_cost'] == first['compute_cost']
    records[0]['target'] = 7.
    corrected = invoke(common, model, records, source='other_episode.json')
    expected = propose_partial_role_experiment(spec, model, records,
        sources=sources, selected_values={'fa': 0., 'fb': 1.}, compute_limit=10000,
        remaining_rows=common.config.observation_budget)
    assert corrected['compute_cost'] == first['compute_cost']
    assert decisions(corrected) == decisions(expected)
    assert invoke(common, model, records[:1], source='other_episode.json')['compute_cost'] == 6


def test_unfunded_suffix_does_not_advance_or_charge_the_index(tmp_path):
    common, _, model, records, _ = consumer(tmp_path)
    first = invoke(common, model, records)
    action = first['actions'][0]
    records.append({'controls': action['sequence'][-1],
        'before': {'qa': 1., 'qb': 1.}, 'after': {'qa': 2., 'qb': 1.},
        'target': 2., 'history': [], 'position': 0, 'probe_id': 'suffix'})
    denied = invoke(common, model, records, previous=first, allowance=5)
    assert denied['compute_cost'] == 0
    assert common._partial_role_transition_index['methods'].indexed_rows == 2
    resumed = invoke(common, model, records, previous=first, allowance=6)
    assert resumed['compute_cost'] == 6
    assert common._partial_role_transition_index['methods'].indexed_rows == 3
