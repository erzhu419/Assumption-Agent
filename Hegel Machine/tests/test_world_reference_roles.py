"""Public role changes must not overwrite calibration references or omit targets."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition.world_relation_research import research_relation


def public_case(reference_count, target_count=2, *, measured=True):
    count = reference_count + target_count
    instruments = [{'uuid': 91, 'name': 'densitometer'}, {'uuid': 92, 'name': 'microscope'}]
    objects = []
    for number in range(1, count + 1):
        active = number <= reference_count
        status = 'activated' if active else 'no crystal present'
        objects += [
            {'uuid': 100 + number, 'name': 'crystal reactor (' + status + ')'},
            {'uuid': 200 + number, 'name': 'quantum crystal ' + str(number),
             'description': ('quantum crystal ' + str(number) +
                 (' (in crystal reactor (activated) [uuid: ' + str(100 + number) + '])'
                  if active else ' (on table [uuid: 300])'))}]
    view = {'inventoryObjects': instruments, 'accessibleEnvironmentObjects': objects,
            'dialog_box': {'is_in_dialog': False}}
    records = [{'index': 0, 'action': {'action': 'TELEPORT_TO_LOCATION', 'arg1': 'reactor lab'},
                'feedback': {'success': True}, 'observation': deepcopy(view)}]
    # The third reference disagrees with a density line but agrees with the gap
    # line. This exercises the existing refitter, not a mocked rejection.
    gaps = [1., 2., 4., 5., 6.]
    for number in range(1, count + 1):
        opened = deepcopy(view)
        frequency = 100 + 20 * gaps[number - 1] if number <= reference_count else 5000
        opened['dialog_box'] = {'is_in_dialog': True,
            'dialogIn': 'Hello, I am Crystal Reactor #' + str(number) +
                '. The current resonance frequence is: ' + str(frequency) + ' Hertz.',
            'dialogOptions': {'1': 'Exit'}}
        records.append({'index': len(records), 'action': {'action': 'TALK', 'arg1': 100 + number},
                        'feedback': {'success': True}, 'observation': opened})
    if measured:
        for number in range(1, count + 1):
            for instrument, text in [
                (91, 'The density is approximately ' + str(number) + ' grams.'),
                (92, 'The quantum gap is estimated to be ' + str(gaps[number - 1]) + ' nm.')]:
                observed = deepcopy(view)
                observed['lastActionMessage'] = 'You examine quantum crystal ' + str(number) + '. ' + text
                records.append({'index': len(records), 'action': {
                    'action': 'USE', 'arg1': instrument, 'arg2': 200 + number},
                    'feedback': {'success': True}, 'observation': observed})
    spec = {'current_observation': deepcopy(view), 'capabilities': {
        'teleport_locations': {'reactor lab': {}, 'crystal bench': {}, 'instruments': {}}}}
    memory = {'program_path': 'shared/method.py', 'relation_research': {}, 'questions': [],
              'model_available': False, 'limits': {'max_model_calls': 4}, 'costs': {'model_calls': 0}}
    return spec, records, memory


@pytest.fixture(autouse=True)
def working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def state():
    return json.loads(recipe.STATE.read_text())


def test_visible_fifth_reactor_is_censused_before_measurement_or_prediction():
    spec, records, memory = public_case(3, measured=False)
    records.pop()  # Its object is visible; its public number has not been read.
    result = recipe.select_actions(spec, records, memory)
    assert result == {'actions': [{'action': 'TALK', 'arg1': 105}]}
    assert not recipe.STATE.exists()


def test_fifth_crystal_is_measured_and_all_three_references_are_retained():
    spec, records, memory = public_case(3)
    records = [row for row in records if not (
        row['action'].get('arg1') == 91 and row['action'].get('arg2') == 205)]
    result = recipe.select_actions(spec, records, memory)
    assert result == {'actions': [{'action': 'USE', 'arg1': 91, 'arg2': 205}]}
    assert state()['reference_numbers'] == [1, 2, 3]
    assert state()['target_numbers'] == [4, 5]


def test_three_reference_refutation_tries_existing_gap_family_then_targets_four():
    spec, records, memory = public_case(3)
    result = recipe.select_actions(spec, records, memory)
    assert result['arguments']['question']['id'] == 'reference_fit_density_4'
    density_binding = state()['binding']
    assert [row['target'] for row in density_binding['rows']] == [120., 140., 180.]
    density = research_relation(density_binding, records)
    assert density['status'] == 'refuted' and density['selected_probe'] is None
    memory['relation_research']['reference_fit_density_4'] = density

    result = recipe.select_actions(spec, records, memory)
    assert result['arguments']['question']['id'] == 'reference_fit_gap_4'
    current = state()
    assert current['rejected'] == {'4': ['density']}
    assert current['fit_refutations'][0]['evidence'] == density['evidence']
    assert len(current['binding']['rows']) == 3
    gap = research_relation(current['binding'], records)
    assert gap['selected_probe']['predictions'] == pytest.approx([200.])
    memory['relation_research']['reference_fit_gap_4'] = gap

    # A reference is never picked up or retuned as a prediction target.
    result = recipe.select_actions(spec, records, memory)
    assert result == {'actions': [{'action': 'PICKUP', 'arg1': 204}]}
    current = state()
    current['completed'] = [4]
    recipe._save(current)
    for obj in spec['current_observation']['accessibleEnvironmentObjects']:
        if obj['uuid'] == 104:
            obj['name'] = 'crystal reactor (activated)'
        if obj['uuid'] == 204:
            obj['description'] = 'quantum crystal 4 (in crystal reactor (activated) [uuid: 104])'
    observed = deepcopy(spec['current_observation'])
    observed['dialog_box'] = {'is_in_dialog': True,
        'dialogIn': 'Reactor #4. The current resonance frequence is: 200 Hertz.'}
    records.append({'index': len(records), 'action': {'action': 'TALK', 'arg1': 104},
                    'feedback': {'success': True}, 'observation': observed})
    result = recipe.select_actions(spec, records, memory)
    assert result['arguments']['question']['id'] == 'reference_fit_density_5'
    assert state()['reference_numbers'] == [1, 2, 3]


def test_normal_keeps_two_references_and_two_targets():
    spec, records, memory = public_case(2)
    result = recipe.select_actions(spec, records, memory)
    assert result['arguments']['question']['id'] == 'reference_fit_density_3'
    assert state()['reference_numbers'] == [1, 2]
    assert state()['target_numbers'] == [3, 4]
    assert len(state()['binding']['rows']) == 2


def test_previously_tested_activated_target_does_not_become_a_reference_on_continuation():
    spec, records, memory = public_case(2)
    view = deepcopy(spec['current_observation'])
    for obj in view['accessibleEnvironmentObjects']:
        if obj['uuid'] == 103:
            obj['name'] = 'crystal reactor (activated)'
        if obj['uuid'] == 203:
            obj['description'] = 'quantum crystal 3 (in crystal reactor (activated) [uuid: 103])'
    opened = deepcopy(view)
    opened['dialog_box'] = {'is_in_dialog': True,
        'dialogIn': 'Reactor #3. The current resonance frequence is: 160 Hertz.',
        'dialogOptions': {'1': 'Exit'}}
    records.append({'index': len(records), 'action': {'action': 'TALK', 'arg1': 103},
                    'feedback': {'success': True}, 'observation': opened})
    spec['current_observation'] = view
    result = recipe.select_actions(spec, records, memory)
    assert result['arguments']['question']['id'] == 'reference_fit_density_4'
    assert state()['completed'] == [3]
    assert state()['reference_numbers'] == [1, 2]
    assert state()['target_numbers'] == [3, 4]
    assert len(state()['binding']['rows']) == 2


def test_ambiguous_fit_is_not_treated_as_a_refuted_family():
    spec, records, memory = public_case(3)
    memory['relation_research']['reference_fit_density_4'] = {
        'status': 'inconclusive', 'diagnosis': 'observational_ambiguity', 'selected_probe': None}
    result = recipe.select_actions(spec, records, memory)
    assert 'observational_ambiguity' in result['stop']
    assert state()['rejected'] == {}


def test_child_receives_current_reference_fit_refutation_without_other_target_failures(monkeypatch):
    monkeypatch.setattr(recipe, 'FAMILIES', (('density', 'densitometer'),))
    spec, records, memory = public_case(3)
    recipe.select_actions(spec, records, memory)
    fitted = research_relation(state()['binding'], records)
    assert fitted['status'] == 'refuted'
    assert len(fitted['fit_rows']) == 3
    memory['relation_research']['reference_fit_density_4'] = fitted
    memory['model_available'] = True
    current = state()
    # This case checks escalation after the reusable finite grammar is exhausted.
    current['expanded_features'] = ['density', 'gap']
    other = {'number': 5, 'family': 'density', 'fit_question': 'reference_fit_density_5',
             'diagnosis': fitted['diagnosis'], 'evidence': fitted['evidence']}
    current['fit_refutations'] = [other]
    recipe._save(current)
    Path(memory['program_path']).write_text(Path(recipe.__file__).read_text())

    result = recipe.select_actions(spec, records, memory)

    assert result['arguments']['operator'] == 'inquire'
    question = result['arguments']['question']
    assert question['id'] == 'reference_extension_4'
    assert 'current-reference fit failures, out-of-range predictions and physical refutations' in question['question']
    failures = json.loads(Path('shared/reference_extension_4.json').read_text())
    assert failures['fit_refutations'] == [{
        'number': 4, 'family': 'density', 'fit_question': 'reference_fit_density_4',
        'diagnosis': fitted['diagnosis'], 'evidence': fitted['evidence']}]
    assert set(fitted['evidence']) == {i for row in fitted['fit_rows'] for i in row['evidence']}
    assert failures['physical_tests'] == []
    assert failures['out_of_range_predictions'] == []
    assert state()['fit_refutations'][0] == other
