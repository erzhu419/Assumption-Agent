"""Retained failures remain usable evidence without manufacturing novelty."""
import importlib.util
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.abstractions import extract_templates
from hegel_machine.cognition.language import Program


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def audited_history(tmp_path_factory):
    specification = importlib.util.spec_from_file_location('historical_relation_novelty',
        ROOT / 'scripts/audit_historical_relation_novelty.py')
    module = importlib.util.module_from_spec(specification)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / 'scripts'))
        specification.loader.exec_module(module)
    if not module.POINTER.is_file():
        pytest.skip('The retained source pointer is not present in this checkout')
    if not Path(json.loads(module.POINTER.read_text())['archive']).is_file():
        pytest.skip('The retained source archive is not mounted')
    payload, _ = module.load_source(module.POINTER)
    receipt = module.audit(pointer=module.POINTER,
        output=tmp_path_factory.mktemp('historical_relation_novelty') / 'receipt.json')
    return payload, receipt


def _resolve(payload, reference):
    value = payload
    for component in reference['json_path']:
        value = value[component]
    return value


@pytest.mark.parametrize('case_id,variable_count', [('nonlinear_square', 1), ('cross_variable_interaction', 2)])
def test_replaced_programs_supply_actual_reconstructible_square_and_interaction_fragments(audited_history, case_id, variable_count):
    payload, receipt = audited_history
    case = next(item for item in receipt['cases'] if item['id'] == case_id)
    fragment = Program.from_dict(_resolve(payload, case['old_fragment_ref']))
    assert fragment.op == 'mul' and len({name for name, _ in fragment.observed_variables()}) == variable_count
    source_path = case['old_fragment_ref']['json_path']
    assert source_path[:3] == ['episodes', 8, 'actions']
    archived = Program.from_dict(_resolve(payload,
        {'json_path': source_path[:6]}))
    extracted = extract_templates(archived, 'old_retained_failure')
    assert any(template.kind == 'subexpression' and template.instantiate(
        dict(zip(template.roles, template.source_features))) == fragment for template in extracted)
    assert receipt['retained_replaced_program']['replacement_ref']['json_path'][3] > source_path[3]
    witness, = case['witnesses']
    assert witness['exact_ordered_Program_reconstruction'] is True
    assert witness['target_constants_used_only_for_representation_audit'] is True
    assert witness['prospective_prediction'] is False and witness['scientific_support_added'] is False
    assert all(reference['json_path'][1] < witness['target_ref']['json_path'][1]
               for reference in witness['earlier_program_refs'])
    for generation in case['actual_old_generation']:
        revision = _resolve(payload, generation['revision_ref'])
        assert revision['action'] == 'revise' and revision['details']['operation'] == 'composition'
        assert revision['details']['after_program'] == payload['episodes'][generation['episode_index']]['program']


def test_reconstructing_t08_does_not_erase_its_saved_terminal_failure(audited_history):
    payload, receipt = audited_history
    case = next(item for item in receipt['cases'] if item['id'] == 'boolean_history_relation')
    assert case['status'] == 'reconstructed_by_old'
    comparison = case['no_counterexample_gain_at_final_revision']
    assert comparison['fresh_gain'] == 0
    assert comparison['saved_terminal_loss'] == payload['episodes'][7]['evaluation']['loss']
    assert comparison['saved_terminal_loss'] > 0.14
    assert case['actual_old_generation'][0]['terminal_evaluation_loss'] == comparison['saved_terminal_loss']
    assert case['retained_failure_refs']
    assert all(_resolve(payload, reference) is not None for reference in case['retained_failure_refs'])


def test_repeated_task_name_keeps_both_original_episode_positions_and_cutoffs(audited_history):
    payload, receipt = audited_history
    repeated = receipt['protocol']['duplicate_task_id']
    assert repeated['task_id'] == 't02' and repeated['distinct_episode_indices'] == [1, 16]
    assert payload['episodes'][1]['task_id'] == payload['episodes'][16]['task_id'] == 't02'
    assert payload['episodes'][1]['program'] != payload['episodes'][16]['program']
    assert receipt['source_episode_count'] == len(payload['episodes']) == 17
    for case in receipt['cases']:
        for witness in case['witnesses']:
            assert all(reference['json_path'][1] < witness['target_ref']['json_path'][1]
                       for reference in witness['earlier_program_refs'])


def test_incomplete_material_is_unknown_and_does_not_start_an_unqualified_ablation(audited_history):
    _, receipt = audited_history
    assert any(item['status'].startswith('UNKNOWN') for item in receipt['unresolved'])
    assert receipt['qualified_new_shared_relation_count'] == 0
    assert receipt['ablation_status'] == 'not_started_no_qualified_relation'
    assert not any(receipt['no_acquisition'].values())
