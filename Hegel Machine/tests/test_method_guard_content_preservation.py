"""Whole-policy comparisons cannot erase or relabel its existing branches."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.research_programs import (
    rebuild_research_selection_guard, export_research_selection_program,
)
from test_method_startup_witness_binding import selection_plan, witness
from test_native_representation_method_selection import prepared, current_memory
from test_v8_structural_method_body import planning_option


def choice():
    return {'general_guard_supported': True,
        'condition': Program('ge', children=(Program('feature', name='different'),
            Program('const', value=.5))).to_dict(),
        'coordinate_paths': [{'path': ['research', 'stage'], 'kind': 'equals', 'value': 'fit'}]}


def test_whole_method_label_retains_every_existing_branch_and_condition():
    plan = {**selection_plan(), 'ordinary_method_path': 'ordinary.py',
        'condition_feature_paths': [{'path': ['research', 'mature_error'], 'kind': 'number'}],
        'source_refs': [{'file': 'old_source.json'}]}
    original, evidence = deepcopy(plan), choice()
    result = rebuild_research_selection_guard(plan, evidence)
    assert result['status'] == 'conditioned_method_retained'
    assert result['reason'] == 'whole_method_pairs_do_not_identify_branch_replacement'
    assert result['plan'] == original and plan == original
    assert not result['changed'] and result['existing_selection_evidence_retained']
    assert export_research_selection_program(result['plan']) == export_research_selection_program(original)
    result['plan']['when_false']['first_segment']['parent_binding'] = 'changed_copy'
    assert plan == original


def test_same_guard_text_does_not_replace_coverage_with_whole_policy_evidence():
    plan = selection_plan()
    evidence = choice()
    evidence['condition'] = deepcopy(plan['condition'])
    result = rebuild_research_selection_guard(plan, evidence)
    assert not result['changed']
    assert result['existing_selection_evidence_retained']
    assert result['plan'] == plan


@pytest.mark.parametrize('value,supported,priority', [
    (0, True, ['old']), (1, True, []), (0, False, [])])
def test_retained_body_keeps_local_whole_and_unknown_actions(
        tmp_path, monkeypatch, value, supported, priority):
    agent, lab, _, _ = prepared(tmp_path, monkeypatch)
    memory = current_memory(agent, lab)
    memory.update(ordinary_method_path=str(agent.workspace / 'ordinary.py'),
        method_condition_features={'whole': value},
        method_condition_support={'can_apply': supported},
        method_candidate_pool={'status': 'observed', 'witnesses': [witness(['old.py'])]})
    previous = selection_plan()
    revised = rebuild_research_selection_guard(previous, choice())['plan']
    outputs = []
    for plan in (previous, revised):
        namespace = {}
        exec(export_research_selection_program(plan), namespace)
        outputs.append(namespace['select_actions'](lab.public_spec(), agent.observations, memory))
    assert outputs[0] == outputs[1]
    native = outputs[1]['arguments']['native_representation_plan']
    assert native.get('startup_parent_priority', []) == priority
    assert native['max_candidates'] == 4
    assert native['parent_theory_ids'] == ['old', 'constant']


def test_unconditional_method_still_gets_a_validated_guard_over_its_complete_body():
    plan = {'ordinary_method_path': 'ordinary.py',
        'first_segment': {'current_option': {'id': 'native_representation:ordinary', 'family': 'N'},
                          'parent_binding': 'paid_startup_witnesses'},
        'source_refs': [{'file': 'original_execution.json'}]}
    original = deepcopy(plan)
    evidence = choice()
    result = rebuild_research_selection_guard(plan, evidence)
    assert result['changed']
    assert result['plan']['when_true'] == {key: value for key, value in plan.items()
                                         if key != 'ordinary_method_path'}
    assert result['plan']['when_false'] == {}
    assert result['plan']['condition'] == evidence['condition']
    assert result['plan']['condition_feature_paths'] == evidence['coordinate_paths']
    assert plan == original
    compile(export_research_selection_program(result['plan']), 'guarded_method.py', 'exec')


def test_unsupported_guard_preserves_unconditional_method():
    plan = {'ordinary_method_path': 'ordinary.py', 'first_segment': {
        'current_option': {'id': 'native_representation:whole_state', 'family': 'N'}}}
    result = rebuild_research_selection_guard(plan, {'general_guard_supported': False})
    assert not result['changed'] and result['plan'] == plan


def test_first_guard_keeps_the_whole_two_segment_dependency():
    option = planning_option()
    plan = {'ordinary_method_path': 'ordinary.py',
        'first_segment': option['semantic_action'],
        'second_segment': {'followup_of_first_segment': True,
                           'purpose': 'conditional_history_structure'},
        'source_refs': [{'file': 'paid_two_segment_return.json'}]}
    revised = rebuild_research_selection_guard(plan, choice())['plan']
    assert revised['when_true'] == {key: value for key, value in plan.items()
                                    if key != 'ordinary_method_path'}
    compile(export_research_selection_program(revised), 'two_segment_method.py', 'exec')


@pytest.mark.parametrize('conditioned', [True, False])
def test_actual_method_research_preserves_tested_content_and_old_evidence(
        tmp_path, monkeypatch, conditioned):
    from hegel_machine.cognition import finite_native_author as author
    from hegel_machine.cognition import method_feedback
    from hegel_machine.cognition import theory_consolidation as library
    from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
    from hegel_machine.cognition.method_action_research import research_method_actions
    from test_inquiry_revision import PublicLab, action, make_agent

    agent = make_agent(tmp_path, 'hegel', cls=CommonArchiveInquiryAgent, compute=1000)

    def save(name, value):
        agent._save(name, json.dumps(value))
        return {'file': name, 'json_path': []}

    plan = selection_plan() if conditioned else selection_plan()['when_false']
    plan = {**plan, 'ordinary_method_path': 'ordinary.py'}
    agent._save('ordinary.py', author.export_finite_native_author())
    old_support = {**choice(), 'coverage': {'source': 'old_actual_method_pairs'}}
    selection_ref = save('method_feedback/action_selection_local_P1.json', old_support)
    previous_analysis = {'status': 'previous_whole_policy_analysis'}
    previous_ref = save('method_feedback/action_selection_local_P1_0000.json', previous_analysis)
    candidate = library.install_python_research_method(agent, candidate_id='local:P1',
        source=export_research_selection_program(plan), selection_plan=plan,
        source_refs=[], selection_ref=selection_ref)
    source = save('source.json', {'method_results': [{'path': 'ordinary.py',
        'proposal': {'action': 'query', 'arguments': {'actions': [action(0.)]}},
        'feedback': {'status': 'completed'}}]})
    root = {**source, 'json_path': ['method_results', 0]}
    tested_ref = save('tested_body.json', candidate)
    outcome_ref = save('actual_return.json', {'method_artifact_ref': tested_ref,
        'method_ref': {'id': candidate['id'], 'version': candidate['version']}})
    pair_ref = save('pair.json', {'challenger_ref': outcome_ref})
    learned = {**choice(), 'local_program_supported': True,
        'identified_effects': {'policy': 'whole_tested_method'}, 'evaluation_use': 'development',
        'new_compute_cost': 0, 'local_support_refs': [pair_ref]}
    monkeypatch.setattr(method_feedback, 'research_paired_method_conditions',
                        lambda *args, **kwargs: deepcopy(learned))
    monkeypatch.setattr(author, 'publish_research_options', lambda *args: [])
    monkeypatch.setattr(author, 'research_segment_options',
                        lambda *args, **kwargs: {'options': [], 'excluded': []})
    question = agent._open_question({'id': 'method_problem', 'kind': 'method_use',
        'question': 'Which complete method should serve the parent?', 'parent_id': 'task',
        'method_use_plan': {'candidate_id': candidate['id'], 'binding': {}}})
    result = research_method_actions(agent, question, {
        'ordinary_method_path': 'ordinary.py',
        'ordinary_proposal_ref': {**root, 'json_path': [*root['json_path'], 'proposal']},
        'segments': [{'root_result_ref': root, 'decision_start_ref': root}],
        'tested_candidate_ref': tested_ref, 'paired_candidate_id': candidate['id'],
        'pair_refs': [pair_ref], 'revised_candidate_id': 'local:P2'},
        public_spec=PublicLab().public_spec(), compute_limit=1000)

    assert result['status'] == ('complete_supported_method_retained' if conditioned
                                else 'local_method_action_revision_selected')
    assert agent.theory_consolidation['candidates'][candidate['id']] == candidate
    assert json.loads((tmp_path / selection_ref['file']).read_text()) == old_support
    assert json.loads((tmp_path / previous_ref['file']).read_text()) == previous_analysis
    installed = result['installed_candidate']
    assert result['policy_selection_ref'] == {**installed['selection_ref'],
        'file': str(agent.workspace / installed['selection_ref']['file'])}
    if conditioned:
        assert installed == candidate
        assert installed['selection_plan']['when_false'] == plan['when_false']
        assert installed['selection_ref'] == selection_ref
        assert result['guard_reconstruction']['reason'] == (
            'whole_method_pairs_do_not_identify_branch_replacement')
        assert 'local:P2' not in agent.theory_consolidation['candidates']
    else:
        assert installed['id'] == 'local:P2'
        assert installed['selection_plan']['when_true'] == {
            key: value for key, value in plan.items() if key != 'ordinary_method_path'}
        assert installed['selection_ref'] != selection_ref
